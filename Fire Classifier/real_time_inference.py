import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.ops import nms 
import cv2
import numpy as np
from PIL import Image
import time
import json
import os
from datetime import datetime

# ================= [설정값 세팅] =================
MODEL_PATH = 'Fire Classifier/models/fire_classifier_resnet.pth' # 모델 경로
VIDEO_PATH = 'Fire Classifier/test_video.mp4'  # 분석할 비디오 경로

# 추론 관련 설정
WINDOW_SIZE = 256       # 슬라이딩 윈도우 크기
STRIDE = 256            # 윈도우 이동 간격
CONF_THRESHOLD = 0.9    # 화재 판단 확신도(Confidence) 임계값
SKIP_FRAMES = 10        # 몇 프레임마다 추론할지 (성능 최적화용)
IOU_THRESHOLD = 0.2     # 중복 박스 제거(NMS) 임계값
DISPLAY_SCALE = 0.5     # 화면 출력 시 크기 비율

# [설정] 녹화 종료 대기 시간 (초)
# 화재가 사라져도 이 시간만큼은 더 녹화한 뒤 종료 (불꽃 깜빡임 대응)
RECORD_COOLDOWN = 5.0 

# [설정] 화재 검증 대기 시간 (초)
# 화재가 이 시간 이상 지속적으로 감지되어야 실제 화재로 인정하고 녹화 시작
# (일시적인 오검지 방지 목적)
FIRE_VERIFICATION_TIME = 1.0

# 로그 파일 이름
JSON_FILENAME = 'fire_events.json'

# GPU 사용 가능 여부 확인
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    """학습된 ResNet 모델을 로드하고 설정합니다."""
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) # 이진 분류 (화재 O/X)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def video_inference_view():
    """비디오를 읽어 화재를 감지하고, 검증 과정을 거쳐 녹화합니다."""
    model = load_model()
    
    # 이미지 전처리 (모델 학습시와 동일하게 설정)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print("오류: 비디오 파일을 열 수 없습니다.")
        return

    # 비디오 정보 가져오기
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30 

    frame_count = 0
    final_boxes = [] # 최종 검출된 화재 박스 리스트

    # 상태 관리를 위한 변수들
    is_recording = False # 현재 녹화 중인지 여부
    video_writer = None  # 비디오 저장 객체
    
    # 시간 관련 변수
    last_fire_time = 0              # 화재가 마지막으로 감지된 시점 (쿨다운용)
    fire_detection_start_time = None # 화재 감지가 시작된 시점 (검증용)
    
    event_logs = []         

    print(f"비디오 분석을 시작합니다... (종료하려면 'q' 키를 누르세요)")
    print(f"- 장치: {DEVICE}")
    print(f"- 검증 대기 시간: {FIRE_VERIFICATION_TIME}초")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break # 비디오 종료

            frame_count += 1
            
            # ================= 1. AI 추론 로직 =================
            if frame_count % SKIP_FRAMES == 0:
                final_boxes = [] 
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                batch_tensors = []
                batch_coords = [] 

                # 슬라이딩 윈도우로 이미지 조각내기
                for y in range(0, height - WINDOW_SIZE + 1, STRIDE):
                    for x in range(0, width - WINDOW_SIZE + 1, STRIDE):
                        patch = pil_img.crop((x, y, x + WINDOW_SIZE, y + WINDOW_SIZE))
                        tensor = preprocess(patch)
                        batch_tensors.append(tensor)
                        batch_coords.append([x, y, x + WINDOW_SIZE, y + WINDOW_SIZE])
                
                # 배치 단위로 모델 추론 수행
                if batch_tensors:
                    batch_input = torch.stack(batch_tensors).to(DEVICE)
                    with torch.no_grad():
                        outputs = model(batch_input)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        scores, preds = torch.max(probs, 1)

                        candidate_boxes = []
                        candidate_scores = []

                        # 화재(클래스 1)로 예측되고 점수가 임계값보다 높은 경우 수집
                        for i in range(len(preds)):
                            if preds[i] == 1 and scores[i] > CONF_THRESHOLD:
                                candidate_boxes.append(batch_coords[i])
                                candidate_scores.append(scores[i].item())
                        
                        # NMS (Non-Maximum Suppression) 적용하여 중복 박스 제거
                        if candidate_boxes:
                            boxes_tensor = torch.tensor(candidate_boxes, dtype=torch.float32).to(DEVICE)
                            scores_tensor = torch.tensor(candidate_scores, dtype=torch.float32).to(DEVICE)
                            keep_indices = nms(boxes_tensor, scores_tensor, IOU_THRESHOLD)
                            
                            for idx in keep_indices:
                                x1, y1, x2, y2 = candidate_boxes[idx]
                                score = candidate_scores[idx]
                                final_boxes.append((int(x1), int(y1), score))

            # ================= 2. 검증 및 녹화 판단 로직 =================
            current_sys_time = time.time()
            has_fire = len(final_boxes) > 0

            # [단계 1] 화재 존재 여부 확인 및 타이머 갱신
            if has_fire:
                last_fire_time = current_sys_time # 쿨다운 타이머 리셋
                
                # 이번 프레임에 처음 화재가 감지되었다면 시작 시간 기록
                if fire_detection_start_time is None:
                    fire_detection_start_time = current_sys_time
            else:
                # 녹화 중이 아닌데 화재가 사라지면 검증 타이머 초기화 (연속성 깨짐)
                if not is_recording:
                    fire_detection_start_time = None

            # [단계 2] 지속 시간 검증 (아직 녹화 중이 아닐 때)
            should_start_recording = False
            
            if has_fire and not is_recording and fire_detection_start_time is not None:
                duration = current_sys_time - fire_detection_start_time
                
                # 설정된 검증 시간(예: 3초)을 초과했는지 확인
                if duration >= FIRE_VERIFICATION_TIME:
                    should_start_recording = True

            # [단계 3] 녹화 시작 처리
            if should_start_recording:
                is_recording = True
                
                start_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                output_filename = f"fire_record_{start_time_str}.mp4"
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
                
                video_pos_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                
                # JSON 로그 기록 생성
                log_entry = {
                    "event_time": start_time_str,
                    "video_timestamp_sec": round(video_pos_sec, 2),
                    "message": f"화재 확정 (지속시간 {FIRE_VERIFICATION_TIME}초 달성), 녹화 시작."
                }
                event_logs.append(log_entry)
                
                with open(JSON_FILENAME, 'w', encoding='utf-8') as f:
                    json.dump(event_logs, f, indent=4, ensure_ascii=False)
                
                print(f"[경고] 화재가 확정되었습니다! 녹화를 시작합니다: {output_filename}")

            # [단계 4] 녹화 진행 및 종료 처리 (쿨다운 적용)
            if is_recording:
                # 종료 조건: 화재가 사라진 지 쿨다운 시간(5초)이 지났을 때
                if (current_sys_time - last_fire_time) > RECORD_COOLDOWN:
                    print(f"[정보] 화재 소실 확인. 녹화를 저장하고 종료합니다.")
                    is_recording = False
                    fire_detection_start_time = None # 검증 타이머 초기화
                    if video_writer:
                        video_writer.release()
                        video_writer = None
                else:
                    # 녹화 중이라면 프레임 저장
                    if video_writer:
                        video_writer.write(frame)

            # ================= 3. 화면 그리기 및 출력 =================

            # 박스 및 점수 그리기
            for (x, y, score) in final_boxes:
                cv2.rectangle(frame, (x, y), (x + WINDOW_SIZE, y + WINDOW_SIZE), (0, 0, 255), 2)
                # 참고: OpenCV putText는 한글 지원이 안 되므로 영어로 유지합니다.
                cv2.putText(frame, f"Fire {score:.2f}", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # UI 상태 표시
            if is_recording:
                # 상태: 녹화 중 (REC)
                cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1) 
                cv2.putText(frame, "REC", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            elif has_fire and fire_detection_start_time is not None:
                # 상태: 검증 중 (Verifying) - 3초 카운트다운 표시
                elapsed = current_sys_time - fire_detection_start_time
                cv2.putText(frame, f"Verifying: {elapsed:.1f}/{FIRE_VERIFICATION_TIME}s", (30, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 화면 크기 조절 (보기 편하게)
            new_width = int(width * DISPLAY_SCALE)
            new_height = int(height * DISPLAY_SCALE)
            display_frame = cv2.resize(frame, (new_width, new_height))

            cv2.imshow('Fire Detection Real-time', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # 자원 해제
        if video_writer:
            video_writer.release()
        cap.release()
        cv2.destroyAllWindows()
        print("\n완료! 프로세스가 종료되었습니다.")

if __name__ == '__main__':
    video_inference_view()