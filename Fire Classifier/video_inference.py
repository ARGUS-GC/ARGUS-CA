import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
from PIL import Image
import time

# ================= 설정 =================
MODEL_PATH = 'models/fire_classifier_resnet.pth'
VIDEO_PATH = 'test_video.mp4'  # 분석할 영상 경로
OUTPUT_PATH = 'output_fire_detection.mp4' # 저장할 결과 영상 이름
WINDOW_SIZE = 256
STRIDE = 128
CONF_THRESHOLD = 0.8
SKIP_FRAMES = 10  # 10프레임마다 검사

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def video_inference_save():
    model = load_model()
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print("영상을 열 수 없습니다.")
        return

    # [수정됨] 영상 저장 설정
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 코덱 설정 (mp4v)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    frame_count = 0
    detected_boxes = [] 

    print(f"영상 처리 및 저장 시작... (결과 파일: {OUTPUT_PATH})")
    
    # 진행률 표시를 위한 총 프레임 수
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        
        # 진행 상황 출력 (100프레임마다)
        if frame_count % 100 == 0:
            print(f"진행 중: {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
        
        # AI 연산 (SKIP_FRAMES 마다)
        if frame_count % SKIP_FRAMES == 0:
            detected_boxes = [] 
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            batch_tensors = []
            batch_coords = [] 

            for y in range(0, height - WINDOW_SIZE + 1, STRIDE):
                for x in range(0, width - WINDOW_SIZE + 1, STRIDE):
                    patch = pil_img.crop((x, y, x + WINDOW_SIZE, y + WINDOW_SIZE))
                    tensor = preprocess(patch)
                    batch_tensors.append(tensor)
                    batch_coords.append((x, y))
            
            if batch_tensors:
                batch_input = torch.stack(batch_tensors).to(DEVICE)
                with torch.no_grad():
                    outputs = model(batch_input)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    scores, preds = torch.max(probs, 1)

                    for i in range(len(preds)):
                        if preds[i] == 1 and scores[i] > CONF_THRESHOLD:
                            x, y = batch_coords[i]
                            score = scores[i].item()
                            detected_boxes.append((x, y, score))

        # 박스 그리기
        for (x, y, score) in detected_boxes:
            cv2.rectangle(frame, (x, y), (x + WINDOW_SIZE, y + WINDOW_SIZE), (0, 0, 255), 2)
            cv2.putText(frame, f"Fire {score:.2f}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # [수정됨] 화면 출력(imshow) 대신 파일 저장(write)
        out.write(frame)

    cap.release()
    out.release() # 저장 파일 닫기
    print("\n완료! 결과 영상이 저장되었습니다.")

if __name__ == '__main__':
    video_inference_save()