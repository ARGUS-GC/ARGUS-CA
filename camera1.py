import cv2
import threading
import queue
import time
import json
import os
import datetime
from ultralytics import YOLO
import supervision as sv

# -------------------------------------------------------
# [최적화 클래스] 별도 스레드를 이용한 최신 프레임 읽기
# CCTV 영상이 버퍼에 쌓여 딜레이가 생기는 것을 방지합니다.
# -------------------------------------------------------
class FreshFrameReader:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        # 버퍼 사이즈를 1로 줄여 지연 시간 최소화
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.ret = False
        self.stopped = False
        self.lock = threading.Lock()
        
        # 데몬 스레드로 실행 (메인 프로그램 종료 시 같이 종료됨)
        self.t = threading.Thread(target=self._update, daemon=True)
        self.t.start()

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break
            
            # 최신 프레임만 변수에 저장 (Lock 사용으로 충돌 방지)
            with self.lock:
                self.ret = ret
                self.frame = frame
            
            # 너무 빠른 루프 방지를 위한 미세 딜레이
            time.sleep(0.001)

    def read(self):
        # 현재 저장된 가장 최신 프레임을 반환
        with self.lock:
            return self.ret, self.frame if self.frame is not None else None

    def release(self):
        self.stopped = True
        self.t.join()
        self.cap.release()

    def get(self, propId):
        return self.cap.get(propId)

# -------------------------------------------------------
# [로그 저장 함수] 카운트 변경 시 JSON 파일에 기록
# -------------------------------------------------------
def save_count_log_json(total_in, total_out, current_people):
    file_path = "count_log.json"
    
    # 저장할 데이터 형식 정의
    log_data = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_in": total_in,
        "total_out": total_out,
        "current_people": current_people,
        "message": f"입장: {total_in}명 / 퇴장: {total_out}명 / 현재: {current_people}명"
    }

    try:
        # 1. 기존 파일 읽기
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    logs = json.load(f)
                    if not isinstance(logs, list):
                        logs = []
                except json.JSONDecodeError:
                    logs = []
        else:
            logs = []

        # 2. 새 데이터 추가
        logs.append(log_data)

        # 3. 파일에 다시 쓰기 (들여쓰기 적용)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=4)
            
    except Exception as e:
        print(f"로그 저장 실패: {e}")

# -------------------------------------------------------
# 메인 실행 함수
# -------------------------------------------------------
def main():
    # 1. YOLO 모델 로드 (사람 감지용)
    model = YOLO('yolov8n.pt')

    # 2. RTSP 스트림 주소 설정
    rtsp_url = "rtsp://hyun00:hyun0000@172.25.86.124/stream1"
    
    # 커스텀 리더 클래스로 비디오 연결
    cap = FreshFrameReader(rtsp_url)
    time.sleep(1.0) # 카메라 예열 대기

    # 카메라 해상도 출력 (디버깅용)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"카메라 해상도: {frame_width}x{frame_height}")

    # ====================================================
    # 3. 카운팅 기준선(Line) 좌표 설정
    # START와 END 점을 연결한 선을 넘어가면 카운트됩니다.
    START = sv.Point(994, 322)
    END = sv.Point(808, 342)
    # ====================================================

    # 4. LineZone(카운팅 구역) 설정
    line_zone = sv.LineZone(
        start=START, 
        end=END,
        # [중요] 사람의 '발(BOTTOM_CENTER)'이 선을 넘을 때 감지 (정확도 높음)
        triggering_anchors=[sv.Position.BOTTOM_CENTER]
    )
    
    # --- 시각화 도구들 설정 ---
    # 선 그리기 도구
    line_zone_annotator = sv.LineZoneAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )
    # 사람 박스 그리기 도구
    box_annotator = sv.BoxAnnotator(
        thickness=2
    )
    # 라벨(ID) 표시 도구
    label_annotator = sv.LabelAnnotator(
        text_thickness=2,
        text_scale=1
    )
    # 이동 경로(Trace) 표시 도구
    trace_annotator = sv.TraceAnnotator(
        thickness=2,
        trace_length=50,
        position=sv.Position.BOTTOM_CENTER 
    )

    print("시스템 시작됨. 종료하려면 'q'를 누르세요.")

    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        
        # 프레임이 비어있으면(로딩 중이면) 잠시 대기
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        # 5. 객체 추적 (Tracking) 수행
        # persist=True: 이전 프레임의 사람 ID를 유지함
        results = model.track(frame, persist=True, verbose=False, imgsz=640)

        # 6. Supervision 형식으로 결과 변환
        detections = sv.Detections.from_ultralytics(results[0])

        # 사람(Class ID 0)만 필터링하고 ID가 있는 경우만 처리
        if detections.tracker_id is not None:
            detections = detections[detections.class_id == 0]
            
            # 7. 선 통과 여부 확인 (핵심 로직)
            crossed_in, crossed_out = line_zone.trigger(detections=detections)
        
            # 누군가 선을 넘었다면(IN 또는 OUT)
            if crossed_in.any() or crossed_out.any():
                total_in = line_zone.in_count
                total_out = line_zone.out_count
                current_people = total_in - total_out
                
                # 콘솔 출력
                print(f"업데이트 -> 입장: {total_in}, 퇴장: {total_out}, [현재 인원: {current_people}명]")

                # [변경됨] JSON 파일로 로그 저장
                save_count_log_json(total_in, total_out, current_people)

            # 8. 화면에 정보 그리기 (시각화)
            labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
            
            # (1) 이동 경로 (발자취) 그리기
            frame = trace_annotator.annotate(
                scene=frame,
                detections=detections
            )

            # (2) 박스와 ID 라벨 그리기
            frame = box_annotator.annotate(scene=frame, detections=detections)
            frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

            # (3) 카운팅 선과 숫자 그리기
            line_zone_annotator.annotate(frame=frame, line_counter=line_zone)

        # 화면 출력 (너무 크면 줄여서 출력)
        display_frame = cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5)
        cv2.imshow("CCTV Counting System", display_frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 리소스 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()