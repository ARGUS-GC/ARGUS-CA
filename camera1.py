import cv2
import threading
import queue
import time
from ultralytics import YOLO
import supervision as sv

# -------------------------------------------------------
# [최적화] 별도 스레드 프레임 리더
class FreshFrameReader:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.ret = False
        self.stopped = False
        self.lock = threading.Lock()
        
        self.t = threading.Thread(target=self._update, daemon=True)
        self.t.start()

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break
            
            with self.lock:
                self.ret = ret
                self.frame = frame
            
            time.sleep(0.001)

    def read(self):
        with self.lock:
            return self.ret, self.frame if self.frame is not None else None

    def release(self):
        self.stopped = True
        self.t.join()
        self.cap.release()

    def get(self, propId):
        return self.cap.get(propId)

# -------------------------------------------------------

def main():
    # 1. 모델 로드
    model = YOLO('yolov8n.pt')

    # 2. RTSP 스트림 연결
    rtsp_url = "rtsp://hyun00:hyun0000@172.25.86.124/stream1"
    
    cap = FreshFrameReader(rtsp_url)
    time.sleep(1.0) 

    # 해상도 확인
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera Resolution: {frame_width}x{frame_height}")

    # ====================================================
    # 3. 선(Line) 정의 (방향 반전을 위해 START와 END 값을 서로 교체함)
    START = sv.Point(994, 322)
    END = sv.Point(808, 342)
    # ====================================================

    # 4. LineZone 설정
    line_zone = sv.LineZone(
        start=START, 
        end=END,
        # 정확도를 위해 '발(BOTTOM_CENTER)' 기준 유지
        triggering_anchors=[sv.Position.BOTTOM_CENTER]
    )
    
    # 시각화 도구 설정
    line_zone_annotator = sv.LineZoneAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )
    box_annotator = sv.BoxAnnotator(
        thickness=2
    )
    label_annotator = sv.LabelAnnotator(
        text_thickness=2,
        text_scale=1
    )
    
    # 이동 경로(Trace) 시각화 도구
    trace_annotator = sv.TraceAnnotator(
        thickness=2,
        trace_length=50,
        position=sv.Position.BOTTOM_CENTER 
    )

    print("System Started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        # 5. 추적 수행
        results = model.track(frame, persist=True, verbose=False, imgsz=640)

        # 6. Supervision 형식 변환
        detections = sv.Detections.from_ultralytics(results[0])

        if detections.tracker_id is not None:
            detections = detections[detections.class_id == 0]
            
            # 7. 선 통과 감지
            crossed_in, crossed_out = line_zone.trigger(detections=detections)
        
            if crossed_in.any() or crossed_out.any():
                total_in = line_zone.in_count
                total_out = line_zone.out_count
                current_people = total_in - total_out
                
                print(f"Update -> In: {total_in}, Out: {total_out}, [Current: {current_people} people]")

                with open("count_log.txt", "a") as f:
                    from datetime import datetime
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"[{now}] In: {total_in}, Out: {total_out}, Current: {current_people}\n")

            # 8. 화면 그리기
            labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
            
            # (1) 이동 경로 (발 기준)
            frame = trace_annotator.annotate(
                scene=frame,
                detections=detections
            )

            # (2) 박스와 라벨
            frame = box_annotator.annotate(scene=frame, detections=detections)
            frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

            # (3) 카운팅 선
            line_zone_annotator.annotate(frame=frame, line_counter=line_zone)

        display_frame = cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5)
        cv2.imshow("Tapo Camera Counting", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()