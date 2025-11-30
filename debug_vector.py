import cv2
import threading
import time
from ultralytics import YOLO
import supervision as sv

# 프레임 최적화 클래스 (그대로 유지)
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

def main():
    model = YOLO('yolov8n.pt')

    # RTSP 주소
    rtsp_url = "rtsp://hyun00:hyun0000@172.25.86.124/stream1"
    cap = FreshFrameReader(rtsp_url)
    time.sleep(1.0) 

    # 해상도 확인
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera Resolution: {frame_width}x{frame_height}")

    # --- [중요] 선 길이를 늘리세요 ---
    # 화면 중앙을 넓게 커버하도록 수정 (예: 400 ~ 1500)
    START = sv.Point(400, 310)
    END   = sv.Point(1500, 308)

    line_zone = sv.LineZone(start=START, end=END)
    
    line_zone_annotator = sv.LineZoneAnnotator(
        thickness=2, text_thickness=2, text_scale=1
    )
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1)

    # --- [추가] 이동 경로(Trace)를 그려주는 도구 ---
    trace_annotator = sv.TraceAnnotator(
        thickness=2,
        trace_length=50 # 지난 50프레임의 경로를 보여줌
    )

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        # Tracking
        results = model.track(frame, persist=True, verbose=False, imgsz=640)
        detections = sv.Detections.from_ultralytics(results[0])

        if detections.tracker_id is not None:
            detections = detections[detections.class_id == 0]
            
            crossed_in, crossed_out = line_zone.trigger(detections=detections)
        
            if crossed_in.any() or crossed_out.any():
                total_in = line_zone.in_count
                total_out = line_zone.out_count
                current_people = total_in - total_out
                print(f"Update -> In: {total_in}, Out: {total_out}, [Current: {current_people}]")

            # Annotate
            labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
            
            # 1. Trace (이동 경로) 그리기 - 디버깅에 매우 도움됨
            frame = trace_annotator.annotate(scene=frame, detections=detections)

            # 2. Box & Label
            frame = box_annotator.annotate(scene=frame, detections=detections)
            frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
            
            # 3. Line
            line_zone_annotator.annotate(frame=frame, line_counter=line_zone)

        display_frame = cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5)
        cv2.imshow("Tapo Camera Counting", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()