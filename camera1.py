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
# [최적화 1] 로그 저장 함수
# -------------------------------------------------------
def save_log_append(filename, log_data):
    try:
        with open(filename, 'a', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False)
            f.write('\n') 
    except Exception as e:
        print(f"[Log Error] {e}")

# -------------------------------------------------------
# [클래스] RTSP 영상 읽기 최적화
# -------------------------------------------------------
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
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def release(self):
        self.stopped = True
        self.t.join()
        self.cap.release()

    def get(self, propId):
        return self.cap.get(propId)

def main():
    RTSP_URL = "rtsp://hyun00:hyun0000@172.25.85.156/stream1"
    LOG_COOLDOWN = 3.0
    
    # 모델 로드
    # 사람이 확실할 때만 잡도록 conf=0.3 정도 주는 것이 '우르르' 올라가는 것 방지에 도움됨
    model = YOLO('best.pt') 

    cap = FreshFrameReader(RTSP_URL)
    time.sleep(1.0)

    # -------------------------------------------------------
    # 2. Supervision 설정
    # -------------------------------------------------------
    # [중요] 라인 좌표: 사람이 '지나가는' 길목이어야 하며
    END = sv.Point(694, 378)
    START   = sv.Point(970, 202)
    
    line_zone = sv.LineZone(
        start=START, end=END,
        triggering_anchors=[sv.Position.CENTER]
    )
    
    line_zone_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=1)
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1)
    trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=50)

    last_helmet_log_time = 0
    
    print("시스템 시작. 'q' 종료.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        # -------------------------------------------------------
        # 3. 추론 및 필터링
        # -------------------------------------------------------
        # conf=0.5 추가 (노이즈에 따라 조절)
        # iou=0.5 (겹침 방지)
        results = model.track(frame, persist=True, verbose=False, imgsz=640, conf=0.5, iou=0.5)
        detections = sv.Detections.from_ultralytics(results[0])

        # 추적 ID가 있는 경우 처리
        if detections.tracker_id is not None:
            
            all_people = detections[(detections.class_id == 0) | (detections.class_id == 1)]
            unsafe_people = detections[detections.class_id == 1]

            # [기능 1] 라인 카운팅
            crossed_in, crossed_out = line_zone.trigger(detections=all_people)
            
            if crossed_in.any() or crossed_out.any():
                print(f"[Count] IN:{line_zone.in_count} OUT:{line_zone.out_count}")
                # (로그 저장 로직은 동일하게 유지)
                log_data = {
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "type": "count",
                    "in": line_zone.in_count, 
                    "out": line_zone.out_count, 
                    "current": line_zone.in_count - line_zone.out_count
                }
                save_log_append("count_log.jsonl", log_data)

            # [기능 2] 헬멧 미착용 감지
            if len(unsafe_people) > 0:
                current_time = time.time()
                cv2.putText(frame, "WARNING: NO HELMET!", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                
                if current_time - last_helmet_log_time > LOG_COOLDOWN:
                    tracker_ids = unsafe_people.tracker_id.tolist()
                    save_log_append("safety_log.jsonl", {
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "type": "violation",
                        "count": len(unsafe_people),
                        "ids": tracker_ids
                    })
                    last_helmet_log_time = current_time

            
            labels = [
                f"{model.names[cid]} #{tid}" 
                for cid, tid in zip(all_people.class_id, all_people.tracker_id)
            ]
            frame = trace_annotator.annotate(scene=frame, detections=all_people)
            frame = box_annotator.annotate(scene=frame, detections=all_people)
            frame = label_annotator.annotate(scene=frame, detections=all_people, labels=labels)

        # -------------------------------------------------------
        line_zone_annotator.annotate(frame=frame, line_counter=line_zone)

        # 화면 출력
        display_frame = cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5)
        cv2.imshow("CCTV System", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()