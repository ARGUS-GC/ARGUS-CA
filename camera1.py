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
# [최적화 1] 로그 저장 함수 (이어쓰기 방식, 랙 방지)
# -------------------------------------------------------
def save_log_append(filename, log_data):
    try:
        with open(filename, 'a', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False)
            f.write('\n') 
    except Exception as e:
        print(f"[Log Error] {e}")

# -------------------------------------------------------
# [클래스] RTSP 영상 읽기 최적화 (별도 스레드)
# -------------------------------------------------------
class FreshFrameReader:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # 버퍼 최소화
        self.frame = None
        self.ret = False
        self.stopped = False
        self.lock = threading.Lock()
        
        # 데몬 스레드로 실행 (메인 종료 시 자동 종료)
        self.t = threading.Thread(target=self._update, daemon=True)
        self.t.start()

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break
            # 최신 프레임만 lock 걸고 업데이트
            with self.lock:
                self.ret = ret
                self.frame = frame
            # CPU 점유율 낮추기 위한 미세한 대기
            time.sleep(0.001)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy() # 안전하게 복사본 전달

    def release(self):
        self.stopped = True
        self.t.join()
        self.cap.release()

    def get(self, propId):
        return self.cap.get(propId)

def main():
    # -------------------------------------------------------
    # 1. 설정 및 모델 로드
    # -------------------------------------------------------
    RTSP_URL = "rtsp://hyun00:hyun0000@172.25.86.124/stream1"
    LOG_COOLDOWN = 3.0  # (초) 로그 저장 간격 (도배 방지)
    
    try:
        model = YOLO('best.pt')
        print("✅ 모델(best.pt) 로드 완료")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return

    # 스레드 방식 카메라 연결
    cap = FreshFrameReader(RTSP_URL)
    time.sleep(1.0) # 카메라 예열 대기

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resoluton: {frame_width}x{frame_height}")

    # -------------------------------------------------------
    # 2. Supervision 설정 (LineZone)
    # -------------------------------------------------------
    # 라인 좌표 (화면 해상도에 맞춰 조정 필요)
    START = sv.Point(994, 322)
    END = sv.Point(808, 342)
    
    line_zone = sv.LineZone(
        start=START, end=END,
        triggering_anchors=[sv.Position.BOTTOM_CENTER]
    )
    
    # Annotators
    line_zone_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=1)
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1)
    trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=50)

    # 로그 쿨다운용 변수
    last_helmet_log_time = 0
    last_count_log_time = 0
    
    print("시스템 시작. 'q' 종료.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        # -------------------------------------------------------
        # 3. 추론 및 필터링
        # -------------------------------------------------------
        results = model.track(frame, persist=True, verbose=False, imgsz=640)
        detections = sv.Detections.from_ultralytics(results[0])

        # 추적 ID가 있는 경우만 처리
        if detections.tracker_id is not None:
            
            # 클래스 정의 (0: Helmet, 1: No Helmet 가정)
            # --> 모든 사람(0+1)과 위반자(1) 분리
            all_people = detections[(detections.class_id == 0) | (detections.class_id == 1)]
            unsafe_people = detections[detections.class_id == 1]

            # [기능 1] 라인 카운팅 (모든 사람 대상)
            crossed_in, crossed_out = line_zone.trigger(detections=all_people)
            
            # 카운트 변화가 있을 때만 로그 저장
            if crossed_in.any() or crossed_out.any():
                total_in = line_zone.in_count
                total_out = line_zone.out_count
                current = total_in - total_out
                
                print(f"[Update] IN:{total_in} OUT:{total_out} CUR:{current}")
                
                # 카운트 로그 저장
                log_data = {
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "type": "count",
                    "in": total_in, "out": total_out, "current": current
                }
                save_log_append("count_log.jsonl", log_data)

            # [기능 2] 헬멧 미착용 감지 (로그 도배 방지 적용)
            if len(unsafe_people) > 0:
                current_time = time.time()
                
                # 화면 경고는 항상 표시
                cv2.putText(frame, "WARNING: NO HELMET!", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                
                # 로그 저장은 쿨다운 시간(3초)이 지났을 때만
                if current_time - last_helmet_log_time > LOG_COOLDOWN:
                    tracker_ids = unsafe_people.tracker_id.tolist()
                    
                    log_data = {
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "type": "violation",
                        "count": len(unsafe_people),
                        "ids": tracker_ids
                    }
                    save_log_append("safety_log.jsonl", log_data)
                    last_helmet_log_time = current_time # 시간 갱신

            # -------------------------------------------------------
            # 4. 시각화 (Visualization)
            # -------------------------------------------------------
            labels = [
                f"{model.names[cid]} #{tid}" 
                for cid, tid in zip(all_people.class_id, all_people.tracker_id)
            ]
            
            frame = trace_annotator.annotate(scene=frame, detections=all_people)
            frame = box_annotator.annotate(scene=frame, detections=all_people)
            frame = label_annotator.annotate(scene=frame, detections=all_people, labels=labels)
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