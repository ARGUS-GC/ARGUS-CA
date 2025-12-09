import cv2
import numpy as np
import time
import datetime
import json
import os
from ultralytics import YOLO
import supervision as sv

# -------------------------------------------------------
# [로그 저장 함수] 파일 이어쓰기 (JSONL 방식) - 성능 최적화
# -------------------------------------------------------
def save_log_append(filename, log_data):
    """
    로그를 파일 끝에 한 줄씩 추가합니다 (Append 모드).
    파일이 커져도 매번 전체를 읽고 쓰지 않으므로 영상 끊김(Lag)을 방지합니다.
    """
    try:
        with open(filename, 'a', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False)
            f.write('\n') # 로그 한 줄마다 줄바꿈
    except Exception as e:
        print(f"[오류] 로그 저장 실패: {e}")

# -------------------------------------------------------
# [로그 래퍼 1] 나홀로 작업 경고 저장
# -------------------------------------------------------
def save_alone_log(zone_id, elapsed_time):
    file_path = "safety_log_alone.jsonl"
    log_data = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": "working_alone",
        "zone_id": zone_id,
        "elapsed_seconds": round(elapsed_time, 2),
        "message": f"{zone_id}번 구역 나홀로 작업 경고 ({round(elapsed_time, 1)}초 경과)"
    }
    save_log_append(file_path, log_data)

# -------------------------------------------------------
# [로그 래퍼 2] 헬멧 미착용 경고 저장
# -------------------------------------------------------
def save_helmet_log(count, location="Global"):
    file_path = "safety_log_helmet.jsonl"
    log_data = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": "no_helmet",
        "location": location,
        "violation_count": count,
        "message": f"헬멧 미착용자 {count}명 감지됨!"
    }
    save_log_append(file_path, log_data)

def main():
    # ==========================================
    # 1. 설정 변수
    # ==========================================
    # RTSP 주소 
    RTSP_URL = "rtsp://hyun00:hyun0000@172.25.86.124/stream1"
    
    WARNING_TIME = 5        # (초) 나홀로 작업 경고 기준 시간
    FRAME_SKIP_INTERVAL = 3 # 3프레임마다 1번 처리 (부하 감소)
    
    # 모델의 클래스 ID (0: Helmet, 1: No Helmet)
    CLASS_ID_HELMET = 0
    CLASS_ID_NO_HELMET = 1

    # ==========================================
    # 2. 감지 구역(Polygon) 좌표 정의
    # ==========================================
    polygons = [
        np.array([[48, 222], [328, 220], [306, 612], [48, 652]]),   # 1번 구역
        np.array([[600, 100], [1000, 100], [1000, 500], [600, 500]]), # 2번 구역
        np.array([[1100, 100], [1500, 100], [1500, 500], [1100, 500]]) # 3번 구역
    ]

    # 구역별 타이머 (나홀로 작업 감지용)
    zone_timers = [None] * len(polygons) 

    # YOLO 모델 로드
    try:
        model = YOLO('best.pt')
        print("✅ 모델(best.pt) 로드 완료.")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return
    
    cap = cv2.VideoCapture(RTSP_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # ==========================================
    # 3. Supervision 설정
    # ==========================================
    zones = []
    zone_annotators = []
    
    # 시각화 도구 설정
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    for polygon in polygons:
        zone = sv.PolygonZone(
            polygon=polygon, 
            triggering_anchors=[sv.Position.CENTER] 
        )
        zones.append(zone)
        
        zone_annotator = sv.PolygonZoneAnnotator(
            zone=zone, 
            color=sv.Color.RED, 
            thickness=2, 
            text_thickness=2, 
            text_scale=1
        )
        zone_annotators.append(zone_annotator)

    frame_count = 0
    print("시스템 시작됨. 종료하려면 'q'를 누르세요.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        if frame_count % FRAME_SKIP_INTERVAL != 0:
            continue

        # ------------------------------------------------------
        # A. 추적 및 전역(Global) 감지
        # ------------------------------------------------------
        results = model.track(frame, persist=True, verbose=False, imgsz=640)
        detections = sv.Detections.from_ultralytics(results[0])

        # 필요한 클래스만 필터링 (Helmet & No Helmet)
        if detections.tracker_id is not None:
            detections = detections[(detections.class_id == CLASS_ID_HELMET) | (detections.class_id == CLASS_ID_NO_HELMET)]

        # --- [로직 1: 전체 화면 헬멧 미착용 감지] ---
        # 구역 상관없이 화면 전체에서 'No Helmet' 개수 파악
        global_no_helmet_detections = detections[detections.class_id == CLASS_ID_NO_HELMET]
        global_no_helmet_count = len(global_no_helmet_detections)

        if global_no_helmet_count > 0:
            # 화면 경고 (OpenCV putText는 한글 지원이 안 되므로 영문 표기)
            cv2.putText(frame, f"WARNING: NO HELMET ({global_no_helmet_count})", (50, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            # 로그 저장
            save_helmet_log(count=global_no_helmet_count, location="Global/Screen")

        # ------------------------------------------------------
        # B. 구역별 로직 (나홀로 작업 감지)
        # ------------------------------------------------------
        for i, zone in enumerate(zones):
            # 현재 구역 안에 있는 객체만 필터링 (Masking)
            is_inside_mask = zone.trigger(detections=detections)
            zone_detections = detections[is_inside_mask]

            # 구역 내 총 인원 수 계산
            person_count_in_zone = len(zone_detections)
            
            status_text = ""
            status_color = (0, 255, 0) # 초록색

            # --- [로직 2: 나홀로 작업 모니터링] ---
            if person_count_in_zone == 1:
                # 1명일 때 타이머 시작 또는 갱신
                if zone_timers[i] is None:
                    zone_timers[i] = time.time()
                
                elapsed = time.time() - zone_timers[i]
                
                if elapsed > WARNING_TIME:
                    status_text = f"ALONE WARN({elapsed:.1f}s)"
                    status_color = (0, 0, 255) # 빨간색
                    # 로그 저장
                    save_alone_log(zone_id=i+1, elapsed_time=elapsed)
                else:
                    status_text = f"ALONE CHECK({elapsed:.1f}s)"
                    status_color = (0, 255, 255) # 노란색
            
            else:
                # 0명이거나 2명 이상이면 타이머 리셋
                zone_timers[i] = None
                if person_count_in_zone == 0:
                    status_text = "EMPTY"
                else:
                    status_text = "TEAM OK"

            # --- [시각화: 구역] ---
            # 구역 다각형 그리기
            zone_annotators[i].annotate(scene=frame)
            
            # 구역 상태 텍스트 표시
            text_pos = (polygons[i][0][0], polygons[i][0][1] - 10)
            cv2.putText(frame, f"Zone {i+1}: {person_count_in_zone}P {status_text}", 
                        text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # ------------------------------------------------------
        # C. 전체 시각화 (박스 및 라벨)
        # ------------------------------------------------------
        # 라벨 생성: "Helmet 0.95" 등
        labels = [
            f"{model.model.names[class_id]} {confidence:.2f}"
            for class_id, confidence
            in zip(detections.class_id, detections.confidence)
        ]

        # 화면 전체의 객체(구역 내/외 모두) 박스 그리기
        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        # 출력 화면 표시
        display_frame = cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5)
        cv2.imshow("Safety Monitor", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()