import cv2
import numpy as np
import time
import datetime
import json
import os
from ultralytics import YOLO
import supervision as sv

def save_log_json(zone_id, elapsed_time):
    """
    경고 발생 시 로그를 JSON 파일에 저장하는 함수
    기존 파일을 읽어 리스트에 추가하고 다시 저장합니다.
    """
    file_path = "safety_log.json"
    
    # 저장할 데이터 구조 생성
    log_data = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "zone_id": zone_id,
        "event": "working_alone_warning",
        "elapsed_seconds": round(elapsed_time, 2),
        "message": f"{zone_id}번 구역 나홀로 작업 경고"
    }

    try:
        # 기존 파일이 있으면 읽어오기
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    logs = json.load(f)
                    # 파일이 비어있거나 리스트가 아닌 경우 초기화
                    if not isinstance(logs, list):
                        logs = []
                except json.JSONDecodeError:
                    logs = []
        else:
            logs = []

        # 새 로그 추가
        logs.append(log_data)

        # 파일에 다시 쓰기
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=4)
            
    except Exception as e:
        print(f"로그 저장 중 오류 발생: {e}")

def main():
    # ==========================================
    # 1. 설정 변수 (Configuration)
    # ==========================================
    RTSP_URL = "rtsp://hyun00:hyun0000@172.25.86.124/stream1"  # CCTV 주소
    # RTSP_URL = 0  # 웹캠 테스트용
    
    WARNING_TIME = 5  # (초) 경고 기준 시간. 테스트용 5초 (실제 운영 시 30초 권장)
    
    # [최적화] 프레임 건너뛰기 설정 (N 프레임마다 1번씩 처리)
    # 딜레이가 심할 경우 이 값을 3, 4, 5 등으로 높이세요.
    FRAME_SKIP_INTERVAL = 3 

    # ==========================================
    # 2. 감지 구역(Polygon) 좌표 정의
    # ==========================================
    # ⚠️ setup_tool을 통해 얻은 좌표를 여기에 붙여넣으세요.
    # 현재는 화면을 3분할하는 임시 좌표입니다.
    polygons = [
        np.array([[48, 222], [328, 220], [306, 612], [48, 652]]),   # 1번 구역
        np.array([[600, 100], [1000, 100], [1000, 500], [600, 500]]), # 2번 구역
        np.array([[1100, 100], [1500, 100], [1500, 500], [1100, 500]]) # 3번 구역
    ]

    # 구역별 상태 관리 변수 (타이머 시작 시간 저장)
    # None: 카운팅 안 함, Time: 카운팅 시작됨
    zone_timers = [None] * len(polygons) 

    # 모델 로드
    # [최적화] GPU 사용 가능 시 자동 사용 (device=0 or mps), 없으면 CPU
    model = YOLO('yolov8n.pt') 
    
    # 비디오 연결
    cap = cv2.VideoCapture(RTSP_URL)
    
    # [최적화] RTSP 지연 시간을 줄이기 위해 버퍼 크기를 1로 설정
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # ==========================================
    # 3. Supervision 구역(Zone) 설정
    # ==========================================
    zones = []
    zone_annotators = []
    
    # 박스 그리기 도구 설정
    box_annotator = sv.BoxAnnotator(thickness=2)

    for polygon in polygons:
        # [해결책] 감지 기준점을 바닥 중앙(BOTTOM_CENTER)에서 중앙(CENTER)으로 변경
        # 이렇게 하면 발이 라인에 걸칠 때 민감하게 반응하는 것을 줄일 수 있습니다.
        zone = sv.PolygonZone(
            polygon=polygon, 
            triggering_anchors=[sv.Position.CENTER] 
        )
        zones.append(zone)
        
        # 구역 시각화 도구 생성
        zone_annotator = sv.PolygonZoneAnnotator(
            zone=zone, 
            color=sv.Color.RED, 
            thickness=2,
            text_thickness=2,
            text_scale=1
        )
        zone_annotators.append(zone_annotator)

    frame_count = 0

    # ==========================================
    # 4. 메인 루프 실행
    # ==========================================
    while True:
        ret, frame = cap.read()
        if not ret: break

        # [최적화] 프레임 건너뛰기 로직
        frame_count += 1
        if frame_count % FRAME_SKIP_INTERVAL != 0:
            continue  # 이번 프레임은 처리를 건너뛰고 다음 프레임으로 넘어감

        # 객체 추적 (Tracking) 수행
        # persist=True는 객체 ID 유지를 위해 필수입니다.
        results = model.track(frame, persist=True, verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])

        # 사람(Class ID 0)만 필터링
        if detections.tracker_id is not None:
            detections = detections[detections.class_id == 0]

        # 각 구역별 로직 수행
        for i, zone in enumerate(zones):
            # 해당 구역 안에 사람이 있는지 확인
            is_inside = zone.trigger(detections=detections)
            # True 개수를 합산하여 사람 수 계산
            person_count = is_inside.sum()
            
            # --- [핵심 로직: 2인 1조 작업 모니터링] ---
            status_text = "OK"
            color = (0, 255, 0) # 초록색 (정상)

            # 사람이 1명일 때 (혼자 작업 중)
            if person_count == 1: 
                if zone_timers[i] is None:
                    zone_timers[i] = time.time() # 타이머 시작
                
                # 경과 시간 계산
                elapsed = time.time() - zone_timers[i]
                
                # 경고 시간 초과 시 (알람 상황)
                if elapsed > WARNING_TIME:
                    status_text = f"ALARM! ALONE {elapsed:.1f}s"
                    color = (0, 0, 255) # 빨간색 (알람)
                    
                    # [변경됨] JSON 파일로 로그 저장
                    # 주의: 매 프레임 저장하면 부하가 걸릴 수 있으므로, 실제 환경에서는
                    # 일정 간격으로 저장하거나 상태가 변할 때만 저장하는 것이 좋습니다.
                    save_log_json(zone_id=i+1, elapsed_time=elapsed)

                else:
                    status_text = f"WARNING {elapsed:.1f}s"
                    color = (0, 255, 255) # 노란색 (주의)
            
            # 사람이 0명이거나 2명 이상일 때 (안전)
            else: 
                zone_timers[i] = None # 타이머 초기화
                status_text = "SAFE" if person_count >= 2 else "EMPTY"
                
            # --- [시각화: 화면 그리기] ---
            # 1. 구역(Polygon) 그리기
            zone_annotators[i].annotate(scene=frame)
            
            # 2. 상태 텍스트 표시 (구역 상단)
            text_pos = (polygons[i][0][0], polygons[i][0][1] - 10)
            cv2.putText(frame, f"Zone {i+1}: {person_count}P - {status_text}", 
                        text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 사람 감지 박스 그리기
        frame = box_annotator.annotate(scene=frame, detections=detections)

        # 화면 출력 (크기 조절)
        display_frame = cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5)
        cv2.imshow("Safety Monitor", display_frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
