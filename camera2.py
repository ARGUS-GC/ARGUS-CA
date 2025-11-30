#작업장을 바라보는 카메라
#2인1조 탐지 / 사람들 장비착용여부(추후에 파인튜닝)
import cv2
import numpy as np
import time
from ultralytics import YOLO
import supervision as sv

def main():
    # 1. 설정 변수
    RTSP_URL = "rtsp://hyun00:hyun0000@172.25.86.124/stream1" # 본인 주소
    # RTSP_URL = 0 # 웹캠 테스트용
    WARNING_TIME = 5  # (초) 테스트용 5초. 실전은 30초로 변경하세요.

    # 2. 구역(Polygon) 좌표 정의 (3개)
    # ⚠️ setup_tool로 좌표를 따서 여기에 넣어야 합니다. (아래 설명 참고)
    # 일단 화면을 3등분해서 임시 좌표를 넣어둡니다.
    polygons = [
        np.array([[100, 100], [500, 100], [500, 500], [100, 500]]),   # 구역 1
        np.array([[600, 100], [1000, 100], [1000, 500], [600, 500]]), # 구역 2
        np.array([[1100, 100], [1500, 100], [1500, 500], [1100, 500]]) # 구역 3
    ]

    # 각 구역별 상태 관리 변수 (타이머 시작 시간 저장용)
    # None이면 카운트 안 하는 중, 시간이 들어있으면 카운트 중
    zone_timers = [None] * len(polygons) 

    # 모델 로드
    model = YOLO('yolov8n.pt') 
    
    # 비디오 연결
    cap = cv2.VideoCapture(RTSP_URL)
    
    # 3. Supervision Zone 설정
    zones = []
    zone_annotators = []
    box_annotator = sv.BoxAnnotator(thickness=2)

    for polygon in polygons:
        # 구역 생성
        zone = sv.PolygonZone(polygon=polygon)
        zones.append(zone)
        
        # 구역 그리는 도구 생성
        zone_annotator = sv.PolygonZoneAnnotator(
            zone=zone, 
            color=sv.Color.RED, 
            thickness=2,
            text_thickness=2,
            text_scale=1
        )
        zone_annotators.append(zone_annotator)

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 4. 추적 수행
        results = model.track(frame, persist=True, verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])

        # 사람(0)만 필터링
        if detections.tracker_id is not None:
            detections = detections[detections.class_id == 0]

        # 5. 각 구역별 로직 검사
        for i, zone in enumerate(zones):
            # 이 구역 안에 있는 사람 판단
            is_inside = zone.trigger(detections=detections)
            # True인 개수를 세면 현재 구역 내 인원수
            person_count = is_inside.sum()
            
            # --- [핵심 로직: 2인 1조 감시] ---
            status_text = "OK"
            color = (0, 255, 0) # 초록색

            if person_count == 1: # 혼자 있음!
                if zone_timers[i] is None:
                    zone_timers[i] = time.time() # 타이머 시작 (현재시간 기록)
                
                # 경과 시간 계산
                elapsed = time.time() - zone_timers[i]
                
                if elapsed > WARNING_TIME:
                    status_text = f"ALARM! ALONE {elapsed:.1f}s"
                    color = (0, 0, 255) # 빨간색 (알람)
                    with open("safety_log.txt", "a", encoding="utf-8") as f:
                        import datetime
                        # 현재 날짜와 시간 가져오기
                        now_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        # 파일에 한 줄 쓰기 (줄바꿈 \n 필수)
                        f.write(f"[{now_time}] {i+1}번 구역 경고! 혼자 작업중 - {elapsed:.1f}초 경과\n")
                else:
                    status_text = f"WARNING {elapsed:.1f}s"
                    color = (0, 255, 255) # 노란색 (주의)
            
            else: # 0명이거나 2명 이상 (정상)
                zone_timers[i] = None # 타이머 리셋
                status_text = "SAFE" if person_count >= 2 else "EMPTY"
                
            # --- [시각화] ---
            # 1. 구역 그리기
            zone_annotators[i].annotate(scene=frame)
            
            # 2. 상태 텍스트 화면에 띄우기 (구역 상단)
            text_pos = (polygons[i][0][0], polygons[i][0][1] - 10)
            cv2.putText(frame, f"Zone {i+1}: {person_count}P - {status_text}", 
                        text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 박스 그리기
        frame = box_annotator.annotate(scene=frame, detections=detections)

        # 화면 출력 (크기 조절)
        display_frame = cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5)
        cv2.imshow("Safety Monitor", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()