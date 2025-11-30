import cv2
import numpy as np

# ==========================================
# 설정 (RTSP 주소 및 화면 배율)
RTSP_URL = "rtsp://hyun00:hyun0000@172.25.86.124/stream1"
# RTSP_URL = 0 # 웹캠 테스트용
SCALE = 0.5    # 화면 배율
# ==========================================

points = [] # 화면상 좌표 저장

def mouse_callback(event, x, y, flags, param):
    global points

    # [좌클릭] 점 추가
    if event == cv2.EVENT_LBUTTONDOWN:
        # 점이 4개 미만일 때만 추가
        if len(points) < 4:
            points.append((x, y))
            print(f"📍 점 {len(points)}번 찍힘")
        
        # 점이 4개가 되는 순간 계산 및 출력 (여기로 옮겼습니다!)
        if len(points) == 4:
            real_points = []
            for pt in points:
                real_x = int(pt[0] / SCALE)
                real_y = int(pt[1] / SCALE)
                real_points.append([real_x, real_y])
            
            print("\n" + "="*40)
            print("🎉 [좌표 완성] 아래 코드를 복사하세요:")
            print(f"np.array({real_points})")
            print("="*40 + "\n")
            print("👉 다시 찍으려면 마우스 [우클릭] 하세요.")

    # [우클릭] 초기화
    elif event == cv2.EVENT_RBUTTONDOWN:
        points = []
        print("🔄 초기화되었습니다. 다시 찍으세요.")

def main():
    cap = cv2.VideoCapture(RTSP_URL)
    
    window_name = "Polygon Setup Tool (V3)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("영상 연결 실패")
            break

        # 1. 화면 리사이즈
        frame_display = cv2.resize(frame, dsize=(0, 0), fx=SCALE, fy=SCALE)

        # 2. 시각화 (그리기만 담당)
        if len(points) > 0:
            # 점 그리기
            for pt in points:
                cv2.circle(frame_display, pt, 5, (0, 0, 255), -1)
            
            # 선 이어주기 (4개면 닫힌 도형, 아니면 열린 도형)
            is_closed = (len(points) == 4)
            if len(points) > 1:
                cv2.polylines(frame_display, [np.array(points)], isClosed=is_closed, color=(0, 255, 0), thickness=2)

        cv2.imshow(window_name, frame_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()