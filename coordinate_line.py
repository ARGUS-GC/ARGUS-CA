import cv2
import numpy as np

# ==========================================
# 사용자 설정 (여기에 RTSP 주소 넣으세요)
RTSP_URL = "rtsp://hyun00:hyun0000@172.25.86.124/stream1"
# RTSP_URL = 0 # 웹캠 테스트 시
# ==========================================

# 전역 변수
drawing = False
ix, iy = -1, -1
tx, ty = -1, -1
scale_factor = 0.5  # 화면 표시 배율 (0.5면 절반 크기)

def mouse_callback(event, x, y, flags, param):
    global ix, iy, tx, ty, drawing

    # 마우스 좌표를 원본 해상도로 복원
    original_x = int(x / scale_factor)
    original_y = int(y / scale_factor)

    # 마우스 왼쪽 버튼 누름 (드래그 시작)
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = original_x, original_y
        tx, ty = original_x, original_y

    # 마우스 이동 (선 그리기 미리보기)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            tx, ty = original_x, original_y

    # 마우스 왼쪽 버튼 뗌 (드래그 끝 -> 좌표 출력)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        tx, ty = original_x, original_y
        
        # 터미널에 복사하기 좋은 형태로 출력
        print("\n" + "="*40)
        print(f"🎉 좌표 획득 완료! 아래 코드를 복사해서 사용하세요:")
        print(f"START = sv.Point({ix}, {iy})")
        print(f"END   = sv.Point({tx}, {ty})")
        print("="*40 + "\n")

def main():
    cap = cv2.VideoCapture(RTSP_URL)
    
    cv2.namedWindow('Tapo Setup Tool')
    cv2.setMouseCallback('Tapo Setup Tool', mouse_callback)

    print("👉 화면에서 원하는 위치를 마우스로 '드래그' 하세요.")
    print("👉 'q' 키를 누르면 종료됩니다.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("영상 연결 실패")
            break

        # 1. 화면에 현재 긋고 있는 선 표시 (시각적 피드백)
        # 원본 프레임에 그립니다.
        if ix != -1 and iy != -1:
            # 시작점 표시
            cv2.circle(frame, (ix, iy), 5, (0, 0, 255), -1)
            
            # 드래그 중이거나 끝난 경우 선 그리기
            if tx != -1 and ty != -1:
                cv2.line(frame, (ix, iy), (tx, ty), (0, 255, 0), 2)
                cv2.circle(frame, (tx, ty), 5, (0, 0, 255), -1)

        # 2. 화면 출력용 리사이즈 (0.5배)
        display_frame = cv2.resize(frame, dsize=(0, 0), fx=scale_factor, fy=scale_factor)
        
        cv2.imshow('Tapo Setup Tool', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()