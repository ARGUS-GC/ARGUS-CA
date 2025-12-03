import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np

# ================= 설정 =================
MODEL_PATH = 'models/fire_classifier_resnet.pth' # 저장된 모델 경로
# IMAGE_PATH = '/home/user/ihson/ARGUS/Data/input_frames/Bframe_00894.jpg' 
IMAGE_PATH = '/home/user/ihson/ARGUS/Data/fire_frames/Bframe_00050.png' 
WINDOW_SIZE = 256                                # 검사할 창문 크기 (학습 데이터 크기와 맞춤)
STRIDE = 128                                     # 이동 간격 (겹치게 검사해서 놓치는 거 방지, 작을수록 꼼꼼함)
CONF_THRESHOLD = 0.8                             # 확신이 80% 이상일 때만 불이라고 판단

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    # 학습 때와 똑같은 구조로 모델 불러오기
    model = models.resnet18(pretrained=False) # 구조만 가져옴
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)         # 출력 2개 (Normal, Fire)
    
    # 저장된 가중치 입히기
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval() # 평가 모드 (필수!)
    return model

def inference():
    model = load_model()
    
    # 학습 때와 동일한 전처리 (Resize 224는 ResNet 필수)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 1. OpenCV로 이미지 읽기 (그리기 용도)
    original_img = cv2.imread(IMAGE_PATH)
    if original_img is None:
        print("이미지를 찾을 수 없습니다.")
        return
    
    # 2. PIL로 변환 (모델 입력 용도)
    pil_img = Image.fromarray(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    width, height = pil_img.size

    print(f"이미지 크기: {width}x{height} | 스캔 시작...")

    fire_detected = False
    
    # 3. 슬라이딩 윈도우 (Sliding Window) 루프
    # y방향, x방향으로 창문을 이동하며 스캔
    for y in range(0, height - WINDOW_SIZE + 1, STRIDE):
        for x in range(0, width - WINDOW_SIZE + 1, STRIDE):
            
            # (1) 이미지 조각 자르기
            patch = pil_img.crop((x, y, x + WINDOW_SIZE, y + WINDOW_SIZE))
            
            # (2) 모델 입력 형태로 변환
            input_tensor = preprocess(patch).unsqueeze(0).to(DEVICE) # 배치 차원 추가
            
            # (3) 예측
            with torch.no_grad():
                outputs = model(input_tensor)
                # 확률 계산 (Softmax)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                score, predicted = torch.max(probs, 1)
                
                # 클래스 1이 'Fire'라고 가정 (학습 시 B폴더가 1번이었음)
                # 확률이 설정한 임계값(0.8)보다 높고, 클래스가 1(Fire)이면
                if predicted.item() == 1 and score.item() > CONF_THRESHOLD:
                    print(f"🔥 화재 감지! 위치: ({x}, {y}) | 확신도: {score.item()*100:.1f}%")
                    
                    # (4) 원본 이미지에 빨간 박스 그리기
                    cv2.rectangle(original_img, (x, y), (x + WINDOW_SIZE, y + WINDOW_SIZE), (0, 0, 255), 2)
                    cv2.putText(original_img, f"Fire {score.item():.2f}", (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    fire_detected = True

    # 4. 결과 저장 및 출력
    if fire_detected:
        save_path = "result_inference.jpg"
        cv2.imwrite(save_path, original_img)
        print(f"\n✅ 화재가 감지되었습니다! 결과가 '{save_path}'에 저장되었습니다.")
    else:
        print("\n✅ 화재가 감지되지 않았습니다. (안전)")

if __name__ == '__main__':
    inference()