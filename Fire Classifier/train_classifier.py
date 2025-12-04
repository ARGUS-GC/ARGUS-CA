import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

# ================= 설정 =================
DATA_DIR = './augmented_data'  
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_classifier():
    print(f"사용 장치: {DEVICE}")

    # 1. 데이터 전처리 
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # ResNet 기본 크기
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. 데이터 불러오기 (폴더 이름을 클래스로 자동 인식)
    # A폴더(일반) -> 0번, B폴더(불) -> 1번
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    
    # 클래스 이름 확인 (A, B가 아니라 Normal, Fire로 출력하고 싶다면 나중에 매핑 필요)
    print(f"클래스 매핑: {full_dataset.class_to_idx}") # {'A': 0, 'B': 1} 예상

    # 학습용(80%) / 검증용(20%) 나누기
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"학습 데이터: {len(train_dataset)}장, 검증 데이터: {len(val_dataset)}장")

    # 3. 모델 정의 (ResNet18 - 가볍고 빠름)
    # pretrained=True: 미리 학습된 똑똑한 모델을 가져와서 튜닝 (전이 학습)
    model = models.resnet18(pretrained=True)
    
    # 마지막 레이어를 우리 목적(2개 클래스: 불O/불X)에 맞게 교체
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) 
    
    model = model.to(DEVICE)

    # 손실 함수와 최적화 도구
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. 학습 루프
    train_losses, val_accs = [], []

    print("--- 학습 시작 ---")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        # 검증 (Validation)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        train_losses.append(epoch_loss)
        val_accs.append(epoch_acc)

        print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

    # 5. 모델 저장
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/fire_classifier_resnet.pth')
    print("--- 학습 완료! 모델이 'models/fire_classifier_resnet.pth'에 저장되었습니다. ---")

    # 6. 결과 그래프
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    
    plt.savefig('models/training_result.png')
    print("결과 그래프가 'models/training_result.png'에 저장되었습니다.")

if __name__ == '__main__':
    train_classifier()