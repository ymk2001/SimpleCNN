import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 중인 Device : ", DEVICE)

## 데이터 불러오기 (CIFAR-10 예제) ##
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 30

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

print("훈련 데이터셋 이미지 개수 : ", len(train_dataset))
print("테스트 데이터셋 이미지 개수 : ", len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

## 클래스 목록 및 클래스 별 예시 사진 확인 ##
# print("CIFAR-10 클래스 목록:")
# for class_name in classes:
#     print(f"- {class_name}")
# print("-" * 20)


# # --- 3. 각 클래스별 예시 이미지 찾기 및 저장 ---
# # 각 클래스의 이미지를 하나씩 찾아서 저장할 딕셔너리
# class_images = {}
# # 각 클래스의 레이블(인덱스)을 저장할 딕셔너리
# class_labels = {}

# # 데이터셋을 순회하며 각 클래스별로 첫 번째 이미지를 찾음
# for image, label in train_dataset:
#     # 아직 딕셔너리에 없는 클래스라면 추가
#     if label not in class_images:
#         class_images[label] = image
#         class_labels[label] = classes[label]
    
#     # 10개 클래스 이미지를 모두 찾았으면 반복 중단
#     if len(class_images) == 10:
#         break

# # --- 4. 예시 이미지 시각화 ---
# fig, axes = plt.subplots(2, 5, figsize=(10, 5))
# fig.suptitle("CIFAR-10 Class Examples (Improved)", fontsize=16)

# sorted_labels = sorted(class_images.keys())

# for i, ax in enumerate(axes.flat):
#     label_index = sorted_labels[i]
#     image = class_images[label_index]
#     class_name = class_labels[label_index]
    
#     img_for_display = np.transpose(image.numpy(), (1, 2, 0))
#     ax.imshow(img_for_display, interpolation='nearest') 
#     ax.set_title(class_name, fontsize=12)
#     ax.axis('off')

# plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.show()

# --- 3. 3계층 CNN 모델 정의 ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # 첫 번째 합성곱 블록
        # Input: (Batch, 3, 32, 32)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) # 배치 정규화
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: (Batch, 32, 16, 16)
        
        # 두 번째 합성곱 블록
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: (Batch, 64, 8, 8)
        
        # 세 번째 합성곱 블록
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: (Batch, 128, 4, 4)
        
        # 완전 연결 계층 (Classifier)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.5) # 드롭아웃
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # 합성곱 블록 순차 적용
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(-1, 128 * 4 * 4)
        
        # 완전 연결 계층 적용
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- 4. 모델, 손실 함수, 옵티마이저 정의 ---
model = SimpleCNN(num_classes=10).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 5. 모델 학습 및 평가 루프 ---
for epoch in range(NUM_EPOCHS):
    # 학습
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {total_loss/len(train_loader):.4f}", end=', ')
    
    # 평가
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

print("--- 학습 완료 ---")
SAVE_PATH = './cifar10_cnn_model.pth' 
torch.save(model.state_dict(), SAVE_PATH)  ## 가중치 저장

print(f"훈련된 모델이 {SAVE_PATH} 경로에 저장되었습니다.")