import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# --- 1. 기본 설정 및 모델 클래스 정의 ---
# ※※※ 중요 ※※※
# 저장된 가중치(state_dict)를 불러오려면, 그 가중치가 저장될
# 모델의 '뼈대'(클래스 구조)가 반드시 동일하게 정의되어 있어야 합니다.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DecentCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(DecentCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- 2. 모델 인스턴스 생성 및 저장된 가중치 불러오기 ---
model = DecentCNN(num_classes=10).to(DEVICE)
LOAD_PATH = './cifar10_cnn_model.pth'
model.load_state_dict(torch.load(LOAD_PATH))
print(f"'{LOAD_PATH}' 파일에서 훈련된 모델 가중치를 성공적으로 불러왔습니다.")

# --- 3. 테스트 데이터 준비 ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# --- 4. 전체 테스트 데이터셋에 대한 성능 평가 ---
model.eval()  # 모델을 평가 모드로 설정
correct = 0
total = 0

with torch.no_grad(): # 기울기 계산 비활성화
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'\n[전체 테스트 데이터에 대한 성능]')
print(f'정확도: {100 * correct / total:.2f} % ({correct}/{total})')
print("-" * 30)


# --- 5. 예측 예시 시각화 ---
print('[예측 결과 시각화]')
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

def unnormalize(img):
    img = img * 0.5 + 0.5
    return img

# 테스트 로더에서 새로운 배치 하나를 가져옴
dataiter = iter(test_loader)
images, labels = next(dataiter)
images, labels = images.to(DEVICE), labels.to(DEVICE)

# 예측 수행
outputs = model(images)
_, predicted = torch.max(outputs, 1)

# 시각화
fig, axes = plt.subplots(5, 6, figsize=(15, 7))
for i, ax in enumerate(axes.flat):
    if i >= 30: break # 10개 이미지만 표시

    img = images[i].cpu()
    img = unnormalize(img)
    img_for_display = np.transpose(img.numpy(), (1, 2, 0))

    ax.imshow(img_for_display, interpolation='nearest')

    true_label = classes[labels[i]]
    predicted_label = classes[predicted[i]]

    title_color = "green" if true_label == predicted_label else "red"
    ax.set_title(f"Pred: {predicted_label}\n(True: {true_label})", color=title_color)
    ax.axis('off')

plt.tight_layout()
plt.show()