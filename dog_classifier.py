import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

# GPU 정보 출력 함수 추가
def print_gpu_info():
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Free GPU Memory: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    else:
        print("CUDA is not available. Using CPU.")

# 데이터 경로 설정
DATA_ROOT = os.path.join(os.path.expanduser("~"), "workspace", "projects", "intotheai", "data", "Dogs Vs AiDogs", "Dogs Vs AiDogs")
TRAIN_DIR = os.path.join(DATA_ROOT, "train", "images")
VALID_DIR = os.path.join(DATA_ROOT, "valid", "images")
TEST_DIR = os.path.join(DATA_ROOT, "test", "images")

# 이미지 변환 정의
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 커스텀 데이터셋 클래스 정의
class DogDataset(Dataset):
    def __init__(self, dir_path, transform=None):
        self.dir_path = dir_path
        self.transform = transform
        self.images = [f for f in os.listdir(dir_path) if f.endswith('.jpg')]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.dir_path, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # 레이블 생성 (ai_: 0, real_: 1)
        label = 0 if img_name.startswith('ai_') else 1
        
        return image, label

# 데이터셋 및 데이터로더 생성
def get_dataloaders(batch_size=32):
    train_dataset = DogDataset(TRAIN_DIR, transform=train_transform)
    valid_dataset = DogDataset(VALID_DIR, transform=test_transform)
    test_dataset = DogDataset(TEST_DIR, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)
    
    return train_loader, valid_loader, test_loader

# 모델 정의 (ResNet-18 사용)
def get_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # 마지막 FC 레이어를 2개 클래스(AI 생성 vs 실제 개)로 바꿈
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model

# 학습 함수
def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=10):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0.0
    model_save_path = "/raid/home/a202021017/workspace/projects/intotheai/best_dog_classifier.pth"
    
    for epoch in range(num_epochs):
        # 학습 모드
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = 100 * correct / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        
        # 검증 모드
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
                
        epoch_val_loss = running_loss / len(valid_loader.dataset)
        epoch_val_acc = 100 * correct / total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, "
              f"Valid Loss: {epoch_val_loss:.4f}, Valid Acc: {epoch_val_acc:.2f}%")
        
        # 최고 성능 모델 저장
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved with validation accuracy: {best_val_acc:.2f}%")
    
    # 학습 히스토리 반환
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs
    }
    
    return model, history

# 모델 평가 함수
def evaluate_model(model, test_loader):
    output_path = "/raid/home/a202021017/workspace/projects/intotheai/confusion_matrix.png"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    # 정확도, 혼동 행렬 및 분류 보고서 출력
    accuracy = sum(1 for i, j in zip(y_true, y_pred) if i == j) / len(y_true)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # AI Generated (0) vs Real (1) 클래스 이름
    class_names = ['AI Generated', 'Real']
    cr = classification_report(y_true, y_pred, target_names=class_names)
    print("Classification Report:")
    print(cr)
    
    # 혼동 행렬 시각화
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # 행렬 안에 숫자 표시
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(output_path)
    plt.close()  # 시각화 출력 제거
    
    return accuracy, cm, cr

# 학습 곡선 시각화 함수
def plot_training_history(history):
    output_path = "/raid/home/a202021017/workspace/projects/intotheai/training_history.png"
    plt.figure(figsize=(12, 5))
    
    # 손실 그래프
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 정확도 그래프
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()  # 시각화 출력 제거

# 이미지를 예측하고 결과를 시각화하는 함수
def predict_and_visualize(model, test_loader, num_samples=10):
    output_path = "/raid/home/a202021017/workspace/projects/intotheai/prediction_samples.png"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # 원본 이미지 시각화를 위한 inverse 정규화
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    # 예측할 이미지와 레이블 샘플링
    all_images = []
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                all_images.append(inv_normalize(images[i]).cpu())
                all_labels.append(labels[i].item())
                all_preds.append(preds[i].item())
                
                if len(all_images) >= num_samples:
                    break
            
            if len(all_images) >= num_samples:
                break
    
    # 각 클래스에 맞는 레이블 텍스트
    class_names = ['AI Generated', 'Real']
    
    # 이미지 시각화
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, (img, label, pred) in enumerate(zip(all_images[:num_samples], all_labels[:num_samples], all_preds[:num_samples])):
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        color = 'green' if label == pred else 'red'
        title = f"True: {class_names[label]}\nPred: {class_names[pred]}"
        axes[i].set_title(title, color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()  # 시각화 출력 제거

# 클래스별 이미지 개수 계산 최적화
def calculate_class_distribution(dataset):
    labels = [0 if img.startswith('ai_') else 1 for img in dataset.images]
    ai_count = labels.count(0)
    real_count = labels.count(1)
    return ai_count, real_count

if __name__ == "__main__":
    # 배치 사이즈 및 에포크 설정
    BATCH_SIZE = 512
    NUM_EPOCHS = 10  # 에포크 수 증가
    
    # GPU 정보 출력
    print_gpu_info()
    
    # 데이터로더 가져오기
    train_loader, valid_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)
    
    # 클래스별 이미지 개수 확인
    print("Dataset Statistics:")
    train_ai, train_real = calculate_class_distribution(train_loader.dataset)
    print(f"Training - AI Generated: {train_ai}, Real: {train_real}")

    val_ai, val_real = calculate_class_distribution(valid_loader.dataset)
    print(f"Validation - AI Generated: {val_ai}, Real: {val_real}")

    test_ai, test_real = calculate_class_distribution(test_loader.dataset)
    print(f"Testing - AI Generated: {test_ai}, Real: {test_real}")
    
    # 모델 생성
    model = get_model()
    
    # 손실 함수와 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # GPU 정보 출력
    print_gpu_info()
    
    # 모델 학습
    trained_model, history = train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=NUM_EPOCHS)
    
    # 학습 곡선 시각화
    plot_training_history(history)
    
    model_save_path = "/raid/home/a202021017/workspace/projects/intotheai/best_dog_classifier.pth"
    # 최고 성능 모델 로드
    model.load_state_dict(torch.load(model_save_path))
    
    # 테스트 데이터셋에서 모델 평가
    accuracy, confusion_mat, class_report = evaluate_model(model, test_loader)
    
    # 일부 이미지 예측 및 시각화
    predict_and_visualize(model, test_loader, num_samples=10)
    
    print("Training and evaluation completed!")
torch.backends.cudnn.benchmark = True  # GPU 성능 최적화