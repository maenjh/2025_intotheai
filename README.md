다음은 `dog_classifier.ipynb` Jupyter Notebook을 설명하는 `README.md` 예시입니다. 이 노트북은 **AI로 생성된 강아지 이미지와 실제 강아지 이미지를 분류하는 딥러닝 모델 학습 및 평가 파이프라인**을 포함하고 있습니다:

---

### 🐶 AI vs Real Dog Classifier

이 프로젝트는 **AI로 생성된 강아지 이미지와 실제 강아지 이미지**를 분류하는 딥러닝 모델을 구축하고, 학습 및 평가하는 전 과정을 담은 Jupyter Notebook(`dog_classifier.ipynb`)입니다. 
데어셋은 kaggle의 [Dogs Vs AiDogs](https://www.kaggle.com/datasets/akshaybabloo/dogs-vs-aidogs)에서 제공하는 강아지 이미지 데이터셋을 사용합니다.

---

### 📁 프로젝트 구성

- **`dog_classifier.ipynb`**  
  전체 파이프라인을 단계별 코드 셀로 구성한 Jupyter Notebook 파일입니다.

- **데이터 구조**
  ```
  └── Dogs Vs AiDogs/
      ├── train/
      │   └── images/
      ├── valid/
      │   └── images/
      └── test/
          └── images/
  ```
  - 파일명 예시:
    - `ai_123.jpg` → AI가 생성한 이미지 (label 0)
    - `real_456.jpg` → 실제 강아지 이미지 (label 1)

---

### 🧠 모델 구조

- **Base Model:** `ResNet-18` (torchvision pretrained)
- **수정 사항:** 마지막 Fully Connected layer를 2개 클래스 분류용으로 수정

---

### 🚀 주요 기능

| 기능                         | 설명 |
|----------------------------|------|
| ✅ 데이터 로딩 및 전처리      | 이미지 resize, normalize, 증강 등 |
| ✅ 커스텀 Dataset 클래스      | 파일 이름으로 레이블 추출 |
| ✅ 학습 루프                  | 훈련/검증 손실 및 정확도 기록 |
| ✅ 검증 정확도 기반 저장      | best 모델 저장 (`.pth`) |
| ✅ 평가 지표 출력             | 정확도, 혼동행렬, 분류 리포트 |
| ✅ 시각화 기능                | 학습 곡선 / 예측 이미지 결과 |

---

### 📊 출력 예시

- **Training Curve**
  - Epoch별 손실 및 정확도 그래프
- **Confusion Matrix**
  - AI vs Real 클래스 혼동 행렬 시각화
- **샘플 예측 이미지**
  - 실제 레이블 vs 예측 결과를 색상으로 구분 표시

---

### 🛠️ 실행 방법

1. 데이터 폴더 구성 확인 (`Dogs Vs AiDogs` 경로)
2. `dog_classifier.ipynb` 실행
3. 아래 셀 순서대로 실행하면 전체 학습 및 평가 과정이 자동 수행됨

```bash
# Jupyter 환경에서 실행 권장
pip install torch torchvision matplotlib scikit-learn tqdm pillow
```

---

### 🖥️ 학습 환경

- Python 3.10 이상
- PyTorch + torchvision
- GPU (CUDA) 사용 가능 시 자동 선택됨

---

### 📌 기타 참고

- 모델 파일 저장 경로 및 출력 이미지 파일 경로는 필요시 노트북 내에서 사용자 환경에 맞게 수정 필요
- 데이터셋이 큰 경우 `batch_size` 및 `num_workers` 설정 조절 가능

---

필요시 `README.md`를 프로젝트 루트에 저장해 바로 볼 수 있도록 하면 좋습니다. 원하시면 바로 `.md` 파일로도 만들어드릴게요.