# GAN 기반 이미지 생성 모델 (MNIST / FashionMNIST)

이 프로젝트는 **GAN(Generative Adversarial Network)** 을 활용하여  
MNIST 또는 FashionMNIST 데이터셋 기반의 이미지를 생성하는 모델입니다.

---

## 실행 방법

### 1) 환경 세팅

```bash
pip install torch torchvision
```

### 2) 학습 실행

```bash
# MNIST로 학습
python simple_gan.py

# FashionMNIST로 학습 (코드에서 dataset 부분만 변경)
# dataset = datasets.FashionMNIST(...)
```

---

## 코드 구조

- **데이터 전처리**

  - 픽셀 값을 `[-1, 1]` 범위로 정규화
  - `DataLoader`를 통해 batch 단위 공급

- **Generator**

  - 입력: 잠재 공간 벡터(z, 기본 100차원)
  - 출력: `28x28x1` 이미지
  - 마지막 활성화 함수: `Tanh`

- **Discriminator**

  - 입력: 이미지(`28x28x1`)
  - 출력: 진짜일 확률(0~1)
  - 마지막 활성화 함수: `Sigmoid`

- **학습**

  1. Discriminator: 진짜=1, 가짜=0으로 분류하도록 학습
  2. Generator: 가짜 이미지를 진짜로 속이도록 학습
  3. 손실 함수: `BCELoss`
  4. 옵티마이저: `Adam`

- **결과 저장**
  - `samples/epoch_10.png`, `samples/epoch_20.png`, ...
  - 10 에폭마다 Generator가 만든 샘플 이미지 저장

---

## 실행 결과

학습 로그 출력:

```
[에폭 1/200] 판별자 손실: 1.3872 | 생성자 손실: 0.8123
[에폭 2/200] 판별자 손실: 1.0245 | 생성자 손실: 1.1456
...
[에폭 200/200] 판별자 손실: 0.6721 | 생성자 손실: 1.0324
```

저장된 샘플 이미지(`samples/epoch_10.png`, `samples/epoch_50.png` 등)에서  
점점 더 MNIST 숫자(또는 FashionMNIST 의류)의 형태가 뚜렷해지는 것을 확인할 수 있습니다.

---
