# ImageNet Classification with Deep Convolutional Neural Networks

(작성중)
## Abstract
- max-pooling layer, dropout 등을 적용하여 ILSVRC-2012에서 큰 차이로 1등을 차지

## Introduction
- 기존의 (라벨링된) 이미지 데이터셋들은 수만개 정도의 이미지에 불과
  - 이런 간단한 인식 작업의 경우 기존의 머신러닝 방법 + 적절한 증강을 통해서 충분히 좋은 성능이 나왔음(MNIST 등)
- 그러나 현실의 객체의 경우 가변성이 더 크기 때문에 더 큰 training set를 필요로 함
- 대규모의 데이터셋(LabelMe, ImageNet 등)은 최근(논문 작성 당시 기준)에서야 구축

<br/>

- 수백만장의 이미지에서 수천가지의 객체를 학습하기 위해서는 큰 학습 용량이 필요
- 그렇기 때문에 저자들은 CNN을 선택
  - 깊이, 폭 등을 제어하여 다양한 이미지 특성을 파악할 수 있기 때문
  - 비슷한 레이어를 가진 feedforward network와 비교했을 때 더 적은 연결과 파라미터를 가졌기 때문에 학습시키기 쉽다.

<br/>

- 이러한 장점에도 불구하고, 고해상도 이미지를 CNN으로 처리하기 위해서는 비싼 비용이 필요
  - 2D convoluton 연산에 최적화된 GPU를 사용

<br/>

## Dataset

- ImageNet 데이터셋은 약 22000개의 카테고리, 1500만장 이상의 라벨링된 고해상도 이미지
- ImageNet Large-Scale Visual Recognition Challenge(ILSVRC)라는 이미지 분류 대회에 사용
  - top-1, top-5 error rate를 평가
- 다양한 해상도의 이미지를 포함, 그러나 이 논문의 시스템은 고정된 해상도를 요구하기 때문에 모든 이미지를 256x256의 해상도로 downsample
  - 직사각형 이미지의 경우 짧은 변의 길이를 256픽셀로 한 뒤, 가운데에서 256x256 사이즈로 crop하여 사용
- 다른 전처리 과정 없이 raw RGB 값을 그대로 사용하여 학습

<br />

## Architecture
- 8개의 레이어로 구성
  - 5개의 convolutional layer, 3개의 fully-connected layer

### ReLU Nonlinearity
- 뉴런의 출력에 주로 사용되는 활성화 함수(비선형 함수)는 $f(x) = tanh(x)$ 또는 $f(x) = (1 + e^{-x})^{-1}$
  - 경사하강법의 학습 시간 측면에서 이러한 saturating nonlinearity 함수(아래서 설명)는 non-saturating nonlinearity 함수 $f(x) = max(0,x)$에 비해서 더 느리다

> ## saturating nonlinearity vs non-saturating nonlinearity <br/>
> - non-saturating nonlinearity 함수: 어떤 입력 $x$가 무한대로 갈 때 함수의 값이 무한대로 가는 함수 <br/>
> ex. ReLU 
> - saturating nonlinearity 함수: 어떤 입력 $x$가 무한대로 갈 때 함수의 값이 어떤 범위 내에서 만 움직이는 함수 (공역? 치역?의 범위에 제한이 있다) <br/>
> ex. sigmoid, tanh
>
> ### non-saturating의 정의 
> - $f$ is non-saturating iff $(|\lim_{z \rightarrow -\infin}f(z)| = +\infin) \lor (|\lim_{z \rightarrow \infin}f(z)|= +\infin)$ <br/>

- ReLU(Rectified Linear Unit)를 활성화 함수로 사용
  - ReLU를 사용하는 CNN 네트워크가 tanh를 사용하는 네트워크에 비해 학습이 수렴하는 속도가 몇 배 더 빠르다
  - 