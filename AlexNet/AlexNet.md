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
> - $f$ is non-saturating iff $(|\lim_{z \rightarrow -\infty}f(z)| = +\infty) \lor (|\lim_{z \rightarrow \infty}f(z)|= +\infty)$ <br/>

- ReLU(Rectified Linear Unit)를 활성화 함수로 사용
  - ReLU를 사용하는 CNN 네트워크가 tanh를 사용하는 네트워크에 비해 학습이 수렴하는 속도가 몇 배 더 빠르다
- 다른 연구에서도 활성화 함수를 대체하려는 연구는 있었으나 학습 속도를 위해 적용한 것은 처음이다.

### Training on Multiple GPUs
- GTX 580 3GB를 사용
  - 전체 데이터셋을 하나의 GPU에 올리기에는 용량이 부족하다
  - 저자들은 두 개의 GPU를 사용
  - GPU는 시스템 메모리를 거치지 않고 서로의 메모리에서 직접 읽고 쓸 수 있음 &rarr; 병렬화에 적합
- 특정 레이어에서만 2개의 GPU가 서로 데이터를 교환 가능하도록, 나머지 레이어는 같은 GPU로부터 연산을 이어받음

### Local Response Normalization
- ReLU는 포화(기울기가 점점 작아지는 현상)를 방지하기 위해 정규화를 필요로 하지 않음
  - 양수인 input이 조금이라도 있다면 학습이 진행된다
- 하지만 저자들은 일반화에 도움이 되는 새로운 로컬 정규화 방식을 도입

$$ b^i_{x,y}  = a^i_{x,y} / \left(k + \alpha \sum\limits_{j=max(0,i-n/2)}^{min(N-1,i+n/2)}(a^j_{x,y})^2 \right)^\beta $$

- $a^i_{x,y}$ : 위치 $(x,y)$에 커널 $i$를 적용한 다음 ReLU를 적용 &rarr; 뉴런 $a$의 출력
- $b^i_{x,y}$ : 정규화된 출력 $b$, 주어진 식은 $n$개의 인접 커널 맵에 걸쳐 실행(동일한 공간 위치?)
- $k$, $n$, $\alpha$, $\beta$는 hyperparameter
- LRN 적용 후 성능이 더 좋아짐을 확인(top-1 에러에서 1.4%, top-5 에러에서 1.2%의 성능 향상)

- **batch normalization의 등장으로 현재는 잘 사용하지 않음**

### Overlapping Pooling

- CNN에서의 풀링 레이어는 같은 커널 내의 인접한 뉴런의 출력을 요약
- 전통적으로 풀링 레이어의 커널을 겹쳐서 사용하지 않음
- 하지만 저자들은 풀링 레이어의 커널들을 서로 겹치게 설정함으로써 더 높은 성능을 거둠(stride = 2, 3x3 kernel 사용)

### Overall Architecture
![img](./img/alexnet_architecture.PNG)
- 전체 네트워크는 총 8개의 가중치 레이어로 구성
  - 첫 5개의 레이어는 convolutional layer, 나머지 3개의 레이어는 fully-connected layer
  - 마지막 fc 레이어의 출력은 1000개의 출력을 가진 softmax &rarr; 1000개의 클래스에 대응
- 2, 4, 5번째 convolutional layer는 동일한 GPU의 이전 레이어에만 연결 되어있음
- 3번째 레이어는 두번째 레이어의 모든 커널 맵과 연결되어 있음(두 개의 GPU 모두 해당하는듯)
- fc 레이어의 경우 이전 레이어의 모든 뉴런과 연결되어 있음
- Response normalization 레이어(LRN)는 첫번째와 두번째 conv 레이어 뒤에 옴
- Max pooling layer의 경우 LRN 레이어와 5번째 conv 레이어 뒤에 옴
- 모든 conv 레이어와 fc 레이어가 ReLU 활성화 함수를 사용
  - 레이어 구조는 conv - ReLU - (LRN) - (max-pooling) 레이어 순서<br/><br/>

- first convolutional layer
  - input : 227 * 227 * 3(논문에는 224 * 224 * 3으로 나와있는데 실제는 227이라고함)
  - output : 55 * 55 * 96 * 3
  - 11 * 11 * 3 kernel size, 96개의 커널
  - stride 4px
  - padding 0px
  
<details>
  <summary>Convolutional layer의 output</summary>

  - 입력 데이터: $W_1 \times H_1 \times D_1$ ($W_1$: 가로, $H_1$: 세로, $D_1$: 채널의 수)
  - 필터(커널)의 수: $K$
  - 필터의 크기(가로=세로): $F$
  - 스트라이드: $S$
  - 패딩: $P$
  <br/><br/>
  - 출력
    - $W_2 = (W_1 - F + 2P)/S+1$
    - $H_2 = (H_1 - F + 2P)/S+1$
    - $D_2 = K$
  - 가중치의 수: $[F_2 \times D_1 + D_1] \times K$
</details>

- first max-pooling layer
  - input : 55 * 55 * 96
  - 3 * 3 kernel size
  - stride 2px
  - output : 27 * 27 * 96 * 3 ($input-kernel/stride+1$)