# Long Short-Term Memory Recurrent Neural Network Architectures for Large Scale Acoustic Modeling(2014)

## Abstract
- 장단기 메모리(Long Short-Term Memory)는 시간적 순서와 장거리 종속성을 더 정확하게 표현하기 위해 개발된 특별한 RNN 구조
- 저자들은 음성 인식 음향 모델링의 관점에서 LSTM을 탐구
- LSTM은 음향 모델링 분야에서 일반적인 DNN과 RNN에 비해 효율적
- 대규모 머신 클러스터에서 비동기 확률적 경사하강법을 사용하는 LSTM 분산 학습을 소개

## 1. Introduction
- 음성: 시간에 따라 변하는 복잡한 상관 관계를 가진 신호
  - RNN은 일반 순방향 신경망보다 시퀀스 데이터를 모델링하기 유리 - 순환 연결을 포함하기 때문

- RNN은 필기 인식 및 언어 모델링 작업에서 큰 성공을 거둔 반면 음향 모델링에서는 큰 주목을 받지 못하고 있음

- DNN은 고정 크기 슬라이딩 윈도우에서만 작동해서 제한된 시간 모델링만 제공할 수 있음
  - RNN은 이전 시간 단계의 값을 현재 시간의 입력으로 활용하여 시간 단계 예측에 영향을 줌
  - 입력 시퀀스에 따라 동적으로 변화
  - 특히 LSTM의 경우 RNN의 일부 약점을 극복한 LSTM의 경우 더 매력적임

- Bidirectional LSTM은 양방향으로 입력 시퀀스에 대해 작동하는 LSTM 네트워크
    - BLSTM 네트워크는 기존 필기 인식 SOTA 모델보다 우수
    - 여러 task에서 BLSTM에 대한 연구가 진행중

## 2. LSTM Network Architectures
### 2.1 Conventional LSTM
- LSTM은 순환 은닉층(recurrent hidden layer)에 memory block이라는 특별한 유닛을 갖고 있음
- 메모리 블록에는 정보의 흐름을 제어하는 게이트와 네트워크의 시간적 상태에 대한 강한 자체 연결을 갖는 메모리셀로 구성
- 각 메모리 블록은 입력 게이트와 출력 게이트, 그리고 망각 게이트(forget gate)를 포함
  - 망각 게이트의 추가를 통해서 하위 시퀀스로 분할되지 않은 연속 입력 스트림을 처리하지 못하는 약점을 해결
  
- 망각 게이트(forget gate)
  - 셀의 자체 반복 연결을 통해 적응적으로 셀의 메모리를 망각하거나 재설정
- 추가적으로 출력의 정확한 타이밍을 학습하기 위한? peephole connection이 존재                                                                          

### 2.2 Deep LSTM
- 음성 인식 분야에 심층 LSTM 적용
  - 여러 층의 LSTM 레이어를 쌓음
- LSTM에서 특정 시점의 feature는 단일 비선형 레이어만 거침
  - 따라서 여러 층의 LSTM 레이어를 쌓아서 성능을 올릴 수 있다?

### 2.3 LSTMP - LSTM with Recurrent Projection Layer
- 표준 LSTM: input layer, LSTM layer, output layer로 구성
  - 입력 레이어는 LSTM 레이어에 연결
  - LSTM 레이어의 순환 연결은 셀 출력 유닛에서 input unit, input gate, output gate, forget gate로 직접 연결
  - 셀 출력 유닛은 네트워크의 output layer와도 연결
- 적당한 수의 입력을 가진 LSTM 네트워크의  계산 복잡도는 $n_c \times (4 \times n_c + n_o)$ 값에 의해 좌우됨 ($n_c$: 메모리 셀(메모리 블록)의 수, $n_o$: output unit의 수)

- LSTM의 계산 복잡성 문제를 해결하기 위한 대안으로 LSTMP(Long Short-Term Memory Projected) 
  - LSTM 레이어 뒤에 별도의 선형 투영 레이어가 존재
  - 순환 투영 레이어에서 LSTM 레이어의 입력으로 연결(재귀적)
  - LSTMP 네트워크의 계산 복잡도는 $n_r \times (4 \times n_c + n_o)$의 값에 의해 결정 ($n_r$ : recurrent projection layer의 수)
  - 따라서 파라미터의 수를 $\frac{n_r}{n_c}$의 비율 만큼 줄일 수 있음

### 2.4 Deep LSTMP
- Deep LSTM과 유사
- 모델의 메모리를 출력 레이어 및 반복 연결과 독립적으로 늘릴 수 있음 : LSTMP의 장점
  - 대신 메모리 크기를 늘리면 모델이 과적합되기 쉽다 : LSTMP의 단점
- 과적합을 줄이기 위해 네트워크의 깊이를 늘리는 방식을 적용

## 3. Distributed Training: Scaling up to Large Models with Parallelization
- 저자들은 GPU 대신 멀티코어 CPU에서 LSTM 구조를 학습시키기로 선택
  - 구현이 간단하고 디버깅이 용이해서
  - 이 논문이 꽤 오래전에 쓰여진 논문이기 때문에 지금은 크게 의미없는 내용일듯
  - SIMD(Single Instruction Multiple Data) 연산을 사용하여 행렬 연산을 구현 &rarr; 병렬화에 이점 (하나의 명령어로 동일한 형태의 여러 데이터를 한번에 처리)
- BPTT(BackPropagation Through Time) 알고리즘 사용
- 비동기 SGD(Asynchronous Stochastic Gradient Descent) 사용
  - 여러 스레드에서 파라미터를 비동기적으로 업데이트


## 4. Experiments
- 대규모 음성 인식 태스크로 LSTM RNN 구조에 대한 평가 및 비교 진행
  - Google Voice Search Task
- Hidden Markov Model과 혼합한 하이브리드 방식 사용
- 데이터의 구조, 학습시 사용한 learning rate 등 언급
- 여러 LSTM과 LSTMP 구조에 대한 결과 언급
  - 대규모 작업에서 어느정도 까지는 LSTM 레이어가 많을수록 성능 향상(1 < 2 < 5, 7개의 레이어는 학습에 너무 오랜 시간 소요)
  - 단일 레이어 LSTMP의 경우 overfitting되는 경향
    - 레이어의 수를 늘리는 것이 더 나은 일반화를 가져오는 것으로 저자들은 추정 (5개의 레이어를 가진 모델이 약간 더 우수)
- LSTMP 구조가 LSTM 구조보다 수렴이 빠른 것을 확인
- 더 많은 레이어를 사용하면 일반화에 도움, 단 훈련이 더 어려워지고 수렴이 느려짐
- 특정 파라미터 수(13M) 이상에서 성능 향상이 없었음

## 5. Conclusions
- 대규모 음성 모델링에서 deep LSTM 구조를 통해 SOTA를 달성
- LSTMP 구조는 LSTM 구조보다 성능이 뛰어남
- ASGD(Asynchronous SGD)를 사용하여 LSTM 모델을 빠르게 학습시킬 수 있음을 보임