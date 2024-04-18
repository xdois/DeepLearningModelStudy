# Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation (2014)
## 0. Abstract
- 저자들은 이 논문에서 새로운 신경망 구조를 제안 - RNN Encoder-Decoder
  - 이 신경망 구조는 두 개의 RNN으로 구성됨
  - 하나의 RNN은 일련의 기호들을 정해진 길이의 벡터 표현으로 인코딩
  - 다른 하나의 RNN은 벡터 표현을 일련의 기호들로 디코딩
  - 제안된 인코더와 디코더는 타겟 시퀀스의 조건부 확률을 최대화 하는 방향으로 공동으로 훈련됨
  - RNN 인코더-디코더 모델을 통해 통계 기반 번역 시스템의 성능을 향상시킬 수 있음을 경험적으로 확인

## 1. Introduction
- 인공 신경망이 다양한 분야에서 성공을 거둠
  - NLP의 다양한 분야 포함
- 해당 논문에서는 구문 기반 통계 기계 번역(SMT) 시스템의 일부로 사용될 수 있는 새로운 신경망 구조에 중점을 둠
- 인코더-디코더 구조 외에도 기억력과 훈련의 용이성을 모두 향상시키기 위한 정교한 hidden unit 사용 제시
- 제안된 모델은 구문 기반 SMT 시스템에서 구문 테이블에서 각 구문 쌍들의 점수를 매기는데 사용됨
  - 기존의 구문 기반 SMT 시스템과 비교했을 때 일반적으로 더 우수한 성능을 기록
  - 연속된 공간에 대한 표현을 학습 -> 의미와 구조를 더 잘 보존

## 2. RNN Encoder-Decoder
### 2.1  Preliminary: Recurrent Neural Networks
- RNN 구조에 대한 대략적인 설명
- hidden state h와 가변 길이 input x에 대한 출력 y로 구성
- 시퀀스의 다음에 올 symbol을 학습 -> 시퀀스의 확률 분포를 학습한다

### 2.2 RNN Encoder-Decoder
- 이 논문에서 저자들이 새롭게 제안하는 신경망 구조
- 인코더와 디코더로 구성
- 인코더
  - 가변 길이 시퀀스를 고정 길이 벡터 표현으로 변환
- 디코더
  - 고정 길이 벡터 표현을 가변 길이 시퀀스로 변환

- 인코더는 입력의 각 심볼을 순차적으로 읽음
  - 각 심볼을 읽을 때 마다 RNN의 hidden state가 $\mathbf{h}_{<t>} = f(\mathbf{h}_{<t-1>}, x_t)$에 따라에 따라 변함
  - 시퀀스를 끝까지 읽은 후 (시퀀스의 끝을 의미하는 심볼까지 읽은 후) hidden state가 전체 시퀀스를 요약한 내용 c가 됨
  
- 디코더는 hidden state가 주어지면 다음 심볼을 생성하도록 훈련된 또다른 RNN 모델
  - 디코더의 hidden state는 다음과 같이 계산됨
  - $\mathbf{h}_{<t>} = f(\mathbf{h}_{<t-1>}, y_{t-1}, \mathbf{c})$
  
- 인코더와 디코더는 조건부 log-likelihood 함수를 최대화 하는 방향으로 함께 학습됨
- 모델이 학습되면 두가지 방법으로 사용 가능
  - 한가지 사용법은 모델을 주어진 입력 시퀀스에 대한 타겟 시퀀스를 생성하는데 사용하는 것
  - 다른 사용법은 입력 및 출력 시퀀스 쌍의 점수를 매기는데 사용하는 것

### 2.3 Hidden Unit that Adaptively Remembers and Forgets
- 저자들은 새로운 모델 아키텍처 뿐 아니라 새로운 hidden unit을 제시
  - $\mathbf{h}_{<t>} = f(\mathbf{h}_{<t-1>}, x_t)$
  - LSTM에서 영감을 받았지만 구현과 계산이 더 단순
    - reset gate와 update gate로 구성

- reset gate가 0에 가까워지면 hidden state가 무시되고 현재 입력으로 재설정됨
- update gate는 이전 hidden state의 정보를 얼마나 현재 hidden state로 전달할지를 결정
  - 이는 LSTM의 memory cell과 유사하게 동작
  - 다양한 시간 범위에 걸쳐 종속성을 학습
  
## 3. Statistical Machine Translation
- SMT의 두 요소
  - 번역 모델(translation model)
  - 언어 모델(language model)
- BLEU score(번역이 얼마나 잘 되었는지 평가하는 정량적 지표)를 최대화 하는 방향으로 학습
- 2003년 SMT 시스템에서 신경망이 처음 도입된 후(Bengio et al., 2003) SMT 시스템에서 신경망이 널리 사용됨
  - 특히 최근에는 신경망을 학습시켜서 번역된 문장의 점수를 매기는 데 관심을 가짐

### 3.1 Scoring Phrase Pairs with RNN Encoder-Decoder
- RNN 인코더-디코더를 학습시킬 때 원본 말뭉치에서 각 구 쌍(phrase pair)의 빈도는 무시
  - 1) 계산 비용을 줄이기 위해서
  - 2) 단순히 발생 빈도 순위를 가지고 구문의 내용을 유추하는 것을 방지하기 위해
- 모델이 언어적 규칙성을 학습하는데 초점을 맞춤
- 기존 phrase table을 완전히 대체 가능

### 3.2 Related Approaches: Neural Networks in Machine Translation
연관된 선행 연구 내용
- (Schwenk, 2012)
  - RNN 대신 feedforward network를 사용 (고정 길이 input, output)
- (Delvin et al., 2014)
  - 마찬가지로 feedforward network를 translation model에 사용
  - 한 번에 대상 문구의 한 단어씩만 예측
  - 성능이 향상되었지만 아직도 input의 최대 길이를 미리 지정해줘야 하는 단점이 존재
  
- (Zou et al., 2013)
  - 단어/구의 두 언어의 임베딩을 학습시키는 것을 제시
  - 학습된 임베딩을 사용하여 어구 쌍 간의 거리를 계산하는 과정을 SMT 시스템에 활용
  
- (Chander et al., 2014)
  - input과 output 간의 매핑을 통해 feedforward network를 학습
- 이외에도 유사한 연구 여러개 소개, 유사한 연구 간 차이점 소개(단어 순서 상관 여부 등)

- (Kalchbrenner and Blunsom, 2013)
  - 이 논문과 가장 유사한 연구
  - 인코더, 디코더와 유사한 모델을 제안
  - 인코더로 convolutional n-gram model(CGM), 디코더로는 inverse CGM과 RNN을 혼합하여 사용

## 4. Experiments
- 영어 프랑스간 번역을 평가(WMP'14 workshop)
### 4.1 Data and Baseline System
- 대충 엄청 많은 데이터를 사용했다는 내용

- 일반적으로 통계적 모델에 엄청나게 큰 데이터를 사용하는 것이 높은 성능을 보장하지도 않고 결과를 다루기도 어렵다.
  - 대신 가장 적절한 subset만 사용
- 저자들은 Data Selection 기법 사용

- 네트워크 학습 시 자주 등장하는 15000개의 단어만 이용
  - 전체 데이터셋의 약 93%를 커버
  - 해당하지 않는 단어들은 특별한 토큰인 [UNK]로 사용

- baseline 시스템(RNN을 사용하지 않은)은 개발 세트에서 30.64, 테스트 세트에서 33.3의 BLEU 기록

#### 4.1.1 RNN Encoder-Decoder
- 1000개의 hidden unit
- 각 단어를 100차원의 벡터로 임베딩
- 활성화 함수로 tanh 사용
- 디코더에서 출력으로 이어지는 부분은 DNN으로 구현
- 가중치들은 평균 0, 표준편차 0.01의 가우시안 분포로 초기화
  - recurrent 가중치 파라미터는 제외
    - recurrent 가중치 파라미터의 경우 백색 가우시안 분포에서 샘플링하여 왼쪽 특이 행렬(left singular matrix) 사용 (Saxe et al., 2014)
  
- optimizer로 Adadelta와 SGD 사용
- 각 업데이트에는 무작위로 선택된 phrase pair 쌍 64개 사용
- 학습에는 약 3일 소요

#### 4.1.2 Neural Language Model
- 성능 비교를 위해 이전에 사용되던 방식(CLSM)도 시도
  - 여러 기술을 중복해서 사용했을 때 추가적인 성능 향상이 있는지도 조사
- 우선 저자들은 7-grams CSLM 모델을 학습
    - 학습 이후 perplexity는 45.80 기록

### 4.2 Quantitative Analysis
- 다음과 같은 4가지 조합 시도
    1. baseline
    2. baseline + RNN
    3. baseline + CSLM + RNN
    4. baseline + CSLM + RNN + word penalty

- 신경망을 사용한 계산을 추가할 수록 성능이 점진적으로 향상됨
- 가장 우수한 성적을 거둔 모델은 CSLM과 RNN Encoder-Decoder를 모두 사용한 것
  - 이것은 두 방법간 상관관계가 크지 않다는 것을 시사

### 4.3 Qualitative Analysis
- 기존의 번역 모델은 말뭉치의 phrase pair의 통계에만 의존
  - 자주 나오는 문구에 대해선 잘 평가하지만 희귀한 문구에 대해서는 나쁘게 평가
- 반면 RNN Encoder-Decoder는 빈도에 대해서 학습하지 않기 때문에 언어 규칙성에 대해 점수를 매길 것으로 기대

- 4~5개의 단어로 구성된 무작위 phrase에 대한 번역 결과를 비교했을 때 RNN Encoder-Decoder의 결과가 실제 번역과 더 유사

### 4.4 Word and Phrase Representations
- 저자들이 제안한 RNN Encoder-Decoder는 기계 번역만을 위해서 디자인된 모델이 아님
- 신경망을 통해서 학습된 언어 모델이 의미 있는 임베딩을 생성한다는 사실은 이미 알려진 내용
- 생성된 임베딩을 2차원으로 투영하여 시각화한 결과 의미론적으로 유사한 단어들끼리 모여있는 것을 확인
  - Barnes-Hut-SNE 사용 시각화

## 5. Conclusion
- RNN Encoder-Decoder라는 새로운 신경망 구조 제안
  - 임의 길이의 시퀀스에서 다른 시퀀스로 매핑을 학습할 수 있음
- RNN의 새로운 Hidden Unit을 제안
- 여러가지 테스트에서 우수한 성능을 거둠
- 제안한 구조는 무궁무진한 잠재력을 가지고 있다.