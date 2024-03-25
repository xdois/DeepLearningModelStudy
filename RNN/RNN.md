# Recurrent neural network based language model

## 0. Abstract
- 순환 신경망 기반 언어 모델 제시
- 기존 SOTA backoff 언어 모델과 비교해서 약 50% 혼란 감소
- Wall Street Journal 작업에서 동일한 데이터로 훈련된 모델에 비해서 오류율 약 18% 감소
- NIST RT05 작업에 대해서는 더 큰 데이터로 학습된 모델에 비해서도 약 5% 성능이 우수
- N-gram 기술에 비해 우수한 점을 경험적으로 제시
- 다만 계산 복잡성이 높음

## 1. Introduction
- 통계적 언어 모델링의 목표: 텍스트의 다음 단어를 예측하는 것
- 따라서 언어 모델을 sequential data 예측 문제로 볼 수 있음
- 특정한 접근 방식이 필요(parse tree 등) 
  - N-gram 기반 모델도 문장이 단어 시퀀스로 구성되고 문장 기호가 중요하다고 가정
- 캐시 기반, 클래스 기반 모델이 기존에 강력했던 방식
- 기존의 모델들은 제한된 시스템에서만 잘 작동하는 경우가 많음

## 2. Model Description
- 저자들은 sequential data 모델링을 위해 RNN(순환 신경망) 네트워크를 도입
  - 기존에 feedforward network 등을 도입하려는 시도는 있었음
  - 훈련 전에 임의의 고정 길이를 지정해 줘야 하는 문제가 있음
  - 짧은 길이를 사용했기 때문에 문맥을 파악하는데 어려움이 있었음
- 순환 신경망은 제한된 크기의 컨텍스트를 사용하지 않음
- 반복 연결을 통해 네트워크 내에서 정보가 오랫동안 순환되게 함
  - SGD를 통해 장기 종속성을 학습하는 것이 어려울 수 있다는 주장도 있음(값이 0에 가까워져서 소실?)

- 저자들은 simple recurrent neural network라는 네트워크를 사용
  - input layer $x$, hidden layer $s$, output layer $y$로 구성

  - 시간 $t$의 네트워크의 input $x(t)$, output $y(t)$, state $s(t)$
  - 현재 단어를 나타내는 벡터 $w(t)$
  
  $$
    x(t) = w(t) + s(t-1)
    \\
    s_j(t) = f\bigg(\sum_ix_i(t)u_{ji}\bigg) \\
    y_k(t) = g\bigg(\sum_js_j(t)v_{kj}\bigg)
   $$
   $f$: sigmoid function

   $g(z_m)$ : softmax function

- 초기화 시에는 작은 값을 사용
- 벡터 x의 크기는 어휘 사전의 크기 $V$에 context layer(state, hidden layer)의 크기를 더한 것
- hidden layer의 크기는 학습 데이터가 클 수록 더 커야 한다

- SGD 사용
- 초기 learning rate = 0.1
  - 성능의 개선이 없으면 절반으로
- 네트워크 규제는 큰 갯선을 제공하지 못함
  
- 출력 레이어는 이전 단어와 컨텍스트가 주어지면 다음 단어에 대한 확률 분포를 나타냄

- error(t) = desired(t) - y(t)
  - desired: 예측해야하는 단어, y: 실제 output
  - 각각의 벡터는 1-of-N code(one-hot encoding)으로 나타냄

- 네트워크가 테스트 단계 중에도 학습을 계속 할 수 있어야 한다?
  - 저자들은 이러한 모델을 dynamic하다고 표현
- dynamic model의 경우 고정 learning rate 0.1을 사용
- 훈련 단계에서는 모든 데이터가 여러 epoch에 걸쳐서 학습되지만 dynamic model의 경우 테스트 데이터를 처리할 때 한 번만 업데이트됨
  - 이 방식이 최적의 솔루션은 아니지만 정적 모델에 비해선 충분히 좋았다고 설명
  - backoff 모델의 cache 기술과 유사

- 훈련 알고리즘 : t=1인 시간에 따라 잘린 역전파(Backpropagation Through Time)
  - 순방향 신경망(feedforward network)와 주요 차이점: hyperparameter의 양?
  - RNN의 경우 히든 레이어의 크기만 선택하면 됨
  - FFN의 경우 단어를 저차원 공간에 투영하는(임베딩) 레이어의 크기, 히든 레이어의 크기, 컨텍스트의 길이를 조정해야함
  

### 2.1. Optimization
- 훈련 성능을 향상시키기 위해 임계값 보다 덜 나오는 모든 단어를 하나의 특수 토큰으로 병합
- 기존의 연구와 유사한 성능의 모델을 훨씬 빠른 시간내에 학습하였음

## 3. WSJ Experiments
- RNN 모델의 성능을 평가하기 위해 몇가지의 표준 음성 인식 태스크를 선정
- 저자들은 훈련 데이터가 커질 수록 모델이 더욱 개선되는 것을 확인
  - 사용된 지표: PPL(Perplxity), WER(Word Error Rate)
- 기존의 연구에 비해 가장 적은 데이터를 사용하면서도 가장 우수한 성능을 기록

## 4. NIST RT05 experiments
- 이전 실험에 사용된 음향 모델이 최신 기술과 거리가 멀다는 지적
- 이런 지적에서 벗어나기 위해 NIST RT05 실험을 진행
- RT05 실험에서도 우수한 성능을 기록(일부 데이터만을 활용해도 기존 모델보다 약간 우수한 성능 기록)

## 5. Conclusion And Future Work
- RNN 모델은 훨씬 더 많은 데이터로 학습된 (당시) 최신 모델에 비해 훨씬 뛰어난 성능을 기록
  - WSJ task에 대해서 약 18%의 WER 감소(같은 양의 데이터로 학습된 기존 모델), 12%의 WER 감소(많은 양의 데이터로 학습된 기존 모델)
  - NIST RT05 task에 대해서 100배 더 많은 데이터로 학습된 모델과 유사한 성능
  - 기존의 통념(언어 모델링이 n-gram 수를 세는 것에 불과하고 결과를 개선하기 위해선 새로운 데이터를 얻는 것이 유일한 방법임)을 깨트림

- BPTT(Backpropagation Through Time)에 대한 추가적인 조사를 통한 개선이 이루어질 가능성이 있음을 제시
- 기존 언어 모델을 사용하는 모든 종류의 애플리케이션에서 RNN 기반 모델을 쉽게 사용할 수 있음(번역, OCR 등)