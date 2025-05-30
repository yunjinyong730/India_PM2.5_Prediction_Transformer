# India_PM2.5_Prediction_Transformer
미세먼지의 하루를 예측하다 : 시간대별 PM2.5 Transformer 예측 모델 

## ABSTRACT
PM2.5는 계절성과 시간대에 따라 고농도 패턴이 반복되며, 복합적이고 비선형적인 요인에 영향을 받는 특성이 있음 이를 효과적으로 예측하기 위해 Transformer 기반의 딥러닝 시계열 모델을 활용한 분석 및 조기 경고 시스템 구축이 유효함

## Specific Objectives
본 연구는 특정 시간 단위에서의 PM2.5 농도를 예측할 수 있는 알고리즘을 구축하는 것을 주된 목표로 함
이를 통해 도시의 공기질 변화에 대한 사전 대응, 보건 경고 시스템, 건강 관리 서비스 등 다양한 분야에서 응용될 수 있음
예측에 활용되는 주요 변수는 시간 정보(연도, 월, 일, 시간)이며, 이들을 통해 시계열적 변화를 모델링함

## Data Examination
사용 데이터 셋 : Air Quality in India (2017 – 2022)
출처: https://www.kaggle.com/code/mattop/air-quality-in-india-2017-2022-eda/input 
데이터셋은 인도에서의 시간별 대기질(PM2.5)을 기록한 정보로 구성되어 있습니다. 총 36,192개의 행과 6개의 열이 있으며, 열은 다음 내용과 같음
분석 대상 데이터는 2017년부터 2022년까지 인도 내 다양한 지역에서 시간 단위로 수집된 공기질 기록으로, 약 3만6천여 개의 시계열 항목으로 구성되어 있음
각 데이터 포인트는 날짜와 시간, 해당 시점의 PM2.5 농도를 포함하고 있으며, 이를 통해 시계열 예측을 위한 다양한 시간 기반 특성을 추출할 수 있음

![월시간별PM2 5평균히트맵](https://github.com/user-attachments/assets/dc9e123d-a30f-4282-8343-a33852746c22)

![RplotD3](https://github.com/user-attachments/assets/b9eee040-bdb0-4fc0-85fe-84b90bf136d1)


전반적으로 PM2.5 농도는 계절, 시간대, 연도 등 다양한 요인에 의해 복합적으로 영향을 받으며,
그 변화는 일정한 패턴을 따라 반복되는 경향이 있음. 이러한 특성은 단순 통계 분석보다는 시계
열 기반 머신러닝 혹은 딥러닝 모델을 통해 보다 정밀하게 예측할 수 있으며, 실제로 Transformer
구조와 같은 딥러닝 모델은 비선형성과 장기 의존성을 동시에 처리하는 데 적합한 대안이 될 수
있음. 데이터를 기반으로 한 시각적 해석과 모델링은 향후 대기 질 예보 및 환경 정책 수립의 중
요한 의사결정 도구로 활용될 수 있을 것임

## PM2.5 예측 Transformer 모델 구성
1. 전체 구조 개요
시계열 데이터(인도의 대기질 데이터)를 사용하여 PM2.5 농도를 예측하는 Transformer 기반 딥러
닝 모델. Transformer는 원래 NLP를 위해 개발되었지만, 시계열 예측에도 뛰어난 성능을 보임

2. 데이터 전처리
def load_and_preprocess_data(file_path):
주요 특징:
• 시간 정보를 datetime 형식으로 변환
• 요일(0-6)과 연중 일수(1-365)를 추가 특성으로 생성
    2 – 1. 주기적 특성 인코딩
        중요한 이유:
        • 시간(0-23)과 월(1-12)은 순환적 특성을 가짐
        • 23시와 0시는 실제로는 가까운데, 숫자로는 멀리 떨어져 있음
        • sin/cos 변환으로 이런 순환성을 모델이 이해할 수 있게 만듦
    2 - 2. create_sequences 함수
        동작 원리:
        • 슬라이딩 윈도우 방식으로 시퀀스 생성
        • 예: seq_length=24면 과거 24시간 데이터로 다음 시간을 예측
        • target은 PM2.5 값만 추출 (인덱스 0)

3. Transformer 핵심 구성 요소
    3-1. Positional Encoding 클래스
        수식 설명:
        • 각 위치와 차원에 대해 고유한 인코딩 생성
        • 10000은 주기를 결정하는 상수 (큰 값일수록 긴 주기)
        왜 필요한가?
        • RNN과 달리 Transformer는 순서 정보가 없음
        • Positional Encoding으로 시간 순서 정보를 추가함
    3-2. Multi-Head Attention 클래스
        핵심 개념:
        • Query, Key, Value 행렬을 생성
        • 여러 개의 attention head로 분할
        • 각 head는 서로 다른 관계를 학습
    
    3-3. Scaled Dot-Product Attention
         Attention(Q,K,V) = softmax(QK^T/√dk)V
         √dk로 나누는 이유: gradient vanishing 방지
         각 시점이 다른 시점들과 얼마나 관련있는지 계산
    
    3-4. Transformer Block
        구성 요소:
        1. Multi-Head Attention: 시계열 패턴 학습
        2. Feed Forward Network: 비선형 변환
        3. Residual Connection: 학습 안정성
        4. Layer Normalization: 각 층의 출력 정규화
        5. Dropout: 과적합 방지


4. 전체 Transformer 모델 
    4-1. 모델 구조
        주요 하이퍼파라미터:
        • d_model=64: 임베딩 차원
        • num_heads=4: Attention head 개수
        • dff=256: Feed forward 은닉층 크기
        • num_blocks=2: Transformer 블록 개수
        • pred_length=1: 예측할 시간 단계 수

5. 학습 프로세스
    5-1. 데이터 준비
        정규화의 중요성:
        • 각 특성의 스케일을 통일
        • 학습 속도 향상 및 안정성 확보
        • 나중에 역정규화로 실제 값 복원
    5-2. 학습 최적화
        콜백 전략:
        • Early Stopping: 과적합 방지, 최적 가중치 복원
        • Learning Rate Reduction: 학습이 정체될 때 학습률 감소


6. 학습 과정 분석
    학습 수렴 패턴
    • 초기 학습 (Epoch 1-10)
    o Loss: 0.1013 → 0.0461 (54.5% 감소)
    o MAE: 0.2208 → 0.1462 (33.8% 감소)
    o 매우 빠른 수렴 속도를 보임
    • Learning Rate 조정
    o Epoch 11: 0.001 → 0.0005 (첫 번째 감소)
    o Epoch 22: 0.0005 → 0.00025 (두 번째 감소)
    o ReduceLROnPlateau가 효과적으로 작동

   
## 최종 성능

![Figure_1](https://github.com/user-attachments/assets/d2a20fd6-5d55-49b2-9e86-34b7d6728b76)

  • Test Loss: 0.0276 (매우 낮음)
  • Test MAE: 0.1086
  • 정규화된 값 기준이므로, 실제 PM2.5 단위로는 약 5.36 μg/m³의 평균 절대 오차 (평균
  PM2.5가 49.3이므로, 0.1086 × 49.3 ≈ 5.36)
