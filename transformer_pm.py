import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 데이터 로드 및 전처리
def load_and_preprocess_data(file_path):
    """데이터를 로드하고 전처리합니다."""
    df = pd.read_csv(file_path)
    
    # 시간 관련 특성 추가
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['day_of_week'] = df['Timestamp'].dt.dayofweek
    df['day_of_year'] = df['Timestamp'].dt.dayofyear
    
    # 주기적 특성을 sin/cos로 인코딩
    df['hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    
    return df

def create_sequences(data, seq_length, pred_length):
    """시계열 데이터를 시퀀스로 변환합니다."""
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length - pred_length + 1):
        seq = data[i:i + seq_length]
        target = data[i + seq_length:i + seq_length + pred_length, 0]  # PM2.5만 예측
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)

# Positional Encoding
class PositionalEncoding(layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.position = position
        self.d_model = d_model
        
    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles
    
    def call(self, inputs):
        seq_length = tf.shape(inputs)[1]
        position = tf.range(seq_length, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(self.d_model, dtype=tf.float32)[tf.newaxis, :]
        
        angle_rads = self.get_angles(position, i, self.d_model)
        
        # 짝수 인덱스에는 sin, 홀수 인덱스에는 cos 적용
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        
        return inputs + pos_encoding

# Multi-Head Attention
class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % num_heads == 0
        
        self.depth = d_model // num_heads
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        
        self.dense = layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # Scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))
        
        output = self.dense(concat_attention)
        
        return output

# Transformer Block
class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, inputs, training):
        attn_output = self.mha(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2

# Transformer Model for Time Series
class TransformerTimeSeries(keras.Model):
    def __init__(self, num_features, d_model, num_heads, dff, num_blocks, pred_length, dropout_rate=0.1):
        super(TransformerTimeSeries, self).__init__()
        
        self.d_model = d_model
        self.num_blocks = num_blocks
        
        # 입력 projection
        self.input_projection = layers.Dense(d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(500, d_model)
        
        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_blocks)
        ]
        
        # 출력 layers
        self.global_pool = layers.GlobalAveragePooling1D()
        self.output_projection = keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(64, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(pred_length)
        ])
        
        self.dropout = layers.Dropout(dropout_rate)
    
    def call(self, inputs, training=False):
        # 입력 projection
        x = self.input_projection(inputs)
        
        # Positional encoding 추가
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        
        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training)
        
        # Global pooling
        x = self.global_pool(x)
        
        # 출력 예측
        outputs = self.output_projection(x)
        
        return outputs

# 모델 학습 및 평가 함수
def train_model(file_path, seq_length=24, pred_length=1, epochs=50, batch_size=32):
    """모델을 학습하고 평가합니다."""
    
    # 데이터 로드 및 전처리
    df = load_and_preprocess_data(file_path)
    
    # 특성 선택
    feature_columns = ['PM2.5', 'Hour', 'Day', 'Month', 'Year', 
                      'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
    data = df[feature_columns].values
    
    # 데이터 정규화
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # 시퀀스 생성
    X, y = create_sequences(data_scaled, seq_length, pred_length)
    
    # 학습/검증/테스트 데이터 분할
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.111, random_state=42)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # 모델 생성
    model = TransformerTimeSeries(
        num_features=X_train.shape[2],
        d_model=64,
        num_heads=4,
        dff=256,
        num_blocks=2,
        pred_length=pred_length,
        dropout_rate=0.1
    )
    
    # 모델 컴파일
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    # 콜백 설정
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001
    )
    
    # 모델 학습
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # 테스트 데이터 평가
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    # 예측 수행
    predictions = model.predict(X_test[:100])
    
    # 역정규화 (PM2.5 값만)
    pm25_mean = scaler.mean_[0]
    pm25_std = scaler.scale_[0]
    
    predictions_original = predictions * pm25_std + pm25_mean
    y_test_original = y_test[:100] * pm25_std + pm25_mean
    
    # 결과 시각화
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(y_test_original[:50], label='Actual', marker='o')
    plt.plot(predictions_original[:50], label='Predicted', marker='s')
    plt.title('PM2.5 Predictions vs Actual')
    plt.xlabel('Time Step')
    plt.ylabel('PM2.5')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return model, scaler, history

# 사용 예시
if __name__ == "__main__":
    # 모델 학습
    model, scaler, history = train_model(
        'data/air-quality-india.csv',
        seq_length=24,  # 과거 24시간 데이터 사용
        pred_length=1,  # 다음 1시간 예측
        epochs=50,
        batch_size=32
    )
    
    # 모델 저장
    model.save('pm25_transformer_model.keras')
    
    # 다중 스텝 예측을 위한 함수
    def predict_multiple_steps(model, initial_sequence, scaler, steps=24):
        """여러 시간 단계를 예측합니다."""
        predictions = []
        current_sequence = initial_sequence.copy()
        
        for _ in range(steps):
            # 예측
            pred = model.predict(current_sequence[np.newaxis, ...], verbose=0)[0]
            predictions.append(pred[0])
            
            # 시퀀스 업데이트
            new_row = current_sequence[-1].copy()
            new_row[0] = pred[0]  # PM2.5 값 업데이트
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        # 역정규화 (PM2.5 값만)
        predictions = np.array(predictions).reshape(-1, 1)
        # 전체 데이터의 scaler에서 PM2.5 컬럼의 평균과 표준편차를 사용
        pm25_mean = scaler.mean_[0]
        pm25_std = scaler.scale_[0]
        predictions_original = predictions * pm25_std + pm25_mean
        
        return predictions_original