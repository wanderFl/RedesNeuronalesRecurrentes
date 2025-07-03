import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Carga del CSV
df = pd.read_csv('Mall_Customers.csv')

# 2. Selección de variables relevantes
data = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# 3. Normalización a rango [0,1]
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# 4. Creación de secuencias para RNN
def create_sequences(data, time_steps=3):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), :])
        y.append(data[i + time_steps, :])
    return np.array(X), np.array(y)

time_steps = 3
X, y = create_sequences(data_scaled, time_steps)

# 5. División en conjuntos de entrenamiento y prueba
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 6. Definición del modelo RNN (LSTM)
model = Sequential([
    LSTM(50, input_shape=(time_steps, X.shape[2])),
    Dense(X.shape[2])
])
model.compile(optimizer='adam', loss='mse')

# 7. Entrenamiento del modelo
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=1
)

# 8. Gráfica de pérdida de entrenamiento y validación
plt.figure()
plt.plot(history.history['loss'], label='Pérdida entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida validación')
plt.title('Evolución de la pérdida por época')
plt.xlabel('Época')
plt.ylabel('MSE')
plt.legend()
plt.tight_layout()
plt.show()

# 9. Predicción de ejemplo y des-normalización
sample_input = X_test[0].reshape(1, time_steps, X.shape[2])
pred_scaled = model.predict(sample_input)
pred = scaler.inverse_transform(pred_scaled)
actual = scaler.inverse_transform(y_test[0].reshape(1, -1))
print(f"Predicción [Annual Income, Spending Score]: {pred.flatten()}")
print(f"Real      [Annual Income, Spending Score]: {actual.flatten()}")

# 10. Guardar el modelo (opcional)
model.save('rnn_mall_customers.h5')
