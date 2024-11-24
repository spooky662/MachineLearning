import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Carregar os dados
dados = pd.read_csv('winequality-red.csv')  # Substitua pelo caminho correto do seu arquivo

# Checar as primeiras linhas do dataset para garantir que as colunas estão corretas
print(dados.head())

# Carregar variáveis para treinamento
fixed_acidity = dados['fixed acidity'].values
volatile_acidity = dados['volatile acidity'].values
citric_acid = dados['citric acid'].values
residual_sugar = dados['residual sugar'].values
chlorides = dados['chlorides'].values
free_sulfur_dioxide = dados['free sulfur dioxide'].values
total_sulfur_dioxide = dados['total sulfur dioxide'].values
density = dados['density'].values
pH = dados['pH'].values
sulphates = dados['sulphates'].values
alcohol = dados['alcohol'].values

# Usando a coluna 'quality' como a variável de saída (rótulo)
# Vamos assumir que estamos tentando prever se a qualidade do vinho é boa ou ruim (binarização)
# 0 -> Ruim, 1 -> Boa

# Para simplificar, vamos categorizar 'quality' em boa (>= 6) ou ruim (< 6)
y = (dados['quality'] >= 6).astype(int).values  # Binarizando a qualidade

# Organizando as variáveis de entrada
X = np.vstack((fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
               free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol)).T

# Normalização dos dados
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Criar o modelo com TensorFlow (Keras)
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(X_normalized.shape[1],)),  # Número de entradas
    tf.keras.layers.Dense(8, activation='relu'),  # Primeira camada oculta (8 neurônios)
    tf.keras.layers.Dense(5, activation='relu'),  # Segunda camada oculta (5 neurônios)
    tf.keras.layers.Dense(3, activation='relu'),  # Terceira camada oculta (3 neurônios)
    tf.keras.layers.Dense(1, activation='sigmoid')  # Camada de saída (1 neurônio, pois é binário)
])

# Compilar o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinar o modelo
history = model.fit(X_normalized, y, epochs=10, batch_size=32, verbose=1)

# Plotar o erro (loss) por época
plt.plot(history.history['loss'])
plt.title("Erros por Época")
plt.xlabel("Épocas")
plt.ylabel("Erros")
plt.grid()
plt.show()

# Exibir a acurácia final
print("Precisão final:", history.history['accuracy'][-1])
