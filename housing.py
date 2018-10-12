from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split
from neupy import algorithms, layers, environment
from neupy.estimators import rmsle
import numpy as np

# Carga el data set
data_set = datasets.load_boston()

# La variable target contiene los valores de MEDV
full_data, target = data_set.data, data_set.target

# Escogemos las variables independientes
ZN = full_data[:, 1]
RM = full_data[:, 5]
RAD = full_data[:, 8]
TAX = full_data[:, 9]
B = full_data[:, 11]

# Juntamos las variables independientes a un solo array
data = np.column_stack((ZN, RM, RAD, TAX, B))

# Normalizamos los datos
data_scaler = preprocessing.MinMaxScaler()
target_scaler = preprocessing.MinMaxScaler()

# Configuramos la semilla para reproductibilidad
environment.reproducible(47)

# Separamos datos de entrena y prueba a un 70%
x_train, x_test, y_train, y_test = train_test_split(
    data_scaler.fit_transform(data),
    data_scaler.fit_transform(target.reshape(-1, 1)),
    train_size=0.70
)

# Configuramos la red neuronal
red = algorithms.ConjugateGradient(
    connection=[
        layers.Input(5),     # input layer
        layers.Sigmoid(20),   # hidden layer
        layers.Sigmoid(1)
    ],
    show_epoch=25,
    verbose=True
)

# Entrenamos la red neuronal
red.train(x_train, y_train, x_test, y_test, epochs=100)

# Desnormalizamos datos y calculamos error
y_predict = red.predict(x_test).round(1)

error = rmsle(target_scaler.inverse_transform(y_test),
              target_scaler.inverse_transform(y_predict))

print(error)
