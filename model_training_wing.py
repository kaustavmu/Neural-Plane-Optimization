import tensorflow as tf
import keras
from keras import layers
import numpy as np
import pandas as pd

x_dataset = pd.read_csv('X.csv')
X = x_dataset.values

y_dataset = pd.read_csv('y.csv')
y = y_dataset.values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1)

'''
model = keras.Sequential(
    [
        keras.Input(shape=(5,)),
        layers.Dense(20, activation="relu", name="layer1"),
        layers.Dense(10, activation="relu", name="layer2"),
        layers.Dense(1, name="layer3"),
    ]
) #MSE -0.03757881
'''

model = keras.Sequential(
    [
        keras.Input(shape=(5,)),
        layers.Dense(15, activation="relu", name="layer1"),
        layers.Dense(30, activation="relu", name="layer2"),
        layers.Dense(6, activation="relu", name="layer3"),
        layers.Dense(1, name="layer4"),
    ]
) #MSE 0.02303611, 0.00017866 with 300 epochs

# Call model on a test input
x = tf.ones((1, 5))
y = model(x)

model.summary()
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=400, batch_size=8)

final = model.predict(X_test)

for i in range(len(final)):
    print(final[i], y_test[i])

MSE = (1/len(final)) * sum(((y_test[i])-(final[i]))**2 for i in range(len(final)))
print("MSE")
print(MSE)
print(model.predict(np.array([[0.449, 5.763, 18, 12, 1.3376998569532415]])))

model.save("PSOmodel.keras")