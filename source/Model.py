import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense


path_click = "../Data/click.csv"
path_non_click = "../Data/non_click.csv"

# Xử lý dữ liệu
df_click = pd.read_csv(path_click)
df_non_click = pd.read_csv(path_non_click)
df_click.columns = range(df_click.shape[1])
df_non_click.columns = range(df_non_click.shape[1])
df_click['label'] = 1
df_non_click['label'] = 0

df_combined = pd.concat([df_click, df_non_click], ignore_index=True)
df_combined.dropna()

X = df_combined.drop(columns=['label']).values
y = df_combined['label'].values

# Chuẩn bị dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Xây dựng mô hình
model = Sequential()
model.add(Dense(42 ,activation = "relu", input_shape = (X_train.shape[1],)))
model.add(Dense(28 , activation = "relu"))
model.add(Dense((1), activation = "sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, epochs=20, batch_size=32)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')


# Lưu model
# model.save('../Data/my_model.h5')




