from sklearn.model_selection import train_test_split, KFold
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

train_path = '/content/ETH-USD-Train.csv'
test_path = '/content/ETH-USD-Test.csv'

df_train = pd.read_csv(train_path, infer_datetime_format=True)
df_test = pd.read_csv(test_path, infer_datetime_format=True)
df_train['Date'] = pd.to_datetime(df_train.Date)
dataset = df_train.loc[:, ['Close']].values
dataset = dataset.reshape(-1, 1)
scaler = MinMaxScaler(feature_range = (0, 1))
data_scaled = scaler.fit_transform(dataset)

X_train = []
y_train = []
time_step = 10
for i in range(len(data_scaled) - time_step - 1):
    a = data_scaled[i:(i + time_step), 0]
    X_train.append(a)
    y_train.append(data_scaled[i + time_step, 0])
X_train = np.array(X_train)
y_train = np.array(y_train)
print("trainX shape: {}\ntrainY shape: {}". format(X_train.shape, y_train.shape))
print(X_train[0])

layer = [
    LSTMLayer(32, 256),
    DenseLayer(1,"linear")
]
model = ModelLSTM(layer)
model.fit(X_train)
