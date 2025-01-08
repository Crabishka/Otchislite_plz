import matplotlib
import numpy as np
import statsmodels as sm
import sns as sn
import tensorflow as tf
import seaborn as sb
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from keras.src.callbacks import ModelCheckpoint
from keras.src.layers import Input, SimpleRNN, LSTM, GRU, Dense, Dropout, RepeatVector
from keras.src import Model, Input
from keras.src.utils import to_categorical
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

matplotlib.use('TkAgg')

df = pd.read_csv('content/silver.csv')
print(df[:5])

df['date'] = pd.to_datetime(df['date'])

print("\nСтатистика по числовым данным:")
print(df.describe().transpose())

# print("Проверка на пустые значения:")
# print(df.isnull().sum())

# fig = px.scatter_geo(df[:10000], lat='latitude', lon='longitude', color='mag')
#
# fig.show()

# plt.hist(df['mag'], bins=40, color='blue', edgecolor='black')
# plt.xlabel('Mag')
#
# plt.show()

# df['mag_rounded_down'] = df['mag'].astype(int)
#
# df.groupby('mag_rounded_down').agg({'id': 'count'})

df['year'] = df['date'].dt.year

# df_filtered = df[df['year'].isin([2011, 2012])]


df['longitude_band'] = df['longitude'].astype(int)
df['latitude_band'] = df['latitude'].astype(int)
df['square'] = df['longitude_band'].astype(str) + '_' + df['latitude_band'].astype(str)
df['month'] = df['date'].dt.month
df['square_cat'] = pd.Categorical(df['square']).codes

print(df)
# df_filtered = df[
#     (df['latitude'] >= 34) & (df['latitude'] <= 35) & (df['longitude'] >= -119) & (df['longitude'] <= -118) & (
#                 df['year'] == 1994)]
# plt.hist(df_filtered['date'], bins=365)

# plt.show()

print(df[:5])
print(df.dtypes)
corr = df[['latitude', 'longitude', 'depth', 'mag', 'hour', 'year', 'month', 'square_cat']].corr()

# plt.figure(figsize=(10, 8))
# sb.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
# plt.title('Корреляционная матрица')
# plt.show()

df = df.drop(columns=['square_cat', 'longitude_band', 'latitude_band', 'square', 'month', 'year'])

# combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

# combined_df['data_source'] = ['train_df' if i < len(train_df) else
#                               'val_df' if i < len(train_df) + len(val_df) else
#                               'test_df' for i in range(len(combined_df))]
#
# sampled_df = combined_df[::1000]
#
# fig = px.scatter_geo(sampled_df, lat='latitude', lon='longitude',
#                      color_discrete_sequence=['red', 'blue', 'green'],
#                      symbol='data_source',
#                      )
# fig.show()

scaler = MinMaxScaler()

exclude_cols = ['id', 'date']
cols_to_normalize = [col for col in df.columns if col not in exclude_cols]
df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
target_scaler = MinMaxScaler(feature_range=(0, 1))

# целевые
df['latitude'] = target_scaler.fit_transform(df[['latitude']])
df['longitude'] = target_scaler.fit_transform(df[['longitude']])
df['mag'] = target_scaler.fit_transform(df[['mag']])

selected_columns = ['latitude', 'longitude', 'mag']

n = len(df)
train_df = df[0:int(n * 0.8)]
val_df = df[int(n * 0.8):int(n * 0.9)]
test_df = df[int(n * 0.9):]


# Функция для создания окон
def create_sequences(df, window_size, step_size):
    X, y = [], []
    for i in range(window_size, len(df) - window_size + 1, step_size):
        X.append(df.iloc[i - window_size:i].drop(columns=['id', 'date']).values)
        y.append(df.iloc[i:i + window_size][selected_columns])
    return np.array(X), np.array(y)


window_size = 100

train_val_step = 100

X_train, y_train = create_sequences(train_df, window_size, step_size=train_val_step)
X_val, y_val = create_sequences(val_df, window_size, step_size=train_val_step)
X_test, y_test = create_sequences(test_df, window_size, step_size=window_size)

# Убедимся, что формы данных правильные
print("Размерность обучающей выборки: " + str(X_train.shape) + str(y_train.shape))
print("Размерность валидационной выборки: " + str(X_val.shape) + str(y_val.shape))
print("Размерность тестовой выборки: " + str(X_test.shape) + str(y_test.shape))

# sample_idx = 0
# x_window = pd.DataFrame(X_test[sample_idx], columns=['latitude', 'longitude', 'depth', 'mag', 'hour', 'date'])
# y_window = pd.DataFrame(y_test[sample_idx], columns=['latitude', 'longitude', 'mag'])
# combined_df = pd.concat([x_window, y_window], ignore_index=True)
#
# # Выведем пример с окном из тестового набора (возьмём первую временную последовательность)
# combined_df['data_source'] = ['test_x' if i < len(x_window) else
#                               'text_y' for i in range(len(combined_df))]
#
# sampled_df = combined_df
#
# fig = px.scatter_geo(sampled_df, lat='latitude', lon='longitude',
#                      color_discrete_sequence=['red', 'blue', 'green'],
#                      symbol='data_source',
#                      )
# fig.show()

input_shape = X_train.shape[1:3]
num_targets = 3


def createRecurrentNetwork(input_shape):
    input_x = Input(shape=input_shape)
    lstm1 = LSTM(64, return_sequences=True)(input_x)
    lstm2 = LSTM(32, return_sequences=True)(lstm1)
    out = Dense(num_targets)(lstm2)
    return Model(inputs=input_x, outputs=out)


def trainModel(X_train, y_train, X_val, y_val):
    # Сохранение только наилучшей модели с наименьшей ошибкой на валидации
    save_callback = ModelCheckpoint(filepath='best_model.weights.h5',
                                    monitor='val_loss',
                                    save_best_only=True,
                                    save_weights_only=True,
                                    mode='min',
                                    verbose=1)  #
    model = createRecurrentNetwork(
        input_shape)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='mean_squared_error')
    model.summary()
    model.fit(x=X_train,
              y=y_train,
              epochs=50,
              # batch_size=32,
              validation_data=(X_val, y_val),
              callbacks=[save_callback]
              )
    return model


model = createRecurrentNetwork(input_shape)
# model = trainModel(X_train, y_train, X_val, y_val)

# plt.figure()
# plt.plot(model.history.history["loss"], label="training loss")
# plt.plot(model.history.history["val_loss"], label="validation loss")
# plt.legend()
# plt.title("График изменения ошибки модели")
# plt.xlabel("Эпохи")
# plt.ylabel("Ошибка")
# plt.show()

model.load_weights('best_model.weights.h5')

from sklearn.metrics import mean_absolute_percentage_error

# Прогноз на тестовых данных
y_test_pred = model.predict(X_test).squeeze()

# Обратная нормализация только для целевой переменной
new_y_test_pred = target_scaler.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()
new_y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Расчет MAPE
mape = mean_absolute_percentage_error(new_y_test_pred, new_y_test_actual) * 100
print(f"Модель ошибается в среднем на {mape:.2f}% от истинных значений")
#
sample_idx = 0
x_window = pd.DataFrame(y_test_pred[sample_idx], columns=['latitude', 'longitude', 'mag'])
y_window = pd.DataFrame(y_test[sample_idx], columns=['latitude', 'longitude', 'mag'])
combined_df = pd.concat([x_window, y_window], ignore_index=True)

# Выведем пример с окном из тестового набора (возьмём первую временную последовательность)
combined_df['data_source'] = ['y_test_pred' if i < len(x_window) else
                              'y_test_actual' for i in range(len(combined_df))]

sampled_df = combined_df

fig = px.scatter_geo(sampled_df, lat='latitude', lon='longitude',
                     color_discrete_sequence=['red', 'blue', 'green'],
                     symbol='data_source',
                     )
fig.show()