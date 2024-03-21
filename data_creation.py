import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Генерируем данные о количестве электроэнергии, потребляемой каждый час в течение суток
num_hours = 24
days = 365
energy_consumption = np.random.randint(100, 500, num_hours * days)

# Создаем DataFrame для хранения сгенерированных данных
date_range = pd.date_range(start='1/1/2024', periods=num_hours * days, freq='H')
data = {'date': date_range, 'energy_consumption': energy_consumption}
df = pd.DataFrame(data)

# Выводим информацию о df.DataFrame
df.info()

# Делим на обучающую и тестовую выборку
X_train, X_test, y_train, y_test = train_test_split(df['date'], df['energy_consumption'], test_size=0.1, random_state=101)
print('len(X_train):', len(X_train))
print('len(X_test):', len(X_test))

# Сохраняем обучающий набор данных
if not os.path.isdir('data'):
    os.mkdir('data')

if not os.path.isdir('data/train'):
    os.mkdir('data/train')

df_train = pd.DataFrame()
df_train['date'] = X_train
df_train['energy_consumption'] = y_train
df_train.to_csv('data/train/train.csv')

# Сохраняем тестовый набор данных
if not os.path.isdir('data/test'):
    os.mkdir('data/test')

df_test = pd.DataFrame()
df_test['date'] = X_test
df_test['energy_consumption'] = y_test
df_train.to_csv('data/test/test.csv')
