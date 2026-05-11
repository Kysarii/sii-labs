import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv('data/ml_moscow_flats.csv', sep=',')

print(df.head(10))

moscow_center_coordinates = [55.7558, 37.6173]

R = 6371000

lat1 = np.radians(df["latitude"].to_numpy())
lon1 = np.radians(df["longitude"].to_numpy())
lat2 = np.radians(moscow_center_coordinates[0])
lon2 = np.radians(moscow_center_coordinates[1])

dlat = lat2 - lat1
dlon = lon2 - lon1

a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance_m = R * c

df['distance_to_center_m'] = distance_m
df['distance_to_center_m'] = df['distance_to_center_m'].round(0)

labelencoder = LabelEncoder()
standartscaler = StandardScaler()
df['wallsMaterial'] = labelencoder.fit_transform(df['wallsMaterial'])
to_scale = ['floorNumber', 'floorsTotal', 'totalArea', 'kitchenArea', 'distance_to_center_m', 'wallsMaterial']
df[to_scale] = standartscaler.fit_transform(df[to_scale])

print(df.head(10))

y = df['price']

features = [
            'wallsMaterial', 
            'floorNumber', 
            'floorsTotal', 
            'totalArea', 
            'kitchenArea',
            'distance_to_center_m',
           ]

X = df[features]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.3, random_state=42
)

rf_model = RandomForestRegressor(n_estimators=2000, 
                                 n_jobs=-1,  
                                 bootstrap=False,
                                 criterion='friedman_mse',
                                 max_features=3,
                                 random_state=1,
                                 max_depth=55,
                                 min_samples_split=5
                                 )

rf_model.fit(X_train, y_train)
rf_prediction = rf_model.predict(X_valid).round(0)

print('\nПервые предсказания для test:')
print(np.round(rf_prediction[:10], 6))

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def median_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.median(np.abs((y_true - y_pred) / y_true)) * 100

def print_metrics(prediction, val_y):
    val_mae = mean_absolute_error(val_y, prediction)
    median_AE = median_absolute_error(val_y, prediction)
    r2 = r2_score(val_y, prediction)

    print('')
    print('R\u00b2: {:.2}'.format(r2))
    print('')
    print('Средняя абсолютная ошибка: {:.3} %'.format(mean_absolute_percentage_error(val_y, prediction)))
    print('Медианная абсолютная ошибка: {:.3} %'.format(median_absolute_percentage_error(val_y, prediction)))

print_metrics(rf_prediction, y_valid)

plt.figure(figsize=(6, 5))
plt.scatter(y_valid, rf_prediction, alpha=0.4, color='steelblue')
mn = min(y_valid.min(), rf_prediction.min())
mx = max(y_valid.max(), rf_prediction.max())
plt.plot([mn, mx], [mn, mx], 'r--', label='Идеал: y=x')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Настоящая цена')
plt.ylabel('Предсказанная цена')
plt.title('Факт vs предсказание (log-log)')
plt.legend()
plt.tight_layout()
plt.show()

residuals = y_valid - rf_prediction
plt.figure(figsize=(6, 4))
plt.scatter(rf_prediction, residuals, alpha=0.35)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Предсказанная цена')
plt.ylabel('Остаток (y_true - y_pred)')
plt.title('График остатков')
plt.tight_layout()
plt.show()