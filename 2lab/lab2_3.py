import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import tracemalloc
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.metrics import silhouette_score

df = pd.read_csv('datasets/Towns10.csv', sep=';', encoding='utf-8')

cities = df['City'].tolist()
lat = df['Latitude'].tolist()
lon = df['Longitude'].tolist()
n = len(cities)

features = df[['Latitude', 'Longitude']].values

max_k = min(10, n)
k_values = list(range(1, max_k + 1))
inertias = []

print("Расчет инерции для различных K значений...")
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(features)
    inertia = kmeans.inertia_
    inertias.append(inertia)
    print(f"K={k}: Инерция = {inertia:.4f}")

choice = input("Выберите способ определения K (1 - автоматический метод локтя, 2 - вручную): ")

if choice == '1':
    kneedle = KneeLocator(k_values, inertias, curve='convex', direction='decreasing')
    K = kneedle.elbow if kneedle.elbow is not None else 2
    print(f"Автоматически выбрано K = {K}")
else:
    K = int(input("Введите количество кластеров: "))

start_time = time.perf_counter()
tracemalloc.start()

kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
df['Cluster_ID'] = kmeans.fit_predict(features)

end_time = time.perf_counter()
current_mem, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()

plt.figure(figsize=(10, 6))
plt.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Количество кластеров (K)', fontsize=12)
plt.ylabel('Инерция', fontsize=12)
plt.title('Метод локтя - нахождение оптимальных K', fontsize=14)
plt.xticks(k_values)
plt.grid(True, alpha=0.3)
plt.axvline(x=K, color='r', linestyle='--', label=f'Оптимальное K = {K}')
plt.legend()
plt.tight_layout()
plt.show()

print(f"\nОптимальное число кластеров: K = {K}")

kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
df['Cluster_ID'] = kmeans.fit_predict(features)
centers = kmeans.cluster_centers_

print(f"\nРаспределение городов на {K} кластеров:\n")
for k in range(K):
    cluster_df = df[df['Cluster_ID'] == k]
    cluster_names = cluster_df['City'].tolist()
    print(f"Кластер {k + 1}:")
    print(f"  Города: {', '.join(cluster_names)}")
    print(f"  Количество: {len(cluster_names)}")

colors = plt.cm.tab10(np.linspace(0, 1, K))

plt.figure(figsize=(8, 6))
for cluster_idx in range(K):
    cluster_df = df[df['Cluster_ID'] == cluster_idx]
    cluster_lats = cluster_df['Latitude'].tolist()
    cluster_lons = cluster_df['Longitude'].tolist()
    plt.scatter(cluster_lons, cluster_lats, c=[colors[cluster_idx]], s=100, label=f'Кластер {cluster_idx + 1}')
    
    #for _, row in cluster_df.iterrows():
        #plt.annotate(row['City'], (row['Longitude'], row['Latitude']), fontsize=9, ha='center', va='bottom')

plt.scatter(centers[:, 1], centers[:, 0], c='black', marker='X', s=200, label='Центроиды')

print(f"\nВремя выполнения алгоритма кластеризации: {end_time - start_time:.6f}s")
print(f"Пиковое количество памяти: {peak_mem / 1024:.4f} KB")

labels = df['Cluster_ID'].values
features = df[['Latitude', 'Longitude']].values
silhouette = silhouette_score(features, labels)
print(f"Silhouette Score: {silhouette:.4f}")

plt.xlabel('Долгота')
plt.ylabel('Широта')
plt.title(f'Результат кластеризации (K = {K})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
