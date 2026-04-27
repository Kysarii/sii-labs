import csv
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import tracemalloc
from kneed import KneeLocator
from sklearn.metrics import silhouette_score

df = pd.read_csv('datasets/Towns10.csv', sep=';', encoding='utf-8')
cities = df['City'].tolist()
lat = df['Latitude'].tolist()
lon = df['Longitude'].tolist()

n = len(cities)

def euclidean_dist(lat1, lon1, lat2, lon2):
    return math.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)

def build_distance_matrix(n, lat, lon):
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n): 
            d = euclidean_dist(lat[i], lon[i], lat[j], lon[j])
            matrix[i][j] = d
            matrix[j][i] = d
    return matrix

dist_matrix = build_distance_matrix(n, lat, lon)

def compute_inertia(clusters, lat, lon):
    inertia = 0.0
    for cluster in clusters:
        if len(cluster) == 0:
            continue
        centroid_lat = sum(lat[idx] for idx in cluster) / len(cluster)
        centroid_lon = sum(lon[idx] for idx in cluster) / len(cluster)
        for idx in cluster:
            inertia += euclidean_dist(lat[idx], lon[idx], centroid_lat, centroid_lon) ** 2
    return inertia

def hierarchical_clustering(K, dist_matrix, n):
    clusters = [[i] for i in range(n)]
   
    while True:
        min_dist = float('inf')
        best_pair = (-1, -1)
       
        # Ищем самую близкую пару кластеров (по полной связи)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                cluster_a = clusters[i]
                cluster_b = clusters[j]
               
                current_max_cluster_dist = 0
                for idx_a in cluster_a:
                    for idx_b in cluster_b:
                        d = dist_matrix[idx_a][idx_b]
                        if d > current_max_cluster_dist:
                            current_max_cluster_dist = d
               
                if current_max_cluster_dist < min_dist:
                    min_dist = current_max_cluster_dist
                    best_pair = (i, j)
       
        # Условие остановки — это и есть главная правка
        if best_pair[0] == -1 or (min_dist > 0 and len(clusters) <= K):
            break
       
        # Слияние
        i, j = best_pair
        merged_cluster = clusters[i] + clusters[j]
       
        if j > i:
            clusters.pop(j)
            clusters[i] = merged_cluster
        else:
            clusters.pop(i)
            clusters[j] = merged_cluster
   
    # Если после всех слияний кластеров меньше K — добавляем пустые
    while len(clusters) < K:
        clusters.append([])
   
    return clusters

max_k = min(10, n)
k_values = list(range(1, max_k + 1))
inertias = []

print("Расчет инерции для различных K значений...")
for k in k_values:
    clusters = hierarchical_clustering(k, dist_matrix, n)
    inertia = compute_inertia(clusters, lat, lon)
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

clusters = hierarchical_clustering(K, dist_matrix, n)

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

clusters = hierarchical_clustering(K, dist_matrix, n)

print(f"\nРаспределение городов на {K} кластеров:\n")
for k, cluster_indices in enumerate(clusters):
    cluster_names = [cities[idx] for idx in cluster_indices]
    print(f"Кластер {k + 1}:")
    print(f"  Города: {', '.join(cluster_names)}")
    print(f"  Количество: {len(cluster_names)}")

colors = plt.cm.tab10(np.linspace(0, 1, K))

plt.figure(figsize=(8, 6))
for cluster_idx, cluster_indices in enumerate(clusters):
    cluster_lats = [lat[idx] for idx in cluster_indices]
    cluster_lons = [lon[idx] for idx in cluster_indices]
    plt.scatter(cluster_lons, cluster_lats, c=[colors[cluster_idx]], s=100, label=f'Кластер {cluster_idx + 1}')
    
    #for idx in cluster_indices:
        #plt.annotate(cities[idx], (lon[idx], lat[idx]), fontsize=9, ha='center', va='bottom')

print(f"Время выполнения алгоритма кластеризации: {end_time - start_time:.6f}s")
print(f"Пиковое количество памяти: {peak_mem / 1024:.4f} KB")

labels = [0] * n
for cluster_id, cluster_indices in enumerate(clusters):
    for idx in cluster_indices:
        labels[idx] = cluster_id

features = np.array(list(zip(lat, lon)))
silhouette = silhouette_score(features, labels)
print(f"Silhouette Score: {silhouette:.4f}")

plt.xlabel('Долгота')
plt.ylabel('Широта')
plt.title(f'Результат кластеризации (K = {K})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()