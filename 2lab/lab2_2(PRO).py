import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import tracemalloc
from kneed import KneeLocator
from sklearn.metrics import silhouette_score

df = pd.read_csv('datasets/Towns100.csv', sep=';', encoding='utf-8')
cities = df['City'].tolist()
lat = df['Latitude'].tolist()
lon = df['Longitude'].tolist()
n = len(cities)

def euclidean_dist(lat1, lon1, lat2, lon2):
    return math.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)

def compute_inertia(clusters, lat, lon):
    inertia = 0.0
    for cluster in clusters.values():
        if len(cluster) == 0:
            continue
        centroid_lat = sum(lat[idx] for idx in cluster) / len(cluster)
        centroid_lon = sum(lon[idx] for idx in cluster) / len(cluster)
        for idx in cluster:
            inertia += euclidean_dist(lat[idx], lon[idx], centroid_lat, centroid_lon) ** 2
    return inertia

def kmeans_clustering(K, lat, lon, n, n_init=10, max_iterations=100, random_state=None, tolerance=0.0001):
    if random_state is not None:
        random.seed(random_state)
    
    best_inertia = float('inf')
    best_clusters = None
    best_centers_lat = None
    best_centers_lon = None
    
    for init in range(n_init):
        initial_centers_indices = random.sample(range(n), K)
        centers_lat = [lat[i] for i in initial_centers_indices]
        centers_lon = [lon[i] for i in initial_centers_indices]
        
        clusters = {i: [] for i in range(K)}
        
        for iteration in range(max_iterations):
            new_clusters = {i: [] for i in range(K)}
            
            for city_idx in range(n):
                min_dist = float('inf')
                assigned_cluster = -1
                
                for k in range(K):
                    dist = euclidean_dist(lat[city_idx], lon[city_idx], centers_lat[k], centers_lon[k])
                    if dist < min_dist:
                        min_dist = dist
                        assigned_cluster = k
                
                new_clusters[assigned_cluster].append(city_idx)
            
            new_centers_lat = []
            new_centers_lon = []
            
            for k in range(K):
                if len(new_clusters[k]) == 0:
                    new_centers_lat.append(centers_lat[k])
                    new_centers_lon.append(centers_lon[k])
                else:
                    cluster_indices = new_clusters[k]
                    avg_lat = sum(lat[i] for i in cluster_indices) / len(cluster_indices)
                    avg_lon = sum(lon[i] for i in cluster_indices) / len(cluster_indices)
                    new_centers_lat.append(avg_lat)
                    new_centers_lon.append(avg_lon)
            
            moved = False
            for k in range(K):
                old_lat, old_lon = centers_lat[k], centers_lon[k]
                new_lat, new_lon = new_centers_lat[k], new_centers_lon[k]
                if euclidean_dist(old_lat, old_lon, new_lat, new_lon) > tolerance:
                    moved = True
                    
            centers_lat = new_centers_lat
            centers_lon = new_centers_lon
            clusters = new_clusters
            
            if not moved:
                break
        
        inertia = compute_inertia(clusters, lat, lon)
        
        if inertia < best_inertia:
            best_inertia = inertia
            best_clusters = clusters.copy()
            best_centers_lat = centers_lat.copy()
            best_centers_lon = centers_lon.copy()
    
    return best_clusters, best_centers_lat, best_centers_lon

max_k = min(10, n)
k_values = list(range(1, max_k + 1))
inertias = []

print("Расчет инерции для различных K значений...")
for k in k_values:
    best_inertia = float('inf')
    for _ in range(10):
        clusters, _, _ = kmeans_clustering(k, lat, lon, n)
        inertia = compute_inertia(clusters, lat, lon)
        if inertia < best_inertia:
            best_inertia = inertia
    inertias.append(best_inertia)
    print(f"K={k}: Инерция = {best_inertia:.4f}")

choice = input("Выберите способ определения K (1 - автоматический метод локтя, 2 - вручную): ")

if choice == '1':
    kneedle = KneeLocator(k_values, inertias, curve='convex', direction='decreasing')
    K = kneedle.elbow if kneedle.elbow is not None else 2
    print(f"Автоматически выбрано K = {K}")
else:
    K = int(input("Введите количество кластеров: "))

start_time = time.perf_counter()
tracemalloc.start()

clusters, centers_lat, centers_lon = kmeans_clustering(K, lat, lon, n, n_init=10, random_state=42)


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

clusters, centers_lat, centers_lon = kmeans_clustering(K, lat, lon, n, n_init=10, random_state=42)


print(f"\nРаспределение городов на {K} кластеров:\n")
for k in range(K):
    cluster_indices = clusters[k]
    cluster_names = [cities[idx] for idx in cluster_indices]
    print(f"Кластер {k + 1}:")
    print(f"  Города: {', '.join(cluster_names)}")
    print(f"  Количество: {len(cluster_names)}")

colors = plt.cm.tab10(np.linspace(0, 1, K))

plt.figure(figsize=(8, 6))
for cluster_idx in range(K):
    cluster_indices = clusters[cluster_idx]
    cluster_lats = [lat[idx] for idx in cluster_indices]
    cluster_lons = [lon[idx] for idx in cluster_indices]
    plt.scatter(cluster_lons, cluster_lats, c=[colors[cluster_idx]], s=100, label=f'Кластер {cluster_idx + 1}')
    
    #for idx in cluster_indices:
        #plt.annotate(cities[idx], (lon[idx], lat[idx]), fontsize=9, ha='center', va='bottom')

plt.scatter(centers_lon, centers_lat, c='black', marker='X', s=200, label='Центроиды')

print(f"Время выполнения алгоритма кластеризации: {end_time - start_time:.6f}s")
print(f"Пиковое количество памяти: {peak_mem / 1024:.4f} KB")

labels = [0] * n
for cluster_id, cluster_indices in enumerate(clusters.values()):
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
