import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

df = pd.read_csv('insurance.csv')

category=['age','bmi','charges']
x=df[category]

scale = StandardScaler()
x_scale = scale.fit_transform(x)

plot.figure(figsize=(10,7))
plot.title('Customer Hierarchical Dendrogram')
linkage_matrix = linkage(X_scaled, method='ward') 
dendrogram(linkage_matrix, truncate_mode='lastp', p=12) 
plot.xlabel("Cluster size (or index)") 
plot.ylabel("Distance") 
plot.show()

WCSS = [] 
for i in range(1, 11): 
  k_means = KMeans(n_clusters=i, init='k-means++', random_state=36) 
  k_means.fit(X_scaled) 
  WCSS.append(kmeans.inertia_) 
  
plot.figure(figsize=(8, 5)) 
plot.plot(range(1, 11), wcss, marker='o', linestyle='--') 
plot.title('Elbow Method for Optimal K') 
plot.xlabel('Number of Clusters')
plot.ylabel('within-cluster sum of squares error')
plot.show()

k_means= KMeans(n_clusters=4, init='k-means++', random_state=36)
df['Cluster'] = k_means.fit_predict(x_scale)

print("Average attributes per cluster:")
print(df.groupby('Cluster')[category].mean())

plot.figure(figsize=(8,6))
sns.scatterplot(data.df, x='bmi',y='charges', hue='Cluster', palette='viridis')
plot.title('Cluster Segments:BMI vs Charges')
plot.show()
