import pandas as pd
import matplotlib.pyplot as PLT
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

#1. Load Data
df = pd.read_csv('insurance.csv')

#2 Data Preprocessing: Select 'Age', 'BMI' and 'Charges' to divide customers
category=['age','bmi','charges']
x=df[category]
scale = StandardScaler()
x_scale = scale.fit_transform(x)

#3 Use agglomerative hierarchical clustering to minimize variance within clusters
PLT.figure(figsize=(10,7))
PLT.title('Customer Hierarchical Dendrogram')
#for large datasets, we truncate the tree to make it readable
linkage_matrix = linkage(x_scale, method='ward')
dendrogram(linkage_matrix, truncate_mode='lastp', p=12)
PLT.xlabel("Cluster size (or index)")
PLT.ylabel("Distance")
PLT.show()

#4. K-Means Method
WCSS = []
for i in range(1, 11):
  k_means = KMeans(n_clusters=i, init='k-means++', random_state=36)
  k_means.fit(x_scale)
  WCSS.append(k_means.inertia_)

PLT.figure(figsize=(8, 5))
PLT.plot(range(1, 11), WCSS, marker='o', linestyle='--')
PLT.title('Elbow Method for Optimal K')
PLT.xlabel('Number of Clusters')
PLT.ylabel('within-cluster sum of squares error')
PLT.show()

#5 Final K-Means Execution (Choose K=4)
k_means= KMeans(n_clusters=4, init='k-means++', random_state=36)
df['Cluster'] = k_means.fit_predict(x_scale)

#6 Result Interpretation
print("Average attributes per cluster:")
print(df.groupby('Cluster')[category].mean())

#Visualize the final clusters
PLT.figure(figsize=(8,6))
sns.scatterplot(data=df, x='bmi',y='charges', hue='Cluster', palette='viridis')
PLT.title('Cluster Segments:BMI vs Charges')
PLT.show()

