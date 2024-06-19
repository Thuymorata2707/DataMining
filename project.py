import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

df = pd.read_csv('Customers.csv')
print(df)

# #Biểu đồ tìm k
x_data1 = df[['Annual_Income_(k$)', 'Spending_Score']].values
lst = []
for k in range(1, 15):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(x_data1)
    lst.append(kmeans.inertia_)
plt.plot(range(1, 15), lst)
plt.xlabel("Number of k (cluster) values")
plt.ylabel("Inertia - WCSS")
plt.title("Finding Optimum Number of K")
plt.show()
# Áp dụng KMeans với số cụm đã chọn
kmeans_x_data1 = KMeans(n_clusters=5, random_state=0)
clusters = kmeans_x_data1.fit_predict(x_data1)
df["Label1"] = clusters

# Vẽ biểu đồ scatter plot với các cụm đã được phân loại
plt.figure(figsize=(15, 8))
plt.scatter(x_data1[clusters == 0, 0], x_data1[clusters == 0, 1], color="green", label="High income-Low Spending")
plt.scatter(x_data1[clusters == 1, 0], x_data1[clusters == 1, 1], color="red", label="Middle income-Middle Spending")
plt.scatter(x_data1[clusters == 2, 0], x_data1[clusters == 2, 1], color="purple", label="High income-High Spending")
plt.scatter(x_data1[clusters == 3, 0], x_data1[clusters == 3, 1], color="cyan", label="Low income-High Spending")
plt.scatter(x_data1[clusters == 4, 0], x_data1[clusters == 4, 1], color="orange", label="Low income-Low Spending")
plt.scatter(kmeans_x_data1.cluster_centers_[:, 0], kmeans_x_data1.cluster_centers_[:, 1], color="black", label="Centroids", s=75)
plt.xlabel("Annual_Income_k$")
plt.ylabel("Spending_Score")
plt.legend()
plt.title("Segmentation According to Income and Spending Score")
plt.show()

# Biểu đồ tính toán phân cụm

x_data2 = df[['Age', 'Spending_Score']].values
wcss2 = []
for k in range(1, 15):
    kmeans2 = KMeans(n_clusters=k, random_state=0)
    kmeans2.fit(x_data2)
    wcss2.append(kmeans2.inertia_)

plt.plot(range(1, 15), wcss2)
plt.xlabel("Number of k (cluster) values")
plt.ylabel("Inertia - WCSS")
plt.title("Finding Optimum Number of K")
plt.show()

# Áp dụng KMeans với số cụm đã chọn
kmeans_x_data2 = KMeans(n_clusters=4, random_state=0)
clusters2 = kmeans_x_data2.fit_predict(x_data2)
df["Label2"] = clusters2

# Vẽ biểu đồ scatter plot với các cụm đã được phân loại
plt.figure(figsize=(15, 8))
plt.scatter(x_data2[clusters2 == 0, 0], x_data2[clusters2 == 0, 1], color="green", label="D group")
plt.scatter(x_data2[clusters2 == 1, 0], x_data2[clusters2 == 1, 1], color="red", label="A group")
plt.scatter(x_data2[clusters2 == 2, 0], x_data2[clusters2 == 2, 1], color="purple", label="B group")
plt.scatter(x_data2[clusters2 == 3, 0], x_data2[clusters2 == 3, 1], color="cyan", label="C group")
plt.scatter(kmeans_x_data2.cluster_centers_[:, 0], kmeans_x_data2.cluster_centers_[:, 1], color="black", label="Centroids", s=75)
plt.xlabel("Age")
plt.ylabel("Spending_Score")
plt.legend()
plt.title("Segmentation According to Age and Spending Score")
plt.show()

# Tạo biểu đồ phân loại giới tính
gender_counts = df['Gender'].value_counts()
gender_counts.plot(kind='bar', color=['pink', 'blue'])
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

#Biểu đồ thu nhập hằng năm
plt.figure(figsize=(10,6))
sns.distplot(df["Annual_Income_(k$)"])
plt.title("Distribution of Annual Income")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Frequency")
plt.show()

#Biểu đồ theo độ tuổi khách hàng
plt.figure(figsize=(10,6))
sns.distplot(df["Age"])
plt.title("Distribution of Age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

#Biểu đồ biểu diễn điểm chi tiêu
plt.figure(figsize=(10,6))
sns.distplot(df["Spending_Score"])
plt.title("Distribution of Spending Score (1-100)")
plt.xlabel("Spending Score (1-100)")
plt.ylabel("Frequency")
plt.show()

sns.pairplot(df.drop('CustomerID',axis=1),hue='Gender',aspect=1.5)
plt.show()