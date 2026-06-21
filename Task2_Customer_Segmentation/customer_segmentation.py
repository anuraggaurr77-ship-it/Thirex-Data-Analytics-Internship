import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

output_path = r"D:\Internship Projects Thirex\Task2_Customer_Segmentation"

df = pd.read_csv("Mall_Customers.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

df['Cluster'] = kmeans.fit_predict(X)

print("\nCluster Distribution:")
print(df['Cluster'].value_counts())

df.to_csv(
    f"{output_path}\\Customer_Segments.csv",
    index=False
)

plt.figure(figsize=(10, 6))

plt.scatter(
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=df['Cluster']
)

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segmentation using K-Means')

plt.savefig(
    f"{output_path}\\customer_segmentation.png"
)

plt.show()

print("\nFiles Generated Successfully:")
print(f"1. {output_path}\\Customer_Segments.csv")
print(f"2. {output_path}\\customer_segmentation.png")