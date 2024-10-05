import seaborn as sns
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data_sample = read_csv("Data Cluster Market.xlsx - Online Retail.csv")
data_sample = data_sample[:5000]


# Step 1: Dropping rows with missing values
data_cleaned = data_sample.dropna(subset=['CustomerID', 'Quantity', 'UnitPrice'])

# Step 2: Selecting relevant features for clustering
X = data_cleaned[['Quantity', 'UnitPrice']]

# Step 3: Standardizing the data to bring all features to the same scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Applying KMeans algorithm for clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Step 5: Adding the cluster labels to the dataset
data_cleaned['Cluster'] = clusters

# Step 6: Visualizing the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=clusters, palette='viridis', s=10)
plt.title('Customer Clustering based on Quantity and Unit Price')
plt.xlabel('Quantity (Standardized)')
plt.ylabel('Unit Price (Standardized)')
plt.show()

# Displaying the first few rows with cluster labels
data_cleaned.head()