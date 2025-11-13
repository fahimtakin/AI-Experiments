import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


january_df = pd.read_csv("January.csv")
march_df = pd.read_csv("March.csv")


combined_df = pd.concat([january_df, march_df], ignore_index=True)


combined_df.columns = combined_df.columns.str.strip().str.replace('"', '')
for col in combined_df.select_dtypes(include='object').columns:
    combined_df[col] = combined_df[col].astype(str).str.strip().str.replace('"', '')


grouped = combined_df.groupby("RecipientPhone").agg(
    TotalOrders=('ConsignmentId', 'count'),
    SuccessfulDeliveries=('OrderStatus', lambda x: (x == 'Delivered').sum()),
    Returns=('OrderStatus', lambda x: (x.str.lower().str.contains('return')).sum()),
    TotalAmountToCollect=('AmountToCollect', 'sum'),
    TotalCollected=('CollectedAmount', 'sum'),
    AvgFinalFee=('FinalFee', 'mean'),
    UniqueMerchants=('MerchantName', pd.Series.nunique)
)


grouped["ReturnRate"] = grouped["Returns"] / grouped["TotalOrders"]
grouped["SuccessRate"] = grouped["SuccessfulDeliveries"] / grouped["TotalOrders"]
grouped["CollectionGap"] = grouped["TotalAmountToCollect"] - grouped["TotalCollected"]


grouped = grouped.fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(grouped)


kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
grouped["Cluster"] = kmeans.fit_predict(X_scaled)


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
grouped["PC1"] = X_pca[:, 0]
grouped["PC2"] = X_pca[:, 1]


plt.figure(figsize=(10, 6))
sns.scatterplot(data=grouped, x='PC1', y='PC2', hue='Cluster', palette='Set2', s=70)
plt.title('Customer Segmentation (PCA + KMeans)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()


grouped.reset_index().to_csv("segmented_customers.csv", index=False)
