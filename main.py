import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Create a synthetic dataset
# TODO: Create a flow to accept a CSV file with the following dataset
np.random.seed(42)
n_customers = 100
data = {
    'CustomerID': range(1, n_customers + 1),
    'Annual Income (k$)': np.random.normal(50, 20, n_customers).clip(5, 100),
    'Spending Score (1-100)': np.random.randint(1, 101, n_customers)
}

df = pd.DataFrame(data)

# Define a function to train the K-Means model
def train_kmeans(n_clusters):
    features = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(features)
    df['Cluster'] = model.labels_
    return model

# Streamlit app configuration
st.title("Customer Segmentation App")

# Sidebar for user input
st.sidebar.title("Settings")
n_clusters = st.sidebar.slider("Number of clusters", min_value=2, max_value=10, value=3, step=1)

# Train the model with the user-defined number of clusters
model = train_kmeans(n_clusters)

# Display the clustering result
st.write(f"Clustering result with {n_clusters} clusters")

# Plot the clusters
fig, ax = plt.subplots()
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='tab10', ax=ax)
ax.set_title("Customer Segments")
st.pyplot(fig)

# Show cluster centroids
st.write("Cluster Centers:")
st.write(pd.DataFrame(model.cluster_centers_, columns=['Annual Income (k$)', 'Spending Score (1-100)']))
