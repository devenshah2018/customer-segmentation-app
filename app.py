import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Initialize the dataframe
df = pd.read_csv('customer_data.csv')

# Define a function to train the K-Means model
def train_kmeans(n_clusters):
    features = df[['Annual Income', 'Spending Score']]
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
sns.scatterplot(data=df, x='Annual Income', y='Spending Score', hue='Cluster', palette='tab10', ax=ax)
ax.set_title("Customer Segments")
st.pyplot(fig)

# Show cluster centroids
st.write("Cluster Centers:")
st.write(pd.DataFrame(model.cluster_centers_, columns=['Annual Income', 'Spending Score']))