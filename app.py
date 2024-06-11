import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import streamlit.components.v1 as components



st.set_page_config(page_title="Customer Segmentation", page_icon="📊", layout="wide")

# Get the current page parameter
query_params = st.experimental_get_query_params()
page = query_params.get("page", ["landing"])[0]

def checkDF(df):
    required_columns = ['Annual Income', 'Spending Score']
    if not all(column in df.columns for column in required_columns):
        st.error(f"The CSV file must contain the following columns: {', '.join(required_columns)}")
        st.stop()
    return

# Define a function to train the K-Means model
def train_kmeans(n_clusters):
    features = df[['Annual Income', 'Spending Score']]
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(features)
    df['Cluster'] = model.labels_
    return model

if page == 'landing':
    # landingHTML = open("landing.html", 'r', encoding='utf-8')
    # source_code = landingHTML.read()
    # components.html(source_code)
    with open("landing.html", "r") as file:
        landing_page_content = file.read()
    st.markdown(landing_page_content, unsafe_allow_html=True)
elif page == "main":
    inital_file_container = st.empty()
    # Initialize the dataframe
    uploaded_customer_data = inital_file_container.file_uploader("Select Customer Data", type=["csv"], key="initial-file")
    if uploaded_customer_data is not None:
        inital_file_container.empty()
        df = pd.read_csv(uploaded_customer_data)
        checkDF(df)
        st.title("Customer Segmentation App")

        # Sidebar
        st.sidebar.title("Settings")
        n_clusters = st.sidebar.slider("Number of clusters", min_value=2, max_value=10, value=3, step=1)
        st.sidebar.title("File Information")
        st.sidebar.write(f"**Filename:** {uploaded_customer_data.name}")
        st.sidebar.write(f"**File Size:** {uploaded_customer_data.size / 1024:.2f} KB")
        st.sidebar.title("Upload New Customer Data")
        new_uploaded_customer_data = st.sidebar.file_uploader("Select Customer Data", type=["csv"], key="new-file")
        if new_uploaded_customer_data is not None:
            df = pd.read_csv(new_uploaded_customer_data)
            uploaded_customer_data = new_uploaded_customer_data

        # Train the model
        model = train_kmeans(n_clusters)

        # Display the cluster chart
        st.write(f"The following chart visualizes the distribution of customer spending-income data into {n_clusters} clusters:")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='Annual Income', y='Spending Score', hue='Cluster', palette='tab10', ax=ax)
        ax.set_title("Customer Segments")
        st.pyplot(fig)

        # Show cluster centroids
        st.write("Cluster Centers:")
        st.write(pd.DataFrame(model.cluster_centers_, columns=['Annual Income', 'Spending Score']))

        # TODO: Ability to add a customer data and find the cluster
    else:
        st.markdown("""
        **Please Note:**
        - The file must be in **CSV** format.
        - The CSV file should have the following columns:
        - **Annual Income**
        - **Spending Score**
        
        Ensure that your file follows these requirements for proper processing.
        """)
else:
    st.error("Page not found. Please go to the landing page.")


