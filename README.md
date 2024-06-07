# Customer Segmentation App

- Upload customer activity training dataset
```
data = {
    'CustomerID': range(1, n_customers + 1),
    'Annual Income (k$)': np.random.normal(50, 20, n_customers).clip(5, 100),
    'Spending Score (1-100)': np.random.randint(1, 101, n_customers)
}
```
- Shows Graph of Clusters (High/Low Income, High/Low Spending Score)
- Provides data analysis

- Streamlit
- KMeans
- Other classification models
