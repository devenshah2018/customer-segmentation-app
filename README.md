# Customer Segmentation App
This application leverages the power of data to categorize customers into distinct groups based on their annual income and spending habits. Built with Python and Streamlit, this app enables businesses to visualize customer segments, optimize marketing strategies, and tailor products and services to meet the unique needs of each segment. With user-friendly controls and dynamic data visualization, it empowers companies to enhance customer engagement and drive targeted business growth. By clustering customers based on their income and spending habits, businesses can craft highly targeted marketing messages that resonate with specific segments. This precision in targeting leads to increased engagement, higher conversion rates, and more efficient use of marketing resources, significantly boosting the overall effectiveness of marketing efforts.
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
py -m streamlit run main.py
