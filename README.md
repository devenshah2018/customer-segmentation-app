# Customer Segmentation App
This application leverages the power of data to categorize customers into distinct groups based on their annual income and spending habits. Built with Python and Streamlit, this app enables businesses to visualize customer segments, optimize marketing strategies, and tailor products and services to meet the unique needs of each segment. With user-friendly controls and dynamic data visualization, it empowers companies to enhance customer engagement and drive targeted business growth. By clustering customers based on their income and spending habits, businesses can craft highly targeted marketing messages that resonate with specific segments. This precision in targeting leads to increased engagement, higher conversion rates, and more efficient use of marketing resources, significantly boosting the overall effectiveness of marketing efforts.

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)

## Features

- Allows users to upload their own customer datasets in CSV format and automatically processes the data for analysis.
- Provides interactive graphs and charts to visualize the customer segments based on their annual income and spending score.
- Enables users to customize the clustering process by adjusting parameters such as the number of clusters and experimenting with different clustering algorithms.

## Tech Stack

- Language: [Python 3.10](https://www.python.org/)
- Framework: [Streamlit](https://streamlit.io/)
- Machine Learning Model(s): [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans)

## Getting Started

Install dependencies using:
```pip install -r requirements.txt```

Run application locally using:
```streamlit run main.py```
