import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.title("Customer Segmentation Dashboard")

df = pd.read_csv("customer_data.csv")

X = df[["Annual Income", "Spending Score"]]

kmeans = KMeans(n_clusters=3, random_state=0)
df["Cluster"] = kmeans.fit_predict(X)

st.write(df)

fig, ax = plt.subplots()

ax.scatter(
    df["Annual Income"],
    df["Spending Score"],
    c=df["Cluster"]
)

ax.set_xlabel("Annual Income")
ax.set_ylabel("Spending Score")
ax.set_title("Customer Segmentation")

st.pyplot(fig)