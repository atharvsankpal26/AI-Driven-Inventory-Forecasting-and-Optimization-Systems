import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Streamlit Page Setup ===
st.set_page_config(page_title="AI Inventory Insights", layout="wide")

# === Load Dataset ===
@st.cache_data
def load_data():
    df = pd.read_excel("AI_Inventory_Management_Dataset_2000.xlsx")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.to_period('M')
    return df

df = load_data()

# === Header ===
st.markdown("""
    <style>
        .main-title {
            font-size: 32px;
            font-weight: 600;
            color: #2c3e50;
        }
        .highlight-box {
            background-color: #f2f2f2;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>AI-Driven Inventory Management Dashboard</div>", unsafe_allow_html=True)

st.markdown("""
This dashboard presents the outcomes of an AI-powered inventory management analysis conducted as part of the MIS790 Capstone Project. It highlights sales trends, promotional impacts, forecasting performance, and AI-generated demand scenarios.
""")

st.markdown("---")

# === KPI Metrics Section ===
st.header("Key Performance Indicators")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("<div class='highlight-box'><strong>Total Units Sold</strong><br>{:,} units</div>".format(df['Sales'].sum()), unsafe_allow_html=True)
with col2:
    st.markdown("<div class='highlight-box'><strong>Average Daily Sales</strong><br>{:.1f}</div>".format(df['Sales'].mean()), unsafe_allow_html=True)
with col3:
    st.markdown("<div class='highlight-box'><strong>Number of Products</strong><br>{}</div>".format(df['Product_Name'].nunique()), unsafe_allow_html=True)

st.markdown("---")

# === Monthly Sales Trend Chart ===
st.header("Monthly Sales Trends by Product")
monthly_sales = df.groupby(['Month', 'Product_Name'])['Sales'].sum().reset_index()
monthly_sales['Month'] = monthly_sales['Month'].dt.to_timestamp()
selected_product = st.selectbox("Select a Product", df['Product_Name'].unique())
filtered = monthly_sales[monthly_sales['Product_Name'] == selected_product]

fig1, ax1 = plt.subplots(figsize=(12, 5))
sns.lineplot(data=filtered, x='Month', y='Sales', marker='o', color='#1f77b4', ax=ax1)
ax1.set_title(f"Sales Timeline: {selected_product}", fontsize=14)
ax1.set_xlabel("Month")
ax1.set_ylabel("Units Sold")
st.pyplot(fig1)
st.markdown("*This visualization shows the monthly sales trends for a selected product, helping identify seasonal patterns, peaks, and any inventory-related inconsistencies over time.*")

st.markdown("---")

# === Promotion Impact ===
st.header("Sales Performance During Promotions")
promo = df.copy()
promo['Promotion_Status'] = promo['Promotion'].map({0: "No Promo", 1: "Promo"})
promo_summary = promo.groupby(['Product_Name', 'Promotion_Status'])['Sales'].mean().reset_index()

fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.barplot(data=promo_summary, x='Product_Name', y='Sales', hue='Promotion_Status', palette='Set2', ax=ax2)
ax2.set_title("Average Sales With and Without Promotions")
st.pyplot(fig2)
st.markdown("*This comparison reveals the impact of promotions on product sales, highlighting the effectiveness of promotional strategies in boosting demand.*")

st.markdown("---")

# === Correlation Heatmap ===
st.header("Feature Correlation Matrix")
corr = df[['Sales', 'Stock_Level', 'Supplier_Lead_Time', 'Price_Per_Unit', 'Weather_Index']].corr()
fig3, ax3 = plt.subplots(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax3)
ax3.set_title("Correlation Between Sales and Related Features")
st.pyplot(fig3)
st.markdown("*The correlation matrix helps understand relationships between features and sales. It guides feature selection for forecasting models and exposes potential drivers of inventory performance.*")

st.markdown("---")

# === Forecasting Insights (Static Summary) ===
st.header("Forecasting Model Performance")
st.markdown("""
- **Model Used**: Random Forest Regressor  
- **RMSE**: 24.63  
- **MAPE**: 44.99%  
- **Service Level Achieved**: >95%  
- Variational Autoencoder (VAE) was applied for generating synthetic demand scenarios to test model robustness.
""")

st.markdown("---")

# === Inventory Turnover Ratio by Product ===
st.header("Inventory Turnover Ratio by Product")
inventory_turnover = df.groupby('Product_Name').apply(
    lambda x: x['Sales'].sum() / x['Stock_Level'].mean()
).reset_index(name='Inventory_Turnover')

fig4, ax4 = plt.subplots(figsize=(8, 5))
sns.barplot(data=inventory_turnover, x='Product_Name', y='Inventory_Turnover', palette='Blues', ax=ax4)
ax4.set_title('Inventory Turnover Ratio by Product')
st.pyplot(fig4)
st.markdown("*This visual highlights product-level inventory efficiency, guiding reorder strategy and AI policy tuning.*")

st.markdown("---")

# === VAE Scenario Visualization ===
st.header("VAE-Generated Demand Scenario")
try:
    import numpy as np
    vae_samples = pd.read_csv("vae_synthetic_samples.csv")  # placeholder file
    st.line_chart(vae_samples)
    st.caption("This line chart represents synthetic demand sequences generated using a trained Variational Autoencoder (VAE).")
    st.markdown("*These scenarios simulate real-world variability in demand, enabling stress testing of AI-driven forecasting and inventory strategies.*")
except Exception as e:
    st.warning("Synthetic demand sample not available. Please upload or generate using the VAE model.")





# === Footer ===
st.caption("Antara More & Atharv Sankpal | MIS790 Project")
