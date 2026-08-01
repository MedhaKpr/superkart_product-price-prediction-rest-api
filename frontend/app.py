import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend
BACKEND_URL = "http://backend:7860"

st.set_page_config(page_title="SuperKart Sales Predictor", layout="wide")
st.title("🛒 SuperKart Product Sales Prediction")

# Sidebar for navigation
option = st.sidebar.selectbox("Choose Prediction Type", ["Single Product Prediction", "Batch Prediction (CSV)"])

if option == "Single Product Prediction":
    st.subheader("Enter Product and Store Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        product_weight = st.number_input("Product Weight", min_value=0.0, value=12.5)
        product_sugar = st.selectbox("Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
        product_mrp = st.number_input("Product MRP ($)", min_value=0.0, value=150.0)
        product_allocated_area = st.number_input("Allocated Area Ratio", min_value=0.0, max_value=1.0, value=0.05)
        product_type = st.selectbox("Product Type", ["Fruits and Vegetables", "Snack Foods", "Household", "Frozen Foods", "Dairy", "Canned", "Baking Goods", "Health and Hygiene", "Soft Drinks", "Meat", "Breads", "Hard Drinks", "Others", "Starchy Foods", "Breakfast", "Seafood"])

    with col2:
        store_establishment_year = st.number_input("Store Establishment Year", min_value=1980, max_value=2024, value=2000)
        store_size = st.selectbox("Store Size", ["Medium", "Small", "High"])
        store_location = st.selectbox("Store Location (City Type)", ["Tier 1", "Tier 2", "Tier 3"])
        store_type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])

    # The backend handles feature engineering, but we send the raw inputs as expected by our app.py logic
    payload = {
        "Product_Weight": product_weight,
        "Product_Sugar_Content": product_sugar,
        "Product_Allocated_Area": product_allocated_area,
        "Product_Type": product_type,
        "Product_MRP": product_mrp,
        "Store_Establishment_Year": store_establishment_year,
        "Store_Size": store_size,
        "Store_Location_City_Type": store_location,
        "Store_Type": store_type
    }

    if st.button("Predict Sales", type="primary"):
        with st.spinner('Calculating...'):
            try:
                response = requests.post(f"{BACKEND_URL}/v1/product", json=payload)
                if response.status_code == 200:
                    prediction = response.json().get('Predicted Price (in dollars)', 'N/A')
                    st.metric("Estimated Total Sales", f"${prediction}")
                else:
                    st.error(f"Error: Backend returned status code {response.status_code}")
            except Exception as e:
                st.error(f"Connection failed: {e}")

else:
    st.subheader("Batch Sales Prediction")
    st.write("Upload a CSV file containing the required product and store features.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        if st.button("Run Batch Prediction", type="primary"):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            try:
                response = requests.post(f"{BACKEND_URL}/v1/batchproducts", files=files)
                if response.status_code == 200:
                    st.success("Batch Processing Complete!")
                    results = response.json()
                    # Convert dictionary to DataFrame for display
                    res_df = pd.DataFrame(list(results.items()), columns=['Product_ID', 'Predicted_Sales'])
                    st.dataframe(res_df)
                else:
                    st.error("Batch prediction failed. Check file format.")
            except Exception as e:
                st.error(f"Connection failed: {e}")
