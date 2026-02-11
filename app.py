import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="Bangalore House Price Prediction", layout="centered")

st.title("🏠 Bangalore House Price Prediction")
st.write("End-to-End Machine Learning Project using Random Forest")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    return df

df = load_data()

# -------------------------------
# DATA CLEANING
# -------------------------------
df = df.dropna()
df = df[df['total_sqft'].str.isnumeric()]
df['total_sqft'] = df['total_sqft'].astype(float)

df['bhk'] = df['size'].apply(lambda x: int(x.split()[0]))
df = df[df['bath'] <= df['bhk'] + 2]

df['price_per_sqft'] = (df['price'] * 100000) / df['total_sqft']

# Remove extreme outliers
df = df[(df['price_per_sqft'] > 1000) & (df['price_per_sqft'] < 50000)]

# Handle rare locations
location_stats = df['location'].value_counts()
rare_locations = location_stats[location_stats <= 10].index
df['location'] = df['location'].apply(lambda x: 'other' if x in rare_locations else x)

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
X = df[['location', 'total_sqft', 'bath', 'bhk']]
y = df['price']

X = pd.get_dummies(X)
columns = X.columns.tolist()

# -------------------------------
# TRAIN TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# TRAIN MODEL
# -------------------------------
@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train)

# -------------------------------
# MODEL EVALUATION
# -------------------------------
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.success(f"Test R² Score: {round(r2 * 100, 2)}%")
st.info(f"RMSE: {round(rmse, 2)}")

# -------------------------------
# FEATURE IMPORTANCE
# -------------------------------
importances = model.feature_importances_

feature_df = pd.DataFrame({
    'Feature': columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

st.subheader("🔍 Top 10 Feature Importance")
st.dataframe(feature_df.head(10))

st.bar_chart(feature_df.set_index("Feature").head(10))

# Save model & columns
pickle.dump(model, open("model.pkl", "wb"))
json.dump(columns, open("columns.json", "w"))

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("About Project")
st.sidebar.write("""
This project predicts Bangalore house prices 
using Random Forest Regression.

Features:
- Location
- Total Square Feet
- Bathrooms
- BHK
""")

# -------------------------------
# USER INPUT
# -------------------------------
st.subheader("Enter House Details")

location = st.selectbox("Location", sorted(df['location'].unique()))
sqft = st.number_input("Total Square Feet", min_value=300.0)
bath = st.number_input("Bathrooms", min_value=1)
bhk = st.number_input("BHK", min_value=1)

# -------------------------------
# PREDICTION FUNCTION
# -------------------------------
def predict_price(location, sqft, bath, bhk):
    columns = json.load(open("columns.json"))
    model = pickle.load(open("model.pkl", "rb"))

    x = np.zeros(len(columns))

    if 'total_sqft' in columns:
        x[columns.index('total_sqft')] = sqft

    if 'bath' in columns:
        x[columns.index('bath')] = bath

    if 'bhk' in columns:
        x[columns.index('bhk')] = bhk

    loc_column = f'location_{location}'
    if loc_column in columns:
        x[columns.index(loc_column)] = 1

    return model.predict([x])[0]

# -------------------------------
# PREDICT BUTTON
# -------------------------------
if st.button("Predict Price"):
    price = predict_price(location, sqft, bath, bhk)
    st.success(f"Estimated Price: ₹ {round(price, 2)} Lakhs")
