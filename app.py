import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Melbourne Housing Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Load model and preprocessing components
@st.cache_resource
def load_model():
    try:
        model = joblib.load('linear_regression_model.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return model, scaler, feature_names
    except FileNotFoundError:
        st.error("Model files not found. Please ensure all model files are in the same directory as this app.")
        return None, None, None

# Main app
def main():
    st.title("🏠 Melbourne Housing Price Predictor")
    st.markdown("### Predict property prices in Box Hill, Burwood, and Richmond")
    
    # Load model
    model, scaler, feature_names = load_model()
    
    if model is None:
        st.stop()
    
    # Create two columns for input and results
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Property Details")
        
        # Basic property features
        st.subheader("Basic Features")
        bedrooms = st.slider("Number of Bedrooms", min_value=1, max_value=6, value=3)
        bathrooms = st.slider("Number of Bathrooms", min_value=1, max_value=5, value=2)
        garage_spaces = st.slider("Garage Spaces", min_value=0, max_value=4, value=1)
        
        # Building type
        st.subheader("Property Type")
        building_type = st.selectbox(
            "Building Type",
            ["House", "Townhouse", "Unit", "Apartment", "Flat"],
            index=0
        )
        
        # Proximity features
        st.subheader("Nearby Amenities")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("**Childcare Facilities**")
            childcare_count = st.number_input("Number within 1km", min_value=0, max_value=10, value=2, key="childcare_count")
            childcare_distance = st.slider("Distance to closest (km)", min_value=0.1, max_value=1.0, value=0.5, step=0.1, key="childcare_dist")
        
        with col_b:
            st.write("**Primary Schools**")
            primary_count = st.number_input("Number within 1.5km", min_value=0, max_value=10, value=3, key="primary_count")
            primary_distance = st.slider("Distance to closest (km)", min_value=0.1, max_value=1.5, value=0.8, step=0.1, key="primary_dist")
        
        st.write("**Secondary Schools**")
        secondary_count = st.number_input("Number within 2km", min_value=0, max_value=10, value=2, key="secondary_count")
        secondary_distance = st.slider("Distance to closest (km)", min_value=0.1, max_value=2.0, value=1.0, step=0.1, key="secondary_dist")
    
    with col2:
        st.header("Price Prediction")
        
        if st.button("🔮 Predict Price", type="primary"):
            # Prepare input data
            input_data = prepare_input_data(
                bedrooms, bathrooms, garage_spaces, building_type,
                childcare_count, childcare_distance,
                primary_count, primary_distance,
                secondary_count, secondary_distance,
                feature_names
            )
            
            # Prepare input for scaling and prediction
            input_df = pd.DataFrame([input_data], columns=feature_names)
            
            # Scale only the numerical features (same as training)
            numerical_cols = ['Bedrooms', 'Bathrooms', 'Garage Spaces', 
                             'nearby_childcare_count', 'nearby_childcare_closest_distance',
                             'nearby_primary_count', 'nearby_primary_closest_distance', 
                             'nearby_secondary_count', 'nearby_secondary_closest_distance']
            
            # Create scaled copy
            input_scaled = input_df.copy()
            input_scaled[numerical_cols] = scaler.transform(input_df[numerical_cols])
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            
            # Display result
            st.success(f"## ${prediction:,.0f}")
            st.write(f"**Estimated Price Range:**")
            st.write(f"${prediction-173000:,.0f} - ${prediction+173000:,.0f}")
            st.caption("Range based on model's average error of $173K")
            
            # Model confidence info
            with st.expander("📊 Model Information"):
                st.write("**Model Performance:**")
                st.write("- Average Error: $173,000")
                st.write("- R² Score: 78.2%")
                st.write("- Model: Linear Regression")
                st.write("- Data: 150 Melbourne properties")
        
        # Feature importance info
        with st.expander("📈 What Affects Price Most?"):
            st.write("**Top Price Drivers:**")
            st.write("1. 🏠 **Building Type** - Houses are most valuable")
            st.write("2. 🛏️ **Bedrooms** - More rooms = higher price")
            st.write("3. 🚗 **Garage Spaces** - Parking adds value")
            st.write("4. 🚿 **Bathrooms** - Additional convenience")
            st.write("5. 🏫 **School Proximity** - Education access matters")

def prepare_input_data(bedrooms, bathrooms, garage_spaces, building_type,
                      childcare_count, childcare_distance,
                      primary_count, primary_distance,
                      secondary_count, secondary_distance,
                      feature_names):
    """Prepare input data for prediction"""
    
    # Initialize with zeros
    input_data = [0] * len(feature_names)
    
    # Set basic features
    input_data[feature_names.index('Bedrooms')] = bedrooms
    input_data[feature_names.index('Bathrooms')] = bathrooms
    input_data[feature_names.index('Garage Spaces')] = garage_spaces
    
    # Set building type (one-hot encoded)
    building_type_col = f'BuildingType_{building_type}'
    if building_type_col in feature_names:
        input_data[feature_names.index(building_type_col)] = 1
    
    # Set proximity features
    input_data[feature_names.index('nearby_childcare_count')] = childcare_count
    input_data[feature_names.index('nearby_childcare_closest_distance')] = childcare_distance
    input_data[feature_names.index('nearby_primary_count')] = primary_count
    input_data[feature_names.index('nearby_primary_closest_distance')] = primary_distance
    input_data[feature_names.index('nearby_secondary_count')] = secondary_count
    input_data[feature_names.index('nearby_secondary_closest_distance')] = secondary_distance
    
    return input_data

if __name__ == "__main__":
    main()
