import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Melbourne Housing Price Predictor",
    page_icon="🏠"
)

# Simple model loading without caching
def load_model():
    try:
        model = joblib.load('linear_regression_model.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Make sure model files are uploaded to the repository")
        return None, None, None

# Main app
def main():
    st.title("🏠 Melbourne Housing Price Predictor")
    st.markdown("### Predict property prices in Box Hill, Burwood, and Richmond")
    
    # Debug info
    import os
    st.sidebar.write("Debug Info:")
    st.sidebar.write(f"Current dir: {os.getcwd()}")
    st.sidebar.write(f"Files: {os.listdir('.')}")
    
    # Test if we can load models
    model, scaler, feature_names = load_model()
    
    if model is None:
        st.error("❌ Cannot load model files")
        st.stop()
    
    st.success("✅ Model loaded successfully!")
    st.info(f"Features: {len(feature_names) if feature_names is not None else 0}")
    
    # Simple form inputs
    st.header("Property Details")
    
    # Basic features
    bedrooms = st.slider("Number of Bedrooms", 1, 6, 3)
    bathrooms = st.slider("Number of Bathrooms", 1, 5, 2)
    garage_spaces = st.slider("Garage Spaces", 0, 4, 1)
    
    # Building type
    building_type = st.selectbox("Building Type", ["House", "Townhouse", "Unit", "Apartment", "Flat"])
    
    # Simplified proximity features
    st.subheader("Nearby Amenities (simplified)")
    childcare_count = st.number_input("Childcare within 1km", 0, 10, 2)
    primary_count = st.number_input("Primary schools within 1.5km", 0, 10, 3)
    secondary_count = st.number_input("Secondary schools within 2km", 0, 10, 2)
    
    # Fixed distances for simplicity
    childcare_distance = 0.5
    primary_distance = 0.8
    secondary_distance = 1.0
    
    if st.button("🔮 Predict Price", type="primary"):
        try:
            # Prepare input data
            input_data = prepare_input_data(
                bedrooms, bathrooms, garage_spaces, building_type,
                childcare_count, childcare_distance,
                primary_count, primary_distance,
                secondary_count, secondary_distance,
                feature_names
            )
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data], columns=feature_names)
            
            # Scale numerical features only
            numerical_cols = ['Bedrooms', 'Bathrooms', 'Garage Spaces', 
                             'nearby_childcare_count', 'nearby_childcare_closest_distance',
                             'nearby_primary_count', 'nearby_primary_closest_distance', 
                             'nearby_secondary_count', 'nearby_secondary_closest_distance']
            
            input_scaled = input_df.copy()
            input_scaled[numerical_cols] = scaler.transform(input_df[numerical_cols])
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            
            # Display result
            st.success(f"## Estimated Price: ${prediction:,.0f}")
            st.info(f"Range: ${prediction-173000:,.0f} - ${prediction+173000:,.0f}")
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
    
    # Simple info section
    with st.expander("ℹ️ About this model"):
        st.write("- Model: Linear Regression")
        st.write("- Average Error: $173,000") 
        st.write("- Accuracy: 78% R²")
        st.write("- Data: 150 Melbourne properties")

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
