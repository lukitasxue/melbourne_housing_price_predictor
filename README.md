# Melbourne Housing Price Predictor 🏠

A machine learning web application that predicts housing prices in Melbourne suburbs (Box Hill, Burwood, Richmond) using property features and nearby amenities.

## Model Performance
- **Accuracy**: 78.2% R² score
- **Average Error**: $173,000 MAE
- **Algorithm**: Linear Regression (outperformed Random Forest and Gradient Boosting)
- **Dataset**: 150 properties from Box Hill, Burwood, and Richmond

## `Quick Start

### Local Development
```bash
# Clone the repository
git clone https://github.com/lukitasxue/melbourne_housing_price_predictor
cd https://github.com/lukitasxue/melbourne_housing_price_predictor

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Live Demo
Visit the deployed app: [[Your Streamlit Cloud URL](https://melbournehousingpricepredictor.streamlit.app)]

## Features
- **Property Type**: House, Townhouse, Unit, Apartment, Flat
- **Basic Features**: Bedrooms, Bathrooms, Garage Spaces
- **Proximity Amenities**: Nearby childcare, primary schools, secondary schools
- **Real-time Predictions**: Instant price estimates with confidence ranges

## Model Insights
The Linear Regression model identified these key price drivers:
1. **Building Type** - Houses command highest premiums
2. **Bedrooms** - Each additional bedroom adds significant value
3. **Property Size** - Garage spaces and bathrooms matter
4. **Location Amenities** - School and childcare proximity provide moderate boost

## Deployment
This app is deployed using:
- **Framework**: Streamlit
- **Hosting**: Streamlit Cloud
- **CI/CD**: Automatic deployment via GitHub integration

## Technical Details
- **Feature Engineering**: One-hot encoding for categorical variables, standardization for numerical features
- **Model Selection**: Compared Linear Regression, Random Forest, and Gradient Boosting
- **Validation**: 5-fold cross-validation with MAE, RMSE, and R² metrics
- **Hyperparameter Tuning**: Optimized Gradient Boosting (still couldn't beat Linear Regression!)

## Project Structure
```
├── app.py                          # Main Streamlit application
├── requirements.txt                # Dependencies
├── linear_regression_model.pkl     # Trained model
├── feature_scaler.pkl             # Feature preprocessing
├── feature_names.pkl              # Feature column names
├── validation_samples.pkl         # Test samples
└── README.md                      # This file
```

## 🏆 Model Comparison Results
| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| **Linear Regression** | **$172,959** | **$227,470** | **0.7822** |
| Random Forest | $189,571 | $250,922 | 0.7434 |
| Gradient Boosting | $184,988 | $246,146 | 0.7541 |

## Academic Context
This project was developed as part of SIT307 Machine Learning coursework, demonstrating:
- End-to-end ML pipeline development
- Model comparison and selection
- Feature importance analysis
- Web application deployment
- Real estate domain application

## Contributing
Feel free to open issues or submit pull requests for improvements!


