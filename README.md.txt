# Drought Prediction Using Remote Sensing and Machine Learning

## Overview
This project focuses on predicting drought conditions by leveraging remote sensing datasets and machine learning techniques. By integrating spatiotemporal data from sources like MODIS, Sentinel-1, and Sentinel-2, the project aims to provide accurate and actionable insights for drought risk management and hydrological resource allocation.

## Objectives
- Develop an advanced spatiotemporal predictive model for drought forecasting.
- Incorporate remote sensing data and machine learning to analyze evolving drought dynamics.
- Create geospatial visualizations and interactive maps to support decision-making.

## Key Features
- **Remote Sensing Integration**: Utilized MODIS, Sentinel-1, and Sentinel-2 datasets to derive features like NDVI and precipitation.
- **Machine Learning Models**: Implemented Random Forest, LSTM, and GridSearchCV for hyperparameter tuning to achieve optimal model performance.
- **Spatiotemporal Analysis**: Captured dynamic changes in drought conditions using time-series analysis and geospatial mapping.
- **Geospatial Visualizations**: Created scatter plots and interactive maps using GeoPandas, Matplotlib, and Folium.

## Dataset
The dataset includes:
- Remote sensing-derived features: NDVI, Precipitation, etc.
- Spatial information: Latitude and Longitude
- Temporal data: Time series measurements

## Methodology
1. **Data Preprocessing**:
   - Cleaned the dataset by handling missing values in NDVI and precipitation.
   - Defined drought conditions based on NDVI and precipitation thresholds.
   - Normalized the features for machine learning.

2. **Machine Learning Models**:
   - **Random Forest**:
     - Tuned using GridSearchCV to achieve optimal hyperparameters.
     - Accuracy: **78%**
     - Precision: **80%**
     - Recall: **78%**
     - F1-Score: **79%**
   - **LSTM (Long Short-Term Memory)**:
     - Architecture with 100 neurons and ReLU activation.
     - Accuracy: **76%**
     - Precision: **78%**
     - Recall: **75%**
     - F1-Score: **76%**

3. **Feature Importance**:
   - Analyzed feature contributions using the Random Forest model.
   - Identified the most critical factors influencing drought conditions.

4. **Spatial Analysis**:
   - Used GeoPandas to map drought and non-drought regions.
   - Created scatter plots for precipitation levels and spatial drought distribution.
   - Developed interactive drought maps using Folium.

## Visualizations
- Scatter plots illustrating precipitation levels across different regions.
- Geospatial maps showing drought and non-drought areas.
- Interactive HTML maps for dynamic exploration of drought conditions.

## Tools and Technologies
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, Keras
- **Geospatial Analysis**: GeoPandas, Matplotlib, Folium
- **Remote Sensing**: MODIS, Sentinel-1, Sentinel-2 datasets
- **Visualization**: Seaborn, Matplotlib, Folium

## Results
- Achieved **78% accuracy** in drought prediction using the Random Forest model.
- LSTM model performance:
  - Accuracy: **76%**
  - Precision: **78%**
  - Recall: **75%**
  - F1-Score: **76%**
- Generated geospatial drought maps to support actionable insights.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Vishhh25/Geospatial-Remote-Sensing-and-Machine-Learning-for-Drought-Prediction.git
   cd Geospatial-Remote-Sensing-and-Machine-Learning-for-Drought-Prediction
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook Drought_Prediction_Remotesensing_Machinelearning.ipynb
   ```

4. Generate the interactive map:
   - Ensure `folium` is installed.
   - Run the notebook cells to save the HTML map.

## Conclusion
This project demonstrates the integration of remote sensing data with machine learning models to predict and analyze drought conditions. By leveraging geospatial analytics and advanced modeling, it provides a comprehensive framework for addressing drought-related challenges effectively.

## Acknowledgments
- Remote sensing data sources: MODIS, Sentinel-1, Sentinel-2
- Libraries: GeoPandas, Folium, Scikit-learn, Keras

## Contact
For further information, please contact [ravalvishwa2501@gmail.com].