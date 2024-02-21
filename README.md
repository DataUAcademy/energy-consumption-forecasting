# Average Energy Consumption Forecasting

**Brief Overview**

This project aims to accurately forecast energy consumption patterns using a Seasonal Autoregressive Integrated Moving Average (SARIMAX) model and provide interactive visualizations with Streamlit. This empowers data-driven insights for stakeholders in the energy sector.

**Data Collection Limitations and Impact**

The original energy consumption dataset exhibited inconsistencies in data collection across individual households. To create a workable model, the following approach was taken:

*   **Averaging:** Total energy consumption for each day was divided by the number of households with data recordings for that day. This provides an aggregate view of energy usage patterns within the community.

**Importance of Transparency**

*   While this approach enables forecasting, it's important to acknowledge that it obscures individual household consumption behavior.
*   Obtaining consistent data at the individual household level would significantly improve model accuracy and granularity of insights.

**Future Directions**

*   A key area for future improvement is securing a  dataset with reliable, comprehensive recording across all households.
*   Exploring methods to impute or simulate realistic missing data patterns could also be a valuable research direction. 



**Technologies Used**

* Python
* SARIMAX (Statsmodels)
* Pandas 
* NumPy
* Streamlit 

**Key Features**

*   Robust time series forecasting model tailored to capture energy demand seasonality and trends.
*   Interactive Streamlit dashboard for user-friendly data exploration and forecast visualization.
*   Clear visual presentation of actual vs. predicted energy consumption.

**How to Run**

1.  **Clone Repository:**
    ```bash
    git clone https://github.com/SombrathnaSout/energy-consumption-forecasting
    ```

2.  **Environment Setup:**
    ```bash
    pip install -r requirements.txt 
    ```

3.  **Launch App:**
    ```bash
    streamlit run app.py 
    ```

**Project Structure**

*   `data/`: Contains raw energy consumption datasets (.csv, .h5, or other format)
*   `models/`: Houses the trained SARIMAX model  (e.g., a .pkl or .joblib file)
*   `Code/`:  Script for data cleaning, feature engineering, and train/test splits.
*   `app.py`: Core Streamlit application code. 

**Results**

![Demo](https://github.com/SombrathnaSout/energy-consumption-forecasting/assets/138176913/788b53bb-b30f-4fb5-a409-673edc0ea564)

*   The SARIMAX model achieved an RMSE of 0.66 on the held-out test set, demonstrating its predictive power.
*   The Streamlit dashboard enables  exploration of how factors like weather conditions and holidays impact energy demand. 

**Contact & Connect**

For further inquiries or to discuss potential collaborations, please feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/sombrathna-sout/). 
