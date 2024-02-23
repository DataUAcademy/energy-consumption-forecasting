# Average Energy Consumption Forecasting

**Brief Overview**

This project provides an interactive platform for forecasting average daily energy consumption and analyzing consumer behavior insights. By integrating a Seasonal Autoregressive Integrated Moving Average (SARIMAX) model with a Streamlit dashboard, stakeholders can visualize energy consumption patterns and predict future demand.

**Data Collection Limitations and Impact**

The original dataset had inconsistencies in data collection across households. To mitigate this and create a viable model, we averaged the total energy consumption for each day by the number of contributing households, offering an aggregate perspective of community energy patterns.

**Insights and Visualizations**

The Streamlit dashboard delivers a suite of insightful visualizations:

* Comparative analysis of energy consumption on holidays vs. non-holidays.
* Distribution of average daily energy consumption through interactive histograms.
* Correlation between energy usage and environmental factors like temperature, humidity, and wind speed.
* Clustering of households to identify different energy usage behaviors.
* Forecast vs. actual energy consumption with RMSE evaluation metrics.
* Cost predictions based on forecasted energy consumption.

**Technologies Used**

* Python
* SARIMAX (Statsmodels)
* Pandas 
* NumPy
* Streamlit 
* Plotly

**How to Run**

1. **Clone Repository:**
   ```bash
git clone https://github.com/your-username/energy-consumption-forecasting.git
   ```
2. Environment Setup:
```bash
pip install -r requirements.txt 
```
3. Launch App:
```bash
streamlit run app.py 
```

**Project Structure**

- `data/`: Contains the energy consumption datasets.
- `models/`: Stores the trained SARIMAX model file.
- `app.py`: The Streamlit application code for running the dashboard.

**Results**
![Untitled design](https://github.com/DataUAcademy/energy-consumption-forecasting/assets/138176913/b50b9c83-35e9-4087-8cde-e912ad4fb075)

- The SARIMAX model, with an RMSE of 0.66 on the test set, demonstrates the model's predictive capabilities.
- Interactive dashboard facilitates real-time insights into energy demand influenced by weather and holiday factors.

**Contact & Connect**

For inquiries or collaboration opportunities, connect with me on [LinkedIn](https://www.linkedin.com/in/sombrathna-sout/).
