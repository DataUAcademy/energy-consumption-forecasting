import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the SARIMAX model
Sarimax_Model = joblib.load('../Model/EnergyForecast.sav')

# Load the model data
model_data = pd.read_hdf('../Data/model_data.h5', key='df')

# Split the data into train and test sets
train = model_data.iloc[:-30]
test = model_data.iloc[-30:-1]

#swtich
combined_data = pd.concat([
    train.assign(type='train'),
    test.assign(type='test')
    ])
st.set_page_config(layout="wide")
col1, col2 = st.columns([1, 10]) # Adjust ratio for desired sidebar width
with col1:
    selected_option = st.sidebar.selectbox("", options=["Model Assessment", "Forecasting"], index=0) # Set Default choice
    if selected_option == "Model Assessment":
        st.sidebar.write("Analyze model performance on historical data.")
    else:
        st.sidebar.write("Generate predictions for upcoming energy consumption.")
# Streamlit UI setup
with col2:
    if selected_option == "Model Assessment":
        st.header("Model Prediction and Evaluation")

        # # Train and Test avg_energy KW/D Line Chart on the same graph
        # st.title("Training and Testing set")
        # st.subheader("Train vs. Test avg_energy KW/D")

        # # Plot train and test data using Streamlit line_chart
        # # st.line_chart(train[['avg_energy', 'avg_energy']], use_container_width=True, color=['#ffaa00', 'rgba(255, 170, 0, 0.2)'])
        # # st.line_chart(test[['avg_energy', 'avg_energy']], use_container_width=True, color=['rgba(255, 170, 0, 0.2)', '#0000FF'])
        # st.line_chart(combined_data, y='avg_energy', color='type', use_container_width=True)

        # Forecast 29 days after the training set
        st.subheader("Forecast vs Actual Average Energy Consumption Per household")
        forecast_days = 29
        predict = Sarimax_Model.predict(n_periods=forecast_days, exogenous=test[['weather_cluster', 'holiday_ind']])
        test['Predicted'] = predict.values


        # Actual vs. Predicted Energy Consumption Line Chart
        st.line_chart(test[['avg_energy', 'Predicted']], use_container_width=True, color=['#ffaa00', '#0000FF'])
        # st.pyplot(fig)  # Display the current figure
        # Calculate RMSE for forecast
        rmse = np.sqrt(mean_squared_error(test['avg_energy'], test['Predicted']))
        # mae = mean_absolute_error(test['avg_energy'], test['Predicted'])

        # Display the  metrics with explanatory text
        st.subheader("Forecast Evaluation")
        st.metric("Average Forecast Off-Target (KW/D)", f"{rmse:.2f}") 
        st.write("Lower value means our predictions are generally closer to reality.")
        # st.metric("Typical Deviation (KW/D)", mae)
        # st.write("A way to gauge the size of error we might expect day-to-day.")

    else:
        # User input for forecast days
        st.subheader("Forecast average energy consumption/household")
        forecast_days = st.number_input("Enter number of days to forecast:", min_value=1, value=7, max_value = 30)

        # Generate forecast
        predict = Sarimax_Model.predict(n_periods=forecast_days, exogenous=test[['weather_cluster', 'holiday_ind']])

        # Create DataFrame for forecast visualization
        forecast_dates = pd.date_range(start=test.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
        forecast_data = pd.DataFrame({'avg_energy': predict.values}, index=forecast_dates)
        forecast_data['type'] = 'forecast' 

        # # Combine test and forecast data
        # updated_combined_data = pd.concat([combined_data, forecast_data])

        test_and_forecast_data = pd.concat([test.assign(type='test'), forecast_data.assign(type='forecast')])
        # Plot TEST, and forecast data
        st.subheader("Time Series Comparison with Forecast")
        st.line_chart(test_and_forecast_data, y='avg_energy', color='type', use_container_width=True)
        st.subheader(f"{forecast_days}" + " Days Forward Average Energy Consumption Per Household")
        forecast_table = forecast_data.reset_index().rename(columns={'index': 'Date'})[['Date', 'avg_energy']] 
        st.table(forecast_table)