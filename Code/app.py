import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.express as px
import plotly.figure_factory as ff

# Load the SARIMAX model
Sarimax_Model = joblib.load('../Model/EnergyForecast.sav')

# Load the model data
model_data = pd.read_hdf('../Data/model_data.h5', key='df')
# Load the additional dataset for insights
insight_data = pd.read_csv('../Data/data.csv')
insight_data = insight_data.iloc[:-1,:]
insight_data['Day'] = pd.to_datetime(insight_data['Day'])
# Split the data into train and test sets
train = model_data.iloc[:-30]
test = model_data.iloc[-30:-1]
def generate_adjusted_forecast(weather_cluster, is_holiday, model, test_data):
    # Adjust 'weather_cluster' and 'holiday_ind' in test_data based on user inputs
    test_data['weather_cluster'] = weather_cluster
    test_data['holiday_ind'] = int(is_holiday)
    
    # Generate forecast using SARIMAX model with adjusted exogenous variables
    forecast_days = len(test_data)
    adjusted_predict = model.predict(n_periods=forecast_days, exogenous=test_data[['weather_cluster', 'holiday_ind']])
    
    return adjusted_predict
# Function to compare energy consumption on holidays vs non-holidays using a box plot
def plot_holiday_impact_on_energy_consumption(df):
    # Ensure the 'Day' column is in datetime format and 'holiday_ind' is categorical
    df['Day'] = pd.to_datetime(df['Day'])
    df['holiday_ind'] = df['holiday_ind'].astype('category')
    
    # Map the 'holiday_ind' to more descriptive labels
    df['Holiday'] = df['holiday_ind'].map({0: 'Non-Holiday', 1: 'Holiday'})
    
    # Creating the box plot
    fig = px.box(df, x='Holiday', y='avg_energy',
                 title='Impact of Holidays on Energy Consumption',
                 labels={'avg_energy': 'Average Energy Consumption per Household (kWh)'},
                 color='Holiday',
                 color_discrete_map={'Non-Holiday': 'blue', 'Holiday': 'red'})

    # Update layout for better readability
    fig.update_layout(showlegend=False)
    return fig
# Function to create a Plotly plot with twin y-axes and a date range slider
def plot_energy_consumption_distribution_plotly(df):
    # Creating the histogram using Plotly Express
    fig = px.histogram(df, x='avg_energy',
                       title='Distribution of Average Daily Energy Consumption',
                       labels={'avg_energy': 'Average Energy Consumption per Household (kWh)'},
                       marginal='box',  # Adds a box plot to the side for additional distribution insight
                       color_discrete_sequence=['#636EFA'])  # Color of the histogram bars

    fig.update_layout(bargap=0.2)  # Adjust the gap between bars
    return fig

def plot_energy_consumption_and_participation_plotly(df):
    # Make sure 'Day' is a datetime type
    df['Day'] = pd.to_datetime(df['Day'])
    
    # Create subplots with shared x-axis (date)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add Average Energy Consumption trace
    fig.add_trace(
        go.Scatter(x=df['Day'], y=df['avg_energy'], name='Avg Energy Consumption (kWh)', marker_color='blue'),
        secondary_y=False,
    )

    # Add Number of Households trace
    fig.add_trace(
        go.Scatter(x=df['Day'], y=df['LCLid'], name='Number of Households', marker_color='red'),
        secondary_y=True,
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Date")

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Average Energy</b> Consumption (kWh)", secondary_y=False, color='blue')
    fig.update_yaxes(title_text="<b>Number of Households</b>", secondary_y=True, color='red')

    # Add slider
    fig.update_layout(
        title="Average Daily Energy Consumption and Household Participation",
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            type="date"
        )
    )

    return fig
#Consumer Behavior Insight
# Ensure 'insight_data' is correctly loaded and 'Day' is properly converted to datetime
insight_data['Day'] = pd.to_datetime(insight_data['Day'])

# Aggregate average energy consumption by household (LCLid)
household_avg_consumption = insight_data['avg_energy'].reset_index()

# Reshape data for clustering (assuming 'avg_energy' is the feature we're clustering on)
X = household_avg_consumption[['avg_energy']].values

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
# Add cluster labels back to the household_avg_consumption DataFrame
household_avg_consumption['Cluster'] = kmeans.labels_

#swtich
combined_data = pd.concat([
    train.assign(type='train'),
    test.assign(type='test')
    ])
st.set_page_config(layout="wide")
col1, col2 = st.columns([1, 10]) # Adjust ratio for desired sidebar width
with col1:
    # Adding "Insight" to the sidebar options
    selected_option = st.sidebar.selectbox("", options=["Model Assessment", "Forecasting", "Insight"], index=0) # Set Default choice
    if selected_option == "Model Assessment":
        st.sidebar.write("Analyze model performance on historical data.")
    elif selected_option == "Forecasting":
        st.sidebar.write("Generate predictions for upcoming energy consumption.")
    elif selected_option == "Insight":
        st.sidebar.write("Explore insights from additional data.")
# Streamlit UI setup
with col2:
    if selected_option == "Model Assessment":
        st.header("Model Prediction and Evaluation")
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

        # Display the  metrics with explanatory text
        st.subheader("Forecast Evaluation")
        st.metric("Average Forecast Off-Target (KW/D)", f"{rmse:.2f}") 
        st.write("Lower value means our predictions are generally closer to reality.")
        # Streamlit UI setup
        st.title("What-If Analysis: Energy Consumption Forecast Adjustment")

        # User inputs for weather cluster and holiday scenario
        weather_cluster = st.selectbox("Select Weather Condition:", 
                                    options=[0, 1, 2], 
                                    format_func=lambda x: ["Hot & Dry Days", "Mild & Humid", "Windy & Temperate"][x])
        is_holiday = st.checkbox("Is it a holiday?")
        test_data = model_data.iloc[-30:-1].copy()  # Example to select the last 30 days as test data

        # Generate adjusted forecast based on user inputs (uncomment and adapt in your application)
        adjusted_predict = generate_adjusted_forecast(weather_cluster, is_holiday, Sarimax_Model, test_data)

        # Visualize the adjusted forecast (uncomment and adapt in your application)
        st.write("Adjusted Forecast:", adjusted_predict)
        st.write("Adjust the parameters above to see how different weather conditions and holidays might affect energy consumption forecasts.")

    elif selected_option == "Forecasting":
        # User input for forecast days
        st.subheader("Forecast average energy consumption/household")
        forecast_days = st.number_input("Enter number of days to forecast:", min_value=1, value=7, max_value = 30)

        # Generate forecast
        predict = Sarimax_Model.predict(n_periods=forecast_days, exogenous=test[['weather_cluster', 'holiday_ind']])

        # Create DataFrame for forecast visualization
        forecast_dates = pd.date_range(start=test.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
        forecast_data = pd.DataFrame({'avg_energy': predict.values}, index=forecast_dates)
        forecast_data['type'] = 'forecast'
        # Assume a flat rate price per kWh (update this to your actual rate)
        st.subheader("Note that this is based on energy rates in 2014 in UK with 0.20$ per KWh")
        price_per_kwh = st.number_input("Enter number of days to forecast:", min_value=0.12, value=0.20, max_value = 1.5)
        # Calculate the cost for the forecasted energy consumption
        forecast_data['Predicted_Cost'] = forecast_data['avg_energy'] * price_per_kwh
        # Create a bar chart for costs
        cost_fig = px.bar(forecast_data, x=forecast_data.index, y='Predicted_Cost',
                        title="Predicted Cost of Energy Consumption",
                        labels={'index': 'Date', 'Predicted_Cost': 'Predicted Cost ($)'})

        st.plotly_chart(cost_fig)
        st.subheader(f"Forecast for the Next {forecast_days} Days")
        detailed_forecast_table = forecast_data[['avg_energy', 'Predicted_Cost']].copy()
        detailed_forecast_table.rename(columns={'avg_energy': 'Predicted Energy (kWh)',
                                                'Predicted_Cost': 'Predicted Cost ($)'},inplace=True)
        st.table(detailed_forecast_table)
    elif selected_option == "Insight":
        st.header("Energy Consumption Insights")
        fig = plot_energy_consumption_and_participation_plotly(insight_data)
        st.plotly_chart(fig, use_container_width=True)
        fig_d = plot_energy_consumption_distribution_plotly(insight_data)
        st.plotly_chart(fig_d, use_container_width=True)
        st.header("Environmental Factors Correlation")
        # Specify the environmental factors you want to include in the correlation matrix
        import plotly.graph_objects as go
        # Example: Comparing avg_energy with temperatureMax
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # Add energy consumption trace
        fig.add_trace(
            go.Scatter(x=insight_data['Day'], y=insight_data['avg_energy'], name="Avg Energy Consumption", marker_color='blue'),
            secondary_y=False,
        )

        # Add temperature trace
        fig.add_trace(
            go.Scatter(x=insight_data['Day'], y=insight_data['temperatureMax'], name="Max Temperature", marker_color='red'),
            secondary_y=True,
        )
        # Add figure title
        fig.update_layout(title_text="Energy Consumption vs. Max Temperature Over Time")
        # Set x-axis title
        fig.update_xaxes(title_text="Date")
        # Set y-axes titles
        fig.update_yaxes(title_text="<b>Average Energy Consumption</b> (kWh)", secondary_y=False)
        fig.update_yaxes(title_text="<b>Max Temperature</b> (°C)", secondary_y=True)

        # Display the figure in Streamlit
        st.plotly_chart(fig)
        st.header("Holiday Impact on Energy Consumption")
        fig_h = plot_holiday_impact_on_energy_consumption(insight_data)
        st.plotly_chart(fig_h, use_container_width=True)
        #Cluster
        fig_c = px.histogram(household_avg_consumption, x='avg_energy', color='Cluster',
                   title='Household Energy Consumption Clusters',
                   labels={'avg_energy': 'Average Energy Consumption (kWh)'},
                   barmode='overlay')
        fig_c.update_traces(opacity=0.75)
        st.plotly_chart(fig_c,use_container_width=True)
        insight_data['Weekday'] = insight_data['Day'].dt.day_name()

        # Aggregate data to find average consumption by day of the week
        avg_consumption_by_weekday = insight_data.groupby('Weekday')['avg_energy'].mean().reset_index()

        # Visualize
        fig_w = px.bar(avg_consumption_by_weekday, x='Weekday', y='avg_energy',
                    title='Average Energy Consumption by Day of the Week',
                    labels={'avg_energy': 'Average Energy Consumption (kWh)'})
        st.plotly_chart(fig_w,use_container_width=True)