# -----------------------------------------------------------------------------
# IMPORTS & CONFIGURATION
# -----------------------------------------------------------------------------
# We import the libraries we need.
import streamlit as st # Streamlit is the framework we use to build the web app.
import pandas as pd # Pandas is the tool we use for tabular data handling.
import yfinance as yf # yfinance is the library we use tu fetch stock data from the Yahoo Finance API.
import numpy as np # Numpy is the library we use for mathematical calculations.
import altair as alt # Altair is used for advanced charts.
from sklearn.ensemble import RandomForestRegressor # We need Sklearn for the Machine Learning part.
from sklearn.metrics import mean_absolute_error # We need Sklearn for the Machine Learning part.

# This must be the first Streamlit command. It sets up the page title and layout.
st.set_page_config(page_title="SMI Stock & Portfolio Comparator", layout="wide") # This sets up the page title in the browser

# Main title of the application
st.title("📈 SMI Stock & Portfolio Comparator")

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def calculate_KPI(df): # 
    """
    We want to calculate different KPI's of the stocks to later compare the stocks to each other in the application.
    We put it all in one function for efficiency and simplicity, as there are some variables we need for different KPI's.
    """
    
    summary = pd.DataFrame(index=df.columns) # We first create an empty data frame to store our results.
    
    # 1. Daily Returns
    returns = df.pct_change().dropna() # We calculate the precentage change in price from yesterday to today. The command ".dropna()" removes the first row, which has no yesterday to compare to.
    
    # 2. Annualized Return
    summary['Ann. Return'] = returns.mean() * 252 # We calculate the average daily return and scale it up to a year. We asssume 252 trading days in a year.

    # 3. Cumulative Return
    summary['Cumulative Return'] = (1 + returns).prod() - 1 # We calculate the total percentage gain/loss over the selected period.
    
    # 4. Annualized Volatility (Risk)
    summary['Ann. Volatility'] = returns.std() * np.sqrt(252) # We calculate the annualized volatility. We assume 252 trading days in a year.
    
    # 5. Sharpe Ratio
    summary['Sharpe Ratio'] = summary['Ann. Return'] / summary['Ann. Volatility'] # We calculate the Sharpe Ratio. A Risk Free Rate of 0% is assumed for simplicity.

    # 6. Sortino Ratio
    downside_returns = returns.copy() # We copy the returns into the new variable "downside_returns" to further process the data.
    downside_returns[downside_returns > 0] = np.nan # For the Sortino Ratio, we neglect upside volatility, therefore we do not consider positive returns for the calculation.
    annual_downside_vol = downside_returns.std() * np.sqrt(252) # We calculate the annual volatility only for negative days.
    summary['Sortino Ratio'] = summary['Ann. Return'] / annual_downside_vol # We calculate the Sortino Ratio.
    
    # 7. Max Drawdown
    # The "Worst Case Scenario": buying at the peak and selling at the bottom.
    # We calculate the cumulative return (growth of $1).
    cumulative_returns_series = (1 + returns).cumprod() # We calculate the cumulative return series, so that we get a value for each day.
    running_max = cumulative_returns_series.cummax() # We store the highest value seen so far as the "running_max"
    drawdown = (cumulative_returns_series / running_max) - 1 # We get a drawdown-value for every day.
    summary['Max Drawdown'] = drawdown.min() # We define the miminum value (the most negative value) of the "drawdown" data frame as the "Max Drawdown".

    # 8. Value at Risk (VaR)
    summary['Value at Risk (95%)'] = returns.quantile(0.05) # We define the 5th percentile of daily returns as the VaR at the 95%-level.
    
    return summary

def prepare_regression_data(series, window=21):
    """
    For the Machine Learning part, we want to predict next-day absolute return (volatility) of a stock.
    We use the absolute returns of the last trading month (21 days) for this.
    """
    df = series.to_frame(name='Close') # We convert the input (series) to a DataFrame
    df['Abs_Return'] = df['Close'].pct_change().abs() # We calculate the absolute daily returns
    df['Target'] = df['Abs_Return'].shift(-1) # We create a new DataFrame to predict tomorrow's volatility using today's data
    for i in range(1, window + 1): # We start a loop which will run 21 times
        df[f'Vol_Lag_{i}'] = df['Abs_Return'].shift(i) # We create new columns for the volatility of each day  
    df = df.dropna() # We remove any row that has missing data to avoid a crash of the model
    feature_cols = [f'Vol_Lag_{i}' for i in range(1, window + 1)] # This creates a list of the column names we created
    return df[feature_cols], df['Target'] # The function returns two separate tables, one containing the lag-columns, one containing the "Target". 

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
# We use 'with st.sidebar' to place everything inside this block on the left side.
with st.sidebar: # We use st.sidebar to place everything inside this block on the sidebar on the left.
    st.header("🎛️ Controls") # This is the header of the sidebar.

    # 1. STOCK SELECTION
    smi_companies = { # We open a dictionary for the stocks that can be picked.
        "^SSMI": "🇨🇭 Swiss Market Index (Benchmark)", # We add the SMI as the Benchmark for our Risk-Return-Analysis.
        "ROG.SW": "Roche", # For each stock, we ad the ticker symbols to get the data from yfinance aswell as the company name to make the application more user-friendly.
        "NESN.SW": "Nestlé",
        "NOVN.SW": "Novartis",
        "UBSG.SW": "UBS Group",
        "ZURN.SW": "Zurich Insurance",
        "CFR.SW": "Richemont",
        "ABBN.SW": "ABB",
        "SIKA.SW": "Sika",
        "LONN.SW": "Lonza",
        "ALC.SW": "Alcon",
        "GIVN.SW": "Givaudan",
        "HOLN.SW": "Holcim",
        "SCMN.SW": "Swisscom",
        "PGHN.SW": "Partners Group",
        "SLHN.SW": "Swiss Life",
        "GEBN.SW": "Geberit",
        "SOON.SW": "Sonova",
        "SREN.SW": "Swiss Re",
        "KNIN.SW": "Kuehne + Nagel",
        "LOGN.SW": "Logitech"
    }


    selectable_tickers = [t for t in smi_companies.keys() if t != "^SSMI"] # We exclude the SMI from the dropdown options to make sure the Benchmark Index cannot be manually deselected.

    tickers = st.multiselect( # This creates the dropdown to pick certain stocks.
        "Select Stocks", 
        options=selectable_tickers, 
        format_func=lambda x: f"{smi_companies[x]} ({x})", # We include a lambda function so that the full company names are shown instead of the ticker symbols.
        default=["NESN.SW", "ROG.SW", "UBSG.SW"]  # By default, the stocks of Nestle, Roche and UBS Group will be selected.
    )

    # 2. DATE SELECTION 
    col1, col2 = st.columns(2) # This builds two columns for the start and end date
    with col1:
        start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01")) # The default Start Date is 2020-01-01
    with col2:
        end_date = st.date_input("End Date", value=pd.to_datetime("today")) # The default End Date is today's date

    # 3. PORTFOLIO BUILDER
    st.markdown("---")
    st.header("⚖️ Portfolio Builder") # This is the second header of the sidebar.
    
    weights = {} # We open up an empty dictionary to store the weights the user enters
    
    # Only show weights input if tickers are selected
    if tickers:
        with st.expander("Assign Weights (%)", expanded=True): # We open an expander to assign the weights, it is expanded by default.
            st.write("Assign percentage weights. Must sum to 100%.") # Descripion for the user.
            
            default_weight = round(100.0 / len(tickers), 2) # We set all weights of the selected stocks equal by default
            
            for t in tickers:
                name = smi_companies[t] # This makes sure we use the company name, not the ticker symbol.
                # Input for Percentage (0-100)
                weights[t] = st.number_input(f"{name} (%)", min_value=0.0, max_value=100.0, value=default_weight, step=1.0) # For any stock, a weight between 0% and 100% can be chosen.
                # If the plus or minus sign are used, the weight will increase or decrease by 1%.

            current_total = sum(weights.values())
            st.write(f"**Total Allocation:** {current_total:.1f}%") # We display the current total allocation
            
            if abs(current_total - 100.0) > 0.1: # We allow for a tiny float error of 0.1% of the chosen weights
                st.error("⚠️ Total must be exactly 100%") # If the total of the picked weights is outside of the float error, an error message will be displayed.
            else:
                st.success("✅ Portfolio Ready") # If the total of the picked weights is good, this message occurs.

# -----------------------------------------------------------------------------
# LOADING DATA
# -----------------------------------------------------------------------------
# This function downloads the data. 
@st.cache_data # This is a decorator provided by the Streamlit library, it saves the data it loaded once in the memory. This saves time as the data is not loaded again.
def load_data(ticker_list, start, end): # We define a function to download the data selected at the sidebar.
    if not ticker_list:
        return pd.DataFrame() # If the list is empty, nothing is downloaded
    
    data = yf.download(ticker_list, start=start, end=end, auto_adjust=True) # We download the data from yfinance. We use auto_adjust=True tu handle splits/dividends.
    
    # Formatting Fix for Single Ticker vs Multiple Tickers
    if len(ticker_list) == 1:
        return data['Close'].to_frame(name=ticker_list[0]) # We need this logic so that yfinance defines a single-stock-ticker the same as a multi-stock table.
    
    return data['Close']

# -----------------------------------------------------------------------------
# MAIN APP LOGIC
# -----------------------------------------------------------------------------
try:
    # 1. PREPARE TICKER LIST
    tickers_to_load = list(set(tickers + ["^SSMI"])) # The tickers that are selected are loaded. The SMI is always loaded.

    # 2. CALL THE FUNCTION
    stock_df = load_data(tickers_to_load, start_date, end_date) # We create the DataFrame for the further analysis.
    
    # 3. CHECK IF DATA IS EMPTY
    if stock_df.empty:
        st.warning("No data found. Please check your date range.") # If the DataFrame is completely empty, a warning is shown. This should never be the case as the SMI is always loaded.
    else:
        if not tickers:
            st.info("Showing Benchmark (SMI) only. Select stocks in the sidebar to compare.") # If only the SMI is loaded, it shows this message.
        else:
            st.success(f"Data loaded successfully for {len(tickers_to_load)} tickers (including Benchmark)!") # If any stocks have been successfully selected, this message shows up.
        
        # Raw Data Preview (Hidden by default)
        with st.expander("📄 View Last 21 Trading Days"): # We show a preview of the loaded data, it is hidden by default. The data can also be downloaded.
             preview_df = stock_df.rename(columns=lambda x: smi_companies.get(x, x)) # The lambda function makes sure that the full company names are shown instead of the ticker symbols.
             st.dataframe(preview_df.tail(21))
             
        csv_data = preview_df.to_csv().encode('utf-8') # We encode the data to a CSV-string to make it ready for the export
             
        st.download_button( # We create a button which allows to download the data
            label="⬇️ Download Raw Price Data (CSV)", # This is the label of the download button
            data=csv_data, 
            file_name="stock_price_data.csv",
            mime="text/csv",
            help="Click to download the full historical price data as a CSV file."
        )
             
        # -----------------------------------------------------------------------------
        # DATA PRE-PROCESSING & PORTFOLIO CALCULATION
        # -----------------------------------------------------------------------------
        cleaned_df = stock_df.dropna() # We drop rows with missing data to ensure a fair comparison
        
        valid_portfolio = False # We set the portfolio as unvalid by default, unless its weight's sum up to 100%
        current_total = sum(weights.values())
        
        if tickers and not cleaned_df.empty and abs(current_total - 100.0) <= 0.1: # Those are the criteria for the portfolio to be valid
            valid_portfolio = True # If the criteria are fulfilled, the portfolio is valid
    
            selected_tickers = cleaned_df[tickers] # We exclude the benchmark from the selected stocks
            
            daily_returns = selected_tickers.pct_change() # We calculate the daily returns for the individual stocks seleted
            
            final_weights = [weights[t] / 100.0 for t in tickers] # We divide the chosen weights by 100 to get the percentages
            
            portfolio_ret = daily_returns.dot(final_weights) # We calculate the portfolio return. The dot-product weighs the single stock returns
            
            my_portfolio_price = (1 + portfolio_ret).cumprod() * 100 # We construct the portfolio price series
            my_portfolio_price.iloc[0] = 100 # We norm it so that the start is at 100
            
            cleaned_df["💼 My Portfolio"] = my_portfolio_price # We add the portfolio to the main DataFrame

        elif tickers and abs(current_total - 100.0) > 0.1:
            st.warning("⚠️ 'My Portfolio' not calculated: Weights do not sum to 100%.") # If the portfolio weights do not sum to 100, this warning is shown

        # -----------------------------------------------------------------------------
        # DYNAMIC KPI VISUALIZER
        # -----------------------------------------------------------------------------
        st.subheader("📊 KPI Visualizer over Time")
        
        if not cleaned_df.empty:
            metric_options = [ # We create a dropdown-menu where users can compare the chosen stocks according to the 7 KPI's we defined
                "Cumulative Return (Indexed to 100)",
                "Annualized Return (30-Day Rolling)",
                "Volatility (30-Day Rolling)",
                "Sharpe Ratio (30-Day Rolling)",
                "Sortino Ratio (30-Day Rolling)",
                "Drawdown (Historical)",
                "Value at Risk 95% (30-Day Rolling)"
            ]
            
            selected_metric = st.selectbox("Select Metric to Plot", metric_options)
            
            returns = cleaned_df.pct_change().dropna() # We calculate the Daily Returns, as we need it for the other KPI's
            window = 30 # For the rolling calculations, we set a window size of 30 days
            
            # We set up a logic that calculates the time series for the chosen KPI. The calculations are the same as for the calculate_KPI function constructed earlier.
            # The difference is that we now need a time series plot instead of only one value for each stock and KPI.
            if selected_metric == "Cumulative Return (Indexed to 100)":
                plot_data = cleaned_df / cleaned_df.iloc[0] * 100
                
            elif selected_metric == "Annualized Return (30-Day Rolling)":
                plot_data = returns.rolling(window=window).mean() * 252
            
            elif selected_metric == "Volatility (30-Day Rolling)":
                plot_data = returns.rolling(window=window).std() * np.sqrt(252)
                
            elif selected_metric == "Sharpe Ratio (30-Day Rolling)":
                rolling_return = returns.rolling(window=window).mean() * 252
                rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
                plot_data = rolling_return / rolling_vol
            
            elif selected_metric == "Sortino Ratio (30-Day Rolling)":
                downside = returns.copy()
                downside[downside > 0] = np.nan
                rolling_downside_vol = downside.rolling(window=window).std() * np.sqrt(252)
                rolling_return = returns.rolling(window=window).mean() * 252
                plot_data = rolling_return / rolling_downside_vol
                
            elif selected_metric == "Drawdown (Historical)":
                cumulative_rets = (1 + returns).cumprod()
                running_max = cumulative_rets.cummax()
                plot_data = (cumulative_rets / running_max) - 1
                
            elif selected_metric == "Value at Risk 95% (30-Day Rolling)":
                plot_data = returns.rolling(window=window).quantile(0.05)


            plot_data = plot_data.rename(columns=lambda x: smi_companies.get(x, x)) # The lambda function makes sure that the full company names are shown instead of the ticker symbols.
            
            st.line_chart(plot_data) # We create a line chart that shows the time series of the selected stocks and KPI.
            
        else:
            st.info("Not enough shared data points to plot a comparison. Try adjusting dates.") # This comes up if there is no data in the cleaned_df.

        # -----------------------------------------------------------------------------
        # CALCULATE RISK & RETURN METRICS
        # -----------------------------------------------------------------------------
        # Our goal here is to create a scatterplot where the user can compare two different metrices to each other. This will allow to make a risk-return-analysis.
        st.subheader("📉 Risk & Return Analysis")
        
        metrics_df = calculate_KPI(cleaned_df) # We call the helper function 
        
        metrics_df = metrics_df.rename(index=lambda x: smi_companies.get(x, x)) # The lambda function makes sure that the full company names are shown instead of the ticker symbols.

        metrics_df.index.name = "Stock" # We name the index "Stock".
        scatter_data = metrics_df.reset_index() # We reset the index so that "Stock" is a column, not an index.
        
        # Map the internal column names to nice labels for the chart
        # We replace dots with spaces or underscores to avoid Altair errors
        col_mapping = {
            'Ann. Return': 'Annualized Return',
            'Cumulative Return': 'Cumulative Return',
            'Ann. Volatility': 'Annualized Volatility',
            'Sharpe Ratio': 'Sharpe Ratio',
            'Sortino Ratio': 'Sortino Ratio',
            'Max Drawdown': 'Max Drawdown',
            'Value at Risk (95%)': 'Value at Risk 95%'
        }
        
        # Rename the columns in our data to match the nice labels
        scatter_data = scatter_data.rename(columns=col_mapping)
        
        # 3. Create Dropdowns for X and Y Axes
        st.markdown("##### Compare Metrics (Scatter Plot)")
        col_x, col_y = st.columns(2)
        
        # Get the list of available options (the nice labels)
        chart_opts = list(col_mapping.values())
        
        with col_x:
            # Default X: Volatility
            x_axis = st.selectbox("X-Axis", chart_opts, index=chart_opts.index('Annualized Volatility'))
        with col_y:
            # Default Y: Return
            y_axis = st.selectbox("Y-Axis", chart_opts, index=chart_opts.index('Annualized Return'))
            
        # 4. Dynamic Formatting
        # If the user selects a Ratio (Sharpe/Sortino), display as number (2.50).
        # Otherwise display as percentage (15%).
        x_format = ".2f" if "Ratio" in x_axis else "%"
        y_format = ".2f" if "Ratio" in y_axis else "%"
        
        # 5. Create Altair Chart
        chart = alt.Chart(scatter_data).mark_circle(size=100).encode(
            x=alt.X(x_axis, title=x_axis, axis=alt.Axis(format=x_format)),
            y=alt.Y(y_axis, title=y_axis, axis=alt.Axis(format=y_format)),
            color='Stock',
            # Tooltip will show all metrics for easy inspection
            tooltip=['Stock'] + chart_opts
        ).interactive() 
        
        st.altair_chart(chart, use_container_width=True)
        
        # 6. Format and Display Summary Table
        # We use specific formatting for percentages and decimals
        formatted_metrics = metrics_df.style.format({
            'Ann. Return': '{:.2%}',
            'Cumulative Return': '{:.2%}',
            'Ann. Volatility': '{:.2%}',
            'Sharpe Ratio': '{:.2f}',
            'Sortino Ratio': '{:.2f}',
            'Max Drawdown': '{:.2%}',
            'Value at Risk (95%)': '{:.2%}'
        })
        
        st.markdown("##### Detailed Metrics Table")
        st.dataframe(formatted_metrics)

        # -----------------------------------------------------------------------------
        # SNIPPET 7: MACHINE LEARNING (Updated for Regression)
        # -----------------------------------------------------------------------------
        st.markdown("---")
        st.header("🤖 Machine Learning: Volatility Prediction")
        
        st.write("""
        This model predicts the **Exact Volatility** (Absolute Daily Return) for the next trading day.
        It uses the past 21 days of volatility to learn patterns using a Random Forest Regressor.
        """)
        
        # 1. User Selects a Stock
        ml_opts = [t for t in tickers if t in cleaned_df.columns]
        ml_ticker = st.selectbox("Select Stock to Predict", ml_opts, format_func=lambda x: smi_companies.get(x, x))
        
        if ml_ticker:
            # 2. Prepare Data
            subset_series = cleaned_df[ml_ticker]
            
            # Use our new helper function for Regression
            X, y = prepare_regression_data(subset_series)
            
            if len(X) > 50:
                # 3. Split Data
                split_index = int(len(X) * 0.8)
                X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
                y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
                
                # 4. Train Model (Random Forest Regressor)
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # 5. Evaluate
                preds = model.predict(X_test)
                mae = mean_absolute_error(y_test, preds)
                
                # Display Results
                st.markdown(f"#### Volatility Forecast for **{smi_companies.get(ml_ticker, ml_ticker)}**")
                
                # Show the predicted volatility for the NEXT day (using the very latest data)
                last_5_days = X.iloc[-1:].values
                next_day_pred = model.predict(last_5_days)[0]
                
                col1, col2 = st.columns(2)
                col1.metric("Predicted Volatility (Next Day)", f"{next_day_pred:.2%}")
                col2.metric("Mean Absolute Error (Test Set)", f"{mae:.2%}")
                
                # Visualization: Predicted vs Actual
                # Create a DataFrame for plotting
                results_df = pd.DataFrame({
                    'Date': y_test.index,
                    'Actual Volatility': y_test.values,
                    'Predicted Volatility': preds
                }).set_index('Date')
                
                st.write("**Predicted vs. Actual Volatility (Test Set):**")
                st.line_chart(results_df)
                
                st.caption("Lower MAE is better. If the lines overlap, the model is doing a good job.")
                
            else:
                st.warning("Not enough data. Try a longer date range.")

except Exception as e:
    # st.error shows a red error box if something crashes
    st.error(f"An error occurred: {e}")
