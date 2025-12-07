# This application was built using Google Gemini 3.0 Pro. We built the base code in cooperation with Google Gemini, modified and commented the code ourselves.

# -----------------------------------------------------------------------------
# IMPORTS & CONFIGURATION
# -----------------------------------------------------------------------------

# We import the libraries we need.
import streamlit as st # Streamlit is the framework we use to build the web app.
import pandas as pd # Pandas is the tool we use for tabular data handling.
import yfinance as yf # yfinance is the library we use tu fetch stock data from the Yahoo Finance API.
import numpy as np # Numpy is the library we use for mathematical calculations.
import altair as alt # We use Altair for advanced charts. We use it instead of matplotlip which was introduced in the lecture, because we liked the interaction possibilities.
from sklearn.ensemble import RandomForestRegressor # We need Sklearn for the Machine Learning part.
from sklearn.metrics import mean_absolute_error # We need Sklearn for the Machine Learning part.

# This must be the first Streamlit command. It sets up the page title and layout.
st.set_page_config(page_title="The Extra SMIle", layout="wide") # This sets up the page title in the browser.

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def calculate_KPI(df, risk_free_rate=0.0): 
   # We want to calculate different KPI's of the stocks to later compare the stocks to each other in the application.
   # We put it all in one function for efficiency and simplicity, as there are some variables we need for different KPI's.
   
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
    # We use the user-defined risk_free_rate passed to the function.
    summary['Sharpe Ratio'] = (summary['Ann. Return'] - risk_free_rate) / summary['Ann. Volatility'] 

    # 6. Sortino Ratio
    downside_returns = returns.copy() # We copy the returns into the new variable "downside_returns" to further process the data.
    downside_returns[downside_returns > 0] = np.nan # For the Sortino Ratio, we neglect upside volatility, therefore we do not consider positive returns for the calculation.
    annual_downside_vol = downside_returns.std() * np.sqrt(252) # We calculate the annual volatility only for negative days.
    # We use the user-defined risk_free_rate passed to the function.
    summary['Sortino Ratio'] = (summary['Ann. Return'] - risk_free_rate) / annual_downside_vol # We calculate the Sortino Ratio.
    
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

def prepare_regression_data(series, window=21, horizon=1):
    # For the Machine Learning part, we want to predict the absolute return (volatility) of a stock.
    # We use the absolute returns of the last trading month (21 days) for this.
    
    # Robust check: If the input is already a DataFrame, we use it directly. If the input is a Series, we convert it to a DataFrame.
    if isinstance(series, pd.DataFrame):
        df = series.copy()
        if len(df.columns) > 0:
            df.columns = ['Close']
    else:
        df = series.to_frame(name='Close') 
        
    df['Abs_Return'] = df['Close'].pct_change().abs() # We calculate the absolute daily returns.
    
    # TARGET CREATION:
    # If the horizon is 1, we just shift by -1 (tomorrow).
    # If the horizon is > 1, we calculate the rolling average of the FUTURE and shift it back.
    if horizon == 1:
        df['Target'] = df['Abs_Return'].shift(-1)
    else:
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=horizon)
        df['Target'] = df['Abs_Return'].rolling(window=indexer).mean()

    # Features: Recent volatility (Lag 1 to Lag 21)
    for i in range(1, window + 1): # We start a loop which will run 21 times.
        df[f'Vol_Lag_{i}'] = df['Abs_Return'].shift(i) # We create new columns for the volatility of each day.
    
    df = df.dropna() # We remove any row that has missing data to avoid a crash of the model.
    feature_cols = [f'Vol_Lag_{i}' for i in range(1, window + 1)] # This creates a list of the column names we created.
    return df[feature_cols], df['Target'] # The function returns two separate tables, one containing the lag-columns, one containing the "Target". 

# -----------------------------------------------------------------------------
# LOADING DATA FUNCTION
# -----------------------------------------------------------------------------

# This function downloads the data. 
@st.cache_data # This is a decorator provided by the Streamlit library, it saves the data it loaded once in the memory. This saves time as the data is not loaded again.
def load_data(ticker_list, start, end): # We define a function to download the data selected at the sidebar.
    if not ticker_list:
        return pd.DataFrame() # If the list is empty, nothing is downloaded
    
    # CRITICAL CHANGE: We always download from a "safe start" (2 years ago) to ensure ML works.
    # We will filter for the user's dates later for display purposes.
    safe_start = pd.Timestamp.today() - pd.DateOffset(years=2)
    # We use the earlier of the two dates (User's start or Safe start)
    download_start = min(start, safe_start)
    
    data = yf.download(ticker_list, start=download_start, end=end, auto_adjust=True) # We download the data from yfinance. We use auto_adjust=True to handle splits/dividends.
    
    if len(ticker_list) == 1:
        return data['Close'].to_frame(name=ticker_list[0]) # We need this logic so that yfinance defines a single-stock-ticker the same as a multi-stock table.
    
    return data['Close']

# -----------------------------------------------------------------------------
# MAIN LAYOUT & CONTROLS
# -----------------------------------------------------------------------------

st.title("The Extra SMIle") # This is the main title of the app.

st.caption("*Going the extra SMIle for your returns.*") # This is our slogan.

# We split the page into four different latches to make it more user-friendly.
page = st.radio("Navigation", ["Guide", "KPI Visualizer", "Risk & Correlation", "Volatility Forecasting"], horizontal=True, label_visibility="collapsed")
st.markdown("---")

# SIDEBAR: CONTROLS
with st.sidebar: # We use st.sidebar to place everything inside this block on the sidebar on the left.
    st.header("Controls") # This is the header of the sidebar.
    
    smi_companies = { # We open a dictionary for the stocks that can be picked.
        "^SSMI": "Swiss Market Index (Benchmark)", # We add the SMI as the Benchmark for our Risk-Return-Analysis.
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

    # 1. STOCK SELECTION
    selectable_tickers = [t for t in smi_companies.keys() if t != "^SSMI"]# We exclude the SMI from the dropdown options to make sure the Benchmark Index cannot be manually deselected.
    tickers = st.multiselect( # This creates the dropdown to pick certain stocks.
        "Select Stocks", 
        options=selectable_tickers, 
        format_func=lambda x: f"{smi_companies[x]} ({x})", # We include a lambda function so that the full company names are shown instead of the ticker symbols.
        default=["NESN.SW", "NOVN.SW", "UBSG.SW"] # By default, the stocks of Nestle, Roche and UBS Group will be selected.
    )

    # 2. DATE SELECTION
    col_d1, col_d2 = st.columns(2) # This builds two columns for the start and end date.
    with col_d1:
        start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01")) # The default Start Date is 2020-01-01.
    with col_d2:
        end_date = st.date_input("End Date", value=pd.to_datetime("today")) # The default End Date is today's date.

    # 3. PORTFOLIO BUILDER
    st.markdown("---")
    st.header(" Portfolio Builder") # This is the second header of the sidebar.
    
    weights = {} # We open up an empty dictionary to store the weights the user enters
    
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
                st.error("Total must be exactly 100%") # If the total of the picked weights is outside of the float error, an error message will be displayed.
            else:
                st.success("Portfolio Ready") # If the total of the picked weights is good, this message occurs.
    else:
        st.info("Please select a stock.") # If no stock is selected, the app asks the user to select a stock.

    # 4. RISK FREE RATE
    # For the calculation of the Sharpe and Sortino Ratios, we need an assumption for the risk-free interest rate.
    st.markdown("---")
    st.header("Risk Free Rate Assumption")
    rf_input = st.number_input(
        "Risk Free Rate (%)",
        value=1.0, # The value for the risk-free rate is set to 1% by default.
        step=0.1,
        help="Used for Sharpe/Sortino Ratios."
    )
    risk_free_rate_val = rf_input / 100.0 # We divide the selected number by 100 to make it a percentage.

# -----------------------------------------------------------------------------
# MAIN APP LOGIC
# -----------------------------------------------------------------------------
try:
    # 0. CHECK IF STOCKS ARE SELECTED
    if not tickers:
        st.warning("Please select a stock") # If no stock is selected, the app asks the user to select a stock.
        st.stop()
        
     # 0.1 DATE CHECKS (NEW)
    if start_date > end_date:
        st.error("Error: Start Date must be before End Date.") # If the selected start date is later than the selected end date, the app asks the user to fix this.
        st.stop()
        
    if start_date > pd.Timestamp.now().date():
        st.error("Error: Start Date must not be in the future.") # If the selected start date is in the future, the app asks the user to fix this.
        st.stop()
    
    if end_date > pd.Timestamp.now().date():
        st.warning("Warning: Future End Date selected. Data will only be available up to the last trading day.") # If the selected end date is in the future, the app warns the user.
    
    # 1. PREPARE TICKER LIST
    tickers_to_load = list(set(tickers + ["^SSMI"])) # The tickers that are selected are loaded. The SMI is always loaded.

    # 2. CALL THE FUNCTION
    # Note: load_data now handles fetching extended history internally
    full_history_df = load_data(tickers_to_load, pd.Timestamp(start_date), pd.Timestamp(end_date)) 
    
    # 3. CHECK IF DATA IS EMPTY
    if full_history_df.empty:
        st.warning("No data found. Please check your date range.") # If the DataFrame is completely empty, a warning is shown.
    else:
        # -----------------------------------------------------------------------------
        # DATA PRE-PROCESSING & PORTFOLIO CALCULATION
        # -----------------------------------------------------------------------------
        
        cleaned_df = full_history_df.dropna() # We remove rows with NaN values.
        
        valid_portfolio = False 
        current_total = sum(weights.values()) # We calculate the sum of the weights entered by the user to verify it equals 100%.
        
        if tickers and not cleaned_df.empty and abs(current_total - 100.0) <= 0.1: # We check if we have tickers, data and that the weights sum to 100%.
            valid_portfolio = True 
    
            selected_tickers = cleaned_df[tickers] # We slice the dataframe to keep only the selected stocks.
            daily_returns = selected_tickers.pct_change() # We calculate daily returns.
            final_weights = [weights[t] / 100.0 for t in tickers] # We calculate the weights of the stocks.
            
            portfolio_ret = daily_returns.dot(final_weights) # We calculate the portfolio returns.
            
            my_portfolio_price = (1 + portfolio_ret).cumprod() * 100 # We construct the price series for the portfolio.
            my_portfolio_price.iloc[0] = 100 # We ensure that the first day is exactly at 100.
            
            cleaned_df["My Portfolio"] = my_portfolio_price  # We add the portfolio to our main dataframe.

        # -----------------------------------------------------------------------------
        # CREATE DISPLAY DATAFRAME (FILTERED BY USER DATE)
        # -----------------------------------------------------------------------------
        display_start = pd.Timestamp(start_date).tz_localize(None) # We make sure the timezones match to prevent slicing errors.
        display_end = pd.Timestamp(end_date).tz_localize(None)
        display_df = cleaned_df.loc[display_start:display_end] # We keep only rows between the user's chosen Start and End dates.

        # -----------------------------------------------------------------------------
        # PAGE 1: GUIDE 
        # -----------------------------------------------------------------------------
        # This is our first page, which gives fundamental information about the use case of the app.
        if page == "Guide":
            st.header("Welcome to the SMI Stock & Portfolio Comparator") # This is teh header of the page.
            
            st.markdown("""
            This application is designed to help investors look beyond simple price charts. 
            It enables you to analyze historical performance, understand complex risk metrics, and forecast future volatility.
            This will help you to make better investment decisions.
            """) # We describe the main goal of the app.
            
            st.subheader("Sidebar Controls & Portfolio Builder")
            
            col_f1, col_f2, col_f3 = st.columns(3) # We create three columns, in which we describe the three further pages of the app.
            
            with col_f1:
                st.info("**Controls**")
                st.write("Pick the stocks you want to compare and want to build a portfolio with. Choose the start and end date of the analysis. The earliest date you might pick is 2010/01/01.")
                
            with col_f2:
                st.warning("**Portfolio Builder**")
                st.write("Assign weights to your selected stocks to build your own portfolio. You can then compare your portfolio to the single stocks and the SMI Benchmark in the analyses. Make sure your assigned weights sum up to 100% (0.1% tolerance).")
                
            with col_f3:
                st.success("**Risk Free Rate Assumption**")
                st.write("Choose your risk free rate. This rate is used for the calculation of the Sharpe and Sortino Ratios.")
            
            st.subheader("Analysis Pages")
            
            col_g1, col_g2, col_g3 = st.columns(3) # We create three columns, in which we describe the three further pages of the app.
            
            with col_g1:
                st.info("**KPI Visualizer**")
                st.write("View historical performance over time. Compare your portfolio against the SMI Benchmark and visualize metrics like cumulative returns and volatility.")
                
            with col_g2:
                st.warning("**Risk & Correlation**")
                st.write("Dive deep into the risk-return relationship. Use the scatter plot to find the 'efficient frontier' and use the correlation matrix to check if your stocks are diversified.")
                
            with col_g3:
                st.success("**Volatility Forecasting**")
                st.write("Use our random forest machine learning model to predict how volatile a stock might be in the future.")
            
            st.markdown("---")
            st.subheader("Financial Glossary")
            st.write("Understanding the metrics used in this application:")
            
            with st.expander("See Definitions", expanded=True): # We explain the different metrics we use in the app.
                st.markdown("""
                * **Cumulative Return:** The total percentage change in the price of an investment over a specific period.
                * **Annualized Return:** The geometric average amount of money earned by an investment each year over a given time period. It allows for comparison between investments held for different lengths of time.
                * **Volatility:** A statistical measure of the dispersion of returns. High volatility means the price swings up and down drastically.
                * **Sharpe Ratio:** Measures the performance of an investment compared to a risk-free asset, after adjusting for its risk. (Formula: Excess Return / Volatility)
                * **Sortino Ratio:** Similar to the Sharpe Ratio, but it only penalizes *negative* volatility (downside risk). It ignores upside volatility, which is usually good for investors.
                * **Maximum Drawdown:** The maximum observed loss from a peak to a low of a portfolio, before a new peak is attained. It indicates the "worst case scenario" for holding a stock.
                * **Value at Risk (95%):** Estimates how much a set of investments might lose on a single day (with a 95% confidence level), given normal market conditions. For example, if VaR is -2%, it means in 95 out of 100 days, you won't lose more than 2%.
                * **Volatility:** A statistical measure of the dispersion of returns. High volatility means the price swings up and down drastically.        """)

        # -----------------------------------------------------------------------------
        # PAGE 2: KPI VISUALIZER
        # -----------------------------------------------------------------------------
        elif page == "KPI Visualizer": 
            st.subheader("KPI Visualizer over Time")
            
            st.write("""
            This plot shows the historical development of a chosen KPI over the selected time period. You also have the opportunity to download the raw price data of your selected assets.
            """) 
            
            # Raw Data Preview
            with st.expander("View Last 21 Trading Days"): # We show a preview of the loaded data, it is hidden by default. The data can also be downloaded.
                 preview_df = display_df.rename(columns=lambda x: smi_companies.get(x, x)) # The lambda function makes sure that the full company names are shown instead of the ticker symbols.
                 st.dataframe(preview_df.tail(21))
                 
            csv_data = preview_df.to_csv().encode('utf-8') # We encode the data to a CSV-string to make it ready for the export
                 
            st.download_button( # We create a button which allows to download the data
                label="Download Raw Price Data (CSV)", 
                data=csv_data, 
                file_name="stock_price_data.csv",
                mime="text/csv"
            )

            if not display_df.empty:
                metric_options = [ # We create a dropdown-menu where users can compare the chosen stocks according to the 7 KPI's we defined
                    "Cumulative Return (Indexed to 100)",
                    "Annualized Return (30-Day Rolling)",
                    "Volatility (30-Day Rolling)",
                    "Sharpe Ratio (30-Day Rolling)",
                    "Sortino Ratio (30-Day Rolling)",
                    "Maximum Drawdown (Historical)",
                    "Value at Risk 95% (30-Day Rolling)"
                ]
                
                selected_metric = st.selectbox("Select Metric to Plot", metric_options)
                
                # We set up a logic that calculates the time series for the chosen KPI. The calculations are the same as for the calculate_KPI function constructed earlier.
                # The difference is that we now need a time series plot instead of only one value for each stock and KPI.
                returns = display_df.pct_change().dropna() 
                window = 30 
                
                if selected_metric == "Cumulative Return (Indexed to 100)":
                    plot_data = display_df / display_df.iloc[0] * 100
                    
                elif selected_metric == "Annualized Return (30-Day Rolling)":
                    plot_data = returns.rolling(window=window).mean() * 252
                
                elif selected_metric == "Volatility (30-Day Rolling)":
                    plot_data = returns.rolling(window=window).std() * np.sqrt(252)
                    
                elif selected_metric == "Sharpe Ratio (30-Day Rolling)":
                    rolling_return = returns.rolling(window=window).mean() * 252
                    rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
                    plot_data = (rolling_return - risk_free_rate_val) / rolling_vol
                
                elif selected_metric == "Sortino Ratio (30-Day Rolling)":
                    downside = returns.copy()
                    downside[downside > 0] = np.nan
                    rolling_downside_vol = downside.rolling(window=window).std() * np.sqrt(252)
                    rolling_return = returns.rolling(window=window).mean() * 252
                    plot_data = (rolling_return - risk_free_rate_val) / rolling_downside_vol
                    
                elif selected_metric == "Maximum Drawdown (Historical)":
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
        # PAGE 3: RISK & CORRELATION
        # -----------------------------------------------------------------------------
        elif page == "Risk & Correlation":
            # Our goal here is to create a scatterplot where the user can compare two different metrices to each other. This will allow to make a risk-return-analysis.
            st.subheader("Risk & Return Analysis")
            
            st.write("In this plot, you can choose two metrics to compare your assets.")
            
            metrics_df = calculate_KPI(display_df, risk_free_rate=risk_free_rate_val) # We call the helper function 
            
            metrics_df = metrics_df.rename(index=lambda x: smi_companies.get(x, x)) # The lambda function makes sure that the full company names are shown instead of the ticker symbols.

            metrics_df.index.name = "Stock" # We name the index "Stock".
            scatter_data = metrics_df.reset_index() # We reset the index so that "Stock" is a column, not an index.
            
            # We map the internal column names to labels for the chart
            col_mapping = {
                'Ann. Return': 'Annualized Return',
                'Cumulative Return': 'Cumulative Return',
                'Ann. Volatility': 'Annualized Volatility',
                'Sharpe Ratio': 'Sharpe Ratio',
                'Sortino Ratio': 'Sortino Ratio',
                'Max Drawdown': 'Max Drawdown',
                'Value at Risk (95%)': 'Value at Risk 95%'
            }
            
            scatter_data = scatter_data.rename(columns=col_mapping) # We rename the columns in our data to match the new labels

            st.markdown("##### Compare Metrics (Scatter Plot)") # This is the title of the scatter plot.
            col_x, col_y = st.columns(2) # We create the two columns to pick the values for the x and y axes.
            
            chart_opts = list(col_mapping.values()) # We create the list of the available metrics to get the chart options afterwards.
            
            # We create the dropdown menu.
            # We set the Annualized Volatility and Anualized Return as the default metrics
            with col_x:
                x_axis = st.selectbox("X-Axis", chart_opts, index=chart_opts.index('Annualized Volatility'))
            with col_y:
                y_axis = st.selectbox("Y-Axis", chart_opts, index=chart_opts.index('Annualized Return'))
             
            # We use dynamic formatting so that if the user selects the Sharepe or Sortino ratio, it is displayed as a number. The other metrics are displayed as a percentage.
            x_format = ".2f" if "Ratio" in x_axis else "%"
            y_format = ".2f" if "Ratio" in y_axis else "%"
            
            # We create the scatterplot using Altair.
            chart = alt.Chart(scatter_data).mark_circle(size=100).encode(
                x=alt.X(x_axis, title=x_axis, axis=alt.Axis(format=x_format)),
                y=alt.Y(y_axis, title=y_axis, axis=alt.Axis(format=y_format)),
                color='Stock',
                tooltip=['Stock'] + chart_opts
            ).interactive() 
            
            st.altair_chart(chart, use_container_width=True) # We load the chart in streamlit
            
            # We format the summary table. The numbers are rounded to two decimal places.
            formatted_metrics = metrics_df.style.format({
                'Ann. Return': '{:.2%}',
                'Cumulative Return': '{:.2%}',
                'Ann. Volatility': '{:.2%}',
                'Sharpe Ratio': '{:.2f}',
                'Sortino Ratio': '{:.2f}',
                'Max Drawdown': '{:.2%}',
                'Value at Risk (95%)': '{:.2%}'
            })
            
            st.markdown("##### Detailed Metrics Table") # This is the title of the table.
            st.dataframe(formatted_metrics) # We create the table with all KPI's for the selected stocks.

            # Correlation Section
            st.markdown("---")
            st.subheader("Correlation Matrix")
            
            st.write("This matrix shows how the returns of assets move together. +1 means they move perfectly in sync (blue), -1 means they move in opposite directions (red).")

            if not display_df.empty:
                corr_returns = display_df.pct_change().dropna() # We calculate the correlation of the daily returns of the chosen assets.
                corr_matrix = corr_returns.corr() 
                
                if len(corr_matrix.columns) > 1: # If we have at least two assets selected (which we always should have, since we have the SMI-benchmark), the matrix is created.
                    corr_matrix_renamed = corr_matrix.rename(index=lambda x: smi_companies.get(x, x), columns=lambda x: smi_companies.get(x, x)) # The lambda function makes sure that the full company names are shown instead of the ticker symbols.
                    corr_data = corr_matrix_renamed.reset_index() # We transform the data to use it in the chart afterwards.
                    corr_data = corr_data.rename(columns={corr_data.columns[0]: 'Stock A'})
                    corr_data = corr_data.melt(id_vars='Stock A')
                    corr_data.columns = ['Stock A', 'Stock B', 'Correlation']

                    heatmap = alt.Chart(corr_data).mark_rect().encode( # We build the heatmap chart.
                        x=alt.X('Stock A', title=None),
                        y=alt.Y('Stock B', title=None),
                        color=alt.Color('Correlation', scale=alt.Scale(domain=[-1, 1], scheme='redblue')),
                        tooltip=['Stock A', 'Stock B', alt.Tooltip('Correlation', format='.2f')]
                    ).properties(
                        height=500
                    )

                    text = heatmap.mark_text(baseline='middle').encode( # We overlay the correlation values as text in the matrix.
                        text=alt.Text('Correlation', format='.2f'),
                        color=alt.condition(
                            (alt.datum.Correlation > 0.5) | (alt.datum.Correlation < -0.5),
                            alt.value('white'), # We use white text if the absolute correlation value is high, as we then have darker backgrounds.
                            alt.value('black') # We use black text if the absolute correlation value is low, as we then have lighter backgrounds.
                        )
                    )

                    st.altair_chart(heatmap + text, use_container_width=True)
                else:
                    st.info("Select at least 2 assets to view correlations.") # If there are not at least two assets selected, this is shown. This should never be the case as the SMI-benchmark is always selected.

        # -----------------------------------------------------------------------------
        # PAGE 4: VOLATILITY FORECASTING
        # -----------------------------------------------------------------------------
        elif page == "Volatility Forecasting":
            st.subheader("Volatility Prediction") 
            
            st.write("""
            This model predicts the volatility (average absolute daily return) over the chosen time horizon. The time horizon can be chosen between one day, one week (5 trading days) or one  month (21 trading days).
            It uses the past 21 days of volatility to learn patterns using a random forest regressor.
            """) 
            
            # The user has to select an asset for the volatility forecasting.
            ml_opts = list(cleaned_df.columns) # The options to choose from are the selected stocks, aswell as the created portfolio and the SMI-benchmark.
            col_ml_1, col_ml_2 = st.columns(2) # We create two columns. In the first one, the asset can be picked. In the second one, the time horizon can be picked.
            
            with col_ml_1: # We create a ticker to select the stocks.
                ml_ticker = st.selectbox("Select Asset to Predict", ml_opts, format_func=lambda x: smi_companies.get(x, x)) # The lambda function makes sure that the full company names are shown instead of the ticker symbols.
             
            with col_ml_2: # We create a ticker to select the time horizon.
                horizon_dict = {"Next Day": 1, "Next Week (5 Days)": 5, "Next Month (21 Days)": 21}
                horizon_label = st.selectbox("Select Forecast Horizon", list(horizon_dict.keys()))
                horizon_val = horizon_dict[horizon_label]

            if ml_ticker:
                subset_series = cleaned_df[ml_ticker].dropna()
                X, y = prepare_regression_data(subset_series, window=21, horizon=horizon_val) # We use the helper function we defined earlier.
                
                if len(X) > 50: # We use at least 50 data points (days) for the model.
                    split_index = int(len(X) * 0.8) # We split the model into 80% trainig data and 20% testing data
                    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
                    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
                    
                    model = RandomForestRegressor(n_estimators=100, random_state=42) # We implement a regressor which uses 100 different analyses.
                    model.fit(X_train, y_train) # We train the model. It looks at the past volatility patterns (X_train) and the actual volatility of the next day (y_train).
                    
                    preds = model.predict(X_test) # We make our prediction by taking the average of all 100 analyses.
                    mae = mean_absolute_error(y_test, preds) # We compute the mean absolute error.
                    
                    st.markdown(f"#### Volatility Forecast for **{smi_companies.get(ml_ticker, ml_ticker)}** ({horizon_label})") # This renders a sub-header, including the picked stock and the picked time horizon.
                    
                    last_21_days = X.iloc[-1:].values # We grab the last rows of our table, which includes the volatility of the past 21 days.
                    next_val_pred = model.predict(last_21_days)[0] # The next day prediction is the last value.
                    
                    col1, col2 = st.columns(2) # We create two columns to show the predicted volatility and the MAE.
                    col1.metric(f"Predicted Volatility ({horizon_label})", f"{next_val_pred:.2%}")
                    col2.metric("Mean Absolute Error (Test Set)", f"{mae:.2%}")

                    # We create a DataFrame for the line chart.
                    results_df = pd.DataFrame({
                        'Date': y_test.index, 
                        'Actual Volatility': y_test.values, 
                        'Predicted Volatility': preds 
                    }).set_index('Date') 
                    
                    st.write(f"**Predicted vs. Actual Volatility ({horizon_label}):**") # We add this label above the graph.
                    st.line_chart(results_df) 
                    
                    # We add a caption to the model, explaining how to interpret the values.
                    st.caption("The lower the ratio of MAE to volatility, the more accurate our model is.")
                    
                else:
                    st.warning("Not enough data. Try a longer date range.") # If there is not enough data selected, this warning occurs.

except Exception as e:
    st.error(f"An error occurred: {e}") # If there is any problem in the main app logic, this occurs.
