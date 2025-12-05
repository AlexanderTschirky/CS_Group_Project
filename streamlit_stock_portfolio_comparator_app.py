# -----------------------------------------------------------------------------
# IMPORTS & CONFIGURATION
# -----------------------------------------------------------------------------

# This application was built using Google Gemini. We created the base code in cooperation with Google Gemini, modified and commented the code ourselves.

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
    # Added a small Risk Free Rate assumption (1%) for more realism
    rf_rate = 0.01 
    summary['Sharpe Ratio'] = (summary['Ann. Return'] - rf_rate) / summary['Ann. Volatility'] 

    # 6. Sortino Ratio
    downside_returns = returns.copy() # We copy the returns into the new variable "downside_returns" to further process the data.
    downside_returns[downside_returns > 0] = np.nan # For the Sortino Ratio, we neglect upside volatility, therefore we do not consider positive returns for the calculation.
    annual_downside_vol = downside_returns.std() * np.sqrt(252) # We calculate the annual volatility only for negative days.
    summary['Sortino Ratio'] = (summary['Ann. Return'] - rf_rate) / annual_downside_vol # We calculate the Sortino Ratio.
    
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
    """
    For the Machine Learning part, we want to predict next-day absolute return (volatility) of a stock.
    We use the absolute returns of the last trading month (21 days) for this.
    
    Added 'horizon' parameter:
    - horizon=1: Predict next day's volatility.
    - horizon=5: Predict average volatility over the next 5 days (1 Week).
    - horizon=21: Predict average volatility over the next 21 days (1 Month).
    """
    # Robust check: If input is already a DataFrame, use it directly. If Series, convert.
    if isinstance(series, pd.DataFrame):
        df = series.copy()
        if len(df.columns) > 0:
            df.columns = ['Close']
    else:
        df = series.to_frame(name='Close') 
        
    df['Abs_Return'] = df['Close'].pct_change().abs() # We calculate the absolute daily returns
    
    # TARGET CREATION:
    # If horizon is 1, we just shift by -1 (tomorrow).
    # If horizon is > 1, we calculate the rolling average of the FUTURE and shift it back.
    if horizon == 1:
        df['Target'] = df['Abs_Return'].shift(-1)
    else:
        # Calculate rolling mean of the NEXT 'horizon' days.
        # .rolling() looks backwards, so we calculate it first, then shift backwards by 'horizon'.
        # This aligns "today's row" with the "average of the next N days".
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=horizon)
        df['Target'] = df['Abs_Return'].rolling(window=indexer).mean()

    # Features: Recent volatility (Lag 1 to Lag 21)
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
        default=["NESN.SW", "NOVN.SW", "UBSG.SW"]  # By default, the stocks of Nestle, Roche and UBS Group will be selected.
    )

    # 2. DATE SELECTION 
    col1, col2 = st.columns(2) # This builds two columns for the start and end date
    with col1:
        start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01")) # The default Start Date is 2020-01-01
    with col2:
        end_date = st.date_input("End Date", value=pd.to_datetime("today")) # The default End Date is today's date

    # 3. PORTFOLIO BUILDER (With Auto-Balancing)
    st.markdown("---")
    st.header("⚖️ Portfolio Builder")
    
    # Initialize session state for weights if not present or if tickers changed
    if 'portfolio_weights' not in st.session_state or 'last_tickers' not in st.session_state or st.session_state.last_tickers != tickers:
        st.session_state.last_tickers = tickers
        # Initialize with equal weights
        if tickers:
            equal_weight = 100.0 / len(tickers)
            st.session_state.portfolio_weights = {t: equal_weight for t in tickers}
        else:
            st.session_state.portfolio_weights = {}

    def balance_weights(changed_ticker):
        """
        Callback function to auto-balance weights.
        When one stock changes, the difference is distributed equally among the others.
        """
        if len(tickers) <= 1:
            return # Cannot balance if only 1 or 0 stocks
            
        current_val = st.session_state[f"weight_{changed_ticker}"]
        old_val = st.session_state.portfolio_weights[changed_ticker]
        diff = current_val - old_val
        
        # We need to subtract this difference from the other stocks
        other_tickers = [t for t in tickers if t != changed_ticker]
        adjustment = diff / len(other_tickers)
        
        # Apply adjustment
        for t in other_tickers:
            new_w = st.session_state.portfolio_weights[t] - adjustment
            # Update the session state widget key directly
            st.session_state[f"weight_{t}"] = max(0.0, new_w) # Prevent negative weights
            
        # Update our master dictionary
        for t in tickers:
            st.session_state.portfolio_weights[t] = st.session_state[f"weight_{t}"]

    weights = {} 
    
    # Only show weights input if tickers are selected
    if tickers:
        with st.expander("Assign Weights (%)", expanded=True): 
            st.write("Adjust one weight, and the others will automatically balance to 100%.") 
            
            for t in tickers:
                name = smi_companies[t] 
                
                # We use the session state to store values. 
                # on_change calls our balancing function immediately when a number is typed.
                st.number_input(
                    f"{name} (%)", 
                    min_value=0.0, 
                    max_value=100.0, 
                    value=st.session_state.portfolio_weights.get(t, 0.0), 
                    step=1.0,
                    key=f"weight_{t}",
                    on_change=balance_weights,
                    args=(t,)
                )
                weights[t] = st.session_state.portfolio_weights.get(t, 0.0)

            current_total = sum(weights.values())
            st.write(f"**Total Allocation:** {current_total:.1f}%") 
            
            # Simple check for the visual confirmation, though math should enforce 100% mostly
            if abs(current_total - 100.0) < 0.1:
                st.success("✅ Portfolio Ready") 
            else:
                st.info("ℹ️ Weights are balancing...")

# -----------------------------------------------------------------------------
# LOADING DATA
# -----------------------------------------------------------------------------
# This function downloads the data. 
@st.cache_data # This is a decorator provided by the Streamlit library, it saves the data it loaded once in the memory. This saves time as the data is not loaded again.
def load_data(ticker_list, start, end): # We define a function to download the data selected at the sidebar.
    if not ticker_list:
        return pd.DataFrame() # If the list is empty, nothing is downloaded
    
    # CRITICAL CHANGE: We always download from a "safe start" (2 years ago) to ensure ML works.
    # We will filter for the user's dates later for display purposes.
    safe_start = pd.Timestamp.today() - pd.DateOffset(years=2)
    # Use the earlier of the two dates (User's start or Safe start)
    download_start = min(start, safe_start)
    
    data = yf.download(ticker_list, start=download_start, end=end, auto_adjust=True) # We download the data from yfinance. We use auto_adjust=True tu handle splits/dividends.
    
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
    # Note: load_data now handles fetching extended history internally
    full_history_df = load_data(tickers_to_load, pd.Timestamp(start_date), pd.Timestamp(end_date)) 
    
    # 3. CHECK IF DATA IS EMPTY
    if full_history_df.empty:
        st.warning("No data found. Please check your date range.") # If the DataFrame is completely empty, a warning is shown. This should never be the case as the SMI is always loaded.
    else:
        if not tickers:
            st.info("Showing Benchmark (SMI) only. Select stocks in the sidebar to compare.") # If only the SMI is loaded, it shows this message.
        else:
            st.success(f"Data loaded successfully for {len(tickers_to_load)} tickers (including Benchmark)!") # If any stocks have been successfully selected, this message shows up.
        
        # -----------------------------------------------------------------------------
        # DATA PRE-PROCESSING & PORTFOLIO CALCULATION (ON FULL HISTORY)
        # -----------------------------------------------------------------------------
        # We calculate the portfolio on the FULL history first, so ML has enough data.
        
        # Drop rows with missing data
        cleaned_df = full_history_df.dropna()
        
        valid_portfolio = False 
        current_total = sum(weights.values())
        
        # Adjusted tolerance slightly for auto-balancing float math
        if tickers and not cleaned_df.empty and abs(current_total - 100.0) <= 0.5: 
            valid_portfolio = True 
    
            selected_tickers = cleaned_df[tickers]
            daily_returns = selected_tickers.pct_change()
            final_weights = [weights[t] / 100.0 for t in tickers] 
            
            portfolio_ret = daily_returns.dot(final_weights) 
            
            # Construct Price Series (Start at 100 on the very first day of history)
            my_portfolio_price = (1 + portfolio_ret).cumprod() * 100 
            my_portfolio_price.iloc[0] = 100 
            
            cleaned_df["💼 My Portfolio"] = my_portfolio_price 

        elif tickers and abs(current_total - 100.0) > 0.5:
            st.warning("⚠️ 'My Portfolio' not calculated: Weights do not sum to 100%.")

        # -----------------------------------------------------------------------------
        # CREATE DISPLAY DATAFRAME (FILTERED BY USER DATE)
        # -----------------------------------------------------------------------------
        # Now we slice the data to show ONLY what the user asked for in the charts/tables
        display_start = pd.Timestamp(start_date).tz_localize(None)
        display_end = pd.Timestamp(end_date).tz_localize(None)
        display_df = cleaned_df.loc[display_start:display_end]

        # Raw Data Preview (Using Display DF)
        with st.expander("📄 View Last 21 Trading Days"): 
             preview_df = display_df.rename(columns=lambda x: smi_companies.get(x, x)) 
             st.dataframe(preview_df.tail(21))
             
        csv_data = preview_df.to_csv().encode('utf-8') 
             
        st.download_button( 
            label="⬇️ Download Raw Price Data (CSV)", 
            data=csv_data, 
            file_name="stock_price_data.csv",
            mime="text/csv",
            help="Click to download the full historical price data as a CSV file."
        )

        # -----------------------------------------------------------------------------
        # DYNAMIC KPI VISUALIZER (USING DISPLAY DF)
        # -----------------------------------------------------------------------------
        st.subheader("📊 KPI Visualizer over Time")
        
        if not display_df.empty:
            metric_options = [ 
                "Cumulative Return (Indexed to 100)",
                "Annualized Return (30-Day Rolling)",
                "Volatility (30-Day Rolling)",
                "Sharpe Ratio (30-Day Rolling)",
                "Sortino Ratio (30-Day Rolling)",
                "Drawdown (Historical)",
                "Value at Risk 95% (30-Day Rolling)"
            ]
            
            selected_metric = st.selectbox("Select Metric to Plot", metric_options)
            
            # Use display_df for visual analysis
            returns = display_df.pct_change().dropna() 
            window = 30 
            
            if selected_metric == "Cumulative Return (Indexed to 100)":
                # Re-index to 100 for the start of the visible period
                plot_data = display_df / display_df.iloc[0] * 100
                
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


            plot_data = plot_data.rename(columns=lambda x: smi_companies.get(x, x)) 
            st.line_chart(plot_data) 
            
        else:
            st.info("Not enough shared data points to plot a comparison. Try adjusting dates.") 

        # -----------------------------------------------------------------------------
        # CALCULATE RISK & RETURN METRICS (USING DISPLAY DF)
        # -----------------------------------------------------------------------------
        st.subheader("📉 Risk & Return Analysis")
        
        metrics_df = calculate_KPI(display_df) 
        
        metrics_df = metrics_df.rename(index=lambda x: smi_companies.get(x, x)) 

        metrics_df.index.name = "Stock" 
        scatter_data = metrics_df.reset_index() 
        
        col_mapping = {
            'Ann. Return': 'Annualized Return',
            'Cumulative Return': 'Cumulative Return',
            'Ann. Volatility': 'Annualized Volatility',
            'Sharpe Ratio': 'Sharpe Ratio',
            'Sortino Ratio': 'Sortino Ratio',
            'Max Drawdown': 'Max Drawdown',
            'Value at Risk (95%)': 'Value at Risk 95%'
        }
        
        scatter_data = scatter_data.rename(columns=col_mapping) 

        st.markdown("##### Compare Metrics (Scatter Plot)") 
        col_x, col_y = st.columns(2) 
        
        chart_opts = list(col_mapping.values()) 
        
        with col_x:
            x_axis = st.selectbox("X-Axis", chart_opts, index=chart_opts.index('Annualized Volatility'))
        with col_y:
            y_axis = st.selectbox("Y-Axis", chart_opts, index=chart_opts.index('Annualized Return'))
            
        x_format = ".2f" if "Ratio" in x_axis else "%"
        y_format = ".2f" if "Ratio" in y_axis else "%"
        
        chart = alt.Chart(scatter_data).mark_circle(size=100).encode(
            x=alt.X(x_axis, title=x_axis, axis=alt.Axis(format=x_format)),
            y=alt.Y(y_axis, title=y_axis, axis=alt.Axis(format=y_format)),
            color='Stock',
            tooltip=['Stock'] + chart_opts
        ).interactive() 
        
        st.altair_chart(chart, use_container_width=True) 
        
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
        # CORRELATION MATRIX (NEW SECTION)
        # -----------------------------------------------------------------------------
        st.markdown("---")
        st.subheader("🔗 Correlation Matrix")
        st.write("This heatmap shows how stocks move together. +1 means they move perfectly in sync (blue), -1 means they move in opposite directions (red).")

        # Get returns again specifically for the Correlation Matrix
        if not display_df.empty:
            corr_returns = display_df.pct_change().dropna()
            
            # Calculate Correlation
            corr_matrix = corr_returns.corr()
            
            # Only show if we have more than 1 stock to compare
            if len(corr_matrix.columns) > 1:
                # 1. Rename tickers to Company Names
                corr_matrix_renamed = corr_matrix.rename(index=lambda x: smi_companies.get(x, x), columns=lambda x: smi_companies.get(x, x))
                
                # 2. Reset index to move the rows into a column called "index" (or whatever the index name was)
                corr_data = corr_matrix_renamed.reset_index()
                
                # 3. Rename that new column to 'Stock A' explicitly to avoid "index" not found errors
                # The column at index 0 is always the one we just reset.
                corr_data = corr_data.rename(columns={corr_data.columns[0]: 'Stock A'})
                
                # 4. Melt: Turn the wide table into a long table
                # id_vars='Stock A' keeps that column fixed, while the other columns (Stock Bs) get melted
                corr_data = corr_data.melt(id_vars='Stock A')
                
                # 5. Rename columns for clarity
                corr_data.columns = ['Stock A', 'Stock B', 'Correlation']

                # Create Heatmap
                heatmap = alt.Chart(corr_data).mark_rect().encode(
                    x=alt.X('Stock A', title=None),
                    y=alt.Y('Stock B', title=None),
                    color=alt.Color('Correlation', scale=alt.Scale(domain=[-1, 1], scheme='redblue')),
                    tooltip=['Stock A', 'Stock B', alt.Tooltip('Correlation', format='.2f')]
                ).properties(
                    height=500
                )

                # Add text labels (numbers inside boxes)
                text = heatmap.mark_text(baseline='middle').encode(
                    text=alt.Text('Correlation', format='.2f'),
                    color=alt.condition(
                        (alt.datum.Correlation > 0.5) | (alt.datum.Correlation < -0.5),
                        alt.value('white'),
                        alt.value('black')
                    )
                )

                st.altair_chart(heatmap + text, use_container_width=True)
            else:
                st.info("Select at least 2 stocks (or 1 stock + Benchmark) to view correlations.")

        # -----------------------------------------------------------------------------
        # MACHINE LEARNING (USING FULL HISTORY)
        # -----------------------------------------------------------------------------
        st.markdown("---") 
        st.header("🤖 Machine Learning: Volatility Prediction") 
        
        st.write("""
        This model predicts the **Exact Volatility** (Average Absolute Daily Return) over a specific time horizon.
        It uses the past 21 days of volatility to learn patterns using a Random Forest Regressor.
        """) 
        
        # User selects stock (options based on cleaned_df which has portfolio)
        ml_opts = list(cleaned_df.columns) 
        col_ml_1, col_ml_2 = st.columns(2)
        
        with col_ml_1:
            ml_ticker = st.selectbox("Select Stock to Predict", ml_opts, format_func=lambda x: smi_companies.get(x, x)) 
        
        with col_ml_2:
            # Dropdown for Forecast Horizon
            horizon_dict = {"Next Day": 1, "Next Week (5 Days)": 5, "Next Month (21 Days)": 21}
            horizon_label = st.selectbox("Select Forecast Horizon", list(horizon_dict.keys()))
            horizon_val = horizon_dict[horizon_label]

        if ml_ticker:
            # IMPORTANT: We use 'cleaned_df' here, which has the FULL 2-year history
            # This ensures ML always works even if 'display_df' is short.
            subset_series = cleaned_df[ml_ticker].dropna()
            
            # Pass the selected horizon to the helper function
            X, y = prepare_regression_data(subset_series, window=21, horizon=horizon_val)
            
            if len(X) > 50: 
                split_index = int(len(X) * 0.8) 
                X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
                y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
                
                model = RandomForestRegressor(n_estimators=100, random_state=42) 
                model.fit(X_train, y_train) 
                
                preds = model.predict(X_test) 
                mae = mean_absolute_error(y_test, preds) 
                
                st.markdown(f"#### Volatility Forecast for **{smi_companies.get(ml_ticker, ml_ticker)}** ({horizon_label})") 
                
                # Predict Next Period
                last_21_days = X.iloc[-1:].values 
                next_val_pred = model.predict(last_21_days)[0] 
                
                col1, col2 = st.columns(2) 
                col1.metric(f"Predicted Volatility ({horizon_label})", f"{next_val_pred:.2%}")
                col2.metric("Mean Absolute Error (Test Set)", f"{mae:.2%}")

                # Chart for ML (Test Set Only)
                results_df = pd.DataFrame({
                    'Date': y_test.index, 
                    'Actual Volatility': y_test.values, 
                    'Predicted Volatility': preds 
                }).set_index('Date') 
                
                st.write(f"**Predicted vs. Actual Volatility ({horizon_label}):**") 
                st.line_chart(results_df) 
                
                st.caption("The lower the ratio of MAE to Volatility, the more accurate our model is. If the lines overlap, the model is doing a good job.")
                
            else:
                st.warning("Not enough data. Try a longer date range.") 

except Exception as e:
    st.error(f"An error occurred: {e}")
