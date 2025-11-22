# -----------------------------------------------------------------------------
# SNIPPET 1: IMPORTS & CONFIGURATION
# -----------------------------------------------------------------------------
# We import the libraries we need.
# streamlit: The framework to build the web app.
# pandas: The standard tool for handling tabular data (DataFrames).
# yfinance: The library that fetches stock data from Yahoo Finance.
# numpy: Needed for math calculations (sqrt, etc.).
# altair: Needed for advanced charts like the Scatter Plot.
# sklearn: Needed for Machine Learning (Logistic Regression).
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np 
import altair as alt 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# This must be the first Streamlit command. It sets up the page title and layout.
st.set_page_config(page_title="Stock Comparator", layout="wide")

# Main title of the application
st.title("📈 Stock & Portfolio Comparator")

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def calculate_metrics(df):
    """
    Takes a DataFrame of Stock Prices and calculates key financial metrics.
    Input: df where columns are tickers and rows are dates.
    Output: DataFrame with metrics (Return, Risk, Sharpe, etc.) for each ticker.
    """
    # 1. Calculate Daily Returns
    # We don't care about the absolute price (e.g., 100 vs 200).
    # We care about the percentage change from yesterday to today.
    # .dropna() removes the first row (which has no "yesterday" to compare to).
    returns = df.pct_change().dropna()
    
    # Create an empty DataFrame to store our results
    summary = pd.DataFrame(index=df.columns)
    
    # 2. Annualized Return
    # We calculate the average daily return and scale it up to a year.
    # There are roughly 252 trading days in a year (excluding weekends/holidays).
    summary['Ann. Return'] = returns.mean() * 252

    # 3. Cumulative Return
    # The total percentage gain/loss over the entire selected period.
    # Formula: (1 + r1) * (1 + r2) * ... - 1
    summary['Cumulative Return'] = (1 + returns).prod() - 1
    
    # 4. Annualized Volatility (Risk)
    # Standard Deviation measures how much the stock jumps around.
    # To annualize volatility, we multiply by the square root of time (sqrt(252)).
    summary['Ann. Volatility'] = returns.std() * np.sqrt(252)
    
    # 5. Sharpe Ratio
    # Measures "Return per unit of Risk".
    # Formula: (Return - RiskFreeRate) / Volatility.
    # We assume RiskFreeRate = 0 for simplicity.
    summary['Sharpe Ratio'] = summary['Ann. Return'] / summary['Ann. Volatility']

    # 6. Sortino Ratio
    # Like Sharpe, but only penalizes "bad" volatility (downside risk).
    # We assume that upside volatility (price jumping up) is good.
    downside_returns = returns.copy()
    downside_returns[downside_returns > 0] = np.nan # Ignore positive returns
    
    # Calculate standard deviation only for negative days
    annual_downside_vol = downside_returns.std() * np.sqrt(252)
    summary['Sortino Ratio'] = summary['Ann. Return'] / annual_downside_vol
    
    # 7. Max Drawdown
    # The "Worst Case Scenario": buying at the peak and selling at the bottom.
    # We calculate the cumulative return (growth of $1).
    cumulative_returns_series = (1 + returns).cumprod()
    
    # running_max tracks the highest price seen SO FAR.
    running_max = cumulative_returns_series.cummax()
    
    # Drawdown is the percentage distance from the running_max.
    drawdown = (cumulative_returns_series / running_max) - 1
    
    # The minimum value (most negative) is the Maximum Drawdown.
    summary['Max Drawdown'] = drawdown.min()

    # 8. Value at Risk (VaR)
    # The 5th percentile of daily returns (95% confidence).
    # Meaning: "95% of the time, your daily loss won't be worse than this."
    summary['Value at Risk (95%)'] = returns.quantile(0.05)
    
    return summary

def prepare_volatility_data(series, window=5):
    """
    Prepares data to predict HIGH VOLATILITY.
    - Input: Price Series
    - Target: 1 if tomorrow's absolute return is in the top 20% of history (High Vol).
    - Features: Past 5 days' absolute returns (volatility clusters).
    """
    df = series.to_frame(name='Close')
    
    # Calculate absolute daily returns (magnitude of move)
    df['Abs_Return'] = df['Close'].pct_change().abs()
    
    # Define "High Volatility" as the top 25% of moves
    # This threshold is dynamic based on the stock's own history
    threshold = df['Abs_Return'].quantile(0.75)
    
    # Target: 1 if tomorrow's move is big, 0 if normal
    df['Target'] = (df['Abs_Return'].shift(-1) > threshold).astype(int)
    
    # Features: Recent volatility (Lag 1 to Lag 5)
    for i in range(1, window + 1):
        df[f'Vol_Lag_{i}'] = df['Abs_Return'].shift(i)
        
    df = df.dropna()
    
    feature_cols = [f'Vol_Lag_{i}' for i in range(1, window + 1)]
    return df[feature_cols], df['Target'], threshold

# -----------------------------------------------------------------------------
# SNIPPET 2: SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
# We use 'with st.sidebar' to place everything inside this block on the left side.
with st.sidebar:
    st.header("🎛️ Controls")

    # 1. TICKER SELECTION
    # Dictionary: Keys = Ticker Symbols (for code), Values = Full Names (for display).
    smi_companies = {
        "^SSMI": "🇨🇭 Swiss Market Index (Benchmark)", # Added Benchmark
        "NESN.SW": "Nestlé",
        "ROG.SW": "Roche",
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

    # We filter the keys to exclude ^SSMI from the dropdown options.
    # This ensures the user cannot manually select/deselect the Benchmark.
    selectable_tickers = [t for t in smi_companies.keys() if t != "^SSMI"]

    # st.multiselect creates the dropdown.
    tickers = st.multiselect(
        "Select Tickers", 
        options=selectable_tickers, 
        format_func=lambda x: f"{smi_companies[x]} ({x})", # Show full names
        default=["NESN.SW", "ROG.SW", "UBSG.SW"] 
    )

    # 2. DATE SELECTION
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    with col2:
        end_date = st.date_input("End Date", value=pd.to_datetime("today"))

    # -------------------------------------------------------------------------
    # SNIPPET 6: PORTFOLIO WEIGHTS (Modified)
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.header("⚖️ Portfolio Builder")
    
    # Dictionary to store the weights user enters
    weights = {}
    
    # Only show weights input if tickers are selected
    if tickers:
        with st.expander("Assign Weights (%)", expanded=True): # Expanded by default
            st.write("Assign percentage weights. Must sum to 100%.")
            
            # Calculate a sensible default (e.g., 33.3 for 3 stocks)
            default_weight = round(100.0 / len(tickers), 1)
            
            for t in tickers:
                # We use the full name for the label
                name = smi_companies[t]
                # Input for Percentage (0-100)
                weights[t] = st.number_input(f"{name} (%)", min_value=0.0, max_value=100.0, value=default_weight, step=1.0)
            
            # Display the current total sum
            current_total = sum(weights.values())
            st.write(f"**Total Allocation:** {current_total:.1f}%")
            
            # Validation Message
            if abs(current_total - 100.0) > 0.1: # Allow tiny float error
                st.error("⚠️ Total must be exactly 100%")
            else:
                st.success("✅ Portfolio Ready")

# -----------------------------------------------------------------------------
# SNIPPET 3: LOADING DATA
# -----------------------------------------------------------------------------
# This function downloads the data. 
# @st.cache_data prevents redownloading on every interaction.
@st.cache_data
def load_data(ticker_list, start, end):
    if not ticker_list:
        return pd.DataFrame()
    
    # yfinance download with auto_adjust=True to handle splits/dividends.
    data = yf.download(ticker_list, start=start, end=end, auto_adjust=True)
    
    # Formatting Fix for Single Ticker vs Multiple Tickers
    if len(ticker_list) == 1:
        return data['Close'].to_frame(name=ticker_list[0])
    
    return data['Close']

# -----------------------------------------------------------------------------
# MAIN APP LOGIC
# -----------------------------------------------------------------------------
try:
    # 1. PREPARE TICKER LIST
    # We take the user's selection AND forcefully add the SMI Index.
    # set() removes duplicates just in case.
    tickers_to_load = list(set(tickers + ["^SSMI"]))

    # 2. CALL THE FUNCTION
    stock_df = load_data(tickers_to_load, start_date, end_date)
    
    # 3. CHECK IF DATA IS EMPTY
    if stock_df.empty:
        st.warning("No data found. Please check your date range or tickers.")
    else:
        st.success(f"Data loaded successfully for {len(tickers_to_load)} tickers (including Benchmark)!")
        
        # Raw Data Preview (Hidden by default)
        with st.expander("📄 View Raw Data Preview"):
             st.dataframe(stock_df.head())

        # -----------------------------------------------------------------------------
        # DATA PRE-PROCESSING & PORTFOLIO CALCULATION (Updated)
        # -----------------------------------------------------------------------------
        # Drop rows with missing values to ensure fair comparison (same start date)
        cleaned_df = stock_df.dropna()
        
        # CHECK if user has valid portfolio configuration
        valid_portfolio = False
        current_total = sum(weights.values())
        
        # Logic: Only calculate portfolio if tickers selected AND weights sum to 100
        if tickers and not cleaned_df.empty and abs(current_total - 100.0) <= 0.1:
            valid_portfolio = True
            
            # 1. Isolate the selected stocks (exclude benchmark)
            selected_stocks = cleaned_df[tickers]
            
            # 2. Calculate Daily Returns for individual stocks
            daily_returns = selected_stocks.pct_change()
            
            # 3. Convert Percentages (50) to Decimals (0.5)
            # We iterate through the list 'tickers' to ensure order matches columns
            final_weights = [weights[t] / 100.0 for t in tickers]
            
            # 4. Calculate "My Portfolio" Returns
            # Dot Product: returns matrix dot weights vector
            portfolio_ret = daily_returns.dot(final_weights)
            
            # 5. Construct "My Portfolio" Price Series (Starting at 100)
            my_portfolio_price = (1 + portfolio_ret).cumprod() * 100
            my_portfolio_price.iloc[0] = 100
            
            # 6. Add to the Main DataFrame
            cleaned_df["💼 My Portfolio"] = my_portfolio_price
            
            # REMOVED: Equal-Weighted Portfolio logic as requested.

        elif tickers and abs(current_total - 100.0) > 0.1:
            st.warning("⚠️ 'My Portfolio' not calculated: Weights do not sum to 100%.")

        # -----------------------------------------------------------------------------
        # SNIPPET 4: DYNAMIC KPI VISUALIZER
        # -----------------------------------------------------------------------------
        st.subheader("📊 KPI Visualizer over Time")
        
        if not cleaned_df.empty:
            # Metric Selection Dropdown
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
            
            # Calculate Daily Returns (needed for most metrics)
            returns = cleaned_df.pct_change().dropna()
            window = 30 # Window size for rolling calculations
            
            # Logic for each Metric
            if selected_metric == "Cumulative Return (Indexed to 100)":
                # Use the data as is (since we already have prices/indices)
                # We re-normalize just to be safe
                plot_data = cleaned_df / cleaned_df.iloc[0] * 100
                
            elif selected_metric == "Annualized Return (30-Day Rolling)":
                # Rolling Average Return * 252
                plot_data = returns.rolling(window=window).mean() * 252
            
            elif selected_metric == "Volatility (30-Day Rolling)":
                # Rolling Std Dev * sqrt(252)
                plot_data = returns.rolling(window=window).std() * np.sqrt(252)
                
            elif selected_metric == "Sharpe Ratio (30-Day Rolling)":
                # Rolling Mean / Rolling Std over 30 days
                rolling_return = returns.rolling(window=window).mean() * 252
                rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
                plot_data = rolling_return / rolling_vol
            
            elif selected_metric == "Sortino Ratio (30-Day Rolling)":
                # Rolling Return / Rolling Downside Volatility
                # 1. Isolate negative returns (keep positives as NaN)
                downside = returns.copy()
                downside[downside > 0] = np.nan
                # 2. Calculate Rolling Std of these negative returns
                # (Pandas rolling.std() ignores NaNs by default, which is what we want)
                rolling_downside_vol = downside.rolling(window=window).std() * np.sqrt(252)
                rolling_return = returns.rolling(window=window).mean() * 252
                plot_data = rolling_return / rolling_downside_vol
                
            elif selected_metric == "Drawdown (Historical)":
                # Historical Drawdown from Running Max
                cumulative_rets = (1 + returns).cumprod()
                running_max = cumulative_rets.cummax()
                plot_data = (cumulative_rets / running_max) - 1
                
            elif selected_metric == "Value at Risk 95% (30-Day Rolling)":
                # Rolling 5th Percentile (0.05 quantile)
                plot_data = returns.rolling(window=window).quantile(0.05)

            # Rename columns and plot
            # We use the dictionary for tickers, but keep "My Portfolio" as is
            # The .get(x, x) method returns the name if found, otherwise keeps the original text
            plot_data = plot_data.rename(columns=lambda x: smi_companies.get(x, x))
            
            st.line_chart(plot_data)
            
        else:
            st.info("Not enough shared data points to plot a comparison. Try adjusting dates.")

        # -----------------------------------------------------------------------------
        # SNIPPET 5: CALCULATE RISK & RETURN METRICS
        # -----------------------------------------------------------------------------
        st.subheader("📉 Risk & Return Analysis")
        
        # 1. Call helper function
        metrics_df = calculate_metrics(cleaned_df)
        
        # 2. Rename the Index (Rows) from Tickers to Full Names
        metrics_df = metrics_df.rename(index=lambda x: smi_companies.get(x, x))

        # 3. SCATTER PLOT CONFIGURATION
        # Prepare data for Altair: Reset index so 'Company' is a column, not an index.
        # Explicitly name the index 'Company' to avoid "None" or "Ticker" errors.
        metrics_df.index.name = "Company"
        scatter_data = metrics_df.reset_index()
        
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
            color='Company',
            # Tooltip will show all metrics for easy inspection
            tooltip=['Company'] + chart_opts
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
        # SNIPPET 7: MACHINE LEARNING (Updated for Volatility)
        # -----------------------------------------------------------------------------
        st.markdown("---")
        st.header("🤖 Machine Learning: Volatility Prediction")
        
        st.write("""
        This model predicts whether **Tomorrow's Volatility** will be **HIGH** (Top 25% of movements) or **NORMAL**.
        It uses the past 5 days of volatility to learn patterns (Volatility Clustering).
        """)
        
        # 1. User Selects a Stock
        ml_opts = [t for t in tickers if t in cleaned_df.columns]
        ml_ticker = st.selectbox("Select Stock to Predict", ml_opts, format_func=lambda x: smi_companies.get(x, x))
        
        if ml_ticker:
            # 2. Prepare Data
            subset_series = cleaned_df[ml_ticker]
            
            # Use our new helper function for Volatility Classification
            X, y, threshold = prepare_volatility_data(subset_series)
            
            if len(X) > 50:
                # 3. Split Data
                split_index = int(len(X) * 0.8)
                X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
                y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
                
                # 4. Train Model (Random Forest is often better for this than LogReg)
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # 5. Evaluate
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                
                # Display Results
                st.markdown(f"#### Volatility Forecast for **{smi_companies.get(ml_ticker, ml_ticker)}**")
                st.write(f"**High Volatility Threshold:** > {threshold:.2%} move per day")
                st.metric("Model Accuracy", f"{acc:.2%}")
                
                # Confusion Matrix
                st.write("**Confusion Matrix:**")
                cm = confusion_matrix(y_test, preds)
                cm_df = pd.DataFrame(cm, 
                                     index=['Actual Normal', 'Actual High Vol'], 
                                     columns=['Pred Normal', 'Pred High Vol'])
                st.dataframe(cm_df)
                
                st.caption("Volatility often 'clusters' (big moves follow big moves). This model tries to exploit that pattern.")
                
            else:
                st.warning("Not enough data. Try a longer date range.")

except Exception as e:
    # st.error shows a red error box if something crashes
    st.error(f"An error occurred: {e}")
