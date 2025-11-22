# -----------------------------------------------------------------------------
# SNIPPET 1: IMPORTS & CONFIGURATION
# -----------------------------------------------------------------------------
# We import the libraries we need.
# streamlit: The framework to build the web app.
# pandas: The standard tool for handling tabular data (DataFrames).
# yfinance: The library that fetches stock data from Yahoo Finance.
# numpy: Needed for math calculations (sqrt, etc.).
# altair: Needed for advanced charts like the Scatter Plot.
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np 
import altair as alt 

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
        # SNIPPET 4: DYNAMIC KPI VISUALIZER
        # -----------------------------------------------------------------------------
        st.subheader("📊 KPI Visualizer over Time")
        
        # Drop rows with missing values to ensure fair comparison (same start date)
        cleaned_df = stock_df.dropna()
        
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

            # Rename columns and plot
            plot_data = plot_data.rename(columns=smi_companies)
            st.line_chart(plot_data)
            
        else:
            st.info("Not enough shared data points to plot a comparison. Try adjusting dates.")

        # -----------------------------------------------------------------------------
        # SNIPPET 5: CALCULATE RISK & RETURN METRICS
        # -----------------------------------------------------------------------------
        st.subheader("📉 Risk & Return Analysis")
        
        # 1. Call helper function
        metrics_df = calculate_metrics(cleaned_df)
        metrics_df = metrics_df.rename(index=smi_companies)

        # 2. SCATTER PLOT CONFIGURATION
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

except Exception as e:
    st.error(f"An error occurred: {e}")
