# -----------------------------------------------------------------------------
# SNIPPET 1: IMPORTS & CONFIGURATION
# -----------------------------------------------------------------------------
# We import the libraries we need.
# streamlit: The framework to build the web app.
# pandas: The standard tool for handling tabular data (DataFrames).
# yfinance: The library that fetches stock data from Yahoo Finance.
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np # Needed for math calculations (sqrt, etc.)
import altair as alt # Needed for the advanced Risk/Return scatter plot

# This must be the first Streamlit command. It sets up the page title and layout.
st.set_page_config(page_title="Stock Comparator", layout="wide")

# Main title of the application
st.title("📈 Stock & Portfolio Comparator")

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS (New Section)
# -----------------------------------------------------------------------------
def calculate_metrics(df):
    """
    Takes a DataFrame of Stock Prices and calculates:
    - Annualized Return
    - Cumulative Return
    - Annualized Volatility
    - Sharpe Ratio
    - Sortino Ratio
    - Max Drawdown
    - Value at Risk (VaR)
    """
    # 1. Calculate Daily Returns (Percentage change from previous day)
    returns = df.pct_change().dropna()
    
    # 2. Calculate Metrics
    # We assume 252 trading days in a year for annualization
    summary = pd.DataFrame(index=df.columns)
    
    # Annualized Return: Average daily return * 252 days
    summary['Ann. Return'] = returns.mean() * 252

    # Cumulative Return: Total return over the entire period
    # Formula: (1 + r1) * (1 + r2) ... - 1
    summary['Cumulative Return'] = (1 + returns).prod() - 1
    
    # Annualized Volatility: Standard deviation of daily returns * Square root of 252
    summary['Ann. Volatility'] = returns.std() * np.sqrt(252)
    
    # Sharpe Ratio: Return / Volatility (assuming 0% Risk Free Rate for simplicity)
    summary['Sharpe Ratio'] = summary['Ann. Return'] / summary['Ann. Volatility']

    # Sortino Ratio: Return / Downside Volatility
    # Downside Volatility only looks at days where the stock lost money.
    downside_returns = returns.copy()
    downside_returns[downside_returns > 0] = np.nan # Ignore positive returns
    annual_downside_vol = downside_returns.std() * np.sqrt(252)
    summary['Sortino Ratio'] = summary['Ann. Return'] / annual_downside_vol
    
    # Max Drawdown: The worst drop from a peak
    # We calculate the running maximum, then how far the current price is below that peak
    cumulative_returns_series = (1 + returns).cumprod()
    running_max = cumulative_returns_series.cummax()
    drawdown = (cumulative_returns_series / running_max) - 1
    summary['Max Drawdown'] = drawdown.min()

    # Value at Risk (VaR) at 95% Confidence
    # The 5th percentile of daily returns.
    # Meaning: "95% of the time, the daily loss will not be worse than this number."
    summary['Value at Risk (95%)'] = returns.quantile(0.05)
    
    return summary

# -----------------------------------------------------------------------------
# SNIPPET 2: SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
# We use 'with st.sidebar' to place everything inside this block on the left side.
# This keeps the main area clean for data and charts.
with st.sidebar:
    st.header("🎛️ Controls")

    # 1. TICKER SELECTION
    # We define a dictionary. 
    # Keys (left) are the ticker symbols yfinance needs.
    # Values (right) are the readable names we want to show the user.
    smi_companies = {
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

    # st.multiselect creates a dropdown where users can pick multiple items.
    # options: We pass the Keys of our dictionary (the tickers).
    # format_func: This function tells Streamlit: "When you show the option 'NESN.SW',
    #              instead show 'Nestlé (NESN.SW)' found in our dictionary."
    tickers = st.multiselect(
        "Select Tickers", 
        options=smi_companies.keys(), 
        format_func=lambda x: f"{smi_companies[x]} ({x})",
        default=["NESN.SW", "ROG.SW", "UBSG.SW"] # These are selected by default on load
    )

    # 2. DATE SELECTION
    # We create two columns inside the sidebar to put Start and End dates side-by-side.
    col1, col2 = st.columns(2)
    with col1:
        # Default start date is set to Jan 1, 2020
        start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    with col2:
        # Default end date is Today
        end_date = st.date_input("End Date", value=pd.to_datetime("today"))

# -----------------------------------------------------------------------------
# SNIPPET 3: LOADING DATA
# -----------------------------------------------------------------------------
# This function downloads the data. 
# @st.cache_data is a "decorator". It tells Streamlit: "If I have downloaded this
# specific list of tickers for these specific dates before, don't download it again.
# Just grab the result from memory." This makes the app much faster.
@st.cache_data
def load_data(ticker_list, start, end):
    # Safety check: If the user deleted all tickers, return an empty table.
    if not ticker_list:
        return pd.DataFrame()
    
    # yfinance allows downloading multiple tickers at once.
    # auto_adjust=True: This adjusts prices for stock splits and dividends.
    # This is crucial for long-term comparison (otherwise a split looks like a crash).
    data = yf.download(ticker_list, start=start, end=end, auto_adjust=True)
    
    # Formatting Fix:
    # If we select only 1 ticker, yfinance returns a DataFrame with columns like [Open, Close...].
    # If we select 2+ tickers, it returns a Multi-Index DataFrame.
    # We want to standardize this so we always get just the 'Close' prices.
    
    if len(ticker_list) == 1:
        # Keep only the Close column and rename it to the ticker name for consistency
        return data['Close'].to_frame(name=ticker_list[0])
    
    # If multiple tickers, we just want the 'Close' price level.
    return data['Close']

# -----------------------------------------------------------------------------
# MAIN APP LOGIC
# -----------------------------------------------------------------------------
# We use a 'try-except' block to handle errors gracefully.
# If the internet is down or Yahoo is blocking us, the app won't crash; it will show an error message.
try:
    # 1. CALL THE FUNCTION
    stock_df = load_data(tickers, start_date, end_date)
    
    # 2. CHECK IF DATA IS EMPTY
    if stock_df.empty:
        # st.warning shows a yellow warning box
        st.warning("No data found. Please check your date range or tickers.")
    else:
        # st.success shows a green success bar
        st.success(f"Data loaded successfully for {len(tickers)} tickers!")
        
        # 3. RAW DATA PREVIEW
        # st.expander creates a collapsible box. Good for hiding details like raw data tables.
        with st.expander("📄 View Raw Data Preview"):
             st.dataframe(stock_df.head())

        # -----------------------------------------------------------------------------
        # SNIPPET 4: DATA VISUALIZATION
        # -----------------------------------------------------------------------------
        st.subheader("📊 Relative Performance (Indexed to 100)")
        
        # PROBLEM: Raw prices are hard to compare (Nestlé is ~80 CHF, Givaudan is ~3000 CHF).
        # SOLUTION: Normalize them so they all start at 100.
        # Formula: (Price_Today / Price_Day_1) * 100
        
        # 1. Drop rows with missing values (NaN).
        #    This ensures all lines start at the same date (e.g. if one company IPO'd later).
        cleaned_df = stock_df.dropna()
        
        if not cleaned_df.empty:
            # 2. RENAME COLUMNS FOR DISPLAY
            # We swap the Ticker (NESN.SW) for the Name (Nestlé) using our dictionary.
            # This makes the chart legend readable.
            display_df = cleaned_df.rename(columns=smi_companies)
            
            # 3. Normalize
            # .iloc[0] selects the very first row (Day 1).
            # Dividing the whole DataFrame by this row indexes everything to 1.
            # Multiplying by 100 indexes everything to 100.
            normalized_df = display_df / display_df.iloc[0] * 100
            
            # 4. Plot
            # st.line_chart is a built-in wrapper for Altair charts. It's interactive by default.
            st.line_chart(normalized_df)
        else:
            st.info("Not enough shared data points to plot a comparison. Try adjusting dates.")

        # -----------------------------------------------------------------------------
        # SNIPPET 5: CALCULATE RISK & RETURN METRICS
        # -----------------------------------------------------------------------------
        st.subheader("📉 Risk & Return Metrics")
        
        # 1. Call our helper function
        metrics_df = calculate_metrics(cleaned_df)
        
        # 2. Rename the Index (Rows) to Full Company Names
        metrics_df = metrics_df.rename(index=smi_companies)

        # 3. NEW: SCATTER PLOT (Risk vs Return)
        # We need to transform the data slightly for the scatter plot.
        # FIX 1: Explicitly name the index 'Company'
        metrics_df.index.name = "Company"
        scatter_data = metrics_df.reset_index()
        
        # FIX 2: Rename columns to remove dots (.) because dots break Altair/Vega-Lite parsing
        scatter_data = scatter_data.rename(columns={
            "Ann. Return": "Return", 
            "Ann. Volatility": "Volatility",
            "Sharpe Ratio": "Sharpe"
        })
        
        # Create an Altair Chart using the CLEAN column names
        # X-Axis: Volatility (Risk)
        # Y-Axis: Return (Reward)
        # Color: Company
        chart = alt.Chart(scatter_data).mark_circle(size=100).encode(
            x=alt.X('Volatility', title='Risk (Annualized Volatility)', axis=alt.Axis(format='%')),
            y=alt.Y('Return', title='Return (Annualized)', axis=alt.Axis(format='%')),
            color='Company',
            tooltip=['Company', 
                     alt.Tooltip('Return', format='.2%'), 
                     alt.Tooltip('Volatility', format='.2%'), 
                     alt.Tooltip('Sharpe', format='.2f')]
        ).interactive() 
        
        st.altair_chart(chart, use_container_width=True)
        
        # 4. Format the numbers for the Table display
        # We want percentages (e.g., 0.12 -> "12.00%") and 2 decimal places.
        # .style.format() is a pandas trick to make tables look pretty without changing the data.
        formatted_metrics = metrics_df.style.format({
            'Ann. Return': '{:.2%}',
            'Cumulative Return': '{:.2%}',
            'Ann. Volatility': '{:.2%}',
            'Sharpe Ratio': '{:.2f}',
            'Sortino Ratio': '{:.2f}',
            'Max Drawdown': '{:.2%}',
            'Value at Risk (95%)': '{:.2%}'
        })
        
        # 5. Display the table
        st.dataframe(formatted_metrics)

except Exception as e:
    # st.error shows a red error box if something crashes
    st.error(f"An error occurred: {e}")
