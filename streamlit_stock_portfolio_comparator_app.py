# -----------------------------------------------------------------------------
# IMPORTS & CONFIGURATION
# -----------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import altair as alt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Page Config: Sets up the browser tab title and wide layout
st.set_page_config(page_title="SMI Pro Comparator", layout="wide", page_icon="📈")

# -----------------------------------------------------------------------------
# GLOBAL CONSTANTS
# -----------------------------------------------------------------------------
SMI_COMPANIES = {
    "^SSMI": "🇨🇭 SMI Index (Benchmark)",
    "ROG.SW": "Roche",
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

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def calculate_KPI(df, risk_free_rate=0.0):
    """
    Calculates detailed financial KPIs for the summary table.
    Now includes 'risk_free_rate' for a more accurate Sharpe Ratio calculation.
    """
    summary = pd.DataFrame(index=df.columns)
    
    # 1. Daily Returns
    returns = df.pct_change().dropna()
    
    # 2. Annualized Return (assuming 252 trading days)
    summary['Ann. Return'] = returns.mean() * 252

    # 3. Cumulative Return
    summary['Cumulative Return'] = (1 + returns).prod() - 1
    
    # 4. Annualized Volatility
    summary['Ann. Volatility'] = returns.std() * np.sqrt(252)
    
    # 5. Sharpe Ratio (Return - RiskFree) / Volatility
    summary['Sharpe Ratio'] = (summary['Ann. Return'] - risk_free_rate) / summary['Ann. Volatility']

    # 6. Sortino Ratio (Downside Deviation)
    # We replace positive returns with 0 to calculate downside risk only
    downside_returns = returns.copy()
    downside_returns[downside_returns > 0] = 0 
    annual_downside_vol = downside_returns.std() * np.sqrt(252)
    
    # Avoid division by zero
    summary['Sortino Ratio'] = np.where(
        annual_downside_vol == 0, 
        np.nan, 
        (summary['Ann. Return'] - risk_free_rate) / annual_downside_vol
    )
    
    # 7. Max Drawdown
    cumulative_returns_series = (1 + returns).cumprod()
    running_max = cumulative_returns_series.cummax()
    drawdown = (cumulative_returns_series / running_max) - 1
    summary['Max Drawdown'] = drawdown.min()

    # 8. Value at Risk (95%)
    summary['VaR (95%)'] = returns.quantile(0.05)
    
    return summary

def prepare_regression_data(series, window=21, horizon=1):
    """
    Prepares data for the Machine Learning Volatility prediction.
    """
    if isinstance(series, pd.DataFrame):
        df = series.copy()
        df.columns = ['Close']
    else:
        df = series.to_frame(name='Close')
        
    df['Abs_Return'] = df['Close'].pct_change().abs()
    
    # Target Creation: Predicting future volatility
    if horizon == 1:
        df['Target'] = df['Abs_Return'].shift(-1)
    else:
        # Calculate rolling mean of the NEXT 'horizon' days
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=horizon)
        df['Target'] = df['Abs_Return'].rolling(window=indexer).mean()

    # Feature Engineering: Creating Lags (Past Volatility)
    for i in range(1, window + 1):
        df[f'Vol_Lag_{i}'] = df['Abs_Return'].shift(i)
    
    df = df.dropna()
    feature_cols = [f'Vol_Lag_{i}' for i in range(1, window + 1)]
    return df[feature_cols], df['Target']

@st.cache_data
def load_data(ticker_list, start_date):
    """
    Downloads data using yfinance. 
    Cached to prevent re-downloading on every interaction.
    """
    if not ticker_list:
        return pd.DataFrame()
    
    try:
        # We assume today is the end date for loading
        data = yf.download(ticker_list, start=start_date, group_by='ticker', auto_adjust=True)
        
        # Handle MultiIndex / Single Index nuances of yfinance
        if len(ticker_list) == 1:
            # If single ticker, yf returns a simple dataframe, we just want Close
            return data['Close'].to_frame(name=ticker_list[0])
        else:
            # If multiple, it returns a MultiIndex (Ticker, OHLCV). We robustly extract Close.
            close_data = pd.DataFrame()
            for t in ticker_list:
                try:
                    close_data[t] = data[t]['Close']
                except KeyError:
                    st.warning(f"Could not find Close data for {t}")
            return close_data
            
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# MAIN LAYOUT & SIDEBAR
# -----------------------------------------------------------------------------

st.title("📈 SMI Professional Portfolio Manager")

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎛️ Configuration")
    
    # 1. Stock Selection
    selectable_tickers = [t for t in SMI_COMPANIES.keys() if t != "^SSMI"]
    
    tickers = st.multiselect(
        "Select Portfolio Assets", 
        options=selectable_tickers, 
        format_func=lambda x: f"{SMI_COMPANIES[x]} ({x})",
        default=["NESN.SW", "NOVN.SW", "UBSG.SW"]
    )
    
    # 2. Date Selection
    st.subheader("Time Horizon")
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start", value=pd.to_datetime("2020-01-01"))
    end_date = col2.date_input("End", value=pd.to_datetime("today"))
    
    # 3. Risk Free Rate (New Feature)
    rf_rate = st.number_input("Risk Free Rate (%)", value=1.0, step=0.1) / 100

    # 4. Portfolio Weights Logic (Improved UX)
    st.markdown("---")
    st.header("⚖️ Portfolio Allocation")
    
    weights = {}
    if tickers:
        with st.expander("Adjust Weights", expanded=True):
            # FEATURE: Auto-Equalize Weights Button
            if st.button("Auto-Equalize Weights"):
                equal_weight = 100.0 / len(tickers)
                for t in tickers:
                    st.session_state[f"w_{t}"] = equal_weight
            
            total_weight = 0
            for t in tickers:
                # Key logic: Use session_state to allow programmatic updates
                key = f"w_{t}"
                if key not in st.session_state:
                    st.session_state[key] = 100.0 / len(tickers)
                
                weights[t] = st.number_input(
                    f"{SMI_COMPANIES[t]} (%)", 
                    min_value=0.0, max_value=100.0, 
                    step=1.0, 
                    key=key
                )
                total_weight += weights[t]

            # Dynamic Feedback on Total Weight
            diff = total_weight - 100.0
            if abs(diff) < 0.1:
                st.success(f"Total: {total_weight:.1f}% ✅")
            else:
                st.error(f"Total: {total_weight:.1f}% (Diff: {diff:.1f}%) ❌")

# -----------------------------------------------------------------------------
# DATA LOADING & PROCESSING
# -----------------------------------------------------------------------------

# We calculate the "Safe Start" date (2 years back) to ensure ML has enough history
buffer_date = pd.Timestamp(start_date) - pd.DateOffset(years=2)
tickers_to_load = list(set(tickers + ["^SSMI"]))

full_data = load_data(tickers_to_load, buffer_date)

if full_data.empty:
    st.warning("No data loaded. Please check your ticker selection.")
    st.stop()

# -----------------------------------------------------------------------------
# PORTFOLIO CALCULATION
# -----------------------------------------------------------------------------
cleaned_df = full_data.dropna()

# Calculate Portfolio logic only if weights are valid
has_portfolio = False
if tickers and abs(sum(weights.values()) - 100.0) < 0.1:
    has_portfolio = True
    
    # Isolate selected stocks
    selection = cleaned_df[tickers]
    
    # Calculate weighted returns
    w_list = [weights[t] / 100.0 for t in tickers]
    portfolio_daily_ret = selection.pct_change().dot(w_list)
    
    # Create Index (Start at 100)
    portfolio_idx = (1 + portfolio_daily_ret).cumprod() * 100
    portfolio_idx.iloc[0] = 100
    
    cleaned_df["💼 My Portfolio"] = portfolio_idx

# -----------------------------------------------------------------------------
# FILTERING FOR DISPLAY
# -----------------------------------------------------------------------------
# Now we cut the dataframe to the user's specific requested date range for the charts
mask = (cleaned_df.index >= pd.Timestamp(start_date).tz_localize(None)) & \
       (cleaned_df.index <= pd.Timestamp(end_date).tz_localize(None))
display_df = cleaned_df.loc[mask]

if display_df.empty:
    st.error("Selected date range resulted in empty data (maybe weekends selected?).")
    st.stop()

# -----------------------------------------------------------------------------
# VISUALIZATION TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Analysis over Time", "📉 Risk & Correlation", "🤖 AI Volatility Predictor"])

# We create a mapping for the legends
legend_map = SMI_COMPANIES.copy()
legend_map["💼 My Portfolio"] = "💼 My Portfolio"

# --- TAB 1: DYNAMIC ANALYSIS (Best of Both Worlds) ---
with tab1:
    col_kpi_1, col_kpi_2 = st.columns([1, 3])
    
    with col_kpi_1:
        st.subheader("Metric Selection")
        # FEATURE: Dynamic KPI Selector from V3 restored!
        metric_options = [ 
            "Cumulative Return (Indexed to 100)",
            "Annualized Return (30-Day Rolling)",
            "Volatility (30-Day Rolling)",
            "Sharpe Ratio (30-Day Rolling)",
            "Drawdown (Historical)",
            "Value at Risk 95% (30-Day Rolling)"
        ]
        selected_metric = st.radio("Select Metric to Visualize", metric_options)
    
    with col_kpi_2:
        st.subheader("Evolution Chart")
        
        # Calculate Plot Data based on selection
        returns = display_df.pct_change().dropna()
        window = 30
        
        if selected_metric == "Cumulative Return (Indexed to 100)":
            plot_data = display_df / display_df.iloc[0] * 100
        elif selected_metric == "Annualized Return (30-Day Rolling)":
            plot_data = returns.rolling(window=window).mean() * 252
        elif selected_metric == "Volatility (30-Day Rolling)":
            plot_data = returns.rolling(window=window).std() * np.sqrt(252)
        elif selected_metric == "Sharpe Ratio (30-Day Rolling)":
            rolling_ret = returns.rolling(window=window).mean() * 252
            rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
            plot_data = (rolling_ret - rf_rate) / rolling_vol
        elif selected_metric == "Drawdown (Historical)":
            cum_rets = (1 + returns).cumprod()
            running_max = cum_rets.cummax()
            plot_data = (cum_rets / running_max) - 1
        elif selected_metric == "Value at Risk 95% (30-Day Rolling)":
            plot_data = returns.rolling(window=window).quantile(0.05)
            
        # --- ALTAIR CHART ---
        # Convert to long format for Altair
        plot_long = plot_data.reset_index().melt('Date', var_name='Asset', value_name='Value')
        
        # Apply readable names
        plot_long['Asset Name'] = plot_long['Asset'].map(lambda x: legend_map.get(x, x))
        
        chart = alt.Chart(plot_long).mark_line().encode(
            x='Date:T',
            y=alt.Y('Value:Q', title=selected_metric),
            color='Asset Name:N',
            tooltip=['Date:T', 'Asset Name:N', alt.Tooltip('Value:Q', format='.2f')]
        ).interactive()
        
        st.altair_chart(chart, use_container_width=True)

    # Raw Data Download
    with st.expander("See Raw Data"):
        st.dataframe(display_df.tail(10))
        st.download_button(
            "Download CSV", 
            display_df.to_csv().encode('utf-8'), 
            "portfolio_data.csv", 
            "text/csv"
        )

# --- TAB 2: RISK & CORRELATION ---
with tab2:
    col1, col2 = st.columns([1, 1])
    
    # KPI Table
    with col1:
        st.subheader("Financial KPIs Summary")
        kpi_df = calculate_KPI(display_df, risk_free_rate=rf_rate)
        
        format_dict = {
            'Ann. Return': '{:.2%}',
            'Cumulative Return': '{:.2%}',
            'Ann. Volatility': '{:.2%}',
            'Sharpe Ratio': '{:.2f}',
            'Sortino Ratio': '{:.2f}',
            'Max Drawdown': '{:.2%}',
            'VaR (95%)': '{:.2%}'
        }
        
        # Highlight Portfolio Row
        st.dataframe(kpi_df.style.format(format_dict).apply(
            lambda x: ['background: #e6f3ff' if x.name == "💼 My Portfolio" else '' for i in x], axis=1
        ))

    # Correlation Matrix
    with col2:
        st.subheader("Correlation Matrix")
        if tickers:
            corr_df = display_df[tickers].pct_change().corr()
            
            # Use readable names
            corr_df.index = corr_df.index.map(lambda x: SMI_COMPANIES.get(x, x))
            corr_df.columns = corr_df.columns.map(lambda x: SMI_COMPANIES.get(x, x))
            
            # Prepare for Altair
            corr_melt = corr_df.reset_index().rename(columns={'index': 'Asset 1'}).melt(
                id_vars='Asset 1', var_name='Asset 2', value_name='Correlation'
            )

            base = alt.Chart(corr_melt).encode(
                x='Asset 1:O',
                y='Asset 2:O'
            )

            # Heatmap
            heatmap = base.mark_rect().encode(
                color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='redblue', domain=[-1, 1])),
                tooltip=['Asset 1', 'Asset 2', alt.Tooltip('Correlation', format='.2f')]
            )

            # Text Labels
            text = base.mark_text(baseline='middle').encode(
                text=alt.Text('Correlation:Q', format='.2f'),
                color=alt.condition(
                    alt.datum.Correlation > 0.5, 
                    alt.value('white'),
                    alt.value('black')
                )
            )

            st.altair_chart(heatmap + text, use_container_width=True)
        else:
            st.info("Select stocks to see correlation.")

    # Risk-Return Scatter
    st.subheader("Risk vs. Return Landscape")
    
    scatter_data = kpi_df.reset_index().rename(columns={'index': 'Asset'})
    scatter_data['Asset Name'] = scatter_data['Asset'].map(lambda x: legend_map.get(x, x))
    
    # Altair Scatter
    scatter = alt.Chart(scatter_data).mark_circle(size=100).encode(
        x=alt.X('Ann. Volatility', axis=alt.Axis(format='%', title='Risk (Ann. Volatility)')),
        y=alt.Y('Ann. Return', axis=alt.Axis(format='%', title='Return (Ann. Return)')),
        color='Asset Name',
        tooltip=['Asset Name', 'Ann. Return', 'Ann. Volatility', 'Sharpe Ratio', 'Max Drawdown']
    ).interactive()

    # Add Crosshairs (0,0)
    rule_x = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(strokeDash=[5, 5]).encode(x='x')
    rule_y = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(strokeDash=[5, 5]).encode(y='y')

    st.altair_chart(scatter + rule_x + rule_y, use_container_width=True)

# --- TAB 3: MACHINE LEARNING ---
with tab3:
    st.subheader("🤖 AI Volatility Forecast")
    st.write("Using **Random Forest Regression** to predict future market turbulence based on recent patterns.")
    
    col_ml_1, col_ml_2 = st.columns(2)
    
    with col_ml_1:
        ml_options = list(display_df.columns)
        ml_ticker = st.selectbox("Select Asset to Predict", ml_options, format_func=lambda x: legend_map.get(x, x))
    
    with col_ml_2:
        horizon_map = {"Next Day": 1, "Next Week (5 Days)": 5, "Next Month (21 Days)": 21}
        horizon_lbl = st.selectbox("Prediction Horizon", list(horizon_map.keys()))
        horizon_val = horizon_map[horizon_lbl]

    if st.button("Train Model & Predict"):
        with st.spinner("Training Random Forest Model..."):
            # Use CLEANED_DF (full history) for training
            series = cleaned_df[ml_ticker]
            X, y = prepare_regression_data(series, window=21, horizon=horizon_val)
            
            if len(X) > 100:
                # 80/20 Split
                split = int(len(X) * 0.8)
                X_train, X_test = X.iloc[:split], X.iloc[split:]
                y_train, y_test = y.iloc[:split], y.iloc[split:]
                
                # Model
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # Evaluation
                preds = model.predict(X_test)
                mae = mean_absolute_error(y_test, preds)
                
                # Future Prediction
                latest_features = X.iloc[-1:].values
                future_pred = model.predict(latest_features)[0]
                
                # --- RESULTS ---
                c1, c2, c3 = st.columns(3)
                c1.metric(f"Predicted Volatility ({horizon_lbl})", f"{future_pred:.2%}")
                c2.metric("Model Accuracy (MAE)", f"{mae:.2%}", delta_color="inverse")
                c3.metric("Current Hist. Volatility (Last 21d)", f"{series.pct_change().abs().tail(21).mean():.2%}")
                
                # Altair Chart for Prediction
                res_df = pd.DataFrame({'Date': y_test.index, 'Actual': y_test.values, 'Predicted': preds})
                res_melt = res_df.melt('Date', var_name='Type', value_name='Volatility')
                
                ml_chart = alt.Chart(res_melt).mark_line().encode(
                    x='Date:T',
                    y=alt.Y('Volatility:Q', axis=alt.Axis(format='%')),
                    color='Type:N',
                    tooltip=['Date:T', 'Type:N', alt.Tooltip('Volatility:Q', format='.2%')]
                ).properties(title="Model Validation: Actual vs Predicted Volatility (Test Set)").interactive()
                
                st.altair_chart(ml_chart, use_container_width=True)
                
            else:
                st.error("Not enough historical data to train model. Try selecting an older start date.")
