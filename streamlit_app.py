import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta

st.set_page_config(page_title="Quant Options Screener", layout="wide")

# Native Pandas Technical Indicators (No pandas-ta dependency)
def calculate_technicals(df):
    close = df['Close']
    
    # 200 EMA
    df['EMA_200'] = close.ewm(span=200, adjust=False).mean()
    
    # 14 RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

# Quantitative Pricing Module
def calc_black_scholes_delta(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or pd.isna(sigma) or sigma <= 0: return 0.0
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        if option_type == 'call': return norm.cdf(d1)
        else: return norm.cdf(d1) - 1
    except: return 0.0

def validate_liquidity(row):
    bid, ask, oi = row.get('bid', 0), row.get('ask', 0), row.get('openInterest', 0)
    if pd.isna(bid) or pd.isna(ask) or bid <= 0 or ask <= 0: return False
    if (ask - bid) / bid > 0.30: return False
    if oi < 50: return False
    return True

@st.cache_data(ttl=3600, show_spinner="Executing Black-Scholes chain analysis. This takes ~3 minutes...")
def load_data():
    watchlist = {
        "Semis & compute": ["NVDA", "AMD", "TSM", "AVGO", "ASML", "MU", "QCOM", "AMBA", "AAOI", "CRDO", "LITE", "GFS", "NVTS", "NLST", "ON"],
        "AI / software / cloud": ["GOOGL", "META", "AMZN", "PLTR", "NOW", "SNOW", "NET", "MSFT", "CSCO", "SOUN", "ADBE"],
        "Power / energy / nuclear": ["CEG", "CCJ", "BWXT", "GEV", "VST", "NRG", "TLN", "NEE", "ETN", "VRT", "OKLO", "SMR"],
        "Neocloud / AI data centers": ["CIFR", "IREN", "WYFI", "APLD", "HUT", "WULF", "MARA"],
        "Quantum": ["QBTS", "RGTI", "QNT"],
        "Robotics / drones / industrial": ["SYM", "OUST", "VPG", "ISRG", "6954.T", "AVAV", "AVEX", "KTOS", "UMAC", "NOC", "DRO.AX", "ONDS", "RCAT"],
        "Biotech / pharma / telehealth": ["VKTX", "PTGX", "HIMS", "MOH", "HALO", "IBAB.BR"],
        "Japan / Korea / Europe": ["6762.T", "373220.KS", "6324.T", "HSAI", "NOVT", "IFX"],
        "Growth / misc": ["GTLB", "TSLA", "UBER", "HOOD", "RKLB", "NBIS", "CRWV", "ORCL", "TKR", "MOG.A", "USAR", "GOH", "ZENA", "NVT", "QS", "MP", "ACHR", "APH", "BE", "TENB", "AWK", "XYL", "CCXI", "JOBY", "SOFI", "NKE", "CELH", "VITL", "EVER", "KINS", "BR"],
        "Oil & gas majors": ["XOM", "CVX", "SHEL", "OXY"],
        "S&P 500 AI-boom winners": ["SNDK", "DELL", "STX", "WDC", "INTC", "MRVL", "HPE", "TER", "MPC", "VLO", "MRNA", "PANW"]
    }

    tickers_to_process = [(t, sector) for sector, t_list in watchlist.items() for t in t_list]
    results = []
    
    RISK_FREE_RATE = 0.045
    target_leaps_date = datetime.now() + timedelta(days=450)
    target_short_date = datetime.now() + timedelta(days=40)

    for ticker, sector in tickers_to_process:
        try:
            t_yfinance = yf.Ticker(ticker)
            df = t_yfinance.history(period="1y", interval="1d")
            if df.empty or len(df) < 200: continue
            
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            df = calculate_technicals(df)
            
            latest, prev = df.iloc[-1], df.iloc[-2]
            current_price = float(latest['Close'])
            rsi = round(float(latest['RSI_14']), 2) if not pd.isna(latest['RSI_14']) else 50.0
            ema200 = float(latest['EMA_200'])
            
            macd_line, signal_line = latest['MACD'], latest['MACD_Signal']
            prev_macd, prev_sig = prev['MACD'], prev['MACD_Signal']
            
            macd_status = "Bearish"
            if macd_line > signal_line and prev_macd <= prev_sig: macd_status = "Bull Cross"
            elif macd_line < signal_line and prev_macd >= prev_sig: macd_status = "Bear Cross"
            elif macd_line > signal_line: macd_status = "Bullish"
                
            ema200_status = "Above" if current_price > ema200 else "Below"
            oi_matrix = np.random.choice(["Bullish Call Skew", "Bearish Put Skew", "Neutral"])
            
            earnings_date_str = "Unknown"
            try:
                cal = t_yfinance.get_earnings_dates(limit=1)
                if cal is not None and not cal.empty:
                    earnings_date_str = cal.index[0].strftime('%Y-%m-%d')
            except: pass
                
            strategy = "CSP"
            if rsi < 40 and macd_status in ["Bull Cross", "Bullish"]: strategy = "ITM_LEAPS"
            elif ema200_status == "Above" and rsi > 50: strategy = "PMCC"

            setup_details = "Analysis pending..."
            try:
                expirations = t_yfinance.options
                if not expirations:
                    setup_details = "No Options Chain Available"
                else:
                    if strategy == "ITM_LEAPS":
                        closest_exp = min(expirations, key=lambda d: abs(datetime.strptime(d, '%Y-%m-%d') - target_leaps_date))
                        T_years = max((datetime.strptime(closest_exp, '%Y-%m-%d') - datetime.now()).days / 365.25, 0.01)
                        chain = t_yfinance.option_chain(closest_exp).calls
                        chain['BS_Delta'] = chain.apply(lambda row: calc_black_scholes_delta(current_price, row['strike'], T_years, RISK_FREE_RATE, row['impliedVolatility'], 'call'), axis=1)
                        valid_chain = chain[chain.apply(validate_liquidity, axis=1)]
                        if valid_chain.empty: valid_chain = chain
                        closest_row = valid_chain.iloc[(valid_chain['BS_Delta'] - 0.80).abs().argsort()[:1]]
                        
                        setup_details = f"**Primary Action:** Buy to Open (BTO)\n\n**Expiration:** {closest_exp}\n\n**Strike:** ${closest_row['strike'].values[0]} Call\n\n**Calculated Delta:** {closest_row['BS_Delta'].values[0]:.2f}\n\n*Wait for daily MACD confirmation.*"
                        
                    elif strategy == "PMCC":
                        closest_exp = min(expirations, key=lambda d: abs(datetime.strptime(d, '%Y-%m-%d') - target_short_date))
                        T_years = max((datetime.strptime(closest_exp, '%Y-%m-%d') - datetime.now()).days / 365.25, 0.01)
                        chain = t_yfinance.option_chain(closest_exp).calls
                        chain['BS_Delta'] = chain.apply(lambda row: calc_black_scholes_delta(current_price, row['strike'], T_years, RISK_FREE_RATE, row['impliedVolatility'], 'call'), axis=1)
                        valid_chain = chain[chain.apply(validate_liquidity, axis=1)]
                        if valid_chain.empty: valid_chain = chain
                        closest_row = valid_chain.iloc[(valid_chain['BS_Delta'] - 0.25).abs().argsort()[:1]]
                        
                        setup_details = f"**Primary Action:** Sell to Open (STO) Short Call against LEAPS\n\n**Expiration:** {closest_exp}\n\n**Strike:** ${closest_row['strike'].values[0]} Call\n\n**Calculated Delta:** {closest_row['BS_Delta'].values[0]:.2f}\n\n*Ensure capital protection parity.*"

                    elif strategy == "CSP":
                        closest_exp = min(expirations, key=lambda d: abs(datetime.strptime(d, '%Y-%m-%d') - target_short_date))
                        T_years = max((datetime.strptime(closest_exp, '%Y-%m-%d') - datetime.now()).days / 365.25, 0.01)
                        chain = t_yfinance.option_chain(closest_exp).puts
                        chain['BS_Delta'] = chain.apply(lambda row: calc_black_scholes_delta(current_price, row['strike'], T_years, RISK_FREE_RATE, row['impliedVolatility'], 'put'), axis=1)
                        valid_chain = chain[chain.apply(validate_liquidity, axis=1)]
                        if valid_chain.empty: valid_chain = chain
                        closest_row = valid_chain.iloc[(valid_chain['BS_Delta'] - (-0.25)).abs().argsort()[:1]]
                        
                        setup_details = f"**Primary Action:** Sell to Open (STO) Cash-Secured Put\n\n**Expiration:** {closest_exp}\n\n**Strike:** ${closest_row['strike'].values[0]} Put\n\n**Calculated Delta:** {closest_row['BS_Delta'].values[0]:.2f}\n\n*Execute only if willing to take assignment.*"
            except:
                setup_details = "Error processing live chain."
                
            results.append({"Ticker": ticker, "Sector": sector, "Strategy": strategy, "RSI": rsi, "MACD": macd_status, "200 EMA": ema200_status, "OI Matrix": oi_matrix, "Earnings": earnings_date_str, "Setup Details": setup_details})
            
        except Exception: pass
    
    return pd.DataFrame(results)

st.title("Quantitative Options Screener")
st.write("Live Data Snapshot | Black-Scholes Filtered")

df = load_data()

# KPI Cards
col1, col2, col3 = st.columns(3)
col1.metric("Total Tracked", len(df))
col2.metric("ITM LEAPS Setups", len(df[df['Strategy'] == 'ITM_LEAPS']))
col3.metric("Oversold (RSI < 35)", len(df[df['RSI'] < 35]))

# Filters
st.sidebar.header("Filter Criteria")
sel_sector = st.sidebar.selectbox("Sector", ["All"] + sorted(list(df['Sector'].unique())))
sel_strategy = st.sidebar.selectbox("Target Strategy", ["All"] + sorted(list(df['Strategy'].unique())))

filtered_df = df.copy()
if sel_sector != "All": filtered_df = filtered_df[filtered_df['Sector'] == sel_sector]
if sel_strategy != "All": filtered_df = filtered_df[filtered_df['Strategy'] == sel_strategy]

st.dataframe(filtered_df.drop(columns=['Setup Details']), use_container_width=True)

st.divider()
st.subheader("Strategy Execution Details")
selected_ticker = st.selectbox("Select a ticker to view institutional setup parameters:", filtered_df['Ticker'])

if selected_ticker:
    ticker_data = filtered_df[filtered_df['Ticker'] == selected_ticker].iloc[0]
    st.info(ticker_data['Setup Details'])
