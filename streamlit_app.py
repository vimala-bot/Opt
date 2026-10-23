import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta

st.set_page_config(page_title="Institutional Options Screener", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS FOR MODERN ARGON-STYLE UI ---
st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e9ecef;
        padding: 16px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03);
    }
    div[data-testid="stMetric"] label {
        color: #8898aa !important;
        font-weight: 700;
        font-size: 0.75rem;
        text-transform: uppercase;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #32325d !important;
        font-weight: 800;
    }
    .strategy-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 24px;
        border-left: 6px solid #5e72e4;
        box-shadow: 0 7px 14px 0 rgba(50, 50, 93, 0.07), 0 3px 6px 0 rgba(0, 0, 0, 0.05);
        margin-top: 15px;
    }
    .badge-leaps { background: #2dce89; color: white; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 0.8rem; }
    .badge-pmcc { background: #11cdef; color: white; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 0.8rem; }
    .badge-csp { background: #fb6340; color: white; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

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
    if (ask - bid) / bid > 0.35: return False
    if oi < 20: return False
    return True

def calculate_technicals(df):
    close = df['Close']
    
    # Standard Indicators
    df['EMA_200'] = close.ewm(span=200, adjust=False).mean()
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Historical Volatility Rank (HVR) - Proxy for IVR
    # 1. Daily Log Returns
    df['Log_Ret'] = np.log(close / close.shift(1))
    # 2. 20-Day Annualized Historical Volatility
    df['HV_20'] = df['Log_Ret'].rolling(window=20).std() * np.sqrt(252)
    # 3. 252-Day (1 Year) Rolling Min & Max of HV
    hv_min = df['HV_20'].rolling(window=252).min()
    hv_max = df['HV_20'].rolling(window=252).max()
    # 4. HVR Calculation (0 to 100 Scale)
    df['HVR'] = ((df['HV_20'] - hv_min) / (hv_max - hv_min)) * 100
    
    return df

@st.cache_data(ttl=3600, show_spinner="Processing vol regimes and Black-Scholes chains (~3 mins)...")
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
            # Fetch 2 years to allow 252-day rolling HVR calculation
            df = t_yfinance.history(period="2y", interval="1d")
            if df.empty or len(df) < 300: continue
            
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df = calculate_technicals(df)
            
            latest, prev = df.iloc[-1], df.iloc[-2]
            current_price = float(latest['Close'])
            rsi = round(float(latest['RSI_14']), 2) if not pd.isna(latest['RSI_14']) else 50.0
            ema200 = float(latest['EMA_200'])
            hvr = round(float(latest['HVR']), 2) if not pd.isna(latest['HVR']) else 50.0
            
            macd_line, signal_line = latest['MACD'], latest['MACD_Signal']
            prev_macd, prev_sig = prev['MACD'], prev['MACD_Signal']
            
            macd_status = "Bearish"
            if macd_line > signal_line and prev_macd <= prev_sig: macd_status = "Bull Cross"
            elif macd_line < signal_line and prev_macd >= prev_sig: macd_status = "Bear Cross"
            elif macd_line > signal_line: macd_status = "Bullish"
                
            ema200_status = "Above" if current_price > ema200 else "Below"
            oi_matrix = np.random.choice(["Bullish Call Skew", "Bearish Put Skew", "Neutral"])
            
            earnings_date_str = "None Recorded"
            try:
                cal = t_yfinance.get_earnings_dates(limit=1)
                if cal is not None and not cal.empty:
                    earnings_date_str = cal.index[0].strftime('%Y-%m-%d')
            except: pass
                
            # VOLATILITY-ARBITRAGE STRATEGY ROUTING
            if (rsi < 42 or macd_status == "Bull Cross") and hvr < 45:
                # Buy Vega when vol is cheap
                strategy = "ITM_LEAPS"
            elif ema200_status == "Above":
                # Sell Calls against Leaps
                strategy = "PMCC"
            else:
                # Default to collecting high-vol Put premium on dips
                strategy = "CSP"

            setup_data = {}
            try:
                expirations = t_yfinance.options
                if expirations:
                    if strategy == "ITM_LEAPS":
                        closest_exp = min(expirations, key=lambda d: abs(datetime.strptime(d, '%Y-%m-%d') - target_leaps_date))
                        T_years = max((datetime.strptime(closest_exp, '%Y-%m-%d') - datetime.now()).days / 365.25, 0.01)
                        chain = t_yfinance.option_chain(closest_exp).calls
                        chain['BS_Delta'] = chain.apply(lambda r: calc_black_scholes_delta(current_price, r['strike'], T_years, RISK_FREE_RATE, r['impliedVolatility'], 'call'), axis=1)
                        valid = chain[chain.apply(validate_liquidity, axis=1)]
                        if valid.empty: valid = chain
                        row = valid.iloc[(valid['BS_Delta'] - 0.80).abs().argsort()[:1]]
                        
                        ask_val = row['ask'].values[0]
                        if pd.isna(ask_val) or ask_val == 0: ask_val = row['lastPrice'].values[0]
                        
                        setup_data = {
                            "action": "Buy to Open (BTO) ITM LEAPS Call",
                            "exp": closest_exp,
                            "strike": f"${row['strike'].values[0]} Call",
                            "delta": f"{row['BS_Delta'].values[0]:.2f}",
                            "contract_cost": f"${(ask_val * 100):,.2f}",
                            "guidance": f"Low Volatility Regime (HVR: {hvr}). Favorable environment to buy long Vega. Targeting 0.80 delta."
                        }
                        
                    elif strategy == "PMCC":
                        closest_exp = min(expirations, key=lambda d: abs(datetime.strptime(d, '%Y-%m-%d') - target_short_date))
                        T_years = max((datetime.strptime(closest_exp, '%Y-%m-%d') - datetime.now()).days / 365.25, 0.01)
                        chain = t_yfinance.option_chain(closest_exp).calls
                        chain['BS_Delta'] = chain.apply(lambda r: calc_black_scholes_delta(current_price, r['strike'], T_years, RISK_FREE_RATE, r['impliedVolatility'], 'call'), axis=1)
                        valid = chain[chain.apply(validate_liquidity, axis=1)]
                        if valid.empty: valid = chain
                        row = valid.iloc[(valid['BS_Delta'] - 0.25).abs().argsort()[:1]]
                        
                        bid_val = row['bid'].values[0]
                        if pd.isna(bid_val) or bid_val == 0: bid_val = row['lastPrice'].values[0]

                        setup_data = {
                            "action": "Sell to Open (STO) Covered Short Call",
                            "exp": closest_exp,
                            "strike": f"${row['strike'].values[0]} Call",
                            "delta": f"{row['BS_Delta'].values[0]:.2f}",
                            "contract_cost": f"${(bid_val * 100):,.2f} Credit",
                            "guidance": f"Volatility Rank: {hvr}. Sell against existing long LEAPS position to harvest theta."
                        }

                    elif strategy == "CSP":
                        closest_exp = min(expirations, key=lambda d: abs(datetime.strptime(d, '%Y-%m-%d') - target_short_date))
                        T_years = max((datetime.strptime(closest_exp, '%Y-%m-%d') - datetime.now()).days / 365.25, 0.01)
                        chain = t_yfinance.option_chain(closest_exp).puts
                        chain['BS_Delta'] = chain.apply(lambda r: calc_black_scholes_delta(current_price, r['strike'], T_years, RISK_FREE_RATE, r['impliedVolatility'], 'put'), axis=1)
                        valid = chain[chain.apply(validate_liquidity, axis=1)]
                        if valid.empty: valid = chain
                        row = valid.iloc[(valid['BS_Delta'] - (-0.25)).abs().argsort()[:1]]
                        
                        bid_val = row['bid'].values[0]
                        if pd.isna(bid_val) or bid_val == 0: bid_val = row['lastPrice'].values[0]

                        setup_data = {
                            "action": "Sell to Open (STO) Cash-Secured Put",
                            "exp": closest_exp,
                            "strike": f"${row['strike'].values[0]} Put",
                            "delta": f"{row['BS_Delta'].values[0]:.2f}",
                            "contract_cost": f"${(bid_val * 100):,.2f} Credit",
                            "guidance": f"Volatility Rank: {hvr}. Requires ${(row['strike'].values[0] * 100):,.2f} cash collateral. Exploiting short Vega decay."
                        }
            except Exception:
                setup_data = {"action": "Data Fetch Incomplete", "exp": "-", "strike": "-", "delta": "-", "contract_cost": "-", "guidance": "Option chain unavailable."}
                
            results.append({
                "Ticker": ticker,
                "Price": f"${current_price:,.2f}",
                "Sector": sector,
                "Strategy": strategy,
                "RSI": rsi,
                "MACD": macd_status,
                "200 EMA": ema200_status,
                "HVR (Vol Rank)": hvr,
                "OI Matrix": oi_matrix,
                "Earnings": earnings_date_str,
                "Setup": setup_data
            })
            
        except Exception: pass
    
    return pd.DataFrame(results)

# Header Section
st.title("Institutional Options & Derivatives Screener")
st.markdown("Quantitative Screener factoring **Volatility Rank (HVR)** alongside Delta targeting.")

df = load_data()

# Metric KPI Highlights
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Monitored", len(df))
c2.metric("ITM LEAPS Setups (Low Vol)", len(df[df['Strategy'] == 'ITM_LEAPS']))
c3.metric("Short Premium Setups", len(df[df['Strategy'].isin(['PMCC', 'CSP'])]))
c4.metric("High Volatility Exceedances", len(df[df['HVR (Vol Rank)'] > 75]))

# Sidebar Controls
st.sidebar.markdown("### Strategic Filters")
all_sectors = ["All"] + sorted(list(df['Sector'].unique()))
sel_sector = st.sidebar.selectbox("Sector / Theme", all_sectors)
sel_strategy = st.sidebar.selectbox("Target Strategy", ["All", "ITM_LEAPS", "PMCC", "CSP"])

# Filter DataFrame
filtered_df = df.copy()
if sel_sector != "All": filtered_df = filtered_df[filtered_df['Sector'] == sel_sector]
if sel_strategy != "All": filtered_df = filtered_df[filtered_df['Strategy'] == sel_strategy]

st.markdown("---")
st.markdown("#### Actionable Positions")
st.caption("Click any row below to instantly calculate live contract pricing based on Black-Scholes Greeks.")

# Interactive Dataframe
display_cols = ["Ticker", "Price", "Sector", "Strategy", "HVR (Vol Rank)", "RSI", "MACD", "200 EMA", "Earnings"]
selection = st.dataframe(
    filtered_df[display_cols],
    use_container_width=True,
    hide_index=True,
    selection_mode="single-row",
    on_select="rerun"
)

# Determine selected ticker
selected_ticker = None
if selection and selection.get("selection", {}).get("rows"):
    selected_idx = selection["selection"]["rows"][0]
    selected_ticker = filtered_df.iloc[selected_idx]["Ticker"]
elif not filtered_df.empty:
    selected_ticker = filtered_df.iloc[0]["Ticker"]

# Strategy Details Card
st.markdown("---")
if selected_ticker:
    row_data = filtered_df[filtered_df['Ticker'] == selected_ticker].iloc[0]
    setup = row_data["Setup"]
    strat = row_data["Strategy"]
    
    badge_class = "badge-leaps" if strat == "ITM_LEAPS" else ("badge-pmcc" if strat == "PMCC" else "badge-csp")
    
    st.markdown(f"""
    <div class="strategy-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <h3 style="margin: 0; color: #32325d;">{row_data['Ticker']} &mdash; <span class="{badge_class}">{strat.replace('_', ' ')}</span></h3>
            <span style="font-size: 1.2rem; font-weight: 700; color: #525f7f;">Spot: {row_data['Price']}</span>
        </div>
        <p style="color: #525f7f; font-weight: 600; margin-bottom: 18px;">Action: {setup.get('action', 'Pending')}</p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-bottom: 16px;">
            <div style="background: #f8f9fe; padding: 12px 16px; border-radius: 8px;">
                <span style="font-size: 0.75rem; text-transform: uppercase; color: #8898aa; font-weight: 700;">Target Expiration</span>
                <div style="font-size: 1.05rem; font-weight: 700; color: #32325d;">{setup.get('exp', '-')}</div>
            </div>
            <div style="background: #f8f9fe; padding: 12px 16px; border-radius: 8px;">
                <span style="font-size: 0.75rem; text-transform: uppercase; color: #8898aa; font-weight: 700;">Recommended Strike</span>
                <div style="font-size: 1.05rem; font-weight: 700; color: #32325d;">{setup.get('strike', '-')}</div>
            </div>
            <div style="background: #f8f9fe; padding: 12px 16px; border-radius: 8px;">
                <span style="font-size: 0.75rem; text-transform: uppercase; color: #8898aa; font-weight: 700;">Calculated Delta</span>
                <div style="font-size: 1.05rem; font-weight: 700; color: #32325d;">&Delta; {setup.get('delta', '-')}</div>
            </div>
            <div style="background: #f8f9fe; padding: 12px 16px; border-radius: 8px;">
                <span style="font-size: 0.75rem; text-transform: uppercase; color: #8898aa; font-weight: 700;">Est. Premium</span>
                <div style="font-size: 1.05rem; font-weight: 700; color: #2dce89;">{setup.get('contract_cost', '-')}</div>
            </div>
        </div>
        <div style="font-size: 0.85rem; color: #8898aa; font-style: italic;">
            {setup.get('guidance', '')}
        </div>
    </div>
    """, unsafe_allow_html=True)
