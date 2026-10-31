import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta

st.set_page_config(page_title="Institutional Options Screener", layout="wide", initial_sidebar_state="expanded")

# --- WEALTHSIMPLE & QUANT INSTITUTIONAL CSS ---
st.markdown("""
<style>
    .stApp {
        background-color: #FBFBFA;
        color: #32302F;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #EAE8E5;
        padding: 18px 22px;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    div[data-testid="stMetric"] label {
        color: #767370 !important;
        font-weight: 600;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #1A1918 !important;
        font-weight: 700;
        font-size: 1.5rem;
    }
    .strategy-card {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 28px;
        border: 1px solid #EAE8E5;
        border-left: 5px solid #178754;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        margin-top: 20px;
    }
    .badge-leaps { background: #E6F4ED; color: #178754; padding: 5px 12px; border-radius: 20px; font-weight: 700; font-size: 0.75rem; }
    .badge-pmcc { background: #E1F5FE; color: #0288D1; padding: 5px 12px; border-radius: 20px; font-weight: 700; font-size: 0.75rem; }
    .badge-csp { background: #FFF3E0; color: #E65100; padding: 5px 12px; border-radius: 20px; font-weight: 700; font-size: 0.75rem; }
    .badge-none { background: #F1F0EE; color: #767370; padding: 5px 12px; border-radius: 20px; font-weight: 700; font-size: 0.75rem; }
    .risk-alert {
        background: #FAFAF9; 
        padding: 16px 20px; 
        border-radius: 10px; 
        border-left: 4px solid #D32F2F; 
        border: 1px solid #EAE8E5;
        margin-top: 20px;
    }
    section[data-testid="stSidebar"] {
        background-color: #F4F3F1;
        border-right: 1px solid #EAE8E5;
    }
</style>
""", unsafe_allow_html=True)

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
    volume = df['Volume']
    df['EMA_200'] = close.ewm(span=200, adjust=False).mean()
    
    # Stock Volume Ratio (Today Volume / 20-day Volume SMA)
    df['Vol_SMA_20'] = volume.rolling(window=20).mean()
    df['Stock_Vol_Ratio'] = np.where(df['Vol_SMA_20'] > 0, round(volume / df['Vol_SMA_20'], 2), 1.0)

    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    df['Log_Ret'] = np.log(close / close.shift(1))
    df['HV_20'] = df['Log_Ret'].rolling(window=20).std() * np.sqrt(252)
    hv_min = df['HV_20'].rolling(window=252).min()
    hv_max = df['HV_20'].rolling(window=252).max()
    df['HVR'] = ((df['HV_20'] - hv_min) / (hv_max - hv_min)) * 100
    return df

def get_market_state_key():
    now = pd.Timestamp.now('US/Eastern')
    if now.dayofweek >= 5:
        return f"weekend_{now.strftime('%Y-%W')}"
    
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    if now < market_open:
        return f"premarket_{now.strftime('%Y-%m-%d')}"
    elif now >= market_close:
        return f"postmarket_{now.strftime('%Y-%m-%d')}"
    else:
        minute_block = (now.minute // 15) * 15
        return f"open_{now.strftime('%Y-%m-%d_%H')}_{minute_block:02d}"

@st.cache_data(show_spinner="Running Institutional Quant Engine (~3 mins)...")
def load_data(market_state_key):
    stocks = [
        "NVDA", "AMD", "TSM", "AVGO", "ASML", "MU", "QCOM", "AMBA", "AAOI", "CRDO", "LITE", "GFS", "NVTS", "NLST", "ON", "LRCX", "SKHY", "CAMT", "ALAB", "AMAT", "KLAC", "ARM",
        "GOOGL", "META", "AMZN", "PLTR", "NOW", "SNOW", "NET", "MSFT", "CSCO", "SOUN", "ADBE", "VEEV", "SHOP", "FTNT", "DDOG", "CRWD", "IBM", "DOCN",
        "CEG", "CCJ", "BWXT", "GEV", "VST", "NRG", "TLN", "NEE", "ETN", "VRT", "OKLO", "SMR",
        "CIFR", "IREN", "WYFI", "APLD", "HUT", "WULF", "MARA",
        "QBTS", "RGTI", "QNT",
        "SYM", "OUST", "VPG", "ISRG", "6954.T", "AVAV", "AVEX", "KTOS", "UMAC", "NOC", "DRO.AX", "ONDS", "RCAT", "LHX", "CGNX",
        "VKTX", "PTGX", "HIMS", "MOH", "HALO", "IBAB.BR",
        "6762.T", "373220.KS", "6324.T", "HSAI", "NOVT", "IFX", "010130.KS",
        "LAC", "GT",
        "GTLB", "TSLA", "UBER", "HOOD", "RKLB", "NBIS", "CRWV", "ORCL", "TKR", "MOG.A", "USAR", "ZENA", "NVT", "QS", "MP", "ACHR", "APH", "BE", "TENB", "AWK", "XYL", "CCXI", "JOBY", "SOFI", "NKE", "CELH", "VITL", "EVER", "KINS", "BR",
        "XOM", "CVX", "SHEL", "OXY",
        "SNDK", "DELL", "STX", "WDC", "INTC", "MRVL", "HPE", "TER", "MPC", "VLO", "MRNA", "PANW",
        "NFLX", "TMTG", "RDDT",
        "SPCX",
        "COHR", "NOK", "SNPS",
        "TMUS", "T", "VZ",
        "EQIX", "DLR", "ANET", "SMCI",
        "COP", "ABT", "CMCSA", "ZTS", "DIS",
        "ANT",
        "JBLU",
        "AAPL",
        "MTSI", "CIEN", "GLW", "FN", "AEHR",
        "ZS", "RBRK", "NTSK", "OKTA", "CHKP", "S", "QLYS", "GD", "FFIV", "VRNS", "FSLY", "4704.T", "CVLT",
        "TGT",
        "EH", "PONY", "XPEV", "JD", "3690.HK",
        "GM", "KO", "BA", "JPM", "WMT", "PFE", "MCD"
    ]
    etfs = [
        "GRID", "FIW", "PHO", "HTEC", "CIBR", "HACK", "BUG", "KROB", "CGW", "SOXX", "VGT", "XEQT", "AIA", "DRAM", "AIQ", "ARKQ", "BBH", "ARKG", "NUKZ", "AIPO", "NCLD",
        "SMH", "QTUM", "URA", "NLR", "XLU", "XLI", "QQQ", "CLOU", "ROBO", "BOTZ"
    ]
    leveraged = [
        "KORU", "TQQQ", "SOXL", "TMF", "TNA", "UPRO", "UDOW", "QLD", "SSO", "UWM", "UYG"
    ]

    tickers_to_process = [(t, "Stock") for t in stocks] + [(t, "ETF") for t in etfs] + [(t, "Leveraged ETF") for t in leveraged]
    results = []
    
    RISK_FREE_RATE = 0.045
    target_leaps_date = datetime.now() + timedelta(days=450)
    target_short_date = datetime.now() + timedelta(days=40)

    for ticker, eq_type in tickers_to_process:
        try:
            t_yfinance = yf.Ticker(ticker)
            df = t_yfinance.history(period="2y", interval="1d")
            if df.empty or len(df) < 300: continue
            
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df = calculate_technicals(df)
            
            latest, prev = df.iloc[-1], df.iloc[-2]
            current_price = float(latest['Close'])
            rsi = round(float(latest['RSI_14']), 2) if not pd.isna(latest['RSI_14']) else 50.0
            ema200 = float(latest['EMA_200'])
            hvr = round(float(latest['HVR']), 2) if not pd.isna(latest['HVR']), 2) if not pd.isna(latest['HVR']) else 50.0
            stock_vol_ratio = float(latest['Stock_Vol_Ratio'])
            
            macd_line, signal_line = latest['MACD'], latest['MACD_Signal']
            prev_macd, prev_sig = prev['MACD'], prev['MACD_Signal']
            
            macd_status = "Bearish"
            if macd_line > signal_line and prev_macd <= prev_sig: macd_status = "Bull Cross"
            elif macd_line < signal_line and prev_macd >= prev_sig: macd_status = "Bear Cross"
            elif macd_line > signal_line: macd_status = "Bullish"
                
            ema200_status = "Above" if current_price > ema200 else "Below"
            
            earnings_date_str = "None Recorded"
            try:
                cal = t_yfinance.get_earnings_dates(limit=1)
                if cal is not None and not cal.empty:
                    earnings_date_str = cal.index[0].strftime('%Y-%m-%d')
            except: pass
                
            if eq_type in ["ETF", "Leveraged ETF"]:
                if ema200_status == "Above":
                    strategy = "PMCC"
                else:
                    strategy = "None"
            else:
                if (rsi < 42 or macd_status == "Bull Cross") and hvr < 45:
                    strategy = "ITM_LEAPS"
                elif ema200_status == "Above":
                    strategy = "PMCC"
                else:
                    strategy = "CSP"

            iv_crush_status = "N/A"
            if strategy == "ITM_LEAPS":
                iv_crush_status = "Optimal (Avoids Crush)" if hvr < 45 else "Warning (Vega Risk)"
            elif strategy in ["PMCC", "CSP"]:
                iv_crush_status = "Optimal (Targets Crush)" if hvr > 50 else "Sub-optimal"

            setup_data = {}
            opt_vol_ratio = 0.0
            try:
                expirations = t_yfinance.options
                if expirations and strategy != "None":
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
                        
                        opt_vol = row['volume'].values[0] if not pd.isna(row['volume'].values[0]) else 0
                        opt_oi = row['openInterest'].values[0] if not pd.isna(row['openInterest'].values[0]) else 1
                        opt_vol_ratio = round(opt_vol / max(opt_oi, 1), 2)

                        setup_data = {
                            "action": "Buy to Open (BTO) ITM LEAPS Call", 
                            "exp": closest_exp, 
                            "strike": f"${row['strike'].values[0]} Call", 
                            "delta": f"{row['BS_Delta'].values[0]:.2f}", 
                            "contract_cost": f"${(ask_val * 100):,.2f}", 
                            "stop_loss": f"Close if underlying closes below 200 EMA (${ema200:.2f}) OR option premium draws down 50% (${(ask_val * 50):,.2f}).",
                            "stop_loss_col": f"EMA 200 (${ema200:.1f}) / -50% Premium"
                        }
                        
                    elif strategy == "PMCC":
                        closest_exp = min(expirations, key=lambda d: abs(datetime.strptime(d, '%Y-%m-%d') - target_short_date))
                        T_years = max((datetime.strptime(closest_exp, '%Y-%m-%d') - datetime.now()).days / 365.25, 0.01)
                        chain = t_yfinance.option_chain(closest_exp).calls
                        chain['BS_Delta'] = chain.apply(lambda r: calc_black_scholes_delta(current_price, r['strike'], T_years, RISK_FREE_RATE, r['impliedVolatility'], 'call'), axis=1)
                        valid = chain[chain.apply(validate_liquidity, axis=1)]
                        if valid.empty: valid = chain
                        row = valid.iloc[(valid['BS_Delta'] - 0.25).abs().argsort()[:1]]
                        strike_val = row['strike'].values[0]
                        bid_val = row['bid'].values[0]
                        if pd.isna(bid_val) or bid_val == 0: bid_val = row['lastPrice'].values[0]

                        opt_vol = row['volume'].values[0] if not pd.isna(row['volume'].values[0]) else 0
                        opt_oi = row['openInterest'].values[0] if not pd.isna(row['openInterest'].values[0]) else 1
                        opt_vol_ratio = round(opt_vol / max(opt_oi, 1), 2)

                        setup_data = {
                            "action": "Sell to Open (STO) Covered Short Call", 
                            "exp": closest_exp, 
                            "strike": f"${strike_val} Call", 
                            "delta": f"{row['BS_Delta'].values[0]:.2f}", 
                            "contract_cost": f"${(bid_val * 100):,.2f} Credit", 
                            "stop_loss": f"Buy-to-Close (BTC) if underlying breaches Short Strike (${strike_val:.2f}). Manage delta if spot drops below 200 EMA.",
                            "stop_loss_col": f"Short Strike Breach (${strike_val:.1f})"
                        }

                    elif strategy == "CSP":
                        closest_exp = min(expirations, key=lambda d: abs(datetime.strptime(d, '%Y-%m-%d') - target_short_date))
                        T_years = max((datetime.strptime(closest_exp, '%Y-%m-%d') - datetime.now()).days / 365.25, 0.01)
                        chain = t_yfinance.option_chain(closest_exp).puts
                        chain['BS_Delta'] = chain.apply(lambda r: calc_black_scholes_delta(current_price, r['strike'], T_years, RISK_FREE_RATE, r['impliedVolatility'], 'put'), axis=1)
                        valid = chain[chain.apply(validate_liquidity, axis=1)]
                        if valid.empty: valid = chain
                        row = valid.iloc[(valid['BS_Delta'] - (-0.25)).abs().argsort()[:1]]
                        strike_val = row['strike'].values[0]
                        bid_val = row['bid'].values[0]
                        if pd.isna(bid_val) or bid_val == 0: bid_val = row['lastPrice'].values[0]

                        opt_vol = row['volume'].values[0] if not pd.isna(row['volume'].values[0]) else 0
                        opt_oi = row['openInterest'].values[0] if not pd.isna(row['openInterest'].values[0]) else 1
                        opt_vol_ratio = round(opt_vol / max(opt_oi, 1), 2)

                        setup_data = {
                            "action": "Sell to Open (STO) Cash-Secured Put", 
                            "exp": closest_exp, 
                            "strike": f"${strike_val} Put", 
                            "delta": f"{row['BS_Delta'].values[0]:.2f}", 
                            "contract_cost": f"${(bid_val * 100):,.2f} Credit", 
                            "stop_loss": f"BTC if premium inflates 300% (${(bid_val * 300):,.2f}), OR take assignment at ${strike_val:.2f}.",
                            "stop_loss_col": f"+300% Premium / Strike (${strike_val:.1f})"
                        }
                else:
                    setup_data = {"action": "Restricted / No Chain", "exp": "-", "strike": "-", "delta": "-", "contract_cost": "-", "stop_loss": "N/A", "stop_loss_col": "N/A"}
                    opt_vol_ratio = 0.0
            except Exception:
                setup_data = {"action": "Data Fetch Incomplete", "exp": "-", "strike": "-", "delta": "-", "contract_cost": "-", "stop_loss": "Unavailable", "stop_loss_col": "N/A"}
                opt_vol_ratio = 0.0
                
            results.append({
                "Ticker": ticker,
                "Type": eq_type,
                "Price": f"${current_price:,.2f}",
                "Strategy": strategy,
                "HVR (Vol Rank)": hvr,
                "IV Crush Setup": iv_crush_status,
                "Stock Vol Ratio": stock_vol_ratio,
                "Opt Vol/OI Ratio": opt_vol_ratio,
                "RSI": rsi,
                "MACD": macd_status,
                "200 EMA": ema200_status,
                "Stop-Loss Setup": setup_data.get("stop_loss_col", "N/A"),
                "Earnings": earnings_date_str,
                "Setup": setup_data
            })
            
        except Exception: pass
    
    return pd.DataFrame(results)

st.title("Institutional Options & Derivatives Screener")
st.markdown("Quantitative framework combining **Volatility Rank (HVR)**, Volume Ratios, Delta targeting, and structured stop-losses.")

market_key = get_market_state_key()
st.caption(f"Market State: **{market_key.replace('_', ' ').title()}** (Resets automatically M-F every 15 minutes)")

df = load_data(market_key)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Monitored", len(df), help="Total assets successfully tracked.")
c2.metric("ITM LEAPS Setups", len(df[df['Strategy'] == 'ITM_LEAPS']), help="Low volatility setups (HVR < 45) optimal for long calls.")
c3.metric("Short Premium Setups", len(df[df['Strategy'].isin(['PMCC', 'CSP'])]))
c4.metric("High Vol Exceedances", len(df[df['HVR (Vol Rank)'] > 75]), help="Extreme volatility rank (HVR > 75).")

st.sidebar.markdown("### Filters")
sel_type = st.sidebar.selectbox("Equity Type", ["All", "Stock", "ETF", "Leveraged ETF"])
sel_strategy = st.sidebar.selectbox("Target Strategy", ["All", "ITM_LEAPS", "PMCC", "CSP", "None"])

filtered_df = df.copy()
if sel_type != "All": filtered_df = filtered_df[filtered_df['Type'] == sel_type]
if sel_strategy != "All": filtered_df = filtered_df[filtered_df['Strategy'] == sel_strategy]

st.markdown("---")
st.markdown("#### Actionable Positions")
st.caption("Select any row below to view live contract pricing and execution rules.")

def color_formatting(val):
    if isinstance(val, str):
        if any(w in val for w in ['Optimal', 'Bullish', 'Bull Cross', 'Above']):
            return 'background-color: #E6F4ED; color: #178754; font-weight: 600;'
        elif any(w in val for w in ['Warning', 'Bearish', 'Bear Cross', 'Below', 'Sub-optimal']):
            return 'background-color: #FCE8E6; color: #C5221F; font-weight: 600;'
    return ''

display_cols = ["Ticker", "Type", "Price", "Strategy", "HVR (Vol Rank)", "IV Crush Setup", "Stock Vol Ratio", "Opt Vol/OI Ratio", "RSI", "MACD", "200 EMA", "Stop-Loss Setup", "Earnings"]

styled_df = filtered_df[display_cols].style.map(color_formatting, subset=['IV Crush Setup', 'MACD', '200 EMA'])

selection = st.dataframe(
    styled_df,
    use_container_width=True,
    hide_index=True,
    selection_mode="single-row",
    on_select="rerun",
    column_config={
        "Stock Vol Ratio": st.column_config.NumberColumn("Stock Vol Ratio", help="Today's stock volume relative to its 20-day Volume SMA (>1.0 indicates abnormal volume accumulation)."),
        "Opt Vol/OI Ratio": st.column_config.NumberColumn("Opt Vol/OI Ratio", help="Option Volume to Open Interest ratio on the target strike (>1.0 indicates heavy fresh positioning or sweep activity)."),
        "HVR (Vol Rank)": st.column_config.NumberColumn("HVR (Vol Rank)", help="Historical Volatility Rank (0-100)."),
        "Stop-Loss Setup": st.column_config.TextColumn("Stop-Loss Setup", help="Systematic stop-loss or risk control trigger for the active position.")
    }
)

selected_ticker = None
if selection and selection.get("selection", {}).get("rows"):
    selected_idx = selection["selection"]["rows"][0]
    selected_ticker = filtered_df.iloc[selected_idx]["Ticker"]
elif not filtered_df.empty:
    selected_ticker = filtered_df.iloc[0]["Ticker"]

st.markdown("---")
if selected_ticker:
    row_data = filtered_df[filtered_df['Ticker'] == selected_ticker].iloc[0]
    setup = row_data["Setup"]
    strat = row_data["Strategy"]
    
    badge_class = "badge-leaps" if strat == "ITM_LEAPS" else ("badge-pmcc" if strat == "PMCC" else ("badge-csp" if strat == "CSP" else "badge-none"))
    
    st.markdown(f"""
    <div class="strategy-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
            <h3 style="margin: 0; color: #1A1918; font-size: 1.25rem;">{row_data['Ticker']} &mdash; <span class="{badge_class}">{strat.replace('_', ' ')}</span></h3>
            <span style="font-size: 1.1rem; font-weight: 700; color: #1A1918;">Spot: {row_data['Price']}</span>
        </div>
        <p style="color: #4A4744; font-weight: 600; margin-bottom: 20px;">Action: {setup.get('action', 'Pending')}</p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-bottom: 16px;">
            <div style="background: #F4F3F1; padding: 14px 18px; border-radius: 10px;">
                <span style="font-size: 0.7rem; text-transform: uppercase; color: #767370; font-weight: 700; letter-spacing: 0.05em;">Target Expiration</span>
                <div style="font-size: 1rem; font-weight: 700; color: #1A1918; margin-top: 4px;">{setup.get('exp', '-')}</div>
            </div>
            <div style="background: #F4F3F1; padding: 14px 18px; border-radius: 10px;">
                <span style="font-size: 0.7rem; text-transform: uppercase; color: #767370; font-weight: 700; letter-spacing: 0.05em;">Recommended Strike</span>
                <div style="font-size: 1rem; font-weight: 700; color: #1A1918; margin-top: 4px;">{setup.get('strike', '-')}</div>
            </div>
            <div style="background: #F4F3F1; padding: 14px 18px; border-radius: 10px;">
                <span style="font-size: 0.7rem; text-transform: uppercase; color: #767370; font-weight: 700; letter-spacing: 0.05em;">Calculated Delta</span>
                <div style="font-size: 1rem; font-weight: 700; color: #1A1918; margin-top: 4px;">&Delta; {setup.get('delta', '-')}</div>
            </div>
            <div style="background: #F4F3F1; padding: 14px 18px; border-radius: 10px;">
                <span style="font-size: 0.7rem; text-transform: uppercase; color: #767370; font-weight: 700; letter-spacing: 0.05em;">Est. Premium</span>
                <div style="font-size: 1rem; font-weight: 700; color: #178754; margin-top: 4px;">{setup.get('contract_cost', '-')}</div>
            </div>
        </div>
        <div class="risk-alert">
            <span style="font-size: 0.7rem; text-transform: uppercase; color: #D32F2F; font-weight: 800; letter-spacing: 0.05em;">Risk Management Constraint</span>
            <div style="font-size: 0.9rem; font-weight: 600; color: #1A1918; margin-top: 6px;">
                {setup.get('stop_loss', '')}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
