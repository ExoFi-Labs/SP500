import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm, skew, kurtosis, genpareto
from copulas.multivariate import GaussianMultivariate
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

# Load S&P 500 stock list
df_sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
df_sp500 = df_sp500[['Symbol', 'Security', 'GICS Sector']]
df_sp500.columns = ['Ticker', 'Name', 'Sector']

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Layout
app.layout = dbc.Container([
    html.H1("S&P 500 Portfolio Risk Analysis"),
    html.P("Build a portfolio and analyze its risk using advanced models like VaR, ES, EVT, and Copulas."),

    dbc.Row([
        dbc.Col([
            html.H4("Asset Selection"),
            dcc.Dropdown(
                id='sector-filter',
                options=[{'label': s, 'value': s} for s in sorted(df_sp500['Sector'].unique())],
                multi=True,
                placeholder="Filter by Sector"
            ),
            dcc.Dropdown(
                id='asset-selector',
                options=[{'label': f"{row['Ticker']} - {row['Name']}", 'value': row['Ticker']} for _, row in df_sp500.iterrows()],
                multi=True,
                placeholder="Select assets"
            )
        ], width=4),

        dbc.Col([
            html.H4("Set Weights"),
            html.Div(id='weight-inputs'),
            html.Button("Run Analysis", id='run-btn', className='btn btn-primary mt-2')
        ], width=4),

        dbc.Col([
            html.H4("Portfolio Pie Chart"),
            dcc.Graph(id='pie-chart')
        ], width=4),
    ]),

    html.Hr(),

    html.H3("Risk Analysis Results"),
    html.Div(id='risk-output'),
    html.Div(id='feedback')
])

# Update asset dropdown by sector
@app.callback(
    Output('asset-selector', 'options'),
    Input('sector-filter', 'value')
)
def filter_assets_by_sector(selected_sectors):
    if not selected_sectors:
        return [{'label': f"{row['Ticker']} - {row['Name']}", 'value': row['Ticker']} for _, row in df_sp500.iterrows()]
    df_filtered = df_sp500[df_sp500['Sector'].isin(selected_sectors)]
    return [{'label': f"{row['Ticker']} - {row['Name']}", 'value': row['Ticker']} for _, row in df_filtered.iterrows()]

# Create weight inputs
@app.callback(
    Output('weight-inputs', 'children'),
    Input('asset-selector', 'value')
)
def generate_weight_inputs(tickers):
    if not tickers:
        return html.Div("Select assets to assign weights")
    return [
        dbc.InputGroup([
            dbc.InputGroupText(ticker),
            dbc.Input(id={'type': 'weight-input', 'index': ticker}, placeholder="e.g. 0.1", type='number', step=0.01)
        ], className='mb-1')
        for ticker in tickers
    ]

# Run analysis
@app.callback(
    [Output('pie-chart', 'figure'),
     Output('risk-output', 'children'),
     Output('feedback', 'children')],
    Input('run-btn', 'n_clicks'),
    State('asset-selector', 'value'),
    State({'type': 'weight-input', 'index': dash.dependencies.ALL}, 'value')
)
def run_analysis(n, tickers, weights):
    if n is None: # Prevent callback from firing on initial load
        return dash.no_update, dash.no_update, dash.no_update
        
    if not tickers or not weights or all(w is None for w in weights) or sum([w for w in weights if w]) == 0:
        return {}, "Please select valid assets and weights. Ensure weights sum to a non-zero value.", ""

    valid_tickers_weights = [(t, w) for t, w in zip(tickers, weights) if w is not None and w != 0]
    if not valid_tickers_weights:
         return {}, "Please assign non-zero weights to selected assets.", ""

    tickers = [item[0] for item in valid_tickers_weights]
    weights_values = [item[1] for item in valid_tickers_weights]
    
    weights_np = np.array(weights_values, dtype=float) # Ensure float type
    weights_np /= weights_np.sum() # Normalize weights

    try:
        raw_data = yf.download(tickers, period='1y', group_by='ticker', auto_adjust=True, progress=False)
    except Exception as e:
        return {}, f"Error downloading data: {e}", ""

    if raw_data.empty:
        return {}, "No data downloaded for the selected tickers. Please check ticker symbols.", ""

    if len(tickers) == 1:
        if tickers[0] in raw_data.columns: # For single ticker, yf might return a DataFrame directly
             df_data = raw_data[['Close']].copy()
             df_data.columns = tickers
        elif 'Close' in raw_data.columns: # yf might return a Series for a single ticker
            df_data = raw_data['Close'].to_frame(tickers[0])
        else:
            return {}, f"Could not find 'Close' data for ticker {tickers[0]}", ""
    else:
        df_data_list = []
        for ticker in tickers:
            if ticker in raw_data and 'Close' in raw_data[ticker]:
                df_data_list.append(raw_data[ticker]['Close'].rename(ticker))
            elif ticker in raw_data.columns and 'Close' in raw_data.columns: # Fallback if structure is flat
                 if isinstance(raw_data[ticker], pd.Series): # Check if it's a Series (e.g. from a failed multi-ticker download)
                    df_data_list.append(raw_data[ticker].rename(ticker)) # Assume this column is 'Close'
                 else: # Should not happen with group_by='ticker' usually
                    return {}, f"Unexpected data structure for {ticker}. 'Close' column missing.", ""
            else:
                return {}, f"Data for ticker {ticker} or its 'Close' column not found in downloaded data.", ""
        if not df_data_list:
            return {}, "No close price data could be extracted for any selected ticker.", ""
        df_data = pd.concat(df_data_list, axis=1)
        df_data.columns = tickers # Ensure columns are correctly named

    df_data = df_data.dropna()
    if df_data.empty or len(df_data) < 2: # Need at least 2 points for pct_change
        return {}, "Not enough data after dropping NaNs to calculate returns.", ""
        
    returns = df_data.pct_change().dropna()
    if returns.empty:
        return {}, "Not enough data to calculate percentage returns.", ""
        
    portfolio_returns = returns.dot(weights_np)

    # Risk metrics
    mu = portfolio_returns.mean()
    sigma = portfolio_returns.std()
    skewness = skew(portfolio_returns)
    kurt = kurtosis(portfolio_returns)
    var_95 = norm.ppf(0.05, mu, sigma)
    es_95 = mu - sigma * norm.pdf(norm.ppf(0.05)) / 0.05
    hist_var_95 = np.percentile(portfolio_returns, 5)
    hist_es_95 = portfolio_returns[portfolio_returns < hist_var_95].mean()

    # EVT Tail Risk
    threshold = portfolio_returns.quantile(0.9) # Use a common quantile for tail definition
    excess = portfolio_returns[portfolio_returns > threshold] - threshold
    tail_var = np.nan # Default value
    if len(excess) > 10: # Need sufficient points for GPD fit
        try:
            shape, loc, scale = genpareto.fit(excess, floc=0) # Fix location to 0 as we're fitting excesses
            # VaR from GPD: u + (beta/xi) * ( (n/Nu * (1-alpha))^(-xi) -1 )
            # Here, alpha is confidence level for VaR (e.g., 0.95), so 1-alpha is the tail probability (0.05)
            # n/Nu is roughly 1 / (1 - quantile_used_for_threshold), so here 1 / (1 - 0.9) = 10
            # P(X > x | X > u) = (1 + xi * (x-u)/beta)^(-1/xi)
            # We want x_q such that P(X > x_q) = q_tail (e.g., 0.05)
            # P(X > x_q) = P(X > x_q | X > u) * P(X > u)
            # q_tail = (1 + xi * (x_q-u)/beta)^(-1/xi) * (1 - quantile_for_threshold)
            # (q_tail / (1-quantile_for_threshold)) = (1 + xi * (x_q-u)/beta)^(-1/xi)
            # (q_tail / (1-quantile_for_threshold))^(-xi) = 1 + xi * (x_q-u)/beta
            # ((q_tail / (1-quantile_for_threshold))^(-xi) - 1) * beta / xi = x_q - u
            # x_q = u + (beta/xi) * ( ( (1-confidence_level_for_VaR) / (1-quantile_for_threshold) )^(-xi) - 1 )
            
            # Confidence level for VaR (e.g., 95%) means we're interested in the 5% tail
            confidence_level_var = 0.95 
            tail_probability_var = 1 - confidence_level_var # 0.05

            # The proportion of data above threshold 'u' is (1 - quantile_for_threshold)
            # Here quantile_for_threshold is 0.9, so P(X > u) = 0.1
            prob_exceed_threshold = 1 - 0.9 

            if shape != 0: # Avoid division by zero
                tail_var = threshold + (scale / shape) * ( ( (tail_probability_var / prob_exceed_threshold) )**(-shape) - 1)
            else: # Exponential tail (xi -> 0 limit of GPD)
                tail_var = threshold - scale * np.log(tail_probability_var / prob_exceed_threshold)

        except Exception as e:
            print(f"EVT fit error: {e}")
            tail_var = np.nan
    else:
        tail_var = np.nan

    # Copula Simulation
    if not returns.empty and returns.shape[1] > 0 : # Check if returns DataFrame is not empty and has columns
        u = returns.rank(axis=0, pct=True) # Use pct=True for ranks scaled to [0,1]
        copula = GaussianMultivariate()
        try:
            copula.fit(u)
            sim = copula.sample(1000)  # Shape: (1000, num_assets)

            # --- Start of Fix ---
            num_simulations = sim.shape[0]
            num_assets = sim.shape[1]
            sim_inv_values = np.zeros((num_simulations, num_assets)) # This will be 2D: (1000, num_assets)

            for i in range(num_assets):
                # historical_asset_returns: 1D array of actual returns for asset i
                historical_asset_returns = returns.iloc[:, i].values # Use .iloc for column access
                # simulated_quantiles_for_asset: 1D array of simulated quantiles for asset i, scaled to 0-100
                simulated_quantiles_for_asset = sim[:, i] * 100
                
                # Apply inverse CDF (empirical quantile function) for the current asset
                sim_inv_values[:, i] = np.percentile(
                    historical_asset_returns, 
                    simulated_quantiles_for_asset, 
                    axis=0 # Operates along the single axis of historical_asset_returns
                )

            # sim_inv_values now has the correct 2D shape (num_simulations, num_assets)
            sim_inv = pd.DataFrame(sim_inv_values, columns=returns.columns)
            # --- End of Fix ---

            sim_portfolio_returns = sim_inv.dot(weights_np)
            copula_var = np.percentile(sim_portfolio_returns, 5)
            copula_es = sim_portfolio_returns[sim_portfolio_returns < copula_var].mean()
        except Exception as e:
            print(f"Copula error: {e}")
            copula_var = np.nan
            copula_es = np.nan
    else:
        copula_var = np.nan
        copula_es = np.nan


    fig = px.pie(values=weights_np, names=tickers, title="Portfolio Allocation")

    risk_summary_items = [
        html.Li(f"Mean Return: {mu:.4f}"),
        html.Li(f"Std Dev (Volatility): {sigma:.4f}"),
        html.Li(f"Skewness: {skewness:.4f}"),
        html.Li(f"Kurtosis: {kurt:.4f}"),
        html.Li(f"Parametric VaR (95%): {var_95:.4f}"),
        html.Li(f"Parametric ES (95%): {es_95:.4f}"),
        html.Li(f"Historical VaR (95%): {hist_var_95:.4f}"),
        html.Li(f"Historical ES (95%): {hist_es_95:.4f}"),
    ]
    if not np.isnan(tail_var):
        risk_summary_items.append(html.Li(f"EVT VaR (95%): {tail_var:.4f}"))
    else:
        risk_summary_items.append(html.Li("EVT VaR (95%): Not Calculated (insufficient tail data or fit error)"))
    
    if not np.isnan(copula_var):
        risk_summary_items.append(html.Li(f"Copula VaR (95%): {copula_var:.4f}"))
        risk_summary_items.append(html.Li(f"Copula ES (95%): {copula_es:.4f}"))
    else:
        risk_summary_items.append(html.Li("Copula VaR (95%): Not Calculated (error in copula process)"))
        risk_summary_items.append(html.Li("Copula ES (95%): Not Calculated (error in copula process)"))

    risk_summary = html.Ul(risk_summary_items)


    feedback_msg = []
    if not np.isnan(var_95) and not np.isnan(hist_var_95) and var_95 < hist_var_95:
        feedback_msg.append("Parametric VaR (Normal assumption) may underestimate historical losses.")
    if not np.isnan(copula_var) and not np.isnan(hist_var_95) and copula_var < hist_var_95: # Compare copula VaR to historical
        feedback_msg.append("Copula-based VaR suggests potential losses could be more severe than simple historical VaR.")
    if len(weights_np) > 0 and np.max(weights_np) > 0.4:
        feedback_msg.append(f"High concentration: Asset '{tickers[np.argmax(weights_np)]}' has {np.max(weights_np)*100:.1f}% weight.")
    if len(tickers) > 3 and np.abs(skewness) < 0.5 and kurt > 2.5 and kurt < 3.5: # Looser bounds for kurtosis
        feedback_msg.append("The portfolio appears reasonably diversified with relatively low skewness and near-normal kurtosis.")
    elif np.abs(skewness) > 1:
        feedback_msg.append(f"Portfolio returns are significantly skewed ({'positively' if skewness > 1 else 'negatively'}).")
    if kurt > 4: # Leptokurtic
        feedback_msg.append("Portfolio returns exhibit high kurtosis (fat tails), indicating higher probability of extreme events than a normal distribution.")


    return fig, risk_summary, html.Ul([html.Li(msg) for msg in feedback_msg])

# Run app
if __name__ == '__main__':
    app.run(debug=True)