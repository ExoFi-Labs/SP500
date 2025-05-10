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
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

# --- Data Loading ---
# Load S&P 500 stock list
try:
    df_sp500_list = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    df_sp500 = df_sp500_list[0]
    df_sp500 = df_sp500[['Symbol', 'Security', 'GICS Sector']].copy()
    df_sp500.columns = ['Ticker', 'Name', 'Sector']
    # Clean common ticker issues from Wikipedia (e.g., BRK.B -> BRK-B)
    df_sp500['Ticker'] = df_sp500['Ticker'].str.replace('.', '-', regex=False)
except Exception as e:
    print(f"Error loading S&P 500 list: {e}. Using a placeholder.")
    # Fallback or raise error if essential
    data = {'Ticker': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
            'Name': ['Apple Inc.', 'Microsoft Corp.', 'Alphabet Inc. (C)', 'Amazon.com Inc.', 'Tesla Inc.'],
            'Sector': ['Information Technology', 'Information Technology', 'Communication Services', 'Consumer Discretionary', 'Consumer Discretionary']}
    df_sp500 = pd.DataFrame(data)

# --- App Initialization ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.COSMO]) # Using a different theme
server = app.server

# --- App Layout ---
app.layout = dbc.Container([
    # 1. Intro Section
    dbc.Row([
        dbc.Col([
            html.H1("Advanced Portfolio Risk Analyzer", className="text-primary mb-4"),
            html.P(
                "This interactive tool empowers you to build a customized S&P 500 portfolio and "
                "delve into its risk profile using sophisticated Quantitative Risk Management techniques. "
                "Go beyond simple volatility to understand potential extreme losses and complex asset dependencies, "
                "especially crucial during market stress. Explore Value-at-Risk (VaR), Expected Shortfall (ES), "
                "Extreme Value Theory (EVT), and Copula modeling to make more informed investment decisions."
            ),
            html.Hr(),
        ])
    ]),

    dbc.Row([
        # 2. Asset Selection & Portfolio Builder
        dbc.Col([
            html.H4("1. Portfolio Construction", className="text-secondary"),
            dbc.Label("Filter by Sector:"),
            dcc.Dropdown(
                id='sector-filter',
                options=[{'label': s, 'value': s} for s in sorted(df_sp500['Sector'].unique())],
                multi=True,
                placeholder="All Sectors"
            ),
            dbc.Label("Select Assets:", className="mt-3"),
            dcc.Dropdown(
                id='asset-selector',
                options=[{'label': f"{row['Ticker']} - {row['Name']}", 'value': row['Ticker']}
                         for _, row in df_sp500.sort_values('Ticker').iterrows()],
                multi=True,
                placeholder="Select S&P 500 stocks"
            ),
            html.Div(id='weight-inputs', className="mt-3"),
        ], md=4),

        # Risk Model Configuration
        dbc.Col([
            html.H4("2. Risk Model Configuration", className="text-secondary"),
            dbc.Label("EVT Threshold (Quantile for Peaks Over Threshold):"),
            dcc.Slider(id='evt-threshold-slider', min=0.85, max=0.99, step=0.01, value=0.95,
                       marks={i/100: str(i/100) for i in range(85, 100, 2)}),

            dbc.Label("Copula Type:", className="mt-3"),
            dcc.RadioItems(
                id='copula-type-selector',
                options=[
                    {'label': 'Gaussian Copula', 'value': 'gaussian'},
                ],
                value='gaussian',
                labelStyle={'display': 'block'},
                className="mb-3"
            ),
            dbc.Button("Run Analysis", id='run-btn', color="primary", className="w-100 mt-4", size="lg")
        ], md=4),

        # Portfolio Pie Chart
        dbc.Col([
            html.H4("Portfolio Allocation", className="text-secondary"),
            dcc.Graph(id='pie-chart', config={'displayModeBar': False})
        ], md=4),
    ]),

    html.Hr(className="my-4"),

    # Risk Analysis Results
    dbc.Row([
        dbc.Col([
            html.H3("3. Risk Analysis Results", className="text-info"),
            dcc.Loading(
                id="loading-risk-output",
                type="default",
                children=[
                    html.Div(id='risk-output'),
                    html.Div(id='feedback', className="mt-3") # Educational Feedback
                ]
            )
        ], md=6),
        dbc.Col([
            html.H4("Tail Risk Visualization (EVT)", className="text-secondary"),
             dcc.Loading(
                id="loading-tail-plot",
                type="default",
                children=dcc.Graph(id='tail-risk-plot', config={'displayModeBar': False})
             )
        ], md=6)
    ]),
    html.Footer(
        dbc.Container(
            html.Small("Disclaimer: This tool is for educational purposes only and not financial advice. Market data from yfinance."),
            className="text-muted text-center mt-5 py-3"
        )
    )
], fluid=True, className="py-4")


# --- Helper Functions for Analysis ---
def get_portfolio_data(tickers, period='1y', auto_adjust=True):
    if not tickers:
        return pd.DataFrame(), pd.Series(dtype=float), "No tickers selected."
    try:
        raw_data = yf.download(tickers, period=period, auto_adjust=auto_adjust, progress=False)
        if raw_data.empty:
            return pd.DataFrame(), pd.Series(dtype=float), "No data downloaded. Check tickers or period."

        if len(tickers) == 1:
            if 'Close' in raw_data.columns:
                df_data = raw_data[['Close']].copy()
                df_data.columns = tickers
            else:
                return pd.DataFrame(), pd.Series(dtype=float), f"Could not find 'Close' data for {tickers[0]}."
        else:
            df_data = raw_data['Close'].copy()
            if isinstance(df_data, pd.Series): # If only one ticker downloaded successfully out of many
                if df_data.name in tickers:
                    df_data = df_data.to_frame(name=df_data.name)
                else: # Should not happen if download was for multiple tickers
                    return pd.DataFrame(), pd.Series(dtype=float), "Unexpected data format for single successful ticker."

            # Ensure all selected tickers are present
            missing_tickers = [t for t in tickers if t not in df_data.columns]
            if missing_tickers:
                # Attempt to get data for missing tickers individually (slower, but robust)
                for mt in missing_tickers:
                    try:
                        mt_data = yf.download(mt, period=period, auto_adjust=auto_adjust, progress=False)['Close']
                        if not mt_data.empty:
                            df_data[mt] = mt_data
                    except Exception:
                        pass # If individual download fails, it will be caught later
            df_data = df_data[tickers] # Reorder and select only requested tickers

        df_data = df_data.dropna(how='all') # Drop rows where all values are NaN
        returns = df_data.pct_change().dropna(how='any') # Drop rows with any NaN after pct_change

        if returns.empty or len(returns) < 20: # Need sufficient data
            return pd.DataFrame(), pd.Series(dtype=float), "Not enough historical data after processing for reliable analysis (min 20 days)."
        return returns, None, None # Return asset returns, error if any
    except Exception as e:
        return pd.DataFrame(), pd.Series(dtype=float), f"Error fetching or processing data: {str(e)}"

def calculate_basic_stats(portfolio_returns):
    mu = portfolio_returns.mean()
    sigma = portfolio_returns.std()
    skewness_val = skew(portfolio_returns)
    kurtosis_val = kurtosis(portfolio_returns) # Fisher's kurtosis (normal is 0)
    return mu, sigma, skewness_val, kurtosis_val

def calculate_parametric_risk(mu, sigma, conf_level=0.95):
    alpha = 1 - conf_level
    var_parametric = norm.ppf(alpha, mu, sigma)
    es_parametric = mu - sigma * norm.pdf(norm.ppf(alpha)) / alpha
    return var_parametric, es_parametric

def calculate_historical_risk(portfolio_returns, conf_level=0.95):
    alpha_percentile = (1 - conf_level) * 100
    var_historical = np.percentile(portfolio_returns, alpha_percentile)
    es_historical = portfolio_returns[portfolio_returns < var_historical].mean()
    return var_historical, es_historical

def calculate_evt_risk(portfolio_returns, threshold_quantile, conf_level=0.95):
    if len(portfolio_returns) < 50: # Need more data for EVT
        return np.nan, np.nan, None, pd.Series(dtype=float), "Insufficient data for EVT (min 50 days)."
    
    threshold_val = portfolio_returns.quantile(threshold_quantile)
    excesses = portfolio_returns[portfolio_returns > threshold_val] - threshold_val
    
    if len(excesses) < 20: # Need enough points over threshold
        return np.nan, np.nan, None, excesses, f"Insufficient excesses ({len(excesses)}) for GPD fit at {threshold_quantile*100:.0f}th percentile. Try lower threshold."

    try:
        # Fit GPD to excesses (loc=0 as we are fitting excesses y = x - u)
        params_gpd = genpareto.fit(excesses, floc=0)
        shape_gpd, loc_gpd, scale_gpd = params_gpd # loc_gpd will be 0

        # VaR from GPD
        prob_exceed_u = 1.0 - threshold_quantile # P(X > u)
        tail_prob_var = 1.0 - conf_level       # Alpha for VaR (e.g., 0.05 for VaR95)

        if shape_gpd == 0: # Exponential tail
            evt_var = threshold_val - scale_gpd * np.log(tail_prob_var / prob_exceed_u)
        else:
            evt_var = threshold_val + (scale_gpd / shape_gpd) * (((tail_prob_var / prob_exceed_u))**(-shape_gpd) - 1)
        
        # ES from GPD (for shape < 1)
        if shape_gpd < 1:
            # ES_q(X) = VaR_q(X) + (scale_gpd + shape_gpd * (VaR_q(X) - threshold_val)) / (1 - shape_gpd)
            evt_es = evt_var + (scale_gpd + shape_gpd * (evt_var - threshold_val)) / (1 - shape_gpd)
        else:
            evt_es = np.nan # ES is infinite or undefined for shape >= 1
            
        return evt_var, evt_es, params_gpd, excesses, None
    except Exception as e:
        return np.nan, np.nan, None, excesses, f"EVT GPD fit error: {str(e)}"

def calculate_copula_risk(asset_returns, weights, copula_type='gaussian', n_simulations=1000, conf_level=0.95):
    if asset_returns.empty or asset_returns.shape[1] == 0:
        return np.nan, np.nan, "Asset returns are empty for copula."
    
    # Transform marginals to uniform
    u_empirical = asset_returns.rank(axis=0, pct=True) # pct=True scales to [0,1]

    copula = GaussianMultivariate()
    
    try:
        copula.fit(u_empirical)
        sim_uniform = copula.sample(n_simulations) # Shape: (n_simulations, num_assets)

        num_assets = asset_returns.shape[1]
        sim_returns_inv = np.zeros((n_simulations, num_assets))

        for i in range(num_assets):
            historical_asset_ret = asset_returns.iloc[:, i].values
            sim_quantiles_asset = sim_uniform[:, i] * 100
            sim_returns_inv[:, i] = np.percentile(historical_asset_ret, sim_quantiles_asset, axis=0)
        
        sim_inv_df = pd.DataFrame(sim_returns_inv, columns=asset_returns.columns)
        sim_portfolio_returns = sim_inv_df.dot(weights)

        alpha_percentile = (1 - conf_level) * 100
        copula_var = np.percentile(sim_portfolio_returns, alpha_percentile)
        copula_es = sim_portfolio_returns[sim_portfolio_returns < copula_var].mean()
        return copula_var, copula_es, None
    except Exception as e:
        return np.nan, np.nan, f"Copula simulation error: {str(e)}"

def calculate_sector_concentration(tickers, weights, df_sp500_ref):
    if not tickers or len(weights) == 0:
        return pd.DataFrame(columns=['Sector', 'Weight'])
    
    asset_to_sector = df_sp500_ref.set_index('Ticker')['Sector']
    sector_allocations = {}
    for ticker, weight in zip(tickers, weights):
        sector = asset_to_sector.get(ticker, "Unknown Sector")
        sector_allocations[sector] = sector_allocations.get(sector, 0) + weight
    
    return pd.DataFrame(list(sector_allocations.items()), columns=['Sector', 'Weight']).sort_values(by='Weight', ascending=False)

def generate_feedback_messages(results, sector_allocations):
    feedback = []
    # VaR comparisons
    if not np.isnan(results['var_parametric']) and not np.isnan(results['var_historical']) and results['var_parametric'] < results['var_historical']:
        feedback.append(html.Li("Parametric VaR (Normal assumption) may underestimate historical losses. Consider non-normal models."))
    if not np.isnan(results['copula_var']) and not np.isnan(results['var_historical']) and results['copula_var'] < results['var_historical']:
        feedback.append(html.Li("Copula-based VaR suggests potential tail losses could be more severe than simple Historical VaR, indicating complex dependencies."))
    
    # High concentration
    if not sector_allocations.empty:
        max_sector_weight = sector_allocations['Weight'].iloc[0]
        max_sector_name = sector_allocations['Sector'].iloc[0]
        if max_sector_weight > 0.5: # Example threshold for sector concentration
            feedback.append(html.Li(f"High concentration: {max_sector_weight*100:.1f}% of portfolio in '{max_sector_name}' sector. Diversify further?"))
    
    if len(results['weights_final']) > 0 and np.max(results['weights_final']) > 0.4:
         feedback.append(html.Li(f"Asset concentration: '{results['tickers_final'][np.argmax(results['weights_final'])]}' has {np.max(results['weights_final'])*100:.1f}% weight."))

    # Skewness and Kurtosis
    if not np.isnan(results['skewness']):
        if results['skewness'] < -0.5:
            feedback.append(html.Li(f"Portfolio has significant negative skewness ({results['skewness']:.2f}), indicating a longer left tail (more frequent large losses than gains)."))
        elif results['skewness'] > 0.5:
            feedback.append(html.Li(f"Portfolio has significant positive skewness ({results['skewness']:.2f}), indicating a longer right tail (more frequent large gains than losses)."))
    
    if not np.isnan(results['kurtosis']): # Fisher's Kurtosis
        if results['kurtosis'] > 1: # Significantly leptokurtic (Normal = 0)
            feedback.append(html.Li(f"High Kurtosis ({results['kurtosis']:.2f}): Portfolio returns have fatter tails than a normal distribution, suggesting higher probability of extreme outcomes."))

    # EVT Specific Feedback
    if results.get('gpd_params'):
        shape_gpd = results['gpd_params'][0]
        if not np.isnan(shape_gpd):
            feedback.append(html.Li(f"EVT GPD Fit: Shape parameter (ξ) is {shape_gpd:.3f}."))
            if shape_gpd > 0 and shape_gpd < 0.5:
                feedback.append(html.Li("This GPD shape (0 < ξ < 0.5) indicates fat tails (heavier than exponential/normal) but finite variance, common for financial returns."))
            elif shape_gpd >= 0.5:
                feedback.append(html.Li("This GPD shape (ξ >= 0.5) indicates very fat tails with infinite variance. Extreme caution advised if fit is robust."))
            elif shape_gpd == 0:
                 feedback.append(html.Li("This GPD shape (ξ ≈ 0) suggests tails similar to an exponential or normal distribution (after thresholding)."))
            else: # shape_gpd < 0
                 feedback.append(html.Li("This GPD shape (ξ < 0) suggests bounded tails, which is unusual for financial asset returns. Verify model fit."))
    
    if not feedback:
        feedback.append(html.Li("Analysis complete. Review the metrics for insights."))
    return html.Ul(feedback)

def plot_tail_distribution_graph(portfolio_returns, evt_threshold_val, excesses, gpd_params):
    fig = go.Figure()
    if excesses is not None and len(excesses) > 0 and gpd_params is not None:
        shape_gpd, loc_gpd, scale_gpd = gpd_params
        
        # Histogram of excesses
        fig.add_trace(go.Histogram(x=excesses, name='Excesses over Threshold', nbinsx=30, histnorm='probability density'))
        
        # Fitted GPD PDF
        x_gpd = np.linspace(excesses.min(), excesses.max(), 200)
        pdf_gpd = genpareto.pdf(x_gpd, c=shape_gpd, loc=loc_gpd, scale=scale_gpd) # loc_gpd is 0
        fig.add_trace(go.Scatter(x=x_gpd, y=pdf_gpd, mode='lines', name='Fitted GPD PDF', line=dict(color='red')))
        
        fig.update_layout(
            title_text=f"EVT: Histogram of Excesses over {evt_threshold_val*100:.2f}% Return Threshold & Fitted GPD",
            xaxis_title="Excess Return (Actual Return - Threshold)",
            yaxis_title="Density",
            legend_title_text='Tail Data'
        )
    else:
        fig.update_layout(title_text="Tail Distribution Plot (Not Available)", xaxis_title="Data", yaxis_title="Density")
        fig.add_annotation(text="Insufficient data or error in EVT calculation for tail plot.",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    return fig


# --- Callbacks ---
@app.callback(
    Output('asset-selector', 'options'),
    Input('sector-filter', 'value')
)
def filter_assets_by_sector(selected_sectors):
    if not selected_sectors:
        filtered_df = df_sp500
    else:
        filtered_df = df_sp500[df_sp500['Sector'].isin(selected_sectors)]
    return [{'label': f"{row['Ticker']} - {row['Name']}", 'value': row['Ticker']}
            for _, row in filtered_df.sort_values('Ticker').iterrows()]

@app.callback(
    Output('weight-inputs', 'children'),
    Input('asset-selector', 'value')
)
def generate_weight_inputs(tickers):
    if not tickers:
        return html.Div("Select assets to assign weights.", className="text-muted")
    
    inputs = [
        dbc.Row([
            dbc.Col(dbc.Label(ticker, html_for={'type': 'weight-input', 'index': ticker}), width=3),
            dbc.Col(
                dcc.Input(
                    id={'type': 'weight-input', 'index': ticker},
                    type='number', min=0, max=1, step=0.01, placeholder="e.g. 0.1",
                    className="form-control form-control-sm" # Bootstrap classes
                ), width=9
            )
        ], className="mb-2 align-items-center") for ticker in tickers
    ]
    return html.Div(inputs)

@app.callback(
    [Output('pie-chart', 'figure'),
     Output('risk-output', 'children'),
     Output('feedback', 'children'),
     Output('tail-risk-plot', 'figure')],
    [Input('run-btn', 'n_clicks')],
    [State('asset-selector', 'value'),
     State({'type': 'weight-input', 'index': dash.dependencies.ALL}, 'value'),
     State('evt-threshold-slider', 'value'),
     State('copula-type-selector', 'value')]
)
def master_run_analysis(n_clicks, selected_tickers, weight_values, evt_threshold_quantile, copula_type):
    if n_clicks is None or not selected_tickers:
        empty_fig = go.Figure()
        empty_fig.update_layout(title_text="Portfolio Allocation")
        empty_tail_fig = go.Figure()
        empty_tail_fig.update_layout(title_text="Tail Risk Visualization (EVT)")
        return empty_fig, "Please select assets, assign weights, and click 'Run Analysis'.", "", empty_tail_fig

    # Validate weights
    if not weight_values or all(w is None for w in weight_values):
        return dash.no_update, "Please assign weights to selected assets.", "", dash.no_update
    
    valid_tickers_weights = []
    for ticker, weight_val in zip(selected_tickers, weight_values):
        if weight_val is not None:
            try:
                w = float(weight_val)
                if w < 0: # Negative weights not allowed by UI, but good to check
                     return dash.no_update, f"Weight for {ticker} cannot be negative.", "", dash.no_update
                if w > 0: # Only include assets with positive weights
                    valid_tickers_weights.append((ticker, w))
            except ValueError:
                return dash.no_update, f"Invalid weight for {ticker}. Please enter a number.", "", dash.no_update

    if not valid_tickers_weights:
        return dash.no_update, "Please assign non-zero weights to at least one asset.", "", dash.no_update

    tickers_final = [item[0] for item in valid_tickers_weights]
    weights_final_raw = np.array([item[1] for item in valid_tickers_weights], dtype=float)
    
    if weights_final_raw.sum() == 0: # Should be caught by w > 0 check above, but as safeguard
        return dash.no_update, "Sum of weights cannot be zero.", "", dash.no_update
        
    weights_final = weights_final_raw / weights_final_raw.sum() # Normalize

    # --- 1. Data Fetching & Preprocessing ---
    asset_returns, portfolio_returns_series, error_msg = get_portfolio_data(tickers_final)
    if error_msg:
        return dash.no_update, error_msg, "", dash.no_update
    
    portfolio_returns = asset_returns.dot(weights_final)


    # --- 2. Risk Calculations ---
    results = {'tickers_final': tickers_final, 'weights_final': weights_final}
    results['mu'], results['sigma'], results['skewness'], results['kurtosis'] = calculate_basic_stats(portfolio_returns)
    results['var_parametric'], results['es_parametric'] = calculate_parametric_risk(results['mu'], results['sigma'])
    results['var_historical'], results['es_historical'] = calculate_historical_risk(portfolio_returns)
    
    results['evt_var'], results['evt_es'], results['gpd_params'], excesses_for_plot, evt_error_msg = \
        calculate_evt_risk(portfolio_returns, evt_threshold_quantile)
    
    results['copula_var'], results['copula_es'], copula_error_msg = \
        calculate_copula_risk(asset_returns, weights_final, copula_type)

    # --- 3. Sector Concentration ---
    sector_allocations = calculate_sector_concentration(tickers_final, weights_final, df_sp500)

    # --- 4. Pie Chart ---
    pie_fig = px.pie(
        values=weights_final, 
        names=tickers_final, 
        title="Portfolio Allocation by Asset",
        hole=0.3
    )
    if not sector_allocations.empty:
        sector_pie_fig = px.pie(
            sector_allocations,
            values='Weight',
            names='Sector',
            title="Portfolio Allocation by Sector",
            hole=0.3
        )
        # For now, let's stick to asset pie chart, or one could combine/choose
        # pie_fig = sector_pie_fig # Uncomment to show sector pie instead

    # --- 5. Risk Summary HTML ---
    risk_summary_items = [
        html.H5("Portfolio Overview", className="text-success"),
        html.Li(f"Mean Daily Return: {results['mu']:.4%}"),
        html.Li(f"Daily Volatility (Std Dev): {results['sigma']:.4%}"),
        html.Li(f"Skewness: {results['skewness']:.4f}"),
        html.Li(f"Kurtosis (Fisher): {results['kurtosis']:.4f} (Normal=0)"),

        html.H5("Value-at-Risk (VaR 95%) - Potential loss not exceeded with 95% confidence", className="mt-3 text-success"),
        html.Li(f"Parametric VaR: {results['var_parametric']:.4%}"),
        html.Li(f"Historical VaR: {results['var_historical']:.4%}"),
        html.Li(f"EVT VaR: {results['evt_var']:.4%}" if not np.isnan(results['evt_var']) else f"EVT VaR: Not Calculated ({evt_error_msg or 'Error'})"),
        html.Li(f"Copula VaR: {results['copula_var']:.4%}" if not np.isnan(results['copula_var']) else f"Copula VaR: Not Calculated ({copula_error_msg or 'Error'})"),

        html.H5("Expected Shortfall (ES 95%) - Expected loss given VaR is breached", className="mt-3 text-success"),
        html.Li(f"Parametric ES: {results['es_parametric']:.4%}"),
        html.Li(f"Historical ES: {results['es_historical']:.4%}"),
        html.Li(f"EVT ES: {results['evt_es']:.4%}" if not np.isnan(results['evt_es']) else f"EVT ES: Not Calculated ({evt_error_msg or 'Error'})"),
        html.Li(f"Copula ES: {results['copula_es']:.4%}" if not np.isnan(results['copula_es']) else f"Copula ES: Not Calculated ({copula_error_msg or 'Error'})"),
    ]
    risk_summary_html = html.Div([
        dbc.Alert(f"Analysis based on {len(portfolio_returns)} daily returns.", color="info"),
        html.Ul(risk_summary_items, className="list-unstyled")
    ])

    # --- 6. Feedback Messages ---
    feedback_html = generate_feedback_messages(results, sector_allocations)
    
    # --- 7. Tail Distribution Plot ---
    evt_threshold_val_abs = portfolio_returns.quantile(evt_threshold_quantile)
    tail_plot_fig = plot_tail_distribution_graph(portfolio_returns, evt_threshold_val_abs, excesses_for_plot, results['gpd_params'])

    return pie_fig, risk_summary_html, feedback_html, tail_plot_fig

# --- Run App ---
if __name__ == '__main__':
    app.run(debug=True)