import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, MATCH
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm, skew, kurtosis, genpareto
from copulas.multivariate import GaussianMultivariate # Ensure copulas is installed
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

# --- Data Loading (Same as before) ---
try:
    df_sp500_list = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    df_sp500 = df_sp500_list[0]
    df_sp500 = df_sp500[['Symbol', 'Security', 'GICS Sector']].copy()
    df_sp500.columns = ['Ticker', 'Name', 'Sector']
    df_sp500['Ticker'] = df_sp500['Ticker'].str.replace('.', '-', regex=False)
    df_sp500['label'] = df_sp500['Ticker'] + " - " + df_sp500['Name']
except Exception as e:
    print(f"Error loading S&P 500 list: {e}. Using a placeholder.")
    data = {'Ticker': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
            'Name': ['Apple Inc.', 'Microsoft Corp.', 'Alphabet Inc. (C)', 'Amazon.com Inc.', 'Tesla Inc.'],
            'Sector': ['Information Technology', 'Information Technology', 'Communication Services', 'Consumer Discretionary', 'Consumer Discretionary']}
    df_sp500 = pd.DataFrame(data)
    df_sp500['label'] = df_sp500['Ticker'] + " - " + df_sp500['Name']

# --- App Initialization (Same as before) ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True)
server = app.server

# --- Helper Functions for Collapsible Sections ---
def make_collapsible_section(title, content_children, section_index, start_open=True):
    """
    Creates a collapsible section.
    section_index will be used to create unique dictionary IDs.
    """
    return html.Div([
        html.Div([
            title,
            html.Span("▲" if start_open else "▼", id={'type': 'arrow-generic', 'index': section_index}, className="float-end", style={'cursor': 'pointer'})
        ], id={'type': 'header-generic', 'index': section_index}, className="collapsible-header", style={'cursor': 'pointer', 'padding': '0.5rem 0.25rem', 'borderBottom': '1px solid #ddd', 'fontWeight': 'bold', 'fontSize': '0.9rem'}),
        dbc.Collapse(
            html.Div(children=content_children, style={'padding': '0.5rem 0.25rem', 'fontSize': '0.85rem'}),
            id={'type': 'collapse-generic', 'index': section_index}, # This ID will be MATCHed
            is_open=start_open
        )
    ], className="mb-2")

# --- App Layout ---
app.layout = dbc.Container([
    # 1. Header (Same as before)
    dbc.Row([
        dbc.Col(html.H1("Riskwise Portfolio", style={'fontSize': '2rem', 'fontWeight': 'bold', 'marginBottom': '0'}), md=6),
        dbc.Col(
            dbc.Nav([
                dbc.NavLink("Build Portfolio", href="#", active="exact", style={'fontSize': '0.9rem'}),
                dbc.NavLink("Theory", href="#", style={'fontSize': '0.9rem'}),
                dbc.NavLink("About Us", href="#", style={'fontSize': '0.9rem'}),
            ], pills=False, className="justify-content-end align-items-center h-100")
        , md=6)
    ], className="py-3 mb-3", style={'borderBottom': '1px solid #eee'}),

    # 2. Main Content Area
    dbc.Row([
        # Left Column
        dbc.Col([
            html.H4("Portfolio Construction", style={'fontSize': '1.1rem', 'marginBottom':'0.2rem'}),
            html.Div([ "S&P 500", html.Span("▲", className="float-end") ], style={'padding': '0.5rem 0.25rem', 'border': '1px solid #ccc', 'borderRadius': '4px', 'marginBottom': '10px', 'fontSize': '0.9rem', 'background': '#f8f9fa'}),
            dcc.Dropdown( id='asset-selector', options=[{'label': row['label'], 'value': row['Ticker']} for _, row in df_sp500.sort_values('Ticker').iterrows()], multi=True, placeholder="Select stocks...", className="mb-2"),
            dbc.Button("Add", id='add-analyze-button', color="secondary", size="sm", className="mb-2 w-100"),
            html.Div([ dbc.Table([ html.Thead(html.Tr([ html.Th("Stock", style={'fontSize': '0.85rem'}), html.Th("Weight (%)", style={'fontSize': '0.85rem', 'width':'100px'}), html.Th("Sector", style={'fontSize': '0.85rem'}) ])), html.Tbody(id='portfolio-table-body') ], bordered=True, striped=True, hover=True, size="sm") ], id='portfolio-table-container', className="mb-3", style={'maxHeight': '200px', 'overflowY': 'auto'}),
            html.H4("Risk Analysis", style={'fontSize': '1.1rem', 'marginTop': '20px'}),
            
            make_collapsible_section("Descriptive Stats", html.Div(id='content-desc-stats'), 'desc-stats'),
            make_collapsible_section("Parametric VaR & ES", html.Div(id='content-param-var'), 'param-var'),
            make_collapsible_section("Historical VaR & ES", html.Div(id='content-hist-var'), 'hist-var'),
            make_collapsible_section("EVT", html.Div(id='content-evt'), 'evt'),
            html.Div(id='evt-charts-placeholder', style={'display':'none'}),
            make_collapsible_section("Risk Concentration", html.Div(id='content-risk-conc'), 'risk-conc'),
            
            html.Div([ html.H5("Alerts", style={'fontSize': '1rem', 'borderBottom': '1px solid #eee', 'paddingBottom':'0.2rem'}), html.Div(id="alerts-content", style={'padding': '0.5rem 0', 'fontSize': '0.85rem'}) ], className="mt-3")
        ], md=6),

        # Right Column
        dbc.Col([
            html.H4("Portfolio Visualization", style={'fontSize': '1.1rem'}),
            dcc.Graph(id='pie-chart', config={'displayModeBar': False}, style={'height': '300px'}),
            
            make_collapsible_section("Threshold", 
                                     dcc.Slider(id='evt-threshold-slider', min=0.85, max=0.99, step=0.01, value=0.95, marks={i/100: {'label': str(i/100), 'style': {'fontSize':'0.7rem'}} for i in range(85, 100, 2)}, tooltip={"placement": "bottom", "always_visible": False}),
                                     'threshold'),
            make_collapsible_section("Interactive Tail Plot", html.Div(id='content-tail-plot'), 'tail-plot'),
            
            # Copula Section - keeping its unique ID structure for now, handled by a separate callback
            html.Div([
                html.Div([ "Copula Modeling Simulation", html.Span("▲", id='arrow-copula-section', className="float-end", style={'cursor': 'pointer'}) ], id='header-copula-section', className="collapsible-header", style={'cursor': 'pointer', 'padding': '0.5rem 0.25rem', 'borderBottom': '1px solid #ddd', 'fontWeight': 'bold', 'fontSize': '0.9rem'}),
                dbc.Collapse( html.Div([ html.Ul([ html.Li("Transform marginals to uniform"), html.Li("Fit Gaussian or Student-t copula (Current: Gaussian)"), html.Li("Simulate joint returns"), html.Li("Aggregate returns by portfolio weights"), html.Li("Estimate VaR / ES from simulated results") ], style={'paddingLeft': '20px', 'margin':0, 'fontSize': '0.85rem'}), html.Div(id="copula-results-display", className="mt-2", style={'fontSize': '0.85rem'}) ], style={'padding': '0.5rem 0.25rem'}), id='collapse-copula-section', is_open=True )
            ], className="mb-2"),
        ], md=6),
    ]),
    dcc.Store(id='analysis-results-store'),
    html.Div(id='placeholder-output', style={'display':'none'})
], fluid=True, className="py-2 px-3")


# --- Helper Functions for Analysis (Same as before) ---
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
            if isinstance(df_data, pd.Series): 
                if df_data.name in tickers:
                    df_data = df_data.to_frame(name=df_data.name)
                else: 
                    return pd.DataFrame(), pd.Series(dtype=float), "Unexpected data format for single successful ticker."
            
            missing_tickers = [t for t in tickers if t not in df_data.columns]
            if missing_tickers:
                for mt in missing_tickers:
                    try:
                        mt_data = yf.download(mt, period=period, auto_adjust=auto_adjust, progress=False)['Close']
                        if not mt_data.empty:
                            df_data[mt] = mt_data
                    except Exception:
                        pass 
            df_data = df_data.reindex(columns=tickers) 

        df_data = df_data.dropna(how='all') 
        returns = df_data.pct_change().dropna(how='any') 

        if returns.empty or len(returns) < 20: 
            return pd.DataFrame(), pd.Series(dtype=float), "Not enough historical data after processing for reliable analysis (min 20 days)."
        return returns, None, None
    except Exception as e:
        return pd.DataFrame(), pd.Series(dtype=float), f"Error fetching or processing data: {str(e)}"

def calculate_basic_stats(portfolio_returns):
    mu = portfolio_returns.mean()
    sigma = portfolio_returns.std()
    skewness_val = skew(portfolio_returns)
    kurtosis_val = kurtosis(portfolio_returns) 
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
    if len(portfolio_returns) < 50:
        return np.nan, np.nan, None, pd.Series(dtype=float), "Insufficient data for EVT (min 50 days)."
    losses = -portfolio_returns 
    if losses.empty or threshold_quantile is None:
        return np.nan, np.nan, None, pd.Series(dtype=float), "Losses series is empty or threshold is None."
    try: # Added try-except for quantile calculation
        threshold_val_loss = losses.quantile(threshold_quantile) 
    except Exception as e:
        return np.nan, np.nan, None, pd.Series(dtype=float), f"Error calculating quantile for EVT: {e}"

    excesses = losses[losses > threshold_val_loss] - threshold_val_loss
    if len(excesses) < 20:
        return np.nan, np.nan, None, excesses, f"Insufficient excesses ({len(excesses)}) for GPD fit at {threshold_quantile*100:.0f}th percentile of losses. Try lower threshold."
    try:
        params_gpd = genpareto.fit(excesses, floc=0) 
        shape_gpd, _, scale_gpd = params_gpd
        prob_exceed_u = (1.0 - threshold_quantile) 
        tail_prob_var = 1.0 - conf_level      
        if shape_gpd == 0: 
            evt_var_loss = threshold_val_loss + scale_gpd * np.log(prob_exceed_u / tail_prob_var)
        else:
            evt_var_loss = threshold_val_loss + (scale_gpd / shape_gpd) * (((prob_exceed_u / tail_prob_var))**(shape_gpd) - 1)
        if shape_gpd < 1:
            evt_es_loss = evt_var_loss + (scale_gpd + shape_gpd * (evt_var_loss - threshold_val_loss)) / (1 - shape_gpd)
        else:
            evt_es_loss = np.nan 
        return -evt_var_loss, -evt_es_loss, params_gpd, excesses, None
    except Exception as e:
        return np.nan, np.nan, None, excesses, f"EVT GPD fit error: {str(e)}"

def calculate_copula_risk(asset_returns, weights, n_simulations=1000, conf_level=0.95):
    if asset_returns.empty or asset_returns.shape[1] == 0:
        return np.nan, np.nan, "Asset returns are empty for copula."
    if len(weights) != asset_returns.shape[1]:
        return np.nan, np.nan, "Mismatch between number of assets and weights for copula."
    u_empirical = asset_returns.rank(axis=0, pct=True)
    copula_model = GaussianMultivariate() 
    try:
        copula_model.fit(u_empirical.to_numpy()) 
        sim_uniform = copula_model.sample(n_simulations) 
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
        return pd.DataFrame(columns=['Sector', 'Weight (%)'])
    asset_to_sector = df_sp500_ref.set_index('Ticker')['Sector']
    sector_allocations = {}
    for ticker, weight in zip(tickers, weights):
        sector = asset_to_sector.get(ticker, "Unknown Sector")
        sector_allocations[sector] = sector_allocations.get(sector, 0) + (weight * 100) 
    return pd.DataFrame(list(sector_allocations.items()), columns=['Sector', 'Weight (%)']).sort_values(by='Weight (%)', ascending=False)

def generate_feedback_messages(results, sector_allocations_df):
    feedback = []
    if results.get('error'):
        feedback.append(html.Li(f"Error during analysis: {results['error']}"))
        return html.Ul(feedback, style={'listStyleType': 'disc', 'paddingLeft': '20px'})
    if not np.isnan(results.get('var_parametric', np.nan)) and not np.isnan(results.get('var_historical', np.nan)) and results['var_parametric'] > results['var_historical']:
        feedback.append(html.Li("Parametric VaR (Normal assumption) may underestimate tail risk compared to Historical VaR if returns are not normal."))
    if not np.isnan(results.get('copula_var', np.nan)) and not np.isnan(results.get('var_historical', np.nan)) and results['copula_var'] < results['var_historical']:
        feedback.append(html.Li("Copula-based VaR suggests potential tail losses could be more severe than simple Historical VaR, indicating complex dependencies."))
    if not sector_allocations_df.empty:
        max_sector_weight = sector_allocations_df['Weight (%)'].iloc[0]
        max_sector_name = sector_allocations_df['Sector'].iloc[0]
        if max_sector_weight > 50: 
            feedback.append(html.Li(f"High concentration: {max_sector_weight:.1f}% of portfolio in '{max_sector_name}' sector."))
    if 'weights_final' in results and len(results['weights_final']) > 0 :
        max_asset_weight_fraction = np.max(results['weights_final'])
        if max_asset_weight_fraction > 0.4:
             feedback.append(html.Li(f"Asset concentration: '{results['tickers_final'][np.argmax(results['weights_final'])]}' has {max_asset_weight_fraction*100:.1f}% weight."))
    if not np.isnan(results.get('skewness', np.nan)):
        if results['skewness'] < -0.5:
            feedback.append(html.Li(f"Portfolio has significant negative skewness ({results['skewness']:.2f}), indicating a longer left tail (more frequent large losses)."))
    if not np.isnan(results.get('kurtosis', np.nan)): 
        if results['kurtosis'] > 1: 
            feedback.append(html.Li(f"High Kurtosis ({results['kurtosis']:.2f}): Portfolio returns have fatter tails than normal, suggesting higher probability of extreme outcomes."))
    if results.get('gpd_params'):
        shape_gpd = results['gpd_params'][0]
        if not np.isnan(shape_gpd):
            feedback.append(html.Li(f"EVT GPD Fit (Losses): Shape parameter (ξ) is {shape_gpd:.3f}."))
            if shape_gpd > 0:
                feedback.append(html.Li("Positive GPD shape (ξ > 0) indicates fat tails (heavier than exponential/normal), common for financial losses."))
    if not feedback:
        feedback.append(html.Li("Analysis complete. Review metrics for insights."))
    return html.Ul(feedback, style={'listStyleType': 'disc', 'paddingLeft': '20px', 'margin':0})

def plot_tail_distribution_graph_new(losses, threshold_val_loss, excesses, gpd_params):
    fig = go.Figure()
    if excesses is not None and len(excesses) > 0 and gpd_params is not None and not np.isnan(threshold_val_loss):
        shape_gpd, _, scale_gpd = gpd_params 
        fig.add_trace(go.Histogram(x=excesses, name='Excess Losses over Threshold', nbinsx=30, histnorm='probability density'))
        x_gpd = np.linspace(excesses.min(), excesses.max(), 200)
        pdf_gpd = genpareto.pdf(x_gpd, c=shape_gpd, loc=0, scale=scale_gpd)
        fig.add_trace(go.Scatter(x=x_gpd, y=pdf_gpd, mode='lines', name='Fitted GPD PDF', line=dict(color='red')))
        fig.update_layout(
            title_text=f"EVT: Excess Losses over {threshold_val_loss:.2%} Threshold & GPD Fit",
            xaxis_title="Excess Loss (Actual Loss - Threshold Loss)",
            yaxis_title="Density",
            legend_title_text='Tail Data',
            margin=dict(l=20, r=20, t=40, b=20), font=dict(size=10), height=280
        )
    else:
        fig.update_layout(title_text="Tail Distribution Plot (Not Available)", xaxis_title="Data", yaxis_title="Density",
                          margin=dict(l=20, r=20, t=40, b=20), font=dict(size=10), height=280)
        fig.add_annotation(text="Insufficient data or error for tail plot.",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    return fig

# --- Callbacks ---
@app.callback(
    Output('portfolio-table-body', 'children'),
    Input('asset-selector', 'value'),
    prevent_initial_call=True 
)
def generate_portfolio_table_rows(selected_tickers):
    if not selected_tickers: return []
    rows = []
    selected_df = df_sp500[df_sp500['Ticker'].isin(selected_tickers)].set_index('Ticker')
    for ticker in selected_tickers:
        sector = selected_df.loc[ticker, 'Sector'] if ticker in selected_df.index else "N/A"
        rows.append(html.Tr([ html.Td(ticker), html.Td(dcc.Input( id={'type': 'weight-input', 'index': ticker}, type='number', min=0, max=100, step=1, placeholder="e.g. 25", className="form-control form-control-sm", style={'width': '70px', 'padding': '0.2rem 0.4rem', 'fontSize': '0.8rem', 'textAlign':'right'})), html.Td(sector) ], key=ticker))
    return rows

# UPDATED Callback for toggling generic collapsible sections using dictionary IDs
@app.callback(
    [Output({'type': 'collapse-generic', 'index': MATCH}, 'is_open'),
     Output({'type': 'arrow-generic', 'index': MATCH}, 'children')],
    [Input({'type': 'header-generic', 'index': MATCH}, 'n_clicks')],
    [State({'type': 'collapse-generic', 'index': MATCH}, 'is_open')],
    prevent_initial_call=True
)
def toggle_generic_section_collapse(n_clicks, is_open):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    new_is_open = not is_open
    arrow = "▲" if new_is_open else "▼"
    return new_is_open, arrow

# Specific callback for Copula section (remains the same if its ID structure is different)
@app.callback(
    [Output('collapse-copula-section', 'is_open'),
     Output('arrow-copula-section', 'children')],
    [Input('header-copula-section', 'n_clicks')],
    [State('collapse-copula-section', 'is_open')],
    prevent_initial_call=True
)
def toggle_copula_section_collapse(n_clicks, is_open):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    new_is_open = not is_open
    arrow = "▲" if new_is_open else "▼"
    return new_is_open, arrow

@app.callback(
    [Output('analysis-results-store', 'data'),
     Output('pie-chart', 'figure'),
     Output('content-desc-stats', 'children'), 
     Output('content-param-var', 'children'),
     Output('content-hist-var', 'children'),
     Output('content-evt', 'children'),
     Output('content-risk-conc', 'children'),
     Output('alerts-content', 'children'),
     Output('content-tail-plot', 'children'), 
     Output('copula-results-display', 'children')
     ],
    [Input('add-analyze-button', 'n_clicks')],
    [State('asset-selector', 'value'),
     State({'type': 'weight-input', 'index': dash.dependencies.ALL}, 'value'),
     State({'type': 'weight-input', 'index': dash.dependencies.ALL}, 'id'), 
     State('evt-threshold-slider', 'value')]
)
def master_run_analysis(n_clicks, selected_tickers_list, weight_values_list, weight_ids_list, evt_threshold_quantile):
    ctx = dash.callback_context
    if not ctx.triggered or n_clicks is None or not selected_tickers_list:
        empty_pie_fig = go.Figure().update_layout(title_text=None, margin=dict(l=20, r=20, t=10, b=10), font=dict(size=10), height=300)
        no_data_msg = "Select assets, enter weights (ensure sum > 0), and click 'Add'."
        no_data_html = html.P(no_data_msg, style={'textAlign': 'center', 'color': 'grey'})
        empty_tail_plot_graph_component = dcc.Graph(figure=go.Figure().update_layout(title_text="Interactive Tail Plot", margin=dict(l=20, r=20, t=40, b=20), font=dict(size=10), height=280), config={'displayModeBar': False})
        alerts_initial_msg = html.Ul([html.Li("No analysis run yet.")]) if not ctx.triggered else ""
        return (dash.no_update, empty_pie_fig, no_data_html, no_data_html, no_data_html, no_data_html, no_data_html, alerts_initial_msg, empty_tail_plot_graph_component, "")

    if evt_threshold_quantile is None:
        error_msg = "EVT threshold slider value is missing. Please ensure it's in the layout."
        err_html = html.Ul([html.Li(error_msg)])
        empty_pie_fig = go.Figure().update_layout(title_text=None, margin=dict(l=20, r=20, t=10, b=10), font=dict(size=10), height=300)
        empty_tail_plot_graph_component = dcc.Graph(figure=go.Figure().update_layout(title_text="Interactive Tail Plot", margin=dict(l=20, r=20, t=40, b=20), font=dict(size=10), height=280), config={'displayModeBar': False})
        return (dash.no_update, empty_pie_fig, err_html, err_html, err_html, err_html, err_html, err_html, empty_tail_plot_graph_component, "")

    valid_tickers_weights = []
    temp_weights_dict = {comp_id['index']: val for comp_id, val in zip(weight_ids_list, weight_values_list)}
    for ticker in selected_tickers_list:
        weight_val = temp_weights_dict.get(ticker)
        if weight_val is not None:
            try:
                w_pct = float(weight_val)
                if w_pct < 0: error_msg = f"Weight for {ticker} cannot be negative."
                elif w_pct > 0: valid_tickers_weights.append((ticker, w_pct / 100.0)); continue
                else: continue # Weight is 0, skip
                err_html = html.Ul([html.Li(error_msg)])
                return (None, dash.no_update, err_html, err_html, err_html, err_html, err_html, err_html, dash.no_update, "")
            except ValueError:
                error_msg = f"Invalid weight for {ticker}. Please enter a number."
                err_html = html.Ul([html.Li(error_msg)])
                return (None, dash.no_update, err_html, err_html, err_html, err_html, err_html, err_html, dash.no_update, "")

    if not valid_tickers_weights:
        error_msg = "Please assign non-zero weights to at least one asset."
        err_html = html.Ul([html.Li(error_msg)])
        return (None, dash.no_update, err_html, err_html, err_html, err_html, err_html, err_html, dash.no_update, "")

    tickers_final = [item[0] for item in valid_tickers_weights]
    weights_final_raw = np.array([item[1] for item in valid_tickers_weights], dtype=float)
    total_weight_sum = weights_final_raw.sum()
    if total_weight_sum == 0:
        error_msg = "Sum of weights cannot be zero. Please ensure weights are positive."
        err_html = html.Ul([html.Li(error_msg)])
        return (None, dash.no_update, err_html, err_html, err_html, err_html, err_html, err_html, dash.no_update, "")
    weights_final = weights_final_raw / total_weight_sum

    asset_returns_df, _, data_error_msg = get_portfolio_data(tickers_final)
    if data_error_msg:
        err_html = html.Ul([html.Li(data_error_msg)])
        return (None, dash.no_update, err_html, err_html, err_html, err_html, err_html, err_html, dash.no_update, "")
    
    if asset_returns_df.shape[1] != len(weights_final):
        mismatch_error = "Mismatch: fetched asset returns columns do not match number of weights. Some assets might have no data."
        err_html = html.Ul([html.Li(mismatch_error)])
        return (None, dash.no_update, err_html, err_html, err_html, err_html, err_html, err_html, dash.no_update, "")

    portfolio_returns_series = asset_returns_df.dot(weights_final)
    results_dict = {'tickers_final': tickers_final, 'weights_final': weights_final.tolist()}
    try:
        results_dict['mu'], results_dict['sigma'], results_dict['skewness'], results_dict['kurtosis'] = calculate_basic_stats(portfolio_returns_series)
        results_dict['var_parametric'], results_dict['es_parametric'] = calculate_parametric_risk(results_dict['mu'], results_dict['sigma'])
        results_dict['var_historical'], results_dict['es_historical'] = calculate_historical_risk(portfolio_returns_series)
        results_dict['evt_var'], results_dict['evt_es'], results_dict['gpd_params'], excesses_for_plot, evt_error_msg = calculate_evt_risk(portfolio_returns_series, evt_threshold_quantile)
        if evt_error_msg: results_dict['evt_error'] = evt_error_msg
        results_dict['copula_var'], results_dict['copula_es'], copula_error_msg = calculate_copula_risk(asset_returns_df, weights_final)
        if copula_error_msg: results_dict['copula_error'] = copula_error_msg
    except Exception as e:
        calc_error_msg = f"Error during risk calculation: {str(e)}"
        results_dict['error'] = calc_error_msg
        err_html = html.Ul([html.Li(calc_error_msg)])
        return (results_dict, dash.no_update, err_html, err_html, err_html, err_html, err_html, err_html, dash.no_update, "")

    sector_allocations_df = calculate_sector_concentration(tickers_final, weights_final, df_sp500)
    pie_fig = px.pie(values=weights_final * 100, names=tickers_final, title=None, hole=0.3)
    pie_fig.update_layout(margin=dict(l=20, r=20, t=10, b=10), font=dict(size=10), height=300, showlegend=True, legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
    pie_fig.update_traces(textposition='inside', textinfo='percent+label', insidetextorientation='radial')

    desc_stats_content = html.Ul([ html.Li(f"Mean Daily Return: {results_dict.get('mu', np.nan):.4%}"), html.Li(f"Daily Volatility: {results_dict.get('sigma', np.nan):.4%}"), html.Li(f"Skewness: {results_dict.get('skewness', np.nan):.4f}"), html.Li(f"Kurtosis (Fisher): {results_dict.get('kurtosis', np.nan):.4f}"), html.Li(f"Data points: {len(portfolio_returns_series)}") ], style={'listStyleType': 'none', 'paddingLeft': '0', 'margin':0})
    param_var_content = html.Ul([ html.Li(f"VaR (95%): {results_dict.get('var_parametric', np.nan):.4%}"), html.Li(f"ES (95%): {results_dict.get('es_parametric', np.nan):.4%}"), ], style={'listStyleType': 'none', 'paddingLeft': '0', 'margin':0})
    hist_var_content = html.Ul([ html.Li(f"VaR (95%): {results_dict.get('var_historical', np.nan):.4%}"), html.Li(f"ES (95%): {results_dict.get('es_historical', np.nan):.4%}"), ], style={'listStyleType': 'none', 'paddingLeft': '0', 'margin':0})
    evt_content_list = [ html.Li(f"VaR (95%): {results_dict.get('evt_var', np.nan):.4%}"), html.Li(f"ES (95%): {results_dict.get('evt_es', np.nan):.4%}"), ]
    if 'evt_error' in results_dict: evt_content_list.append(html.Li(f"Note: {results_dict['evt_error']}", style={'color':'orange'}))
    if results_dict.get('gpd_params'): evt_content_list.extend([html.Li(f"GPD Shape (ξ): {results_dict['gpd_params'][0]:.3f}"), html.Li(f"GPD Scale (σ): {results_dict['gpd_params'][2]:.4f}")])
    evt_content = html.Ul(evt_content_list, style={'listStyleType': 'none', 'paddingLeft': '0', 'margin':0})
    
    losses_series = -portfolio_returns_series
    evt_threshold_val_loss = losses_series.quantile(evt_threshold_quantile) if not losses_series.empty and evt_threshold_quantile is not None else np.nan
    tail_plot_fig = plot_tail_distribution_graph_new(losses_series, evt_threshold_val_loss, excesses_for_plot, results_dict.get('gpd_params'))
    tail_plot_content = dcc.Graph(figure=tail_plot_fig, config={'displayModeBar': False})

    risk_conc_content_list = [html.Li(f"{row['Sector']}: {row['Weight (%)']:.2f}%") for _, row in sector_allocations_df.iterrows()]
    risk_conc_content = html.Ul(risk_conc_content_list, style={'listStyleType': 'none', 'paddingLeft': '0', 'margin':0})

    copula_results_list = [ html.Li(f"VaR (95%): {results_dict.get('copula_var', np.nan):.4%}"), html.Li(f"ES (95%): {results_dict.get('copula_es', np.nan):.4%}"), ]
    if 'copula_error' in results_dict: copula_results_list.append(html.Li(f"Note: {results_dict['copula_error']}", style={'color':'orange'}))
    copula_results_text = html.Ul(copula_results_list, style={'listStyleType': 'none', 'paddingLeft': '0', 'margin':0, 'marginTop':'5px', 'borderTop':'1px solid #eee', 'paddingTop':'5px'})
    
    feedback_html = generate_feedback_messages(results_dict, sector_allocations_df)
    
    return (results_dict, pie_fig, desc_stats_content, param_var_content, hist_var_content, evt_content, risk_conc_content, feedback_html, tail_plot_content, copula_results_text)

# --- Run App ---
if __name__ == '__main__':
    app.run(debug=True)