# Riskwise Portfolio Analyzer (V4)

Riskwise Portfolio Analyzer is an interactive web application built with Python and Dash for constructing and analyzing the risk of equity portfolios. Users can select stocks from the S&P 500, assign weights, and then perform a comprehensive risk assessment using various quantitative finance techniques.

## Overview

This tool allows users to:
1.  **Build a custom portfolio** by selecting S&P 500 stocks and specifying their weights.
2.  **Visualize portfolio allocation** through an interactive pie chart.
3.  **Analyze portfolio risk** using multiple methodologies, including:
    *   Descriptive Statistics (mean, volatility, skewness, kurtosis).
    *   Parametric Value-at-Risk (VaR) and Expected Shortfall (ES).
    *   Historical VaR and ES.
    *   Extreme Value Theory (EVT) for tail risk modeling (VaR, ES, GPD fit).
    *   Copula Modeling for simulating joint returns and estimating VaR/ES.
4.  **Assess risk concentration** by sector.
5.  Receive **contextual alerts and feedback** based on the analysis results.

The application features a user-friendly interface with collapsible sections to present detailed risk metrics and visualizations.

## Features

*   **Portfolio Construction:**
    *   Select multiple stocks from an S&P 500 list (data fetched from Wikipedia).
    *   Dynamically add selected stocks to a table.
    *   Input percentage weights for each selected stock in the table.
    *   "Add" button triggers portfolio construction and subsequent risk analysis.
*   **Portfolio Visualization:**
    *   Interactive pie chart displaying asset allocation by weight.
*   **Risk Analysis Modules (Collapsible Sections):**
    *   **Descriptive Stats:** Mean daily return, daily volatility, skewness, and kurtosis of portfolio returns.
    *   **Parametric VaR & ES:** Calculates Value-at-Risk and Expected Shortfall assuming a normal distribution of returns (95% confidence level).
    *   **Historical VaR & ES:** Calculates Value-at-Risk and Expected Shortfall based on historical simulation (95% confidence level).
    *   **Extreme Value Theory (EVT):**
        *   Calculates VaR and ES using the Generalized Pareto Distribution (GPD) fitted to the tail of portfolio losses.
        *   Displays GPD fit parameters (shape and scale).
        *   User-adjustable threshold quantile for GPD fitting (via a slider in the "Threshold" section).
    *   **Risk Concentration:** Shows the portfolio's allocation across different GICS sectors.
*   **Advanced Risk Modeling (Right Panel):**
    *   **Threshold:** Contains a slider to adjust the quantile for EVT analysis.
    *   **Interactive Tail Plot (EVT):** Visualizes the histogram of excess losses over the selected threshold and the fitted GPD probability density function.
    *   **Copula Modeling Simulation:**
        *   Displays the steps of the copula modeling process.
        *   Shows the calculated VaR and ES from the copula simulation.
*   **Alerts:** Provides qualitative feedback and potential warnings based on the calculated risk metrics (e.g., high concentration, significant skewness/kurtosis, comparison of different VaR models).
*   **Data Source:** Historical stock price data is fetched using `yfinance`. S&P 500 stock list is scraped from Wikipedia.

## Technologies Used

*   **Python 3.x**
*   **Dash:** Core framework for building the web application.
    *   **Dash Bootstrap Components:** For styling and layout.
    *   **Dash Core Components (dcc):** For interactive UI elements like dropdowns, sliders, graphs.
    *   **Dash HTML Components (html):** For structuring the app with HTML-like tags.
*   **Plotly:** For creating interactive charts and visualizations.
*   **Pandas:** For data manipulation and analysis.
*   **NumPy:** For numerical operations.
*   **SciPy:** For statistical functions (e.g., `norm`, `skew`, `kurtosis`, `genpareto` for GPD fitting).
*   **yfinance:** For downloading historical market data.
*   **Copulas:** For multivariate modeling of dependencies (specifically `GaussianMultivariate`).

## Setup and Installation

1.  **Clone the repository (if applicable) or download the `app.py` file.**
    ```bash
    # git clone <repository-url>
    # cd <repository-name>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    # venv\Scripts\activate
    # On macOS/Linux
    # source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    Create a `requirements.txt` file with the following content:
    ```
    dash
    dash-bootstrap-components
    pandas
    numpy
    scipy
    yfinance
    plotly
    copulas
    lxml # Often needed by pandas.read_html
    ```
    Then run:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the application:**
    Navigate to the directory containing `app.py` and execute:
    ```bash
    python app.py
    ```
2.  **Open your web browser** and go to `http://127.0.0.1:8050/` (or the address shown in your terminal).

3.  **Interact with the application:**
    *   **Portfolio Construction:**
        *   Use the "Select stocks..." dropdown to choose S&P 500 stocks.
        *   Selected stocks appear in the table below.
        *   Enter the desired percentage weight for each stock in the table's "Weight (%)" column.
        *   Click the "Add" button. This normalizes weights (if they don't sum to 100%) and triggers the risk analysis.
    *   **View Portfolio Allocation:** The pie chart (right panel) updates.
    *   **Explore Risk Analysis:**
        *   Results for different risk metrics populate the collapsible sections (left panel). Click headers to expand/collapse.
        *   Adjust the EVT threshold using the slider in the "Threshold" section (right panel). Click "Add" again to re-analyze with the new threshold.
        *   The "Interactive Tail Plot" (right panel) shows the GPD fit for EVT.
        *   The "Copula Modeling Simulation" section (right panel) displays process steps and results.
    *   **Check Alerts:** The "Alerts" section (bottom-left) provides qualitative feedback.

## File Structure

*   `app.py`: The main Python script containing the Dash application layout, callbacks, and analysis logic.
*   `README.md`: This file.
*   `requirements.txt` (to be created): Lists project dependencies.

## Disclaimer

This application is for educational and informational purposes only. It should not be considered financial advice. The risk metrics and analyses provided are based on historical data and specific models, which may not accurately predict future market behavior or risk. Always conduct your own thorough research or consult with a qualified financial advisor before making any investment decisions.