# Financial Indicators & System Parameters

This document outlines the key financial metrics, parameters, and data points extracted from external APIs (e.g., Alpha Vantage, Financial Modeling Prep). These parameters serve as the foundational context for the Multi-Agent System (MAS).

| Category | Parameter / Indicator | Description & Use Case in MAS | API Function / Endpoint |
| :--- | :--- | :--- | :--- |
| **Fundamental** | **Revenue & Net Income** | Top-line money vs. bottom-line profit. Used to assess growth trajectories. | `INCOME_STATEMENT` |
| **Fundamental** | **EPS (Earnings Per Share)** | Net profit divided by outstanding shares. Indicates per-share profitability. | `OVERVIEW` |
| **Fundamental** | **P/E Ratio** | Price-to-Earnings ratio. Calculates if a stock is overvalued or undervalued. | `OVERVIEW` |
| **Fundamental** | **P/B Ratio** | Price-to-Book ratio. Compares market valuation to book value; crucial for value agents. | `OVERVIEW` |
| **Fundamental** | **ROE (Return on Equity)** | Measures how efficiently the management generates income from shareholders' equity. | `OVERVIEW` |
| **Fundamental** | **Debt-to-Equity Ratio** | Highlights financial leverage and risk. High ratios trigger risk-management protocols. | `BALANCE_SHEET` (Requires calc) or FMP `key-metrics` |
| **Market & Price**| **OHLC (Open, High, Low, Close)**| Core components of daily price action. High/Low spread determines daily volatility. | `TIME_SERIES_DAILY` |
| **Market & Price**| **Volume** | Number of shares traded. Validates price breakouts, providing a "conviction" score. | `TIME_SERIES_DAILY` |
| **Market & Price**| **52-Week High/Low** | Highest and lowest price over the past year. Used to gauge momentum and support levels. | `OVERVIEW` |
| **Alternative** | **News Sentiment Score** | Quantitative representation of media tone regarding a ticker (Scale: `-0.35` to `+0.35`). | `NEWS_SENTIMENT` |
| **Alternative** | **Article Summaries** | Textual 1-2 sentence summaries of the news. Provides qualitative context for LLMs to read between the lines of the numerical score. | `NEWS_SENTIMENT` (Extracted from the `summary` field) |
| **Alternative** | **Insider Trading Activity**| Tracks if C-suite executives are buying/selling. Insider buying is a strong bullish signal. | FMP `insider-trading` |
| **Alternative** | **Analyst Consensus** | Aggregated ratings (Buy/Hold/Sell) and Target Prices from Wall Street institutions. | `OVERVIEW` or FMP `analyst-stock-recommendations` |
| **Macroeconomic** | **Federal Funds Rate** | Dictates the cost of borrowing. High rates generally pressure growth/tech stocks. | `FEDERAL_FUNDS_RATE` |
| **Macroeconomic** | **CPI (Inflation)** | Consumer Price Index. The standard measure of inflation. | `CPI` |
| **Macroeconomic** | **Real GDP Growth** | Indicates whether the broader economy is expanding or contracting. | `REAL_GDP` |