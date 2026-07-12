# ADR-0012: Use Lightweight Charts for Financial Visualization

## Status
Accepted

## Context
NGLab's core value proposition involves visualizing financial market data (candlesticks, order book depth, ticks) in real-time. Standard charting libraries (Chart.js, Recharts) are often built on SVG or lack the specialized features needed for financial plotting (e.g., time-scale handling, massive datasets).

## Decision
We will use **Lightweight Charts** (by TradingView) as our primary library for financial data visualization.
- **Use Case**: OHLCV candlesticks, simple line charts, and real-time tick updates.
- **Performance**: Canvas-based rendering optimized for high-frequency updates.

We will use **Highcharts** as a secondary library for general statistical visualization.
- **Use Case**: Histograms, complex statistical plots (e.g., distribution of returns), 3D surfaces (if needed).
- **Justification**: Lightweight Charts is specialized for "TradingView-style" charts and lacks general plotting capabilities.

## Consequences
- **Easier**:
    - **Professional Look**: Out-of-the-box financial aesthetics familiar to traders.
    - **Performance**: Can handle thousands of data points without UI lag.
- **Difficult**:
    - **Fragmentation**: Developers must learn two charting APIs.
    - **Customization**: Canvas-based charts are harder to style dynamically (CSS) than SVG-based ones.

## Alternatives Considered
- **D3.js**: Extremely flexible but requires building every chart type from scratch (axes, scales, interactions). Too high implementation cost.
- **Plotly.js**: Great for scientific plotting (Python ecosystem match), but the JS library is very heavy (large bundle size) and slower than canvas-based alternatives for real-time streaming.
