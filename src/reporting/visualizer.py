"""Visualization module for creating charts and plots for analysis reports."""

# pylint: disable=import-outside-toplevel, wrong-import-position, use-dict-literal
# Note: use-dict-literal disabled because dict() is more readable in Plotly code.
import logging
import os

import matplotlib

matplotlib.use("Agg")  # Set backend to non-interactive to avoid threading issues
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

logger = logging.getLogger(__name__)


class Visualizer:
    """Creates static and interactive visualizations for trading strategy analysis."""

    def __init__(self, output_dir="reports"):
        self.output_dir = output_dir
        # Use exist_ok=True to avoid TOCTOU race condition
        os.makedirs(output_dir, exist_ok=True)

    def plot_forecast(self, strategy_data, forecast_values, strategy_id, method_name):
        """Plot historical profit and forecasted profit."""
        plt.figure(figsize=(12, 6))

        # Historical
        plt.plot(
            strategy_data["Date"],
            strategy_data["Profit"],
            label="Historical Profit",
            color="blue",
        )

        # Forecast - We need dates for forecast. Assuming weekly steps.
        last_date = strategy_data["Date"].iloc[-1]

        # Infer frequency from data
        if len(strategy_data) > 2:
            freq = pd.infer_freq(strategy_data["Date"])
            if freq is None:
                # Fallback: Calculate median time difference
                freq = strategy_data["Date"].diff().median()
        else:
            freq = pd.Timedelta(weeks=1)  # Default fallback

        # Generate forecast dates
        if isinstance(freq, str):
            forecast_dates = pd.date_range(
                start=last_date, periods=len(forecast_values) + 1, freq=freq
            )[1:]
        else:
            forecast_dates = [last_date + (freq * (i + 1)) for i in range(len(forecast_values))]

        plt.plot(
            forecast_dates,
            forecast_values,
            label=f"Forecast ({method_name})",
            color="red",
            linestyle="--",
        )

        plt.title(f"Strategy {strategy_id} Performance & Forecast")
        plt.xlabel("Date")
        plt.ylabel("Profit")
        plt.legend()
        plt.grid(True)

        filename = os.path.join(self.output_dir, f"strategy_{strategy_id}_forecast.png")
        plt.savefig(filename)
        plt.close()
        return filename

    def plot_comparison(self, results_df):
        """
        Bar chart of top 10 strategies by 1M forecast.
        """
        plt.figure(figsize=(12, 6))
        top_10 = results_df.head(10)
        # Use explicit x/y without hue to avoid seaborn deprecation warning
        _ax = sns.barplot(x="No.", y="Forecast_1M", data=top_10, palette="viridis")
        plt.title("Top 10 Strategies Forecast (1 Month)")
        plt.xlabel("Strategy No.")
        plt.ylabel("Forecasted Profit")

        filename = os.path.join(self.output_dir, "top_strategies_comparison.png")
        plt.savefig(filename)
        plt.close()
        return filename

    def plot_monthly_heatmap(self, pivot_table, strategy_id):
        """
        Plots a heatmap of monthly returns.
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap="RdYlGn", center=0)
        plt.title(f"Strategy {strategy_id} Monthly Performance")

        filename = os.path.join(self.output_dir, f"strategy_{strategy_id}_heatmap.png")
        plt.savefig(filename)
        plt.close()
        return filename

    def plot_drawdown_curve(self, strategy_data, strategy_id):
        """
        Plots the drawdown curve.
        """
        plt.figure(figsize=(12, 6))

        # Validate data before calculation
        if strategy_data is None or strategy_data.empty or "Profit" not in strategy_data.columns:
            plt.text(
                0.5,
                0.5,
                "Insufficient data for drawdown calculation",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
            filename = os.path.join(self.output_dir, f"strategy_{strategy_id}_drawdown.png")
            plt.savefig(filename)
            plt.close()
            return filename

        # Calculate Drawdown
        equity = strategy_data["Profit"].cumsum()
        running_max = equity.cummax()
        drawdown = equity - running_max

        plt.fill_between(strategy_data["Date"], drawdown, 0, color="red", alpha=0.3)
        plt.plot(strategy_data["Date"], drawdown, color="red", linewidth=1)

        plt.title(f"Strategy {strategy_id} Drawdown Curve")
        plt.xlabel("Date")
        plt.ylabel("Drawdown ($)")
        plt.grid(True)

        filename = os.path.join(self.output_dir, f"strategy_{strategy_id}_drawdown.png")
        plt.savefig(filename)
        plt.close()
        return filename

    def create_interactive_forecast(self, strategy_data, forecast_values, strategy_id, method_name):
        """
        Creates an interactive Plotly forecast chart.
        """
        import plotly.graph_objects as go

        last_date = strategy_data["Date"].iloc[-1]

        # Infer frequency from data
        if len(strategy_data) > 2:
            freq = pd.infer_freq(strategy_data["Date"])
            if freq is None:
                # Fallback: Calculate median time difference
                freq = strategy_data["Date"].diff().median()
        else:
            freq = pd.Timedelta(weeks=1)  # Default fallback

        # Generate forecast dates
        if isinstance(freq, str):
            forecast_dates = pd.date_range(
                start=last_date, periods=len(forecast_values) + 1, freq=freq
            )[1:]
        else:
            forecast_dates = [last_date + (freq * (i + 1)) for i in range(len(forecast_values))]

        fig = go.Figure()

        # Historical
        fig.add_trace(
            go.Scatter(
                x=strategy_data["Date"],
                y=strategy_data["Profit"],
                mode="lines",
                name="Historical Profit",
                line=dict(color="blue"),
            )
        )

        # Forecast
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=forecast_values,
                mode="lines+markers",
                name=f"Forecast ({method_name})",
                line=dict(color="red", dash="dash"),
            )
        )

        fig.update_layout(
            title=f"Strategy {strategy_id} Interactive Forecast",
            xaxis_title="Date",
            yaxis_title="Profit",
            template="plotly_white",
            hovermode="x unified",
        )
        return fig

    def create_interactive_drawdown(self, strategy_data, strategy_id):
        """Create an interactive Plotly drawdown chart."""
        import plotly.graph_objects as go

        equity = strategy_data["Profit"].cumsum()
        running_max = equity.cummax()
        drawdown = equity - running_max

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=strategy_data["Date"],
                y=drawdown,
                fill="tozeroy",
                mode="lines",
                name="Drawdown",
                line=dict(color="red"),
            )
        )

        fig.update_layout(
            title=f"Strategy {strategy_id} Interactive Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown ($)",
            template="plotly_white",
            hovermode="x unified",
        )
        return fig

    def create_interactive_heatmap(self, pivot_table, strategy_id):
        """
        Creates an interactive Plotly heatmap.
        """
        import plotly.express as px

        # Calculate symmetric range to ensure 0 is center (Yellow/Green/Red)
        max_val = max(abs(pivot_table.min().min()), abs(pivot_table.max().max()))

        fig = px.imshow(
            pivot_table,
            text_auto=".0f",
            aspect="auto",
            color_continuous_scale="RdYlGn",
            range_color=[-max_val, max_val],
            title=f"Strategy {strategy_id} Monthly Performance",
        )
        fig.update_layout(template="plotly_white")
        return fig

    def plot_returns_distribution(self, strategy_data, strategy_id):
        """
        Plots the distribution of returns (Histogram and KDE).
        """
        plt.figure(figsize=(10, 6))

        # Calculate returns if not present
        strategy_data = strategy_data.copy()
        if "Returns" not in strategy_data.columns:
            strategy_data["Returns"] = strategy_data["Profit"].pct_change().fillna(0)

        sns.histplot(strategy_data["Returns"], kde=True, bins=30, color="purple")
        plt.title(f"Strategy {strategy_id} Returns Distribution")
        plt.xlabel("Returns")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)

        filename = os.path.join(self.output_dir, f"strategy_{strategy_id}_returns_dist.png")
        plt.savefig(filename)
        plt.close()
        return filename

    def plot_rolling_metrics(self, strategy_data, strategy_id, window=30):
        """
        Plots rolling volatility and Sharpe ratio.
        """
        if "Returns" not in strategy_data.columns:
            strategy_data = strategy_data.copy()
            strategy_data["Returns"] = strategy_data["Profit"].pct_change().fillna(0)

        rolling_vol = strategy_data["Returns"].rolling(window=window).std()
        # Simple Sharpe: Mean / Std (assuming 0 risk-free rate for simplicity in this context)
        rolling_sharpe = strategy_data["Returns"].rolling(window=window).mean() / (
            rolling_vol + 1e-9
        )

        _fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        ax1.plot(
            strategy_data["Date"],
            rolling_vol,
            color="orange",
            label=f"{window}-Period Rolling Volatility",
        )
        ax1.set_title(f"Strategy {strategy_id} Rolling Volatility")
        ax1.set_ylabel("Volatility")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(
            strategy_data["Date"],
            rolling_sharpe,
            color="green",
            label=f"{window}-Period Rolling Sharpe",
        )
        ax2.set_title(f"Strategy {strategy_id} Rolling Sharpe Ratio")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Sharpe Ratio")
        ax2.legend()
        ax2.grid(True)

        filename = os.path.join(self.output_dir, f"strategy_{strategy_id}_rolling_metrics.png")
        plt.savefig(filename)
        plt.close()
        return filename

    def plot_lag_plot(self, strategy_data, strategy_id, lag=1):
        """
        Plots a lag plot to check for autocorrelation.
        """
        plt.figure(figsize=(8, 8))
        pd.plotting.lag_plot(strategy_data["Profit"], lag=lag)
        plt.title(f"Strategy {strategy_id} Lag Plot (Lag={lag})")
        plt.grid(True)

        filename = os.path.join(self.output_dir, f"strategy_{strategy_id}_lag_plot.png")
        plt.savefig(filename)
        plt.close()
        return filename

    def plot_decomposition(self, strategy_data, strategy_id, period=52):
        """
        Plots STL decomposition (Trend, Seasonal, Residual).
        """
        from statsmodels.tsa.seasonal import seasonal_decompose

        # Ensure we have enough data and no NaNs
        series = strategy_data["Profit"].dropna()
        if len(series) < period * 2:
            return None

        try:
            result = seasonal_decompose(
                series, model="additive", period=period, extrapolate_trend="freq"
            )

            _fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

            ax1.plot(series.index, result.observed)
            ax1.set_ylabel("Observed")
            ax1.set_title(f"Strategy {strategy_id} Decomposition")

            ax2.plot(series.index, result.trend)
            ax2.set_ylabel("Trend")

            ax3.plot(series.index, result.seasonal)
            ax3.set_ylabel("Seasonal")

            ax4.plot(series.index, result.resid)
            ax4.set_ylabel("Residual")
            ax4.set_xlabel("Index")

            plt.tight_layout()

            filename = os.path.join(self.output_dir, f"strategy_{strategy_id}_decomposition.png")
            plt.savefig(filename)
            plt.close()
            return filename
        except (ValueError, AttributeError) as e:
            logger.debug("Decomposition failed: %s", e)
            return None

    def create_interactive_returns_distribution(self, strategy_data, strategy_id):
        """
        Creates an interactive Plotly returns distribution chart.
        """
        import plotly.express as px

        strategy_data = strategy_data.copy()
        if "Returns" not in strategy_data.columns:
            strategy_data["Returns"] = strategy_data["Profit"].pct_change().fillna(0)

        fig = px.histogram(
            strategy_data,
            x="Returns",
            nbins=50,
            marginal="box",
            title=f"Strategy {strategy_id} Returns Distribution",
            color_discrete_sequence=["purple"],
        )
        fig.update_layout(template="plotly_white")
        return fig

    def create_interactive_rolling_metrics(self, strategy_data, strategy_id, window=30):
        """
        Creates an interactive Plotly rolling metrics chart.
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        strategy_data = strategy_data.copy()
        if "Returns" not in strategy_data.columns:
            strategy_data["Returns"] = strategy_data["Profit"].pct_change().fillna(0)

        rolling_vol = strategy_data["Returns"].rolling(window=window).std()
        rolling_sharpe = strategy_data["Returns"].rolling(window=window).mean() / (
            rolling_vol + 1e-9
        )

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Rolling Volatility", "Rolling Sharpe Ratio"),
        )

        fig.add_trace(
            go.Scatter(
                x=strategy_data["Date"],
                y=rolling_vol,
                name="Volatility",
                line=dict(color="orange"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=strategy_data["Date"],
                y=rolling_sharpe,
                name="Sharpe Ratio",
                line=dict(color="green"),
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            title_text=f"Strategy {strategy_id} Rolling Metrics ({window}-period)",
            template="plotly_white",
        )
        return fig

    def create_interactive_lag_plot(self, strategy_data, strategy_id, lag=1):
        """
        Creates an interactive Plotly lag plot.
        """
        import plotly.graph_objects as go

        series = strategy_data["Profit"]

        # Validate series length before lag operation to avoid IndexError
        if len(series) <= lag:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Insufficient data for lag={lag} (need >{lag} points, have {len(series)})",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            fig.update_layout(
                title=f"Strategy {strategy_id} Lag Plot (Lag={lag}) - Insufficient Data"
            )
            return fig

        y_t = series.iloc[lag:]
        y_t_lag = series.shift(lag).iloc[lag:]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=y_t_lag,
                y=y_t,
                mode="markers",
                name=f"Lag {lag}",
                marker=dict(color="blue", opacity=0.6),
            )
        )

        fig.update_layout(
            title=f"Strategy {strategy_id} Lag Plot (Lag={lag})",
            xaxis_title=f"Profit (t-{lag})",
            yaxis_title="Profit (t)",
            template="plotly_white",
        )
        return fig

    def create_interactive_decomposition(self, strategy_data, strategy_id, period=52):
        """
        Creates an interactive Plotly decomposition chart.
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from statsmodels.tsa.seasonal import seasonal_decompose

        series = strategy_data["Profit"].dropna()
        # Need dates for x-axis
        dates = strategy_data["Date"].loc[series.index]

        if len(series) < period * 2:
            return None

        try:
            result = seasonal_decompose(
                series, model="additive", period=period, extrapolate_trend="freq"
            )

            fig = make_subplots(
                rows=4,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
            )

            fig.add_trace(
                go.Scatter(
                    x=dates, y=result.observed, name="Observed", line=dict(color="blue")
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=dates, y=result.trend, name="Trend", line=dict(color="red")),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=dates, y=result.seasonal, name="Seasonal", line=dict(color="green")
                ),
                row=3,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=result.resid,
                    name="Residual",
                    line=dict(color="gray"),
                    mode="markers",
                ),
                row=4,
                col=1,
            )

            fig.update_layout(
                title_text=f"Strategy {strategy_id} Decomposition", template="plotly_white"
            )
            return fig

        except (ValueError, AttributeError) as e:
            logger.debug("Interactive decomposition failed: %s", e)
            return None
