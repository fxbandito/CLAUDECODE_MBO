"""
Financial calculation metrics for strategy analysis.

This module provides the FinancialMetrics class which handles the calculation
of various trading performance metrics such as Profit, Win Rate, Sharpe Ratio,
and Drawdown from raw trading data.
"""

import numpy as np
import pandas as pd


class FinancialMetrics:
    """
    Utility class for calculating financial performance metrics.

    Provides static methods to compute comprehensive statistics and
    aggregated returns from trading dataframes.
    """

    @staticmethod
    def calculate_all(df):
        """
        Calculates a comprehensive set of financial metrics for a strategy dataframe.
        Expects 'Profit' and 'Date' columns.
        """
        if df is None or df.empty:
            return {}

        metrics = {}

        # Ensure Date is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
            df["Date"] = pd.to_datetime(df["Date"])

        # Sort by date
        df = df.sort_values("Date")

        # Basic Stats
        metrics["Total Profit"] = df["Profit"].sum()
        metrics["Total Trades"] = len(df)

        # Win Rate - Only count non-zero trades (exclude zero-profit trades from loss count)
        wins = df[df["Profit"] > 0]
        losses = df[df["Profit"] < 0]  # Strictly negative, not <= 0
        total_decisive = len(wins) + len(losses)
        metrics["Win Rate"] = len(wins) / total_decisive if total_decisive > 0 else 0

        # Profit Factor
        gross_profit = wins["Profit"].sum()
        gross_loss = abs(losses["Profit"].sum())
        metrics["Profit Factor"] = gross_profit / gross_loss if gross_loss > 0 else np.inf

        # Average Trade
        metrics["Avg Trade"] = df["Profit"].mean()

        # Drawdown Analysis
        # Construct equity curve
        equity = df["Profit"].cumsum()
        running_max = equity.cummax()
        drawdown = equity - running_max

        # Max Drawdown (Absolute amount)
        metrics["Max Drawdown ($)"] = abs(drawdown.min())

        # Sharpe Ratio (Annualized)
        # Using sample standard deviation (ddof=1) for unbiased estimate
        returns = df["Profit"]
        returns_std = returns.std(ddof=1) if len(returns) > 1 else 0
        if returns_std > 0:
            # Annualized assuming ~52 weeks for weekly data
            metrics["Sharpe Ratio"] = (returns.mean() / returns_std) * np.sqrt(52)
        else:
            metrics["Sharpe Ratio"] = 0

        # Sortino Ratio (Downside risk only)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 1:
            downside_std = downside_returns.std(ddof=1)
            if downside_std > 0:
                metrics["Sortino Ratio"] = (returns.mean() / downside_std) * np.sqrt(52)
            else:
                metrics["Sortino Ratio"] = 0
        else:
            metrics["Sortino Ratio"] = 0

        return metrics

    @staticmethod
    def get_monthly_returns(df):
        """
        Aggregates profits by Month/Year for heatmap.
        """
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
            df["Date"] = pd.to_datetime(df["Date"])

        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month

        monthly = df.groupby(["Year", "Month"])["Profit"].sum().reset_index()
        pivot = monthly.pivot(index="Year", columns="Month", values="Profit")

        return pivot
