"""
Data processing module for strategy analysis.

Handles data cleaning, feature engineering (original, forward, rolling),
and data preparation for ML models.
"""

import gc
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataProcessor:
    """Data processing and feature engineering for strategy analysis."""

    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw dataframe.

        - Converts numerical columns to proper types
        - Handles percentage strings (e.g. "5%" -> 5.0)
        - Handles accounting format (e.g. "(123)" -> -123)
        - Removes NaN rows if necessary

        Args:
            df: Raw DataFrame

        Returns:
            Cleaned DataFrame
        """
        if df is None or df.empty:
            return df

        df = df.copy()

        # Numeric columns to clean
        numeric_cols = ["Profit", "Trades", "PF", "DD %"]
        existing_cols = [c for c in numeric_cols if c in df.columns]

        for col in existing_cols:
            if df[col].dtype == object:
                # Convert to string for processing
                df[col] = df[col].astype(str)

                # Handle accounting format: (123.45) -> -123.45
                mask_parens = df[col].str.match(r"^\(.*\)$", na=False)
                df.loc[mask_parens, col] = "-" + df.loc[mask_parens, col].str.replace(
                    r"[()]", "", regex=True
                )

                # Remove currency symbols, % signs, spaces
                df[col] = df[col].str.replace(r"[^\d.-]", "", regex=True)

                # Convert to numeric
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Handle empty PF cells as 0
        if "PF" in df.columns:
            df["PF"] = df["PF"].fillna(0)

        # Convert 'No.' to integer
        if "No." in df.columns:
            df["No."] = pd.to_numeric(df["No."], errors="coerce").fillna(0).astype(int)

        return df

    @staticmethod
    def prepare_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for time-series or ML analysis.
        Groups by 'No.' (Strategy ID) and sorts by time.

        Args:
            df: Cleaned DataFrame

        Returns:
            Sorted DataFrame
        """
        if "No." in df.columns and "Date" in df.columns:
            return df.sort_values(by=["No.", "Date"]).reset_index(drop=True)
        if "Date" in df.columns:
            return df.sort_values(by="Date").reset_index(drop=True)
        return df

    @staticmethod
    def add_features_forward(df: pd.DataFrame) -> pd.DataFrame:
        """
        Forward calculation: Add EXPANDING WINDOW features for each strategy.
        At each time point, features are computed from all data UP TO that point.

        Features added:
        - feat_weeks_count: Number of weeks elapsed
        - feat_active_ratio: Percentage of weeks with trades so far
        - feat_profit_consistency: Positive weeks / (Positive + Negative weeks)
        - feat_total_profit: Cumulative profit
        - feat_cumulative_trades: Total trades so far
        - feat_volatility: Standard deviation of profits
        - feat_sharpe_ratio: Mean profit / Std profit
        - feat_max_drawdown: Maximum drawdown so far

        Args:
            df: DataFrame with strategy data

        Returns:
            DataFrame with added features
        """
        if df is None or df.empty:
            return df

        if "No." not in df.columns or "Profit" not in df.columns:
            logger.warning("Cannot add features: 'No.' or 'Profit' column missing")
            return df

        logger.info("Adding forward-calculated features (expanding window)...")

        df = df.sort_values(["No.", "Date"]).copy()

        # Helper columns
        df["_is_active"] = (df["Trades"] > 0).astype(float) if "Trades" in df.columns else 1.0
        df["_is_positive"] = (df["Profit"] > 0).astype(float)
        df["_is_negative"] = (df["Profit"] < 0).astype(float)

        grouped = df.groupby("No.", group_keys=False)

        # Weeks count
        df["feat_weeks_count"] = grouped.cumcount() + 1

        # Active ratio
        df["feat_active_ratio"] = grouped["_is_active"].cumsum() / df["feat_weeks_count"]

        # Profit consistency
        cum_pos = grouped["_is_positive"].cumsum()
        cum_neg = grouped["_is_negative"].cumsum()
        cum_total = cum_pos + cum_neg
        df["feat_profit_consistency"] = np.where(cum_total > 0, cum_pos / cum_total, 0.5)

        # Cumulative profit and trades
        df["feat_total_profit"] = grouped["Profit"].cumsum()
        if "Trades" in df.columns:
            df["feat_cumulative_trades"] = grouped["Trades"].cumsum()
        else:
            df["feat_cumulative_trades"] = df["feat_weeks_count"]

        # Expanding mean and std
        expanding_mean = grouped["Profit"].expanding(min_periods=1).mean()
        expanding_mean = expanding_mean.reset_index(level=0, drop=True)

        expanding_std = grouped["Profit"].expanding(min_periods=2).std()
        expanding_std = expanding_std.reset_index(level=0, drop=True)

        df["feat_volatility"] = expanding_std.fillna(0)
        df["feat_sharpe_ratio"] = np.where(expanding_std > 0, expanding_mean / expanding_std, 0)

        # Max drawdown
        cumsum = grouped["Profit"].cumsum()
        running_max = grouped["Profit"].apply(lambda x: x.cumsum().cummax())
        df["feat_max_drawdown"] = (cumsum - running_max).clip(upper=0)

        # Cleanup
        df = df.drop(columns=["_is_active", "_is_positive", "_is_negative"])

        feat_count = len([c for c in df.columns if c.startswith("feat_")])
        logger.info("Forward features added: %d new columns", feat_count)

        return df

    @staticmethod
    def add_features_rolling(df: pd.DataFrame, window: int = 13) -> pd.DataFrame:
        """
        Rolling window calculation: Add features computed from a rolling window.

        Features added:
        - feat_rolling_active_ratio: Rolling percentage of weeks with trades
        - feat_rolling_profit_consistency: Rolling positive weeks ratio
        - feat_rolling_sharpe: Rolling Sharpe-like ratio
        - feat_rolling_volatility: Rolling standard deviation
        - feat_rolling_avg_profit: Rolling average profit
        - feat_rolling_momentum_4w: 4-week momentum
        - feat_rolling_momentum_13w: 13-week momentum
        - feat_rolling_profit_sum: Rolling sum of profits
        - feat_rolling_max_dd: Rolling maximum drawdown

        Args:
            df: DataFrame with strategy data
            window: Rolling window size in weeks (default 13 = ~3 months)

        Returns:
            DataFrame with added features
        """
        if df is None or df.empty:
            return df

        if "No." not in df.columns or "Profit" not in df.columns:
            logger.warning("Cannot add rolling features: 'No.' or 'Profit' column missing")
            return df

        logger.info("Adding rolling features with window=%d...", window)

        df = df.sort_values(["No.", "Date"]).copy()
        min_periods = max(4, window // 3)

        # Helper columns
        df["_is_active"] = (df["Trades"] > 0).astype(float) if "Trades" in df.columns else 1.0
        df["_is_positive"] = (df["Profit"] > 0).astype(float)
        df["_is_negative"] = (df["Profit"] < 0).astype(float)

        grouped = df.groupby("No.", group_keys=False)

        # Rolling active ratio
        df["feat_rolling_active_ratio"] = grouped["_is_active"].transform(
            lambda x: x.rolling(window, min_periods=min_periods).mean()
        )

        # Rolling profit consistency
        rolling_pos = grouped["_is_positive"].transform(
            lambda x: x.rolling(window, min_periods=min_periods).sum()
        )
        rolling_neg = grouped["_is_negative"].transform(
            lambda x: x.rolling(window, min_periods=min_periods).sum()
        )
        rolling_active = rolling_pos + rolling_neg
        df["feat_rolling_profit_consistency"] = np.where(
            rolling_active > 0, rolling_pos / rolling_active, 0.5
        )

        # Rolling mean and std
        rolling_mean = grouped["Profit"].transform(
            lambda x: x.rolling(window, min_periods=min_periods).mean()
        )
        rolling_std = grouped["Profit"].transform(
            lambda x: x.rolling(window, min_periods=min_periods).std()
        )

        df["feat_rolling_sharpe"] = np.where(rolling_std > 0, rolling_mean / rolling_std, 0)
        df["feat_rolling_volatility"] = rolling_std
        df["feat_rolling_avg_profit"] = rolling_mean

        # Momentum indicators
        df["feat_rolling_momentum_4w"] = grouped["Profit"].transform(
            lambda x: x.rolling(4, min_periods=2).mean()
        )
        df["feat_rolling_momentum_13w"] = grouped["Profit"].transform(
            lambda x: x.rolling(13, min_periods=4).mean()
        )

        # Rolling sum
        df["feat_rolling_profit_sum"] = grouped["Profit"].transform(
            lambda x: x.rolling(window, min_periods=min_periods).sum()
        )

        # Rolling max drawdown
        cumsum = grouped["Profit"].cumsum()
        rolling_max = grouped["Profit"].transform(
            lambda x: x.cumsum().rolling(window, min_periods=min_periods).max()
        )
        df["feat_rolling_max_dd"] = (cumsum - rolling_max).clip(upper=0)

        # Cleanup and fill NaN
        df = df.drop(columns=["_is_active", "_is_positive", "_is_negative"])
        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        df[feat_cols] = df[feat_cols].fillna(0)

        logger.info("Rolling features added: %d new columns", len(feat_cols))

        # Memory cleanup after rolling (memory-intensive operation)
        gc.collect()

        return df

    @staticmethod
    def get_feature_mode_description(mode: str) -> str:
        """
        Get description of what each feature mode does.

        Args:
            mode: 'Original', 'Forward Calc', or 'Rolling Window'

        Returns:
            Description string
        """
        descriptions = {
            "Original": "No additional features - raw data only",
            "Forward Calc": "Features from entire history (expanding window)",
            "Rolling Window": "Features from rolling 13-week window (dynamic)",
        }
        return descriptions.get(mode, "Unknown mode")

    @staticmethod
    def get_data_summary(df: pd.DataFrame) -> dict:
        """
        Get summary statistics for loaded data.

        Args:
            df: DataFrame

        Returns:
            Dictionary with summary stats
        """
        if df is None or df.empty:
            return {"rows": 0, "columns": 0, "strategies": 0, "date_range": "N/A"}

        summary = {
            "rows": len(df),
            "columns": len(df.columns),
            "strategies": df["No."].nunique() if "No." in df.columns else 0,
        }

        if "Date" in df.columns:
            try:
                dates = pd.to_datetime(df["Date"], errors="coerce")
                summary["date_range"] = f"{dates.min()} - {dates.max()}"
            except Exception:
                summary["date_range"] = "N/A"
        else:
            summary["date_range"] = "N/A"

        if "SourceFile" in df.columns:
            summary["files"] = df["SourceFile"].nunique()
        else:
            summary["files"] = 1

        # Feature count
        summary["features"] = len([c for c in df.columns if c.startswith("feat_")])

        return summary

    @staticmethod
    def get_feature_count(df: pd.DataFrame) -> int:
        """
        Get the number of feature columns in the DataFrame.

        Args:
            df: DataFrame

        Returns:
            Number of feature columns (columns starting with 'feat_')
        """
        if df is None or df.empty:
            return 0
        return len([c for c in df.columns if c.startswith("feat_")])
