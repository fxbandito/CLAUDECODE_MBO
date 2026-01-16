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
            except Exception:  # pylint: disable=broad-exception-caught
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

    # ==================== RANKING SYSTEM ====================

    @staticmethod
    def calculate_stability_metrics(raw_data, results_df):
        """
        Calculate stability metrics for each strategy based on historical data.

        Metrics calculated:
        - activity_consistency: How consistent is the trading activity
        - profit_consistency: Ratio of positive weeks among active weeks
        - sharpe_ratio: Risk-adjusted return (mean/std of profits)
        - data_confidence: Confidence based on amount of data

        Args:
            raw_data: Original DataFrame with historical strategy data
            results_df: DataFrame with forecast results

        Returns:
            DataFrame with stability metrics added
        """
        if raw_data is None or results_df is None or results_df.empty:
            logger.warning("calculate_stability_metrics: raw_data or results_df is None/empty")
            return results_df

        logger.info("Calculating stability metrics for ranking...")

        # Get strategy IDs we need to calculate
        try:
            strategy_ids = set(results_df["No."].astype(int).values)
        except (ValueError, TypeError):
            strategy_ids = set(results_df["No."].values)

        # Pre-filter raw_data to only include relevant strategies
        filtered_data = raw_data.copy()
        if "No." in filtered_data.columns:
            try:
                filtered_data["No."] = (
                    pd.to_numeric(filtered_data["No."], errors="coerce").fillna(-1).astype(int)
                )
            except (ValueError, TypeError):
                pass
        filtered_data = filtered_data[filtered_data["No."].isin(strategy_ids)]

        has_trades_col = "Trades" in filtered_data.columns
        grouped = filtered_data.groupby("No.")

        def calc_metrics(group):
            profits = group["Profit"].values
            trades = group["Trades"].values if has_trades_col else (profits != 0).astype(int)

            total_weeks = len(profits)
            active_weeks = np.sum(trades > 0)
            pos_weeks = np.sum(profits > 0)
            neg_weeks = np.sum(profits < 0)

            # Activity Consistency
            if total_weeks >= 13:
                rolling_activity = pd.Series(trades > 0).rolling(13, min_periods=4).mean()
                activity_mean = rolling_activity.mean()
                activity_std = rolling_activity.std()
                if activity_mean > 0:
                    activity_cv = activity_std / activity_mean
                    activity_consistency = max(0, 1 - activity_cv)
                else:
                    activity_consistency = 0.0
            else:
                activity_consistency = active_weeks / total_weeks if total_weeks > 0 else 0.0

            # Profit Consistency
            if (pos_weeks + neg_weeks) > 0:
                profit_consistency = pos_weeks / (pos_weeks + neg_weeks)
            else:
                profit_consistency = 0.5

            # Sharpe Ratio
            mean_profit = np.mean(profits)
            std_profit = np.std(profits)
            sharpe_ratio = mean_profit / std_profit if std_profit > 0 else 0.0

            # Data Confidence
            data_confidence = min(1.0, total_weeks / 104)

            # Combined Stability Score
            stability_score = (
                0.30 * activity_consistency
                + 0.35 * profit_consistency
                + 0.20 * min(1.0, max(0.0, (sharpe_ratio + 1) / 3))
                + 0.15 * data_confidence
            )

            return pd.Series({
                "Stability_Score": round(stability_score, 4),
                "Activity_Consistency": round(activity_consistency, 4),
                "Profit_Consistency": round(profit_consistency, 4),
                "Sharpe_Ratio": round(sharpe_ratio, 4),
                "Data_Confidence": round(data_confidence, 4),
            })

        stability_df = grouped.apply(calc_metrics, include_groups=False).reset_index()

        # Handle strategies not found in raw_data
        missing_ids = strategy_ids - set(stability_df["No."].values)
        if missing_ids:
            logger.warning("%d strategies not found in raw_data, using defaults", len(missing_ids))
            default_rows = pd.DataFrame([
                {
                    "No.": sid,
                    "Stability_Score": 0.5,
                    "Activity_Consistency": 0.5,
                    "Profit_Consistency": 0.5,
                    "Sharpe_Ratio": 0.0,
                    "Data_Confidence": 0.5,
                }
                for sid in missing_ids
            ])
            stability_df = pd.concat([stability_df, default_rows], ignore_index=True)

        # Merge with results
        try:
            results_to_merge = results_df.copy()
            results_to_merge["No."] = (
                pd.to_numeric(results_to_merge["No."], errors="coerce").fillna(-1).astype(int)
            )
            stability_df["No."] = (
                pd.to_numeric(stability_df["No."], errors="coerce").fillna(-1).astype(int)
            )
            results_with_stability = results_to_merge.merge(stability_df, on="No.", how="left")
        except (ValueError, TypeError) as e:
            logger.error("Merge failed: %s. Falling back to original merge.", e)
            results_with_stability = results_df.merge(stability_df, on="No.", how="left")

        logger.info("Stability metrics calculated for %d strategies", len(stability_df))
        return results_with_stability

    @staticmethod
    def apply_ranking(results_df, ranking_mode="forecast", sort_column="Forecast_1M"):
        """
        Apply ranking based on selected mode.

        Ranking modes:
        - 'forecast': Sort by forecast only (default)
        - 'stability': Percentile-based (60% forecast, 40% stability)
        - 'risk_adjusted': Percentile-based (50% forecast, 30% Sharpe, 20% consistency)

        Args:
            results_df: DataFrame with results and stability metrics
            ranking_mode: One of 'forecast', 'stability', 'risk_adjusted'
            sort_column: Which forecast column to use (default: Forecast_1M)

        Returns:
            tuple: (sorted DataFrame, sort_column_name used)
        """
        logger.info("Applying ranking mode: '%s' (sort_column: %s)", ranking_mode, sort_column)

        if results_df is None or results_df.empty:
            logger.warning("apply_ranking: results_df is None or empty")
            return results_df, sort_column

        df = results_df.copy()

        if ranking_mode == "forecast":
            sort_col = sort_column
            df = df.sort_values(sort_col, ascending=False)

        elif ranking_mode == "stability":
            if "Stability_Score" not in df.columns:
                logger.warning("Stability metrics not found, falling back to forecast ranking")
                sort_col = sort_column
                df = df.sort_values(sort_col, ascending=False)
            else:
                forecast_pct = df[sort_column].rank(pct=True) * 100
                stability_pct = df["Stability_Score"].rank(pct=True) * 100
                df["Ranking_Score"] = (forecast_pct * 0.60) + (stability_pct * 0.40)
                sort_col = "Ranking_Score"
                df = df.sort_values(sort_col, ascending=False)

        elif ranking_mode == "risk_adjusted":
            if "Sharpe_Ratio" not in df.columns:
                logger.warning("Risk metrics not found, falling back to forecast ranking")
                sort_col = sort_column
                df = df.sort_values(sort_col, ascending=False)
            else:
                forecast_pct = df[sort_column].rank(pct=True) * 100
                sharpe_pct = df["Sharpe_Ratio"].rank(pct=True) * 100

                if "Profit_Consistency" in df.columns:
                    consistency_pct = df["Profit_Consistency"].rank(pct=True) * 100
                    df["Ranking_Score"] = (
                        (forecast_pct * 0.50) +
                        (sharpe_pct * 0.30) +
                        (consistency_pct * 0.20)
                    )
                else:
                    df["Ranking_Score"] = (forecast_pct * 0.60) + (sharpe_pct * 0.40)

                sort_col = "Ranking_Score"
                df = df.sort_values(sort_col, ascending=False)

        else:
            logger.warning("Unknown ranking mode: %s, using forecast", ranking_mode)
            sort_col = sort_column
            df = df.sort_values(sort_col, ascending=False)

        # Add rank column
        df["Rank"] = range(1, len(df) + 1)

        return df, sort_col

    @staticmethod
    def group_strategies(
        df: pd.DataFrame,
        strategy_col: str = "No.",
        sort_by: str = "Date",
        exclude_cols: list = None
    ) -> dict:
        """
        Stratégiák előcsoportosítása O(1) lookup-hoz.

        Ez a PRE_GROUPING optimalizáció - egyszer csoportosít,
        utána dict lookup O(n) szűrés helyett O(1).

        Használat:
            - Batch feldolgozásnál sok stratégiával
            - Párhuzamos végrehajtásnál memória optimalizálás
            - Bármikor amikor többször kell stratégia adatot lekérni

        Args:
            df: Bemeneti DataFrame
            strategy_col: Stratégia azonosító oszlop neve
            sort_by: Rendezési oszlop (tipikusan "Date")
            exclude_cols: Kihagyandó oszlopok listája

        Returns:
            Dict[strategy_id, DataFrame] - előcsoportosított adatok

        Példa:
            ```python
            groups = DataProcessor.group_strategies(df)
            for strategy_id in strategy_ids:
                strat_data = groups[strategy_id]  # O(1) lookup!
            ```
        """
        if df is None or df.empty:
            return {}

        if strategy_col not in df.columns:
            logger.warning("Strategy column '%s' not found", strategy_col)
            return {}

        # Default exclude columns (MT5 specifikus, nem kellenek a modelleknek)
        if exclude_cols is None:
            exclude_cols = [
                "Start", "1st Candle", "Shift", "Position",
                "Param. Sum", "SourceFile"
            ]

        strategy_groups = {}
        for strat_id, group in df.groupby(strategy_col):
            # Rendezés ha van ilyen oszlop
            if sort_by and sort_by in group.columns:
                sorted_group = group.sort_values(sort_by)
            else:
                sorted_group = group

            # Felesleges oszlopok eltávolítása
            cols_to_drop = [c for c in exclude_cols if c in sorted_group.columns]
            if cols_to_drop:
                sorted_group = sorted_group.drop(columns=cols_to_drop)

            strategy_groups[strat_id] = sorted_group

        logger.debug("Pre-grouped %d strategies for O(1) lookup", len(strategy_groups))
        return strategy_groups

    @staticmethod
    def detect_data_mode(df: pd.DataFrame) -> str:
        """
        Detektálja az adat módot az oszlopnevek alapján.

        Három mód lehetséges:
        - "rolling": Rolling feature oszlopok jelenléte (több memória kell)
        - "forward": Forward-looking feature oszlopok jelenléte
        - "original": Alap mód, feature engineering nélkül

        Használat:
            - Worker szám optimalizálásnál (rolling = kevesebb worker)
            - Memória becslésénél

        Args:
            df: DataFrame az oszlopnevekkel

        Returns:
            str: "rolling", "forward", vagy "original"
        """
        if df is None or df.empty:
            return "original"

        columns = set(df.columns)

        # Rolling mód indikátorok
        rolling_indicators = [
            "feat_rolling_active_ratio",
            "feat_rolling_profit_consistency",
            "feat_rolling_sharpe",
            "feat_rolling_volatility",
        ]

        if any(col in columns for col in rolling_indicators):
            return "rolling"

        # Forward mód indikátorok
        forward_indicators = [
            "feat_forward_return_1w",
            "feat_forward_return_4w",
            "feat_forward_activity_1w",
        ]

        if any(col in columns for col in forward_indicators):
            return "forward"

        return "original"
