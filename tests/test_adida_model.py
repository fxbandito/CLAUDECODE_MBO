"""
ADIDA Model Validation Test
Teszt az AUDJPY adatokon a program betöltési módszerével.
"""
# pylint: disable=wrong-import-position

import sys
import os
import time
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd


def load_data_like_gui(parquet_path: str) -> pd.DataFrame:
    """
    Adatok betöltése pontosan úgy, ahogy a GUI Data Loading tab teszi.
    """
    from data.loader import DataLoader
    from data.processor import DataProcessor

    print(f"Loading data from: {parquet_path}")
    raw_data = DataLoader.load_parquet_files([parquet_path])

    if raw_data is None or raw_data.empty:
        raise ValueError("No data loaded from parquet file")

    print(f"Raw data shape: {raw_data.shape}")

    # Clean data like GUI does
    processed_data = DataProcessor.clean_data(raw_data)
    print(f"Processed data shape: {processed_data.shape}")

    return processed_data


def extract_strategy_data(df: pd.DataFrame, strategy_col: str = "No.") -> dict:
    """
    Kinyeri az egyes stratégiák profit idősorait.
    """
    strategies = {}

    if strategy_col not in df.columns:
        print(f"Warning: Column '{strategy_col}' not found. Using first numeric column.")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            return {"default": df[numeric_cols[0]].values.tolist()}
        return {}

    # Group by strategy
    for strategy_id in df[strategy_col].unique():
        strategy_df = df[df[strategy_col] == strategy_id]
        if "Profit" in strategy_df.columns:
            profit_series = strategy_df["Profit"].values.tolist()
            if len(profit_series) >= 10:  # Minimum data requirement
                strategies[str(strategy_id)] = profit_series

    return strategies


def test_adida_model():
    """
    ADIDA modell tesztelése az AUDJPY adatokon.
    """
    print("=" * 70)
    print("ADIDA MODEL VALIDATION TEST")
    print("=" * 70)

    # Load data
    parquet_path = Path(__file__).parent.parent / "testdata" / "AUDJPY_2020_2021_2022_2023_Weekly_Fix.parquet"

    if not parquet_path.exists():
        print(f"ERROR: Test data file not found: {parquet_path}")
        return False

    try:
        df = load_data_like_gui(str(parquet_path))
    except Exception as e:
        print(f"ERROR loading data: {e}")
        traceback.print_exc()
        return False

    # Extract strategies
    strategies = extract_strategy_data(df)
    print(f"\nFound {len(strategies)} strategies with sufficient data")

    if not strategies:
        # Fallback: use Profit column directly
        if "Profit" in df.columns:
            strategies = {"all": df["Profit"].dropna().values.tolist()}
        else:
            print("ERROR: No usable data found")
            return False

    # Import ADIDA model
    try:
        from models.statistical.adida import ADIDAModel
        print("\n[OK] ADIDA model imported successfully")
    except Exception as e:
        print(f"ERROR importing ADIDA model: {e}")
        traceback.print_exc()
        return False

    # Create model instance
    try:
        model = ADIDAModel()
        print(f"[OK] ADIDA model instantiated: {model}")
        print(f"  - Category: {model.MODEL_INFO.category}")
        print(f"  - GPU Support: {model.MODEL_INFO.supports_gpu}")
        print(f"  - Batch Support: {model.MODEL_INFO.supports_batch}")
    except Exception as e:
        print(f"ERROR creating model instance: {e}")
        traceback.print_exc()
        return False

    # Test parameters
    print("\n" + "-" * 70)
    print("PARAMETER VALIDATION")
    print("-" * 70)
    print(f"Default parameters: {model.PARAM_DEFAULTS}")
    print(f"Parameter options: {model.PARAM_OPTIONS}")

    # Test different methods
    test_configs = [
        {"method": "standard", "aggregation_level": "4", "base_model": "SES"},
        {"method": "standard", "aggregation_level": "auto", "base_model": "ARIMA"},
        {"method": "croston", "alpha": "0.1", "beta": "0.1"},
        {"method": "sba", "alpha": "0.15", "beta": "0.15"},
        {"method": "tsb", "alpha": "0.1", "beta": "0.2"},
    ]

    # Select a test strategy
    test_strategy_name = list(strategies.keys())[0]
    test_data = strategies[test_strategy_name]
    print(f"\nUsing strategy '{test_strategy_name}' with {len(test_data)} data points")

    results = {}
    steps = 12  # Forecast 12 periods ahead

    print("\n" + "-" * 70)
    print("FORECAST TESTS")
    print("-" * 70)

    for config in test_configs:
        method = config.get("method", "standard")
        print(f"\nTesting: {method.upper()}")
        print(f"  Config: {config}")

        try:
            start_time = time.perf_counter()
            forecast = model.forecast(test_data, steps, config)
            elapsed = (time.perf_counter() - start_time) * 1000

            results[method] = {
                "forecast": forecast,
                "time_ms": elapsed,
                "success": True,
                "error": None
            }

            print(f"  [OK] Success in {elapsed:.2f} ms")
            print(f"  Forecast (first 5): {[f'{x:.4f}' if not np.isnan(x) else 'NaN' for x in forecast[:5]]}")

            # Validate forecast
            if all(np.isnan(f) for f in forecast):
                print("  [WARN] WARNING: All forecasts are NaN")
            elif len(forecast) != steps:
                print(f"  [WARN] WARNING: Expected {steps} forecasts, got {len(forecast)}")

        except Exception as e:
            results[method] = {
                "forecast": None,
                "time_ms": 0,
                "success": False,
                "error": str(e)
            }
            print(f"  [FAIL] FAILED: {e}")
            traceback.print_exc()

    # Test batch mode
    print("\n" + "-" * 70)
    print("BATCH MODE TEST")
    print("-" * 70)

    # Use first 5 strategies for batch test
    batch_data = dict(list(strategies.items())[:5])
    if len(batch_data) < 2:
        batch_data = {"strategy1": test_data[:50], "strategy2": test_data[50:100]}

    print(f"Testing batch mode with {len(batch_data)} strategies")

    try:
        start_time = time.perf_counter()
        batch_results = model.forecast_batch(batch_data, steps, {"method": "standard"})
        elapsed = (time.perf_counter() - start_time) * 1000

        print(f"[OK] Batch mode completed in {elapsed:.2f} ms")
        print(f"  Results for {len(batch_results)} strategies")

        for name, forecast in list(batch_results.items())[:3]:
            print(f"  - {name}: {[f'{x:.4f}' for x in forecast[:3]]}")

    except Exception as e:
        print(f"[FAIL] Batch mode FAILED: {e}")
        traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    success_count = sum(1 for r in results.values() if r["success"])
    total_count = len(results)

    print(f"Passed: {success_count}/{total_count} tests")

    for method, result in results.items():
        status = "[PASS]" if result["success"] else "[FAIL]"
        time_str = f"{result['time_ms']:.2f}ms" if result["success"] else "N/A"
        print(f"  {status} {method}: {time_str}")

    return success_count == total_count


def test_cpu_gpu_handling():
    """
    CPU/GPU kezelés tesztelése.
    """
    print("\n" + "=" * 70)
    print("CPU/GPU HANDLING TEST")
    print("=" * 70)

    from models.statistical.adida import ADIDAModel

    model = ADIDAModel()

    # Test get_device method
    print("\nDevice selection tests:")

    # Small data - should use CPU
    device = model.get_device(100, use_gpu=True)
    print(f"  Data size 100, GPU requested: {device}")

    # Large data - might use GPU if available
    device = model.get_device(5000, use_gpu=True)
    print(f"  Data size 5000, GPU requested: {device}")

    # GPU not requested
    device = model.get_device(5000, use_gpu=False)
    print(f"  Data size 5000, GPU not requested: {device}")

    # Check CUDA availability
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"\nCUDA Available: {cuda_available}")
        if cuda_available:
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("\nPyTorch not installed - GPU not available")

    print("\n[OK] CPU/GPU handling test completed")
    return True


def test_feature_mode_compatibility():
    """
    Feature mode kompatibilitas tesztelese.
    """
    print("\n" + "=" * 70)
    print("FEATURE MODE COMPATIBILITY TEST")
    print("=" * 70)

    from models.statistical.adida import ADIDAModel

    # Test Original mode
    compatible, msg = ADIDAModel.check_feature_mode_compatibility("Original")
    print(f"\nOriginal mode: {'COMPATIBLE' if compatible else 'NOT COMPATIBLE'}")
    assert compatible, "Original mode should be compatible"
    print("  [OK] Original mode test passed")

    # Test Forward Calc mode
    compatible, msg = ADIDAModel.check_feature_mode_compatibility("Forward Calc")
    print(f"\nForward Calc mode: {'COMPATIBLE' if compatible else 'NOT COMPATIBLE'}")
    assert not compatible, "Forward Calc mode should NOT be compatible"
    print(f"  Warning message: {msg[:50]}...")
    print("  [OK] Forward Calc warning test passed")

    # Test Rolling Window mode
    compatible, msg = ADIDAModel.check_feature_mode_compatibility("Rolling Window")
    print(f"\nRolling Window mode: {'COMPATIBLE' if compatible else 'NOT COMPATIBLE'}")
    assert not compatible, "Rolling Window mode should NOT be compatible"
    print(f"  Warning message: {msg[:50]}...")
    print("  [OK] Rolling Window warning test passed")

    print("\n[OK] Feature mode compatibility tests passed")
    return True


def test_model_capabilities():
    """
    Model kepessegek tesztelese (panel, dual, gpu, batch).
    """
    print("\n" + "=" * 70)
    print("MODEL CAPABILITIES TEST")
    print("=" * 70)

    from models import (
        supports_gpu,
        supports_batch,
        supports_forward_calc,
        supports_rolling_window,
        supports_panel_mode,
        supports_dual_mode,
        get_model_info
    )

    info = get_model_info("ADIDA")
    print(f"\nADIDA Model Capabilities:")
    print(f"  GPU Support: {supports_gpu('ADIDA')} (expected: False)")
    print(f"  Batch Support: {supports_batch('ADIDA')} (expected: True)")
    print(f"  Forward Calc Support: {supports_forward_calc('ADIDA')} (expected: False)")
    print(f"  Rolling Window Support: {supports_rolling_window('ADIDA')} (expected: False)")
    print(f"  Panel Mode Support: {supports_panel_mode('ADIDA')} (expected: False)")
    print(f"  Dual Mode Support: {supports_dual_mode('ADIDA')} (expected: False)")

    # Assertions
    assert not supports_gpu("ADIDA"), "ADIDA should NOT support GPU"
    assert supports_batch("ADIDA"), "ADIDA should support batch mode"
    assert not supports_forward_calc("ADIDA"), "ADIDA should NOT support Forward Calc"
    assert not supports_rolling_window("ADIDA"), "ADIDA should NOT support Rolling Window"
    assert not supports_panel_mode("ADIDA"), "ADIDA should NOT support Panel Mode"
    assert not supports_dual_mode("ADIDA"), "ADIDA should NOT support Dual Mode"

    print("\n[OK] Model capabilities tests passed")
    return True


def test_parameter_loading():
    """
    Parameter betoltes tesztelese (GUI integracio).
    """
    print("\n" + "=" * 70)
    print("PARAMETER LOADING TEST (GUI Integration)")
    print("=" * 70)

    try:
        from models import (
            get_categories,
            get_models_in_category,
            get_param_defaults,
            get_param_options,
            supports_gpu,
            supports_batch,
            supports_forward_calc,
            supports_rolling_window,
            supports_panel_mode,
            supports_dual_mode,
            get_model_info
        )

        # Check if ADIDA is registered
        categories = get_categories()
        print(f"\nRegistered categories: {len(categories)}")

        # Find Statistical Models
        if "Statistical Models" in categories:
            models = get_models_in_category("Statistical Models")
            print(f"Models in 'Statistical Models': {models}")

            if "ADIDA" in models:
                print("\n[OK] ADIDA is registered in the model registry")

                # Get parameters
                defaults = get_param_defaults("ADIDA")
                options = get_param_options("ADIDA")

                print(f"\nDefault parameters loaded:")
                for key, val in defaults.items():
                    print(f"  {key}: {val}")

                print(f"\nParameter options:")
                for key, opts in options.items():
                    print(f"  {key}: {opts}")

                # Check capabilities
                info = get_model_info("ADIDA")
                print(f"\nModel capabilities:")
                print(f"  GPU Support: {supports_gpu('ADIDA')}")
                print(f"  Batch Support: {supports_batch('ADIDA')}")

                return True
            else:
                print("[FAIL] ADIDA not found in Statistical Models")
                return False
        else:
            print("[FAIL] 'Statistical Models' category not found")
            return False

    except Exception as e:
        print(f"[FAIL] Parameter loading test FAILED: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("# ADIDA MODEL COMPREHENSIVE VALIDATION TEST")
    print("#" * 70)

    all_passed = True

    # Run tests
    try:
        if not test_parameter_loading():
            all_passed = False
    except Exception as e:
        print(f"Parameter loading test error: {e}")
        all_passed = False

    try:
        if not test_adida_model():
            all_passed = False
    except Exception as e:
        print(f"ADIDA model test error: {e}")
        all_passed = False

    try:
        if not test_cpu_gpu_handling():
            all_passed = False
    except Exception as e:
        print(f"CPU/GPU test error: {e}")
        all_passed = False

    try:
        if not test_feature_mode_compatibility():
            all_passed = False
    except Exception as e:
        print(f"Feature mode compatibility test error: {e}")
        all_passed = False

    try:
        if not test_model_capabilities():
            all_passed = False
    except Exception as e:
        print(f"Model capabilities test error: {e}")
        all_passed = False

    # Final result
    print("\n" + "#" * 70)
    if all_passed:
        print("# ALL TESTS PASSED")
    else:
        print("# SOME TESTS FAILED")
    print("#" * 70)

    sys.exit(0 if all_passed else 1)
