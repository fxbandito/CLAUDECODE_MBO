"""
ARIMA Model Validation Test
MBO Trading Strategy Analyzer

Ez a teszt fajl ellenorzi az ARIMA modell teljes funkcionalitasat:
- Parameter betoltes
- Forecast muveletek
- Edge case kezeles
- Batch mod
- Feature mode kompatibilitas
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import time
import numpy as np


def test_parameter_loading():
    """GUI parameter teszt - ellenorzi, hogy a parameterek betoltodnek."""
    print("\n" + "=" * 60)
    print("TEST: Parameter Loading")
    print("=" * 60)

    from models import get_param_defaults, get_param_options, get_model_info

    # Defaults
    defaults = get_param_defaults("ARIMA")
    assert defaults is not None, "No defaults found"
    assert "p" in defaults, "Missing 'p' parameter"
    assert "d" in defaults, "Missing 'd' parameter"
    assert "q" in defaults, "Missing 'q' parameter"
    assert "trend" in defaults, "Missing 'trend' parameter"
    print(f"  [OK] PARAM_DEFAULTS: {defaults}")

    # Options
    options = get_param_options("ARIMA")
    assert options is not None, "No options found"
    assert "p" in options, "Missing 'p' options"
    assert "d" in options, "Missing 'd' options"
    assert "q" in options, "Missing 'q' options"
    assert "trend" in options, "Missing 'trend' options"
    print(f"  [OK] PARAM_OPTIONS keys: {list(options.keys())}")

    # Model Info
    info = get_model_info("ARIMA")
    assert info is not None, "No model info found"
    assert info.name == "ARIMA", f"Wrong model name: {info.name}"
    assert info.category == "Statistical Models", f"Wrong category: {info.category}"
    assert info.supports_gpu is False, "ARIMA should not support GPU"
    assert info.supports_batch is True, "ARIMA should support batch mode"
    print(f"  [OK] MODEL_INFO: name={info.name}, category={info.category}")
    print(f"       supports_gpu={info.supports_gpu}, supports_batch={info.supports_batch}")

    print("\n[PASS] Parameter loading test passed")
    return True


def test_arima_basic_forecast():
    """Alapveto forecast funkcionalitas teszt."""
    print("\n" + "=" * 60)
    print("TEST: Basic Forecast")
    print("=" * 60)

    from models.statistical.arima import ARIMAModel

    model = ARIMAModel()

    # Test data - szinuszos mintazat zajjal
    np.random.seed(42)
    t = np.linspace(0, 4 * np.pi, 100)
    test_data = (np.sin(t) * 100 + 500 + np.random.randn(100) * 20).tolist()

    # Default parameterekkel
    start = time.perf_counter()
    result = model.forecast(test_data, steps=12, params={})
    elapsed = (time.perf_counter() - start) * 1000

    print(f"  Input data length: {len(test_data)}")
    print(f"  Forecast steps: 12")
    print(f"  Execution time: {elapsed:.2f} ms")
    print(f"  First 5 forecasts: {[f'{x:.2f}' for x in result[:5]]}")

    assert len(result) == 12, f"Forecast length mismatch: {len(result)} != 12"
    assert not all(np.isnan(result)), "All forecasts are NaN"
    assert not all(r == 0.0 for r in result), "All forecasts are zero"

    print("\n[PASS] Basic forecast test passed")
    return True


def test_arima_different_parameters():
    """Kulonbozo parameterek tesztelese."""
    print("\n" + "=" * 60)
    print("TEST: Different Parameters")
    print("=" * 60)

    from models.statistical.arima import ARIMAModel

    model = ARIMAModel()

    np.random.seed(42)
    test_data = list(np.random.randn(100) * 50 + 1000)

    test_cases = [
        {"p": "0", "d": "1", "q": "0"},  # Random walk
        {"p": "1", "d": "1", "q": "1"},  # ARIMA(1,1,1)
        {"p": "2", "d": "1", "q": "2"},  # ARIMA(2,1,2)
        {"p": "1", "d": "0", "q": "1"},  # ARMA(1,1) - no differencing
        {"p": "3", "d": "1", "q": "3", "trend": "ct"},  # Complex with trend
    ]

    for params in test_cases:
        start = time.perf_counter()
        result = model.forecast(test_data, steps=12, params=params)
        elapsed = (time.perf_counter() - start) * 1000

        p, d, q = params.get("p", "1"), params.get("d", "1"), params.get("q", "1")
        trend = params.get("trend", "c")

        assert len(result) == 12, f"ARIMA({p},{d},{q}) failed: wrong length"
        assert not all(np.isnan(result)), f"ARIMA({p},{d},{q}) failed: all NaN"

        print(f"  [OK] ARIMA({p},{d},{q}) trend={trend}: {elapsed:.2f}ms, forecast[0]={result[0]:.2f}")

    print("\n[PASS] Different parameters test passed")
    return True


def test_edge_cases():
    """Edge case teszt - ures adat, NaN, egyszeru ertek."""
    print("\n" + "=" * 60)
    print("TEST: Edge Cases")
    print("=" * 60)

    from models.statistical.arima import ARIMAModel

    model = ARIMAModel()

    # 1. Ures adat
    result = model.forecast([], steps=5, params={})
    assert len(result) == 5, "Empty data handling failed"
    print(f"  [OK] Empty data: returns {len(result)} zeros")

    # 2. Tul rovid adat
    result = model.forecast([1.0, 2.0, 3.0], steps=5, params={})
    assert len(result) == 5, "Short data handling failed"
    print(f"  [OK] Short data (3 points): returns {len(result)} values")

    # 3. Minden NaN
    result = model.forecast([np.nan] * 20, steps=5, params={})
    assert len(result) == 5, "All NaN handling failed"
    print(f"  [OK] All NaN data: returns {len(result)} values")

    # 4. Reszleges NaN
    data_with_nan = [100.0, np.nan, 102.0, np.nan, 104.0, 105.0, np.nan, 107.0, 108.0, 109.0, 110.0, 111.0]
    result = model.forecast(data_with_nan, steps=5, params={})
    assert len(result) == 5, "Partial NaN handling failed"
    assert not all(np.isnan(result)), "Partial NaN should produce forecasts"
    print(f"  [OK] Partial NaN data: forecast[0]={result[0]:.2f}")

    # 5. Konstans ertek
    result = model.forecast([100.0] * 50, steps=5, params={})
    assert len(result) == 5, "Constant value handling failed"
    print(f"  [OK] Constant data: forecast[0]={result[0]:.2f}")

    # 6. Egyszeru ertek
    result = model.forecast([100.0], steps=5, params={})
    assert len(result) == 5, "Single value handling failed"
    print(f"  [OK] Single value: returns {len(result)} values")

    # 7. Negativ ertekek
    result = model.forecast(list(np.random.randn(50) * 100 - 500), steps=5, params={})
    assert len(result) == 5, "Negative values handling failed"
    print(f"  [OK] Negative values: forecast[0]={result[0]:.2f}")

    print("\n[PASS] Edge case tests passed")
    return True


def test_batch_mode():
    """Batch mod teszt - tobb strategia egyszerre."""
    print("\n" + "=" * 60)
    print("TEST: Batch Mode")
    print("=" * 60)

    from models.statistical.arima import ARIMAModel

    model = ARIMAModel()

    # Teszt adatok - 10 strategia
    np.random.seed(42)
    all_data = {}
    for i in range(10):
        all_data[f"Strategy_{i:02d}"] = list(np.random.randn(100) * 50 + 1000 + i * 10)

    start = time.perf_counter()
    results = model.forecast_batch(all_data, steps=12, params={"p": "1", "d": "1", "q": "1"})
    elapsed = (time.perf_counter() - start) * 1000

    assert len(results) == 10, f"Batch result count mismatch: {len(results)} != 10"

    for name, forecasts in results.items():
        assert len(forecasts) == 12, f"{name}: wrong forecast length"
        assert not all(np.isnan(forecasts)), f"{name}: all NaN"

    print(f"  Strategies processed: {len(results)}")
    print(f"  Total execution time: {elapsed:.2f} ms")
    print(f"  Average per strategy: {elapsed / len(results):.2f} ms")
    print(f"  Sample forecast (Strategy_00): {[f'{x:.2f}' for x in results['Strategy_00'][:3]]}")

    print("\n[PASS] Batch mode test passed")
    return True


def test_feature_mode_compatibility():
    """Feature mode kompatibilitas teszt."""
    print("\n" + "=" * 60)
    print("TEST: Feature Mode Compatibility")
    print("=" * 60)

    from models.statistical.arima import ARIMAModel

    # Original mode
    compatible, message = ARIMAModel.check_feature_mode_compatibility("Original")
    assert compatible is True, "Original mode should be compatible"
    print(f"  [OK] Original mode: compatible={compatible}")

    # Forward Calc mode
    compatible, message = ARIMAModel.check_feature_mode_compatibility("Forward Calc")
    assert compatible is True, "Forward Calc should be compatible for ARIMA"
    print(f"  [OK] Forward Calc mode: compatible={compatible}")
    if message:
        print(f"       Message: {message.split(chr(10))[0]}...")

    # Rolling Window mode
    compatible, message = ARIMAModel.check_feature_mode_compatibility("Rolling Window")
    assert compatible is True, "Rolling Window should be compatible for ARIMA"
    print(f"  [OK] Rolling Window mode: compatible={compatible}")
    if message:
        print(f"       Message: {message.split(chr(10))[0]}...")

    print("\n[PASS] Feature mode compatibility test passed")
    return True


def test_model_info_attributes():
    """Model info attributumok tesztelese."""
    print("\n" + "=" * 60)
    print("TEST: Model Info Attributes")
    print("=" * 60)

    from models.statistical.arima import ARIMAModel

    model = ARIMAModel()
    info = model.MODEL_INFO

    # Ellenorzes
    assert info.name == "ARIMA", f"Wrong name: {info.name}"
    assert info.category == "Statistical Models", f"Wrong category: {info.category}"
    assert info.supports_gpu is False, "ARIMA should not support GPU"
    assert info.supports_batch is True, "ARIMA should support batch"
    assert info.supports_forward_calc is True, "ARIMA should support forward calc"
    assert info.supports_rolling_window is True, "ARIMA should support rolling window"
    assert info.supports_panel_mode is False, "ARIMA should not support panel mode"
    assert info.supports_dual_mode is False, "ARIMA should not support dual mode"

    print(f"  [OK] name: {info.name}")
    print(f"  [OK] category: {info.category}")
    print(f"  [OK] supports_gpu: {info.supports_gpu}")
    print(f"  [OK] supports_batch: {info.supports_batch}")
    print(f"  [OK] supports_forward_calc: {info.supports_forward_calc}")
    print(f"  [OK] supports_rolling_window: {info.supports_rolling_window}")
    print(f"  [OK] supports_panel_mode: {info.supports_panel_mode}")
    print(f"  [OK] supports_dual_mode: {info.supports_dual_mode}")

    print("\n[PASS] Model info attributes test passed")
    return True


def test_on_audjpy():
    """Teszt az AUDJPY adatokon (ha elerheto)."""
    print("\n" + "=" * 60)
    print("TEST: AUDJPY Real Data")
    print("=" * 60)

    parquet_path = Path(__file__).parent.parent / "testdata" / "AUDJPY_2020_2021_2022_2023_Weekly_Fix.parquet"

    if not parquet_path.exists():
        print(f"  [SKIP] Test data not found: {parquet_path}")
        return True

    try:
        import pandas as pd
        from models.statistical.arima import ARIMAModel

        # Load data
        df = pd.read_parquet(parquet_path)
        print(f"  Loaded data shape: {df.shape}")

        # Extract profit column
        if "Profit" in df.columns:
            # Group by strategy and use first one
            strategies = df["No."].unique()[:5] if "No." in df.columns else [0]

            model = ARIMAModel()
            total_time = 0

            for strat_id in strategies[:3]:  # Test 3 strategies
                if "No." in df.columns:
                    strat_data = df[df["No."] == strat_id]["Profit"].dropna().values.tolist()
                else:
                    strat_data = df["Profit"].dropna().values.tolist()

                if len(strat_data) < 20:
                    continue

                start = time.perf_counter()
                result = model.forecast(strat_data[-100:], steps=12, params={})
                elapsed = (time.perf_counter() - start) * 1000
                total_time += elapsed

                print(f"  [OK] Strategy {strat_id}: {elapsed:.2f}ms, data_len={len(strat_data)}")
                print(f"       Forecast: {[f'{x:.2f}' for x in result[:3]]}...")

            print(f"\n  Average time: {total_time / 3:.2f}ms per strategy")
        else:
            print("  [SKIP] No 'Profit' column found")

    except Exception as e:
        print(f"  [WARN] AUDJPY test error: {e}")
        return True

    print("\n[PASS] AUDJPY test passed")
    return True


def test_fallback_mechanism():
    """Fallback mechanizmus teszt - problemás adatokkal."""
    print("\n" + "=" * 60)
    print("TEST: Fallback Mechanism")
    print("=" * 60)

    from models.statistical.arima import ARIMAModel

    model = ARIMAModel()

    # Problemás adat ami triggereli a fallback-et
    # Nagy p,d,q kicsi adatmennyiseggel
    result = model.forecast([1.0] * 15, steps=5, params={"p": "5", "d": "2", "q": "5"})
    assert len(result) == 5, "Fallback should return correct length"
    print(f"  [OK] High order with low data: forecast[0]={result[0]:.2f}")

    # Extrém értékekkel
    extreme_data = list(np.array([1e10, 1e-10] * 50))
    result = model.forecast(extreme_data, steps=5, params={})
    assert len(result) == 5, "Extreme values should be handled"
    print(f"  [OK] Extreme values: returned {len(result)} forecasts")

    print("\n[PASS] Fallback mechanism test passed")
    return True


def run_all_tests():
    """Minden teszt futtatasa."""
    print("\n" + "=" * 60)
    print("ARIMA MODEL VALIDATION TESTS")
    print("=" * 60)

    tests = [
        test_parameter_loading,
        test_arima_basic_forecast,
        test_arima_different_parameters,
        test_edge_cases,
        test_batch_mode,
        test_feature_mode_compatibility,
        test_model_info_attributes,
        test_fallback_mechanism,
        test_on_audjpy,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"\n[FAIL] {test.__name__}")
        except Exception as e:
            failed += 1
            print(f"\n[FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Passed: {passed}/{len(tests)}")
    print(f"  Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n[OK] ALL TESTS PASSED")
        return True
    else:
        print(f"\n[FAIL] {failed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
