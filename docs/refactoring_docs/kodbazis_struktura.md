# Kódbázis Struktúra és Felülvizsgálati Dokumentum

Használja ezt a dokumentumot az egyes fájlok áttekintésére, leírások hozzáadására és a refaktorálás követésére.


## Gyökér (`src/`)
- [ ] `auto.txt` | **Funkció:** [TODO]
- [ ] `clear_pycache.bat` | **Funkció:** [TODO]
- [ ] `clear_pycache.ps1` | **Funkció:** [TODO]
- [ ] `cloudflared.exe` | **Funkció:** [TODO]
- [ ] `install_requirements.bat` | **Funkció:** [TODO]
- [ ] `install_requirements.sh` | **Funkció:** [TODO]
- [ ] `main.py` | **Funkció:** [TODO]
- [ ] `main_debug.py` | **Funkció:** [TODO]
- [ ] `requirements.txt` | **Funkció:** [TODO]
- [ ] `start_tunnel.ps1` | **Funkció:** [TODO]
- [ ] `start_tunnel.sh` | **Funkció:** [TODO]
- [ ] `start_web.bat` | **Funkció:** [TODO]
- [ ] `start_web.sh` | **Funkció:** [TODO]
- [ ] `web_ui.py` | **Funkció:** [TODO]
- [ ] `secrets.toml` | **Funkció:** [TODO]

### Elemzés (`src/analysis`)
- [ ] `cpu_manager.py` | **Funkció:** [TODO]
- [ ] `dual_executor.py` | **Funkció:** [TODO]
- [ ] `dual_mode_task.py` | **Funkció:** [TODO]
- [ ] `engine.py` | **Funkció:** [TODO]
- [ ] `engine_utils.py` | **Funkció:** [TODO]
- [ ] `inspection.py` | **Funkció:** [TODO]
- [ ] `metrics.py` | **Funkció:** [TODO]
- [ ] `panel_executor.py` | **Funkció:** [TODO]
- [ ] `performance.py` | **Funkció:** [TODO]
- [ ] `strategy_analyzer.py` | **Funkció:** [TODO]

#### Összehasonlító (`src/analysis/comparator`)
- [ ] `base.py` | **Funkció:** [TODO]
- [ ] `horizon.py` | **Funkció:** [TODO]
- [ ] `main_data.py` | **Funkció:** [TODO]
- [ ] `main_data_ar.py` | **Funkció:** [TODO]

#### Modellek (`src/analysis/models`)

##### Klasszikus ML (`src/analysis/models/classical_ml`)
- [ ] `base.py` | **Funkció:** [TODO]
- [ ] `gradient_boosting.py` | **Funkció:** [TODO]
- [ ] `knn_regressor.py` | **Funkció:** [TODO]
- [ ] `lightgbm_model.py` | **Funkció:** [TODO]
- [ ] `random_forest.py` | **Funkció:** [TODO]
- [ ] `regressors.py` | **Funkció:** [TODO]
- [ ] `svr.py` | **Funkció:** [TODO]
- [ ] `xgboost_model.py` | **Funkció:** [TODO]

##### DL CNN (`src/analysis/models/dl_cnn`)
- [ ] `dlinear.py` | **Funkció:** [TODO]
- [ ] `dlinear_batch.py` | **Funkció:** [TODO]
- [ ] `nbeats.py` | **Funkció:** [TODO]
- [ ] `nbeats_batch.py` | **Funkció:** [TODO]
- [ ] `nhits.py` | **Funkció:** [TODO]
- [ ] `nhits_batch.py` | **Funkció:** [TODO]
- [ ] `tcn.py` | **Funkció:** [TODO]
- [ ] `tcn_batch.py` | **Funkció:** [TODO]
- [ ] `tide.py` | **Funkció:** [TODO]
- [ ] `tide_batch.py` | **Funkció:** [TODO]
- [ ] `timesnet.py` | **Funkció:** [TODO]
- [ ] `timesnet_batch.py` | **Funkció:** [TODO]

##### DL Gráf Specializált (`src/analysis/models/dl_graph_specialized`)
- [ ] `diffusion.py` | **Funkció:** [TODO]
- [ ] `diffusion_batch.py` | **Funkció:** [TODO]
- [ ] `kan.py` | **Funkció:** [TODO]
- [ ] `kan_batch.py` | **Funkció:** [TODO]
- [ ] `mtgnn.py` | **Funkció:** [TODO]
- [ ] `mtgnn_batch.py` | **Funkció:** [TODO]
- [ ] `neural_arima.py` | **Funkció:** [TODO]
- [ ] `neural_arima_batch.py` | **Funkció:** [TODO]
- [ ] `neural_gam.py` | **Funkció:** [TODO]
- [ ] `neural_ode.py` | **Funkció:** [TODO]
- [ ] `neural_quantile_regression.py` | **Funkció:** [TODO]
- [ ] `neural_var.py` | **Funkció:** [TODO]
- [ ] `neural_volatility.py` | **Funkció:** [TODO]
- [ ] `rbf.py` | **Funkció:** [TODO]
- [ ] `snn.py` | **Funkció:** [TODO]
- [ ] `stemgnn.py` | **Funkció:** [TODO]

##### DL RNN (`src/analysis/models/dl_rnn`)
- [ ] `deepar.py` | **Funkció:** [TODO]
- [ ] `deepar_batch.py` | **Funkció:** [TODO]
- [ ] `es_rnn.py` | **Funkció:** [TODO]
- [ ] `es_rnn_batch.py` | **Funkció:** [TODO]
- [ ] `gru.py` | **Funkció:** [TODO]
- [ ] `gru_batch.py` | **Funkció:** [TODO]
- [ ] `lstm.py` | **Funkció:** [TODO]
- [ ] `lstm_batch.py` | **Funkció:** [TODO]
- [ ] `mqrnn.py` | **Funkció:** [TODO]
- [ ] `mqrnn_batch.py` | **Funkció:** [TODO]
- [ ] `seq2seq.py` | **Funkció:** [TODO]
- [ ] `seq2seq_batch.py` | **Funkció:** [TODO]

##### DL Transformer (`src/analysis/models/dl_transformer`)
- [ ] `autoformer.py` | **Funkció:** [TODO]
- [ ] `autoformer_batch.py` | **Funkció:** [TODO]
- [ ] `fedformer.py` | **Funkció:** [TODO]
- [ ] `fedformer_batch.py` | **Funkció:** [TODO]
- [ ] `fits.py` | **Funkció:** [TODO]
- [ ] `fits_batch.py` | **Funkció:** [TODO]
- [ ] `informer.py` | **Funkció:** [TODO]
- [ ] `informer_batch.py` | **Funkció:** [TODO]
- [ ] `itransformer.py` | **Funkció:** [TODO]
- [ ] `itransformer_batch.py` | **Funkció:** [TODO]
- [ ] `patchtst.py` | **Funkció:** [TODO]
- [ ] `patchtst_batch.py` | **Funkció:** [TODO]
- [ ] `tft.py` | **Funkció:** [TODO]
- [ ] `tft_batch.py` | **Funkció:** [TODO]
- [ ] `transformer.py` | **Funkció:** [TODO]
- [ ] `transformer_batch.py` | **Funkció:** [TODO]

##### DL Segédek (`src/analysis/models/dl_utils`)
- [ ] `utils.py` | **Funkció:** [TODO]

##### Együttes (`src/analysis/models/ensemble`)
- [ ] `ensemble.py` | **Funkció:** [TODO]

##### Meta Tanulás (`src/analysis/models/meta_learning`)
- [ ] `darts.py` | **Funkció:** [TODO]
- [ ] `fforma.py` | **Funkció:** [TODO]
- [ ] `gfm.py` | **Funkció:** [TODO]
- [ ] `meta_learning.py` | **Funkció:** [TODO]
- [ ] `moe.py` | **Funkció:** [TODO]
- [ ] `mtl.py` | **Funkció:** [TODO]
- [ ] `nas.py` | **Funkció:** [TODO]

##### Valószínűségi (`src/analysis/models/probabilistic`)
- [ ] `bsts.py` | **Funkció:** [TODO]
- [ ] `conformal_prediction.py` | **Funkció:** [TODO]
- [ ] `gaussian_process.py` | **Funkció:** [TODO]
- [ ] `monte_carlo.py` | **Funkció:** [TODO]
- [ ] `prophet.py` | **Funkció:** [TODO]

##### Hasonlóság (`src/analysis/models/similarity`)
- [ ] `dtw.py` | **Funkció:** [TODO]
- [ ] `knn.py` | **Funkció:** [TODO]
- [ ] `kshape.py` | **Funkció:** [TODO]
- [ ] `matrix_profile.py` | **Funkció:** [TODO]

##### Simítás és Felbontás (`src/analysis/models/smoothing_and_decomposition`)
- [ ] `ets.py` | **Funkció:** [TODO]
- [ ] `holt_winters.py` | **Funkció:** [TODO]
- [ ] `mstl.py` | **Funkció:** [TODO]
- [ ] `stl.py` | **Funkció:** [TODO]
- [ ] `theta.py` | **Funkció:** [TODO]

##### Spektrális (`src/analysis/models/spectral`)
- [ ] `base_spectral.py` | **Funkció:** [TODO]
- [ ] `dft.py` | **Funkció:** [TODO]
- [ ] `fft.py` | **Funkció:** [TODO]
- [ ] `periodogram.py` | **Funkció:** [TODO]
- [ ] `spectral_analysis.py` | **Funkció:** [TODO]
- [ ] `ssa.py` | **Funkció:** [TODO]
- [ ] `wavelet.py` | **Funkció:** [TODO]
- [ ] `welch.py` | **Funkció:** [TODO]

##### Állapottér (`src/analysis/models/state_space`)
- [ ] `kalman_filter.py` | **Funkció:** [TODO]
- [ ] `state_space.py` | **Funkció:** [TODO]

##### Statisztikai Modellek (`src/analysis/models/statistical_models`)
- [ ] `adida.py` | **Funkció:** [TODO]
- [ ] `arima.py` | **Funkció:** [TODO]
- [ ] `arimax.py` | **Funkció:** [TODO]
- [ ] `auto_arima.py` | **Funkció:** [TODO]
- [ ] `ces.py` | **Funkció:** [TODO]
- [ ] `change_point.py` | **Funkció:** [TODO]
- [ ] `gam.py` | **Funkció:** [TODO]
- [ ] `garch.py` | **Funkció:** [TODO]
- [ ] `ogarch.py` | **Funkció:** [TODO]
- [ ] `quantile_regression.py` | **Funkció:** [TODO]
- [ ] `sarima.py` | **Funkció:** [TODO]
- [ ] `var.py` | **Funkció:** [TODO]
- [ ] `vecm.py` | **Funkció:** [TODO]

##### Szimbolikus Regresszió (`src/analysis/models/symbolic_regression`)
- [ ] `gplearn_model.py` | **Funkció:** [TODO]
- [ ] `pysindy_model.py` | **Funkció:** [TODO]
- [ ] `pysr_model.py` | **Funkció:** [TODO]

##### Topológiai (`src/analysis/models/topological`)
- [ ] `tda.py` | **Funkció:** [TODO]

### Konfiguráció (`src/config`)
- [ ] `models.py` | **Funkció:** [TODO]
- [ ] `parameters.py` | **Funkció:** [TODO]
- [ ] `web_state.json` | **Funkció:** [TODO]

### Adat (`src/data`)
- [ ] `loader.py` | **Funkció:** [TODO]
- [ ] `processor.py` | **Funkció:** [TODO]

### GUI (`src/gui`)
- [ ] `auto_execution_mixin.py` | **Funkció:** [TODO]
- [ ] `auto_window.py` | **Funkció:** [TODO]
- [ ] `help_parser.py` | **Funkció:** [TODO]
- [ ] `main_window.py` | **Funkció:** [TODO]
- [ ] `optuna_widget.py` | **Funkció:** [TODO]
- [ ] `sorrend_data.json` | **Funkció:** [TODO]
- [ ] `sorrend_data.py` | **Funkció:** [TODO]
- [ ] `sound_manager.py` | **Funkció:** [TODO]
- [ ] `translations.py` | **Funkció:** [TODO]
- [ ] `window_config.json` | **Funkció:** [TODO]

#### Eszközök (`src/gui/assets`)
- [ ] `app_icon.ico` | **Funkció:** [TODO]
- [ ] `dark_logo.png` | **Funkció:** [TODO]
- [ ] `light_logo.png` | **Funkció:** [TODO]

#### Hangok (`src/gui/sounds`)
- [ ] `app_close.wav` | **Funkció:** [TODO]
- [ ] `app_start.wav` | **Funkció:** [TODO]
- [ ] `button_click.wav` | **Funkció:** [TODO]
- [ ] `checkbox_off.wav` | **Funkció:** [TODO]
- [ ] `checkbox_on.wav` | **Funkció:** [TODO]
- [ ] `model_complete.wav` | **Funkció:** [TODO]
- [ ] `model_start.wav` | **Funkció:** [TODO]
- [ ] `tab_switch.wav` | **Funkció:** [TODO]
- [ ] `toggle_switch.wav` | **Funkció:** [TODO]

#### Fülek (`src/gui/tabs`)
- [ ] `analysis_tab.py` | **Funkció:** [TODO]
- [ ] `compare_tab.py` | **Funkció:** [TODO]
- [ ] `data_tab.py` | **Funkció:** [TODO]
- [ ] `help_system.py` | **Funkció:** [TODO]
- [ ] `inspection_tab.py` | **Funkció:** [TODO]
- [ ] `performance_tab.py` | **Funkció:** [TODO]
- [ ] `results_tab.py` | **Funkció:** [TODO]

#### Segédek (`src/gui/utils`)
- [ ] `filename_utils.py` | **Funkció:** [TODO]
- [ ] `logging_utils.py` | **Funkció:** [TODO]
- [ ] `shutdown_handler.py` | **Funkció:** [TODO]
- [ ] `text_formatting.py` | **Funkció:** [TODO]

#### Widgetek (`src/gui/widgets`)
- [ ] `circular_progress.py` | **Funkció:** [TODO]
- [ ] `core_heatmap.py` | **Funkció:** [TODO]
- [ ] `gauge.py` | **Funkció:** [TODO]
- [ ] `line_chart.py` | **Funkció:** [TODO]

### Súgó (`src/help`)
- [ ] `01. statistical_models_en.md` | **Funkció:** [TODO]
- [ ] `01. statistical_models_hu.md` | **Funkció:** [TODO]
- [ ] `02. smoothing_decomposition_en.md` | **Funkció:** [TODO]
- [ ] `02. smoothing_decomposition_hu.md` | **Funkció:** [TODO]
- [ ] `03. classical_ml_en.md` | **Funkció:** [TODO]
- [ ] `03. classical_ml_hu.md` | **Funkció:** [TODO]
- [ ] `04. dl_rnn_en.md` | **Funkció:** [TODO]
- [ ] `04. dl_rnn_hu.md` | **Funkció:** [TODO]
- [ ] `05. dl_cnn_hybrid_en.md` | **Funkció:** [TODO]
- [ ] `05. dl_cnn_hybrid_hu.md` | **Funkció:** [TODO]
- [ ] `06. dl_transformer_en.md` | **Funkció:** [TODO]
- [ ] `06. dl_transformer_hu.md` | **Funkció:** [TODO]
- [ ] `07. dl_graph_specialized_en.md` | **Funkció:** [TODO]
- [ ] `07. dl_graph_specialized_hu.md` | **Funkció:** [TODO]
- [ ] `08. meta_learning_automl_en.md` | **Funkció:** [TODO]
- [ ] `08. meta_learning_automl_hu.md` | **Funkció:** [TODO]
- [ ] `09. bayesian_probabilistic_en.md` | **Funkció:** [TODO]
- [ ] `09. bayesian_probabilistic_hu.md` | **Funkció:** [TODO]
- [ ] `10. freq_domain_signal_proc_en.md` | **Funkció:** [TODO]
- [ ] `10. freq_domain_signal_proc_hu.md` | **Funkció:** [TODO]
- [ ] `11. distance_similarity_en.md` | **Funkció:** [TODO]
- [ ] `11. distance_similarity_hu.md` | **Funkció:** [TODO]
- [ ] `12. state_space_filtering_en.md` | **Funkció:** [TODO]
- [ ] `12. state_space_filtering_hu.md` | **Funkció:** [TODO]
- [ ] `13. topological_methods_en.md` | **Funkció:** [TODO]
- [ ] `13. topological_methods_hu.md` | **Funkció:** [TODO]
- [ ] `14. ensemble_methods_en.md` | **Funkció:** [TODO]
- [ ] `14. ensemble_methods_hu.md` | **Funkció:** [TODO]
- [ ] `15. symbolic_regression_en.md` | **Funkció:** [TODO]
- [ ] `15. symbolic_regression_hu.md` | **Funkció:** [TODO]
- [ ] `convert_help_to_html.py` | **Funkció:** [TODO]

### Optimalizáció (`src/optimization`)
- [ ] `objective_functions.py` | **Funkció:** [TODO]
- [ ] `optuna_optimizer.py` | **Funkció:** [TODO]
- [ ] `parameter_spaces.py` | **Funkció:** [TODO]

### Jelentés (`src/reporting`)
- [ ] `exporter.py` | **Funkció:** [TODO]
- [ ] `visualizer.py` | **Funkció:** [TODO]

### Eredmények (`src/results`)

#### Optimalizáció (`src/results/optimization`)
- [ ] `test_results.json` | **Funkció:** [TODO]

### Segédek (`src/utils`)
- [ ] `optional_imports.py` | **Funkció:** [TODO]

### Web (`src/web`)

#### Komponensek (`src/web/components`)
- [ ] `header.py` | **Funkció:** [TODO]
- [ ] `help_popup.py` | **Funkció:** [TODO]
- [ ] `log_panel.py` | **Funkció:** [TODO]
- [ ] `progress.py` | **Funkció:** [TODO]

#### Fülek (`src/web/tabs`)
- [ ] `analysis_tab.py` | **Funkció:** [TODO]
- [ ] `compare_tab.py` | **Funkció:** [TODO]
- [ ] `data_tab.py` | **Funkció:** [TODO]
- [ ] `inspect_tab.py` | **Funkció:** [TODO]
- [ ] `perf_tab.py` | **Funkció:** [TODO]
- [ ] `results_tab.py` | **Funkció:** [TODO]

#### Segédek (`src/web/utils`)
- [ ] `file_handler.py` | **Funkció:** [TODO]
- [ ] `state_manager.py` | **Funkció:** [TODO]
