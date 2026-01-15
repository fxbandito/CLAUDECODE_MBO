# Modellek Előrejelzési Mechanizmusának Teljes Körű Elemzése (15 Kategória)

Ez a dokumentum a projektben található **összes (15 db)** modellkategória részletes vizsgálatát tartalmazza. A cél annak tisztázása, hogy az egyes kategóriák a jövőre vonatkozóan (pl. 52 hét) készítenek-e előrejelzést, vagy múltbeli validációt végeznek.

## 🟢 Általános Konklúzió

A teljes kódbázis átvizsgálása után kijelenthető, hogy **két kivételtől eltekintve (Dual Mode, Panel Mode)**, a rendszerben található **összes többi 13 kategória a valós jövőre vonatkozó előrejelzést végez**.

---

## 1. Classical Machine Learning (Független Mód)
*   **Példák:** XGBoost, LightGBM (egyedi stratégiánként futtatva)
*   **Fájl:** `src/analysis/engine.py` (és modell fájlok)
*   **Irány:** ✅ **Jövő**
*   **Működés:** Amikor nem "Panel Mode"-ban futnak, ezek a modellek a `recursive_horizon` (vagy direkt `steps`) paraméter alapján a jövőbeli értékeket becsülik meg a tanított minták alapján.

## 2. Statistical Models
*   **Példák:** ARIMA, Theta, AutoARIMA
*   **Fájl:** `src/analysis/models/statistical_models/arima.py`
*   **Irány:** ✅ **Jövő**
*   **Működés:** A klasszikus statisztikai modellek matematikai definíciójuknál fogva a jövőbeli időszakra (`steps`) vetítik ki a várható értéket a múltbeli autokorreláció alapján.

## 3. Smoothing & Decomposition
*   **Példák:** ETS, Holt-Winters, MSTL
*   **Fájl:** `src/analysis/models/smoothing_and_decomposition/ets.py`
*   **Irány:** ✅ **Jövő**
*   **Működés:**
    *   Az ETS modell (Error, Trend, Seasonal) állapotegyenleteket használ.
    *   A `forecast()` metódus a legutolsó becsült szintből, trendből és szezonalitásból számolja ki a jövőbeli értékeket (`h` lépésre előre).

## 4. Deep Learning - RNN
*   **Példák:** LSTM, GRU, DeepAR
*   **Fájl:** `src/analysis/models/dl_rnn/lstm.py`
*   **Irány:** ✅ **Jövő**
*   **Működés:** A rekurrens hálók a belső "memóriájuk" (hidden state) segítségével lépésről lépésre ("autoregressive" módon) generálják a jövőbeli sorozatot, a saját kimenetüket visszacsatolva.

## 5. Deep Learning - CNN
*   **Példák:** TimesNet, TCN, N-BEATS
*   **Fájl:** `src/analysis/models/dl_cnn/timesnet_batch.py`
*   **Irány:** ✅ **Jövő**
*   **Működés:**
    *   Ezek a modellek (pl. TimesNet) gyakran egy menetben (`pred_len` kimenettel) vagy rekurzívan jósolják meg a jövőt.
    *   A vizsgált `TimesNetBatch` kódja explicit módon `forecast` tömböt épít, és a jövőbe lépteti a bemeneti ablakot.

## 6. Deep Learning - Transformer
*   **Példák:** Transformer, Informer, Autoformer
*   **Fájl:** `src/analysis/models/dl_transformer/transformer.py`
*   **Irány:** ✅ **Jövő**
*   **Működés:** Az "Attention" mechanizmus segítségével a múltbeli releváns pontokból súlyozva állítja elő a jövőbeli sorozatot, jellemzően egy lépésben vagy rekurzívan a teljes horizontra.

## 7. Deep Learning - Graph / Specialized
*   **Példák:** StemGNN, MTGNN, Neural ODE
*   **Fájl:** `src/analysis/models/dl_graph_specialized/stemgnn_batch.py`
*   **Irány:** ✅ **Jövő**
*   **Működés:**
    *   A StemGNN a különböző stratégiák közötti kapcsolatokat (gráf) és az időbeli mintázatokat (spektrális) egyszerre tanulja.
    *   Az `inference` ciklusban a kód (`run_stemgnn_batch`) a jövőbeli lépéseket generálja (`new_step`), és hozzáfűzi a bemenethez a következő lépéshez.

## 8. Meta-Learning
*   **Példák:** MAML, Reptile
*   **Fájl:** `src/analysis/models/meta_learning/meta_learning.py`
*   **Irány:** ✅ **Jövő**
*   **Működés:** A modell a "tanulást tanulja meg", hogy gyorsan alkalmazkodjon az aktuális idősorhoz. A betanulás után a jövőbeli értékeket standard rekurzív módon generálja.

## 9. Probabilistic & Bayesian
*   **Példák:** Gaussian Process, Prophet, BSTS
*   **Fájl:** `src/analysis/models/probabilistic/gaussian_process.py`
*   **Irány:** ✅ **Jövő**
*   **Működés:** Matematikai valószínűségi eloszlásokat illesztenek az adatokra, és ezekből mintavételeznek vagy analitikusan számolnak várható értéket a jövőbeli időpontokra.

## 10. Spectral Analysis
*   **Példák:** SSA (Singular Spectrum Analysis)
*   **Fájl:** `src/analysis/models/spectral/ssa.py`
*   **Irány:** ✅ **Jövő**
*   **Működés:** A jelet frekvencia-komponensekre bontja, majd a legfontosabb komponensekből egy lineáris rekurrencia formula (LRR) segítségével számolja tovább a sorozatot a jövőbe.

## 11. Similarity & Distance
*   **Példák:** KNN (K-Nearest Neighbors), DTW
*   **Fájl:** `src/analysis/models/similarity/knn.py`
*   **Irány:** ✅ **Jövő**
*   **Működés:** "Történelem ismétli önmagát" elv. Megkeresi a múltbeli hasonló szituációkat, és azok *folytatását* átlagolja, hogy megbecsülje a jelenlegi szituáció jövőbeli kimenetelét.

## 12. State Space Models
*   **Példák:** Kalman Filter
*   **Fájl:** `src/analysis/models/state_space/kalman_filter.py`
*   **Irány:** ✅ **Jövő**
*   **Működés:** A rendszer állapotát (szint, tendencia) becsüli, és a fizikai/matematikai modell alapján "vakon" vetíti előre ezt az állapotot a jövőbe.

## 13. Topological Methods
*   **Példák:** TDA (Topological Data Analysis)
*   **Fájl:** `src/analysis/models/topological/tda.py`
*   **Irány:** ✅ **Jövő**
*   **Működés:** Az adatok alakját (topológiáját) vizsgálja, és a kinyert jellemzők alapján egy regressziós modellel becsüli a következő lépést, rekurzívan.

## 14. Ensemble Methods
*   **Példák:** Voting, Weighted Average
*   **Fájl:** `src/analysis/models/ensemble/ensemble.py`
*   **Irány:** ✅ **Jövő**
*   **Működés:** Több másik (jövőbe látó) modell eredményét kombinálja. Mivel az alapmodellek a jövőre jeleznek, az együttes eredmény is a jövőre vonatkozik.

## 15. Symbolic Regression
*   **Példák:** GP-Learn (Genetic Programming)
*   **Fájl:** `src/analysis/models/symbolic_regression/gplearn_model.py`
*   **Irány:** ✅ **Jövő**
*   **Működés:** Egy explicit matematikai képletet (pl. `y = sin(x) + ...`) evolvál, és ebbe a képletbe helyettesíti be lépésről lépésre az értékeket a jövőbeli becsléshez.

---

## ⚠️ Kivételek (Ismétlés)

Az alábbi működések **nem** klasszikus jövőbeli előrejelzések a jelenlegi implementációban:

1.  **Dual Mode (Activity + Profit):**
    *   **Múltbeli Validáció ("Backtest/Walk-Forward"):** Itt a horizont a visszatekintés mértéke. A modell a múltbeli teljesítményét méri, nem a jövőt jósolja.

2.  **Panel Mode (Technikai Korlát):**
    *   **Kényszerített Horizont:** Bár a jövőbe jelez, a kód fixen 52 lépésre kényszeríti a horizontot, figyelmen kívül hagyva a felhasználói beállítást.
