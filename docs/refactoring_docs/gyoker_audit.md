# Gyökér Mappa (`src/`) Audit

Ez a dokumentum a `src/` gyökérkönyvtárban található fájlok részletes elemzését tartalmazza.

## Python Fájlok

### `main.py`
**Funkció:**
Ez az asztali alkalmazás (GUI) fő belépési pontja.
- Inicializálja a környezetet (Julia környezeti változók).
- Beállítja a `cpu_manager`-t.
- Elindítja a `MainWindow` grafikus felületet.

**Belső Szerkezet:**
- **Importok:** `os`, `sys`, `warnings`, `customtkinter`. Késleltetett importok (`cpu_manager`, `main_window`) a rendszerútvonal beállítása után.
- **Függvények:**
    - `main()`: Beállítja a `customtkinter` témáját, letiltja a DPI skálázást, és elindítja a `mainloop`-ot.
- **Megjegyzések/Refaktorálás:**
    - A `warnings.filterwarnings` egy specifikus PyTorch/Python 3.14 hibát kezel – ezt érdemes nyomon követni.
    - A Julia szálkezelés (`_cpu_count // 2`) logikája itt van hardcode-olva.
    - `sys.path.insert` hack a fájl elején.

### `main_debug.py`
**Funkció:**
Hibakeresési (debug) indítófájl. Ugyanazt teszi, mint a `main.py`, de előtte beállítja a `MBO_DEBUG_MODE` környezeti változót.

**Belső Szerkezet:**
- **Importok:** `os`, `main`.
- **Logika:** Csak egy `if __name__ == "__main__":` blokk, ami beállítja a változót és hívja a `main.main()`-t.

### `web_ui.py`
**Funkció:**
A Streamlit alapú webes felület belépési pontja.
- **REFAKTORÁLVA:** A CSS stílusok ki lettek szervezve a `web/assets/styles.css` fájlba.
- **REFAKTORÁLVA:** Az adatbetöltés logikája átkerült a `web/tabs/data_tab.py` fájlba.
- **REFAKTORÁLVA:** `sys.path` hackek eltávolítva, helyette `pyproject.toml` alapú csomagkezelés.

**Belső Szerkezet:**
- **Importok:** Tisztítva. Moduláris importok használata.
- **Függvények:**
    - `load_css()`: Külső CSS betöltése.
    - `check_password()`: Autentikáció.
    - `render_data_tab()`: Hívás a külső modulra.

## Új és Módosított Fájlok (Refaktorálás után)

### `pyproject.toml`
**Funkció:**
A projekt konfigurációs fájlja, amely definiálja a függőségeket és a csomagstruktúrát. Lehetővé teszi a `pip install -e .` használatát, kiváltva a `sys.path` manipulációt.

### `src/web/assets/styles.css`
**Funkció:**
A webes felület összes egyedi CSS stílusát tartalmazza.

### `src/web/tabs/data_tab.py`
**Funkció:**
Az "Adatbetöltés" fül teljes logikája (fájlkezelés, előnézet, feature mode).

## Egyéb Fájlok

### Szkriptek
- **`start_web.bat` / `start_web.sh`**: Parancsfájlok a webes felület indításához. Valószínűleg aktiválják a környezetet és futtatják a `streamlit run src/web_ui.py` parancsot.
- **`install_requirements.bat` / `install_requirements.sh`**: Függőségek telepítése a `requirements.txt`-ből.
- **`clear_pycache.bat` / `clear_pycache.ps1`**: Segédszkriptek a `__pycache__` mappák törlésére (takarítás).
- **`start_tunnel.ps1` / `start_tunnel.sh`**: Cloudflare tunnel indítása (távoli eléréshez).

### Konfiguráció
- **`requirements.txt`**: A Python csomagfüggőségek listája.
- **`auto.txt`**: (Git által mellőzve) Valószínűleg egy helyi konfigurációs fájl az automatikus modellfuttatáshoz, amely tartalmazza a futtatandó modellek listáját.
- **`.streamlit/secrets.toml`**: (Git által mellőzve) Érzékeny adatok (jelszavak, beállítások) a Streamlit alkalmazáshoz.

## Binárisok
- **`cloudflared.exe`**: Cloudflare Tunnel kliens a webes felület publikálásához.

## Összegzés a Refaktoráláshoz
1.  **CSS Leválasztás**: A `web_ui.py` jelentősen tisztítható a CSS kiszervezésével.
2.  **Adatbetöltés Modularizálása**: A `render_data_loading_tab` függvényt ki kell emelni a `web_ui.py`-ból.
3.  **Projekt Útvonal Kezelés**: A `sys.path.insert` és az "ImportError" kezelések helyett érdemes lehet robusztusabb csomagkezelést vagy relatív importokat használni, ha a struktúra engedi.
