# MBO Trading Strategy Analyzer - Fejléc és Navigáció

## Tartalomjegyzék

- [02.00 Fejléc funkciók áttekintése](#0200-fejléc-funkciók-áttekintése)
- [02.01 H-K / SZ-P váltó és beviteli mező](#0201-h-k--sz-p-váltó-és-beviteli-mező)
- [02.02 Fülek (Tab-ok) ismertetése](#0202-fülek-tab-ok-ismertetése)
- [02.03 Mute kapcsoló](#0203-mute-kapcsoló)
- [02.04 HU-EN nyelvváltó](#0204-hu-en-nyelvváltó)
- [02.05 Dark Mode kapcsoló](#0205-dark-mode-kapcsoló)

---

# 02.00 Fejléc funkciók áttekintése

## A fejléc felépítése

A program fejléce (header) egy állandó sáv az ablak tetején, amely a legfontosabb navigációs és beállítási elemeket tartalmazza. A fejléc **4 fő zónára** oszlik:

```
┌────────────────────────────────────────────────────────────────────────────┐
│  [LOGO]  │  [H-K/SZ-P Váltó] [Szám] [Eredmény]  │  [TAB GOMBOK]  │  [VEZÉRLŐK]  │
│          │         Bal oldal                    │     Közép      │   Jobb oldal │
└────────────────────────────────────────────────────────────────────────────┘
```

### Zóna 1: Logo (bal szélső)
- **Pozíció**: Bal felső sarok
- **Méret**: 40px magas, arányos szélesség
- **Funkció**: Alkalmazás azonosítás, vizuális branding
- **Téma-érzékeny**: Dark és Light módban eltérő logó jelenik meg

### Zóna 2: Stratégia keresés (bal oldal)
- **H-K / SZ-P kapcsoló**: Mód választó
- **Szám beviteli mező**: 0-6143 közötti érték
- **Eredmény kijelző**: Stratégia beállítások megjelenítése

### Zóna 3: Navigációs gombok (közép)
- **6 Tab gomb**: Data Loading, Analysis, Results, Comparison, Inspection, Performance Test
- **Dinamikus stílus**: Aktív tab kiemelése
- **Hangvisszajelzés**: Tab váltáskor hangjelzés (ha nincs némítva)

### Zóna 4: Globális vezérlők (jobb oldal)
- **Mute checkbox**: Hangok ki/bekapcsolása
- **HU-EN váltó**: Nyelv választó kapcsoló
- **Dark Mode kapcsoló**: Téma váltó

---

# 02.01 H-K / SZ-P váltó és beviteli mező

## Funkció célja

Ez a keresési funkció lehetővé teszi, hogy egy **stratégia sorszáma** alapján gyorsan lekérdezd annak **kereskedési beállításait**. A rendszer 6144 különböző stratégia konfigurációt ismer, mindegyiket egyedi sorszámmal azonosítva.

## Komponensek

### 1. H-K / SZ-P Mód Kapcsoló

```
┌─────────────────────────────────────────┐
│   H-K  [====○    ]  SZ-P               │
│   fehér            szürke              │
│   (aktív)          (inaktív)           │
└─────────────────────────────────────────┘
```

| Mód | Jelentés | Tartomány |
|-----|----------|-----------|
| **H-K** | Hétfő-Kedd (alapértelmezett) | Hét eleji kereskedési stratégiák |
| **SZ-P** | Szerda-Péntek | Hét közepi/végi kereskedési stratégiák |

**Működés**:
- Kapcsoló **balra** (OFF) = H-K mód aktív (fehér címke)
- Kapcsoló **jobbra** (ON) = SZ-P mód aktív (fehér címke)
- Váltáskor hangjelzés (toggle_switch.wav)

### 2. Szám Beviteli Mező

```
┌──────────────┐
│  0000        │  <- Placeholder szöveg
│  [    ]      │  <- Max 4 karakter
└──────────────┘
```

| Tulajdonság | Érték |
|-------------|-------|
| Szélesség | 60 px |
| Max karakterek | 4 |
| Érvényes tartomány | 0 - 6143 |
| Típus | Csak számok |

**Validáció**:
- Automatikusan szűri a nem-numerikus karaktereket
- Ha 6143-nál nagyobb értéket ír be, automatikusan 6143-ra korlátozza
- Valós időben frissíti az eredményt (KeyRelease eseménynél)

### 3. Eredmény Kijelző

```
┌─────────────────────────────────────┐
│    (0-4-50-1)                       │  <- Zöld szöveg, sötét háttér
│    vagy                             │
│    (....)                           │  <- Nincs találat
└─────────────────────────────────────┘
```

| Tulajdonság | Érték |
|-------------|-------|
| Betűtípus | Arial 12, félkövér |
| Szín | Zöld (#2ecc71) |
| Háttér | Sötét (#1a1a2e) |
| Formátum | (X-Y-Z-W) |

**A beállítás formátuma**: `(Paraméter1-Paraméter2-Paraméter3-Paraméter4)`

Ezek a számok a stratégia konkrét kereskedési paramétereit jelentik (belépési/kilépési szabályok, időzítés, stb.).

## Adatforrás

A keresési adatok a `sorrend_data.json` fájlból származnak:

| Adat | Tartalom |
|------|----------|
| **HK_DATA** | 6144 bejegyzés (0-6143 sorszámok) |
| **SZP_DATA** | 6144 bejegyzés (0-6143 sorszámok) |

**Példa bejegyzések**:

| Sorszám | H-K mód | SZ-P mód |
|---------|---------|----------|
| 0 | (0-4-50-1) | (48-52-50-1) |
| 1 | (1-4-50-1) | (49-52-50-1) |
| 2 | (2-4-50-1) | (50-52-50-1) |
| 100 | (0-8-52-2) | (48-56-52-2) |

## Használati útmutató

1. **Válaszd ki a módot**: H-K (hétfői-keddi) vagy SZ-P (szerdai-pénteki)
2. **Írd be a stratégia sorszámát**: 0 és 6143 között
3. **Olvasd le az eredményt**: A zöld mezőben megjelenik a stratégia beállítása
4. Ha `(....)` jelenik meg: Érvénytelen sorszám vagy üres mező

---

# 02.02 Fülek (Tab-ok) ismertetése

A program 6 fő fület tartalmaz, amelyek különböző funkcionális területeket fednek le:

## Fülek áttekintése

| # | Fül neve | Magyar | Rövid leírás |
|---|----------|--------|--------------|
| 1 | **Data Loading** | Adatbetöltés | Excel/Parquet fájlok betöltése és előnézete |
| 2 | **Analysis** | Elemzés | Modell kiválasztás, paraméterezés és futtatás |
| 3 | **Results** | Eredmények | Előrejelzések megjelenítése, rangsorolás és export |
| 4 | **Comparison** | Összehasonlítás | Több modell/jelentés összehasonlító elemzése |
| 5 | **Inspection** | Vizsgálat | Előrejelzések pontosságának ellenőrzése benchmark adatokkal |
| 6 | **Performance Test** | Teljesítmény Teszt | Rendszer erőforrások valós idejű monitorozása |

## Részletes leírások (1-1 mondat)

### 1. Data Loading (Adatbetöltés)
Kereskedési stratégia adatok betöltése Excel vagy Parquet fájlokból, egyedi fájlonként vagy teljes mappából, előnézettel és Parquet konverziós lehetőséggel.

### 2. Analysis (Elemzés)
A 14 kategóriából és 100+ modellből választható előrejelző algoritmus konfigurálása paraméterekkel, végrehajtási mód beállítása (Independent/Panel/Dual), majd az elemzés futtatása párhuzamos feldolgozással.

### 3. Results (Eredmények)
Az elemzés eredményeinek megjelenítése táblázatos és grafikus formában, stratégiák rangsorolása különböző módszerekkel (Standard, Stability, Risk), valamint Markdown/HTML/CSV export lehetőséggel.

### 4. Comparison (Összehasonlítás)
Különböző modellekkel készített jelentések összehasonlító elemzése horizont-specifikus bontásban (1 hét, 1 hónap, 3 hónap, 6 hónap, 1 év), HTML riport generálással.

### 5. Inspection (Vizsgálat)
Korábbi előrejelzések pontosságának utólagos ellenőrzése valós benchmark adatokkal összevetve, rangsor-eltérések és profit különbségek kimutatásával.

### 6. Performance Test (Teljesítmény Teszt)
CPU, GPU és memória használat valós idejű monitorozása grafikus műszerekkel és diagramokkal, beleértve az alkalmazás és a modellek külön terhelését.

## Tab gombok megjelenése

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  [Data Loading] [Analysis] [Results] [Comparison] [Inspection] [Perf Test] │
│       █████                                                                 │
│      (aktív)      (inaktív) (inaktív)  (inaktív)   (inaktív)   (inaktív)   │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Állapot | Háttérszín | Szövegszín |
|---------|------------|------------|
| **Aktív** | #4a4a6a | Fehér (#FFFFFF) |
| **Inaktív** | #2a2a3e | Szürke (#888888) |
| **Hover** | #3d3d5c | - |

---

# 02.03 Mute kapcsoló

## Funkció

A **Mute** checkbox a program összes hangeffektjének be- és kikapcsolására szolgál.

```
┌───────────────────┐
│  [✓] Mute         │  <- Bejelölve = Némítva (alapértelmezett)
│  [ ] Mute         │  <- Nincs bejelölve = Hangok bekapcsolva
└───────────────────┘
```

## Működés

| Állapot | Hatás |
|---------|-------|
| **Bejelölve (Muted)** | Minden hangeffekt ki van kapcsolva |
| **Nincs bejelölve** | Hangeffektek aktívak |

**Alapértelmezett**: Némítva (True)

## Hangeffektek a programban

Amikor a Mute **nincs** bejelölve, a következő hangok játszódnak le:

| Esemény | Hangfájl | Leírás |
|---------|----------|--------|
| Program indítás | `app_start.wav` | Üdvözlő hang |
| Program bezárás | `app_close.wav` | Búcsú hang |
| Tab váltás | `tab_switch.wav` | Navigációs visszajelzés |
| Gomb kattintás | `button_click.wav` | Műveleti visszajelzés |
| Kapcsoló váltás | `toggle_switch.wav` | Beállítás változás |
| Checkbox bejelölés | `checkbox_on.wav` | Opció aktiválás |
| Checkbox kikapcsolás | `checkbox_off.wav` | Opció deaktiválás |
| Modell indítás | `model_start.wav` | Elemzés kezdete |
| Modell befejezés | `model_complete.wav` | Elemzés vége |

## Technikai részletek

- **Hangkezelő**: `SoundManager` singleton osztály
- **Backend**: Windows-on `winsound`, egyéb platformon `pygame`
- **Hangerő**: Alapértelmezett 50%
- **Szálkezelés**: ThreadPoolExecutor (max 2 párhuzamos hang)
- **Hangfájlok helye**: `src/gui/sounds/`

## Perzisztencia

A Mute állapot mentésre kerül a `window_config.json` fájlba, így a program újraindításakor megőrzi a beállítást.

---

# 02.04 HU-EN nyelvváltó

## Funkció

A **HU-EN kapcsoló** lehetővé teszi a felhasználói felület nyelvének váltását magyar és angol között.

```
┌──────────────────────────────────┐
│   HU  [    ○====]  EN            │
│  szürke           fehér          │
│  (inaktív)        (aktív)        │
└──────────────────────────────────┘
```

## Működés

| Kapcsoló pozíció | Nyelv | Aktív címke |
|------------------|-------|-------------|
| **Balra (OFF)** | Magyar (HU) | "HU" fehér |
| **Jobbra (ON)** | Angol (EN) | "EN" fehér |

**Alapértelmezett**: Angol (EN) - kapcsoló jobbra

## Mit fordít le?

A nyelvváltás az alábbi elemeket érinti:

### Fejléc elemek
- Dark Mode felirat
- Application Log címke

### Tab nevek
| Angol | Magyar |
|-------|--------|
| Data Loading | Adatbetöltés |
| Analysis | Elemzés |
| Results | Eredmények |
| Comparison | Összehasonlítás |
| Inspection | Vizsgálat |
| Performance Test | Teljesítmény Teszt |

### Gombok és címkék
- Összes gomb felirat (Run Analysis → Elemzés Futtatása)
- Minden címke és leírás
- Hibaüzenetek és visszajelzések
- Súgó szövegek

### Nem fordítódik
- Modell nevek (maradnak angolul: ARIMA, LSTM, XGBoost, stb.)
- Technikai paraméter nevek
- Log üzenetek (részben angolul maradnak)

## Fordítási rendszer

A fordítások a `translations.py` fájlban találhatók:

```python
TRANSLATIONS = {
    "HU": {
        "Run Analysis": "Elemzés Futtatása",
        "Results": "Eredmények",
        "Category:": "Kategória:",
        # ... 300+ fordítás
    }
}
```

## Technikai működés

```python
def tr(self, text):
    """Translate text based on current language."""
    if self.current_language == "HU":
        return TRANSLATIONS["HU"].get(text, text)
    return text  # Angol = eredeti szöveg
```

Ha egy szöveghez nincs magyar fordítás, az eredeti angol szöveg jelenik meg.

---

# 02.05 Dark Mode kapcsoló

## Funkció

A **Dark Mode kapcsoló** a program teljes színsémájának váltására szolgál sötét és világos téma között.

```
┌─────────────────────────────────┐
│   [====○    ]  Dark Mode        │  <- Bekapcsolva = Sötét téma
│   [    ○====]  Dark Mode        │  <- Kikapcsolva = Világos téma
└─────────────────────────────────┘
```

## Témák összehasonlítása

| Elem | Dark Mode (Sötét) | Light Mode (Világos) |
|------|-------------------|----------------------|
| Háttér | Sötétszürke/fekete | Fehér/világosszürke |
| Szöveg | Fehér/világosszürke | Fekete/sötétszürke |
| Gombok | Sötét tónusok | Világos tónusok |
| Inputok | Sötét háttér | Világos háttér |
| Tab aktív | #4a4a6a | Világosabb árnyalat |
| Logo | dark_logo.png | light_logo.png |

## Alapértelmezett

- **Dark Mode**: Bekapcsolva (alapértelmezett)
- Ez a program indításakor a `main.py`-ban kerül beállításra:

```python
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")
```

## Működés

Amikor a felhasználó váltja a kapcsolót:

1. **Hang**: `toggle_switch.wav` lejátszása (ha nincs némítva)
2. **Téma váltás**: CustomTkinter beépített témaváltó hívása
3. **Logo csere**: Automatikus (CTkImage light/dark képpel)
4. **Log üzenet**: "Theme changed to Dark/Light mode"

```python
def toggle_theme(self):
    self.sound_manager.play_toggle_switch()
    if self.switch_theme.get():
        ctk.set_appearance_mode("Dark")
    else:
        ctk.set_appearance_mode("Light")
```

## Előnyök

### Dark Mode
- Szemkímélő hosszú használat mellett
- Energiatakarékos OLED kijelzőkön
- Professzionális megjelenés
- Kevesebb fényterhelés éjszakai használatkor

### Light Mode
- Jobb olvashatóság erős fényben
- Megszokott dokumentum-stílus
- Nyomtatásbarát előnézet
- Egyes felhasználók preferenciája

## Perzisztencia

**Megjegyzés**: A téma beállítás jelenleg **nem** mentődik a session-ök között. A program mindig Dark módban indul.

---

## Fejléc összefoglaló diagram

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                      │
│  ┌──────┐  ┌─────────────────────────────┐  ┌────────────────────────┐  ┌─────────┐ │
│  │      │  │ H-K [○===] SZ-P  [0000] (.) │  │ [Tab] [Tab] [Tab] ...  │  │[Vezérlők│ │
│  │ LOGO │  │                             │  │                        │  │         │ │
│  │      │  │ Stratégia keresés           │  │ Navigáció              │  │ Mute    │ │
│  └──────┘  └─────────────────────────────┘  └────────────────────────┘  │ HU-EN   │ │
│                                                                          │ Dark    │ │
│  Col 0       Col 1                          Col 2 (weight=1)            └─────────┘ │
│                                                                          Col 3      │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  [████████████████████████████████████████████████████████████░░░░░░░░░░░░]  75%     │
│                              Globális Progress Bar                                   │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

*Dokumentum készítése: 2024.12.29*
*Program verzió: v3.60.0 Stable*
*Fejezet: 02 - Fejléc és Navigáció*
