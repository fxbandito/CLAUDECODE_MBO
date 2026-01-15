# Refaktorálási és Optimalizálási Megvalósítási Terv

# Cél Leírása
A Gemini MBO kódbázis átfogó refaktorálásának végrehajtása a kódminőség, karbantarthatóság, teljesítmény és felhasználói élmény javítása érdekében. Ez magában foglalja a kódbázis teljes auditálását, a halott kódok tisztítását, az importok optimalizálását, a szintaxis modernizálását, valamint a modell kategorizálás és az Optuna integráció fejlesztését.

## Felhasználói Felülvizsgálat Szükséges
> [!IMPORTANT]
> A `kodbazis_struktura.md` fájlban lévő minden egyes fájl "Kódbázis Funkciójának" ellenőrzése az Ön bevonását igényli. Ezen interaktívan fogunk végigmenni.

## Javasolt Módosítások

### 1. Fázis: Előkészületek és Audit (Jelenlegi)
- **Struktúra Dokumentum**: `kodbazis_struktura.md` elkészítve, amely listázza az összes `src` fájlt.
- **Teendő**: Végigmenni minden fájlon a `kodbazis_struktura.md`-ben:
    - [ ] Dokumentálni a fájl pontos funkcióját.
    - [ ] Megjegyzéseket fűzni a szükséges refaktoráláshoz vagy hibákhoz.

### 2. Fázis: Strukturális és Stilisztikai Refaktorálás
#### [MÓDOSÍTÁS] Összes Python Fájl
- **Import Tisztítás**:
    - Importok rendezése (Standard könyvtár -> Harmadik fél -> Helyi alkalmazás).
    - Nem használt importok eltávolítása.
    - Körkörös függőségek feloldása, ha vannak.
- **Modern Szintaxis**:
    - `super(Class, self).__init__()` cseréje `super().__init__()`-re.
- **Függvény Refaktorálás**:
    - Túl sok lokális változóval (>15) rendelkező függvények azonosítása.
    - Komplex függvények felbontása kisebb segédfüggvényekre.
    - Nem használt változók és halott kód eltávolítása.
    - Redundáns docstringek tisztítása (csak a lényeges dokumentáció maradjon).

### 3. Fázis: Konfiguráció és Paraméterek
#### [MÓDOSÍTÁS] `src/config/parameters.py`, `src/gui/`, Modell Osztályok
- **Harmonizáció**:
    - Biztosítani, hogy a `src/config/parameters.py` legyen az alapértelmezések egyetlen igazságforrása.
    - Modell osztályok (`src/analysis/models/...`) frissítése ezen alapértelmezések használatára.
    - GUI komponensek (`src/gui/`) frissítése az alapértelmezések tiszteletben tartására.
    - Biztosítani, hogy a GUI paraméter változtatások helyesen jussanak el a végrehajtó motorhoz.

### 4. Fázis: Modell Szervezés és Optimalizáció
#### [MÓDOSÍTÁS] `src/analysis/models/` és `src/optimization/`
- **Újrakategorizálás**:
    - `src/config/models.py` (vagy ahol a kategóriák definiálva vannak) felülvizsgálata.
    - Modellek logikus csoportosítása (pl. család, adatigény szerint).
- **Optuna Integráció**:
    - `src/optimization/parameter_spaces.py` kiegészítése minden modell kategóriához.
    - Biztosítani, hogy az `Optuna Optimizer` helyesen működjön az új kategóriákkal.
    - `tests/test_optuna_categories.py` létrehozása a kategóriák futásának ellenőrzésére.

### 5. Fázis: Dokumentáció és Végső Simítások
- **Docstringek**: Biztosítani, hogy minden publikus modul, osztály és függvény rendelkezzen világos docstringgel.
- **Kommentek**: "Kikommentált kódok" eltávolítása, kivéve, ha szigorúan szükségesek jövőbeli hivatkozáshoz (és ekként vannak jelölve).

## Ellenőrzési Terv

### Automatizált Tesztek
- **Linting**:
    - Pylint/flake8 futtatása a szintaxis és stílus javítások ellenőrzésére.
    - `pylint src/ --rcfile=.pylintrc` (Létrehozás, ha hiányzik).
- **Unit Tesztek**:
    - Meglévő tesztek futtatása: `python -m unittest discover tests` (ha létezik).
    - **ÚJ**: `tests/test_imports.py` létrehozása a körkörös importok ellenőrzésére.
    - **ÚJ**: `tests/test_optuna_integration.py` létrehozása a paramétergenerálás ellenőrzésére.

### Manuális Ellenőrzés
- **Kódbázis Áttekintés**:
    - Ön és én manuálisan átnézzük a `kodbazis_struktura.md` haladását.
- **Alkalmazás Teszt**:
    - GUI indítása: `run_gui.bat` (vagy ekvivalens).
    - Ellenőrizni, hogy minden fül betöltődik.
    - Próba elemzés futtatása minden új kategóriából egy modellel.
    - Eredmények helyes megjelenítésének ellenőrzése.
