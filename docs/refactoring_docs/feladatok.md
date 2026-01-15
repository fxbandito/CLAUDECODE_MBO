# Projekt Refaktorálás és Optimalizálás Feladatlista

## 1. Fázis: Előkészületek és Audit
- [x] **Kódbázis Struktúra Dokumentum Generálása** <!-- id: 0 -->
    - [x] Minden fájl és mappa listázása a `src`-ben <!-- id: 1 -->
    - [x] `kodbazis_struktura.md` létrehozása interaktív felülvizsgálathoz <!-- id: 2 -->
- [ ] **Kezdeti Kódbázis Audit** <!-- id: 3 -->
    - [ ] Fájl funkciók áttekintése és leírások hozzáadása a `kodbazis_struktura.md`-hez <!-- id: 4 -->
    - [ ] Felesleges docstringek és halott kódok azonosítása <!-- id: 5 -->

## 2. Fázis: Strukturális és Stilisztikai Refaktorálás
- [ ] **Importok Rendezése és Tisztítása** <!-- id: 6 -->
    - [ ] Importok rendezése PEP 8 szerint <!-- id: 7 -->
    - [ ] Nem használt importok eltávolítása <!-- id: 8 -->
- [ ] **Modern Python Szintaxis** <!-- id: 9 -->
    - [ ] `super(Class, self).__init__()` cseréje `super().__init__()`-re <!-- id: 10 -->
- [ ] **Függvény Refaktorálás** <!-- id: 11 -->
    - [ ] Túl sok lokális változóval rendelkező függvények azonosítása <!-- id: 12 -->
    - [ ] Segédfüggvények kiemelése a komplexitás csökkentésére <!-- id: 13 -->

## 3. Fázis: Konfiguráció és Paraméterek
- [ ] **Paraméter Harmonizáció** <!-- id: 14 -->
    - [ ] `parameters.py` auditálása <!-- id: 15 -->
    - [ ] Modell osztályok alapértelmezett paramétereinek auditálása <!-- id: 16 -->
    - [ ] GUI paraméterkezelés auditálása <!-- id: 17 -->
    - [ ] Konzisztencia biztosítása mindhárom rétegben <!-- id: 18 -->

## 4. Fázis: Modell Szervezés és Optimalizáció
- [ ] **Modell Újrakategorizálás** <!-- id: 19 -->
    - [ ] Jelenlegi modell kategóriák felülvizsgálata <!-- id: 20 -->
    - [ ] Új kategorizálási stratégia implementálása <!-- id: 21 -->
- [ ] **Optuna Integráció és Tesztelés** <!-- id: 22 -->
    - [ ] Optuna paraméterek konfigurálása kategóriánként <!-- id: 23 -->
    - [ ] Optuna optimalizáció tesztelése minden kategóriára <!-- id: 24 -->

## 5. Fázis: Dokumentáció és Végső Simítások
- [ ] **Dokumentáció Felülvizsgálat** <!-- id: 25 -->
    - [ ] Docstringek frissítése az új strukturához igazodva <!-- id: 26 -->
    - [ ] Redundáns kommentek eltávolítása <!-- id: 27 -->
- [ ] **Végső Ellenőrzés** <!-- id: 28 -->
    - [ ] Teljes alkalmazás tesztek futtatása <!-- id: 29 -->
    - [ ] Manuális, automata és kötegelt (batch) módok ellenőrzése <!-- id: 30 -->
