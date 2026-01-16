@echo off
REM Release Script - Egyszerű wrapper a release.py-hoz
REM Használat: release.bat 5.4.3
REM           release.bat 5.4.3 "Custom message"

if "%1"=="" (
    echo Hasznalat: release.bat VERSION [MESSAGE]
    echo Pelda:     release.bat 5.4.3
    echo           release.bat 5.4.3 "Dual mode fixes"
    exit /b 1
)

cd /d "%~dp0.."
python scripts\release.py %1 %2 %3 %4 %5
