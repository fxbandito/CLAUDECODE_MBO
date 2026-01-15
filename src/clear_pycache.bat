@echo off
:: Run PowerShell script as Administrator to clear __pycache__ directories
:: This batch file will elevate to admin and run the PS1 script

cd /d "%~dp0"
powershell -Command "Start-Process powershell -ArgumentList '-ExecutionPolicy Bypass -File \"%~dp0clear_pycache.ps1\"' -Verb RunAs"
