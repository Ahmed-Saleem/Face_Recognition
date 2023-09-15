@echo off
:loop
nvidia-smi
timeout /t 1 /nobreak >nul
goto :loop
