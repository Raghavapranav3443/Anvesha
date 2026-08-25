@echo off
cd /d C:\Users\Rupesh\Desktop\Projects\SatQuery
:wait
timeout /t 30 /nobreak >nul
findstr /C:"VQA2 COMPLETE" runs\train_chain2.log >nul 2>&1
if errorlevel 1 goto wait
echo [%date% %time%] chain3 start >> runs\train_chain3.log
python -u scripts\train_captioner.py --epochs 5 >> runs\train_chain3.log 2>&1
python -u scripts\train_grounding.py --epochs 8 >> runs\train_chain3.log 2>&1
echo [%date% %time%] CHAIN3 COMPLETE >> runs\train_chain3.log
