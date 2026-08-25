@echo off
cd /d C:\Users\Rupesh\Desktop\Projects\SatQuery
:wait
timeout /t 60 /nobreak >nul
findstr /C:"D COMPLETE" runs\d_chain.log >nul 2>&1
if errorlevel 1 goto wait
echo [%date% %time%] E: full-data change training >> runs\e_chain.log
python -u scripts\train_change.py --epochs 10 --batch-size 16 --pos-weight 20 --lr 2e-4 >> runs\e_chain.log 2>&1
echo [%date% %time%] E: scorecard refresh >> runs\e_chain.log
python -u -m satquery.evaluate --all --n 300 >> runs\e_chain.log 2>&1
echo [%date% %time%] E COMPLETE >> runs\e_chain.log
