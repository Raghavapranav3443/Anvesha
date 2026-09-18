@echo off
cd /d "%~dp0.."
:wait
timeout /t 30 /nobreak >nul
findstr /C:"CHAIN COMPLETE" runs\train_chain2.log >nul 2>&1
if errorlevel 1 goto wait
echo [%date% %time%] vqa2 start >> runs\train_chain2.log
python -u scripts\train_vqa.py --epochs 20 --image-size 128 --batch-size 128 --lr 3e-4 >> runs\train_vqa2.log 2>&1
echo [%date% %time%] VQA2 COMPLETE >> runs\train_chain2.log
