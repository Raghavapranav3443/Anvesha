@echo off
cd /d "%~dp0.."
echo [%date% %time%] D-chain start > runs\d_chain.log
python -u scripts\train_change.py --epochs 14 --max-pairs 1400 --batch-size 16 --pos-weight 20 --lr 2e-4 >> runs\d_chain.log 2>&1
echo [%date% %time%] change retrain done >> runs\d_chain.log
python -u scripts\train_grounding_point.py --epochs 10 >> runs\d_chain.log 2>&1
echo [%date% %time%] point grounding done - D COMPLETE >> runs\d_chain.log
