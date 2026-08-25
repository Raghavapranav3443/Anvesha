@echo off
cd /d C:\Users\Rupesh\Desktop\Projects\SatQuery
echo [%date% %time%] chain2 start > runs\train_chain2.log
python -u scripts\train_vqa.py --epochs 20 --image-size 128 --batch-size 128 --lr 3e-4 >> runs\train_chain2.log 2>&1
echo [%date% %time%] vqa done >> runs\train_chain2.log
python -u scripts\train_captioner.py --epochs 5 >> runs\train_chain2.log 2>&1
echo [%date% %time%] captioner done >> runs\train_chain2.log
python -u scripts\train_grounding.py --epochs 8 >> runs\train_chain2.log 2>&1
echo [%date% %time%] grounding done - CHAIN COMPLETE >> runs\train_chain2.log
