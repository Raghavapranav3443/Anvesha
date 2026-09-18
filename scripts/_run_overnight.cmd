@echo off
REM ============================================================
REM  Overnight chain: Phase 2 (VQA CLIP A/B gate) -> Phase 3 (joint change training)
REM  Run from a normal terminal:  scripts\_run_overnight.cmd
REM  Logs land in runs\phase2_vqa_gate.log and runs\phase3_train.log
REM  Neither run can harm production weights: both use candidate
REM  checkpoints gated on pre-registered floors/evaluations.
REM ============================================================
cd /d "%~dp0.."

echo [1/2] Phase 2: CLIP-vs-BOW type-heads A/B gate...
python scripts\train_type_heads_clip.py > runs\phase2_vqa_gate.log 2>&1
echo       done, exit=%ERRORLEVEL%

echo [2/2] Phase 3: joint LEVIR+SECOND change training...
python scripts\train_change.py --dataset both --epochs 30 --patience 8 --pos-weight 10 --batch-size 16 --grad-accum 2 > runs\phase3_train.log 2>&1
echo       done, exit=%ERRORLEVEL%

echo ALL DONE. Check runs\phase2_vqa_gate.log and runs\phase3_train.log
pause
