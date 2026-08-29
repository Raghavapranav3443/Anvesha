@echo off
REM ============================================================
REM  Phase 2b: captioner A/B gate (CLIP vision vs SceneEncoder)
REM  Run from a normal terminal:  scripts\_run_phase2b.cmd
REM  Two stacks train side-by-side on identical data/order;
REM  the CLIP stack is promoted ONLY if it beats the control.
REM  Production captioner.pt is never touched during the run.
REM  Estimated: 1-2 hours. Log: runs\phase2b_caption_gate.log
REM ============================================================
cd /d c:\Users\Rupesh\Desktop\Projects\SatQuery

echo Phase 2b: captioner A/B gate...
python scripts\train_captioner_clip.py > runs\phase2b_caption_gate.log 2>&1
echo done, exit=%ERRORLEVEL%
echo See runs\phase2b_caption_gate.log and runs\captioner_gate.json
pause
