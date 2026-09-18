@echo off
REM ============================================================
REM  Definitive post-promotion gate runs (full test sets)
REM  Estimated: ~15-20 min total on the RTX 5050
REM ============================================================
cd /d "%~dp0.."

echo [1/2] LEVIR-CD FULL test (2048 crops, thr=0.85) - gate: IoU ^>= 0.75 / F1 ^>= 0.88
python -c "import sys; sys.path.insert(0,'.'); from scripts.run_benchmarks import eval_levir; import json; r=eval_levir(99999); print(json.dumps(r, indent=1)); json.dump(r, open('artifacts/phase3_levir_fulltest.json','w'))"

echo [2/2] RSVQA FULL test - baseline: 0.7021
python -c "import sys; sys.path.insert(0,'.'); from scripts.run_benchmarks import eval_rsvqa; import json; r=eval_rsvqa(99999); print(json.dumps({k:v for k,v in r.items() if k!='samples'}, indent=1)); json.dump(r, open('artifacts/phase2_rsvqa_fulltest.json','w'))"

echo ALL DONE - results saved to artifacts\phase3_levir_fulltest.json and artifacts\phase2_rsvqa_fulltest.json
pause
