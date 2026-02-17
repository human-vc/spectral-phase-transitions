#!/bin/bash
cd /Users/jacobcrainic/projects/spectral-phase-transitions/experiments

echo "Starting run_safe.sh at $(date)" > run_safe.log

# Run parallel
echo "Launching scripts..." >> run_safe.log

/Users/jacobcrainic/projects/spectral-phase-transitions/venv/bin/python -u empirical_phase_boundary.py > phase.log 2>&1 &
PID1=$!
echo "Started empirical_phase_boundary.py with PID $PID1" >> run_safe.log

/Users/jacobcrainic/projects/spectral-phase-transitions/venv/bin/python -u spectral_density.py > density.log 2>&1 &
PID2=$!
echo "Started spectral_density.py with PID $PID2" >> run_safe.log

/Users/jacobcrainic/projects/spectral-phase-transitions/venv/bin/python -u spectral_gap.py > gap.log 2>&1 &
PID3=$!
echo "Started spectral_gap.py with PID $PID3" >> run_safe.log

echo "Waiting for PIDs: $PID1 $PID2 $PID3" >> run_safe.log
wait $PID1 $PID2 $PID3
echo "All scripts finished at $(date)" >> run_safe.log
