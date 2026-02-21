#!/bin/bash
while true; do
  count=$(pgrep -f "python.*experiments" | wc -l)
  echo "$(date): Processes: $count"
  ls -lh experiments/results/ | grep "Feb 16 22:05" -v
  sleep 300
done
