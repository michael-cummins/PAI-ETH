bsub -o "$(pwd)/logs.txt" -n 4 -R "rusage[mem=2048]" python -u checker_client.py --data-dir "$(pwd)" --results-dir "$(pwd)"
