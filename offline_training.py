# Offline training, running the script even if connection lost.
import os
import shutil
import sys
from pathlib import Path

# os.environ['OMP_NUM_THREADS'] = "1"
# os.environ['CUDA_VISIBLE_DEVICES'] = ""
# os.environ['PATH'] = str(Path.home()) + "/miniforge3/bin" + ':' + os.environ['PATH']
os.environ['PATH'] = str(Path.home()) + "/anaconda3/bin" + ':' + os.environ['PATH']

os.system(f"nohup {sys.executable} {os.getcwd()}/xgb_prediction.py >log.txt &")
os.system('tail -f log.txt')
