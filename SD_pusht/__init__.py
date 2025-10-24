# define the root directory of the project
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# the datasets is outside the SD_pusht directory
DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")   # this is the directory where the datasets are stored
# the runs is outside the SD_pusht directory
RUNS_DIR = os.path.join(ROOT_DIR, "runs")   # this is the directory where the runs are stored