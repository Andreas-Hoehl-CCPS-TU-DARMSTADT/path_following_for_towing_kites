"""
This module gives global definitions (such as path structure) that can be reused in any other module
"""
from pathlib import Path
import os

# ______ tu colors _______
TU_COLORS = {'1a': '#5D85C3', '9a': '#E9503E', '3a': '#50B695', '7a': '#F8BA3C', '4a': '#AFCC50', '7b': '#F5A300',
             '11a': '#804597'}

# ______ path handling ______
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / 'data'
OPTIMAL_TRAJECTORIES_DIR = DATA_DIR / 'optimal_trajectories'
TRAINING_DATA_DIR = DATA_DIR / 'training_data'
TEST_DATA_DIR = DATA_DIR / 'test_data'
RESIDUAL_Model_DIR = DATA_DIR / 'residual_models'
SIMULATION_DIR = DATA_DIR / 'simulation'
PGF_DIR = DATA_DIR / 'pgf_data'

# create folder structure of data if not available
_dirs = [ROOT_DIR, DATA_DIR, OPTIMAL_TRAJECTORIES_DIR, TRAINING_DATA_DIR, TEST_DATA_DIR, RESIDUAL_Model_DIR,
         SIMULATION_DIR, PGF_DIR]
for _dir in _dirs:
    if not _dir.exists():
        os.makedirs(_dir)
