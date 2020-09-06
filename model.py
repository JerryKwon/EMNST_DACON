# model.py
# cnn models for emnist dataset

import pandas as pd
import numpy as np
import os
import platform
import warnings

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold


warnings.filterwarnings("ignore")