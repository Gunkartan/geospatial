import os
import joblib
import rasterio
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.ndimage import uniform_filter