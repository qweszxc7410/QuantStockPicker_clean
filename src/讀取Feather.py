from time import perf_counter

import numpy as np
import os
import pandas as pd

if 1:
    
    df = pd.read_feather(os.path.join(os.path.normpath(os.getcwd()),"data",'rule_data','signal_007_2.feather'))
    print(df)

