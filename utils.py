import pandas as pd
from log_util import log


def read_data(data=None):
    try:
        data = pd.read_csv(data, dtype=object)
    except:
        data = pd.read_excel(data, dtype=object)
    return data