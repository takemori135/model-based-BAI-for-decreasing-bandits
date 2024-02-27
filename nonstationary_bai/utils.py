import numpy as np
import json
import pickle
from typing import List

def kronecker_prod(x: np.ndarray, y: np.ndarray) -> float:
    dim = x.size
    x = x.reshape((dim, 1))
    y = y.reshape((1, dim))
    return x @ y


def read_dumped_rhos(d: int) -> List[float]:
    fname = f"data/rhos.json"
    with open(fname, "r") as fp:
        dct = json.load(fp)
    return dct[str(d)]


def read_dumped_rhos_cn(d: int) -> List[float]:
    fname = f"data/rhos_condition_number_minimize.json"
    with open(fname, "r") as fp:
        dct = json.load(fp)
    return dct[str(d)]

def read_compatiblity_constants():
    fname = f"data/compatibility_constants.pickle" 
    with open(fname, "rb") as fp:
        return pickle.load(fp)