import numpy as np
import scipy
import pickle
from nonstationary_bai.utils import read_dumped_rhos_cn 
from unittest import TestCase

def sigma_mat_fn(rhos: np.ndarray) -> np.ndarray:
    return np.array([[1/(1 - r - s) for r in rhos] for s in rhos])

def l1_norm(x: np.ndarray) -> float:
    return np.linalg.norm(x, ord=1)

def x0_generator(rng, s: int, d: int, l=1.0) -> np.ndarray:
    b0 = np.random.uniform(low=-1.0, high=1.0, size=(s,))
    b0 = b0 / l1_norm(b0)
    b1 = np.random.uniform(low=-1.0, high=1.0, size=(d - s,))
    l1 = l1_norm(b1)
    if l1 > 1.0 - 1e-5:
        b1 = b1 / l1
        u = rng.uniform(low=0.0, high=l)
        b1 = b1 * u
    return np.concatenate([b0, b1])

def cp_fn(x: np.ndarray, s: int, sigma_mat: np.ndarray) -> float:
    return s * (sigma_mat @ x).dot(x)

def cp_minimize(rhos: np.ndarray, rng, s:int = 2, restart:int = 5, l=1.1, **kwargs):
    sigma_mat = sigma_mat_fn(rhos)
    d = rhos.shape[0]
    
    def const_eq(x):
        return l1_norm(x[:s]) - 1
    def const_ineq(x):
        return l - l1_norm(x[s:])
    def fun(x):
        return cp_fn(x, s, sigma_mat)
    constraints = [{"fun": const_eq, "type": "eq"}, {"fun": const_ineq, "type": "ineq"}]

    opt_res = []
    for i in range(restart):
        x0 = x0_generator(rng, s, d, l)
        assert -1e-8 < const_eq(x0) < 1e-8
        assert const_ineq(x0) > - 1e-8
        opt_res.append(scipy.optimize.minimize(fun, x0, constraints=constraints, **kwargs))
    i = np.argmin([o.fun for o in opt_res])
    return opt_res[i]

def rhos_swapped(rhos: np.ndarray, i) -> np.ndarray:
    if i is None or i == 1:
        return rhos
    assert i > 1
    rhos1 = np.copy(rhos)
    rhos1[1], rhos1[i] = rhos1[i], rhos1[1]
    return rhos1

class TestCC(TestCase):
    def test_compatibility_constant(self):
        res_dct = {}
        rng = np.random.default_rng(seed=0)
        for d in range(3, 11):
            for s_idx in [1, 2]:
                rhos = read_dumped_rhos_cn(d)
                rhos = rhos_swapped(rhos, d - s_idx)
                rhos = np.array(rhos)
                opt_res = cp_minimize(rhos, rng, s=2, restart=10, method="SLSQP")
                res_dct[(d, s_idx)] = {"fun": opt_res.fun, "rhos": rhos, "opt_res": opt_res.x}
        fname = f"data/compatibility_constants.pickle" 
        with open(fname, "wb") as fp:
            pickle.dump(res_dct, fp)