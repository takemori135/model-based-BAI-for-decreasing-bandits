from unittest import TestCase, skip
import json
import numpy as np
import copy
from typing import List, Tuple
import itertools
import scipy.optimize
from nonstationary_bai.experiments import OneGrouofOptArmExp 
from nonstationary_bai.environments import  OneGroupofOptimalArms

def to_dict(a):
    d = vars(a)
    d = copy.deepcopy(d) 
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            d[k] = v.tolist()
    return d


def sigma_mat(rhos1: List[float]) -> np.ndarray:
    rhos = [0] + list(rhos1)
    return np.array([[1/(1 - r1 - r2) for r1 in rhos] for r2 in rhos])

def sigma_mat_det_maximize_grid_search(d: int) -> List[float]:
    rhos1 = np.linspace(0.05, 0.45, 50)
    cands = list(itertools.product(rhos1, repeat=d-1))
    dets = np.array([np.linalg.det(sigma_mat(rho1)) for rho1 in cands])
    i = np.argmax(dets)
    return np.sort([0.0] + list(cands[i])).tolist()

def sigma_mat_det_maximize(d: int, seed=0, repeat=10, method="L-BFGS-B",
                           low=0.05, high=0.45) -> List[float]:
    rng = np.random.default_rng(seed=seed)
    rho0_l = rng.uniform(low=low, high=high, size=(repeat, d-1))
    def fun(x):
        return -np.linalg.det(sigma_mat(x))
    opt_xs = []
    opt_fs = []
    for rho0 in rho0_l:
        res = scipy.optimize.minimize(fun, x0=rho0, method=method,bounds=[(low, high) for _ in range(d-1)])
        opt_fs.append(res.fun)
        opt_xs.append(res.x)
    i = np.argmin(opt_fs) 
    return np.sort([0.0] + list(opt_xs[i])).tolist()


def sigma_mat_cond_num_minimize_grid_search(d: int) -> np.ndarray:
    rhos1 = np.linspace(0.00, 0.45, 50)
    cands = list(itertools.product(rhos1, repeat=d-1))
    def cond_num(m):
        lams = np.linalg.eigh(m)[0] 
        mn = np.min(lams)
        mx = np.max(lams)
        return mx / mn
    vals = np.array([cond_num(sigma_mat(rho)) for rho in cands] )
    i = np.argmin(vals)
    return np.sort([0.0] + list(cands[i])).tolist()


def cond_num(m):
    lams = np.linalg.eigh(m)[0] 
    mn = max(np.min(lams), 0)
    mx = np.max(lams)
    return mx / mn


def sigma_mat_cn_minimize1(d: int, seed=0, repeat=10, method="Nelder-Mead",
                          low=0.00, high=0.45) -> List[float]:
    rng = np.random.default_rng(seed=seed)
    rho0_l = rng.uniform(low=low, high=high, size=(repeat, d-1))
    def fun(x):
        return cond_num(sigma_mat(x))
    opt_xs = []
    opt_fs = []
    for rho0 in rho0_l:
        res = scipy.optimize.minimize(fun, x0=rho0, method=method,bounds=[(low, high) for _ in range(d-1)])
        opt_fs.append(res.fun)
        opt_xs.append(res.x)
    i = np.argmin(opt_fs) 
    return (np.sort([0.0] + list(opt_xs[i])).tolist(), opt_fs[i]) 


def sigma_mat_cn_minimize(d: int, seed=0, repeat=10, methods=["Nelder-Mead", "Powell"],
                          low=0.00, high=0.45) -> Tuple[List[float], float]:
    l = [sigma_mat_cn_minimize1(d, seed=seed, repeat=repeat, method=method, low=low, high=high) for method in methods]
    rhos = [rho for rho, _ in l] 
    vals = [val for _, val in l]
    i = np.argmin(vals)
    return rhos[i]


def binary_search_problem_complexity(env, t_tilde=50):
    obj_val = env.problem_complexity(t_tilde)
    l = 0.89
    u = 0.9
    def problem_complexity(a):
        env = OneGroupofOptimalArms(dim=50, coeffs=[0.8, 0.2, a], index=(10, 40), n_arms=5, sigma=1e-2)
        return env.problem_complexity(50)
    for i in range(100):
        a = (l + u)/2
        pc = problem_complexity(a)
        if pc < obj_val:
            u = a
        else:
            l = a
    return a, abs(pc - obj_val)


class NSBAI(TestCase):
    @skip("done")
    def test_dump_rhos(self):
        rhos_dct = {}
        for d in range(2, 51):
            print(d, flush=True)
            rhos =  sigma_mat_det_maximize(d) 
            rhos_dct[d] = rhos

        fname = f"data/rhos.json"
        with open(fname, "w") as fp:
            json.dump(rhos_dct, fp, indent=True)

    @skip("done")
    def test_dump_rhos_condition_number_minimization(self):
        rhos_dct = {}
        for d in range(2, 11):
            print(d, flush=True)
            rhos = sigma_mat_cn_minimize(d, repeat=100, high=0.99/2)
            rhos_dct[d] = rhos

        fname = f"data/rhos_condition_number_minimize.json"
        with open(fname, "w") as fp:
            json.dump(rhos_dct, fp, indent=True)

    @skip("skip")
    def test_run_onegroupof_opt_arm_exp(self):
        budgets = 50 * np.arange(1, 11)
        for budget in budgets:
            print(budget)
            exp = OneGrouofOptArmExp(
                dim=4, 
                coeffs=[0.5, 0.5, 0.8],
                n_arms=5, 
                index=[1, 3],
                sigma=1e-2,
                budget=budget, t_tilde=budget, repeat=100)
            # exp.run_lasso()
            # exp.run_ols()
            # exp.run_rsh()
            # exp.run_rucbe()
            # exp.run_sh()
            exp.run_rsr()

    @skip("skip")
    def test_run_onegroupof_opt_arm_exp_dim50(self):
        budgets = 50 * np.arange(1, 11)

        env_dim4 = OneGrouofOptArmExp(
            dim=4, 
            coeffs=[0.5, 0.5, 0.8],
            n_arms=5, 
            index=[1, 3],
            sigma=1e-2,
            budget=100, t_tilde=100, repeat=100).env()
        a, err = binary_search_problem_complexity(env_dim4) 
        self.assertLess(err, 1e-10)

        for budget in budgets:
            print(budget, flush=True)
            exp = OneGrouofOptArmExp(
                budget=budget, 
                t_tilde=budget, 
                repeat=100, 
                dim=50, 
                coeffs=[0.8, 0.2, a],
                index=(10, 40), 
                n_arms=5, 
                sigma=1e-2)
            # exp.run_lasso()
            # exp.run_ols()
            # exp.run_rsh()
            # exp.run_rucbe()
            # exp.run_sh()
            exp.run_rsr()
    
    @skip("skip")
    def test_run_onegroupof_opt_arm_exp_dim50_1(self):
        budgets = 50 * np.arange(1, 11)

        for budget in budgets:
            print(budget, flush=True)
            exp = OneGrouofOptArmExp(
                budget=budget, 
                t_tilde=budget, 
                repeat=100, 
                dim=50, 
                coeffs=[0.8, 0.2, 0.9],
                index=(10, 40), 
                n_arms=5, 
                sigma=1e-2)
            exp.run_lasso()
            exp.run_ols()
            exp.run_rsh()
            exp.run_rucbe()
            exp.run_sh()
