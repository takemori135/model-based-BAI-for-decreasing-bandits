from nonstationary_bai.interfaces import Environment, Policy
from typing import List, Tuple
import numpy as np
import pickle
import json
from nonstationary_bai.utils import read_dumped_rhos
from nonstationary_bai.policies import LassoSequentialHalving, OLSSequentialHalving, RUCBE, RSH, SH, RSR
from nonstationary_bai.environments import OneGroupofOptimalArms
from nonstationary_bai.interfaces import Policy


class Log:
    def __init__(self):
        self.estimated_best_arm: int = None
        self.best_arm = None

def run_experiment(policy: Policy, 
                   env: Environment, 
                   budget: int,
                   t_tilde: int,
                   rng: np.random.Generator) -> Log:
    policy.reset()
    env.reset()
    log = Log()
    best_arm = env.best_arm(t_tilde)
    log.best_arm = int(best_arm)
    arm = int(policy.predict_best_arm(budget, t_tilde, env, rng))
    log.estimated_best_arm = arm
    assert env.budget_consumed <= budget
    return log

class OneGrouofOptArmExp:
    def __init__(self,  budget=100, t_tilde=100, repeat=100, 
                 seed=0,
                 n_arms=5,
                 dim=4, 
                 coeffs=None,
                 index=[1, 3],
                 sigma=1e-2,
                 rhos=None,
                 ):
        self.budget = budget
        self.t_tilde = t_tilde
        self.repeat = repeat
        self.seed = seed
        if rhos is not None:
            rhos = np.array(read_dumped_rhos(dim))
        env_args = {"n_arms": n_arms, 
                    "rhos": rhos, 
                    "coeffs": coeffs, 
                    "index": index, 
                    "sigma": sigma, 
                    "dim": dim, 
                    }
        self.env_args = env_args

    def env(self):
        return OneGroupofOptimalArms(**self.env_args)

    def dump_gen(self, pol_fn):
        rng = np.random.default_rng(seed=self.seed)
        env = OneGroupofOptimalArms(**self.env_args)
        fname = self.fname(pol_fn()) 
        res = []
        for i in range(self.repeat):
            env = OneGroupofOptimalArms(**self.env_args)
            l = run_experiment(pol_fn(), env, budget=self.budget, t_tilde=self.t_tilde, rng=rng)
            res.append(vars(l))
        with open(fname, "w") as fp:
            json.dump(res, fp)
    
    def fname(self, pol):
        env = OneGroupofOptimalArms(**self.env_args)
        fname = f"data/pol{pol.name()}_{env.name()}_bd{self.budget}_tt{self.t_tilde}_sd{self.seed}_nr{self.repeat}.json"
        return fname

    def run_ols(self):
        n_arms = self.env_args["n_arms"]
        d = self.env_args["dim"]
        rhos = np.array(read_dumped_rhos(d))
        for lam in [1e-3, 1e-2, 1e-1, 1.0]:
            def pol_fn():
                return OLSSequentialHalving(rhos=rhos, n_arms=n_arms, lam=lam) 
            self.dump_gen(pol_fn)

    def run_lasso(self):
        n_arms = self.env_args["n_arms"]
        d = self.env_args["dim"]
        rhos = np.array(read_dumped_rhos(d))
        for lam in [1e-3, 1e-2, 1e-1, 1.0]:
            def pol_fn():
                return LassoSequentialHalving(rhos=rhos, n_arms=n_arms, lam=lam) 
            self.dump_gen(pol_fn)

    def run_rucbe(self):
        n_arms = self.env_args["n_arms"]
        sigma = self.env_args["sigma"]
        expl_params = np.arange(1, 10) * 1e-1
        # expl_params = np.arange(2, 10) * 1e-1
        for a in expl_params:
            def pol_fn():
                return RUCBE(n_arms=n_arms, eps=0.25, expl_param=a, sigma=sigma)
            self.dump_gen(pol_fn)

    def run_rsh(self):
        n_arms = self.env_args["n_arms"]
        for eps in [0.1, 0.2, 0.3, 0.4, 0.5]:
            def pol_fn():
                return RSH(n_arms=n_arms, eps=eps)
            self.dump_gen(pol_fn)

    def run_rsr(self):
        n_arms = self.env_args["n_arms"]
        for eps in [0.1, 0.2, 0.3, 0.4, 0.5]:
            def pol_fn():
                return RSR(n_arms=n_arms, eps=eps)
            self.dump_gen(pol_fn)

    def run_sh(self):
        n_arms = self.env_args["n_arms"]
        def pol_fn():
            return SH(n_arms=n_arms)
        self.dump_gen(pol_fn)


    def print_acc(self):
        rhos = self.env_args["rhos"]
        n_arms = self.env_args["n_arms"]
        expl_params = [1e+2, 1e+3, 1e+4]
        sigma = self.env_args["sigma"]
        pols = [
            OLSSequentialHalving(rhos=rhos, n_arms=n_arms, lam=1e-2),
            LassoSequentialHalving(rhos=rhos, n_arms=n_arms, lam=1e-2),
                ]
        pols.extend([RSH(n_arms=n_arms, eps=eps) for eps in [0.1, 0.2, 0.3, 0.4, 0.5]])
        pols.extend([RUCBE(n_arms=n_arms, eps=0.25, sigma=sigma, expl_param=a) for a in expl_params])
        for pol in pols:
            fname = self.fname(pol)
            with open(fname, "rb") as fp:
                res = pickle.load(fp)
                print(f"{pol.name()}: {accuracy(res)}")

    def results_zero_one_arrays(self, 
                                regs_lasso=[1e-2, 1e-1],
                                regs_ols=[1e-2, 1e-1],
                                include_sh=True,
                                expl_params=np.arange(1, 10, 2) * 1e-1,
                                epsilons=[0.1, 0.2, 0.3],
                                ):
        rhos = self.env_args["rhos"]
        n_arms = self.env_args["n_arms"]
        sigma = self.env_args["sigma"]
        pols = []
        pols.extend([OLSSequentialHalving(rhos=rhos, n_arms=n_arms, lam=reg) for reg in regs_ols])
        pols.extend([LassoSequentialHalving(rhos=rhos, n_arms=n_arms, lam=reg) for reg in regs_lasso])
        for eps in epsilons:
            pols.append(RSH(n_arms=n_arms, eps=eps))
        for eps in epsilons:
            pols.append(RSR(n_arms=n_arms, eps=eps))
        for a in expl_params:
            pols.append(RUCBE(n_arms=n_arms, eps=0.25, sigma=sigma, expl_param=a)) 
        if include_sh:
            pols.append(SH(n_arms=n_arms))
        res = []
        for pol in pols:
            fname = self.fname(pol)
            with open(fname, "r") as fp:
                arrys = logs_to_zero_one_array(json.load(fp))
            res.append((pol, arrys))
        return res

    def read_log(self, pol) -> np.ndarray:
        fname = self.fname(pol)
        with open(fname, "r") as fp:
            arrys = logs_to_zero_one_array(json.load(fp))
        return arrys
    
def logs_to_zero_one_array(logs) -> np.ndarray:
    def zero_one(d):
        if d["estimated_best_arm"] == d["best_arm"]:
            return 1.0
        else:
            return 0.0
    return np.array([zero_one(l) for l in logs])

def accuracy(logs) -> float:
    return np.mean(logs_to_zero_one_array(logs))