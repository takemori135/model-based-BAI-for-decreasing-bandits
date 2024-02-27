from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
from nonstationary_bai.experiments import OneGrouofOptArmExp
from nonstationary_bai.environments import OneGroupofOptimalArms
from nonstationary_bai.utils import read_dumped_rhos
from nonstationary_bai.policies import LassoSequentialHalving, OLSSequentialHalving, RSH, RUCBE, SH, RSR
import pandas as pd


class PlotData:
    def __init__(self) -> None:
        self.accs = []
        self.linestyle = None
        self.stddevs = []

def line_style(pol):
    if isinstance(pol, RSH):
        return "dashed"
    elif isinstance(pol, RUCBE):
        return "dotted"
    elif isinstance(pol, RSR):
        return "dotted"
    elif isinstance(pol, SH):
        return "dashdot"
    else:
        return "solid"


class PltOneGrouofOptArmExp:
    def __init__(self, 
                 n_arms: int,
                 dim: int, 
                 coeffs: Tuple[float, float, float]=None,
                 index: Tuple[int, int]=None,
                 sigma: float=1e-2,):
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "Helvetica",
            "figure.dpi": 200,
        })

        rhos = np.array(read_dumped_rhos(dim))

        env_args = {"n_arms": n_arms, 
                    "coeffs": coeffs, 
                    "index": index, 
                    "sigma": sigma, 
                    "rhos": rhos,
                    "dim": dim}
        self.env_args = env_args

    def exp(self, budget=100, t_tilde=100, repeat=100):
        exp = OneGrouofOptArmExp(budget=budget, t_tilde=t_tilde, repeat=repeat, **self.env_args) 
        return exp

    def env(self):
        return OneGroupofOptimalArms(**self.env_args) 

    def read_zeroone_arrays(self, 
                        budgets,
                        regs_lasso=[1e-3, 1e-2, 1e-1],
                        regs_ols=[1e-3, 1e-2, 1e-1],
                        epsilons=[0.1, 0.2, 0.3, 0.4, 0.5],
                        explparams=0.1 * np.arange(1, 10),
                        only_best_baselines=True,
                        include_sh=True,
                        ):
        rhos = self.env_args["rhos"]
        n_arms = self.env_args["n_arms"]
        sigma = self.env_args["sigma"]
        pols_ols = [OLSSequentialHalving(rhos=rhos, n_arms=n_arms, lam=reg) for reg in regs_ols]
        pols_lasso = [LassoSequentialHalving(rhos=rhos, n_arms=n_arms, lam=reg) for reg in regs_lasso]
        pols_rsh = [RSH(n_arms=n_arms, eps=eps) for eps in epsilons]
        pols_rsr = [RSR(n_arms=n_arms, eps=eps) for eps in epsilons]
        pols_rucbe = [RUCBE(n_arms=n_arms, eps=0.25, sigma=sigma, expl_param=a) for a in explparams] 
        pols_sh = [SH(n_arms=n_arms)]
        pols_res = []
        zero_one_arrays = []

        def read_zo_ary(pols):
            return [self._read_zo_array(budgets=budgets, pol=pol) for pol in pols]

        pols_res.extend(pols_ols)
        zero_one_arrays.extend(read_zo_ary(pols_ols))

        pols_res.extend(pols_lasso)
        zero_one_arrays.extend(read_zo_ary(pols_lasso))
        if only_best_baselines:
            for pols in [pols_rsh, pols_rsr, pols_rucbe]:
                log, pol = self.best_zero_one_logs(budgets, pols)
                pols_res.append(pol)
                zero_one_arrays.append(log)
        else:
            for pols in [pols_rsh, pols_rsr, pols_rucbe]:
                pols_res.extend(pols)
                zero_one_arrays.extend(read_zo_ary(pols))
        if include_sh:
            pols_res.extend(pols_sh)
            zero_one_arrays.extend(read_zo_ary(pols_sh))
        return pols_res, zero_one_arrays

    def _read_zo_array(self, budgets, pol, repeat=100):
        exps = [self.exp(budget=b, t_tilde=b, repeat=repeat) for b in budgets] 
        return np.array([exp.read_log(pol) for exp in exps])

    def best_zero_one_logs(self, budgets, pols):
        exps = [self.exp(budget=b, t_tilde=b, repeat=100) for b in budgets] 
        mean_arrys = np.array([[np.mean(exp.read_log(pol)) for pol in pols] for exp in exps])
        medians = np.median(mean_arrys, axis=0)
        assert medians.shape == (len(pols),)
        i = np.argmax(medians)
        pol = pols[i]
        return np.array([exp.read_log(pol) for exp in exps]), pol


    def plot_accuracies(self, budgets=None, 
                        **kwargs,
                        ):
        plt.clf()
        fig, ax = plt.subplots()
        if budgets is None:
            budgets = 100 * np.arange(1, 11)
        pols, arys = self.read_zeroone_arrays(budgets=budgets, **kwargs)
        
        for pol, ary in zip(pols, arys):
            accs = ary.mean(axis=1)
            std = ary.std(axis=1)
            n = std.shape[0]
            pol_name = pol.name()
            ls = line_style(pol)
            lb = np.maximum(accs - std, np.zeros(n)) 
            ub = np.minimum(accs + std, np.ones(n)) 
            ax.plot(budgets, accs, label=pol_name, linestyle=ls)
            # ax.errorbar(budgets, y=accs, yerr=[accs - lb, ub - accs], label=pol_name, linestyle=ls, fmt='o', errorevery=2)
            ax.fill_between(budgets, lb, ub, alpha=0.2)
        ax.set_xticks(budgets)
        ax.set_xlabel("Budget")
        ax.set_ylabel("Accuracy")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    def plot_environment_loss_means(self, fname=None, **kwargs):
        plt.clf()
        fig, ax = plt.subplots(**kwargs)
        env = self.env()
        ts = np.arange(1, 100)
        losses = np.array([env.thetas.dot(env.feature_vec(tau)) for tau in ts])
        ax.plot(losses[:, 0], label="Arm 1")
        ax.plot(losses[:, 1], label="Arm 2...5")
        ax.set_xlabel(r"$\tau$")
        ax.set_ylabel("Expected Loss")
        plt.legend()
        if fname is not None:
            plt.savefig(fname, dpi=300, bbox_inches="tight")

    def plot_diff_means(self):
        plt.clf()
        fig, ax = plt.subplots()
        env = self.env()
        ts = np.arange(1, 500)
        losses = np.array([env.thetas.dot(env.feature_vec(tau)) for tau in ts])
        ax.plot(np.abs(np.abs(losses[:, 0] - losses[:, 1])))
        ax.set_ylabel("ABS. Diff. of Expected Losses")

    def plot_problem_complexities(self, ts):
        plt.clf()
        fig, ax = plt.subplots()
        env = self.env()
        pcs = [env.problem_complexity(t) for t in ts] 
        ax.plot(ts, pcs)
        ax.set_xlabel(r"$\tilde{T}$")
        ax.set_ylabel(r"$H(\tilde{T})$")


def policy_name(pol):
    if isinstance(pol, LassoSequentialHalving):
        return f"LASSO-SH({pol.lam:.1e})"
    elif isinstance(pol, OLSSequentialHalving):
        return f"LS-SH({pol.lam:.1e})"
    elif isinstance(pol, RSH):
        return "RSH"
    elif isinstance(pol, RSR):
        return "RSR"
    elif isinstance(pol, RUCBE):
        return "RUCBE"
    elif isinstance(pol, SH):
        return "SH"
    else:
        raise RuntimeError()


def make_d4_df():
    plt_expd4 = PltOneGrouofOptArmExp(dim=4, coeffs=[0.5, 0.5, 0.8], n_arms=5, index=[1, 3], sigma=1e-2,)
    return make_df_base(plt_expd4)

def make_d50_df():
    plt_expd50 = PltOneGrouofOptArmExp(dim=50, coeffs=[0.8, 0.2,0.8954915084635513], index=(10, 40), n_arms=5, sigma=1e-2)
    return make_df_base(plt_expd50)

def make_df_base(plt_exp):
    d = {"method": [], "accuracy": [], "budget": []}
    budgets = 50 * np.arange(1, 11)
    pols, arys = plt_exp.read_zeroone_arrays(budgets=budgets, regs_lasso=[1e-3, 1e-2, 1e-1], regs_ols=[1e-3, 1e-2, 1e-1])
    for pol, ary in zip(pols, arys):
        for i, b in zip(range(10), budgets):
            a = ary[i,:]
            assert a.shape == (100,)
            d["method"].extend([policy_name(pol) for _ in range(100)])
            d["accuracy"].extend(a.tolist())
            d["budget"].extend([b for _ in range(100)])
    return pd.DataFrame(d)