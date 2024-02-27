from nonstationary_bai.interfaces import Arm, Environment, Policy
from abc import abstractmethod
from typing import Dict, List, Optional
from nonstationary_bai.utils import kronecker_prod
import math
import numpy as np
from sklearn.linear_model import Lasso

class SequentialHalvingBase(Policy):
    def __init__(self, n_arms: int):
        super().__init__(n_arms)

    def estimated_losses(self, arms: List[Arm], t_tilde: int, t_r: Optional[int]) -> List[float]:
        return [self.estimated_loss(arm, t_tilde, t_r) for arm in arms] 

    @abstractmethod
    def estimated_loss(self, arm: Arm, t_tilde: int, t_r: Optional[int]) -> float:
        pass

    def predict_best_arm(self, 
                          budget: int, 
                          t_tilde: int,
                          env: Environment, 
                          rng: np.random.Generator) -> Arm:

        good_arms = [i for i in range(self.n_arms)]
        n_phases = math.ceil(math.log2(self.n_arms)) 

        for _ in range(n_phases):
            t_r = budget/(len(good_arms) * n_phases)
            t_r = math.floor(t_r)
            for i in good_arms:
                for _ in range(t_r):
                    self.play(i, env, rng)

            est_losses = np.array(self.estimated_losses(good_arms, t_tilde, t_r))
            sort_index = np.argsort(est_losses)
            m = math.ceil(len(good_arms) / 2)
            new_good_arm_indx = sort_index[sort_index < m] 
            good_arms = [good_arms[i] for i in new_good_arm_indx] 
        if len(good_arms) == 1:
            return good_arms[0]
        if self._round + len(good_arms) <= budget:
            # perform uniform exploration using rest budget
            n = (budget - self._round) // len(good_arms)
            for i in good_arms:
                for _ in range(n):
                    self.play(i, env, rng)
            est_losses = np.array(self.estimated_losses(good_arms, t_tilde, None))
            i = np.argmin(est_losses) 
            return good_arms[i]
        else:
            est_losses = np.array(self.estimated_losses(good_arms, t_tilde, None))
            i = np.argmin(est_losses) 
            return good_arms[i]


class SuccesiveRejectsBase(Policy):
    def __init__(self, n_arms: int):
        super().__init__(n_arms)

    def estimated_losses(self, arms: List[Arm], t_tilde: int) -> List[float]:
        return [self.estimated_loss(arm, t_tilde) for arm in arms] 

    @abstractmethod
    def estimated_loss(self, arm: Arm, t_tilde: int) -> float:
        pass

    def predict_best_arm(self, budget: int, t_tilde: int, env: Environment, rng: np.random.Generator) -> Arm:
        logk_bar = 0.5 + sum(1.0/i for i in range(2, self.n_arms + 1))
        good_arms = [i for i in range(self.n_arms)]
        def n_j(j):
            if j == 0:
                return 0
            return math.ceil((budget - self.n_arms) / ((self.n_arms + 1 - j) * logk_bar))
        for j in range(1, self.n_arms):
            n = n_j(j) - n_j(j - 1)
            for i in good_arms:
                for _ in range(n):
                    self.play(i, env, rng)
            loss_estimates = self.estimated_losses(good_arms, t_tilde)
            idx = np.argmax(loss_estimates)
            del good_arms[idx]
        assert self._round <= budget
        return good_arms[0]


class OLSSequentialHalving(SequentialHalvingBase):
    def __init__(self, rhos: np.ndarray, n_arms: int, lam: float):
        super().__init__(n_arms)
        self.lam = lam
        self.rhos = rhos
        self.dim = rhos.size

    def feature_vec(self, s: int) -> np.ndarray:
        return np.power(s, -self.rhos)

    def estimated_loss(self, arm: Arm, t_tilde: int, t_r: Optional[int]) -> float:
        losses = self._observed_losses[arm]
        tau = len(losses) 
        m = np.eye(self.dim) * self.lam
        feature_vecs = [self.feature_vec(s + 1) for s in range(tau)] 
        for x in feature_vecs:
            m = m + kronecker_prod(x, x)
        theta_hat = np.sum([y * x for x, y in zip(feature_vecs, losses)], axis=0)
        theta_hat = theta_hat.reshape((self.dim, 1))
        theta_hat = np.linalg.inv(m) @ theta_hat
        return self.feature_vec(t_tilde).dot(theta_hat)[0]

    def name(self) -> str:
        return f"OLSSH({self.lam:.2e})"


class LassoSequentialHalving(SequentialHalvingBase):
    def __init__(self, rhos: np.ndarray, n_arms: int, lam: float):
        super().__init__(n_arms)
        self.lam = lam
        self.rhos = rhos
        self.dim = rhos.size

    def feature_vec(self, s: int, tau: int) -> np.ndarray:
        x = np.power(s, -self.rhos)
        norm_factor = np.power(tau, self.rhos)
        return np.multiply(norm_factor, x)

    def estimated_loss(self, arm: Arm, t_tilde: int, t_r: Optional[int]) -> float:
        losses = self._observed_losses[arm]
        tau = len(losses) 
        xs = [self.feature_vec(s + 1, tau) for s in range(tau)]
        xs = np.array(xs).reshape((-1, self.dim))
        alpha = self.lam * 0.5 / math.sqrt(tau)
        est = Lasso(alpha=alpha, fit_intercept=False)
        est.fit(xs, np.array(losses))
        beta = est.coef_
        x = self.feature_vec(t_tilde, tau)
        return beta.dot(x)
    
    def name(self) -> str:
        return f"LASSOSH({self.lam:.2e})"

class SH(SequentialHalvingBase):
    '''
    just Sequential Halving
    '''
    def __init__(self, n_arms: int):
        super().__init__(n_arms)


    def estimated_loss(self, arm: Arm, t_tilde: int, t_r: Optional[int]) -> float:
        if t_r is None:
            losses = self._observed_losses[arm]
        else:
            losses = self._observed_losses[arm][-t_r:]
        return np.mean(losses) 
    
    def name(sefl) -> str:
        return "SH"

class RSH(SequentialHalvingBase):
    """
    A modification of RSR from
    Mussi, M., Montenegro, A., Trovo, F., Restelli, M., and Metelli, A. M. (2023). 
    Best arm identiﬁcation for stochastic rising bandits.
    arXiv preprint arXiv:2302.07510
    """

    def __init__(self, n_arms: int, eps: float):
        super().__init__(n_arms)
        self.eps = eps

    def tau(self, arm: Arm) -> int:
        return len(self._observed_losses[arm]) + 1

    def h_n(self, arm: Arm) -> int:
        return math.floor(self.eps * self.tau(arm))

    def estimated_loss(self, arm: Arm, t_tilde: int, t_r: Optional[int]) -> float:
        h = self.h_n(arm)
        losses = self._observed_losses[arm]
        mu = np.mean(losses[-h:])
        return mu

    def name(self) -> str:
        return f"RSH({self.eps})"


class RSR(SuccesiveRejectsBase):
    def __init__(self, n_arms: int, eps: float):
        super().__init__(n_arms)
        self.eps = eps

    def tau(self, arm: Arm) -> int:
        return len(self._observed_losses[arm]) + 1

    def h_n(self, arm: Arm) -> int:
        return math.floor(self.eps * self.tau(arm))

    def estimated_loss(self, arm: Arm, t_tilde: int) -> float:
        h = self.h_n(arm)
        losses = self._observed_losses[arm]
        mu = np.mean(losses[-h:])
        return mu

    def name(self) -> str:
        return f"RSR({self.eps})"


class RUCBE(Policy):
    """
    Mussi, M., Montenegro, A., Trovo, F., Restelli, M., and Metelli, A. M. (2023). 
    Best arm identiﬁcation for stochastic rising bandits.
    arXiv preprint arXiv:2302.07510
    """
    def __init__(self, n_arms: int, eps: float, expl_param: float, sigma: float) -> None:
        super().__init__(n_arms)
        self.expl_param = expl_param
        self.sigma = sigma
        self.eps = eps

    def tau(self, arm: Arm) -> int:
        return len(self._observed_losses[arm]) + 1

    def h_n(self, arm: Arm) -> int:
        return math.floor(self.eps * self.tau(arm))
    
    def mu(self, arm: Arm, t_tilde: int) -> float:
        h = self.h_n(arm)
        if h == 0:
            return self._observed_losses[arm][-1]
        if 2*h > self.tau(arm):
            return np.mean(self._observed_losses[arm][-h:])

        losses = self._observed_losses[arm]
        mu = np.mean(losses[-h:]) + np.mean([(t_tilde - i) * (losses[-i] - losses[-i - h])/h for i in range(1, h + 1)])
        return mu

    def beta(self, arm: Arm, t_tilde: int) -> float:
        tau = self.tau(arm)
        h = self.h_n(arm)
        if h == 0:
            h = 1
        return self.sigma * (t_tilde - tau + h - 1) * math.sqrt(self.expl_param / h**3)
    
    def lcb(self, arm: Arm, t_tilde: int) -> float:
        return self.mu(arm, t_tilde) - self.beta(arm, t_tilde)

    def predict_best_arm(self, 
                          budget: int, 
                          t_tilde: int,
                          env: Environment, 
                          rng: np.random.Generator) -> Arm:
        # pull all arms once 
        for i in range(self.n_arms):
            self.play(i, env, rng)

        arms = list(range(self.n_arms))
        for _ in range(self.n_arms, budget):
            lcbs = [self.lcb(arm, t_tilde) for arm in arms]
            i = arms[np.argmin(lcbs)]
            self.play(i, env, rng)
       
        lcbs = [self.lcb(arm, t_tilde) for arm in arms]
        i = arms[np.argmin(lcbs)]
        return i
    
    def name(self) -> str:
        return f"RUCBE(eps{self.eps:.2e}_a{self.expl_param:.2e})"