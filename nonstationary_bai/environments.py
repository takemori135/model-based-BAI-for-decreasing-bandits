from nonstationary_bai.interfaces import Arm, Environment
import numpy as np
from typing import Tuple, Optional
from nonstationary_bai.utils import read_dumped_rhos

class LinearEnvBase(Environment):
    def __init__(self, thetas: np.ndarray, rhos: np.ndarray, sigma: float):
        '''
        thetas: np.ndarray with shape (K, d)
        rhos: np.ndarray with shape (d,)
        sigma: stddev of the error normal distribution
        '''
        self.n_arms = thetas.shape[0]
        super().__init__(self.n_arms)
        self.dim = rhos.size
        self.rhos = rhos.reshape((self.dim,))
        assert thetas.shape[1] == self.dim
        self.thetas = thetas.reshape((-1, self.dim))
        self.n_arms = thetas.shape[0]
        self.sigma = sigma

    def loss_with_tau(self, i: Arm, tau: int, rng: np.random.Generator) -> float:
        return self.expected_loss_with_tau(i, tau) + rng.normal(loc=0.0, scale=self.sigma)

    def feature_vec(self, tau: int) -> np.ndarray:
        return np.power(tau, -self.rhos)

    def expected_loss_with_tau(self, i: Arm, tau: int) -> float:
        fvec = self.feature_vec(tau) 
        return self.thetas[i].dot(fvec)


class OneGroupofOptimalArms(LinearEnvBase):
    def __init__(self, 
                 n_arms: int, 
                 coeffs: Tuple[float, float, float],
                 index: Tuple[int, int],
                 sigma: float,
                 rhos: Optional[np.ndarray]=None, 
                 dim: Optional[int]=None,
                 ):
        if rhos is not None:
            dim = rhos.size
            self.dim = dim
        else:
            assert dim is not None
            self.dim = dim
            rhos = read_dumped_rhos(dim) 
            rhos = np.array(rhos)
        assert rhos[0] < 1e-10
        thetas = np.zeros((n_arms, dim)) 
        self.coeffs = coeffs
        self.index = index
        self.sigma = sigma
        m, n = index 
        a, b, a1 = coeffs 
        rho1 = rhos[m] 
        rho2 = rhos[n] 
        assert rho1 < rho2
        thetas[0, m] = a
        thetas[0, n] = b
        for i in range(1, n_arms):
            thetas[i, m] = a1 
        
        super().__init__(thetas=thetas, rhos=rhos, sigma=sigma)

    def best_arm(self, t: int) -> Arm:
        losses = [self.expected_loss_with_tau(i, t) for i in range(self.n_arms)] 
        return np.argmin(losses)
    
    def name(self) -> str:
        fname = f"onegrpoptarms_k{self.n_arms}_d{self.dim}_" + \
            f"cf{self.coeffs[0]:.5e}_{self.coeffs[1]:.5e}_{self.coeffs[2]:.5e}_ind{self.index[0]}_{self.index[1]}_sigma{self.sigma:.2e}"
        return fname
    
    def problem_complexity(self, t_tilde: int):
        losses = [self.expected_loss_with_tau(i, t_tilde) for i in range(self.n_arms)] 
        losses = np.sort(losses)
        l = losses[0]
        return sum((losses[i] - l)**(-2) for i in range(1, self.n_arms))

    def t_lower_bound_mussi(self, t_tilde: int):
        losses = [self.expected_loss_with_tau(i, t_tilde) for i in range(self.n_arms)] 
        losses = np.sort(losses)
        l = losses[0]
        deltas = [losses[i] - l for i in range(1, self.n_arms)]
        rho = np.min([rho for rho in self.rhos if rho > 0])
        return sum([delta ** (-1/rho) for delta in deltas])
