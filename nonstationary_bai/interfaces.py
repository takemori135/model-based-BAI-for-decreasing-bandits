from typing import List, TypedDict, Optional, Dict
from abc import ABC, abstractmethod
import numpy as np

Arm = int

class Environment(ABC):
    def __init__(self, n_arms: int):
        self.tau_dict = {i: 1 for i in range(n_arms)} 
        self._bc = 0

    def reset(self):
        self.tau_dict = {i: 1 for i in range(self.n_arms)} 
        self._bc = 0

    def loss(self, i: Arm, rng: np.random.Generator) -> float:
        """Noisy loss of arm i.
        """
        tau = self.tau_dict[i]
        loss = self.loss_with_tau(i, tau, rng)
        self.tau_dict[i] += 1
        self._bc += 1
        return loss

    @property
    def budget_consumed(self) -> int:
        return self._bc

    @abstractmethod
    def loss_with_tau(self, i: Arm, tau: int, rng: np.random.Generator) -> float:
        pass

    @abstractmethod
    def best_arm(self, t: int) -> Arm:
        pass

    @abstractmethod
    def name(self) -> str:
        pass

class EnvWithExpectedLoss(Environment):
    @abstractmethod
    def expected_loss(self, i: Arm, tau: int) -> float:
        pass

class Policy(ABC):
    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self._observed_losses: Dict[Arm, List[float]] = {i: [] for i in range(n_arms)}
        self._round = 0

    def reset(self):
        self._observed_losses: Dict[Arm, List[float]] = {i: [] for i in range(self.n_arms)}
        self._round = 0

    def play(self, arm: Arm, env: Environment, rng: np.random.Generator):
        loss = env.loss(arm, rng)

        self._observed_losses[arm].append(loss)
        self._round += 1

    @abstractmethod
    def predict_best_arm(self, 
                          budget: int, 
                          t_tilde: int,
                          env: Environment, 
                          rng: np.random.Generator,
                          ) -> Arm:
        """
        Return estimated best arm with a given budget.
        t_tilde is the time step at which the best arm is defined.
        """
        pass

    @abstractmethod
    def name(self) -> str:
        pass