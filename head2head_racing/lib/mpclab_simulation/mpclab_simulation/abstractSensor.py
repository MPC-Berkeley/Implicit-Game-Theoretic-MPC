#!/usr/bin python3

from abc import ABC, abstractmethod

import numpy as np

class abstractSensor(ABC):

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def step(self):
        pass

    def add_white_noise(self, x: float, std: float, sigma_max: float = None) -> float:
        if sigma_max is None:
            return x + np.random.normal(0, std)
        else:
            return x + std * np.clip(np.random.normal(), -sigma_max, sigma_max)
