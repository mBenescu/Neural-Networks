import numpy as np


class FIR_filter:

    def __init__(self, coefficients):
        self.coefficients = coefficients
        self.filter_length = len(coefficients)
        self.buffer = np.zeros(self.filter_length, dtype=np.float64)

    def filter(self, v):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = v
        return np.dot(self.buffer, self.coefficients)

    def lms(self, error, mu=0.01):
        for j in range(self.filter_length):
            self.coefficients[j] = self.coefficients[j] + error * mu * self.buffer[j]


