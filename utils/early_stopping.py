from dataclasses import dataclass

@dataclass
class EarlyStopper:
    patience: int = 6
    min_delta: float = 1e-4
    best_value: float = float("inf")
    best_step: int = -1
    _wait: int = 0

    def update(self, value: float, step: int) -> tuple[bool, bool]:
        val = float(value)
        if val < self.best_value - self.min_delta:
            self.best_value, self.best_step, self._wait = val, step, 0
            return False, True   
        self._wait += 1
        return self._wait >= self.patience, False
