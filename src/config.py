from dataclasses import dataclass

@dataclass
class VSTConfig:
    """Конфігурація параметрів перетворення."""
    a: float = 8.39
    b: float = 1.2
    epsilon: float = 1.0  # Поріг для захисту від log(0)