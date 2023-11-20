from dataclasses import dataclass
from typing import Any

@dataclass
class SARSD():
    """
    MDP transition: [S, A, R, S, D]
    """
    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool

@dataclass
class SSARSSD():
    """
    For taskNet and collisionNet (two different input entries)
    """
    state1: Any
    state2: Any
    action: int
    reward: float
    next_state1: Any
    next_state2: Any
    done: bool

@dataclass
class SARSAD():
    """
    MDP transition: [S, A, R, S, A, D]
    """
    state: Any
    action: int
    reward: float
    next_state: Any
    next_action: int
    done: bool

@dataclass
class SARD():
    """
    MDP transition: [S, A, R, D]
    """
    state: Any
    action: int
    reward: float
    done: bool

@dataclass
class VTLLP():
    """
    Gradients: [Value, Target, Logarithmic policy probability,
    Logarithmic policy probability of certain action, Policy probability]
    """
    value: Any
    target: Any
    log_prob: Any
    log_prob_action: Any
    prob: Any

@dataclass
class SATAP():
    """
    [S, A, Target, Advantage]
    """
    state: Any
    action: int
    target: Any
    advantage: Any
    policy: Any

@dataclass
class SARTAP():
    """
    [S, A, R, Target, Advantage, Policy]
    """
    state: Any
    action: int
    reward: float
    target: Any
    advantage: Any
    policy: Any

@dataclass
class SARTAPE():
    """
    [S, A, R, Target, Advantage, Policy, Effective Recall Factor]
    """
    state: Any
    action: int
    reward: float
    target: Any
    advantage: Any
    policy: Any
    erf: Any
