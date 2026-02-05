from .alpha import (
    compute_alpha as _compute_alpha,
    compute_left_alpha as _compute_left_alpha,
    compute_right_alpha as _compute_right_alpha,
    function_1 as _function_1,
    function_2 as _function_2,
)
from .beta_hard import (
    compute_beta_hard as _compute_beta_hard,
    compute_parent_beta_hard as _compute_parent_beta_hard,
    make_hard_decision as _make_hard_decision,
)
from .beta_soft import compute_beta_soft as _compute_beta_soft
from .encoding import compute_encoding_step as _compute_encoding_step
from .node_types import NodeTypes, get_node_type

# Simple global flop counter for arithmetic operation estimation
_GLOBAL_FLOPS = 0

def reset_flops():
    global _GLOBAL_FLOPS
    _GLOBAL_FLOPS = 0

def add_flops(n: int):
    global _GLOBAL_FLOPS
    _GLOBAL_FLOPS += int(n)

def get_flops():
    return int(_GLOBAL_FLOPS)

# Wrapper functions that call the optimized numba routines and estimate flops.
def compute_left_alpha(llr):
    res = _compute_left_alpha(llr)
    # compute_left_alpha does N = llr.size/2 outputs; approximate flops per element
    N = llr.size // 2
    # estimate: sign, abs, min/comparison, multiplication ~ 4 flops per element
    add_flops(4 * N)
    return res

def compute_right_alpha(llr, left_beta):
    res = _compute_right_alpha(llr, left_beta)
    N = llr.size // 2
    # estimate: multiply, subtract, multiply, subtract ~ 4 flops per element
    add_flops(4 * N)
    return res

def compute_alpha(a, b):
    res = _compute_alpha(a, b)
    N = a.size
    add_flops(4 * N)
    return res

def function_1(a, b, c):
    res = _function_1(a, b, c)
    N = a.size
    add_flops(6 * N)
    return res

def function_2(a, b, c):
    res = _function_2(a, b, c)
    N = a.size
    add_flops(6 * N)
    return res

def compute_encoding_step(level: int, n: int, source=None, result=None):
    res = _compute_encoding_step(level, n, source, result)
    # encoding does roughly 2*N bit ops per stage; approximate as 2*N flops
    N = result.size if result is not None else source.size
    add_flops(2 * N)
    return res

def compute_beta_hard(*args, **kwargs):
    res = _compute_beta_hard(*args, **kwargs)
    add_flops(1)
    return res

def compute_parent_beta_hard(*args, **kwargs):
    res = _compute_parent_beta_hard(*args, **kwargs)
    add_flops(1)
    return res

def make_hard_decision(*args, **kwargs):
    res = _make_hard_decision(*args, **kwargs)
    add_flops(1)
    return res

def compute_beta_soft(*args, **kwargs):
    res = _compute_beta_soft(*args, **kwargs)
    add_flops(1)
    return res
