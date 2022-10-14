import numpy as np
import sympy as sp


def driftMatrix(l: sp.Symbol):
    return sp.Matrix(np.array([[1, l, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0],
                               [0, 0, 1, l, 0, 0],
                               [0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 1]]))


def quadMatrix(k1: sp.Symbol, eff_length: float, k_sign: int) -> sp.Matrix:
    """

    :type eff_length: object
    :param k1:
    :return:
    """
    phi = eff_length * sp.sqrt(sp.Abs(k1))

    if k_sign > 0:
        Rquad = sp.Matrix([[            sp.cos(phi)      ,  sp.sin(phi)/sp.sqrt(sp.Abs(k1)),                0             ,              0               , 0, 0],
                           [-sp.sqrt(sp.Abs(k1))*sp.sin(phi),           sp.cos(phi)        ,                0             ,              0               , 0, 0],
                           [                0            ,                0             ,            sp.cosh(phi)      , sp.sinh(phi)/sp.sqrt(sp.Abs(k1)), 0, 0],
                           [                0            ,                0             , sp.sqrt(sp.Abs(k1))*sp.sinh(phi),          sp.cosh(phi)        , 0, 0],
                           [                0            ,                0             ,                0             ,              0               , 1, 0],
                           [                0            ,                0             ,                0             ,              0               , 0, 1]])
    elif k_sign < 0:
        Rquad = sp.Matrix([[           sp.cosh(phi)       , sp.sinh(phi)/sp.sqrt(sp.Abs(k1)),                 0            ,              0              , 0, 0],
                           [ sp.sqrt(sp.Abs(k1))*sp.sinh(phi),           sp.cosh(phi)       ,                 0            ,              0              , 0, 0],
                           [                0             ,                0             ,             sp.cos(phi)      , sp.sin(phi)/sp.sqrt(sp.Abs(k1)), 0, 0],
                           [                0             ,                0             , -sp.sqrt(sp.Abs(k1))*sp.sin(phi),          sp.cos(phi)        , 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]])
    elif k_sign == 0:
        Rquad = sp.Matrix([[1, eff_length, 0,     0     , 0, 0],
                           [0,     1,      0,     0     , 0, 0],
                           [0,     0,      1, eff_length, 0, 0],
                           [0,     0,      0,     1     , 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]])
    return Rquad
