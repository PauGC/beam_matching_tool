#!/usr/bin/env python3

from typing import Tuple
import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy.constants import c as const_c
from scipy.optimize import minimize_scalar
import sys
import os
from pathlib import Path
from typing import Union
from io import StringIO
from ocelot import *


# functions not supporting multiple axes:
# ======================================================================================================================
def driftMatrix(l: float):
    return np.matrix(np.array([[1, l, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0],
                               [0, 0, 1, l, 0, 0],
                               [0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 1]]))


def quadMatrix(k1: float, eff_length: float, thin_lens: bool = False) -> np.ndarray:
    """

    :type eff_length: object
    :param k1:
    :return:
    """
    K = k1 * eff_length
    phi = eff_length * np.sqrt(abs(k1))

    if thin_lens == False:
        if k1 > 0:
            Rquad = np.array([[            np.cos(phi)      ,  np.sin(phi)/np.sqrt(abs(k1)),                0             ,              0               , 0, 0],
                              [-np.sqrt(abs(k1))*np.sin(phi),           np.cos(phi)        ,                0             ,              0               , 0, 0],
                              [                0            ,                0             ,            np.cosh(phi)      , np.sinh(phi)/np.sqrt(abs(k1)), 0, 0],
                              [                0            ,                0             , np.sqrt(abs(k1))*np.sinh(phi),          np.cosh(phi)        , 0, 0],
                              [                0            ,                0             ,                0             ,              0               , 1, 0],
                              [                0            ,                0             ,                0             ,              0               , 0, 1]])
        elif k1 < 0:
            Rquad = np.array([[           np.cosh(phi)       , np.sinh(phi)/np.sqrt(abs(k1)),                 0            ,              0              , 0, 0],
                              [ np.sqrt(abs(k1))*np.sinh(phi),           np.cosh(phi)       ,                 0            ,              0              , 0, 0],
                              [                0             ,                0             ,             np.cos(phi)      , np.sin(phi)/np.sqrt(abs(k1)), 0, 0],
                              [                0             ,                0             , -np.sqrt(abs(k1))*np.sin(phi),          np.cos(phi)        , 0, 0],
                              [0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 1]])
        elif k1 == 0:
            Rquad = np.array([[1, eff_length, 0,     0     , 0, 0],
                              [0,     1,      0,     0     , 0, 0],
                              [0,     0,      1, eff_length, 0, 0],
                              [0,     0,      0,     1     , 0, 0],
                              [0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 1]])

    elif thin_lens == True:
        if k1 != 0:
            Rquad = np.array([[ 1 , 0, 0, 0],
                              [ -K, 1, 0, 0],
                              [ 0 , 0, 1, 0],
                              [ 0 , 0, K, 1]])
        elif k1 == 0:
            Rquad = np.array([[1, eff_length,  0,     0     ],
                              [0,      1,      0,     0     ],
                              [0,      0,      1, eff_length],
                              [0,      0,      0,     1     ]])
    return Rquad


def dipoleSectorMatrix(angle: float, length: float, energy: float, axis: str = 'x'):
    gamma = energy / m_e_MeV
    rho = length/angle
    if abs(angle) < 1e-4:
        R = driftMatrix(length)
        return R
    else:
        if axis.lower() == 'x':
            R = np.array([[np.cos(angle)           , rho*np.sin(angle)      , 0, 0        , 0, rho*(1 - np.cos(angle))                           ],
                          [-1 / rho * np.sin(angle), np.cos(angle)          , 0, 0        , 0, np.sin(angle)                                     ],
                          [0                       , 0                      , 1, rho*angle, 0, 0                                                 ],
                          [0                       , 0                      , 0, 1        , 0, 0                                                 ],
                          [-np.sin(angle)          , rho*(np.cos(angle) - 1), 0, 0        , 1, rho*angle/gamma**2 - rho*angle + rho*np.sin(angle)],
                          [0                       , 0                      , 0, 0        , 0, 1                                                 ]])
        elif axis.lower() == 'y':
            R = np.array([[1, rho*angle, 0                       , 0                      , 0, 0                                                 ],
                          [0, 1        , 0                       , 0                      , 0, 0                                                 ],
                          [0, 0        , np.cos(angle)           , rho*np.sin(angle)      , 0, rho*(1 - np.cos(angle))                           ],
                          [0, 0        , -1 / rho * np.sin(angle), np.cos(angle)          , 0, np.sin(angle)                                     ],
                          [0, 0        , -np.sin(angle)          , rho*(np.cos(angle) - 1), 1, rho*angle/gamma**2 - rho*angle + rho*np.sin(angle)],
                          [0, 0        , 0                       , 0                      , 0, 1                                                 ]])
        return R


def dipole_kick(arc_length: float, field: np.ndarray, energy: float, beta_rel: float = 1.0):
    """

    :param arc_length: effective
    :param field: in Tesla
    :param energy: in GeV
    :param beta_rel: in general 1.0...
    :return: kick in mrad
    """
    rho_inv = const_c * 1e-9 * field / (beta_rel * energy)
    theta = arc_length * rho_inv  # rad
    return theta


def twissTransportMatrix(R: np.ndarray):
    """
    Generates the matrix B to get the squared beam size in the measuring plane from the the beam matrix
    elements (<x0^2>, <x0*x0'>, <x0'^2>) at the reference plane (s_0) and the transfer matrix between the two planes
    R(s0 --> s) for different quadrupole strengths.

        <x(1)^2>     R11(1)^2  2*R11(1)*R12(1)  R12(1)^2
        <x(2)^2>     R11(2)^2  2*R11(2)*R12(2)  R12(2)^2
        <x(3)^2>     R11(3)^2  2*R11(3)*R12(3)  R12(3)^2       <x0^2>
           .      =     .             .            .       *   <x0*x0'>
           .            .             .            .           <x0'^2>
           .            .             .            .
        <x(n)^2>     R11(n)^2  2*R11(n)*R12(n)  R12(n)^2

    Parameters
    ----------
    k_values : np.ndarray
        1D array containing the different k1 values of the quadrupole with which the scan is done.
    eff_length : float
        Effective length of the quadrupole. It can be found in the middlelayer server. For Q6FLFMAFF = 0.270
    drift : float
        Drift between the quadrupole and the OTR or scintillator screen. Between Q6FLFMAFF and the screen Q8FLFMAFF the
        drift is 2.612 m.
    axis : str ('x' or 'y')

    Returns
    -------
    out : np.ndarray
        Matrix containing the transformation between the beam matrix elements at the reference plane (s_0) and
        the beam size at the measurement plane (s_1).
    """
    if R.shape == (6, 6):
        ttm_x = np.array([[ R[0, 0] ** 2     , -2 * R[0, 0] * R[0, 1]                ,  R[0, 1] ** 2     ],
                          [-R[0, 0] * R[1, 0],  R[0, 1] * R[1, 0] + R[0, 0] * R[1, 1], -R[0, 1] * R[1, 1]],
                          [ R[1, 0] ** 2     , -2 * R[1, 0] * R[1, 1]                ,  R[1, 1] ** 2     ]])
        ttm_y = np.array([[ R[2, 2] ** 2     , -2 * R[2, 2] * R[2, 3]                ,  R[2, 3] ** 2     ],
                          [-R[2, 2] * R[3, 2],  R[2, 3] * R[3, 2] + R[2, 2] * R[3, 3], -R[2, 3] * R[3, 3]],
                          [ R[3, 2] ** 2     , -2 * R[3, 2] * R[3, 3]                ,  R[3, 3] ** 2     ]])
        out = np.zeros((6, 6))
        out[:3, :3] = ttm_x
        out[3:, 3:] = ttm_y
        return out
    elif R.shape == (2, 2):
        Rt = np.array([[R[0, 0] ** 2, -2 * R[0, 0] * R[0, 1], R[0, 1] ** 2],
                       [-R[0, 0] * R[1, 0], R[0, 1] * R[1, 0] + R[0, 0] * R[1, 1], -R[0, 1] * R[1, 1]],
                       [R[1, 0] ** 2, -2 * R[1, 0] * R[1, 1], R[1, 1] ** 2]])
        return Rt


def calc_emittance(R: Union[list, np.ndarray], beamsizes: np.ndarray, beam_energy: float):
    """

    :param R:
    :param beamsizes: in mm
    :param beam_energy: in GeV
    :return:
    """
    if type(R) == list:
        R = np.array(R)
    if len(R.shape) == 3 and R.shape[1:] == (6, 6):
        beamsizes = beamsizes * 1e-3
        sig2_x = np.nanmean(beamsizes[:, :, 0], axis=1) ** 2
        var_sig2_x = 2 * np.nanmean(beamsizes[:, :, 0], axis=1) * np.nanstd(beamsizes[:, :, 0], axis=1)
        b_x = sig2_x / var_sig2_x
        B_x = (np.vstack((R[:, 0, 0] ** 2, 2 * R[:, 0, 0] * R[:, 0, 1], R[:, 0, 1] ** 2)) / var_sig2_x).T
        C_x = np.linalg.inv(np.matmul(B_x.T, B_x))
        a_x = np.matmul(C_x, np.matmul(B_x.T, b_x))

        sig2_y = np.nanmean(beamsizes[:, :, 1], axis=1) ** 2
        var_sig2_y = 2 * np.nanmean(beamsizes[:, :, 1], axis=1) * np.nanstd(beamsizes[:, :, 1], axis=1)
        b_y = sig2_y / var_sig2_y
        B_y = (np.vstack((R[:, 2, 2] ** 2, 2 * R[:, 2, 2] * R[:, 2, 3], R[:, 2, 3] ** 2)) / var_sig2_y).T
        C_y = np.linalg.inv(np.matmul(B_y.T, B_y))
        a_y = np.matmul(C_y, np.matmul(B_y.T, b_y))

        gamma_rel = beam_energy / 0.51099895e-3

        try:
            eg_x = np.sqrt(a_x[0] * a_x[2] - a_x[1] ** 2)
        except:
            print('no success in X!!!')
            raise ValueError
        else:
            en_x = gamma_rel * eg_x
            beta_x = a_x[0] / eg_x
            alpha_x = -a_x[1] / eg_x
            gamma_x = a_x[2] / eg_x

            # from Stephan (who took it from Velizar). Reference: J. Mnich, ""
            NablaF = np.zeros((4, 3))
            NablaF[0, :] = np.array([a_x[2] / (2 * eg_x),
                                     -a_x[1] / eg_x,
                                     a_x[0] / (2 * eg_x)])
            NablaF[1, :] = np.array([1 / eg_x - a_x[0] * NablaF[0, 0] / eg_x ** 2,
                                     -a_x[0] * NablaF[0, 1] / eg_x ** 2,
                                     -a_x[0] * NablaF[0, 2] / eg_x ** 2])
            NablaF[2, :] = np.array([a_x[1] * NablaF[0, 0] / eg_x ** 2,
                                     -1 / eg_x + a_x[1] * NablaF[0, 1] / eg_x ** 2,
                                     a_x[1] * NablaF[0, 2] / eg_x ** 2])
            NablaF[3, :] = np.array([-a_x[2] * NablaF[0, 0] / eg_x ** 2,
                                     -a_x[2] * NablaF[0, 1] / eg_x ** 2,
                                     1 / eg_x - a_x[2] * NablaF[0, 2] / eg_x ** 2])
            sigmaF_x = np.sqrt(np.diag(np.matmul(NablaF, np.matmul(C_x, NablaF.T))))

        try:
            eg_y = np.sqrt(a_y[0] * a_y[2] - a_y[1] ** 2)
        except:
            print('no success in Y!!!')
            raise ValueError
        else:
            en_y = gamma_rel * eg_y
            beta_y = a_y[0] / eg_y
            alpha_y = -a_y[1] / eg_y
            gamma_y = a_y[2] / eg_y

            NablaF = np.zeros((4, 3))
            NablaF[0, :] = np.array([a_y[2] / (2 * eg_y),
                                     -a_y[1] / eg_y,
                                     a_y[0] / (2 * eg_y)])
            NablaF[1, :] = np.array([1 / eg_y - a_y[0] * NablaF[0, 0] / eg_y ** 2,
                                     -a_y[0] * NablaF[0, 1] / eg_y ** 2,
                                     -a_y[0] * NablaF[0, 2] / eg_y ** 2])
            NablaF[2, :] = np.array([a_y[1] * NablaF[0, 0] / eg_y ** 2,
                                     -1 / eg_y + a_y[1] * NablaF[0, 1] / eg_y ** 2,
                                     a_y[1] * NablaF[0, 2] / eg_y ** 2])
            NablaF[3, :] = np.array([-a_y[2] * NablaF[0, 0] / eg_y ** 2,
                                     -a_y[2] * NablaF[0, 1] / eg_y ** 2,
                                     1 / eg_y - a_y[2] * NablaF[0, 2] / eg_y ** 2])
            sigmaF_y = np.sqrt(np.diag(np.matmul(NablaF, np.matmul(C_y, NablaF.T))))

        return np.array([[en_x, gamma_rel * sigmaF_x[0]], [en_y, gamma_rel * sigmaF_y[0]]]), \
               np.array([[beta_x, sigmaF_x[1]], [alpha_x, sigmaF_x[2]], [gamma_x, sigmaF_x[3]]]), \
               np.array([[beta_y, sigmaF_y[1]], [alpha_y, sigmaF_y[2]], [gamma_y, sigmaF_y[3]]])

    elif len(R.shape) == 3 and R.shape[1:] == (2, 2):
        beamsizes = beamsizes * 1e-3
        sig2_x = np.nanmean(beamsizes, axis=1) ** 2
        var_sig2_x = 2 * np.nanmean(beamsizes, axis=1) * np.nanstd(beamsizes, axis=1)
        b_x = sig2_x / var_sig2_x
        B_x = (np.vstack((R[:, 0, 0] ** 2, 2 * R[:, 0, 0] * R[:, 0, 1], R[:, 0, 1] ** 2)) / var_sig2_x).T
        C_x = np.linalg.inv(np.matmul(B_x.T, B_x))
        a_x = np.matmul(C_x, np.matmul(B_x.T, b_x))

        try:
            eg_x = np.sqrt(a_x[0] * a_x[2] - a_x[1] ** 2)
        except:
            print('no success in X!!!')
            raise ValueError
        else:
            gamma_rel = beam_energy / 0.51099895e-3
            en_x = gamma_rel * eg_x
            beta_x = a_x[0] / eg_x
            alpha_x = -a_x[1] / eg_x
            gamma_x = a_x[2] / eg_x
            NablaF = np.zeros((4, 3))
            NablaF[0, :] = np.array([a_x[2] / (2 * eg_x),
                                     -a_x[1] / eg_x,
                                     a_x[0] / (2 * eg_x)])
            NablaF[1, :] = np.array([1 / eg_x - a_x[0] * NablaF[0, 0] / eg_x ** 2,
                                     -a_x[0] * NablaF[0, 1] / eg_x ** 2,
                                     -a_x[0] * NablaF[0, 2] / eg_x ** 2])
            NablaF[2, :] = np.array([a_x[1] * NablaF[0, 0] / eg_x ** 2,
                                     -1 / eg_x + a_x[1] * NablaF[0, 1] / eg_x ** 2,
                                     a_x[1] * NablaF[0, 2] / eg_x ** 2])
            NablaF[3, :] = np.array([-a_x[2] * NablaF[0, 0] / eg_x ** 2,
                                     -a_x[2] * NablaF[0, 1] / eg_x ** 2,
                                     1 / eg_x - a_x[2] * NablaF[0, 2] / eg_x ** 2])
            sigmaF_x = np.sqrt(np.diag(np.matmul(NablaF, np.matmul(C_x, NablaF.T))))

        return np.array([en_x, gamma_rel * sigmaF_x[0]]),\
               np.array([[beta_x, sigmaF_x[1]], [alpha_x, sigmaF_x[2]], [gamma_x, sigmaF_x[3]]])


def calc_centroids(R: np.ndarray, centroids: np.ndarray):
    """

    :param R:
    :param centroids: in mm!!!
    :return:
    """
    centroids *= 1e-3
    if type(R) == list:
        R = np.array(R)
    if len(R.shape) == 3 and R.shape[1:] == (2, 2):
        cm = np.nanmean(centroids, axis=1)
        var_cm = np.nanstd(centroids, axis=1)
        b_x = cm / var_cm
        B_x = (np.vstack((R[:, 0, 0], R[:, 0, 1])) / var_cm).T
        C_x = np.linalg.inv(np.matmul(B_x.T, B_x))
        a_x = np.matmul(C_x, np.matmul(B_x.T, b_x))

        x0 = a_x[0]
        x0_prime = a_x[1]
        sigmaF_x = np.sqrt(np.diag(C_x))

        return np.vstack((np.array([x0, x0_prime]), sigmaF_x)).T


def mismatch(tws_des: np.ndarray, tws_meas: np.ndarray):
    if type(tws_des) == np.ndarray:
        if tws_des.size == 3:
            return 0.5 * (tws_meas[0] * tws_des[2] - 2 * tws_meas[1] * tws_des[1] + tws_meas[2] * tws_des[0])
        elif tws_des.size == 6:
            mx = 0.5 * (tws_meas[0] * tws_des[2] - 2 * tws_meas[1] * tws_des[1] + tws_meas[2] * tws_des[0])
            my = 0.5 * (tws_meas[3] * tws_des[5] - 2 * tws_meas[4] * tws_des[4] + tws_meas[5] * tws_des[3])
            return mx, my
    elif type(tws_des) == Twiss:
        tws_des.gamma_x = (1 + tws_des.alpha_x ** 2) / tws_des.beta_x
        tws_des.gamma_y = (1 + tws_des.alpha_y ** 2) / tws_des.beta_y
        tws_meas.gamma_x = (1 + tws_meas.alpha_x ** 2) / tws_meas.beta_x
        tws_meas.gamma_y = (1 + tws_meas.alpha_y ** 2) / tws_meas.beta_y
        mx = 0.5 * (tws_meas.beta_x * tws_des.gamma_x - 2 * tws_meas.alpha_x * tws_des.alpha_x
                    + tws_meas.gamma_x * tws_des.beta_x)
        my = 0.5 * (tws_meas.beta_y * tws_des.gamma_y - 2 * tws_meas.alpha_y * tws_des.alpha_y
                    + tws_meas.gamma_y * tws_des.beta_y)
        return mx, my


def calc_ellipse_ps(emittance: float, beta: float, alpha: float, return_norm_matrix: bool = False):
    """

    :param emittance: normalised emittance [m] (e.g. at FLASH ~2e-6 m)
    :param beta: beta from Courant-Snyder params [m]
    :param alpha: alpha from Courant-Snyder params
    :param return_coords: coordinates (x, xp) of the curve that defines the ellipse
    :return:
    """
    gamma = (1 + alpha ** 2) / beta
    sqrtterm = np.sqrt((gamma - beta) ** 2 + (2 * alpha) ** 2)
    a = np.sqrt(2 * emittance * (gamma + beta + sqrtterm))
    b = np.sqrt(2 * emittance * (gamma + beta - sqrtterm))
    if alpha == 0 and gamma < beta:
        theta = 0
    elif alpha == 0 and gamma > beta:
        theta = np.pi / 2
    else:
        theta = -1 * np.arctan(0.5 / alpha * (beta - gamma - sqrtterm))
    phi = np.linspace(0, 2 * np.pi, 1000)
    x = a * np.cos(phi) * np.cos(theta) + b * np.sin(phi) * np.sin(theta)
    xp = -a * np.cos(phi) * np.sin(theta) + b * np.sin(phi) * np.cos(theta)
    coords = np.vstack((x, xp))
    if return_norm_matrix:
        # Normalisation
        phi_max = np.arctan(- a / (b * np.tan(theta)))
        xp_max = -a * np.cos(phi_max) * np.sin(theta) + b * np.sin(phi_max) * np.cos(theta)
        R_norm = 1 / abs(beta * xp_max) * np.array([[1, 0], [alpha, beta]])
        return a, b, theta, coords, R_norm
    else:
        return a, b, theta, coords


def get_object_plane(R: np.ndarray, axis: int = 0):
    def get_R12(l: float, R: np.ndarray, axis: int = axis):
        R_tot = np.matmul(R, driftMatrix(l=l))
        return R_tot[int(axis * 2), int(axis * 2) + 1] ** 2
    result = minimize_scalar(get_R12, args=(R,))
    R11 = np.matmul(R, driftMatrix(result.x))[int(axis * 2), int(axis * 2)]
    return result.x, R11


def point2point(lat, axis: int, magnification: float, energy: float,
                bounds: list = None, bounds_mode: str = 'absolute', p0: np.ndarray = None):
    return True