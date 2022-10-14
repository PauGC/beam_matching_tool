#!/usr/bin/env python3

from copy import copy, deepcopy
import matplotlib.pylab as plt
plt.ion()
import numpy as np
from scipy.constants import physical_constants
m0_MeV = physical_constants['electron mass energy equivalent in MeV'][0]
import ocelot
from ocelot import *
from flash_lattice.lattice.lattice_manager import FLASHlattice


def calc_multi_8FLFMAFF(tw0: ocelot.cpbd.beam.Twiss, charge: float, energy: float, emittance_n: float,
                        scan_steps: int = 11, optics_seed: dict = None, randomize: bool = True, direction: int = +1,
                        verbose: bool = False):
    """

    :param tw0: Twiss ocelot object
    :param charge: in nC
    :param energy: in GeV
    :param emittance_n: in um
    :param scan_steps: int
    :param optics_seed: dictionary with structure {quad_name_0: {'k1': value}, quad_name_1: {'k1': value}, etc...}
    :param randomize: True or False
    :param direction: increasing phase advance or decreasing it (+1 or -1)
    :param verbose: ocelot match function verbosity
    :return: optics (json file), new_beta (np.ndarray), new_phase_advance (np.ndarray)
    """

    # input params >>> beta_target to avoid scintillator screen saturation
    gamma_rel = energy * 1e3 / m0_MeV
    beta_target = min(gamma_rel * charge / (2 * np.pi * emittance_n), 100)
    step_val = np.pi / (scan_steps - 1)  # rad

    # lattice with new markers
    lat_maff = FLASHlattice().return_lat_section('FL3_xtds', start='STARTFLFMAFF', stop='SCREEN8FLFMAFF')
    new_seq = deepcopy(lat_maff.lat.sequence)
    marker_0 = Marker(eid='M0')
    marker_1 = Marker(eid='M1')
    marker_2 = Marker(eid='M2')
    marker_3 = Marker(eid='M3')
    new_seq.append(marker_0)
    new_seq.append(marker_1)
    new_seq.append(marker_2)
    new_seq.append(marker_3)
    lat_n = MagneticLattice(new_seq)

    # variables
    quads = [el for el in new_seq if type(el) == Quadrupole]
    if optics_seed:
        for quad in quads:
            quad.k1 = optics_seed[quad.id]['k1']
        lat_n.update_transfer_maps()
    elif randomize:
        for quad, k1 in zip(quads, (np.random.rand(len(quads)) - 0.5) * 5.):
            quad.k1 = k1
            lat_n.update_transfer_maps()

    # weights function
    def match_weights(val):
        if val in ['beta_x', 'beta_y', 'mux', 'muy']: return 10.0
        return 1.0

    # constraints
    constraints = {'global':
                       {'beta_x': ["<", 150],
                        'beta_y': ["<", 150]},
                   marker_0:
                       {'beta_x': ["<", beta_target + beta_target * 0.1],
                        'beta_y': ["<", beta_target + beta_target * 0.1]},
                   marker_1:
                       {'beta_x': [">", beta_target - beta_target * 0.1],
                        'beta_y': [">", beta_target - beta_target * 0.1]}}

    # actual calculation
    results = []
    betass = []
    muss = []
    Rs = []
    for i in range(scan_steps):
        result = match(lat=lat_n, constr=constraints, vars=quads, tw=tw0, weights=match_weights, max_iter=1000000,
                       verbose=verbose)
        for quad, k1 in zip(quads, result):
            quad.k1 = k1
        lat_n.update_transfer_maps()
        tws = twiss(lat_n, tw0)
        Rs.append(lattice_transfer_map(lat_n, energy=energy)[:4, :4])
        if i == 0:
            mux = tws[-1].mux + direction * np.linspace(0, np.pi, scan_steps)
            muy = tws[-1].muy + direction * np.linspace(0, np.pi, scan_steps)
        results.append(result)
        betass.append(np.array([tws[-1].beta_x, tws[-1].beta_y]))
        muss.append(np.array([tws[-1].mux, tws[-1].muy]))
        if i < scan_steps - 1:
            constraints.update({marker_2:
                                    {'mux': ["<", mux[i + 1] + 0.1 * step_val],
                                     'muy': ["<", muy[i + 1] + 0.1 * step_val]},
                                marker_3:
                                    {'mux': [">", mux[i + 1] - 0.1 * step_val],
                                     'muy': [">", muy[i + 1] - 0.1 * step_val]}})
    new_optics = {quad.id: {'k1': list(vals)} for quad, vals in zip(quads, np.array(results).T)}
    new_beta = np.array(betass)
    new_phase_advance = np.array(muss)
    return new_optics, new_beta, new_phase_advance, Rs


def calc_multi_FL2SEED(tw0: ocelot.cpbd.beam.Twiss, charge: float, energy: float, emittance_n: float,
                       scan_steps: int = 11, optics_seed: dict = None, randomize: bool = True, direction: int = +1,
                       verbose: bool = False):
    """

    :param tw0: Twiss ocelot object
    :param charge: in nC
    :param energy: in GeV
    :param emittance_n: in um
    :param scan_steps: int
    :param optics_seed: dictionary with structure {quad_name_0: {'k1': value}, quad_name_1: {'k1': value}, etc...}
    :param randomize: True or False
    :param direction: increasing phase advance or decreasing it (+1 or -1)
    :param verbose: ocelot match function verbosity
    :return: optics (json file), new_beta (np.ndarray), new_phase_advance (np.ndarray)
    """

    # input params >>> beta_target to avoid scintillator screen saturation
    gamma_rel = energy * 1e3 / m0_MeV
    beta_target = min(gamma_rel * charge / (2 * np.pi * emittance_n), 100)
    step_val = np.pi / (scan_steps - 1)  # rad

    # lattice with new markers
    lat_maff = FLASHlattice().return_lat_section('FL2', start='MQ17FL2EXTR.U', stop='OTR1FL2SEED7')
    new_seq = deepcopy(lat_maff.lat.sequence)
    marker_0 = Marker(eid='M0')
    marker_1 = Marker(eid='M1')
    marker_2 = Marker(eid='M2')
    marker_3 = Marker(eid='M3')
    new_seq.append(marker_0)
    new_seq.append(marker_1)
    new_seq.append(marker_2)
    new_seq.append(marker_3)
    lat_n = MagneticLattice(new_seq)

    # variables
    quads = [el for el in new_seq if type(el) == Quadrupole]
    quads.pop(0)
    if optics_seed:
        for quad in quads:
            quad.k1 = optics_seed[quad.id]['k1']
        lat_n.update_transfer_maps()
    elif randomize:
        for quad, k1 in zip(quads, (np.random.rand(len(quads)) - 0.5) * 5.):
            quad.k1 = k1
            lat_n.update_transfer_maps()

    # weights function
    def match_weights(val):
        if val in ['beta_x', 'beta_y', 'mux', 'muy']: return 10.0
        return 1.0

    # constraints
    constraints = {'global':
                       {'beta_x': ["<", 150],
                        'beta_y': ["<", 150]},
                   marker_0:
                       {'beta_x': ["<", beta_target + beta_target * 0.1],
                        'beta_y': ["<", beta_target + beta_target * 0.1]},
                   marker_1:
                       {'beta_x': [">", beta_target - beta_target * 0.1],
                        'beta_y': [">", beta_target - beta_target * 0.1]}}

    # actual calculation
    results = []
    betass = []
    muss = []
    Rs = []
    for i in range(scan_steps):
        result = match(lat=lat_n, constr=constraints, vars=quads, tw=tw0, weights=match_weights, max_iter=1000000,
                       verbose=verbose)
        for quad, k1 in zip(quads, result):
            quad.k1 = k1
        lat_n.update_transfer_maps()
        tws = twiss(lat_n, tw0)
        Rs.append(lattice_transfer_map(lat_n, energy=energy)[:4, :4])
        if i == 0:
            mux = tws[-1].mux + direction * np.linspace(0, np.pi, scan_steps)
            muy = tws[-1].muy + direction * np.linspace(0, np.pi, scan_steps)
        results.append(result)
        betass.append(np.array([tws[-1].beta_x, tws[-1].beta_y]))
        muss.append(np.array([tws[-1].mux, tws[-1].muy]))
        if i < scan_steps - 1:
            constraints.update({marker_2:
                                    {'mux': ["<", mux[i + 1] + 0.1 * step_val],
                                     'muy': ["<", muy[i + 1] + 0.1 * step_val]},
                                marker_3:
                                    {'mux': [">", mux[i + 1] - 0.1 * step_val],
                                     'muy': [">", muy[i + 1] - 0.1 * step_val]}})
    new_optics = {quad.id: {'k1': list(vals)} for quad, vals in zip(quads, np.array(results).T)}
    new_beta = np.array(betass)
    new_phase_advance = np.array(muss)
    return new_optics, new_beta, new_phase_advance, Rs
