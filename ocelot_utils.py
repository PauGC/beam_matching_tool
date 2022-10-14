import importlib
import logging
import numpy as np
import pandas as pd
import os

from copy import deepcopy, copy
from ocelot import *
from ocelot.cpbd.magnetic_lattice import *
# logging.basicConfig(level=logging.DEBUG)
from io import StringIO
from ocelot_utils import *
logger = logging.getLogger(__name__)


def twiss_at(elem_id, lat, tws, edge='start'):
    """
    Returns the twiss paramters at a specific element. The position relative to the element at which the parameters
    shouldd be determined can be"

    :param lat: MagneticLattice element
    :param elem_id: id of the element
    :param tws: twiss list (result of twiss(lat, tw0)
    :param edge: position in the element where to give the twiss params ('start', 'middle', 'end')
    :return: Twiss() object
    """
    try:
        idx = [el.id for el in lat.sequence].index(elem_id)
    except Exception as err:
        raise ValueError(str(err))

    s_pos = tws[0].s
    for el in lat.sequence[:idx]:
        s_pos += el.l

    if edge == 'middle':
        s_pos += lat.sequence[idx].l / 2
    elif edge == 'end':
        s_pos += lat.sequence[idx]

    idxs_up = np.where(np.array([t.s for t in tws]) >= s_pos)[0]
    if idxs_up.shape[0] == 0:
        raise ValueError("element outside the range of the given twiss list!")
    else:
        idx_up = idxs_up[0]
    idxs_down = np.where(np.array([t.s for t in tws]) < s_pos)[0]
    if idxs_down.shape[0] == 0:
        return deepcopy(tws[idx_up])
    else:
        newTwiss = Twiss()
        idx_down = idxs_down[-1]
        pars = ['beta_x', 'alpha_x', 'gamma_x', 'emit_x', 'emit_xn', 'mux', 'Dx', 'Dxp',
                'beta_y', 'alpha_y', 'gamma_y', 'emit_y', 'emit_yn', 'muy', 'Dy', 'Dyp']
        s_l = tws[idx_up].s
        s_u = tws[idx_down].s
        for par in pars:
            val_l = getattr(tws[idx_up], par)
            val_u = getattr(tws[idx_down], par)
            m = (val_u - val_l) / (s_u - s_l)
            value = val_l + m * (s_pos - s_l)
            setattr(newTwiss, par, value)
        return deepcopy(newTwiss)


def twiss_diff(tw0, tw1):
    pars = ['Dx', 'Dxp', 'Dy', 'Dyp', 'E', 'alpha_x', 'alpha_y', 'beta_x', 'beta_y', 'emit_x', 'emit_xn', 'emit_y',
            'emit_yn', 'gamma_x', 'gamma_y', 'mux', 'muy', 'p', 's', 'tau', 'x', 'xp', 'y', 'yp']
    newTwiss = Twiss()
    for par in pars:
        diff = getattr(tw1, par) - getattr(tw0, par)
        setattr(newTwiss, par, diff)
    return newTwiss


def find_slice_emittance_scan(tw0, lat_design, steps, quad_names,
                              beta_scr=[20.0, 30.0], direction='+', global_constraints=None):
    """
    Finds the proper k1 values for the quads between the reference and the marker so that starting with the twiss
    parameters given by tw0 a phase advance in the plane given covers 180 degree in N steps.

    :param tw0: Twiss() object (no matter the s position!)
    :param lat: MagneticLattice object. The twiss resulting from propagating tw0 through the lattice will be used
    to determine the starting phase advance offset between the reference plane and the measuring screen.
    :param m0_id: marker id for the reference plane (must be in the lattice!!!)
    :param m1_id: marker id for the measurement plane (must also be in the lattice!!!)
    :param quads: Quads to be used. If a quad in the given list is not found in the lattice, it will be ignored!!!
    :param steps: Steps between the 180 degrees
    :param beta_scr: Target beta at the measurement screen
    :return: diccionary with quad name as key and list (of length #steps) of quad stength values as value
    """
    # determine the
    idx_new_reff = [el.id for el in lat_design.sequence].index(quad_names[0])
    m_reff = Marker("NEWREFERENCEPLANE")
    lat_design.sequence.insert(idx_new_reff, m_reff)
    lat_n = MagneticLattice(lat_design.sequence)
    tws_design = twiss(lat_n, tw0)

    tw0 = twiss_at(elem_id='NEWREFERENCEPLANE', lat=lat_n, tws=tws_design)
    tw0_scr = twiss_at(elem_id='SCR11FLFXTDS', lat=lat_n, tws=tws_design)
    mux0 = deepcopy(tw0_scr.mux) - deepcopy(tw0.mux)
    muy0 = deepcopy(tw0_scr.muy) - deepcopy(tw0.muy)
    tw0.mux = 0.0
    tw0.muy = 0.0
    mux_list = []
    muy_list = []
    tws_list = []

    names = [el.id for el in lat_design.sequence]
    idx0 = names.index('NEWREFERENCEPLANE')
    idx1 = names.index('SCR11FLFXTDS')
    new_seq = lat_design.sequence[idx0:idx1 + 1]

    names = [el.id for el in new_seq]
    idx_tds = names.index('TDS-C')
    m_tds = new_seq[idx_tds]
    m_scr1 = new_seq[-1]
    m_scr2 = Marker("SCR11FLFXTDS-2")
    new_seq.append(m_scr2)

    quads = [el for el in new_seq if el.id in quad_names]
    quads.pop(1)
    quads.pop(2)
    results = {}
    for quad in quads:
        results.update({quad.id: {'k1': [quad.k1]}})

    constraints = {}
    if global_constraints is None:
        global_constraints = dict({'beta_x': ["<", 200.0], 'beta_y': ["<", 200.0]})
        constraints.update({'global': global_constraints})
    constraints.update({m_tds: {'beta_y': 100.0, 'alpha_y': 0.0}})

    lat_n = MagneticLattice(new_seq)

    for i in range(steps):
        if direction == '+':
            constraints.update({m_scr1: {'beta_y': 1.0, 'beta_x': ["<", beta_scr[1]],
                                         'mux': mux0 + i * np.pi / (steps - 1), 'muy': muy0}})
        elif direction == '-':
            constraints.update({m_scr1: {'beta_y': 1.0, 'beta_x': ["<", beta_scr[1]],
                                         'mux': mux0 - i * np.pi / (steps - 1), 'muy': muy0}})
        constraints.update({m_scr2: {'beta_x': [">", beta_scr[0]]}})
        result = match(lat=lat_n, constr=constraints, vars=quads, tw=tw0, max_iter=100000)
        for q, k1 in zip(quads, result):
            results[q.id]['k1'].append(k1)
            q.k1 = k1
        lat_n.update_transfer_maps()
        tws = twiss(lat_n, tw0)
        mux_list.append(tws[-1].mux)
        muy_list.append(tws[-1].muy)
        tws_list.append(tws)
    return results, mux_list, muy_list, tws_list


def find_XTDS_opt(lat, tws_tds, screen, beta_scr, axis='y'):
    names = [el.id for el in lat.sequence]
    idx_tds = names.index('TDS-C')
    if screen == '11FLFXTDS':
        idx_scr = names.index('SCR11FLFXTDS')
        new_seq = lat.sequence[idx_tds:idx_scr + 1]
        lat_n = MagneticLattice(new_seq)
        m_scr = new_seq[-1]
        if axis == 'y':
            muy = tws_tds.muy + np.pi / 2
            quads = [el for el in new_seq if el.__class__ == Quadrupole]
            contraints = {m_scr: {'beta_y': beta_scr, 'muy': muy}}
            result = match(lat=lat_n, constr=contraints, vars=quads, tw=tws_tds, max_iter=100000)
            for q, k1 in zip(quads, result):
                q.k1 = k1
            lat_n.update_transfer_maps()
            tws = twiss(lat_n, tws0=tws_tds)
            muy_diff = tws[-1].muy - tws[0].muy
            beta_scr = tws[-1].beta_y
            print('phase advance:', muy_diff)
            print('beta at screen:', beta_scr)
            values = {}
            for q in quads:
                values.update({q.id: q.k1})
            return values
        elif axis == 'x':
            mux = tws_tds.mux + np.pi / 2
            quads = [el for el in new_seq if el.__class__ == Quadrupole]
            contraints = {m_scr: {'beta_x': beta_scr, 'mux': mux}}
            result = match(lat=lat_n, constr=contraints, vars=quads, tw=tws_tds, max_iter=100000)
            for q, k1 in zip(quads, result):
                q.k1 = k1
            lat_n.update_transfer_maps()
            tws = twiss(lat_n, tws0=tws_tds)
            mux_diff = tws[-1].mux - tws[0].mux
            beta_scr = tws[-1].beta_x
            print('phase advance:', mux_diff)
            print('beta at screen:', beta_scr)
            values = {}
            for q in quads:
                values.update({q.id: q.k1})
            return values