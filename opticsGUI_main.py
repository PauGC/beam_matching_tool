#!/usr/bin/env python3

from copy import deepcopy, copy
import json
import logging
import numpy as np
from ocelot import *
import os
import pandas as pd
import pickle
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pyqtgraph as pg

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
import re
from scipy.constants import c as c_light
import sys

from ocelot_utils import twiss_at
from flash_lattice.lattice.lattice_manager import FLASHlattice, read_opticsfile, save_opticsfile
from opticsGUI_windows import *

try:
    import pydoocs
except:
    pass


"""
def trap_exc_during_debug(*args):
   # when app raises uncaught exception, print info
   print(args)

# install exception hook: without this, uncaught exception would cause application to exit
sys.excepthook = trap_exc_during_debug
"""


def flatten_items_dict(dictionary):
    items = []
    for k, v in dictionary.items():
        if type(v) == list:
            items += v
        else:
            items.append(v)
    return items


class MatchWorker(QObject):
    signalDone = pyqtSignal()

    def __init__(self, parent, lat, constr, vars, tw0, weights):
        super().__init__()
        # self.preset = preset
        self.parent = parent
        self.lat = lat
        self.constr = constr
        self.vars = vars
        self.tw0 = tw0
        self.weights = weights
        self.result = None
        self._abort = False

    def run(self):
        self.result = match(lat=self.lat, constr=self.constr, vars=self.vars, tw=self.tw0, weights=self.weights,
                            max_iter=100000)
        if self.parent.ui.match_preset.currentText() in ['11FLFXTDS', '8FLFDUMP']:
            for quad, value in zip(self.vars, self.result):
                quad.k1 = value
            self.lat.update_transfer_maps()
            tws = twiss(self.lat, self.tw0, nPoints=1000)
            tw_tds = twiss_at('TDS-C', self.lat, tws)
            for key in self.constr.keys():
                if type(key) == Marker:
                    if key.id == 'SCR11FLFXTDS':
                        if self.parent.ui.streaking_plane.currentText() == 'X':
                            self.constr.update({key: {'mux': tw_tds.mux + np.pi / 2}})
                        elif self.parent.ui.streaking_plane.currentText() == 'Y':
                            self.constr.update({key: {'muy': tw_tds.muy + np.pi / 2}})
                    elif key.id == 'SCR8FLFDUMP':
                        self.constr.update({key: {'muy': tw_tds.muy + np.pi / 2}})
            self.result = match(lat=self.lat, constr=self.constr, vars=self.vars, tw=self.tw0, weights=self.weights,
                                max_iter=100000)
        self.signalDone.emit()
        return

    def abort(self):
        self._abort = True


class mainGui(QMainWindow):
    my_signal = pyqtSignal(object)

    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.ui = ui_MainWindow()
        self.ui.setupUi(self)

        self.lattice_manager = FLASHlattice()
        self.ui.lat_section.addItems([sec.name for sec in self.lattice_manager.sections])
        self.twiss_nPoints = 2000
        self.lat_section_design = None
        self.lat_section_ttf = None
        self.tws_design = None
        self.tws_real = None
        self.tw0_real = None
        self.optics = None
        self.magnet_widget_list = []
        self.magnet_push_list = []
        self.lattice_ui_items = {}
        self.top_plot_ui_items = {'design': [], 'real': []}
        self.bottom_plot_ui_items = {'design': [], 'real': []}
        self.worker = None
        self.thread = None

        # Main settings events
        self.ui.lat_section.currentIndexChanged.connect(self._load_section_items)
        self.ui.section_start.currentIndexChanged.connect(self._load_sliced_section)
        self.ui.section_stop.currentIndexChanged.connect(self._load_sliced_section)

        # Input twiss events
        self.ui.betax.valueChanged.connect(self.set_new_realTw0)
        self.ui.alphax.valueChanged.connect(self.set_new_realTw0)
        self.ui.Dx.valueChanged.connect(self.set_new_realTw0)
        self.ui.Dxp.valueChanged.connect(self.set_new_realTw0)
        self.ui.betay.valueChanged.connect(self.set_new_realTw0)
        self.ui.alphay.valueChanged.connect(self.set_new_realTw0)
        self.ui.Dy.valueChanged.connect(self.set_new_realTw0)
        self.ui.Dyp.valueChanged.connect(self.set_new_realTw0)

        # XTDS time resolution
        # self.ui.xtds_calc_resolution.clicked.connect(self.calc_timeRes)
        self.ui.bunch_energy.valueChanged.connect(self.calc_timeRes)
        self.ui.bunch_emmitance.valueChanged.connect(self.calc_timeRes)
        self.ui.xtds_voltage.valueChanged.connect(self.calc_timeRes)
        self.ui.xtds_calc_resolution_screen.currentIndexChanged.connect(self.calc_timeRes)

        # Optics frame events
        self.ui.top_function.currentTextChanged.connect(self.update_top)
        self.ui.axtop_xactive.stateChanged.connect(self.update_top)
        self.ui.axtop_yactive.stateChanged.connect(self.update_top)
        self.ui.axtop_log.stateChanged.connect(self.set_log_scale)
        self.ui.axtop_autoscale.stateChanged.connect(self.set_autoscale_top)
        self.ui.bottom_function.currentTextChanged.connect(self.update_bottom)
        self.ui.axbottom_xactive.stateChanged.connect(self.update_bottom)
        self.ui.axbottom_yactive.stateChanged.connect(self.update_bottom)
        self.ui.axbottom_log.stateChanged.connect(self.set_log_scale)
        self.ui.axbottom_autoscale.stateChanged.connect(self.set_autoscale_bottom)
        self.ui.match_preset.currentTextChanged.connect(self.select_match_preset)
        self.ui.do_match.clicked.connect(self.do_match)

        # Magnet controls events
        self.ui.magnet_stepsize.currentTextChanged.connect(self._set_magnet_stepsize)
        self.ui.magnets_push_all.stateChanged.connect(self._set_push_all_magnets)
        self.ui.magnets_pull_from_linac.clicked.connect(self.magnets_pull_from_linac)
        self.ui.magnets_push_to_linac.clicked.connect(self.magnets_push_to_linac)
        self.ui.load_reference_optFile.clicked.connect(self._open_optFile)
        self.ui.optics_file_setting.valueChanged.connect(self._load_optics_setting)
        self.ui.save_current_optFile.clicked.connect(self._save_optFile)
        self.ui.magnets_real_to_theory.clicked.connect(self.magnets_reset_to_theory)

    def _block_signals(self, block):
        self.ui.section_start.blockSignals(block)
        self.ui.section_stop.blockSignals(block)
        self.ui.betax.blockSignals(block)
        self.ui.alphax.blockSignals(block)
        self.ui.betay.blockSignals(block)
        self.ui.alphay.blockSignals(block)
        self.ui.Dx.blockSignals(block)
        self.ui.Dy.blockSignals(block)

    def _pull_tw0_des_ui(self):
        if not self.lat_section_design.tw0_des is None:
            for par in ['betax_des', 'alphax_des']:
                label = getattr(self.ui, par)
                label.setText("{:.3f}".format(getattr(self.lat_section_design.tw0_des, par.replace('x_des', '_x'))))
                combo = getattr(self.ui, par.replace('_des', ''))
                combo.setValue(getattr(self.lat_section_design.tw0_des, par.replace('x_des', '_x')))
            for par in ['betay_des', 'alphay_des']:
                label = getattr(self.ui, par)
                label.setText("{:.3f}".format(getattr(self.lat_section_design.tw0_des, par.replace('y_des', '_y'))))
                combo = getattr(self.ui, par.replace('_des', ''))
                combo.setValue(getattr(self.lat_section_design.tw0_des, par.replace('y_des', '_y')))
            for par in ['Dx_des', 'Dxp_des', 'Dy_des', 'Dyp_des']:
                label = getattr(self.ui, par)
                label.setText("{:.2f}".format(getattr(self.lat_section_design.tw0_des, par.replace('_des', '')) * 1e3))
                combo = getattr(self.ui, par.replace('_des', ''))
                combo.setValue(getattr(self.lat_section_design.tw0_des, par.replace('_des', '')) * 1e3)

    def _load_section_items(self, section):
        self._block_signals(True)
        sec_name = [sec.name for sec in self.lattice_manager.sections][section - 1]
        self.lat_section_design = self.lattice_manager.return_lat_section(sec_name)
        self.lat_section_ttf = self.lattice_manager.return_lat_section(sec_name)
        #with open('optics/FLASHForward_R56opt_XQA_shifts.json', 'r') as jf:
        #    optics = json.load(jf)
        #    self.lat_section_design.optics = optics
        #    self.lat_section_ttf.optics = optics
        self.tws_design = twiss(lattice=self.lat_section_design.lat, tws0=self.lat_section_design.tw0_des,
                                nPoints=self.twiss_nPoints)
        self.tws_real = twiss(lattice=self.lat_section_ttf.lat, tws0=self.lat_section_ttf.tw0_des,
                              nPoints=self.twiss_nPoints)
        self.tw0_real = deepcopy(self.lat_section_ttf.tw0_des)
        self.ui.section_start.addItems([el.id for el in self.lat_section_design.seq if type(el) != Aperture])
        self.ui.section_start.setCurrentIndex(0)
        self.ui.section_stop.addItems([el.id for el in self.lat_section_design.seq if type(el) != Aperture])
        self.ui.section_stop.setCurrentIndex(self.ui.section_stop.count() - 1)
        self.ui.match_ref_plane.addItems([el.id for el in self.lat_section_design.lat.sequence
                                          if el.__class__ == Marker])
        self.ui.match_target_plane.addItems([el.id for el in self.lat_section_design.lat.sequence
                                             if el.__class__ == Marker])
        self.ui.match_target_plane.setCurrentIndex(self.ui.match_target_plane.count() - 1)
        self._pull_tw0_des_ui()
        self._build_magnets_grid()
        self._plot_lattice()
        self._plot_top_design()
        self._plot_bottom_design()
        self._plot_top_real()
        self._plot_bottom_real()
        self.calc_timeRes()
        self._block_signals(False)

    def _load_sliced_section(self):
        name_start = self.ui.section_start.currentText()
        name_stop = self.ui.section_stop.currentText()
        self._block_signals(True)
        self.lat_section_design = self.lattice_manager.return_lat_section(self.ui.lat_section.currentText(),
                                                                          start=name_start, stop=name_stop)
        self.lat_section_ttf = self.lattice_manager.return_lat_section(self.ui.lat_section.currentText(),
                                                                       start=name_start, stop=name_stop)
        self.tws_design = twiss(lattice=self.lat_section_design.lat, tws0=self.lat_section_design.tw0_des,
                                nPoints=self.twiss_nPoints)
        self.tws_real = twiss(lattice=self.lat_section_ttf.lat, tws0=self.lat_section_ttf.tw0_des,
                              nPoints=self.twiss_nPoints)
        self.tw0_real = deepcopy(self.lat_section_ttf.tw0_des)
        idx_start = [el.id for el in self.lat_section_ttf.seq if type(el)].index(name_start)
        idx_stop = [el.id for el in self.lat_section_ttf.seq if type(el)].index(name_stop)
        self.ui.section_start.clear()
        self.ui.section_start.addItems([el.id for el in self.lat_section_ttf.seq[:idx_stop + 1] if type(el)])
        self.ui.section_start.setCurrentIndex(idx_start)
        self.ui.section_stop.clear()
        self.ui.section_stop.addItems([el.id for el in self.lat_section_ttf.seq[idx_start:] if type(el)])
        idx_stop_new = [el.id for el in self.lat_section_ttf.seq[idx_start:] if type(el)].index(name_stop)
        self.ui.section_stop.setCurrentIndex(idx_stop_new)
        self.ui.match_ref_plane.clear()
        self.ui.match_ref_plane.addItems([el.id for el in self.lat_section_design.lat.sequence
                                          if el.__class__ == Marker])
        self.ui.match_target_plane.clear()
        self.ui.match_target_plane.addItems([el.id for el in self.lat_section_design.lat.sequence
                                             if el.__class__ == Marker])
        self.ui.match_target_plane.setCurrentIndex(self.ui.match_target_plane.count() - 1)
        self._pull_tw0_des_ui()
        self._build_magnets_grid()
        self._plot_lattice()
        self._plot_top_design()
        self._plot_bottom_design()
        self._plot_top_real()
        self._plot_bottom_real()
        self.calc_timeRes()
        self._block_signals(False)

    def _build_magnets_grid(self):
        if len(self.magnet_widget_list) > 0:
            for i in range(self.ui.magnetsLayout.columnCount() * (self.ui.magnetsLayout.rowCount())):
                if not type(self.ui.magnetsLayout.itemAtPosition(int(i / 5), np.mod(i, 5))) is type(None):
                    widget_to_remove = self.ui.magnetsLayout.itemAtPosition(int(i / 5), np.mod(i, 5)).widget()
                    self.ui.magnetsLayout.removeWidget(widget_to_remove)
                    widget_to_remove.setParent(None)
        self.magnet_widget_list = []
        all_magnets = [elem for elem in self.lat_section_design.lat.sequence if (type(elem) == Quadrupole or
                                                                                 type(elem) == Sextupole)]
        seen = {}
        magnets = [seen.setdefault(x, x) for x in all_magnets if x not in seen]
        for i, magnet in enumerate(magnets):
            magnet_label = QLabel(magnet.id)
            magnet_label.setAlignment(Qt.AlignCenter)
            self.ui.magnetsLayout.addWidget(magnet_label, i, 0)
            magnet_widget = QDoubleSpinBox()
            magnet_widget.setDecimals(4)
            magnet_widget.setSingleStep(0.1000)
            magnet_widget.setRange(-40.0, 40.0)
            magnet_widget.setObjectName(magnet.id)
            if type(magnet) == Quadrupole:
                magnet_value = QLabel("{:.3f}".format(magnet.k1))
                magnet_widget.setValue(magnet.k1)
            elif type(magnet) == Sextupole:
                magnet_value = QLabel("{:.3f}".format(magnet.k2))
                magnet_widget.setValue(magnet.k2)
            # else:
            #    magnet_value = QLabel("{:.3f}".format(np.rad2deg(magnet.angle)))
            #    magnet_widget.setValue(np.rad2deg(magnet.angle))
            magnet_value.setAlignment(Qt.AlignCenter)
            magnet_widget.setAlignment(Qt.AlignCenter)
            self.ui.magnetsLayout.addWidget(magnet_value, i, 1)
            self.ui.magnetsLayout.addWidget(magnet_widget, i, 2)
            magnet_diff = QLabel("0.0")
            magnet_diff.setAlignment(Qt.AlignCenter)
            self.ui.magnetsLayout.addWidget(magnet_diff, i, 3)
            magnet_checkbox = QCheckBox()
            magnet_checkbox.setChecked(False)
            self.ui.magnetsLayout.addWidget(magnet_checkbox, i, 4)
            magnet_widget.valueChanged.connect(self.set_magnet_value)
        self.magnet_widget_list = [self.ui.magnetsLayout.itemAtPosition(i, 2).widget() for i in range(len(magnets))]
        self.magnet_push_list = [self.ui.magnetsLayout.itemAtPosition(i, 4).widget() for i in range(len(magnets))]

    def set_new_realTw0(self):
        new_tw0_vals = [widget.value() for widget in
                        [getattr(self.ui, var) for var in ['betax', 'alphax', 'Dx', 'Dxp',
                                                           'betay', 'alphay', 'Dy', 'Dyp']]]
        new_tw0_vals[2] *= 1e-3
        new_tw0_vals[3] *= 1e-3
        new_tw0_vals[6] *= 1e-3
        new_tw0_vals[7] *= 1e-3
        new_tw0 = Twiss()
        [setattr(new_tw0, var, new_tw0_vals[i]) for i, var in enumerate(['beta_x', 'alpha_x', 'Dx', 'Dxp',
                                                                         'beta_y', 'alpha_y', 'Dy', 'Dyp'])]
        if self.tw0_real.beta_x != 0.0:
            self.tw0_real.gamma_x = (1 + self.tw0_real.alpha_x ** 2) / self.tw0_real.beta_x
        else:
            self.tw0_real.gamma_x = 0.0
        if self.tw0_real.beta_y != 0.0:
            self.tw0_real.gamma_y = (1 + self.tw0_real.alpha_y ** 2) / self.tw0_real.beta_y
        else:
            self.tw0_real.gamma_y = 0.0
        self.tw0_real = new_tw0
        self.tws_real = twiss(lattice=self.lat_section_ttf.lat, tws0=new_tw0, nPoints=self.twiss_nPoints)
        try:
            self.update_plot_real()
        except:
            logging.error('twiss function did not converge!')
            return

    def set_magnet_value(self, name=None, param=None, value=None):
        try:
            sender = self.sender()
            idx = [el.id for el in self.lat_section_ttf.lat.sequence].index(sender.objectName())
            magnet = self.lat_section_ttf.lat.sequence[idx]
            magnet_des = self.lat_section_design.lat.sequence[idx]
            if magnet.__class__ == Quadrupole:
                self.lat_section_ttf.set_value(magnet.id, k1=sender.value())
                try:
                    diff = (magnet.k1 - magnet_des.k1) / magnet_des.k1 * 100
                except ZeroDivisionError:
                    diff = 12345
            elif magnet.__class__ == Sextupole:
                self.lat_section_ttf.set_value(magnet.id, k2=sender.value())
                try:
                    diff = (magnet.k2 - magnet_des.k2) / magnet_des.k2 * 100
                except ZeroDivisionError:
                    diff = 12345
            # elif magnet.__class__ == Bend or magnet.__class__ == SBend or magnet.__class__ == RBend:
            #    self.lat_section_ttf.set_value(magnet.id, 'angle', np.deg2rad(sender.value()))
            idx_grid = [el.objectName() for el in self.magnet_widget_list].index(sender.objectName())
            diff_widget = self.ui.magnetsLayout.itemAtPosition(idx_grid, 3).widget()
            diff_widget.setText("{:.1f}".format(diff))
        except:
            self.lat_section_ttf.set_value(name, param, value)
            idx = [el.id for el in self.lat_section_ttf.lat.sequence].index(name)
            magnet = self.lat_section_ttf.lat.sequence[idx]
        self.update_magnet_value(magnet)
        self.tws_real = twiss(lattice=self.lat_section_ttf.lat, tws0=self.tw0_real, nPoints=self.twiss_nPoints)
        self.update_plot_real()
        self.calc_timeRes()

    def update_magnet_value(self, magnet):
        if magnet.id in self.lattice_ui_items.keys():
            for item in self.lattice_ui_items[magnet.id]:
                self.ui.lattice_plot.removeItem(item)
            self.lattice_ui_items[magnet.id] = []
        if magnet.__class__ == Quadrupole or magnet.__class__ == Sextupole:
            ctrx = magnet.z0
            qlx = ctrx - float(magnet.l) / 2
            qrx = ctrx + float(magnet.l) / 2
            if magnet.__class__ == Quadrupole:
                if magnet.k1 >= 0:
                    ltop = self.ui.lattice_plot.plot([qlx, qrx], [1.0, 1.0], pen=pg.mkPen((51, 102, 0), width=0.1))
                    lbott = self.ui.lattice_plot.plot([qlx, qrx], [0.0, 0.0], pen=pg.mkPen((51, 102, 0), width=0.1))
                elif magnet.k1 < 0:
                    ltop = self.ui.lattice_plot.plot([qlx, qrx], [0.0, 0.0], pen=pg.mkPen((51, 102, 0), width=0.1))
                    lbott = self.ui.lattice_plot.plot([qlx, qrx], [-1.0, -1.0],
                                                      pen=pg.mkPen((51, 102, 0), width=0.1))
                fill = pg.FillBetweenItem(ltop, lbott, brush=pg.mkBrush((51, 102, 0)))
            elif magnet.__class__ == Sextupole:
                if magnet.k2 >= 0:
                    ltop = self.ui.lattice_plot.plot([qlx, qrx], [1.0, 1.0],
                                                     pen=pg.mkPen((190, 64, 245), width=0.1))
                    lbott = self.ui.lattice_plot.plot([qlx, qrx], [0.0, 0.0],
                                                      pen=pg.mkPen((190, 64, 245), width=0.1))
                elif magnet.k2 < 0:
                    ltop = self.ui.lattice_plot.plot([qlx, qrx], [0.0, 0.0],
                                                     pen=pg.mkPen((190, 64, 245), width=0.1))
                    lbott = self.ui.lattice_plot.plot([qlx, qrx], [-1.0, -1.0],
                                                      pen=pg.mkPen((190, 64, 245), width=0.1))
                fill = pg.FillBetweenItem(ltop, lbott, brush=pg.mkBrush((190, 64, 245)))
            self.ui.lattice_plot.addItem(fill)
            self.lattice_ui_items.update({magnet.id: [ltop, lbott, fill]})
        elif magnet.__class__ in [Bend, SBend, RBend]:
            ctrx = magnet.z0
            angle = magnet.angle
            if abs(angle) < 1e-9:
                orientation = 0
            elif angle > 0:
                orientation = +1
            elif angle < 0:
                orientation = -1
            dlx = ctrx - magnet.l / 2
            dlr = ctrx + magnet.l / 2
            if orientation >= 0:
                dbott = - magnet.l * (np.tan(np.pi / 3) - np.tan(np.pi / 6)) / 2
                dtop = magnet.l * np.tan(np.pi / 6) / 2
                if orientation == 0:
                    color = QColor(0, 77, 153, alpha=75)
                elif orientation == 1:
                    color = QColor(0, 77, 153, alpha=255)
                l0 = self.ui.lattice_plot.plot([dlx, dlr], [dtop, dtop], pen=pg.mkPen(color))
                l1 = self.ui.lattice_plot.plot([dlr, ctrx, dlx], [dtop, dbott, dtop], pen=pg.mkPen(color))
                dfill = pg.FillBetweenItem(l0, l1, brush=pg.mkBrush(color))
                dfill.setZValue(1)
            elif orientation < 0:
                dtop = magnet.l * (np.tan(np.pi / 3) - np.tan(np.pi / 6)) / 2
                dbott = - magnet.l * np.tan(np.pi / 6) / 2
                color = QColor(0, 77, 153, alpha=255)
                l0 = self.ui.lattice_plot.plot([dlx, dlr], [dbott, dbott], pen=pg.mkPen(color))
                l1 = self.ui.lattice_plot.plot([dlr, ctrx, dlx], [dbott, dtop, dbott], pen=pg.mkPen(color))
                dfill = pg.FillBetweenItem(l0, l1, brush=pg.mkBrush(color))
                dfill.setZValue(1)
            self.ui.lattice_plot.addItem(dfill)
            self.lattice_ui_items.update({magnet.id: [l0, l1, dfill]})

    def _plot_lattice(self):
        self.ui.plot_top.setXRange(self.lat_section_design.lat.sequence[0].z0,
                                   self.lat_section_design.lat.sequence[-1].z0)
        self.ui.plot_bottom.setXRange(self.lat_section_design.lat.sequence[0].z0,
                                      self.lat_section_design.lat.sequence[-1].z0)
        if len(flatten_items_dict(self.lattice_ui_items)) != 0:
            for item in flatten_items_dict(self.lattice_ui_items):
                self.ui.lattice_plot.removeItem(item)
        self.lattice_ui_items = {}
        monitor_spots = []
        baseline = self.ui.lattice_plot.plot([self.lat_section_design.lat.sequence[0].z0,
                                              self.lat_section_design.lat.sequence[-1].z0],
                                             [0.0, 0.0],
                                             pen=pg.mkPen('#94b8b8'))
        self.lattice_ui_items.update({'baseline': baseline})
        for elem in self.lat_section_design.lat.sequence:
            if type(elem) in [Quadrupole, Sextupole, Bend, SBend, RBend]:
                self.update_magnet_value(elem)
            elif type(elem) == Monitor:
                monitor_spots.append({'pos': (elem.z0, 0.0), 'data': 1})
            elif type(elem) == TDCavity:
                ctrx = elem.z0
                tdsl = ctrx - elem.l / 2
                tdsr = ctrx + elem.l / 2
                l0 = self.ui.lattice_plot.plot([tdsl, tdsr], [0.3, 0.3], pen=pg.mkPen((153, 0, 0)))
                l1 = self.ui.lattice_plot.plot([tdsl, tdsr], [-0.3, -0.3], pen=pg.mkPen((153, 0, 0)))
                tdsfill = pg.FillBetweenItem(l0, l1, brush=pg.mkBrush((153, 0, 0)))
                tdsfill.setZValue(1)
                self.ui.lattice_plot.addItem(tdsfill)
                self.lattice_ui_items.update({elem.id: [l0, l1, tdsfill]})
            elif type(elem) == Marker and not np.any([a in elem.id for a in ['SCR', '-C', 'SCRAPER']]):
                ctrx = elem.z0
                scr = self.ui.lattice_plot.plot([ctrx, ctrx], [-0.5, 0.5], pen=pg.mkPen((255, 71, 26), width=1.))
                self.lattice_ui_items.update({elem.id: scr})
        s1 = pg.ScatterPlotItem(size=5, pen=pg.mkPen((204, 122, 0)), brush=pg.mkBrush(204, 122, 0, 255))
        s1.addPoints(monitor_spots)
        self.ui.lattice_plot.addItem(s1)
        self.lattice_ui_items.update({'bpms': s1})

    def _plot_top_design(self):
        if len(self.top_plot_ui_items['design']) != 0:
            for item in self.top_plot_ui_items['design']:
                self.ui.plot_top.removeItem(item)
            self.top_plot_ui_items['design'] = []
        func = self.ui.top_function.currentText()
        if self.ui.axtop_xactive.isChecked():
            if func == 'beta':
                l = self.ui.plot_top.plot([t.s for t in self.tws_design], [t.beta_x for t in self.tws_design],
                                          pen=pg.mkPen((0, 134, 179), width=1.0, style=Qt.DashDotLine))
                self.top_plot_ui_items['design'].append(l)
            elif func == 'sqrt(beta)':
                l = self.ui.plot_top.plot([t.s for t in self.tws_design], np.sqrt([t.beta_x for t in self.tws_design]),
                                          pen=pg.mkPen((0, 134, 179), width=1.0, style=Qt.DashDotLine))
                self.top_plot_ui_items['design'].append(l)
            elif func == 'alpha':
                l = self.ui.plot_top.plot([t.s for t in self.tws_design], [t.alpha_x for t in self.tws_design],
                                          pen=pg.mkPen((0, 134, 179), width=1.0, style=Qt.DashDotLine))
                self.top_plot_ui_items['design'].append(l)
            elif func == 'phase advance':
                mux0 = self.tws_design[0].mux
                l = self.ui.plot_top.plot([t.s for t in self.tws_design], [t.mux - mux0 for t in self.tws_design],
                                          pen=pg.mkPen((0, 134, 179), width=1.0, style=Qt.DashDotLine))
                self.top_plot_ui_items['design'].append(l)
            elif func == 'dispersion':
                l = self.ui.plot_top.plot([t.s for t in self.tws_design], [t.Dx for t in self.tws_design],
                                          pen=pg.mkPen((0, 134, 179), width=1.0, style=Qt.DashDotLine))
                self.top_plot_ui_items['design'].append(l)
        if self.ui.axtop_yactive.isChecked():
            if func == 'beta':
                l = self.ui.plot_top.plot([t.s for t in self.tws_design], [t.beta_y for t in self.tws_design],
                                          pen=pg.mkPen((153, 0, 0), width=1.0, style=Qt.DashDotLine))
                self.top_plot_ui_items['design'].append(l)
            elif func == 'sqrt(beta)':
                l = self.ui.plot_top.plot([t.s for t in self.tws_design], np.sqrt([t.beta_y for t in self.tws_design]),
                                          pen=pg.mkPen((153, 0, 0), width=1.0, style=Qt.DashDotLine))
                self.top_plot_ui_items['design'].append(l)
            elif func == 'alpha':
                l = self.ui.plot_top.plot([t.s for t in self.tws_design], [t.alpha_y for t in self.tws_design],
                                          pen=pg.mkPen((153, 0, 0), width=1.0, style=Qt.DashDotLine))
                self.top_plot_ui_items['design'].append(l)
            elif func == 'phase advance':
                muy0 = self.tws_design[0].muy
                l = self.ui.plot_top.plot([t.s for t in self.tws_design], [t.muy - muy0 for t in self.tws_design],
                                          pen=pg.mkPen((153, 0, 0), width=1.0, style=Qt.DashDotLine))
                self.top_plot_ui_items['design'].append(l)
            elif func == 'dispersion':
                l = self.ui.plot_top.plot([t.s for t in self.tws_design], [t.Dy for t in self.tws_design],
                                          pen=pg.mkPen((153, 0, 0), width=1.0, style=Qt.DashDotLine))
                self.top_plot_ui_items['design'].append(l)

    def _plot_bottom_design(self):
        if len(self.bottom_plot_ui_items['design']) != 0:
            for item in self.bottom_plot_ui_items['design']:
                self.ui.plot_bottom.removeItem(item)
            self.bottom_plot_ui_items['design'] = []
        func = self.ui.bottom_function.currentText()
        if self.ui.axbottom_xactive.isChecked():
            if func == 'beta':
                l = self.ui.plot_bottom.plot([t.s for t in self.tws_design], [t.beta_x for t in self.tws_design],
                                             pen=pg.mkPen((0, 134, 179), width=1.0, style=Qt.DashDotLine))
                self.bottom_plot_ui_items['design'].append(l)
            elif func == 'sqrt(beta)':
                l = self.ui.plot_bottom.plot([t.s for t in self.tws_design],
                                             np.sqrt([t.beta_x for t in self.tws_design]),
                                             pen=pg.mkPen((0, 134, 179), width=1.0, style=Qt.DashDotLine))
                self.bottom_plot_ui_items['design'].append(l)
            elif func == 'alpha':
                l = self.ui.plot_bottom.plot([t.s for t in self.tws_design], [t.alpha_x for t in self.tws_design],
                                             pen=pg.mkPen((0, 134, 179), width=1.0, style=Qt.DashDotLine))
                self.bottom_plot_ui_items['design'].append(l)
            elif func == 'phase advance':
                mux0 = self.tws_design[0].mux
                l = self.ui.plot_bottom.plot([t.s for t in self.tws_design], [t.mux - mux0 for t in self.tws_design],
                                             pen=pg.mkPen((0, 134, 179), width=1.0, style=Qt.DashDotLine))
                self.bottom_plot_ui_items['design'].append(l)
            elif func == 'dispersion':
                l = self.ui.plot_bottom.plot([t.s for t in self.tws_design], [t.Dx for t in self.tws_design],
                                             pen=pg.mkPen((0, 134, 179), width=1.0, style=Qt.DashDotLine))
                self.bottom_plot_ui_items['design'].append(l)
        if self.ui.axbottom_yactive.isChecked():
            if func == 'beta':
                l = self.ui.plot_bottom.plot([t.s for t in self.tws_design], [t.beta_y for t in self.tws_design],
                                             pen=pg.mkPen((153, 0, 0), width=1.0, style=Qt.DashDotLine))
                self.bottom_plot_ui_items['design'].append(l)
            elif func == 'alpha':
                l = self.ui.plot_bottom.plot([t.s for t in self.tws_design], [t.alpha_y for t in self.tws_design],
                                             pen=pg.mkPen((153, 0, 0), width=1.0, style=Qt.DashDotLine))
                self.bottom_plot_ui_items['design'].append(l)
            elif func == 'phase advance':
                muy0 = self.tws_design[0].muy
                l = self.ui.plot_bottom.plot([t.s for t in self.tws_design], [t.muy - muy0 for t in self.tws_design],
                                             pen=pg.mkPen((153, 0, 0), width=1.0, style=Qt.DashDotLine))
                self.bottom_plot_ui_items['design'].append(l)
            elif func == 'dispersion':
                l = self.ui.plot_bottom.plot([t.s for t in self.tws_design], [t.Dy for t in self.tws_design],
                                             pen=pg.mkPen((153, 0, 0), width=1.0, style=Qt.DashDotLine))
                self.bottom_plot_ui_items['design'].append(l)

    def update_plot_design(self):
        self.set_log_scale()
        self._plot_top_design()
        self._plot_bottom_design()

    def _plot_top_real(self):
        if len(self.top_plot_ui_items['real']) != 0:
            for item in self.top_plot_ui_items['real']:
                self.ui.plot_top.removeItem(item)
            self.top_plot_ui_items['real'] = []
        func = self.ui.top_function.currentText()
        if self.ui.axtop_xactive.isChecked():
            if func == 'beta':
                l = self.ui.plot_top.plot([t.s for t in self.tws_design], [t.beta_x for t in self.tws_real],
                                          pen=pg.mkPen((0, 134, 179), width=2.0))
                self.top_plot_ui_items['real'].append(l)
                self.ui.plot_top.getAxis('left').setLabel(text='beta', units='m')
            elif func == 'sqrt(beta)':
                l = self.ui.plot_top.plot([t.s for t in self.tws_design], np.sqrt([t.beta_x for t in self.tws_real]),
                                          pen=pg.mkPen((0, 134, 179), width=2.0))
                self.top_plot_ui_items['real'].append(l)
                self.ui.plot_top.getAxis('left').setLabel(text='sqrt(beta)', units='m<sup>-1/2</sup>')
            elif func == 'alpha':
                l = self.ui.plot_top.plot([t.s for t in self.tws_design], [t.alpha_x for t in self.tws_real],
                                          pen=pg.mkPen((0, 134, 179), width=2.0))
                self.top_plot_ui_items['real'].append(l)
                self.ui.plot_top.getAxis('left').setLabel(text='alpha')
            elif func == 'phase advance':
                mux0 = self.tws_real[0].mux
                l = self.ui.plot_top.plot([t.s for t in self.tws_design], [t.mux - mux0 for t in self.tws_real],
                                          pen=pg.mkPen((0, 134, 179), width=2.0))
                self.top_plot_ui_items['real'].append(l)
                self.ui.plot_top.getAxis('left').setLabel(text='phase advance', units='rad')
            elif func == 'dispersion':
                l = self.ui.plot_top.plot([t.s for t in self.tws_design], [t.Dx for t in self.tws_real],
                                          pen=pg.mkPen((0, 134, 179), width=2.0))
                self.top_plot_ui_items['real'].append(l)
                self.ui.plot_top.getAxis('left').setLabel(text='R<sub>16</sub> / R<sub>36</sub>', units='m')
        if self.ui.axtop_yactive.isChecked():
            if func == 'beta':
                l = self.ui.plot_top.plot([t.s for t in self.tws_design], [t.beta_y for t in self.tws_real],
                                          pen=pg.mkPen((153, 0, 0), width=2.0))
                self.top_plot_ui_items['real'].append(l)
            elif func == 'sqrt(beta)':
                l = self.ui.plot_top.plot([t.s for t in self.tws_design], np.sqrt([t.beta_y for t in self.tws_real]),
                                          pen=pg.mkPen((153, 0, 0), width=2.0))
                self.top_plot_ui_items['real'].append(l)
            elif func == 'alpha':
                l = self.ui.plot_top.plot([t.s for t in self.tws_design], [t.alpha_y for t in self.tws_real],
                                          pen=pg.mkPen((153, 0, 0), width=2.0))
                self.top_plot_ui_items['real'].append(l)
            elif func == 'phase advance':
                muy0 = self.tws_real[0].muy
                l = self.ui.plot_top.plot([t.s for t in self.tws_design], [t.muy - muy0 for t in self.tws_real],
                                          pen=pg.mkPen((153, 0, 0), width=2.0))
                self.top_plot_ui_items['real'].append(l)
            elif func == 'dispersion':
                l = self.ui.plot_top.plot([t.s for t in self.tws_design], [t.Dy for t in self.tws_real],
                                          pen=pg.mkPen((153, 0, 0), width=2.0))
                self.top_plot_ui_items['real'].append(l)

    def _plot_bottom_real(self):
        if len(self.bottom_plot_ui_items['real']) != 0:
            for item in self.bottom_plot_ui_items['real']:
                self.ui.plot_bottom.removeItem(item)
            self.bottom_plot_ui_items['real'] = []
        func = self.ui.bottom_function.currentText()
        if self.ui.axbottom_xactive.isChecked():
            if func == 'beta':
                l = self.ui.plot_bottom.plot([t.s for t in self.tws_design], [t.beta_x for t in self.tws_real],
                                             pen=pg.mkPen((0, 134, 179), width=2.0))
                self.bottom_plot_ui_items['real'].append(l)
                self.ui.plot_bottom.getAxis('left').setLabel(text='beta', units='m')
            elif func == 'sqrt(beta)':
                l = self.ui.plot_bottom.plot([t.s for t in self.tws_design], np.sqrt([t.beta_x for t in self.tws_real]),
                                             pen=pg.mkPen((0, 134, 179), width=2.0))
                self.bottom_plot_ui_items['real'].append(l)
                self.ui.plot_bottom.getAxis('left').setLabel(text='sqrt(beta)', units='m<sup>-1/2</sup>')
            elif func == 'alpha':
                l = self.ui.plot_bottom.plot([t.s for t in self.tws_design], [t.alpha_x for t in self.tws_real],
                                             pen=pg.mkPen((0, 134, 179), width=2.0))
                self.bottom_plot_ui_items['real'].append(l)
                self.ui.plot_bottom.getAxis('left').setLabel(text='alpha')
            elif func == 'phase advance':
                mux0 = self.tws_real[0].mux
                l = self.ui.plot_bottom.plot([t.s for t in self.tws_design], [t.mux - mux0 for t in self.tws_real],
                                             pen=pg.mkPen((0, 134, 179), width=2.0))
                self.bottom_plot_ui_items['real'].append(l)
                self.ui.plot_bottom.getAxis('left').setLabel(text='phase advance', units='rad')
            elif func == 'dispersion':
                l = self.ui.plot_bottom.plot([t.s for t in self.tws_design], [t.Dx for t in self.tws_real],
                                             pen=pg.mkPen((0, 134, 179), width=2.0))
                self.bottom_plot_ui_items['real'].append(l)
                self.ui.plot_bottom.getAxis('left').setLabel(text='R<sub>16</sub> / R<sub>36</sub>', units='m')
        if self.ui.axbottom_yactive.isChecked():
            if func == 'beta':
                l = self.ui.plot_bottom.plot([t.s for t in self.tws_design], [t.beta_y for t in self.tws_real],
                                             pen=pg.mkPen((153, 0, 0), width=2.0))
                self.bottom_plot_ui_items['real'].append(l)
            elif func == 'alpha':
                l = self.ui.plot_bottom.plot([t.s for t in self.tws_design], [t.alpha_y for t in self.tws_real],
                                             pen=pg.mkPen((153, 0, 0), width=2.0))
                self.bottom_plot_ui_items['real'].append(l)
            elif func == 'phase advance':
                muy0 = self.tws_real[0].muy
                l = self.ui.plot_bottom.plot([t.s for t in self.tws_design], [t.muy - muy0 for t in self.tws_real],
                                             pen=pg.mkPen((153, 0, 0), width=2.0))
                self.bottom_plot_ui_items['real'].append(l)
            elif func == 'dispersion':
                l = self.ui.plot_bottom.plot([t.s for t in self.tws_design], [t.Dy for t in self.tws_real],
                                             pen=pg.mkPen((153, 0, 0), width=2.0))
                self.bottom_plot_ui_items['real'].append(l)

    def update_plot_real(self):
        self.set_log_scale()
        self._plot_top_real()
        self._plot_bottom_real()

    def update_top(self):
        self.set_log_scale()
        self._plot_top_design()
        self._plot_top_real()

    def update_bottom(self):
        self.set_log_scale()
        self._plot_bottom_design()
        self._plot_bottom_real()

    def set_log_scale(self):
        if self.ui.axtop_log.isChecked():
            if self.ui.top_function.currentText() in ['alpha', 'dispersion']:
                self.ui.axtop_log.blockSignals(True)
                self.ui.axtop_log.setChecked(False)
                self.ui.axtop_log.blockSignals(False)
                pass
            else:
                self.ui.plot_top.setLogMode(False, True)
        else:
            self.ui.plot_top.setLogMode(False, False)
        if self.ui.axbottom_log.isChecked():
            if self.ui.bottom_function.currentText() in ['alpha', 'dispersion']:
                self.ui.axbottom_log.blockSignals(True)
                self.ui.axbottom_log.setChecked(False)
                self.ui.axbottom_log.blockSignals(False)
                pass
            else:
                self.ui.plot_bottom.setLogMode(False, True)
        else:
            self.ui.plot_bottom.setLogMode(False, False)

    def set_autoscale_top(self):
        if self.ui.axtop_autoscale.isChecked():
            self.ui.plot_top.enableAutoRange('y', True)
        else:
            self.ui.plot_top.enableAutoRange('y', False)

    def set_autoscale_bottom(self):
        if self.ui.axbottom_autoscale.isChecked():
            self.ui.plot_bottom.enableAutoRange('y', True)
        else:
            self.ui.plot_bottom.enableAutoRange('y', False)

    def _set_magnet_stepsize(self, value):
        for widget in self.magnet_widget_list:
            widget.setSingleStep(float(value))

    def _set_push_all_magnets(self):
        state = self.ui.magnets_push_all.isChecked()
        if self.magnet_push_list:
            for box in self.magnet_push_list:
                box.setChecked(state)

    # matching functions
    def select_match_preset(self, preset):
        self.ui.betax.blockSignals(False)
        self.ui.betay.blockSignals(False)
        self.ui.alphax.blockSignals(False)
        self.ui.alphay.blockSignals(False)
        self.ui.betax.setStyleSheet("background-color: #FFFFFF")
        self.ui.alphax.setStyleSheet("background-color: #FFFFFF")
        self.ui.betay.setStyleSheet("background-color: #FFFFFF")
        self.ui.alphay.setStyleSheet("background-color: #FFFFFF")
        self.ui.Dx.setEnabled(True)
        self.ui.Dxp.setEnabled(True)
        self.ui.Dy.setEnabled(True)
        self.ui.Dyp.setEnabled(True)
        self.ui.tw0_real_label.setText("Real")
        if preset == 'theory':
            self.ui.match_var_1_active.setText('\u03b2x (m) =')
            self.ui.match_var_2_active.setText('\u03b2y (m) =')
            self.ui.match_combo_label_1.setText('match start')
            self.ui.match_combo_label_2.setText('match target')
            self.ui.match_ref_plane.setEnabled(True)
            self.ui.match_target_plane.setEnabled(True)
            self.ui.match_var_1_active.setEnabled(False)
            self.ui.match_var_1.setEnabled(False)
            self.ui.match_var_2_active.setEnabled(False)
            self.ui.match_var_2.setEnabled(False)
            self.ui.match_var_3_active.setEnabled(False)
            self.ui.match_var_3.setEnabled(False)
        else:
            self.ui.match_target_plane.setEnabled(False)
            self.ui.match_var_1_active.setEnabled(True)
            self.ui.match_var_1.setEnabled(True)
            self.ui.match_var_2_active.setEnabled(True)
            self.ui.match_var_2.setEnabled(True)
            self.ui.match_var_3_active.setEnabled(False)
            self.ui.match_var_3.setEnabled(False)
            self.ui.match_combo_label_2.setText('')
            if preset == 'compression':
                self.ui.match_ref_plane.setEnabled(False)
                self.ui.match_combo_label_1.setText('')
                self.ui.match_var_1_active.setText("R56 (mm) =")
                self.ui.match_var_1.setRange(-10.0, 10.0)
                self.ui.match_var_1.setValue(0.0)
                self.ui.match_var_1_active.setChecked(True)
                self.ui.match_var_2_active.setText("\u03b2scraper (m) <")
                self.ui.match_var_2.setRange(0.01, 20.0)
                self.ui.match_var_2.setValue(1.7)
                self.ui.match_var_2_active.setChecked(True)
                QMessageBox.information(self, "Compressopm match", "For \u03b2x at the scraper < 1.6 m, the dispersion "
                                                                   "at the end of FLFCOMP is hardly closed.\n\n"
                                                                   "If additionally a specific R56 is given, the "
                                                                   "optimisation is even harder, but for \u03b2x < 6.0 m"
                                                                   " everything is possible.",
                                        QMessageBox.Ok)
            elif preset == 'plasma cell':
                self.ui.match_ref_plane.setEnabled(False)
                self.ui.match_combo_label_1.setText('')
                self.ui.match_var_1_active.setText('\u03b2x (m) =')
                self.ui.match_var_2_active.setText('\u03b2y (m) =')
                self.ui.match_var_1.setRange(0.001, 100.0)
                self.ui.match_var_2.setRange(0.001, 100.0)
                self.ui.match_var_1.setValue(1.0)
                self.ui.match_var_2.setValue(1.0)
                self.ui.match_var_1_active.setChecked(True)
                self.ui.match_var_2_active.setChecked(True)
                self.ui.match_var_1.setSingleStep(0.001)
                self.ui.match_var_2.setSingleStep(0.001)
                self.ui.match_var_3_active.setEnabled(True)
                self.ui.match_var_3_active.setChecked(True)
                self.ui.match_var_3.setEnabled(True)
                self.ui.match_var_3.setRange(-0.250, 0.250)
                self.ui.match_var_3.setValue(-0.194)
                self.ui.match_var_3_active.setText('z (m) =')
                QMessageBox.information(self, "Plasma cell match", "z = -0.194 is the default position of the plasma "
                                                                   "cell input.",
                                        QMessageBox.Ok)

            elif preset == 'beam capture':
                self.ui.match_ref_plane.setEnabled(False)
                self.ui.match_combo_label_1.setText('')
                self.ui.betax.blockSignals(True)
                self.ui.betay.blockSignals(True)
                self.ui.alphax.blockSignals(True)
                self.ui.alphay.blockSignals(True)
                self.ui.betax.setStyleSheet("background-color: #F4A347")
                self.ui.alphax.setStyleSheet("background-color: #F4A347")
                self.ui.betay.setStyleSheet("background-color: #F4A347")
                self.ui.alphay.setStyleSheet("background-color: #F4A347")
                self.ui.Dx.setEnabled(False)
                self.ui.Dxp.setEnabled(False)
                self.ui.Dy.setEnabled(False)
                self.ui.Dyp.setEnabled(False)
                self.ui.match_var_1_active.setEnabled(False)
                self.ui.match_var_1.setEnabled(False)
                self.ui.match_var_2_active.setEnabled(False)
                self.ui.match_var_2.setEnabled(False)
                self.ui.match_var_3_active.setEnabled(True)
                self.ui.match_var_3_active.setChecked(True)
                self.ui.match_var_3.setEnabled(True)
                self.ui.match_var_3.setRange(-0.25, 0.25)
                self.ui.match_var_3.setValue(0.0)
                self.ui.match_var_3_active.setText('z (m) =')
                self.ui.tw0_real_label.setText("Center target")
                tw0_real_new = twiss_at('STARTFLFDIAG', self.lat_section_ttf.lat, self.tws_real)
                self.ui.betax.setValue(tw0_real_new.beta_x)
                self.ui.alphax.setValue(tw0_real_new.alpha_x)
                self.ui.betay.setValue(tw0_real_new.beta_y)
                self.ui.alphay.setValue(tw0_real_new.alpha_y)
                QMessageBox.information(self, "Beam capture match", "1. The starting Twiss parameters (typically those"
                                                                    " at the output of the cell) must be given "
                                                                    "in the \"Input Twiss params\" group box.\n\n"
                                                                    "2. \"z\" is the distance from the center of the "
                                                                    "plasma chamber (Center target).", QMessageBox.Ok)
            elif preset == 'tomography':
                self.ui.match_ref_plane.setEnabled(False)
                self.ui.match_combo_label_1.setText('')
                self.ui.match_var_1_active.setText('\u03b2_TDS (m) =')
                self.ui.match_var_2_active.setText('\u03b2_SCR (m) =')
                self.ui.match_var_1.setRange(0.001, 100.0)
                self.ui.match_var_2.setRange(0.001, 100.0)
                self.ui.match_var_1.setValue(1.0)
                self.ui.match_var_2.setValue(1.0)
                self.ui.match_var_1_active.setChecked(True)
                self.ui.match_var_1.setSingleStep(0.1)
                self.ui.match_var_2.setSingleStep(0.1)
                self.ui.match_var_3_active.setEnabled(False)
                self.ui.match_var_3.setEnabled(False)
            else:
                self.ui.match_ref_plane.setEnabled(True)
                self.ui.match_combo_label_1.setText('match start')
                if 'TDS-C' in [el.id for el in self.lat_section_ttf.lat.sequence]:
                    # idx = self.ui.match_ref_plane.
                    self.ui.match_ref_plane.setCurrentText('TDS-C')
                self.ui.match_var_1_active.setText('\u03b2x screen (m) =')
                self.ui.match_var_1_active.setChecked(True)
                self.ui.match_var_1_active.setEnabled(True)
                self.ui.match_var_1.setEnabled(True)
                self.ui.match_var_2_active.setText('\u03b2y screen (m) =')
                self.ui.match_var_2_active.setChecked(True)
                self.ui.match_var_2_active.setEnabled(True)
                self.ui.match_var_2.setEnabled(True)
                self.ui.match_var_3_active.setEnabled(False)
                self.ui.match_var_3.setEnabled(False)
                if preset == '11FLFXTDS':
                    if self.ui.streaking_plane.currentText() == 'Y':
                        self.ui.match_var_3_active.setText('\u03b2y TDS (m) =')
                    elif self.ui.streaking_plane.currentText() == 'X':
                        self.ui.match_var_3_active.setText('\u03b2x TDS (m) =')
                elif preset == '8FLFDUMP':
                    self.ui.streaking_plane.setCurrentText('Y')
                    self.ui.match_var_3_active.setText('\u03b2y TDS (m) =')
                self.ui.match_var_3_active.setChecked(True)
                self.ui.match_var_3_active.setEnabled(True)
                self.ui.match_var_3.setEnabled(True)
                self.ui.match_var_1.setRange(0.001, 100.0)
                self.ui.match_var_2.setRange(0.001, 100.0)
                self.ui.match_var_3.setRange(10.0, 150.0)
                self.ui.match_var_1.setValue(1.0)
                self.ui.match_var_2.setValue(1.0)
                self.ui.match_var_3.setValue(100.0)
                self.ui.match_var_1.setSingleStep(0.1)
                self.ui.match_var_2.setSingleStep(0.1)

    def do_match(self):
        self.ui.do_match.setEnabled(False)

        if self.ui.match_preset.currentText() == 'theory':
            ref_name = self.ui.match_ref_plane.currentText()
            idx0 = [el.id for el in self.lat_section_ttf.lat.sequence].index(ref_name)
            tw0 = twiss_at(ref_name, self.lat_section_ttf.lat, self.tws_real)
            target_name = self.ui.match_target_plane.currentText()
            twf = twiss_at(target_name, self.lat_section_ttf.lat, self.tws_design)
            idx1 = [el.id for el in self.lat_section_ttf.lat.sequence].index(target_name)
            new_seq = deepcopy(self.lat_section_ttf.lat.sequence[idx0:idx1 + 1])
            m0 = new_seq[-1]
            quads = [el for el in new_seq if el.__class__ == Quadrupole]
            constraints = {m0: {'beta_x': twf.beta_x, 'alpha_x': twf.alpha_x,
                                'beta_y': twf.beta_y, 'alpha_y': twf.alpha_y}}

            def match_weights(val):
                if val in ['beta_x', 'beta_y', 'alpha_x', 'alpha_y']: return 5.0
                return 1.0

            lat_n = MagneticLattice(new_seq)
        elif self.ui.match_preset.currentText() == 'compression':
            tw0 = deepcopy(self.lat_section_ttf.tw0_des)
            idx0 = [el.id for el in self.lat_section_ttf.lat.sequence].index('STARTFLFEXTR')
            idx1 = [el.id for el in self.lat_section_ttf.lat.sequence].index('ENDFLFCOMP')
            new_seq = deepcopy(self.lat_section_ttf.lat.sequence[idx0:idx1 + 1])
            idx_rb = [el.id for el in new_seq].index('STARTREVBEND')
            m0 = new_seq[idx_rb]
            idx_scraper = [el.id for el in new_seq].index('SCRAPER-C')
            m1 = new_seq[idx_scraper]
            idx_end = [el.id for el in new_seq].index('ENDFLFCOMP')
            m2 = new_seq[idx_end]
            Dx = 0.47826979182009793 - 70.50367135458421 * self.ui.match_var_1.value() * 1e-3
            beta_scraper = self.ui.match_var_2.value()
            quads = [el for el in new_seq if type(el) == Quadrupole]
            if self.ui.match_var_1_active.isChecked() and self.ui.match_var_2_active.isChecked():
                constraints = {'global': {'beta_x': ["<", 150],
                                          'beta_y': ["<", 150],
                                          'Dx': ["<", 2.0]},
                               m0: {'Dx': Dx},
                               m1: {'beta_x': ["<", beta_scraper]},
                               m2: {'beta_x': ["<", 20.0],
                                    'beta_y': ["<", 20.0],
                                    'Dx': 0.0,
                                    'Dxp': 0.0}}
            elif self.ui.match_var_1_active.isChecked():
                constraints = {'global': {'beta_x': ["<", 150],
                                          'beta_y': ["<", 150],
                                          'Dx': ["<", 2.0]},
                               m0: {'Dx': Dx},
                               m2: {'beta_x': ["<", 20.0],
                                    'beta_y': ["<", 20.0],
                                    'Dx': 0.0,
                                    'Dxp': 0.0}}
            elif self.ui.match_var_2_active.isChecked():
                constraints = {'global': {'beta_x': ["<", 150],
                                          'beta_y': ["<", 150],
                                          'Dx': ["<", 2.0]},
                               m1: {'beta_x': ["<", beta_scraper]},
                               m2: {'beta_x': ["<", 20.0],
                                    'beta_y': ["<", 20.0],
                                    'Dx': 0.0,
                                    'Dxp': 0.0}}

            def match_weights(val):
                if val in ['Dx', 'Dxp']: return 10.0
                if val in ['beta_x', 'beta_y']: return 5.0
                return 1.0

            lat_n = MagneticLattice(new_seq)
        elif self.ui.match_preset.currentText() == 'plasma cell':
            idx0 = [el.id for el in self.lat_section_ttf.lat.sequence].index('STARTFLFMAFF')
            tw0 = twiss_at('STARTFLFMAFF', self.lat_section_ttf.lat, self.tws_real)
            idx1 = [el.id for el in self.lat_section_ttf.lat.sequence].index('FOCALPOINT')
            # twf = twiss_at('FOCALPOINT', self.lat_section_ttf.lat, self.tws_design)
            new_seq = deepcopy(self.lat_section_ttf.lat.sequence[idx0:idx1 + 1])
            if self.ui.match_var_3_active.isChecked():
                focalpoint_shift = self.ui.match_var_3.value() + 0.194
                idx_prefocal = [el.id for el in new_seq].index('L030063')
                drift_pre = new_seq[idx_prefocal]
                drift_pre.l += focalpoint_shift
            m0 = new_seq[-1]
            quads = [el for el in new_seq if el.__class__ == Quadrupole]
            """
            if min(self.ui.match_var_1.value(), self.ui.match_var_2.value()) < 1.0:
                with open('./optics/magnets_strong_focusing_seed.json', 'r') as jf:
                    opts_seed = json.load(jf)
                for quad, params in zip(quads, opts_seed.values()):
                    quad.k1 = params['k1']
            """
            beta_max = max(100, 3 / min(self.ui.match_var_1.value(), self.ui.match_var_2.value()))
            if self.ui.match_var_1_active.isChecked() and self.ui.match_var_2_active.isChecked():
                constraints = {'global': {'beta_x': ["<", beta_max],
                                          'beta_y': ["<", beta_max]},
                               m0: {'beta_x': self.ui.match_var_1.value(),
                                    'alpha_x': 0.0,
                                    'beta_y': self.ui.match_var_2.value(),
                                    'alpha_y': 0.0}}
            elif self.ui.match_var_1_active.isChecked() and not self.ui.match_var_2_active.isChecked():
                constraints = {'global': {'beta_x': ["<", beta_max],
                                          'beta_y': ["<", beta_max]},
                               m0: {'beta_x': self.ui.match_var_1.value(),
                                    'alpha_x': 0.0}}
            elif not self.ui.match_var_1_active.isChecked() and self.ui.match_var_2_active.isChecked():
                constraints = {'global': {'beta_x': ["<", beta_max],
                                          'beta_y': ["<", beta_max]},
                               m0: {'beta_x': self.ui.match_var_2.value(),
                                    'alpha_x': 0.0}}

            def match_weights(val):
                if val in ['beta_x', 'beta_y', 'alpha_x', 'alpha_y']: return 10.0
                return 1.0

            lat_n = MagneticLattice(new_seq)
        elif self.ui.match_preset.currentText() == 'beam capture':
            idx_startdiag = [el.id for el in self.lat_section_ttf.lat.sequence].index('STARTFLFDIAG')
            idx_tds = [el.id for el in self.lat_section_ttf.lat.sequence].index('TDS-C')
            tw0 = Twiss()
            tw0.beta_x = self.ui.betax.value()
            tw0.alpha_x = self.ui.alphax.value()
            tw0.beta_y = self.ui.betay.value()
            tw0.alpha_y = self.ui.alphay.value()
            self.tw0_real = deepcopy(tw0)
            new_seq = deepcopy(self.lat_section_ttf.lat.sequence[idx_startdiag:idx_tds + 1])
            if self.ui.match_var_3_active.isChecked():
                focalpoint_shift = self.ui.match_var_3.value()
                idx_postfocal = [el.id for el in new_seq].index('L030064')
                drift_post = new_seq[idx_postfocal]
                drift_post.l -= focalpoint_shift
            m1 = new_seq[-1]
            quads = [el for el in new_seq if type(el) == Quadrupole]
            quads.pop(1)
            quads.pop(2)
            if min(tw0.beta_x, tw0.beta_y) < 1.0:
                seed_file = './optics/magnets_capture_strong_seed.json'
            else:
                seed_file = './optics/magnets_capture_soft_seed.json'
            with open(seed_file, 'r') as jf:
                seed_opts = json.load(jf)
            for quad, param in zip(quads, seed_opts.values()):
                quad.k1 = param['k1']
                print(quad.id, quad.k1)
            beta_max = max(100, 3 / min(self.tw0_real.beta_x, self.tw0_real.beta_y))
            constraints = {'global': {'beta_x': ["<", beta_max],
                                      'beta_y': ["<", beta_max]},
                           m1: {'beta_y': 100.0,
                                'alpha_y': 0.0,
                                'beta_x': ["<", 50.0],
                                'alpha_x': 0.0}}

            def match_weights(val):
                if val in ['beta_x', 'beta_y']: return 10.0
                if val in ['alpha_x', 'alpha_y']: return 1.0
                return 1.0

            lat_n = MagneticLattice(new_seq)
        elif self.ui.match_preset.currentText() == '11FLFXTDS' or self.ui.match_preset.currentText() == '8FLFDUMP':
            idx_start = [el.id for el in self.lat_section_ttf.lat.sequence].index(self.ui.match_ref_plane.currentText())
            idx1 = [el.id for el in self.lat_section_ttf.lat.sequence].index('SCR' + self.ui.match_preset.currentText())
            tw0 = twiss_at(self.ui.match_ref_plane.currentText(), self.lat_section_ttf.lat, self.tws_real)
            new_seq = deepcopy(self.lat_section_ttf.lat.sequence[idx_start:idx1 + 1])
            idx_tds = [el.id for el in new_seq].index('TDS-C')
            m_tds = new_seq[idx_tds]
            m_tds2 = Marker()
            new_seq.insert(idx_tds, m_tds2)
            m0 = new_seq[-1]
            quads = [el for el in new_seq if el.__class__ == Quadrupole]
            constraints = {'global': {'beta_x': ["<", 100],
                                      'beta_y': ["<", 100]},
                           m0: {}}
            if self.ui.match_var_1_active.isChecked():
                constraints[m0].update({'beta_x': self.ui.match_var_1.value()})
            if self.ui.match_var_2_active.isChecked():
                constraints[m0].update({'beta_y': self.ui.match_var_2.value()})
            if self.ui.streaking_plane.currentText() == 'X':
                constraints.update({m_tds: {'beta_x': self.ui.match_var_3.value(),
                                            'alpha_x': ["<", 1.0],
                                            'beta_y': ["<", 50]},
                                    m_tds2: {'alpha_x': [">", -1.0]}})
            else:
                constraints.update({m_tds: {'beta_y': self.ui.match_var_3.value(),
                                            'alpha_y': ["<", 1.0],
                                            'beta_x': ["<", 50]},
                                    m_tds2: {'alpha_y': [">", -1.0]}})

            def match_weights(val):
                if val in ['beta_x', 'beta_y', 'alpha_x', 'alpha_y']: return 10.0
                if val in ['mux', 'muy']: return 10.0
                return 1.0

            lat_n = MagneticLattice(new_seq)
        elif self.ui.match_preset.currentText() == 'tomography':
            idx0 = [el.id for el in self.lat_section_ttf.lat.sequence].index('FOCALPOINT')
            tw0 = twiss_at('FOCALPOINT', self.lat_section_ttf.lat, self.tws_real)
            idx_scr = [el.id for el in self.lat_section_ttf.lat.sequence].index('SCR11FLFXTDS')
            new_seq = deepcopy(self.lat_section_ttf.lat.sequence[idx0:idx_scr + 1])
            idx_tds = [el.id for el in new_seq].index('TDS-C')
            m0 = new_seq[idx_tds]
            m1 = new_seq[-1]
            quads = [el for el in new_seq if el.__class__ == Quadrupole]
            constraints = {m0: {'beta_x': self.ui.match_var_1.value(), 'beta_y': self.ui.match_var_1.value(),
                                'alpha_x': ["<", 0.0], 'alpha_y': ["<", 0.0]},
                           m1: {'beta_x': self.ui.match_var_2.value(), 'beta_y': self.ui.match_var_2.value(),
                                'alpha_x': 0.0, 'alpha_y': 0.0}}

            def match_weights(val):
                if val in ['beta_x', 'beta_y', 'alpha_x', 'alpha_y']: return 10.0
                return 1.0

            lat_n = MagneticLattice(new_seq)
        self.worker = MatchWorker(parent=self,  # preset=self.ui.match_preset.currentText(),
                                  lat=lat_n, constr=constraints, vars=quads, tw0=tw0, weights=match_weights)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.signalDone.connect(self.update_match_result)
        self.thread.start()

    def update_match_result(self):
        for q, k1 in zip(self.worker.vars, self.worker.result):
            self.lat_section_ttf.set_value(q.id, k1=k1)
            idx = [el.id for el in self.lat_section_ttf.lat.sequence].index(q.id)
            self.update_magnet_value(self.lat_section_ttf.lat.sequence[idx])
            idx = [el.objectName() for el in self.magnet_widget_list].index(q.id)
            self.magnet_widget_list[idx].setValue(k1)

        if self.ui.match_preset.currentText() == 'beam capture':
            new_seq = deepcopy(self.lat_section_ttf.lat.sequence)
            focalpoint_shift = self.ui.match_var_3.value()
            tw0 = Twiss()
            tw0.beta_x = self.ui.betax.value()
            tw0.alpha_x = -1 * self.ui.alphax.value()
            tw0.beta_y = self.ui.betay.value()
            tw0.alpha_y = -1 * self.ui.alphay.value()
            if 'L030063A' in [el.id for el in new_seq]:
                idx_pre = [el.id for el in new_seq].index('L030063A')
                drift_pre = new_seq[idx_pre]
                drift_pre.l += focalpoint_shift
                new_seq_pre = new_seq[:idx_pre + 1]
                new_seq_pre.reverse()
                lat_pre = MagneticLattice(new_seq_pre)
                tw0_new = twiss(lattice=lat_pre, tws0=tw0)[-1]
                tw0_new.alpha_x *= -1.0
                tw0_new.alpha_y *= -1.0
                self.tw0_real = tw0_new
            else:
                drift = Drift(l=focalpoint_shift)
                lat_pre = MagneticLattice([drift])
                tw0_new = twiss(lattice=lat_pre, tws0=tw0)[-1]
                tw0_new.alpha_x *= -1.0
                tw0_new.alpha_y *= -1.0
                self.tw0_real = tw0_new
        self.tws_real = twiss(lattice=self.lat_section_ttf.lat, tws0=self.tw0_real, nPoints=self.twiss_nPoints)
        self.update_plot_real()
        self.ui.do_match.setEnabled(True)
        self.thread.quit()
        self.thread.wait()

    # magnets control functions
    def magnets_pull_from_linac(self):
        print('Pulling values from LINAC...')
        magnets = [el for el in self.lat_section_ttf.lat.sequence if
                   el.__class__ == Quadrupole or el.__class__ == Sextupole]
        for magnet in magnets:
            idx = [el.objectName() for el in self.magnet_widget_list].index(magnet.id)
            if magnet.__class__ == Quadrupole:
                try:
                    magnet.k1 = pydoocs.read('FLASH.MAGNETS/MAGNET.ML/' + magnet.id + '/STRENGTH.SP')['data']
                    self.magnet_widget_list[idx].setValue(magnet.k1)
                except:
                    pass
            else:
                try:
                    magnet.k2 = pydoocs.read('FLASH.MAGNETS/MAGNET.ML/' + magnet.id + '/STRENGTH.SP')['data']
                    self.magnet_widget_list[idx].setValue(magnet.k1)
                except:
                    pass
        self.lat_section_ttf.lat.update_transfer_maps()
        self.tws_real = twiss(lattice=self.lat_section_ttf.lat, tws0=self.tw0_real, nPoints=self.twiss_nPoints)
        self.update_plot_real()

    def magnets_push_to_linac(self):
        msg = QMessageBox.warning(self, "Optics tool", "You are going to modify the magnet currents in the LINAC.\n"
                                                       "Proceed?", QMessageBox.Ok | QMessageBox.No)
        if msg == 65536:
            return
        elif msg == 1024:
            print('Pushing values to LINAC...')
            all_magnets = [elem for elem in self.lat_section_ttf.lat.sequence if (type(elem) == Quadrupole or
                                                                                     type(elem) == Sextupole)]
            seen = {}
            magnets = [seen.setdefault(x, x) for x in all_magnets if x not in seen]
            count_pushed = 0
            count_selected = 0
            for i, magnet in enumerate(magnets):
                if self.magnet_push_list[i].isChecked():
                    count_selected += 1
                    if magnet.__class__ == Quadrupole:
                        try:
                            pydoocs.write('FLASH.MAGNETS/MAGNET.ML/' + magnet.id + '/STRENGTH.SP', magnet.k1)
                            count_pushed += 1
                        except:
                            logging.error("Magnet " + magnet.id + " could not be set into the LINAC")
                            pass
                    elif magnet.__class__ == Sextupole:
                        try:
                            pydoocs.write('FLASH.MAGNETS/MAGNET.ML/' + magnet.id + '/STRENGTH.SP', magnet.k2)
                            count_pushed += 1
                        except:
                            logging.error("Magnet " + magnet.id + " could not be set into the LINAC")
                            pass
            QMessageBox.information(self, "PUSH to linac INFO",
                                    "{} magnet values pushed to linac successfully\nfrom {} selected.".format(
                                        count_pushed, count_selected),
                                    QMessageBox.Ok)

    def _open_optFile(self):
        dlg = QFileDialog(caption='Load new reference file', directory='./')
        dlg.setFileMode(QFileDialog.ExistingFile)
        if dlg.exec_():
            fname_fullpath = dlg.selectedFiles()[0]
            if fname_fullpath.endswith('.json'):
                with open(fname_fullpath, 'r') as jf:
                    opts = json.load(jf)
                if type(next(iter(opts.values()))['k1']) == float:
                    self.ui.optics_file_setting.setValue(0)
                    self.ui.optics_file_setting.setEnabled(False)
                    self.lat_section_design.optics = opts
                elif type(next(iter(next(iter(opts.values())).values()))) == list:
                    self.ui.optics_file_setting.setEnabled(True)
                    num_settings = len(next(iter(next(iter(opts.values())).values())))
                    self.ui.optics_file_setting.setMaximum(num_settings - 1)
                    self.ui.optics_file_setting.setValue(0)
                    self.optics = opts
                    opts_idx0 = {}
                    for magnet, params in self.optics.items():
                        opts_idx0.update({magnet: {}})
                        for par, val in params.items():
                            opts_idx0[magnet].update({par: val[0]})
                    self.lat_section_design.optics = opts_idx0
            self.tws_design = twiss(lattice=self.lat_section_design.lat, tws0=self.lat_section_design.tw0_des,
                                    nPoints=self.twiss_nPoints)
            self.update_plot_design()
            magnets_checked = [widget.isChecked() for widget in self.magnet_push_list]
            self._build_magnets_grid()
            magnets_ttf_all = [el for el in self.lat_section_ttf.lat.sequence if
                               ((type(el) == Quadrupole) or (type(el) == Sextupole))]
            seen = {}
            magnets_ttf = [seen.setdefault(x, x) for x in magnets_ttf_all if x not in seen]
            for magnet in magnets_ttf:
                idx_widget = [w.objectName() for w in self.magnet_widget_list].index(magnet.id)
                widget = self.magnet_widget_list[idx_widget]
                if type(magnet) == Quadrupole:
                    widget.setValue(magnet.k1)
                elif type(magnet) == Sextupole:
                    widget.setValue(magnet.k2)
            for widget, checked in zip(self.magnet_push_list, magnets_checked):
                widget.setChecked(checked)
            self.ui.current_optics_file.setText(fname_fullpath.split('/')[-1])
            QMessageBox.information(self, "New optics NOTE", "Are the input Twiss design parameters still valid?",
                                    QMessageBox.Ok)

    def _load_optics_setting(self):
        idx = int(self.ui.optics_file_setting.value())
        opts = {}
        for magnet, params in self.optics.items():
            opts.update({magnet: {}})
            for par, val in params.items():
                opts[magnet].update({par: val[idx]})
        self.lat_section_design.optics = opts
        self.tws_design = twiss(lattice=self.lat_section_design.lat, tws0=self.lat_section_design.tw0_des,
                                nPoints=self.twiss_nPoints)
        self.update_plot_design()
        magnets_checked = [widget.isChecked() for widget in self.magnet_push_list]
        self._build_magnets_grid()
        magnets_ttf_all = [el for el in self.lat_section_ttf.lat.sequence if
                           ((type(el) == Quadrupole) or (type(el) == Sextupole))]
        seen = {}
        magnets_ttf = [seen.setdefault(x, x) for x in magnets_ttf_all if x not in seen]
        for magnet in magnets_ttf:
            idx_widget = [w.objectName() for w in self.magnet_widget_list].index(magnet.id)
            widget = self.magnet_widget_list[idx_widget]
            if type(magnet) == Quadrupole:
                widget.setValue(magnet.k1)
            elif type(magnet) == Sextupole:
                widget.setValue(magnet.k2)
        for widget, checked in zip(self.magnet_push_list, magnets_checked):
            widget.setChecked(checked)

    def _save_optFile(self):
        filename = QFileDialog.getSaveFileName(caption='Save current lattice', directory='./')[0]
        if filename == '':
            return
        optics = self.lat_section_ttf.optics
        if self.ui.save_optFile_format.currentText() == 'json':
            save_opticsfile(filename, optics, mode='json')
        elif self.ui.save_optFile_format.currentText() == 'swesch':
            save_opticsfile(filename, optics, mode='txt')
        return

    def magnets_reset_to_theory(self):
        magnets_ttf_all = [el for el in self.lat_section_ttf.lat.sequence if el.__class__ == Quadrupole or
                           el.__class__ == Sextupole]
        seen = {}
        magnets_ttf = [seen.setdefault(x, x) for x in magnets_ttf_all if x not in seen]
        for i, magnet in enumerate(magnets_ttf):
            idx_design = [el.id for el in self.lat_section_design.lat.sequence].index(magnet.id)
            idx_widget = [el.objectName() for el in self.magnet_widget_list].index(magnet.id)
            if self.magnet_push_list[i].isChecked():
                if magnet.__class__ == Quadrupole:
                    magnet.k1 = self.lat_section_design.lat.sequence[idx_design].k1
                    self.magnet_widget_list[idx_widget].setValue(magnet.k1)
                elif magnet.__class__ == Sextupole:
                    magnet.k2 = self.lat_section_design.lat.sequence[idx_design].k2
                    self.magnet_widget_list[idx_widget].setValue(magnet.k2)
            else:
                pass
            # elif magnet.__class__ == Bend or magnet.__class__ == RBend or magnet.__class__ == SBend:
            #     magnet.angle = self.lat_section_design.lat.sequence[idx_design].angle
            #     self.magnet_widget_list[idx_widget].setValue(magnet.angle)
        self.lat_section_ttf.lat.update_transfer_maps()
        self.tws_real = twiss(lattice=self.lat_section_ttf.lat, tws0=self.tw0_real, nPoints=self.twiss_nPoints)
        self.update_plot_real()

    # PolariX box functions
    def calc_timeRes(self):
        E = self.ui.bunch_energy.value() * 1e6
        VTDS = self.ui.xtds_voltage.value() * 1e6
        en = self.ui.bunch_emmitance.value() * 1e-6
        try:
            tws_TDS = twiss_at(elem_id='TDS-C', lat=self.lat_section_ttf.lat, tws=self.tws_real)
            tws_SCR = twiss_at(elem_id='SCR' + self.ui.xtds_calc_resolution_screen.currentText(),
                               lat=self.lat_section_ttf.lat, tws=self.tws_real)
        except:
            return
        if self.ui.streaking_plane.currentText() == 'X':
            beta_TDS = tws_TDS.beta_x
            beta_SCR = tws_SCR.beta_x
            ph_adv = tws_SCR.mux - tws_TDS.mux
        elif self.ui.streaking_plane.currentText() == 'Y':
            beta_TDS = tws_TDS.beta_y
            beta_SCR = tws_SCR.beta_y
            ph_adv = tws_SCR.muy - tws_TDS.muy
        time_res = np.sqrt(en / ((E / 0.510998946e6) * beta_TDS)) / abs(np.sin(ph_adv)) * E / (
                    VTDS * 251) / c_light * 1e15
        streak = np.sqrt(beta_TDS * beta_SCR) * abs(np.sin(ph_adv)) * VTDS * 251 / E
        self.ui.xtds_timeres.setText("{:.1f} fs".format(time_res))
        self.ui.xtds_streak.setText("{:.2f}".format(streak))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationDisplayName("Optics tool")
    myapp = mainGui(None)
    ssFile = './stylesheet_white.css'
    with open(ssFile, "r") as fh:
        myapp.setStyleSheet(fh.read())
    myapp.move(200, 100)
    myapp.show()
    sys.exit(app.exec_())
