"""Manufacturer-curve receiver model for v5; upstream source is not modified."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

import generate_location_dataset as base
sim = base.sim


class AntennaPattern:
    def __init__(self, path):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text())
        self.angles = np.asarray(self.data["angles_deg_ccw"], float)
        self.frequencies = np.asarray(self.data["pattern_frequencies_mhz"], float)
        self.gain_frequencies = np.asarray(self.data["gain_frequencies_mhz"], float)
        self.peak = np.asarray(self.data["peak_gain_dbi"], float)
        self.horizontal = np.asarray(self.data["horizontal_relative_db"], float)
        self.vertical = np.asarray(self.data["vertical_relative_db"], float)
        if self.horizontal.shape != (len(self.frequencies), len(self.angles)) or self.vertical.shape != self.horizontal.shape:
            raise ValueError("Pattern array shape mismatch")
        if len(self.peak) != len(self.gain_frequencies):
            raise ValueError("Peak-gain array shape mismatch")
        for a in (self.angles, self.frequencies, self.gain_frequencies):
            if not np.all(np.isfinite(a)) or not np.all(np.diff(a) > 0):
                raise ValueError("Invalid antenna coordinate table")
        for a in (self.horizontal, self.vertical, self.peak):
            if not np.all(np.isfinite(a)):
                raise ValueError("Non-finite gain table")
        if np.any(self.horizontal > 0) or np.any(self.vertical > 0):
            raise ValueError("Relative patterns must be non-positive")
        if self.angles[0] != 0 or self.angles[-1] >= 360:
            raise ValueError("Expected one periodic angle cycle starting at zero")
        if np.any(self.horizontal[:, 0] != 0) or np.any(self.vertical[:, 0] != 0):
            raise ValueError("Principal cuts must be normalized at boresight")

    def gain(self, frequencies_hz, azimuth_offset_deg, elevation_offset_deg):
        """Add principal-plane losses in dB; an explicit separable 3D approximation.

        PDF H angles are counter-clockwise in the top-view convention; local
        geographic azimuth offsets are clockwise. V positive angles are up.
        """
        f = np.asarray(frequencies_hz, float) / 1e6
        if not np.all(np.isfinite(f)) or np.any(f < self.frequencies[0]) or np.any(f > self.frequencies[-1]):
            raise ValueError("Signal frequency is outside the digitized pattern range")
        if not np.isfinite(azimuth_offset_deg) or not np.isfinite(elevation_offset_deg):
            raise ValueError("Non-finite arrival angle")
        angles = np.r_[self.angles, 360.]
        h = [np.interp((-azimuth_offset_deg) % 360, angles, np.r_[row, row[0]]) for row in self.horizontal]
        v = [np.interp(elevation_offset_deg % 360, angles, np.r_[row, row[0]]) for row in self.vertical]
        relative = np.interp(f, self.frequencies, h) + np.interp(f, self.frequencies, v)
        # -50 dB is the plotting floor. Do not claim deeper nulls from two cuts.
        return np.interp(f, self.gain_frequencies, self.peak) + np.maximum(relative, -50.)


def rx_arrival_angles(rx, tx, propagation, orientation):
    terrain = propagation["terrain"]
    if terrain is None:
        raise ValueError("v5 expects the same terrain-backed propagation as v3")
    bearing, _, distance = sim.GEOD.inv(rx.lon_deg, rx.lat_deg, tx.lon_deg, tx.lat_deg)
    tx_height = terrain["tx_ground_elevation_m"] + tx.height_agl_m
    rx_height = terrain["rx_ground_elevation_m"] + rx.height_agl_m
    curvature_drop = distance ** 2 / (2 * sim.EARTH_RADIUS_M)
    elevation = np.degrees(np.arctan2(tx_height-rx_height-curvature_drop, distance))
    return {"bearing_rx_to_tx_deg": bearing % 360.,
            "off_axis_azimuth_deg": (bearing-orientation["azimuth_deg"]+180) % 360-180,
            "geometric_arrival_elevation_deg": float(elevation),
            "off_axis_elevation_deg": float(elevation-orientation["elevation_deg"])}


# The function below is copied from upstream make_receiver_spectrogram_v3
# (v3.2, SHA256 0479c66bda13f90c0eb65edf76deccb0b72a81cdb3cd06ef1ec2fa68a2c39a9a).
# Only the two receive-gain additions change: gain can vary by frequency.
# All original random draws, noise, propagation input and detector interface stay the same.

def make_receiver_spectrogram_v3(rx: sim.Receiver, tx: sim.Transmitter, freq_hz: np.ndarray, state_models: dict, state_schedule: np.ndarray, template_schedule: np.ndarray, propagation: dict, nt: int, rng: np.random.Generator, gain_function=None):
    """Time-varying LTE load + within-state LTE template variation + propagation."""
    nf = len(freq_hz)
    geom = propagation['geometry']
    noise_floor_dbm = sim.thermal_noise_floor_dbm(sim.RBW_HZ, rx.noise_figure_db)
    x = np.linspace(0.0, 8.0 * np.pi, nf)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    bandpass_db = (sim.BANDPASS_RIPPLE_DB * np.sin(x + phase) + 0.25 * sim.BANDPASS_RIPPLE_DB * np.sin(0.27 * x + 0.4 * phase)).astype(np.float32)
    gain_drift = sim.gaussian_filter1d(rng.normal(size=nt), sigma=25.0)
    gain_drift -= np.mean(gain_drift)
    if np.std(gain_drift) > 0:
        gain_drift *= sim.GAIN_DRIFT_DB / np.std(gain_drift)
    gain_drift = gain_drift.astype(np.float32)
    spec_dbm = np.empty((nt, nf), dtype=np.float32)
    chunk = 50
    for i0 in range(0, nt, chunk):
        i1 = min(nt, i0 + chunk)
        jitter = rng.normal(0.0, sim.NOISE_JITTER_DB, size=(i1 - i0, nf)).astype(np.float32)
        spec_dbm[i0:i1] = noise_floor_dbm + bandpass_db[None, :] + gain_drift[i0:i1, None] + jitter
    shadowing_db = sim.make_slow_shadowing(nt, sim.SHADOWING_SIGMA_DB, sim.SHADOWING_CORRELATION_S, sim.TIME_RESOLUTION_S, rng)
    template_signal_mw = {}
    template_component_center_dbm = {}
    component_names = []
    for state_name, model in state_models.items():
        for template in model['templates']:
            ti = int(template['template_index'])
            key = (state_name, ti)
            total_signal_mw = np.zeros(nf, dtype=np.float64)
            center_by_component = {}
            for comp in template['components']:
                if comp.name not in component_names:
                    component_names.append(comp.name)
                frac = template['fractions'][comp.name]
                active = frac > 1e-16
                if not np.any(active):
                    continue
                idx = np.flatnonzero(active)
                frac_active = frac[idx]
                path_loss_db = propagation['path_loss_db'][idx]
                if np.any(~np.isfinite(path_loss_db)):
                    raise RuntimeError(f'Missing path loss values for {rx.name}/{state_name}/template{ti}/{comp.name}')
                tx_gain_db = sim.tx_antenna_gain_db(tx, geom['bearing_tx_to_rx_deg'], geom['elevation_tx_to_rx_deg'], harmonic_order=comp.harmonic_order, extra_loss_db=comp.antenna_extra_loss_db)
                tx_bin_dbm = comp.total_power_dbm + 10.0 * np.log10(np.maximum(frac_active, 1e-30))
                rx_bin_dbm = tx_bin_dbm + tx_gain_db + (rx.antenna_gain_dbi if gain_function is None else gain_function(freq_hz[idx])) - path_loss_db - rx.excess_path_loss_db
                total_signal_mw[idx] += sim.dbm_to_mw(rx_bin_dbm)
                center_key = str(int(round(comp.center_hz)))
                if center_key in propagation['center_losses_db']:
                    center_loss_db = propagation['center_losses_db'][center_key]
                else:
                    finite = np.isfinite(propagation['path_loss_db'])
                    center_loss_db = float(np.interp(comp.center_hz, freq_hz[finite], propagation['path_loss_db'][finite]))
                center_by_component[comp.name] = {'center_hz': comp.center_hz, 'kind': comp.kind, 'tx_total_power_dbm': comp.total_power_dbm, 'tx_antenna_gain_db': tx_gain_db, 'path_loss_center_db': center_loss_db, 'rx_center_no_shadow_dbm': comp.total_power_dbm + tx_gain_db + (rx.antenna_gain_dbi if gain_function is None else gain_function(comp.center_hz)) - center_loss_db - rx.excess_path_loss_db}
            template_signal_mw[key] = total_signal_mw
            template_component_center_dbm[key] = center_by_component
    for state_name, model in state_models.items():
        for template in model['templates']:
            ti = int(template['template_index'])
            tidx = np.flatnonzero((state_schedule == state_name) & (template_schedule == ti))
            if len(tidx) == 0:
                continue
            scale = 10.0 ** (shadowing_db[tidx].astype(np.float64) / 10.0)
            signal_mw = scale[:, None] * template_signal_mw[state_name, ti][None, :]
            spec_dbm[tidx] = sim.mw_to_dbm(sim.dbm_to_mw(spec_dbm[tidx]) + signal_mw).astype(np.float32)
    component_truth = []
    for comp_name in component_names:
        rx_t = np.full(nt, np.nan, dtype=np.float32)
        tx_t = np.full(nt, np.nan, dtype=np.float32)
        truth_template = None
        for state_name, model in state_models.items():
            for template in model['templates']:
                ti = int(template['template_index'])
                tidx = np.flatnonzero((state_schedule == state_name) & (template_schedule == ti))
                c = template_component_center_dbm[state_name, ti].get(comp_name)
                if c is None or len(tidx) == 0:
                    continue
                truth_template = c
                rx_t[tidx] = c['rx_center_no_shadow_dbm'] + shadowing_db[tidx]
                tx_t[tidx] = c['tx_total_power_dbm']
        if truth_template is None:
            continue
        component_truth.append({'component': comp_name, 'kind': truth_template['kind'], 'center_hz': truth_template['center_hz'], 'tx_antenna_gain_db': truth_template['tx_antenna_gain_db'], 'path_loss_center_db': truth_template['path_loss_center_db'], 'median_tx_total_power_dbm': float(np.nanmedian(tx_t)), 'median_rx_total_component_dbm': float(np.nanmedian(rx_t)), 'tx_total_power_dbm_t': tx_t, 'rx_total_component_dbm_t': rx_t})
    truth = {**geom, 'noise_floor_per_rbw_dbm': noise_floor_dbm, 'shadowing_db_t': shadowing_db, 'component_truth': component_truth, 'propagation_model': propagation['mode'], 'center_losses_db': propagation['center_losses_db']}
    if propagation['terrain'] is not None:
        truth.update({'tx_ground_elevation_m': propagation['terrain']['tx_ground_elevation_m'], 'rx_ground_elevation_m': propagation['terrain']['rx_ground_elevation_m'], 'midpoint_ground_elevation_m': propagation['terrain']['midpoint_ground_elevation_m']})
    return (spec_dbm, truth)
