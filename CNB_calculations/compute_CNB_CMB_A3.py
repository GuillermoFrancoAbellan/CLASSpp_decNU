"""
compute_CNB_CMB_A3.py

Scenario A3: single decay channel, solar mass gap, inverted ordering.

Set COMPUTE_SPECTRA = True to run CLASS and save results.
Set COMPUTE_SPECTRA = False to skip CLASS and load previously saved files.
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter
import healpy as hp

from classy import Class
from cnb_utils import compute_power_spectrum_new, generate_poisson_fluctuations, \
    l_max, n_kmodes, k_values

# ─── computation flag ──────────────────────────────────────────────────────────
COMPUTE_SPECTRA = False   # set False to load saved files and replot
# ───────────────────────────────────────────────────────────────────────────────

os.makedirs('./Cl_avg',      exist_ok=True)
os.makedirs('./Cl_CMB',      exist_ok=True)
os.makedirs('./plots_residuals', exist_ok=True)

os.environ['PATH'] = "/home/abellan/texlive/2025/bin/x86_64-linux" + os.environ['PATH']


# ─── scenario parameters ───────────────────────────────────────────────────────
m_nuL          = 0.05
m_lightest     = 0.0001
idx_n_stable   = 1    # index_ncdm_to_print for stable case
idx_n_decay    = 2    # index_ncdm_to_print for decay case (daughter ν1)

Gamma_list  = [979.49, 293.85, 97.95]   # [km/s/Mpc]
max_q_list  = [15,     15,     15   ]
n_bins_list = [60,     30,     20   ]   # from plot_PSD_all_scenarios.py
gtag_list   = ['g1000', 'g300', 'g100']

# Poisson fluctuation settings
N_total, n_trials, nside = int(1e5), 1000, 128

k_output_str = ", ".join(f"{k:.2e}" for k in k_values)

# ─── common CLASS parameters ───────────────────────────────────────────────────
COMMON = {
    'output'             : 'tCl,pCl,lCl,mPk',
    'lensing'            : 'yes',
    'l_max_scalars'      : 2500,
    'background_method'  : 0,
    'H0'                 : 67.37,
    'omega_b'            : 0.02233,
    'omega_cdm'          : 0.1198,
    'tau_reio'           : 0.0540,
    'n_s'                : 0.9652,
    'A_s'                : 2.0968e-09,
    'N_ur'               : 0.00641,
    'P_k_max_1/Mpc'      : 1.0,
    'k_output_values'    : k_output_str,
    'threads'            : 16,
    'ncdm_fluid_approximation': 3,
    'l_max_ncdm'         : 17,
}

def gamma_tag(Gamma):
    gtag = gtag_list[Gamma_list.index(Gamma)]
    return f'A3_{gtag}_mL0p05'

def extract_delta_new(cosmo, idx):
    perturbs = cosmo.get_perturbations()['scalar']
    delta_new = np.zeros((n_kmodes, l_max + 1))
    for k_idx in range(n_kmodes):
        for l in range(l_max + 1):
            delta_new[k_idx, l] = perturbs[k_idx][f'Theta_new_n_l[{idx}][{l}]'][-1]
    return delta_new

def run_stable():
    """Stable A3: all 3 neutrinos as standard NCDM, IO."""
    p = dict(COMMON)
    p['index_ncdm_to_print'] = idx_n_stable
    p.update({
        'neutrino_hierarchy'               : 'inverted',
        'N_ncdm_standard'                  : 3,
        'deg_ncdm_standard'                : '1.0, 1.0, 1.0',
        'm_ncdm_lightest'                  : m_lightest,
        'quadrature_strategy_ncdm_standard': '3, 3, 3',
        'maximum_q_ncdm_standard'          : '15, 15, 15',
        'N_momentum_bins_ncdm_standard'    : '15, 15, 15',
    })
    cosmo = Class(); cosmo.set(p); cosmo.compute()
    Cl_cnb = compute_power_spectrum_new(extract_delta_new(cosmo, idx_n_stable))
    Cl_TT  = cosmo.lensed_cl(2500)['tt']
    cosmo.struct_cleanup(); cosmo.empty()
    return Cl_cnb, Cl_TT

def run_decay(Gamma, max_q, n_bins):
    """Decaying A3: ν3 standard (free-streaming), ν2 → ν1 + ϕ (solar gap, IO)."""
    p = dict(COMMON)
    p['index_ncdm_to_print'] = idx_n_decay
    p.update({
        'N_ncdm_decay_dr'                   : 2,
        'has_ncdm_decay_dr_ncdm'            : 'yes',
        'N_ncdm_standard'                   : 1,
        'log10Gamma_nu'                     : np.log10(Gamma),
        'is_ncdm_decay_degenerate'          : 'no',
        'neutrino_hierarchy'                : 'inverted',
        'decay_mass_gap'                    : 'solar',
        'deg_ncdm_decay_dr'                 : '1.0, 1.0',
        'm_ncdm_lightest'                   : m_lightest,
        'l_max_dr'                          : '17, 17',
        'quadrature_strategy_ncdm_standard' : 3,
        'maximum_q_ncdm_standard'           : 15,
        'N_momentum_bins_ncdm_standard'     : 15,
        'quadrature_strategy_ncdm_decay_dr' : '3, 3',
        'maximum_q_ncdm_decay_dr'           : f'{max_q}, {max_q}',
        'N_momentum_bins_ncdm_decay_dr'     : f'{n_bins}, {n_bins}',
        'adjust_q_binning'                  : 'no',
    })
    cosmo = Class(); cosmo.set(p); cosmo.compute()
    Cl_cnb = compute_power_spectrum_new(extract_delta_new(cosmo, idx_n_decay))
    Cl_TT  = cosmo.lensed_cl(2500)['tt']
    cosmo.struct_cleanup(); cosmo.empty()
    return Cl_cnb, Cl_TT

# ─── run CLASS or load ─────────────────────────────────────────────────────────
if COMPUTE_SPECTRA:
    print("Running CLASS for stable A3...")
    Cl_stable, Cl_TT_stable = run_stable()
    np.save('./Cl_avg/Cl_avg_A3_stable_mL0p05.npy', Cl_stable)
    np.save('./Cl_CMB/Cl_TT_A3_stable_mL0p05.npy',  Cl_TT_stable)

    for Gamma, max_q, n_bins in zip(Gamma_list, max_q_list, n_bins_list):
        print(f"Running CLASS for Gamma = {Gamma} km/s/Mpc "
              f"(max_q = {max_q}, N_bins = {n_bins})...")
        Cl_dec, Cl_TT_dec = run_decay(Gamma, max_q, n_bins)
        tag = gamma_tag(Gamma)
        np.save(f'./Cl_avg/Cl_avg_{tag}.npy', Cl_dec)
        np.save(f'./Cl_CMB/Cl_TT_{tag}.npy',  Cl_TT_dec)

# ─── load ──────────────────────────────────────────────────────────────────────
Cl_stable    = np.load('./Cl_avg/Cl_avg_A3_stable_mL0p05.npy')
Cl_TT_stable = np.load('./Cl_CMB/Cl_TT_A3_stable_mL0p05.npy')

print("Generating Poisson fluctuations...")
delta_map = hp.synfast(Cl_stable, nside=nside)
_, _, cls_fluc, _ = generate_poisson_fluctuations(Cl_stable, delta_map, n_trials, N_total)
cl_std = np.std(cls_fluc, axis=0)

Cl_dec_list    = [np.load(f'./Cl_avg/Cl_avg_{gamma_tag(G)}.npy') for G in Gamma_list]
Cl_TT_dec_list = [np.load(f'./Cl_CMB/Cl_TT_{gamma_tag(G)}.npy') for G in Gamma_list]

# ─── plot settings ─────────────────────────────────────────────────────────────
mpl.rcParams.update({
    'text.usetex': True, 'font.family': 'serif', 'font.serif': ['cmr10'],
    'axes.labelsize': 25, 'axes.linewidth': 1.5,
    'xtick.labelsize': 17, 'ytick.labelsize': 17,
    'legend.fontsize': 17, 'font.size': 17,
    'legend.frameon': False, 'axes.formatter.use_mathtext': True,
})

colors     = ['royalblue', 'crimson', 'forestgreen']
tau_labels = [r'$\tau_\nu = 1.0\ \mathrm{Gyr}$',
              r'$\tau_\nu = 3.33\ \mathrm{Gyr}$',
              r'$\tau_\nu = 10.0\ \mathrm{Gyr}$']

lTT, DlTT_mean, _, DlTT_err_plus, _ = np.loadtxt(
    '../error_Planck/Planck2018_errorTT.txt', unpack=True)
ell_cnb = np.arange(1, l_max + 1)
ell_cmb = np.arange(len(Cl_TT_stable))

# CMB residual plot
fig, ax = plt.subplots(figsize=(8, 6), dpi=400)
for Cl_TT_dec, col, lab in zip(Cl_TT_dec_list, colors, tau_labels):
    res = (Cl_TT_dec - Cl_TT_stable) / Cl_TT_stable
    ax.plot(ell_cmb[38:], res[38:], color=col, linewidth=2.0, label=lab)
ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
ax.fill_between(lTT, -DlTT_err_plus/DlTT_mean, DlTT_err_plus/DlTT_mean, color='lightgray')
ax.set_xlim(38, 2500)
ax.set_ylim(-0.009, 0.009)
ax.set_xlabel(r'$\ell$',fontsize=23)
ax.set_ylabel(r'$\Delta C_\ell^{\rm TT}/C_{\ell,{\rm ref}}^{\rm TT}$',fontsize=23)
ax.set_title(r'{\bf CMB} ($m_{\nu l}=%.2f\,{\rm eV}$, solar gap, IO)' % m_nuL, pad=10, fontsize=23)
[s.set_linewidth(2.0) for s in ax.spines.values()]
leg1 = ax.legend(frameon=False, loc='upper right')
ax.add_artist(leg1)
leg2 = ax.legend([Line2D([0],[0], color='lightgray', linewidth=4)],
                 [r'\textsl{Planck} errors'], loc='lower right',
                 frameon=True, facecolor='white', edgecolor='black', framealpha=1.0)
leg2.get_frame().set_linewidth(0.5)
fig.tight_layout()
fig.savefig('plots_residuals/CMB_residuals_A3.pdf')
plt.show()

# CνB residual plot
fig, ax = plt.subplots(figsize=(8, 6), dpi=400)
for Cl_dec, col, lab in zip(Cl_dec_list, colors, tau_labels):
    res = savgol_filter((Cl_dec[1:] - Cl_stable[1:]) / Cl_stable[1:], 11, 3)
    ax.plot(ell_cnb, res, color=col, linewidth=2.0, label=lab)
ax.fill_between(ell_cnb, -np.sqrt(2./(2*ell_cnb+1.)), np.sqrt(2./(2*ell_cnb+1.)),
                color='lightgray')
ax.fill_between(ell_cnb, -cl_std/Cl_stable[1:], cl_std/Cl_stable[1:],
                alpha=0.7, color='gray')
ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
ax.set_xlim(1, 17) 
ax.set_ylim(-0.61, 0.61)
ax.set_xlabel(r'$\ell$',fontsize=23)
ax.set_ylabel(r'$\Delta C_\ell / C_{\ell,{\rm ref}}$',fontsize=23)
ax.set_title(r'{\bf C$\nu$B} ($m_{\nu l}=%.2f\,{\rm eV}$, solar gap, IO)' % m_nuL, pad=10, fontsize=23)
[s.set_linewidth(2.0) for s in ax.spines.values()]
leg1 = ax.legend(frameon=False, loc=(0.01, 0.55)); ax.add_artist(leg1)
leg2 = ax.legend(
    [Line2D([0],[0], color='lightgray', linewidth=4),
     Line2D([0],[0], color='gray',       linewidth=4)],
    [r'Cosmic Variance', r'Counting statistics, $N=10^5$'],
    loc='upper right', frameon=True, facecolor='white', edgecolor='black', framealpha=1.0)
leg2.get_frame().set_linewidth(0.5)
fig.tight_layout()
fig.savefig('plots_residuals/CNB_residuals_A3.pdf')
plt.show()
