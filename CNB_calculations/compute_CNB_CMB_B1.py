"""
compute_CNB_CMB_B1_classy.py

Scenario B1: two decay channels, atmospheric mass gap, normal ordering.

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

# Output directories
os.makedirs('./Cl_avg',     exist_ok=True)
os.makedirs('./Cl_CMB',     exist_ok=True)
os.makedirs('./plots_residuals', exist_ok=True)

os.environ['PATH'] = "/home/abellan/texlive/2025/bin/x86_64-linux" + os.environ['PATH']

# ─── scenario parameters ───────────────────────────────────────────────────────
m_nuL      = 0.03                         # lightest neutrino mass [eV]
idx_n      = 1                            # index_ncdm_to_print (daughter species)

# Gamma values and corresponding maximum_q from plot_CNB_CMB_spectra_B1.ipynb
Gamma_list = [979.49, 489.75, 244.87]     # [km/s/Mpc]
max_q_list = [45,     75,     130]        # maximum_q_ncdm_decay_dr per Gamma
gtag_list  = ['g1000', 'g500', 'g250']   # file tags matching Cl_avg_OLD naming

# Poisson fluctuation settings
N_total  = int(1e5)
n_trials = 1000
nside    = 128

# k-output string (same values as in LCDMnu_dec_toNCDM.ini)
k_output_str = ", ".join(f"{k:.2e}" for k in k_values)

# ─── common CLASS parameters (from LCDMnu_dec_toNCDM.ini) ─────────────────────
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
    'index_ncdm_to_print': idx_n,
    'threads'            : 16,
    'ncdm_fluid_approximation': 3,
    'l_max_ncdm'         : 17,
}

# ─── helpers ───────────────────────────────────────────────────────────────────
def gamma_tag(Gamma):
    gtag = gtag_list[Gamma_list.index(Gamma)]
    return f'B1_{gtag}_mL0p03'

def extract_delta_new(cosmo):
    """Return delta_new[k_idx, l] = Θ̃_ℓ(k, τ₀) from classy perturbations."""
    perturbs = cosmo.get_perturbations()['scalar']
    delta_new = np.zeros((n_kmodes, l_max + 1))
    for k_idx in range(n_kmodes):
        for l in range(l_max + 1):
            key = f'Theta_new_n_l[{idx_n}][{l}]'
            delta_new[k_idx, l] = perturbs[k_idx][key][-1]
    return delta_new

def run_stable():
    """Run CLASS for stable B1 scenario; return (Cl_cnb, Cl_TT)."""
    p = dict(COMMON)
    p.update({
        'neutrino_hierarchy'               : 'normal',
        'N_ncdm_standard'                  : 2,
        'deg_ncdm_standard'                : '1.0, 2.0',
        'm_ncdm_lightest'                  : m_nuL,
        'quadrature_strategy_ncdm_standard': '3, 3',
        'maximum_q_ncdm_standard'          : '15, 15',
        'N_momentum_bins_ncdm_standard'    : '15, 15',
    })
    cosmo = Class()
    cosmo.set(p)
    cosmo.compute()
    delta_new = extract_delta_new(cosmo)
    Cl_cnb = compute_power_spectrum_new(delta_new)
    Cl_TT  = cosmo.lensed_cl(2500)['tt']
    cosmo.struct_cleanup()
    cosmo.empty()
    return Cl_cnb, Cl_TT

def run_decay(Gamma, max_q, adjust_q='no'):
    """Run CLASS for decaying B1 scenario; return (Cl_cnb, Cl_TT)."""
    p = dict(COMMON)
    p.update({
        'N_ncdm_decay_dr'                   : 2,
        'has_ncdm_decay_dr_ncdm'            : 'yes',
        'log10Gamma_nu'                     : np.log10(Gamma),
        'is_ncdm_decay_degenerate'          : 'yes',
        'neutrino_hierarchy'                : 'normal',
        'decay_mass_gap'                    : 'atmospheric',
        'deg_ncdm_decay_dr'                 : '1.0, 2.0',
        'm_ncdm_lightest'                   : m_nuL,
        'l_max_dr'                          : '17, 17',
        'quadrature_strategy_ncdm_decay_dr' : '3, 3',
        'maximum_q_ncdm_decay_dr'           : f'{max_q}, {max_q}',
        'N_momentum_bins_ncdm_decay_dr'     : f'{max_q}, {max_q}',
        'adjust_q_binning'                  : adjust_q,
    })
    cosmo = Class()
    cosmo.set(p)
    cosmo.compute()
    delta_new = extract_delta_new(cosmo)
    Cl_cnb = compute_power_spectrum_new(delta_new)
    Cl_TT  = cosmo.lensed_cl(2500)['tt']
    cosmo.struct_cleanup()
    cosmo.empty()
    return Cl_cnb, Cl_TT

# ─── run CLASS or load saved results ──────────────────────────────────────────
if COMPUTE_SPECTRA:
    print("Running CLASS for stable scenario...")
    Cl_stable, Cl_TT_stable = run_stable()
    np.save('./Cl_avg/Cl_avg_B1_stable_mL0p03.npy', Cl_stable)
    np.save('./Cl_CMB/Cl_TT_B1_stable_mL0p03.npy',  Cl_TT_stable)

    for Gamma, max_q in zip(Gamma_list, max_q_list):
        # Last Gamma value uses adjust_q_binning to let CLASS find optimal q-grid
        adjust_q = 'yes' if Gamma == Gamma_list[-1] else 'no'
        print(f"Running CLASS for Gamma = {Gamma} km/s/Mpc "
              f"(max_q = {max_q}, adjust_q_binning = {adjust_q})...")
        Cl_dec, Cl_TT_dec = run_decay(Gamma, max_q, adjust_q=adjust_q)
        tag = gamma_tag(Gamma)
        np.save(f'./Cl_avg/Cl_avg_{tag}.npy', Cl_dec)
        np.save(f'./Cl_CMB/Cl_TT_{tag}.npy',  Cl_TT_dec)

# ─── load all spectra ──────────────────────────────────────────────────────────
Cl_stable    = np.load('./Cl_avg/Cl_avg_B1_stable_mL0p03.npy')
Cl_TT_stable = np.load('./Cl_CMB/Cl_TT_B1_stable_mL0p03.npy')

# Poisson fluctuations are fast — always recompute from the stable spectrum
print("Generating Poisson fluctuations...")
delta_map = hp.synfast(Cl_stable, nside=nside)
_, _, cls_fluc, _ = generate_poisson_fluctuations(Cl_stable, delta_map, n_trials, N_total)
cl_std = np.std(cls_fluc, axis=0)

Cl_dec_list    = [np.load(f'./Cl_avg/Cl_avg_{gamma_tag(G)}.npy') for G in Gamma_list]
Cl_TT_dec_list = [np.load(f'./Cl_CMB/Cl_TT_{gamma_tag(G)}.npy') for G in Gamma_list]

# ─── plot settings ─────────────────────────────────────────────────────────────
mpl.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['cmr10'],
    'axes.labelsize': 25,
    'axes.linewidth': 1.5,
    'xtick.labelsize': 17,
    'ytick.labelsize': 17,
    'legend.fontsize': 17,
    'font.size': 17,
    'legend.frameon': False,
    'axes.formatter.use_mathtext': True,
})

colors     = ['royalblue', 'crimson', 'forestgreen']
tau_labels = [r'$\tau_\nu = 0.5\ \mathrm{Gyr}$',
              r'$\tau_\nu = 1.0\ \mathrm{Gyr}$',
              r'$\tau_\nu = 2.0\ \mathrm{Gyr}$']

lTT, DlTT_mean, _, DlTT_err_plus, _ = np.loadtxt(
    '../error_Planck/Planck2018_errorTT.txt', unpack=True)

ell_cnb = np.arange(1, l_max + 1)
ell_cmb = np.arange(len(Cl_TT_stable))

# ─── CMB residual plot ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6), dpi=400)
for Cl_TT_dec, col, lab in zip(Cl_TT_dec_list, colors, tau_labels):
    res = (Cl_TT_dec - Cl_TT_stable) / Cl_TT_stable
    ax.plot(ell_cmb[38:], res[38:], color=col, linewidth=2.0, label=lab)
ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
ax.fill_between(lTT, -DlTT_err_plus/DlTT_mean, DlTT_err_plus/DlTT_mean, color='lightgray')
ax.set_xlim(38, 2500)
ax.set_ylim(-0.015, 0.021)
ax.set_xlabel(r'$\ell$', fontsize=23)
ax.set_ylabel(r'$\Delta C_\ell^{\rm TT}/C_{\ell,{\rm ref}}^{\rm TT}$', fontsize=23)
ax.set_title(r'{\bf CMB} ($m_{\nu l}=%.2f\,{\rm eV}$, atmos.\ gap, NO)' % m_nuL, pad=10, fontsize=23)
[s.set_linewidth(2.0) for s in ax.spines.values()]
leg1 = ax.legend(frameon=False, loc='upper center')
ax.add_artist(leg1)
leg2 = ax.legend([Line2D([0], [0], color='lightgray', linewidth=4)],
                 [r'\textsl{Planck} errors'],
                 loc=(0.36, 0.05), frameon=True,
                 facecolor='white', edgecolor='black', framealpha=1.0)
leg2.get_frame().set_linewidth(0.5)
fig.tight_layout()
fig.savefig('plots_residuals/CMB_residuals_B1.pdf')
plt.show()

# ─── CνB residual plot ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6), dpi=400)
for Cl_dec, col, lab in zip(Cl_dec_list, colors, tau_labels):
    res = savgol_filter(
        (Cl_dec[1:] - Cl_stable[1:]) / Cl_stable[1:], 11, 3)
    ax.plot(ell_cnb, res, color=col, linewidth=2.0, label=lab)
ax.fill_between(ell_cnb,
                -np.sqrt(2. / (2*ell_cnb + 1.)),
                 np.sqrt(2. / (2*ell_cnb + 1.)),
                color='lightgray')
ax.fill_between(ell_cnb,
                -cl_std / Cl_stable[1:],
                 cl_std / Cl_stable[1:],
                alpha=0.7, color='gray')
ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
ax.set_xlim(1, 17)
ax.set_ylim(-0.71, 0.71)
ax.set_xlabel(r'$\ell$', fontsize=23)
ax.set_ylabel(r'$\Delta C_\ell / C_{\ell,{\rm ref}}$', fontsize=23)
ax.set_title(r'{\bf C$\nu$B} ($m_{\nu l}=%.2f\,{\rm eV}$, atmos.\ gap, NO)' % m_nuL, pad=10, fontsize=23)
[s.set_linewidth(2.0) for s in ax.spines.values()]
leg1 = ax.legend(frameon=False, loc=(0.01, 0.55))
ax.add_artist(leg1)
leg2 = ax.legend(
    [Line2D([0], [0], color='lightgray', linewidth=4),
     Line2D([0], [0], color='gray',       linewidth=4)],
    [r'Cosmic Variance', r'Counting statistics, $N=10^5$'],
    loc='upper right', frameon=True,
    facecolor='white', edgecolor='black', framealpha=1.0)
leg2.get_frame().set_linewidth(0.5)
fig.tight_layout()
fig.savefig('plots_residuals/CNB_residuals_B1.pdf')
plt.show()
