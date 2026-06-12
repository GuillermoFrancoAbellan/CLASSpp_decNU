"""
compute_PSD_all_scenarios.py

Computes and plots the present-day phase-space distribution q³·f(q,τ₀)
for scenarios A2, A3, B1 and B2, using the classy Python wrapper.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import os
from classy import Class
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import matplotlib.ticker as ticker

# ─── computation flag ──────────────────────────────────────────────────────────
COMPUTE_BACKGROUND = False   # set False to load saved files and replot
# ───────────────────────────────────────────────────────────────────────────────

os.makedirs('./Cl_CMB', exist_ok=True)

# ─── constants ─────────────────────────────────────────────────────────────────
pref = ((2 * np.pi)**3) / 2          # phase-space volume prefactor

# 1 km/s/Mpc → Gyr^{-1}: Γ [km/s/Mpc] / KM_S_MPC_TO_GYR = Γ [Gyr^{-1}]
KM_S_MPC_TO_GYR = 3.0857e19 / 3.156e16  # ≈ 977.7

def gamma_to_tau_gyr(Gamma, tau_factor=1.0):
    """Convert Γ [km/s/Mpc] to τ [Gyr].  tau_factor=0.5 for B1 (two channels)."""
    return tau_factor * KM_S_MPC_TO_GYR / Gamma

# ─── matplotlib style (matching plot_PSD_all_scenarios.ipynb) ─────────────────
mpl.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['cmr10'],
    'axes.labelsize': 24,
    'axes.linewidth': 1.5,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 20,
    'font.size': 18,
    'legend.frameon': False,
    'axes.formatter.use_mathtext': True,
})

Nq_interp = 500   # number of points for interpolated q-grid

# ─── common CLASS background parameters ───────────────────────────────────────
COMMON_BG = {
    'output'            : '',       # background only — no CMB/perturbations
    'background_method' : 0,
    'H0'                : 67.37,
    'omega_b'           : 0.02233,
    'omega_cdm'         : 0.1198,
    'N_ur'              : 0.00641,
    'ncdm_fluid_approximation': 3,
    'threads'           : 16,
}

# ─── helper: run CLASS background and extract daughter PSD ────────────────────
def get_psd(params, idx_daughter):
    """
    Run CLASS background; return (q_vals, lnf) for the daughter species.
    Q-bins are read from CLASS via get_ncdm()  [ncdm_->q_ncdm_[idx][q_id]].
    log(f) comes from get_background()         [lnf_dncdm[idx][q_id], last step].
    To recover the PSD: f = pref * exp(lnf)  where pref = ((2π)³)/2.
    """
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()

    # q-bins (ncdm_->q_ncdm_[idx_daughter][q_id])
    ncdm_data = cosmo.get_ncdm()
    q_size = int(ncdm_data[f'q_size[{idx_daughter}]'])
    q_vals = np.array([ncdm_data[f'q[{idx_daughter}][{q_id}]'] for q_id in range(q_size)])

    # log(f) at today (last time step)
    bg = cosmo.get_background()
    lnf = np.array([bg[f'lnf_dncdm[{idx_daughter}][{q_id}]'][-1] for q_id in range(q_size)])

    cosmo.struct_cleanup()
    cosmo.empty()
    return q_vals, lnf

# ─── scenario definitions ──────────────────────────────────────────────────────
# idx_daughter: global NCDM species index for the daughter neutrino
#   A2/A3: N_ncdm_standard=1 (n=0) + parent (n=1) + daughter (n=2)  → idx=2
#   B1/B2: parent (n=0) + daughter (n=1)                              → idx=1

SCENARIOS = [
    {
        'name'        : 'A2',
        'title'       : r'$m_{\nu l} = 0.03$ eV, solar gap, NO',
        'm_nuL'       : 0.03,
        'mL_str'      : 'mL0p03',
        'Gamma_list'  : [979.49, 293.85, 97.95],
        'gtag_list'   : ['g1000', 'g300', 'g100'],
        'tau_labels'  : ['1.0', '3.33', '10.0'],
        'max_q_list'  : [15,    15,     15   ],   # max_q per Gamma (fixed at 15 for solar gap)
        'n_bins_list' : [35,    20,     20   ],   # N_bins per Gamma (from plot_PSD_all_scenarios.ipynb)
        'idx_daughter': 2,
        'q_max_fd'    : 15,
        'xlim'        : (0, 15),
        'ylim'        : (0.0, 3.1),
        'base_params' : {
            **COMMON_BG,
            'N_ncdm_decay_dr'                   : 2,
            'has_ncdm_decay_dr_ncdm'            : 'yes',
            'N_ncdm_standard'                   : 1,
            'is_ncdm_decay_degenerate'          : 'no',
            'neutrino_hierarchy'                : 'normal',
            'decay_mass_gap'                    : 'solar',
            'deg_ncdm_decay_dr'                 : '1.0, 1.0',
            'quadrature_strategy_ncdm_standard' : 3,
            'maximum_q_ncdm_standard'           : 15,
            'N_momentum_bins_ncdm_standard'     : 15,
            'quadrature_strategy_ncdm_decay_dr' : '3, 3',
            'adjust_q_binning'                  : 'no',
        },
    },
    {
        'name'        : 'A3',
        'title'       : r'$m_{\nu l} = 0.05$ eV, solar gap, IO',
        'm_nuL'       : 0.0001,
        'mL_str'      : 'mL0p05',
        'Gamma_list'  : [979.49, 293.85, 97.95],
        'gtag_list'   : ['g1000', 'g300', 'g100'],
        'tau_labels'  : ['1.0', '3.33', '10.0'],
        'max_q_list'  : [15,    15,     15   ],
        'n_bins_list' : [60,    30,     20   ],
        'idx_daughter': 2,
        'q_max_fd'    : 15,
        'xlim'        : (0, 15),
        'ylim'        : (0.0, 3.1),
        'base_params' : {
            **COMMON_BG,
            'N_ncdm_decay_dr'                   : 2,
            'has_ncdm_decay_dr_ncdm'            : 'yes',
            'N_ncdm_standard'                   : 1,
            'is_ncdm_decay_degenerate'          : 'no',
            'neutrino_hierarchy'                : 'inverted',
            'decay_mass_gap'                    : 'solar',
            'deg_ncdm_decay_dr'                 : '1.0, 1.0',
            'quadrature_strategy_ncdm_standard' : 3,
            'maximum_q_ncdm_standard'           : 15,
            'N_momentum_bins_ncdm_standard'     : 15,
            'quadrature_strategy_ncdm_decay_dr' : '3, 3',
            'adjust_q_binning'                  : 'no',
        },
    },
    {
        'name'        : 'B1',
        'title'       : r'$m_{\nu l} = 0.03$ eV, atmos.\ gap, NO',
        'm_nuL'       : 0.03,
        'mL_str'      : 'mL0p03',
        'Gamma_list'  : [979.49, 489.75, 244.87],
        'gtag_list'   : ['g1000', 'g500', 'g250'],
        'tau_labels'  : ['0.5',  '1.0',  '2.0' ],
        'max_q_list'  : [45,     75,     130   ],  # q_max per Gamma (from plot_PSD_all_scenarios.ipynb)
        'n_bins_list' : [45,     75,     130   ],  # N_bins = max_q for B scenarios
        'idx_daughter': 1,
        'q_max_fd'    : 15,
        'xlim'        : (0, 120),
        'ylim'        : (0.05, 2.3),
        'base_params' : {
            **COMMON_BG,
            'N_ncdm_decay_dr'                   : 2,
            'has_ncdm_decay_dr_ncdm'            : 'yes',
            'is_ncdm_decay_degenerate'          : 'yes',
            'neutrino_hierarchy'                : 'normal',
            'decay_mass_gap'                    : 'atmospheric',
            'deg_ncdm_decay_dr'                 : '1.0, 2.0',
            'quadrature_strategy_ncdm_decay_dr' : '3, 3',
            'adjust_q_binning'                  : 'no',
        },
    },
    {
        'name'        : 'B2',
        'title'       : r'$m_{\nu l} = 0.02$ eV, atmos.\ gap, IO',
        'm_nuL'       : 0.02,
        'mL_str'      : 'mL0p02',
        'Gamma_list'  : [1960.78, 980.39, 490.19],
        'gtag_list'   : ['g2000', 'g1000', 'g500'],
        'tau_labels'  : ['0.5',   '1.0',   '2.0' ],
        'max_q_list'  : [45,      75,      130   ],
        'n_bins_list' : [45,      75,      130   ],
        'idx_daughter': 1,
        'q_max_fd'    : 15,
        'xlim'        : (0, 120),
        'ylim'        : (0.05, 2.3),
        'base_params' : {
            **COMMON_BG,
            'N_ncdm_decay_dr'                   : 2,
            'has_ncdm_decay_dr_ncdm'            : 'yes',
            'is_ncdm_decay_degenerate'          : 'yes',
            'neutrino_hierarchy'                : 'inverted',
            'decay_mass_gap'                    : 'atmospheric',
            'deg_ncdm_decay_dr'                 : '2.0, 1.0',
            'quadrature_strategy_ncdm_decay_dr' : '3, 3',
            'adjust_q_binning'                  : 'no',
        },
    },
]

# ─── plot each scenario separately ────────────────────────────────────────────
colors = ['royalblue', 'crimson', 'forestgreen']

os.makedirs('./plots_residuals', exist_ok=True)

for sc in SCENARIOS:
    name        = sc['name']
    m_nuL       = sc['m_nuL']
    mL_str      = sc['mL_str']
    Gamma_list  = sc['Gamma_list']
    gtag_list   = sc['gtag_list']
    tau_labels  = sc['tau_labels']
    max_q_list  = sc['max_q_list']
    n_bins_list = sc['n_bins_list']
    idx_d       = sc['idx_daughter']
    xlim        = sc['xlim']
    base_params = sc['base_params']

    fig, ax = plt.subplots(figsize=(8, 6), dpi=400)

    # Fermi-Dirac reference (analytic, on fine grid)
    q_max_fd = sc['q_max_fd']
    q_fd = np.linspace(q_max_fd / Nq_interp, q_max_fd, Nq_interp)
    f_fd = 1.0 / (np.exp(q_fd) + 1.0)
    ax.plot(q_fd, q_fd**3 * f_fd, 'k--', linewidth=2.0, label='Fermi-Dirac')

    # Decaying cases
    for Gamma, gtag, tau_lbl, max_q, n_bins, col in zip(
            Gamma_list, gtag_list, tau_labels, max_q_list, n_bins_list, colors):
        label = rf'$\tau_\nu = {tau_lbl}\ \mathrm{{Gyr}}$'
        stem  = f'PSD_{name}_{gtag}_{mL_str}'

        if COMPUTE_BACKGROUND:
            p = dict(base_params)
            p['m_ncdm_lightest']              = m_nuL
            p['log10Gamma_nu']                = np.log10(Gamma)
            p['maximum_q_ncdm_decay_dr']      = f'{max_q}, {max_q}'
            p['N_momentum_bins_ncdm_decay_dr']= f'{n_bins}, {n_bins}'
            print(f"[{name}] Γ = {Gamma} km/s/Mpc  (τ = {tau_lbl} Gyr, max_q={max_q}, N={n_bins})...")
            q_vals, lnf = get_psd(p, idx_d)
            np.savez(f'./Cl_CMB/{stem}.npz', q=q_vals, lnf=lnf)
        else:
            data   = np.load(f'./Cl_CMB/{stem}.npz')
            q_vals = data['q']
            lnf    = data['lnf']

        # Interpolate onto fine uniform grid for smooth curves
        f_coarse = pref * np.exp(lnf)
        q_max_i  = q_vals[-1]
        q_fine   = np.linspace(q_max_i / Nq_interp, q_max_i, Nq_interp)
        f_fine   = interp1d(q_vals, f_coarse, kind='cubic', fill_value='extrapolate')(q_fine)
        y_plot = q_fine**3 * f_fine
        if name == 'B2' and tau_lbl == '2.0':
            y_plot = savgol_filter(y_plot, 21, 3)
        ax.plot(q_fine, y_plot, color=col, linewidth=2.0, label=label)

    ax.set_xlim(xlim)
    ax.set_ylim(sc['ylim'])
    ax.set_xlabel(r'$q$')
    ax.set_ylabel(r'$q^3 \bar{f}_{\nu_l}(q,\tau_0)$', labelpad=15)
    ax.set_title(sc['title'], pad=10)
    ax.legend(loc='upper right')
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    [s.set_linewidth(1.5) for s in ax.spines.values()]
    fig.tight_layout()
    fig.savefig(f'plots_residuals/PSD_{name}.pdf')
    plt.show()
