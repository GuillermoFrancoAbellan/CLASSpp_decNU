import numpy as np
import healpy as hp

from scipy.interpolate import interp1d
from scipy.stats import poisson

pref = ((2 * np.pi)**3)/2

eV = 1.602176487e-19
k_B = 1.3806504e-23
T_nu = 2.7255*0.71611
As = 2.0968e-09
ns = 0.9652
k_pivot = 0.05

l_max = 17
n_kmodes = 75
Nq_interp = 100


k_min, k_max = 1e-4, 1e-1   # 1/Mpc
k_values = np.logspace(np.log(k_min), np.log(k_max), n_kmodes, base=np.e)

# Convert k_values to ln(k)
lnk_values = np.log(k_values)
number_string = ", ".join(f"{elem:.2e}" for elem in k_values)

print("As =",As)
print("ns =",ns)
print("lmax =",l_max)
print("k_output_values =",number_string)
print("")
print("Quantities interpolated over momenta will be evaluated on Nq_interp = %d bins"%Nq_interp)

def read_files(file_path, perturb_base_path, q_ratio, q_ratio2, decay = True):

    """
    Reads background/perturbation files from CLASS to get the neutrino PSD and the neutrino temperature multipoles (defined as in eq. 2.4 of https://arxiv.org/abs/2103.01274)

    Parameters:
        file_path: path to the _background.dat file produced by CLASS
        perturb_base_path: root path to the _perturbations files produced by CLASS, i.e. the path without the subscript _k{i}_s.dat
        q_ratio (np.ndarray): 1D array of q-bins at which CLASS evolves the back. and perts. of the daughter NCDM species (printed on the screen when running .ini files)
        q_ratio2 (np.ndarray): 1D array of q-bins at which CLASS stores the temperature perts. (it's a small subset of q_ratio, also printed on the screen when running .ini files)
        decay: whether we are considering neutrino decays or not

    Returns:
        f (np.ndarray): present-day PSD of the daughter neutrino evaluated at the bins q_int, printed only if "decay = True"
        delta_interpolated (np.ndarray): 3D array of neutrino temperature multipoles (for each q, k, l)
        q_int (np.ndarray): Comoving momentum bins q at which the PSD and the temp. multipoles are evaluated.

    """

    q_size = len(q_ratio)
    q_max = np.max(q_ratio)
    q_size2 = len(q_ratio2)

    # Read background data
    with open(file_path, 'r') as file:
        line_count = 0
        all_columns_bg = []
        for line in file:
            line_count += 1
            if line_count <= 4:  # Skip the first four lines
                continue
            columns = line.split()
            if len(all_columns_bg) == 0:
                all_columns_bg = [[] for _ in range(len(columns))]
            elif len(all_columns_bg) != len(columns):
                diff = len(columns) - len(all_columns_bg)
                all_columns_bg.extend([] for _ in range(diff))

            for j, column_data in enumerate(columns):
                all_columns_bg[j].append(float(column_data))

    # Initialize a list to store data for different wavenumber outputs
    all_columns_list = []

    # Loop over wavenumber outputs
    for i in range(n_kmodes):
        file_path = f"{perturb_base_path}_k{i}_s.dat"
        # Read only the last line of the file
        with open(file_path, 'rb') as file:
            file.seek(-2, 2)  # Move to just before the end of file
            while file.read(1) != b'\n':
                file.seek(-2, 1)  # Move back until you hit the last full line
            last_line = file.readline().decode()

        columns = last_line.split()
        all_columns = [float(value) for value in columns]
        all_columns_list.append(all_columns)

    # Initialize lists to store perturbations
    delta = [[[None for _ in range(l_max + 1)] for _ in range(n_kmodes)] for _ in range(q_size2)]

    # Prepare f using background data
    if decay == True:
        if is_ncdm_decay_degenerate == "yes":
            lnf_all = [all_columns_bg[18 + q_size + i] for i in range(q_size)]
        else:
            lnf_all = [all_columns_bg[21 + q_size + i] for i in range(q_size)]
        lnf_s = [sublist[-1] for sublist in lnf_all]
        lnf_d = np.array(lnf_s, dtype=float)
        f = pref*np.exp(lnf_d)

    q_int = np.linspace(q_max/Nq_interp, q_max, Nq_interp)

    # Interpolate f
    if decay == True:
        interp_func = interp1d(q_ratio, f, kind='cubic', fill_value='extrapolate')
        f = interp_func(q_int)

    # Retrieve values for each momentum bin and multipole
    for k in range(n_kmodes):
        for q in range(q_size2):
            for l in range(l_max + 1):
                if is_ncdm_decay_degenerate == "yes":
                    delta[q][k][l] = all_columns_list[k][25 + l + (l_max+1)*q]
                else:
                    delta[q][k][l] = all_columns_list[k][29 + l + (l_max+1)*q]

    # Interpolate delta to new q-int values
    delta_interpolated = np.zeros((len(q_int), len(delta[0]), len(delta[0][0])))
    for k in range(len(delta[0])):  # Loop over k
        for l in range(len(delta[0][k])):  # Loop over l
            delta_q_k_l = [delta[q][k][l] for q in range(q_size2)]
            interp_func = interp1d(q_ratio2, delta_q_k_l, kind='cubic', fill_value='extrapolate')
            delta_q_int_k_l = interp_func(q_int)
            delta_interpolated[:, k, l] = delta_q_int_k_l

    if decay == True:
        return f, delta_interpolated, q_int
    else:
        return delta_interpolated, q_int



def read_Cl_CMB(file_path):
    """
    Read CMB files from CLASS

    Parameters:
        file_path: path to the _cl_lensed.dat file produced by CLASS

    Returns:
        l_values (np.ndarray): multipoles at which the CMB spectra are evaluated
        Cl_TT_values (np.ndarray): CMB TT angular power spectrum
        Cl_EE_values (np.ndarray): CMB EE angular power spectrum

    """

    l_values = []
    Cl_TT_values = []
    Cl_EE_values = []

    with open(file_path, 'r') as file:
        line_count = 0
        for line in file:
            line_count += 1
            if line_count <= 11:  # Skip the first 11 lines
                continue

            # Split the line into columns
            columns = line.split()

            # Ensure there are at least three columns to read
            if len(columns) < 3:
                raise ValueError(f"Unexpected file format: less than 3 columns in line {line_count}")

            # Append the values from the respective columns
            l_values.append(float(columns[0]))  # First column
            Cl_TT_values.append(float(columns[1]))  # Second column
            Cl_EE_values.append(float(columns[2]))  # Third column

    l_values = np.array(l_values)
    Cl_TT_values = np.array(Cl_TT_values)
    Cl_EE_values = np.array(Cl_EE_values)

    return l_values, Cl_TT_values, Cl_EE_values



def compute_power_spectrum(delta):
    """
    Compute q-dependent CNB angular power spectrum according to eq. 2.6 in https://arxiv.org/abs/2103.01274

    Parameters:
        delta (np.ndarray): 3D array of neutrino temperature multipoles (for each q, k, l) computed by CLASS

    Returns:
        Cl_1 (np.ndarray): 2D array of CNB angular power spectra at different momentum bins q
    """

    # Calculate Cl values
    Cl = []
    for q in range(Nq_interp):
        integral_values = []
        for l in range(1, l_max + 1):
            integral_value = 0.0
            for k in range(len(lnk_values) - 1):
                d_lnk = lnk_values[k + 1] - lnk_values[k]
                delta_squared_midpoint = 0.5 * ( ((k_values[k]/k_pivot)**(ns-1))*delta[q][k][l]**2 +
                                                 ((k_values[k+1]/k_pivot)**(ns-1))*delta[q][k + 1][l]**2 )
                integral_value += delta_squared_midpoint * d_lnk
            integral_values.append(T_nu**2 * (4*np.pi) * As * integral_value)
        Cl.append(integral_values)

    Cl_1 = Cl

    # Add monopole term as 0 and shift values
    for q in range(Nq_interp):
        Cl_1[q] = np.concatenate((Cl_1[q], [0]))
        Cl_1[q] = np.roll(Cl_1[q], 1)

    return Cl_1


def compute_avg_hannestad(f, Cl, m_eV,q_int):
    """
    Compute momentum-averaged CNB angular power spectrum according to eq. 15 in https://arxiv.org/abs/0910.4578

    Parameters:
        f (np.ndarray): present-day PSD of the neutrino evaluated at the bins q_int
        Cl (np.ndarray): q-dependent CNB angular power spectra obtained from compute_power_spectrum(...).
        m_eV (np.ndarray): Mass of the neutrino in eV
        q_int (ndarray): Comoving momentum bins q at which the PSD is evaluated.

    Returns:
        maps_fluc_unscaled (np.ndarray): 2D array of unscaled fluctuated maps (δT/T).
    """

    dq = q_int[1]-q_int[0]
    m_class = (m_eV * eV) / (k_B * T_nu)
    eps = np.sqrt(q_int**2 + m_class**2)

    # Compute energy density integral for decaying case
    energy_d = 0.
    for q in range(Nq_interp - 1):
        en_mid = 0.5 * (q_int[q]**2 * eps[q] * f[q] + q_int[q+1]**2 * eps[q+1] * f[q+1])
        energy_d += dq * en_mid

    # Compute q-average of Cl
    Cl_avg_n = 0.
    for q in range(Nq_interp - 1):
        Cl_mid = 0.5 * (q_int[q]**2 * eps[q] * f[q] * np.sqrt(Cl[q]) +
                        q_int[q+1]**2 * eps[q+1] * f[q+1] * np.sqrt(Cl[q+1]))
        Cl_avg_n += Cl_mid * dq

    Cl_avg_test = (Cl_avg_n/energy_d)**2

    return Cl_avg_test


def generate_poisson_fluctuations(Cl_avg0, delta_map, n_trials, N_total):
    """
    Generate Poisson-fluctuated sky maps and compute their angular power spectra.

    Parameters:
        Cl_avg0 (np.ndarray): Theoretical Cℓ spectrum (including monopole and dipole).
        delta_map (np.ndarray): Input fractional fluctuation map (δT/T).
        n_trials (int): Number of Monte Carlo realizations.
        N_total (int): Total number of neutrino capture events.

    Returns:
        maps_fluc_unscaled (np.ndarray): 2D array of unscaled fluctuated maps (δT/T).
        cls_fluc (np.ndarray): Array of angular power spectra per trial.
        ratios_fluc (list): List of Cℓ / Cℓ_theory ratios for each trial (ℓ > 0).
    """
    n_pixels = len(delta_map)
    lmax = len(Cl_avg0) - 1
    ls = np.arange(1, lmax + 1)  # Exclude monopole (ℓ=0)

    # Add monopole and convert to expected counts
    map_wmono = N_total * (delta_map + 1.95)

    # Allocate arrays
    maps_fluc_scaled = np.empty((n_trials, n_pixels), dtype=np.float64)
    maps_fluc_unscaled = np.empty((n_trials, n_pixels), dtype=np.float64)
    cls_fluc = []
    ratios_fluc = []

    rng = np.random.default_rng(seed=5)
    # Generate Poisson samples and compute Cℓs
    for i in range(n_trials):
        sampled_map = rng.poisson(map_wmono)
        maps_fluc_scaled[i] = sampled_map
        maps_fluc_unscaled[i] = (sampled_map / N_total - 1.95)

    for m in maps_fluc_unscaled:
        cl = hp.anafast(m, lmax=lmax)
        cls_fluc.append(cl[ls])
        ratios_fluc.append(cl[ls] / Cl_avg0[1:])

    return maps_fluc_scaled, maps_fluc_unscaled, np.array(cls_fluc), np.array(ratios_fluc)
