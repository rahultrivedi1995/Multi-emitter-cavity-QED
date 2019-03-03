"""Running experiments for broadened multi-emitter cQED system."""
from typing import List, Tuple

from matplotlib import pyplot as plt
import numpy as np
import os
import logging
import scipy.io
import signal

import multi_emitter_hamiltonian
import single_photon_transport
import two_photon_transport

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

# Parameter values not changed throughout the simulations.
_KAPPA = 2.0 * np.pi * 25.0
_GAMMA = 2.0 * np.pi * 0.3
_DELTA = 0.8 * _KAPPA
_COUP_CONST = 0.2 * _KAPPA


def _get_resonant_dip_indices(
        equal_time_corr: np.ndarray) -> np.ndarray:
    """Compute the indices at which the dips in the given spectrum are observed.

    Args:
        equal_time_corr: The equal time correlation to analyze.

    Returns:
        The numpy array with the resonant indices.
    """
    # The dips are located by computing the points at which the derivative
    # changes sign from negative to positive.
    sign_deriv = np.sign(np.diff(equal_time_corr))

    # Detect the points at which the change in sign is happening. This is done
    # by computing another derivative of `sign_deriv` and checking the points
    # at which it is >0.
    deriv_sign_deriv = np.diff(sign_deriv)

    # The addition of 1 is required since every `np.diff` moves the indices back
    # by 1.
    return np.where(deriv_sign_deriv > 0)[0] + 1


def _analyze_multi_emitter_system(
        hamiltonian: multi_emitter_hamiltonian.MultiEmitterHamiltonian,
        input_freqs: np.ndarray,
        freqs_fine: np.ndarray) -> Tuple[List[float], List[float]]:
    """Analyze a single instance of an inhomogenously broadened system.

    This function computes `g2(0)` for the given Multi-emitter system as
    a function of frequencies. From the resulting data, it extracts the
    `locations and values of the subradiant dips. The location and values are
    made more acccurate by simulating the `g2(0)` at frequencies close to these
    fine dips.

    Args:
        hamiltonian: The hamiltonian of the subradiant dips.
        input_freqs: The frequencies at which to compute the `g2(0)` function.
        freqs_fine: The frequencies around the subradiant dips at which to
            compute `g2(0)`.

    Returns:
        The frequnecies and the values of all the dips.
    """
    # Simulate the two-photon transport at `input_freqs`.
    two_ph_corr = two_photon_transport.two_photon_correlation(
            hamiltonian, np.array([0]), input_freqs)

    # Extract the indices of the dip.
    dip_indices = _get_resonant_dip_indices(two_ph_corr[:, 0])

    # Iterate over all the dip indices, and resimulate `g2(0)` to improve the
    # estimate of the subradiant dips.
    dip_freqs = []
    dip_corr_vals = []
    for dip_index in dip_indices:
        freq = input_freqs[dip_index]
        # Perform a finely grained simulation.
        two_ph_corr_fine = two_photon_transport.two_photon_correlation(
                hamiltonian, np.array([0]), freq + freqs_fine)
        # Extract the dip in the finely grained simulation.
        fine_dip_index = np.argmin(two_ph_corr_fine[:, 0])
        if two_ph_corr_fine[fine_dip_index] < 1:
            dip_freqs.append(freq + freqs_fine[fine_dip_index])
            dip_corr_vals.append(two_ph_corr_fine[fine_dip_index, 0])

    LOGGER.debug("Dip frequencies {}".format(dip_freqs))
    LOGGER.debug("Dip correlation values {}".format(dip_corr_vals))
    return dip_freqs, dip_corr_vals


def run_monte_carlo(results_dir: str,
                    num_samples: int,
                    input_freqs: np.ndarray,
                    num_emitters: int,
                    mean_freq_em: float,
                    sigma_freq_em: float,
                    kappa: float = _KAPPA,
                    gamma: float = _GAMMA,
                    coup_const: float = _COUP_CONST) -> None:
    """Run Monte carlo simulation to analyze inhomogenous broadenings.

    Runs monte-carlo simulations of equal time correlation function for the
    emitters and saves the statistics of the polaritonic and subradiant dips.

    Args:
        results_dir: The directory where to save the results. The run will
            create another directory inside the results directory in which it
            will save the parameters, statistics as well as the log files for
            the current run.
        num_samples: The number of monte-carlo simulations to run.
        input_freqs: The input frequencies at which to compute the equal-time
            correlation function.
        num_emitters: The number of emitters in the multi-emitter sytem.
        mean_freq_em: The mean frequency of the emitters.
        sigma_freq_em: The standard deviation in the frequency of the emitters.
        kappa: The decay rate of the cavity.
        gamma: The homogenous broadening in the emitter.
        coup_const: The coupling constant of the emitters with the optical
            cavity mode.
    """
    # Setup the directory in which to save the results.
    dir_name = os.path.join(
            results_dir,
            "monte_carlo_g_{:0.3f}_delta_{:0.3f}_kappa_{:0.3f}_gamma_{:0.3f}"
            "_inhom_{:0.3f}".format(coup_const, mean_freq_em, kappa,
                                    gamma, sigma_freq_em))

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # Add log file.
    log_file_handler = logging.FileHandler(os.path.join(dir_name, "diary.log"))
    format_file_handler = logging.Formatter(
        "%(asctime)s:%(name)s:%(levelname)s: %(message)s")
    log_file_handler.setFormatter(format_file_handler)
    logging.getLogger("").addHandler(log_file_handler)

    # Since the subradiant peaks are at times difficult to resolve, in order to
    # obtain accurate estimate of the frequencies and g2 values at the
    # subradiant dips, we perform a fine-grained simulation near each of the
    # subradiant dips.
    freqs_fine = np.linspace(-2 * gamma, 2 * gamma, 100)

    # List of parameters.
    params = {"delta_emitters": [],
              "input_freqs": input_freqs,
              "kappa": kappa,
              "coup_const": coup_const,
              "gamma": gamma,
              "mean_freq_em": mean_freq_em,
              "sigma_freq_em": sigma_freq_em,
              "num_emitters": num_emitters}

    # Lists for the emitter statistics.
    results = {"subradiant_freqs": [],
               "polaritonic_freqs": [],
               "subradiant_corr_vals": [],
               "polaritonic_corr_vals": []}

    # Perform monte carlo simulations.
    for sample in range(num_samples):
        LOGGER.info("Simulating sample {}.".format(sample))
        delta_emitters = np.random.normal(mean_freq_em, sigma_freq_em,
                                          num_emitters)
        params["delta_emitters"].append(delta_emitters)

        hamiltonian = multi_emitter_hamiltonian.MultiEmitterHamiltonian(
                delta_cavity=0,
                kappa_in=0.5 * kappa,
                kappa_out=0.5 * kappa,
                kappa_loss=0,
                delta_emitters=delta_emitters,
                decay_emitters=[gamma] * num_emitters,
                cavity_emitter_couplings=[coup_const] * num_emitters)
        dip_freqs, dip_corr_vals = _analyze_multi_emitter_system(
                hamiltonian, input_freqs, freqs_fine)

        if dip_freqs:
            # All the dips except for the first and last are interpreted as
            # subradiant dips.
            results["subradiant_freqs"] += dip_freqs[1:-1]
            results["subradiant_corr_vals"] += dip_corr_vals[1:-1]

            # The first and last dips are the polaritonic dips.
            results["polaritonic_freqs"].append(dip_freqs[0])
            results["polaritonic_freqs"].append(dip_freqs[-1])
            results["polaritonic_corr_vals"].append(dip_corr_vals[0])
            results["polaritonic_corr_vals"].append(dip_corr_vals[-1])


        scipy.io.savemat(os.path.join(dir_name, "statistics.mat"), results)
        scipy.io.savemat(os.path.join(dir_name, "parameters.mat"), params)


if __name__ == "__main__":
    # Simulating the resonant case.
    # Small inhomogenous broadening (5 GHz).
    """
    input_freqs = np.array(np.linspace(-4, -0.1, 250).tolist() +
                           np.linspace(-0.1, 0.1, 100).tolist() +
                           np.linspace(0.1, 4, 250).tolist()) * _KAPPA
    run_monte_carlo("results/nonidentical_emitters",
                    num_samples=100,
                    input_freqs=input_freqs,
                    num_emitters=20,
                    mean_freq_em=0,
                    sigma_freq_em=5)

    # Large inhomogenous broadening (20 GHz).
    input_freqs = np.array(np.linspace(-4, -0.25, 250).tolist() +
                           np.linspace(-0.25, 0.25, 100).tolist() +
                           np.linspace(0.25, 4, 250).tolist()) * _KAPPA
    run_monte_carlo("results/nonidentical_emitters",
                    num_samples=100,
                    input_freqs=input_freqs,
                    num_emitters=20,
                    mean_freq_em=0,
                    sigma_freq_em=20)
    """
    input_freqs = np.linspace(0, 2.5, 2000) * _KAPPA
    run_monte_carlo("results/nonidentical_emitters/detuned",
                    num_samples=100,
                    input_freqs=input_freqs,
                    num_emitters=20,
                    mean_freq_em=0.8 * _KAPPA,
                    sigma_freq_em=5)
    run_monte_carlo("results/nonidentical_emitters/detuned",
                    num_samples=100,
                    input_freqs=input_freqs,
                    num_emitters=20,
                    mean_freq_em=0.8 * _KAPPA,
                    sigma_freq_em=20)
