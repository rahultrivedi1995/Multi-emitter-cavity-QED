"""Running experiments for the multi emitter CQED system."""
import concurrent.futures as futures
from matplotlib import pyplot as plt
import multiprocessing
import numpy as np
import os
import scipy.io
from typing import Tuple, List, Optional

import multi_emitter_hamiltonian
import single_photon_transport
import two_photon_transport
import experiment_utils


# Parameter values not changed throughout the simulations.
_KAPPA = 2.0 * np.pi * 25.0
_GAMMA = 2.0 * np.pi * 0.3


def var_with_coup_strength(serial: bool = False,
                           num_parallel: Optional[int] = None) -> None:
    """Run sweep with respect to coupling constant and number of emitters.

    Args:
        serial: Whether to run the simulation serially or in parallel.
        num_parallel: Number of simulations to launch in parallel.
    """
    # Directory for results.
    dir_name = os.path.join("results/identical_emitters")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # Parameters for the simulation.
    num_emitters = np.arange(1, 51)
    coup_consts = np.linspace(0.1, 2.0, 100) * _KAPPA
    input_freqs = np.linspace(-20, 20, 3000) * _KAPPA
    time_diffs = np.array([0])

    # Number of CPUs to use.
    if num_parallel is None:
        num_parallel = multiprocessing.cpu_count()

    simulator = experiment_utils.MultiProcessingSimulator(num_parallel)
    # Setup the dictionary to save simulation data.
    results_transmission = {str(num_em): np.zeros([input_freqs.size,
                                                   coup_consts.size])
                            for num_em in num_emitters}
    results_two_ph_corr = {str(num_em): np.zeros([input_freqs.size,
                                                  coup_consts.size])
                           for num_em in num_emitters}

    def _simulate_and_save(simulator, num_em, coup_const_index,
                           input_freqs, time_diffs):
        coup_const = coup_consts[coup_const_index]
        hamiltonian = multi_emitter_hamiltonian.MultiEmitterHamiltonian(
                0, 0.5 * _KAPPA, 0.5 * _KAPPA, 0,
                [0] * num_em, [_GAMMA] * num_em, [coup_const] * num_em)
        transmission, two_ph_corr = simulator.simulate(hamiltonian,
                                                       input_freqs,
                                                       time_diffs)
        results_transmission[str(num_em)][:, coup_const_index] = np.abs(
                transmission)**2
        results_two_ph_corr[str(num_em)][:, coup_const_index] = two_ph_corr[:, 0]

    if serial:
        print("Simulating in serial")
        for num_em in num_emitters:
            for coup_const_index, coup_const in enumerate(coup_consts):
                print("Simulating num_emitters {} coup_const {}".format(
                                num_em, coup_const))
                _simulate_and_save(simulator, num_em, coup_const_index,
                                   input_freqs, time_diffs)
    else:
        print("Simulating in parallel")
        with futures.ThreadPoolExecutor(num_parallel) as executor:
            for num_em in num_emitters:
                for coup_const_index, coup_const in enumerate(coup_consts):
                    print("Submitting num_emitters {} coup_const {}".format(
                                num_em, coup_const))
                    executor.submit(_simulate_and_save,
                                    simulator, num_em, coup_const_index,
                                    input_freqs, time_diffs)

    # Save the results.
    scipy.io.savemat(os.path.join(dir_name, "transmission_var_coup_const.mat"),
                     results_transmission)
    scipy.io.savemat(os.path.join(dir_name, "equaltimecorr_var_coup_const.mat"),
                     results_two_ph_corr)


def var_with_detuning(serial: bool = False,
                      num_parallel: Optional[int] = None) -> None:
    """Run sweep with respect to coupling constant and number of emitters.

    Args:
        serial: Whether to run the simulation serially or in parallel.
        num_parallel: Number of simulations to launch in parallel.
    """
    # Directory for results.
    dir_name = os.path.join("results/identical_emitters")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # Parameters for the simulation.
    num_emitters = np.arange(1, 51)
    detunings = np.linspace(0, 2, 100) * _KAPPA
    input_freqs = np.linspace(-3, 3, 3000) * _KAPPA
    time_diffs = np.array([0])

    # Number of CPUs to use.
    if num_parallel is None:
        num_parallel = multiprocessing.cpu_count()

    simulator = experiment_utils.MultiProcessingSimulator(num_parallel)

    # Setup the dictionary to save simulation data.
    results_transmission = {str(num_em): np.zeros([input_freqs.size,
                                                   detunings.size])
                            for num_em in num_emitters}
    results_two_ph_corr = {str(num_em): np.zeros([input_freqs.size,
                                                  detunings.size])
                           for num_em in num_emitters}

    def _simulate_and_save(simulator, num_em, detuning_index,
                           input_freqs, time_diffs):
        detuning = detunings[detuning_index]
        hamiltonian = multi_emitter_hamiltonian.MultiEmitterHamiltonian(
                0, 0.5 * _KAPPA, 0.5 * _KAPPA, 0,
                [detuning] * num_em, [_GAMMA] * num_em,
                [0.25 * _KAPPA] * num_em)
        transmission, two_ph_corr = simulator.simulate(hamiltonian,
                                                       input_freqs,
                                                       time_diffs)
        results_transmission[
            str(num_em)][:, detuning_index] = np.abs(transmission)**2
        results_two_ph_corr[
            str(num_em)][:, detuning_index] = two_ph_corr[:, 0]

    if serial:
        print("Simulating in serial")
        for num_em in num_emitters:
            for detuning_index, detuning in enumerate(detunings):
                print("Simulating num_emitters {} coup_const {}".format(
                                num_em, detuning))
                _simulate_and_save(simulator, num_em, detuning_index,
                                   input_freqs, time_diffs)
    else:
        print("Simulating in parallel")
        with futures.ThreadPoolExecutor(num_parallel) as executor:
            for num_em in num_emitters:
                for detuning_index, detuning in enumerate(detunings):
                    print("Submitting num_emitters {} coup_const {}".format(
                                num_em, detuning))
                    executor.submit(_simulate_and_save,
                                    simulator, num_em, detuning_index,
                                    input_freqs, time_diffs)

    # Save the results.
    scipy.io.savemat(os.path.join(dir_name, "transmission_var_detuning.mat"),
                     results_transmission)
    scipy.io.savemat(os.path.join(dir_name, "equaltimecorr_var_detuning.mat"),
                     results_two_ph_corr)

if __name__ == "__main__":
    print("Running experiment with varying coupling strengths")
    var_with_coup_strength(False, 20)

    print("Running experiments with varying detuning")
    var_with_detuning(False, 20)
