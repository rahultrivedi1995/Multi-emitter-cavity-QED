"""Utilities for running experiments for multi-emitter CQED systems."""
import concurrent.futures as futures
import multiprocessing
import numpy as np
import signal
from typing import Tuple, Optional

import multi_emitter_hamiltonian
import single_photon_transport
import two_photon_transport


def _simulate(
        hamiltonian: multi_emitter_hamiltonian.MultiEmitterHamiltonian,
        input_freqs: np.ndarray,
        time_diffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute two-photon correlation and transmission.

    Args:
        hamiltonian: The hamiltonian for the mult-emitter system under
            consideration.
        input_freqs: The input frequencies to simulate the system at.
        time_diffs: The time-differences to simulate the two-photon
            correlation function at.

    Returns:
        The transmission at `input_freqs` and two-photon correlation at
        `input_freqs` and `time_diffs` as numpy arrays.
    """
    # We cannot pass the `hamiltonian` object directly the the
    # process pool executor - it seems that pickling of objects is not
    # implemented in multiprocessing library.
    transmission = single_photon_transport.single_photon_transmission(
            hamiltonian, input_freqs);
    two_ph_corr = two_photon_transport.two_photon_correlation(
            hamiltonian, time_diffs, input_freqs)

    return transmission, two_ph_corr


class MultiProcessingSimulator:
    """Class to perform the multi-emitter simulations in multiple processes."""
    def __init__(self, num_processes: Optional[int] = None):
        """Created a new `MulitProcessingSimulator` object.

        Args:
            num_processes: The number of processes to use for running
                simulations.
        """
        if num_processes is None:
            num_processes = multiprocessing.cpu_count()

        # Handle SIGINT properly by ignoring in the worker processes.
        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        self._pool = multiprocessing.Pool(num_processes)

    def simulate(
            self,
            hamiltonian: multi_emitter_hamiltonian.MultiEmitterHamiltonian,
            input_freqs: np.ndarray,
            time_diffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Submit a multi-emitter simulation.

        Args:
            hamiltonian: The hamiltonian for the multi-emitter system under
                consideration.
            input_freqs: The input frequencies to simulate.
            time_diffs: The time-differences to simulate the two-photon
                correlation function at.

        Returns:
            The transmission at `input_freqs` and two-photon correlation at
            `input_freqs` and `time_diffs` as numpy arrays.
        """
        try:
            return self._pool.apply(_simulate,
                                    (hamiltonian, input_freqs, time_diffs))
        except KeyboardInterrupt:
            self._pool.terminate()
            self._pool.join()


