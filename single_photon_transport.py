"""Compute the single- and two-photon transport through multi-emitter system.

A weak coherent state exciting a multi-emitter system is equivalent to a single
photon being transported through that system. This module provides functions
to compute the transmission of a single photon through the multi-emitter system.sf
"""

import numpy as np
import multi_emitter_hamiltonian


def _single_photon_green_func(
        hamiltonian: multi_emitter_hamiltonian.MultiEmitterHamiltonian,
        input_freqs: np.ndarray) -> np.ndarray:
    """Computes the frequency domain single-photon Green's function.

    The frequency-domain single-photon Green's function is the fourier transform
    of the time-domain single-photon Green's function. This can be computed as
    a function of frequency using the eigen-value decomposition of the single-
    photon Hamiltonian.

    Args:
        hamiltonian: The hamiltonian of the multi-emitter system.
        input_freqs: The frequencies at which to compute the single-photon
            green's functions.

    Returns:
        The single photon green's function as a function of frequency.
    """
    # Calculate the eigen-value decomposition of single-emitter component of the
    # Hamiltonian.
    eig_values, eig_vectors = hamiltonian.single_photon_eigen_decomposition()

    # Calculate the distance of the specified frequencies from the single-photon
    # eigen-values.
    freq_diff = eig_values[np.newaxis, :] - np.array(input_freqs)[
        :, np.newaxis]

    # Calculate the frequency domain green's functions.
    green_func = -1.0j * np.sum(
        (eig_vectors[[0], :]**2) / freq_diff, axis=1)

    return green_func

def single_photon_transmission(
        hamiltonian: multi_emitter_hamiltonian.MultiEmitterHamiltonian,
        input_freqs: np.ndarray) -> np.ndarray:
    """Computes the single photon transmission for the given Hamiltonian.

    Args:
        hamiltonian: The hamiltonian of the multi-emitter system, specified as a
            multi_emitter_hamiltonian.MultiEmtterHamiltonian object.
        input_freqs: The frequencies (relative to the reference frequency) at
            which to compute the transmission. This should be an array rank 1.

    Returns:
        The complex transmission through the multi-emitter system at the
        specified frequencies.

    Raises:
        ValueError: If the `frequencies` is not a one-dimensional array.
    """
    # Validate `frequencies`.
    if input_freqs.ndim != 1:
        raise ValueError("Expected the `input_freqs` array to be a 1D array "
                         "instead got {} dimensions".format(input_freqs.ndim))

    # Calculate the frequency domain green's functions.
    green_func = _single_photon_green_func(hamiltonian, input_freqs)

    # Calculate the transmission.
    transmission = -np.sqrt(
        hamiltonian.kappa_in * hamiltonian.kappa_out) * green_func

    return transmission
