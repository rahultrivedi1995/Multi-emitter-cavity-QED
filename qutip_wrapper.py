"""Use qutip to perform transmission and correlation computations."""

import numpy as np
import qutip
from typing import Tuple, List

import multi_emitter_hamiltonian


def _setup_qutip_hamiltonian(
        hamiltonian: multi_emitter_hamiltonian.MultiEmitterHamiltonian,
        amp_drive: float,
        delta_drive: float,
        num_cav_states: int) -> Tuple[qutip.qobj.Qobj,
                                      qutip.qobj.Qobj,
                                      List[qutip.qobj.Qobj]]:
    """Setup the qutip hamiltonian for the given multi-emitter Hamiltonian.

    The hamiltonian for the multi-emitter CQED system is given by the following
    operator equation:
        H_0 = delta_c a^dagger a + sum_n delta_n sigma_n^dagger sigma_n +
                sum_n g_n (a sigma_n^dagger + sigma_n a^dagger)
    where `a` is the annihilation operator for the cavity, `sigma_n` is the
    lowering operator for the nth emitter, `delta_c` is the resonant frequency
    of the cavity specified as its detuning from a reference frequency,
    `delta_n` is the detuning of the nth emitter from the reference frequency,
    `g_n` is the coupling constant between the nth emitter and the cavity mode.

    For modelling a multi-emitter cQED system excited with a coherent state, we
    consider a driven hamiltonian of the form:
                        H = H_0 + Omega * (a + a^dagger)
    where `Omega` is the amplitude of the coherent drive used for exciting the
    cavity, with the coherent drive being a continuous wave drive at the
    reference frequency.

    Args:
        hamiltonian: The Hamiltonian for the multi-emitter system specified as
            a `multi_emitter_hamiltonian.MultiEmitterHamiltonian` object. This
            object is primarily used for accessing the various parameters of the
            multi-emitter cQED hamiltonian (e.g. resonant frequency, emitter
            frequency, cavity decay etc.)
        drive_amp: The amplitude of the coherent drive used for driving the
            cavity mode.
        drive_freq: The frequency at which the coherent drive used for driving
            the cavity mode is oscillating.
        num_cav_states: The number of Fock states used for describing the
            Hilbert space of the cavity.

    Returns:
        The qutip object corresponding to the hamiltonian operator for the
        Hamiltonian of the multi-emitter system, and the qutip operator
        corresponding to the annihilation operator of the cavity.
    """
    # Setup the annihilation operator for the cavity. Since the operator needs
    # to be defined in the hilbert space of the cavity and the emitters taken
    # together, it has to be explicitly defined as an identity operator over the
    # hilbert space of the emitters.
    annihil_op = ([qutip.destroy(num_cav_states)] +
                  [qutip.qeye(2)] * hamiltonian.num_emitters)
    annihil_op = qutip.tensor(annihil_op)

    # Initialize the hamiltonian operator with only the cavity hamiltonian.
    # Note that so as to work with a time-dependent hamiltonian, the hamiltonian
    # is written in a frame of reference rotating at the frequency of the input
    # drive.
    qutip_hamiltonian = (
        hamiltonian.delta_cavity - delta_drive) * annihil_op.dag() * annihil_op

    # Add the hamiltonian corresponding to the coherent drive.
    qutip_hamiltonian += amp_drive * (annihil_op + annihil_op.dag())

    # Maintain list of emitter lowering operators.
    sigmas = []

    # Calculate the hamiltonian for all the emitters.
    for k, (delta_em, coup_em) in enumerate(zip(
        hamiltonian.delta_emitters,
        hamiltonian.cavity_emitter_couplings)):

        # Get the lowering operator for the kth emitter. Note that this is an
        # identity operator on the hilbert space of the cavity, the previous
        # `k - 1` emitters and the next `N - k - 1` emitters (where `N` is the
        # total number of emitters).
        sigma = ([qutip.qeye(num_cav_states)] +
                 [qutip.qeye(2)] * k + [qutip.destroy(2)] +
                 [qutip.qeye(2)] * (hamiltonian.num_emitters - k - 1)) 
        sigma = qutip.tensor(sigma)
        sigmas.append(sigma)

        # Add the hamiltonian for the energy of emitter.
        qutip_hamiltonian += ((delta_em - delta_drive) * sigma.dag() * sigma)

        # Add the interaction hamiltonian between the emitter and the cavity.
        qutip_hamiltonian += (coup_em * (sigma * annihil_op.dag() +
                                         annihil_op * sigma.dag()))

    return qutip_hamiltonian, annihil_op, sigmas


def _setup_loss_operators(
        hamiltonian: multi_emitter_hamiltonian.MultiEmitterHamiltonian,
        annihil_op: qutip.qobj.Qobj,
        sigmas: List[qutip.qobj.Qobj]) -> List[qutip.qobj.Qobj]:
    """Setup the loss operators for the multi-emitter cQED system.

    The loss operators describe how the multi-emitter system interacts with the
    baths that it couples to. These operators are required while using the
    master equation formalism for analyzing the multi-emitter system.

    Args:
        hamiltonian: The hamiltonian of the multi-emitter system, specified as
            a multi_emitter_hamiltonian.MultiEmitterHamiltonian object.
        annihil_op: The annihilation operator for the mode of the cavity of the
            multi-emitter system.
        sigmas: A list of lowering operators for the two level atoms in the
            multi-emitter system.

    Returns:
        A list of operators, as `qutip.qobj.Qobj` objects, corresponding to the
        loss channels that the multi-emitter system couples to.
    """
    # Total cavity decay rate.
    kappa = (
        hamiltonian.kappa_loss + hamiltonian.kappa_in + hamiltonian.kappa_out)

    # Loss operator for the cavity - this is defined as `sqrt(kappa) * a`, where
    # `a` is the annihilation operator for the cavity.
    loss_op_cavity = np.sqrt(kappa) * annihil_op

    # Loss operators for the emitters - for the nth operator, this is defined by
    # `sqrt(gamma_n) * sigma_n`, where `sigma_n` is the lowering operator for
    # that emitter.
    loss_ops_em = [np.sqrt(gamma) * sigma
                   for gamma, sigma in zip(hamiltonian.decay_emitters,
                                           sigmas)]

    return [loss_op_cavity] + loss_ops_em


def single_photon_transmissivity(
        hamiltonian: multi_emitter_hamiltonian.MultiEmitterHamiltonian,
        input_freqs: np.ndarray,
        amp_drive: float,
        num_cav_states: int = 4) -> np.ndarray:
    """Computes the single photon transmissivity for the given Hamiltonian.

    This function computes the transmittivity of a multi-emitter using qutip.

    Args:
        hamiltonian: The hamiltonian of the multi-emitter system, specified as a
            multi_emitter_hamiltonian.MultiEmtterHamiltonian object.
        input_freqs: The frequencies (relative to the reference frequency) at
            which to compute the transmission. This should be an array rank 1.
        amp_drive: The amplitude of the coherent drive used for driving the
            cavity mode.
        num_cav_states: The number of Fock states used for describing the
            Hilbert space of the cavity.

    Returns:
        The transmissivity through the multi-emitter system at the specified
        frequencies.

    Raises:
        ValueError: If the `frequencies` is not a one-dimensional array.
    """
    if input_freqs.ndim != 1:
        raise ValueError("Expected the `input_freqs` array to be a 1D array "
                         "instead got {} dimensions".format(input_freqs.ndim))

    # List of transmissions for different frequencies.
    transmissions = []

    for freq in input_freqs:
        # Get the hamiltonian and the operators for cavity and emitters.
        qutip_hamiltonian, annihil_op, sigmas = _setup_qutip_hamiltonian(
            hamiltonian, amp_drive, freq, num_cav_states)

        # Setup the loss operators.
        loss_ops = _setup_loss_operators(hamiltonian, annihil_op, sigmas)

        # Calculate the steady state density matrix.
        density_matrix_ss = qutip.steadystate(qutip_hamiltonian, loss_ops)

        # Calculate the expectation value of the number of photons.
        num_photons = qutip.expect(annihil_op.dag() * annihil_op,
                                   density_matrix_ss)

        # Calculate the transmission.
        norm_factor = hamiltonian.kappa_in * hamiltonian.kappa_out 
        transmissions.append(norm_factor * num_photons / amp_drive**2)

    return np.array(transmissions)


def two_photon_correlation(
        hamiltonian: multi_emitter_hamiltonian.MultiEmitterHamiltonian,
        time_diffs: np.ndarray,
        input_freqs: np.ndarray,
        amp_drive: float,
        num_cav_states: int = 4) -> np.ndarray:
    """Calculate the two-photon correlation using QuTip.

    Note that in many computations, it is required to compute the two-time
    correlation at 0 time-differences (this corresponds to the case when
    `time_diffs` is `np.array([0])`). In this case, this function simply
    computes the following expression:
                < a^dagger * a * a^dagger * a > / < a^dagger * a >^2
    instead of using Qutip's `correlation_3opt_1t` function.

    Args:
        hamiltonian: The hamiltonian of the multi-emitter system specified as a
            a `multi_emitter_hamitlonian.MultiEmitterHamiltonian` object.
        time_diffs: The time-differences in the output state at which to
            evaluate the response.
        input_freqs: The frequencies at which the system is excited.

    Returns:
        The two photon correlation function as a function of the time difference
        between the time-points at which the correlation is being computed.

    Raises:
        ValueError: If either `time_diffs` or `input_freqs` are not one
        dimensional arrays.
    """
    # Validate the input arguments.
    if time_diffs.ndim != 1:
        raise ValueError("Expected the `time_diffs` array to be a 1D array "
                         "instead got {} dimensions".format(time_diffs.ndims))

    if input_freqs.ndim != 1:
        raise ValueError("Expected the `input_freqs` array to be a 1D array "
                         "instead got {} dimensions".format(input_freqs.ndims))

    # List of correlation functions at different frequencies.
    correlations = []

    for freq in input_freqs:
        # Get the hamiltonian for the 
        qutip_hamiltonian, annihil_op, sigmas = _setup_qutip_hamiltonian(
            hamiltonian, amp_drive, freq, num_cav_states)

        # Setup the loss operators.
        loss_ops = _setup_loss_operators(hamiltonian, annihil_op, sigmas)

        # Calculate the steady sate density matrix. This is required for
        # normalizing the correlation function, as well as for handling the
        # case where `time_diffs = np.array([0])`.
        density_matrix_ss = qutip.steadystate(qutip_hamiltonian, loss_ops)

        # Calculate the correlation as a function of time.
        if np.array_equal(time_diffs, np.array([0])):
            g2_tau = np.array(
                [qutip.expect(annihil_op.dag() * annihil_op.dag() *
                              annihil_op * annihil_op,
                              density_matrix_ss)])
        else:
            g2_tau = qutip.correlation_3op_1t(qutip_hamiltonian,
                                              None, np.abs(time_diffs),
                                              loss_ops,
                                              annihil_op.dag(),
                                              annihil_op.dag() * annihil_op,
                                              annihil_op,
                                              solver='es')

        # Normalizing the correlation function. To normalize this correlation
        # function, we compute the correlation at inifinite time - this is the
        # same as computing the square of expection of photon number operator in
        # the steady state.
        num_photons = qutip.expect(annihil_op.dag() * annihil_op,
                                   density_matrix_ss)

        correlations.append(g2_tau / num_photons**2)

    return np.array(correlations)

