"""Computes the two-photon transport through multi-emitter system."""

from typing import Tuple, Dict
import numpy as np
import multi_emitter_hamiltonian
import single_photon_transport

def _flatten(poles: Dict[str, np.ndarray],
             coeffs: Dict[str, np.ndarray]
            ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Flattens a product of three sums of exponentials into a single sum.

    At many points during the computation of the two-photon green's function,
    there are terms of the form:
                    A(t1) B(t2) C(t3)
    where `A(t)`, `B(t)` and `C(t)` are individually expressible as a sum of
    complex exponentials:
            A(t) = sum_n a_n exp(alpha_n t)
            B(t) = sum_n b_n exp(beta_n t)
            C(t) = sum_n c_n exp(gamma_n t)
    The product `A(t1) B(t2) C(t3)` can then be expressed as, on multiplying out
    the summations:
       A(t1) B(t2) C(t3) = sum_n L_n exp(alpha_n' t1 + beta_n' t2 + gamma_n't3)
    where the number of terms in this summation is equal to the product of the
    number of terms in the summations for `A(t)`, `B(t)` and `C(t)`.

    This function computes `L_n`, `alpha_n'`, `beta_n'` and `gamma_n'` from the
    coeffcients `a_n`, `b_n`, `c_n` (specified as a dictionary) and the
    poles `alpha_n`, `beta_n` and `gamma_n` (specified as a dictionary).

    Args:
        poles: A dictionary of one-dimensional arrays corresponding to the
            poles. Note that these arrays need not be of equal sizes.
        coeffs: A dictionary of one-dimensional numpy arrays corresponding to
            the coefficients. Note that this dictionary should have the same
            keys as `poles`, and the arrays should be of lengths consistent
            with the arrays in `poles`.

    Returns:
        The poles `alpha_n'`, `beta_n'` and `gamma_n'` as a dictionary and the
        coefficients `L_n` as an array. Note that the keys in the dictionary for
        the output poles is consistent with the keys in `poles`.
    """
    # Setup all possible combinations of the poles and coefficients. This is
    # done by using meshgrid followed by a flatten command.
    poles_as_grid = np.meshgrid(*[p for _, p in poles.items()])
    poles_as_vec = [p.flatten() for p in poles_as_grid]
    coeffs_as_grid = np.meshgrid(*[c for _, c in coeffs.items()])
    coeffs_as_vec = [c.flatten() for c in coeffs_as_grid]

    # Calculate the product of the coefficients.
    coeffs_prod = np.prod(coeffs_as_vec, axis=0)

    return {k: p for k, p in zip(poles.keys(), poles_as_vec)}, coeffs_prod

def _compute_ft_sum_exp_product(
        poles: Tuple[np.ndarray, np.ndarray, np.ndarray],
        coeffs: Tuple[np.ndarray, np.ndarray, np.ndarray],
        time_diffs: np.ndarray,
        input_freqs: np.ndarray) -> np.ndarray:
    """Compute the fourier transform of a product of sum of exponentials. 
    
    During the computation of the bundled two-photon Green's function, we often
    have to efficiently compute an expression of the following form:
        F(t1, t2, w_0) = exp(1.0j * w_0 * (t1 + t2)) * f(t1 - t2) *
            int_t2'<t2 int_t2'<t1'<t2 [
                g(t2 - t1') * h(t1' - t2') * exp(-1.0j * w_0 * (t1' + t2'))]
                     dt1' dt2' 
    where it is assumed that `t1 > t2` and:
        f(tau) = sum_n f_n exp(-1.0j * lambda_n * t)
        g(tau) = sum_n g_n exp(-1.0j * sigma_n * t)
        h(tau) = sum_n h_n exp(-1.0j * nu_n * t)
    This integral can be analytically evaluated to obtain the result that
    `F` only depends on the time-difference `t1 - t2` and is given by:
        F(tau, w0) = -f(tau) * exp(1.0j * w_0 * tau) * G(w_0) * H(w_0)
    where:
            G(w_0) = sum_n g_n / (sigma_n - 2 * w_0)
            H(w_0) = sum_n h_n / (nu_n - w_0)
    where `tau` is the time-difference `t1 - t2`.

    This function evaluates `F` for a given set of time-differences (`tau`) and
    a given set of input frequencies `w0` provided the poles (`lambda_n`,
    `sigma_n` and `nu_n`) and coefficients (`f_n`, `g_n` and `h_n`) are known.

    Args:
        poles: A tuple of one-dimensional arrays corresponding to the poles
            `lambda_n`, `sigma_n` and `nu_n` (in order). Note that these arrays
            need not be of the same size.
        coeffs: A tuple of one-dimensional arrays corresponding to the
            coefficients `f_n`, `g_n` and `h_n` (in order). Note that these
            arrays need not be of the same size.
        time_diffs: A one-dimensional array with the time-differences at which
             `F` needs to be evaluated.
        input_freqs: An one-dimensional array with the frequencies (`w_0`) at
            which `F` needs to be evaluated.
    """
    # Extract the coefficients `f_n`, `g_n`, `h_n`.
    coeffs_f = coeffs[0]
    coeffs_g = coeffs[1]
    coeffs_h = coeffs[2]

    # Extract the poles `lambda_n`, `sigma_n` and `nu_n`.
    lambdas = poles[0]
    sigmas = poles[1]
    nus = poles[2]

    # Evaluating the complex exponential `exp(1.0j * w_0 * tau)` - this is
    # evaluated to be a `N_w x N_tau` matrix (where `N_w` is the size of
    # `input_freq`, and `N_tau` is the size of `tau`).
    oscillating_exp = np.exp(
        1.0j * input_freqs[:, np.newaxis] @ time_diffs[np.newaxis, :])

    # Evaluating the function `f(tau)`. This is evaluated to obtain a
    # `1 x N_tau` matrix. This done by separately computing
    # `exp(-1.0j * lambda_n * tau)` as a `N_exp x N_tau` matrix (where `N_exp`
    # is the number of exponentials in the summation for `f`), multiplying it by
    # the coefficients `f_n` and summing it over the first axis.
    decaying_exp = np.exp(
        -1.0j * lambdas[:, np.newaxis] @ time_diffs[np.newaxis, :])
    f_tau = np.sum(coeffs_f[:, np.newaxis] * decaying_exp,
                   axis=0, keepdims=True)

    # Evaluating the function `G(w_0)` defined in the docstring. Again, this
    # computation is performed by evaluating the term `1 / (sigma_n - 2 * w_0)`
    # as a `N_exp x N_w` matrix (where `N_exp` is the number of exponentials in
    # the summation for `g`), multiplying it by the coefficients `g_n` and
    # summing it over the first axis.
    freq_dep_coeffs = 1.0 / (
        sigmas[:, np.newaxis] - 2.0 * input_freqs[np.newaxis, :])
    g_omega = np.sum(coeffs_g[:, np.newaxis] * freq_dep_coeffs,
                     axis=0, keepdims=True)

    # Evaluating the function `H(w_0)` defined in the docstring. Again, this
    # computation is performed by evaluating the term `1 / (nu_n - w_0)`
    # as a `N_exp x N_w` matrix (where `N_exp` is the number of exponentials in
    # the summation for `g`), multiplying it by the coefficients `g_n` and
    # summing it over the first axis.
    freq_dep_coeffs = 1.0 / (
        nus[:, np.newaxis] - input_freqs[np.newaxis, :])
    h_omega = np.sum(coeffs_h[:, np.newaxis] * freq_dep_coeffs,
                     axis=0, keepdims=True)

    return -((g_omega * h_omega).T * oscillating_exp) * f_tau


def _two_photon_bundled_response(  # pylint: disable=too-many-locals
        hamiltonian: multi_emitter_hamiltonian.MultiEmitterHamiltonian,
        time_diffs: np.ndarray,
        input_freqs: np.ndarray) -> np.ndarray:
    """Calculate the bundled two-photon correlation function.

    The bundled two-photon response is defined as the two-photon state obtained
    on exciting the multi-emitter system with two continuous-wave photons at an
    input-frequency `w0` and is given by (assuming `t1 > t2`):
        f(t1, t2) = exp(1.0j * w_0 * (t1 + t2)) * int_t2' int_(t1' > t2') [
            G(t1, t2; t1', t2') * exp(-i w_0 * (t1' + t2'))] dt2' dt1'
    and for `t2 >= t1`:
                    f(t2, t1) = f(t1, t2)

    The G(t1, t2; t1', t2') is the two-photon bundled time-domain green's
    function and is given by the following expression:

      < 0; g1 ... gN | a(t1) a(t2) a^dagger(t1') a^dagger(t2') | 0; g1 ... gN>

    where it is assumed that `t1 > t2 > t1' > t2'` (in case this ordering of
    indices is violated, the asymmetric green's function is, by definition, 0).
    Using the definition of `a(t)` and `a^dagger(t)`, this Green's function can
    be expressed in terms of the effective hamiltonian `H_eff` via the following
    equation:
    G(t1, t2; t1', t2') =
        2 * A_1g,1g(t1 - t2) * B_2g,2g(t2 - t1') * A_1g,1g(t1' - t2') +
        sqrt(2) * sum_p [A_1g,0ep'(t1 - t2) * B_1ep,2g(t2 - t1') *
                                                    A_1g,1g(t1' - t2')] +
        sqrt(2) * sum_p [A_1g,1g(t1 - t2) * B_2g,1ep(t2 - t1') *
                                                    A_0ep,1g(t1' - t2')] +
        sum_p sum_q [A_1g,0eq(t1 - t2) * B_1eq,1ep(t2 - t1') *
                                                    A_0ep,1g(t1' - t2')]
    where `A_s,s'(t)` and `B_s,s'(t)` are given by the following expressions:
                A_s,s'(t) = < s | exp(-i H_eff t) | s' >
                B_s,s'(t) = < s | exp(-i H_eff t) | s' >
    Therefore, `G` can be expressed as a sum of four distinct terms,
    corresponding to the four transitions shown below that map the system state
    from `1g` to `1g`:
        * 1g->1g(->)2g->2g(->)1g->1g,
        * 1g->1g(->)2g->1e(->)0e->1g,
        * 1g->0e(->)1e->2g(->)1g->1g,
        * 1g->0e(->)1e->1e(->)0e->1g,
    where `->` corresponds to the evolution under the non-hermitian hamiltonian
    (note that this evolution conserves the number of excitations in the
    mult-emitter system), and `(->)` corresponds to the annihilation or
    creation of a photon in the cavity, which does not conserve the number of
    excitations in the system.

    For computing `A_s,s'(t)`, only the component of the Hamiltonian acting on
    the single-excitation subspace of the multi-emitter system's Hilbert space
    is required and for computing `B_s,s'(t)`, the component of the Hamiltonian
    acting on the two-excitation subspace is required. Assuming that the
    eigen-value decomposition of the effective hamiltonian in the single and
    two-excitation subspace is known:
                    (H_eff)_1 = V_1 Lambda_1 V_1.T
                    (H_eff)_2 = V_2 Lambda_2 V_2.T
    where `V_1` and `V_2` are orthogonal matrices, we can express `A_s,s'(t)`
    and `B_s,s'(t)` as a sum of complex exponentials:
            A_s,s'(t) = sum_n <s, v_1,n> <s', v_1,n> exp(-i lambda_1,n t)
            B_s,s'(t) = sum_n <s, v_2,n> <s', v_2,n> exp(-i lambda_2,n t)
    where `v_1,n` is the nth column of `V_1` and `v_2,n` is the nth column of
    `V_2` and `<v, u> = v.T u`. The pre-factors multiplying the complex
    exponentials are refered to as the coefficients and the slopes of the
    argument of the complex exponentials (`lambda_1,n` and `lambda_2,n` are
    refered to as the poles).

    Using the eigen-decomposition of the two-excitation hamiltonian of the
    multi-emitter system, we can express `G(t1, t2; t1',t2')` as the following
    sum:
        G(t1, t2; t1', t2') = sum_n[ c_n *
            exp(-i lambda_n (t1 - t2)) * exp(-i sigma_n (t2 - t1'))
                * exp(-i * nu_n * (t1' - t2')) ]
    With this expansion, the two-photon response can be expressed as:
        f(t1, t2) = -exp(iw_0 (|t1 - t2|)) * 
            sum_n [exp(-1.0j * lambda_n |t1 - t2|) / (
                (sigma_n - 2 * w_0) * (nu_n - w_0))]

    This function computes `G_s` as a function of `d` for a specified `w0`.

    Args:
        hamiltonian: The hamiltonian of the multi-emitter system, specified as a
            multi_emitter_hamiltonian.MultiEmitterHamiltonian object.
        freq_diffs: The difference between the output frequencies at which to
            evaluate the Green's functions. (Although the green's function has
            two output frequencies, energy conservation requires that the sum of
            the two output frequencies be constant. The output frequencies can
            thus be specified by just the difference between them i.e. instead
            of the output frequencies being an array of tuples of form
            `(w1', w2')`, they can be an array of floats `d = w1' - w2'`.)
        input_freq: The input frequency that the system is excited with.

    Returns:
        The bundled two-photon Green's function.
    """
    # Calculate the single and two-excitation eigen-decompositions.
    evalues_1ex, evectors_1ex = hamiltonian.single_photon_eigen_decomposition()
    evalues_2ex, evectors_2ex = hamiltonian.two_photon_eigen_decomposition()

    # Set up the list of poles (`lambda_n`, `nu_n` and `sigma_n` in the function
    # docstring) and coefficients (`L_n`, `N_n` and `S_n` in the function
    # docstring) needed to evaluate the two-photon bundled green's function.

    # Initialize the lists for both coefficients and the poles. Note that
    # `lambdas` correspond to `lambda_n`, `sigmas` correspond to `sigma_n`, and
    # `nus` correspond to `nu_n` in the function docstring. The lists are
    # initialized with the coefficients and poles corresponding to the
    # transition `1g->1g(->)2g->2g(->)1g-1g`.
    response = 0

    # Calculate the poles and coeficients for the transition
    # `1g->1g(->)2g->2g(->)1g-1g`.
    response += 2 * _compute_ft_sum_exp_product(
        (evalues_1ex, evalues_2ex, evalues_1ex),
        (evectors_1ex[0, :] * evectors_1ex[0, :],
         evectors_2ex[0, :] * evectors_2ex[0, :],
         evectors_1ex[0, :] * evectors_1ex[0, :]),
        time_diffs, input_freqs)

    # Calculate the poles and coefficients corresponding to the transition
    # `1g->1g(->)2g->1e(->)0e->1g`.
    for n in range(hamiltonian.num_emitters):
        # Calculate the poles and coefficients corresponding to the transition
        # with the nth emitter.
        response += np.sqrt(2) * _compute_ft_sum_exp_product(
            (evalues_1ex, evalues_2ex, evalues_1ex),
            (evectors_1ex[0, :] * evectors_1ex[n + 1, :],
             evectors_2ex[0, :] * evectors_2ex[n + 1, :],
             evectors_1ex[0, :] * evectors_1ex[0, :]),
            time_diffs, input_freqs)

    # Calculate the poles and coefficients corresponding to the transition
    # `1g->0e(->)1e->2g(->)1g->1g`.
    for n in range(hamiltonian.num_emitters):
        # Calculate the poles an coefficients correponding to the transition
        # with the nth emitter.
        response += np.sqrt(2) * _compute_ft_sum_exp_product(
            (evalues_1ex, evalues_2ex, evalues_1ex),
            (evectors_1ex[0, :] * evectors_1ex[0, :],
             evectors_2ex[0, :] * evectors_2ex[n + 1, :],
             evectors_1ex[0, :] * evectors_1ex[n + 1, :]),
            time_diffs, input_freqs)        

    # Calculate the poles and coefficients corresponding to the transition
    # `1g->0e(->)1e->1e(->)0e->1g`. Since there are two stages in this
    # transition that the emitters are involved, it is needed to iterate over
    # all possible combinations of emitters.
    for n in range(hamiltonian.num_emitters):
        for m in range(hamiltonian.num_emitters):
            # Calculate the poles and coefficients corresponding to the
            # transition involving the nth and mth emitter.
            response += _compute_ft_sum_exp_product(
                (evalues_1ex, evalues_2ex, evalues_1ex),
                (evectors_1ex[0, :] * evectors_1ex[n + 1, :],
                 evectors_2ex[n + 1, :] * evectors_2ex[m + 1, :],
                 evectors_1ex[0, :] * evectors_1ex[m + 1, :]),
                time_diffs, input_freqs)    

    return response


def _two_photon_consec_response(
        hamiltonian: multi_emitter_hamiltonian.MultiEmitterHamiltonian,
        time_diffs: np.ndarray,
        input_freqs: np.ndarray) -> np.ndarray:
    """Calculate the two-photon state arising from consecutive scattering.

    Args:
        hamiltonian: The hamiltonian of the multi-emitter system specified as
            a multi_emitter_hamiltonian.MultiEmitterHamiltonian object.
        time_diffs: The time-differences in the output state at which to
            evaluate the response. This function assumes that all elements in
            this array is positive.
        input_freqs: The frequencies at which the system is excited.
    """
    # Calculate the eigen-value decomposition for the single excitation part of
    # the effective hamiltonian.
    evalues, evectors = hamiltonian.single_photon_eigen_decomposition()

    # Compute the coefficients in the complex-exponential expansion of the
    # single-photon scattering.
    coeffs = evectors[0, :]**2

    # Calculate the expression
    #                   s_n / (lambda_n - w_0)
    # where `s_n` refer to the coefficients in the array `coeffs`, `lambda_n`
    # refer to the eigen-values in the array `evalues` and `w_0` refer to the
    # input frequency in the array `input_freqs`.
    freq_dep_coeffs = coeffs[:, np.newaxis] / (
        evalues[:, np.newaxis] - input_freqs[np.newaxis, :])

    # Calculate the decaying exponential given by the following expression
    #           exp(-1.0j * lambda_n * tau)
    decaying_exp = np.exp(
        -1.0j * evalues[:, np.newaxis] @ time_diffs[np.newaxis, :])

    # Calculate the decaying exponential given by the following expression
    #           exp(1.0j * w_0 * tau)
    oscillating_exp = np.exp(
        1.0j * input_freqs[:, np.newaxis] @ time_diffs[np.newaxis, :])

    # Calculate the response function. The response at `tau` to infinity is
    # computed first.
    inf_time_resp = np.sum(freq_dep_coeffs, axis=0, keepdims=True)
    
    # Computing the response function as a function of time.
    return inf_time_resp.T * (
        (freq_dep_coeffs.T @ decaying_exp) * oscillating_exp - inf_time_resp.T)


def two_photon_correlation(
        hamiltonian: multi_emitter_hamiltonian.MultiEmitterHamiltonian,
        time_diffs: np.ndarray,
        input_freqs: np.ndarray) -> np.ndarray:
    """Calculate the two photon correlation.

    Args:
        hamiltonian: The hamiltonian of the multi-emitter system specified as
            a multi_emitter_hamiltonian.MultiEmitterHamiltonian object.
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

    # Calculate the two-photon response function.
    bundled_response = _two_photon_bundled_response(hamiltonian,
                                                    np.abs(time_diffs),
                                                    input_freqs)
    consec_response = _two_photon_consec_response(hamiltonian,
                                                  np.abs(time_diffs),
                                                  input_freqs)

    two_ph_response = bundled_response + consec_response

    # Calculate the transmission (this is for normalizing the correlation).
    single_ph_tran = single_photon_transport.single_photon_transmission(
        hamiltonian, input_freqs)

    # Calculate the two-photon correlation function.
    two_ph_corr = hamiltonian.kappa_in**2 * hamiltonian.kappa_out**2 * np.abs(
        two_ph_response)**2 / np.abs(single_ph_tran[:, np.newaxis])**4

    return two_ph_corr
