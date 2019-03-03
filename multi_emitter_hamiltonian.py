"""Setup Hamiltonian and its eigen-decomposition for Multi-Emitter-CQED system.
"""
from typing import List, Tuple
import numpy as np


def _compute_eigen_decomposition(
        matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the eigen-value decomposition of the specified matrix.

    This function wraps np.linalg.eig for computing the eigen-values and
    eigen-vectors of a specified matrix. For a complex matrix, the eigen-vectors
    computed by np.linalg.eig are normalized to unity according to the inner
    product <u, v> = u.dagger v. This function uses np.linalg.eig to compute
    the eigen-values and eigen-vectors of the specified matrix, but normalizes
    the eigen-vectors to unity according to the inner product <u, v> = u.T v.
    Such a normalization is required, for e.g., while computing the eigen-value
    decomposition of a complex symmetric matrix.

    Args:
        matrix: The matrix whose eigen-decomposition is to be computed.

    Returns:
        The eigen-decomposition of `matrix` as a 1D array of eigen-values and a
        2D array whose columns are the eigen-vectors of `matrix` normalized to
        unity according to the inner product <u, v> = u.T v.
    """
    # Computing the eigen-values and eigen-vectors of `matrix`.
    eig_values, eig_vectors = np.linalg.eig(matrix)

    # Normalizing the computed eigen-vectors appropriately.
    eig_vectors /= np.sqrt(np.sum(eig_vectors**2, axis=0, keepdims=True))

    return eig_values, eig_vectors


class MultiEmitterHamiltonian:  # pylint: disable=too-many-instance-attributes
    """Defines a Multi-emitter Hamiltonian.

    A multi-emitter hamiltonian describes the quantum dynamics of a cavity with
    emitters coupled to it. The cavity-emitter system can be modelled as an
    open quantum system to account for the losses in the cavity and the emitters.
    In the scattering matrix picture, to analyze scattering of a low-power
    coherent pulse through a multi-emitter system, it is necessary to compute
    the eigen-values and eigen-vectors of the single- and two-photon components
    of the multi-emitter hamiltonian. This class implements the calculation of
    these eigen-values and eigen-vectors, which can then be used to compute the
    single- and two-photon scattering matrices for the multi-emitter system.
    Additionally, since the computation of eigen-values and eigen-vectors is an
    expensive operation for systems with large number of emitters, this class
    also implements caching of the eigen-values and eigen-vectors to avoid
    repeated computations.
    """
    def __init__(self,  # pylint: disable=too-many-arguments
                 delta_cavity: float,
                 kappa_in: float,
                 kappa_out: float,
                 kappa_loss: float,
                 delta_emitters: List[float],
                 decay_emitters: List[float],
                 cavity_emitter_couplings: List[float]) -> None:
        """Creates a new MultiEmitterHamiltonian object.

        Args:
            delta_cavity: The detuning of the cavity from the reference
                frequency.
            kappa_in: The decay rate of the cavity into the input waveguide.
            kappa_out: The decay rate of the cavity into the output
                waveguide.
            kappa_loss: The decay rate of the cavity into the radiative losses.
            delta_emitters: The detuning of the emitters from the reference
                frequency. Note that this converted to a 1D numpy array inside
                the `__init__` function.
            decay_cavity: The decay rate of the waveguide into the coupled
                waveguide.
            decay_emitters: The decay rate of the emitters into their loss
                channels. Note that this is converted to a 1D numpy array inside
                the `__init__` function.
            cavity_emitter_coupling: The coupling constant between the cavity
                mode and the emitters. Note that this is converted into a 1D
                numpy array inside the `__init__` function.

        Raises:
            ValueError: If the lengths of `delta_emitters`, `decay_emitters` and
                `cavity_emitter_couplings` are not all equal.
        """
        # Validate the emitter parameters.
        if (len(delta_emitters) != len(decay_emitters) or
                len(delta_emitters) != len(cavity_emitter_couplings)):
            raise ValueError("The input lists `delta_emitters`, "
                             "`decay_emitters` and `cavity_emitter_couplings` "
                             "should have the same length, got "
                             "{}, {} and {}".format(
                                 len(delta_emitters),
                                 len(decay_emitters),
                                 len(cavity_emitter_couplings)))

        # Record the various parameters of the Hamiltonian as the object's
        # private variables.
        self._delta_cavity = delta_cavity
        self._kappa_in = kappa_in
        self._kappa_out = kappa_out
        self._kappa_loss = kappa_loss
        self._delta_emitters = np.array(delta_emitters)
        self._decay_emitters = np.array(decay_emitters)
        self._cavity_emitter_couplings = np.array(cavity_emitter_couplings)

        # Initialize private variables for the caching the eigen-decompositions
        # of the single- and two-photon components of the Hamiltonian.
        self._single_ex_eigen_values = None
        self._single_ex_eigen_vectors = None
        self._two_ex_eigen_values = None
        self._two_ex_eigen_vectors = None

    @property
    def num_emitters(self) -> int:
        """Returns the number of emitters in the multi-emitter system."""

        return self._delta_emitters.size

    @property
    def kappa_in(self) -> float:
        """Returns the decay rate of the cavity into the input waveguide."""
        return self._kappa_in

    @property
    def kappa_out(self) -> float:
        """Returns the decay rate of the cavity into the output waveguide."""
        return self._kappa_out

    @property
    def delta_cavity(self) -> float:
        """Returns the detuning of the cavity from the reference frequency."""
        return self._delta_cavity

    @property
    def delta_emitters(self) -> List[float]:
        """Returns the detuning of the emitters from the reference frequency."""
        return self._delta_emitters.tolist()

    @property
    def cavity_emitter_couplings(self) -> List[float]:
        """Returns the coupling constants between the emitters and cavity."""
        return self._cavity_emitter_couplings.tolist()

    @property
    def kappa_loss(self) -> float:
        return self._kappa_loss

    @property
    def decay_emitters(self) -> List[float]:
        return self._decay_emitters.tolist()

    def _single_photon_matrix(self) -> np.ndarray:
        """Computes the matrix corresponding to the single-photon Hamiltonian.

        Returns:
          A np.ndarray corresponding to the single-photon Hamiltonian matrix.
        """
        # Total decay rate of the cavity.
        decay_cavity = self._kappa_in + self._kappa_out + self._kappa_loss

        # Construct the single-emitter component of the Hamiltonian.
        hamiltonian = np.block(
            [[self._delta_cavity - 0.5j * decay_cavity,
              self._cavity_emitter_couplings[np.newaxis, :]],
             [self._cavity_emitter_couplings[:, np.newaxis],
              np.diag(self._delta_emitters - 0.5j * self._decay_emitters)]])

        return hamiltonian

    def single_photon_eigen_decomposition(  # pylint: disable=invalid-name
            self) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the eigen-decomposition of the single photon Hamiltonian.

        Note that this function caches the eigen-decomposition in the private
        variables `_single_eigen_values` and `_single_eigen_vectors`. This
        allows for avoiding repeated computations of the eigen-value
        decomposition, since this can be an expensive operation if the number of
        emitters is large.

        Returns:
          The eigen-values and eigen-vectors of the single-emitter component of
          the effective Hamiltonian. The eigen-values are returned as a 1D array
          of complex numbers, and the eigen-vectors are returned as a 2D array
          with columns being the eigen-vectors.
        """
        if (self._single_ex_eigen_values is None or
                self._single_ex_eigen_vectors is None):
            # Compute the matrix corresponding to the single-emitter effective
            # Hamilonian.
            hamiltonian = self._single_photon_matrix()
            eigen_values, eigen_vectors = _compute_eigen_decomposition(
                hamiltonian)
            self._single_ex_eigen_values = eigen_values
            self._single_ex_eigen_vectors = eigen_vectors

        return self._single_ex_eigen_values, self._single_ex_eigen_vectors

    def _two_photon_matrix(self) -> np.ndarray:
        """Compute the matrix corresponding to the two-photon Hamiltonian.

        The two-photon Hamiltonian can be setup as the following matrix:

                    [H_2g,2g    H_2g,1e         0  ]
                    [H_2g,1e.T  H_1e,1e     H_1e,0e]
                    [  0        H_1e,0e.T   H_0e,0e]

        where `H_2g,2g` is a 1 x 1 matrix given by:
                    H_2g,2g = < 2g | H_eff | 2g >
        This matrix corresponds to the effective hamiltonian mapping the cavity
        being in a two photon state, and all emitters being in the ground state,
        to the same state.

        `H_2g,1e` is a row matrix given by:
                    (H_2g,1e)_n = < 2g | H_eff | 1g, 1e_n >
        Note that `H_1e,2g` is a column matrix with the same elements as the row
        matrix `H_2g,1e`. These corresponds to the effective hamiltonian mapping
        the cavity being in a two photon state, and all the emitters being in
        the ground state, to the cavity being in a single photon state and one
        of the emitters being in the excited state.

        `H_1e,1e` is a `N x N` square matrix (where `N` is the number of
        emitters) given by:
                    (H_1e,1e)_m,n = < 1e_n | H_eff | 1e_m>
        This corresponds to the effective hamiltonain mapping the cavity being
        in the single photon state and the m-th emitter being in the excited
        state to the cavity being in the single-photon state and the n-th
        emitter being in the excited state.

        `H_0e,1e` is a `M x N` matrix (where `M = N * (N - 1)/2`) which maps
        the cavity being in the ground state and two emitters being in the
        excited state, to the cavity being in the single-photon state and
        one of the emitters being in the excited state.

        `H_0e,0e` is a `M x M` matrix (where `M = N * (N - 1)/2`) which maps the
        cavity being in the ground state and two emitters being in the excited
        state to the cavity being in the ground state and two emitters being in
        excited state. Note that since the emitters only interact with each
        other through the cavity, this matrix is a diagonal matrix.

        Returns:
            A np.ndarray corresponding to the two-photon effective Hamiltonian
            matrix.
        """
        # Total decay rate of the cavity.
        decay_cavity = self._kappa_in + self._kappa_out + self._kappa_loss

        # Calculate the cavity pole - this is the effective complex frequency of
        # the cavity that appears in the effective hamiltonian.
        cav_pole = self._delta_cavity - 0.5j * decay_cavity

        # Calculate the emitter pole - this is the effective complex frequency
        # of the emitter that appears in the effective hamiltonian.
        em_pole = self._delta_emitters - 0.5j * self._decay_emitters

        # Calculate the matrix element of the effective hamiltonian in between
        # the cavity with a two photon state, and all emitters being in the
        # ground state (This corresponds to `H_2g,2g` defined above).
        hamil_2g2g = 2 * cav_pole

        # Calculate the matrix `H_2g,1e`.
        hamil_2g1e = np.sqrt(2) * self._cavity_emitter_couplings[np.newaxis, :]

        # Calculate the matrix `H_1e,1e`. This is a diagonal matrix.
        hamil_1e1e = np.diag(cav_pole + em_pole)

        # Calculate the matrices which involve two excited emitters and no
        # photons in the cavity (these includes `H_1e,0e`, `H_0e,1e` and
        # `H_0e,0e`).

        # Number of possible two-emitter excitations.
        num_two_em = (self.num_emitters * (self.num_emitters - 1)) // 2

        # Setup the indices corresponding to two-emitters being in the excited
        # state. A total of `N * (N - 1) / 2` different basis functions can be
        # formed by exciting two emitters - each of these basis functions can be
        # assigned an index between 0 and `N * (N - 1) / 2`. `two_em_indices`
        # stores the mapping of this basis index to the individual indices of
        # the two emitters. In particular, the emitter indices for a basis index
        # `i` is given by `two_em_indices[0][i]` and `two_em_indices[1][i]`.
        two_em_indices = np.triu_indices(self.num_emitters,
                                         k=1)

        # Calculate the matrix `H_1e,0e`, which has `N` rows, and
        # `N * (N - 1) / 2` columns, where `N` is the number of emitters. Note
        # that an element in this matrix is of the form:
        #   <1, e_n | H_eff | 0, e_p, e_q> = g_p delta(q, n) + g_q delta(p, n)
        # where `delta` is the kronecker delta function. This corresponds to the
        # element at the `nth` row and `kth` column, where `k` is the index of
        # the tuple `(p, q)` in `two_em_indices` (i.e. `k` is the integer
        # between 0 and `N * (N - 1) / 2` such that `two_em_indices[0][k] == p`
        # and `two_em_indices[1][k] == q`). To setup the matrix `H_1e,0e`, at
        # each column `k`, we first extract the indices `(p, q)` using
        # `two_em_indices`, and then add `g_q` at the `pth` row, and `g_p` at
        # the `qth` row.
        hamil_1e0e = np.zeros((self.num_emitters, num_two_em),
                              dtype=complex)
        hamil_1e0e[two_em_indices[0],
                   np.arange(num_two_em)] += self._cavity_emitter_couplings[
                       two_em_indices[1]]
        hamil_1e0e[two_em_indices[1],
                   np.arange(num_two_em)] += self._cavity_emitter_couplings[
                       two_em_indices[0]]

        # Calculate the matrix `H_0e,0e`. This is simply a diagonal matrix with
        # the `(k, k)` element being given by:
        #      <0, e_p, e_q | H | 0, e_p, e_q>
        #       = (delta_p + delta_q) - 0.5j * (gamma_p + gamma_q)
        # where `p = two_em_indices[0][k]` and `q = two_em_indices[1][k]`.
        hamil_0e0e = np.diag(
            em_pole[two_em_indices[0]] + em_pole[two_em_indices[1]])

        # Assemble the Hamiltonian.
        zero_matrix = np.zeros((1, num_two_em), dtype=complex)
        hamiltonian = np.block(
            [[hamil_2g2g, hamil_2g1e, zero_matrix],
             [hamil_2g1e.T, hamil_1e1e, hamil_1e0e],
             [zero_matrix.T, hamil_1e0e.T, hamil_0e0e]])

        return hamiltonian

    def two_photon_eigen_decomposition(
            self) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the eigen-decomposition of the two photon Hamiltonian.

        Note that this function caches the eigen-decomposition in the private
        variables `_two_eigen_values` and `_two_eigen_vectors`. This allows for
        avoiding repeated computations of the eigen-value decomposition, since
        this can be an expensive operation if the number of emitters is large.

        Returns:
            The eigen-values and eigen-vectors of the two-emiiter component of
            the Hamiltonian.
        """
        if (self._two_ex_eigen_values is None or
                self._two_ex_eigen_vectors is None):
            # Compute the matrix corresponding to the single-emitter effective
            # Hamilonian.
            hamiltonian = self._two_photon_matrix()
            eigen_values, eigen_vectors = _compute_eigen_decomposition(
                hamiltonian)
            self._two_ex_eigen_values = eigen_values
            self._two_ex_eigen_vectors = eigen_vectors

        return self._two_ex_eigen_values, self._two_ex_eigen_vectors
