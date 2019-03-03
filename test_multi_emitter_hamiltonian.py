"""Unittests for multi_emitter_hamiltonian.py"""

import unittest
import multi_emitter_hamiltonian


class TestMultiEmitterHamiltonian(unittest.TestCase):

    def test_single_photon_eigen_decomposition(self):
        hamiltonian = multi_emitter_hamiltonian.MultiEmitterHamiltonian(
            delta_cavity=0.0,
            kappa_in=0.5,
            kappa_out=0.5,
            kappa_loss=1.0,
            delta_emitters=[0.0, 0.0],
            decay_emitters=[0.5, 0.5],
            cavity_emitter_couplings=[0.5, 0.5])
        eigen_values, eigen_vectors = hamiltonian.single_photon_eigen_decomposition()
 
    def test_two_photon_eigen_decomposition(self):
        hamiltonian = multi_emitter_hamiltonian.MultiEmitterHamiltonian(
            delta_cavity=0.0,
            kappa_in=0.5,
            kappa_out=0.5,
            kappa_loss=1.0,
            delta_emitters=[0.0] * 2,
            decay_emitters=[0.5] * 2,
            cavity_emitter_couplings=[0.5] * 2)
        print(hamiltonian._two_photon_matrix().shape)
        eigen_values, eigen_vectors = hamiltonian.two_photon_eigen_decomposition()


if __name__ == "__main__":
    unittest.main()