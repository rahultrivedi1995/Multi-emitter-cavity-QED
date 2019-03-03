"""Tests for single_photon_transport.py"""

import unittest

from matplotlib import pyplot as plt
import numpy as np

import multi_emitter_hamiltonian
import single_photon_transport


class TestSinglePhotonTransmission(unittest.TestCase):

    def test_single_photon_transmission(self):
        hamiltonian = multi_emitter_hamiltonian.MultiEmitterHamiltonian(
            delta_cavity=0.0,
            kappa_in=0.5,
            kappa_out=0.5,
            kappa_loss=0.0,
            delta_emitters=[0.0, 0.0],
            decay_emitters=[0.0, 0.0],
            cavity_emitter_couplings=[0.5, 0.5])
        frequencies = np.linspace(-10, 10, 1000).tolist()
        transmission = photon_transport.single_photon_transmission(
            hamiltonian,
            frequencies)
        plt.plot(np.abs(transmission)**2)
        plt.show()

    def test_raise_value_error_empty_frequencies(self):
        with self.assertRaisesRegex(ValueError,
                                    "Got empty list for frequencies"):
            hamiltonian = multi_emitter_hamiltonian.MultiEmitterHamiltonian(
                delta_cavity=0.0,
                kappa_in=0.5,
                kappa_out=0.5,
                kappa_loss=1.0,
                delta_emitters=[0.0, 0.0],
                decay_emitters=[0.5, 0.5],
                cavity_emitter_couplings=[0.5, 0.5])

            photon_transport.single_photon_transmission(hamiltonian, [])


if __name__ == "__main__":
    unittest.main()
