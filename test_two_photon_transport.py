"""Tests for two_photon_transport.py"""

import unittest

from matplotlib import pyplot as plt
import numpy as np

import multi_emitter_hamiltonian
import two_photon_transport

class TestTwoPhotonCorrelation(unittest.TestCase):

	def test_two_photon_correlation(self):
		hamiltonian = multi_emitter_hamiltonian.MultiEmitterHamiltonian(
			delta_cavity=0.0,
			kappa_in=0.5,
			kappa_out=0.5,
			kappa_loss=0.0,
			delta_emitters=[0.0, 0.0],
			decay_emitters=[0.0, 0.0],
			cavity_emitter_couplings=[0.5, 0.5])

		input_freqs = np.linspace(-10, 10, 10)
		time_diffs = np.linspace(0, 100, 1000)
		two_ph_corr = two_photon_transport.two_photon_correlation(
			hamiltonian,
			time_diffs,
			input_freqs)
		print(two_ph_corr)
		print(two_ph_corr.shape)

if __name__ == "__main__":
    unittest.main()