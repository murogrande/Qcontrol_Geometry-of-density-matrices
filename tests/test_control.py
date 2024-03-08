import unittest
import numpy as np

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

# sys.path.append("/home/mauro/Documents/personal/sphener-code/Qcontrol_Geometry-of-density-matrices")

from contrlwgeo import geodesic
from contrlwgeo import muk
from contrlwgeo import fidelity
from contrlwgeo import control1setup3
from contrlwgeo import get_time_fidelity


class TestAlgo(unittest.TestCase):
    def setUp(self) -> None:
        self.vect1 = np.array([0.0, 0.0, 0.9])
        self.vect2 = np.array([0.9, 0.0, 0.0])
        self.tau = 0.2

    def tearDown(self) -> None:
        return super().tearDown()

    def test_geodesic(self):
        georesult = geodesic(self.tau, self.vect1, self.vect2)
        self.assertEqual(georesult[0, 0].real, 0.9092521918410296)
        self.assertEqual(georesult[0, 0].imag, 0.0)
        self.assertEqual(georesult[0, 1].real, 0.12487213325637833)
        self.assertEqual(georesult[0, 1].imag, 0.0)

        self.assertAlmostEqual(georesult[1, 0].real, 0.12487213325637833)
        self.assertAlmostEqual(georesult[1, 0].imag, 0.0)
        self.assertAlmostEqual(georesult[1, 1].real, 0.09074781)
        self.assertAlmostEqual(georesult[1, 1].imag, 0.0)

        self.assertAlmostEqual(np.trace(georesult).imag, 0.0)
        self.assertAlmostEqual(np.trace(georesult).real, 1.0)

        georesult = geodesic(-self.tau, self.vect1, self.vect2)

        self.assertEqual(georesult[0, 0].real, 0.9567036024784532)
        self.assertEqual(georesult[0, 0].imag, 0.0)
        self.assertEqual(georesult[0, 1].real, -0.12487213325637835)
        self.assertEqual(georesult[0, 1].imag, 0.0)

        self.assertAlmostEqual(georesult[1, 0].real, -0.12487213325637837)
        self.assertAlmostEqual(georesult[1, 0].imag, 0.0)
        self.assertAlmostEqual(georesult[1, 1].real, 0.043296397521546647)
        self.assertAlmostEqual(georesult[1, 1].imag, 0.0)

        self.assertAlmostEqual(np.trace(georesult).imag, 0.0)
        self.assertAlmostEqual(np.trace(georesult).real, 1.0)

        with self.assertRaises(ValueError):
            geodesic(self.tau, self.vect1, np.array([1.0, 0.0, 0.0]))

    def test_muk(self):
        mukresult = muk(self.vect1, self.vect2)
        self.assertAlmostEqual(mukresult[0], 0.5833833511969478)
        self.assertAlmostEqual(mukresult[1], 0.0)
        self.assertAlmostEqual(mukresult[2], -0.5833833511969478)

    def test_fidelity(self):
        fidelityres1 = fidelity(self.vect1, self.vect2)
        fidelityres2 = fidelity(self.vect1, self.vect1)
        fidelityres3 = fidelity(self.vect2, self.vect2)

        self.assertAlmostEqual(fidelityres1, 0.595)
        self.assertAlmostEqual(fidelityres2, 1.0)
        self.assertAlmostEqual(fidelityres3, 1.0)

    def test_controlSetUp3(self):
        qsri = 1 / np.sqrt(3) * np.array([0.7, 0.8, 0.8])
        qssf = 1 / np.sqrt(3) * np.array([0.2, 0.9, 0.0])
        w0 = 5.0
        gamma_0 = 0.01
        gamma_c = 10
        Nmax = 10
        deltat = 0.0030

        solEstadolist = [
            ([0.40414519, 0.46188022, 0.46188022]),
            ([0.37961291, 0.50613312, 0.43551888]),
            ([0.35305522, 0.54729677, 0.40708988]),
            ([0.32489164, 0.58518751, 0.37663321]),
            ([0.29559258, 0.61965519, 0.34416402]),
            ([0.26569362, 0.65058118, 0.3096603]),
            ([0.23582308, 0.67787255, 0.27304614]),
            ([0.20675607, 0.70144832, 0.23417317]),
            ([0.17952425, 0.72120693, 0.19282441]),
            ([0.15563593, 0.73694718, 0.14889171]),
        ]

        solTiempo = [
            0.0,
            0.003,
            0.006,
            0.009000000000000001,
            0.012,
            0.015,
            0.018,
            0.020999999999999998,
            0.023999999999999997,
            0.026999999999999996,
        ]

        solVeclanbdas = [
            61.2742005627058,
            66.1754754389770,
            72.2489523163035,
            79.9079104001009,
            89.7922670200548,
            102.939219395469,
            121.118307692969,
            147.492623432852,
            187.666770093428,
            248.164669292313,
        ]

        estadoslist, tiempolists, solution, vec_lambda = control1setup3(
            qsri,
            qssf,
            Nmax=Nmax,
            w0=w0,
            gamma_0=gamma_0,
            gamma_c=gamma_c,
            deltat=deltat,
        )

        for i in range(Nmax):
            self.assertAlmostEqual(list(estadoslist[i])[0], solEstadolist[i][0])
            self.assertAlmostEqual(list(estadoslist[i])[1], solEstadolist[i][1])
            self.assertAlmostEqual(list(estadoslist[i])[2], solEstadolist[i][2])
        print(tiempolists)
        for i in range(Nmax):
            self.assertAlmostEqual(solTiempo[i], tiempolists[i])

        for i in range(Nmax):
            self.assertAlmostEqual(solVeclanbdas[i], vec_lambda[i])

    def test_getTimeFidelity(self):
        qsri = 1 / np.sqrt(3) * np.array([0.7, 0.8, 0.8])
        qssf = 1 / np.sqrt(3) * np.array([0.2, 0.9, 0.0])
        Nmax = 40
        imax = 7
        deltat = 0.0030

        solFinalEstados = [
            ([0.11515949, 0.63900878, 0.02595626]),
            ([0.11562241, 0.63325575, 0.02445069]),
            ([0.11602751, 0.62753868, 0.0229844]),
            ([0.11637749, 0.62185806, 0.02155605]),
            ([0.1166751, 0.61621426, 0.02016439]),
            ([0.11692304, 0.61060763, 0.01880823]),
            ([0.11712389, 0.60503845, 0.01748644]),
            ([0.11728012, 0.59950694, 0.01619793]),
            ([0.11739401, 0.59401331, 0.01494169]),
            ([0.11746776, 0.58855771, 0.01371674]),
        ]

        estadoslist, tiempolists, solution, _ = control1setup3(
            qsri, qssf, Nmax=Nmax, deltat=deltat
        )
        finalestados, finaltiempotot, list_lambda_time = get_time_fidelity(
            estadoslist, tiempolists, solution, imax, qssf
        )

        for i in range(30, Nmax):
            self.assertAlmostEqual(list(finalestados[i])[0], solFinalEstados[i - 30][0])
            self.assertAlmostEqual(list(finalestados[i])[1], solFinalEstados[i - 30][1])
            self.assertAlmostEqual(list(finalestados[i])[2], solFinalEstados[i - 30][2])


if __name__ == "__main__":
    unittest.main()
