import unittest
import numpy as np

import geodesic as geo
import muk as muk
from fidelity import fidelity
from controlSetup3 import control1setup3


class TestAlgo(unittest.TestCase):
    def setUp(self) -> None:
        self.vect1 = np.array([0.0, 0.0, 0.9])
        self.vect2 = np.array([0.9, 0.0, 0.0])
        self.tau = 0.2

    def tearDown(self) -> None:
        return super().tearDown()

    def test_geodesic(self):
        georesult = geo.geodesic(self.tau, self.vect1, self.vect2)
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

        georesult = geo.geodesic(-self.tau, self.vect1, self.vect2)

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
            geo.geodesic(self.tau, self.vect1, np.array([1.0, 0.0, 0.0]))

    def test_muk(self):
        mukresult = muk.muk(self.vect1, self.vect2)
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

        for i in range(Nmax):
            self.assertAlmostEqual(solTiempo[i], tiempolists[i])


if __name__ == "__main__":
    unittest.main()
