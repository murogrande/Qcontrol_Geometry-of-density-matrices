import unittest
import numpy as np

import geodesic as geo
import muk as muk
from fidelity import fidelity


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


if __name__ == "__main__":
    unittest.main()
