import unittest
import geodesic as geo
import numpy as np

class TestAlgo(unittest.TestCase):

    def setUp(self) -> None:
        self.vect1 = np.array([0.0,0.0,0.9])
        self.vect2 = np.array([0.9,0.0,0.0])

    def tearDown(self) -> None:
        return super().tearDown()


    def test_geodesic(self):
        
        tau = 0.2
        georesult = geo.geodesic(tau,self.vect1,self.vect2)
        self.assertEqual(georesult[0,0].real,0.9092521918410296)
        self.assertEqual(georesult[0,0].imag,0.0)
        self.assertEqual(georesult[0,1].real,0.12487213325637833)
        self.assertEqual(georesult[0,1].imag,0.0)

        self.assertAlmostEqual(georesult[1,0].real,0.12487213325637833)
        self.assertAlmostEqual(georesult[1,0].imag,0.0)
        self.assertAlmostEqual(georesult[1,1].real,0.09074781)
        self.assertAlmostEqual(georesult[1,1].imag,0.0)

        self.assertAlmostEqual(np.trace(georesult).imag,0.0)
        self.assertAlmostEqual(np.trace(georesult).real,1.0)


        tau = -0.2
        georesult = geo.geodesic(tau,self.vect1,self.vect2)

        self.assertEqual(georesult[0,0].real,0.9567036024784532)
        self.assertEqual(georesult[0,0].imag,0.0)
        self.assertEqual(georesult[0,1].real,-0.12487213325637835)
        self.assertEqual(georesult[0,1].imag,0.0)

        self.assertAlmostEqual(georesult[1,0].real,-0.12487213325637837)
        self.assertAlmostEqual(georesult[1,0].imag,0.0)
        self.assertAlmostEqual(georesult[1,1].real,0.043296397521546647)
        self.assertAlmostEqual(georesult[1,1].imag,0.0)

        self.assertAlmostEqual(np.trace(georesult).imag,0.0)
        self.assertAlmostEqual(np.trace(georesult).real,1.0)

        with self.assertRaises(ValueError):
            geo.geodesic(tau,self.vect1,np.array([1.0,0.0,0.0]))


if __name__ == '__main__':
    unittest.main()