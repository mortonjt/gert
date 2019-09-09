import os
import random
import numpy as np
from util import get_context, GeneInterval
import unittest


class TestContext(unittest.TestCase):

    def setUp(self):
        self.genes = [
            GeneInterval(1, 2, 'G', 1),
            GeneInterval(3, 4, 'H', 1),
            GeneInterval(4, 5, 'I', 1),
            GeneInterval(4, 5, 'K', -1),
            GeneInterval(13, 14, 'J', -1),
            GeneInterval(15, 16, 'L', -1)
        ]

    def test_context(self):
        res = get_context(self.genes, 1, window_size=3)
        exp = GeneInterval(1, 2, 'G', 1)
        self.assertEqual(res[0], exp)

        exp = GeneInterval(4, 5, 'I', 1)
        self.assertEqual(res[1], exp)


    def test_boundaries(self):
        res = get_context(self.genes, 0, window_size=3)
        g1 = GeneInterval(1, 2, 'G', 1)
        n1 = GeneInterval(3, 4, 'H', 1)
        n2 = GeneInterval(4, 5, 'I', 1)

        self.assertNotIn(g1, res)
        self.assertIn(n1, res)
        self.assertIn(n2, res)

        res = get_context(self.genes, 5, window_size=3)
        n1 = GeneInterval(13, 14, 'J', -1)
        g1 = GeneInterval(15, 16, 'L', -1)

        self.assertNotIn(g1, res)
        self.assertIn(n1, res)

    def test_get_operon(self):
        records = [
            GeneInterval(
                268, 469,
                ('MVLRQLSRQASVRVSKTWTGTKRRAQRIFIFILELLLEFCRGEDSVDGKNKSTTALPA'
                 'VKDSVKDS'), 1
            ),
            GeneInterval(
                504, 1560,
                ('MGAALALLGDLVASVSEAAAATGFSVAEIAAGEAAAAIEVQIAS'
                 'LATVEGITSTSEAIAAIGLTPQTYAVIAGAPGAIAGFAALIQTVTGISSLAQVGYRFF'
                 'SDWDHKVSTVGLYQQSGMALELFNPDEYYDILFPGVNTFVNNIQYLDPRHWGPSLFAT'
                 'ISQALWHVIRDDIPAITSQELQRRTERFFRDSLARFLEETTWTIVNAPVNFYNYIQDY'
                 'YSNLSPIRPSMVRQVAEREGTQVNFGHTYRIDDADSIQEVTQRMELRNKENVHSGEFI'
                 'EKTIAPGGANQRTAPQWMLPLLLGLYGTVTPALEAYEDGPNQKKRRVSRGSSQKAKGT'
                 'RASAKTTNKRRSRSSRS'), 1
            ),
            GeneInterval(
                861, 1560,
                ('MALELFNPDEYYDILFPGVNTFVNNIQYLDPRHWGPSLFATISQ'
                 'ALWHVIRDDIPAITSQELQRRTERFFRDSLARFLEETTWTIVNAPVNFYNYIQDYYSN'
                 'LSPIRPSMVRQVAEREGTQVNFGHTYRIDDADSIQEVTQRMELRNKENVHSGEFIEKT'
                 'IAPGGANQRTAPQWMLPLLLGLYGTVTPALEAYEDGPNQKKRRVSRGSSQKAKGTRAS'
                 'AKTTNKRRSRSSRS'), 1
            )
        ]
        operon = get_context(records, 0, 100)
        self.assertGreater(len(operon), 0)

        operon = get_context(records, 1, 100)
        self.assertGreater(len(operon), 0)


if __name__ == '__main__':
    unittest.main()
