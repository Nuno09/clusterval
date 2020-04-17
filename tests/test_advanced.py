# -*- coding: utf-8 -*-

from .context import indexes

import unittest


class AdvancedTestSuite(unittest.TestCase):
    """Advanced test cases."""

    def test_thoughts(self):
        self.assertIsNone(indexes.hmm())


if __name__ == '__main__':
    unittest.main()
