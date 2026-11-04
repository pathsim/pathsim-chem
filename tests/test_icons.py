########################################################################################
##
##                                  TESTS FOR
##                                  block icons
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import xml.etree.ElementTree as ET

from pathsim.blocks import Block

import pathsim_chem


# TESTS ================================================================================

class TestIcons(unittest.TestCase):
    """
    Every block of the toolbox has an icon, read by 'Block.icon()'
    """

    def test_blocks_have_icons(self):

        blocks = [
            obj for obj in vars(pathsim_chem).values()
            if isinstance(obj, type) and issubclass(obj, Block)
            and obj.__module__.startswith("pathsim_chem")
            ]
        self.assertGreater(len(blocks), 0)

        for cls in blocks:
            with self.subTest(block=cls.__name__):
                icon = cls.icon()
                self.assertIsNotNone(icon)
                self.assertEqual(ET.fromstring(icon).get("viewBox"), "0 0 96 64")


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
