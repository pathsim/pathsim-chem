########################################################################################
##
##                                  TESTS FOR
##                     deprecated aliases of the moved blocks
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import importlib.util

import pathsim_chem
import pathsim_chem.tritium

from pathsim.blocks import Splitter


HAS_FUSION = importlib.util.find_spec("pathsim_fusion") is not None


# TESTS ================================================================================

class TestMovedBlocks(unittest.TestCase):
    """
    Test the deprecated aliases for the blocks that moved to pathsim core
    and to pathsim-fusion
    """

    def test_splitter(self):

        #alias of the core block, from the module and the package
        with self.assertWarns(DeprecationWarning):
            self.assertIs(pathsim_chem.tritium.Splitter, Splitter)
        with self.assertWarns(DeprecationWarning):
            from pathsim_chem import Splitter as S
        self.assertIs(S, Splitter)


    def test_not_registered(self):

        #aliases are not part of the package namespace
        self.assertNotIn("Splitter", dir(pathsim_chem))
        self.assertNotIn("GLC", dir(pathsim_chem.tritium))

        #unknown names still raise
        with self.assertRaises(AttributeError):
            pathsim_chem.tritium.NoBlock


    @unittest.skipUnless(HAS_FUSION, "pathsim-fusion not installed")
    def test_fusion(self):

        import pathsim_fusion

        for name in ("GLC", "Bubbler4", "IonisationChamber"):
            with self.assertWarns(DeprecationWarning):
                self.assertIs(getattr(pathsim_chem, name), getattr(pathsim_fusion, name))


    @unittest.skipIf(HAS_FUSION, "pathsim-fusion installed")
    def test_fusion_missing(self):

        with self.assertWarns(DeprecationWarning):
            with self.assertRaises(ImportError):
                pathsim_chem.GLC


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
