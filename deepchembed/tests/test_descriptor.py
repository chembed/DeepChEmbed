import __init__
import descriptor
from unittest import TestCase
from unittest.mock import patch, Mock
from descriptor import Descriptors
from rdkit import Chem


class TestDescriptors(TestCase):
    """
    """
    def test_cannot_instantiate(self):
        """ test the absctract class won't be instantiated"""
        with self.assertRaises(TypeError):
            Descriptors()

        return

    @patch.multiple(Descriptors,__abstractmethods__=set())
    def test_set_molecules(self):
        """ test constuctor and set_molecule """

        descriptor_engine = Descriptors()
        assert descriptor_engine.Molecule == None

        descriptor_engine.set_molecule('c1ccccc1')
        isinstance(descriptor_engine.Molecule, Chem.rdchem.Mol)

        return
