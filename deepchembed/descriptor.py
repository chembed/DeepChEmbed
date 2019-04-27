"""
Wrapper Module using other packages to generate molecular descriptors
"""

# from itertools import zip_longest
import math
import numpy as np
import os
import pandas as pd

from rdkit import Chem
import rdkit.Chem.rdMolDescriptors as rdMolD
from rdkit.Chem import AllChem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig

class rdkitDescriptors:
    """
    A wrapper class using rdkit to generate the different descpritors
    for specrtra prediciton.

    Initilized with SIMLES of a molecule

    Attributes:
    Molecule       -- an object of rdkit.Chem.rdchem.Mol
    __feat_factory -- an object to caluate the featrues for molecules, from
                    rdkit.Chem.rdMolChemicalFeatures.MolChemicalFeatureFactory

    Methods:
    set_molecule
    get_features
    get_properties
    get_coulomb_matrix
    get_Morgan_fingerprint
    __config_feature_factory
    __get_charges_coords

    """

    def __init__(self, SMILES = None):
        """
        spDescriptor Constructor
        """
        if(SMILES is not None):
            self.set_molecule(SMILES)
        else:
            self.Molecule = None

        #functions extend from rdkit package
        # self.__feat_factory  = None
#        self.Features    = None
#        self.Fingerprint = None


    def set_molecule(self, SMILES):
        """ set molecule of the spDecriptor"""
        self.Molecule = Chem.MolFromSmiles(SMILES)
        return

    def compute_all_descriptors(self):

        des_dict = {}
        des_dict.update(self.compute_properties())
        des_dict.update(self.compute_connectivity_and_shape_indexes())
        des_dict.update(self.compute_MOE_descriptors())
        des_dict.update(self.compute_MQN_descriptors())

        return des_dict

    def compute_properties(self, feature_name = None):
        """  """

        assert type(self.Molecule) == Chem.rdchem.Mol

        if feature_name is not None:
            properties = rdMolD.Properties(feature_name)
        else:
            properties = rdMolD.Properties()

        props_dict = dict(zip(properties.GetPropertyNames(),
                              properties.ComputeProperties(self.Molecule)))

        return props_dict

    def compute_connectivity_and_shape_indexes(self):
        """ """
        assert type(self.Molecule) == Chem.rdchem.Mol

        CSI_dict = {}

        CSI_dict['Chi0v'] = rdMolD.CalcChi0v(self.Molecule)
        CSI_dict['Chi1v'] = rdMolD.CalcChi1v(self.Molecule)
        CSI_dict['Chi2v'] = rdMolD.CalcChi2v(self.Molecule)
        CSI_dict['Chi3v'] = rdMolD.CalcChi3v(self.Molecule)
        CSI_dict['Chi4v'] = rdMolD.CalcChi4v(self.Molecule)
        CSI_dict['Chi0n'] = rdMolD.CalcChi0n(self.Molecule)
        CSI_dict['Chi1n'] = rdMolD.CalcChi1n(self.Molecule)
        CSI_dict['Chi2n'] = rdMolD.CalcChi2n(self.Molecule)
        CSI_dict['Chi3n'] = rdMolD.CalcChi3n(self.Molecule)
        CSI_dict['Chi4n'] = rdMolD.CalcChi4n(self.Molecule)
        CSI_dict['HallKierAlpha'] = rdMolD.CalcHallKierAlpha(self.Molecule)
        CSI_dict['Kappa1'] = rdMolD.CalcKappa1(self.Molecule)
        CSI_dict['Kappa2'] = rdMolD.CalcKappa2(self.Molecule)
        CSI_dict['Kappa3'] = rdMolD.CalcKappa3(self.Molecule)

        return CSI_dict

    def compute_MOE_descriptors(self):
        """ """
        assert type(self.Molecule) == Chem.rdchem.Mol

        MOE_dict = {}

        SlogP_names = []
        for i in range(1,13):
            SlogP_names.append('SlogP_VSA' + str(i))

        MOE_dict.update(dict(zip(SlogP_names, rdMolD.SlogP_VSA_(self.Molecule))))

        SMR_names = []
        for i in range(1,11):
            SMR_names.append('SMR_VSA' + str(i))

        MOE_dict.update(dict(zip(SMR_names, rdMolD.SMR_VSA_(self.Molecule))))

        PEOE_names = []
        for i in range(1,15):
            PEOE_names.append('PEOE_VSA' + str(i))

        MOE_dict.update(dict(zip(PEOE_names, rdMolD.PEOE_VSA_(self.Molecule))))

        return MOE_dict

    def compute_MQN_descriptors(self):
        """ """
        assert type(self.Molecule) == Chem.rdchem.Mol

        MQN_names = []
        for i in range(1,43):
            MQN_names.append('MQN' + str(i))

        MQN_dict = dict(zip(MQN_names, rdMolD.MQNs_(self.Molecule)))

        return MQN_dict

    def compute_Morgan_fingerprint(self, radius=2, nBits=2048, use_features=False):
        """  """
        assert type(self.Molecule) == Chem.rdchem.Mol
        fp = AllChem.GetMorganFingerprintAsBitVect(self.Molecule, radius, nBits=nBits,
                                                   useFeatures=use_features)
        return list(fp.ToBinary())

    # def get_features(self):
    #     if(self.__feat_factory is None):
    #         self.__config_feature_factory()
    #
    #     assert type(self.Molecule) == Chem.rdchem.Mol
    #
    #     features = self.__feat_factory.GetFeaturesForMol(self.Molecule)
    #     features_dict = {}
    #     for i in range(len(features)):
    #         f_type = features[i].GetType()
    #         f_ids  = features[i].GetAtomIds()
    #         if (f_type in features_dict.keys()):
    #             features_dict[f_type].append(f_ids)
    #         else:
    #             features_dict[f_type] = [f_ids]
    #
    #     return features_dict



    # def get_coulomb_matrix(self, eig_sort=True):
    #     """
    #     Generates the coulomb matrix for a given molecule from its
    #         SMILES string, of size MxM, where M is the number of
    #         atoms in the molecule. in the training data set.
    #
    #     Args:
    #     -----
    #         SMILES (str) -- the SMILES string representation of the
    #             molecule.
    #     Returns:
    #     --------
    #         coulomb_matrix (numpy.ndarray) -- the coulomb matrix for
    #             a given molecule's nuclear geometry.
    #     """
    #     # Assertions
    #     assert type(self.Molecule) == Chem.rdchem.Mol
    #     # Generating the coulomb matrix
    #     molecule_df = self.__get_charges_coords()
    #     num_atoms = len(molecule_df)
    #     coulomb_matrix = np.zeros(shape=(num_atoms,num_atoms))
    #     for indexi, rowi in molecule_df.iterrows():
    #         for indexj, rowj in molecule_df.iterrows():
    #             Zi = rowi.charge
    #             xi = rowi.x
    #             yi = rowi.y
    #             zi = rowi.z
    #             Zj = rowj.charge
    #             xj = rowj.x
    #             yj = rowj.y
    #             zj = rowj.z
    #             if indexi == indexj:
    #                 element = 0.5 * math.pow(Zi, 2.4)
    #             else:
    #                 norm_diff = math.sqrt(math.pow((xi-xj),2) + math.pow((yi-yj),2) + math.pow((zi-zj),2))
    #                 element = Zi * Zj / norm_diff
    #             coulomb_matrix[indexi][indexj] = element
    #
    #     if eig_sort:
    #         eig = np.linalg.eig(coulomb_matrix)[0]
    #         eig_idx = np.argsort(eig)
    #         sorted_matrix = np.zeros(shape=(num_atoms,num_atoms))
    #         for i in range(num_atoms):
    #             sorted_matrix[i] = coulomb_matrix[eig_idx[i]]
    #         return sorted_matrix
    #     else:
    #         pass
    #
    #     return coulomb_matrix
    #
    # def __config_feature_factory(self):
    #     """
    #     """
    #     fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
    #     self.__feat_factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    #     return
    #
    # def __get_charges_coords(self):
    #     """
    #     Generates a pandas dataframe containing the charges and cartesian
    #         coordinates of each atom in a molecule.
    #
    #     Args:
    #     -----
    #         SMILES (str) -- the SMILES string representation of the
    #             molecule.
    #
    #     Returns:
    #     --------
    #         molecule_df (pandas.DataFrame) -- contains the charge and
    #             cartesian coordinate information for each atom within
    #             a molecule.
    #     """
    #     # Assertions
    #
    #     # Building the benzene molecule and ADDING HYDROGENS
    #     molecule = Chem.AddHs(self.Molecule)
    #     # 'Embedding' the molecular coordinates, optimising structure
    #     AllChem.EmbedMolecule(molecule)
    #     AllChem.MMFFOptimizeMolecule(molecule)
    #     # Generating universal force field model
    #     ff = AllChem.UFFGetMoleculeForceField(molecule)
    #     # Getting the positions of nuclei; returned as a tuple
    #     # of the form (x1, y1, z1, x2, y2, z2, x3, ...)
    #     positions = ff.Positions()
    #
    #     # Creating a list of the atomic numbers
    #     atomic_nums = []
    #     for atom in molecule.GetAtoms():
    #         atomic_nums.append(atom.GetAtomicNum())
    #
    #     # Creating lists of the cartesian coordinates of the atoms
    #     x = []
    #     y = []
    #     z = []
    #     for item1, item2, item3 in self.__grouper(3, positions):
    #         x.append(item1)
    #         y.append(item2)
    #         z.append(item3)
    #
    #     # Building a DF with predictors
    #     molecule_df = pd.DataFrame()
    #     molecule_df['charge'] = atomic_nums
    #     molecule_df['x'] = x
    #     molecule_df['y'] = y
    #     molecule_df['z'] = z
    #
    #     return molecule_df

    # def __grouper(self, n, iterable, fillvalue=None):
    #     """
    #     A function to aggregates items in a list into groups of
    #         assigned size.
    #
    #     Args:
    #     -----
    #         n (int)         -- the number of items per group.
    #         iterable (list) -- the list to iterate over, whose items will
    #             be grouped.
    #         fillvalue       -- the value to fill in empty compartments of a
    #             group if the number of items in the iterator isn't a multiple
    #             of the group size (n).
    #
    #     Returns:
    #     --------
    #         grouper (itertools.zip_longest) -- the iterator for the
    #             aggregation of items in the input list.
    #     """
    #     args = [iter(iterable)] * n
    #     grouper = zip_longest(fillvalue=fillvalue, *args)
    #
    #     return grouper
