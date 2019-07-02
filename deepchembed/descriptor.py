"""
Wrapper Module to generate molecular descriptors by using other packages
"""

import math
import numpy as np
import os
import pandas as pd

from abc import ABC, abstractmethod
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
import rdkit.Chem.rdmolops as rdmolops
import rdkit.Chem.rdMolDescriptors as rdDesc
import rdkit.Chem.EState.EState_VSA as EState
from mordred import Calculator as mdCalc
import mordred.descriptors as mdDesc
import mordred.error as mdError


class Descriptors(ABC):
    """
    An abstract class for descriptor computation.

    Attributes:
        Molecule: a rdkit.Chem.rdchem.Mol object, stores the chemical info.
    """
    def __init__(self, SMILES = None):
        """ Descriptor Constructor """
        if(SMILES is not None):
            self.set_molecule(SMILES)
        else:
            self.Molecule = None

    def set_molecule(self, SMILES):
        """ set molecule of the rdkitDecriptor"""
        self.Molecule = Chem.MolFromSmiles(SMILES)
        return

    def compute_all_descriptors(self):
        """ compute descriptors for one molecule"""
        pass

    @abstractmethod
    def batch_compute_all_descriptors(SMILES_list):
        """ compute descriptors for a list of molecules, must
        implemented as @staticmethod.
        """
        pass


class rdkitDescriptors(Descriptors):
    """
    A wrapper class using rdkit to generate the different descpritors.

    Initilized with SIMLES of a molecule

    Attributes:
        Molecule: an object of rdkit.Chem.rdchem.Mol

    Methods:
        set_molecule

        compute_all_descriptors
        compute_properties
        compute_connectivity_and_shape_indexes
        compute_MOE_descriptors
        compute_MQN_descriptors
        compute_Morgan_fingerprint

    """

    def compute_all_descriptors(self,desc_type='all'):
        """compute all descriptors avaiable from the rdkit package,

        Args:
            desc_type: descriptor type, could be 'all', 'int', or 'float'
        Return:
            desc_dict: descriptor dictionary.
        """
        assert desc_type in ['all', 'int','float']

        desc_dict = {}

        if desc_type == 'all':
            desc_dict.update(self.compute_properties())
            desc_dict.update(self.compute_connectivity_and_shape_indexes())
            desc_dict.update(self.compute_MOE_descriptors())
            desc_dict.update(self.compute_MQN_descriptors())
        elif desc_type == 'int':
            desc_dict.update(self.compute_properties(\
                ['lipinskiHBA', 'lipinskiHBD', 'NumRotatableBonds', 'NumRings',
                 'NumHeteroatoms', 'NumAmideBonds','NumAromaticRings', 'NumHBA',
                 'NumAliphaticRings', 'NumSaturatedRings', 'NumHeterocycles',
                 'NumAromaticHeterocycles', 'NumSaturatedHeterocycles', 'NumHBD',
                 'NumAliphaticHeterocycles', 'NumSpiroAtoms', 'NumBridgeheadAtoms',
                 'NumAtomStereoCenters','NumUnspecifiedAtomStereoCenters']))
            desc_dict.update(self.compute_MQN_descriptors())
        elif desc_type == 'float':
            desc_dict.update(self.compute_properties(\
                ['exactmw','FractionCSP3','labuteASA','tpsa',
                 'CrippenClogP','CrippenMR']))
            desc_dict.update(self.compute_connectivity_and_shape_indexes())
            desc_dict.update(self.compute_MOE_descriptors())

        return desc_dict

    @staticmethod
    def batch_compute_all_descriptors(SMILES_list, desc_type='all'):
        """compute all descriptors avaiable from the rdkit package for a list
        of molecules.

        Args:
            desc_type: descriptor type, could be 'all', 'int', or 'float'
        Return:
            desc_df: descriptors pandas.DataFrame.
        """
        assert len(SMILES_list) >= 1

        Molecules = list(map(Chem.MolFromSmiles, SMILES_list))
        DESC_ENGINE = rdkitDescriptors()
        DESC_ENGINE.set_molecule(SMILES_list[0])
        desc_dict = DESC_ENGINE.compute_all_descriptors(desc_type)
        desc_df = pd.DataFrame(desc_dict, index=[0])

        for i in range(1,len(Molecules)):
            DESC_ENGINE.set_molecule(SMILES_list[i])
            desc_dict = DESC_ENGINE.compute_all_descriptors(desc_type)
            desc_df = desc_df.append(pd.DataFrame(desc_dict, index=[i]))

        return desc_df

    def compute_properties(self, feature_name = None):
        """compute the basic properties from the rdkit package

        Args:
            feature_name: a list of input features names. if not specified, all
                          avaiable features will be calculated.

        Returns:
            prop_dict: property dictionary, mixed with float and int
        """

        assert type(self.Molecule) == Chem.rdchem.Mol

        if feature_name is not None:
            properties = rdDesc.Properties(feature_name)
        else:
            properties = rdDesc.Properties()

        props_dict = dict(zip(properties.GetPropertyNames(),
                              properties.ComputeProperties(self.Molecule)))

        return props_dict

    def compute_connectivity_and_shape_indexes(self):
        """compute the compute connectivity and shape indexes.
        Ref: Rev. Comput. Chem. 2:367-422 (1991)

        Returns:
            CSI_dict: CSI dictionary, data type: float
        """
        assert type(self.Molecule) == Chem.rdchem.Mol

        CSI_dict = {}

        CSI_dict['Chi0v'] = rdDesc.CalcChi0v(self.Molecule)
        CSI_dict['Chi1v'] = rdDesc.CalcChi1v(self.Molecule)
        CSI_dict['Chi2v'] = rdDesc.CalcChi2v(self.Molecule)
        CSI_dict['Chi3v'] = rdDesc.CalcChi3v(self.Molecule)
        CSI_dict['Chi4v'] = rdDesc.CalcChi4v(self.Molecule)
        CSI_dict['Chi0n'] = rdDesc.CalcChi0n(self.Molecule)
        CSI_dict['Chi1n'] = rdDesc.CalcChi1n(self.Molecule)
        CSI_dict['Chi2n'] = rdDesc.CalcChi2n(self.Molecule)
        CSI_dict['Chi3n'] = rdDesc.CalcChi3n(self.Molecule)
        CSI_dict['Chi4n'] = rdDesc.CalcChi4n(self.Molecule)
        CSI_dict['HallKierAlpha'] = rdDesc.CalcHallKierAlpha(self.Molecule)
        CSI_dict['Kappa1'] = rdDesc.CalcKappa1(self.Molecule)
        CSI_dict['Kappa2'] = rdDesc.CalcKappa2(self.Molecule)
        CSI_dict['Kappa3'] = rdDesc.CalcKappa3(self.Molecule)

        return CSI_dict

    def compute_MOE_descriptors(self):
        """compute the MOE-type descriptors.
        Ref:???

        Returns:
            MOE_dict: MOE dictionary, data type: float
        """
        assert type(self.Molecule) == Chem.rdchem.Mol

        MOE_dict = {}

        SlogP_VSA_names = []
        for i in range(1,13):
            SlogP_VSA_names.append('SlogP_VSA' + str(i))

        MOE_dict.update(dict(zip(SlogP_VSA_names,
                                     rdDesc.SlogP_VSA_(self.Molecule))))

        SMR_VSA_names = []
        for i in range(1,11):
            SMR_VSA_names.append('SMR_VSA' + str(i))

        MOE_dict.update(dict(zip(SMR_VSA_names,
                                 rdDesc.SMR_VSA_(self.Molecule))))

        PEOE_VSA_names = []
        for i in range(1,15):
            PEOE_VSA_names.append('PEOE_VSA' + str(i))

        MOE_dict.update(dict(zip(PEOE_VSA_names,
                                 rdDesc.PEOE_VSA_(self.Molecule))))

        EState_VSA_names = []
        for i in range(1,12):
            EState_VSA_names.append('EState_VSA' + str(i))

        MOE_dict.update(dict(zip(EState_VSA_names,
                                 EState.EState_VSA_(self.Molecule))))

        VSA_EState_names = []
        for i in range(1,12):
            VSA_EState_names.append('VSA_EState' + str(i))

        MOE_dict.update(dict(zip(VSA_EState_names,
                                 EState.VSA_EState_(self.Molecule))))

        return MOE_dict

    def compute_MQN_descriptors(self):
        """compute the MQN-type descriptors.
        Ref: Nguyen et al. ChemMedChem 4:1803-5 (2009)

        Returns:
            MOE_dict: MQN dictionary, data type: int
        """
        assert type(self.Molecule) == Chem.rdchem.Mol

        MQN_names = []
        for i in range(1,43):
            MQN_names.append('MQN' + str(i))

        MQN_dict = dict(zip(MQN_names, rdDesc.MQNs_(self.Molecule)))

        return MQN_dict

    @staticmethod
    def batch_compute_rdkit_fingerprints(SMILES_list):
        """batch compute rdkit topological fingerprints for a list of
        input SMILES

        Args:
            SMILES_list: a list of SMILES for computation.

        Returns:
            FPs: a numpy.array of binary rdkit fingerprints.
        """
        assert len(SMILES_list) >= 1

        Molecules = list(map(Chem.MolFromSmiles, SMILES_list))
        FPs = list(map(lambda x: list(rdmolops.RDKFingerprint(x)), Molecules))

        return np.array(FPs)

    @staticmethod
    def batch_compute_MACCSkeys(SMILES_list):
        """batch compute MACCSkeys for a list of input SMILES from rdkit

        Args:
            SMILES_list: a list of SMILES for computation.

        Returns:
            FPs: a numpy.array of binary MACCSkeys.
         """
        assert len(SMILES_list) >= 1

        Molecules = list(map(Chem.MolFromSmiles, SMILES_list.values))
        FPs = list(map(lambda x: list(MACCSkeys.GenMACCSKeys(x)), Molecules))

        return np.array(FPs)

class mordredDescriptors(Descriptors):
    """
    A wrapper class using mordred to generate the different descpritors.

    Initilized with SIMLES of a molecule

    Attributes:
    Molecule       -- an object of rdkit.Chem.rdchem.Mol

    Methods:
    set_molecule

    compute_all_descriptors

    """

    DESC_ENGINE = mdCalc(mdDesc, ignore_3D=True)

    def compute_all_descriptors(self):
        """compute all modred descriptors for one molecule

        Returns:
            desc_list: a list of modred descriptors, mixed datatype.
        """
        assert type(self.Molecule) == Chem.rdchem.Mol
        desc_list = dict(zip(list(self.DESC_ENGINE._name_dict.keys()),
                             self.DESC_ENGINE(self.Molecule)))

        desc_list ={k: self._convert_mdError_to_na(v)
                    for k, v in desc_list.items()}

        return desc_list

    @staticmethod
    def batch_compute_all_descriptors(SMILES_list, desc_type='all',
                                      remove_na=True):
        """compute all descriptors avaiable from the modred package for a list
        of molecules. and remove NA and error if presented.

        Args:
            desc_type: descriptor type, could be 'all', 'int', or 'float'
            remove_na: boolean, True for removing all NA in the return dataFrame
        Return:
            desc_df: descriptors pandas.DataFrame.
        """
        assert len(SMILES_list) >= 1
        assert desc_type in ['all', 'int','float']
        DESC_ENGINE = mordredDescriptors.DESC_ENGINE
        na_coverter = mordredDescriptors._convert_mdError_to_na

        Molecules = list(map(Chem.MolFromSmiles, SMILES_list))
        desc_df = DESC_ENGINE.pandas(Molecules)
        desc_df = desc_df.applymap(na_coverter)

        if remove_na:
            for col in desc_df.columns:
                if len(pd.value_counts(desc_df[col].isna())) > 1:
                    desc_df = desc_df.drop(col, axis=1)

        if desc_type == 'int':
            desc_df = desc_df.select_dtypes(include=['int64'])
        elif desc_type == 'float':
            desc_df = desc_df.select_dtypes(include=['float64'])

        return desc_df

    @staticmethod
    def _convert_mdError_to_na(x):
        """helper function for converting mdError.Missing into numpy.nan
        """
        if type(x) == mdError.Missing:
            return np.nan
        else:
            return x
