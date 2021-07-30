#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import numpy
import rdkit
import torch
import itertools
import pytorch_lightning
from rdkit import Chem
from typing import Any
from typing import Iterable
from typing import Tuple
from typing import List
from typing import Optional
import os
import pandas
import argparse
import logging
import time 

__author__ = "Damien Coupry"
__copyright__ = "GSK"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Damien Coupry"
__email__ = "damien.x.coupry@gsk.com"
__status__ = "Dev"

logger = logging.getLogger("optuna")
rdkit.RDLogger.DisableLog('rdApp.*')
pytorch_lightning.seed_everything(42) 

BOND_ORDERS = [1.0, 1.5, 2.0, 3.0]
NODE_DEGREES = [0, 1, 2, 3, 4, 5, 6]
H_DEGREES = [0, 1, 2, 3, 4]
CHARGES = [-2, -1, 0, 1, 2]
# the following subset was chosen after counting elements in the
# zinc database of druglike compounds
ATOMIC_SYMBOLS = ["C", "N", "O", "S", "F", "Cl", "Br", "I"]
HYBRIDIZATIONS = [Chem.rdchem.HybridizationType.UNSPECIFIED,
                  Chem.rdchem.HybridizationType.S,
                  Chem.rdchem.HybridizationType.SP,
                  Chem.rdchem.HybridizationType.SP2,
                  Chem.rdchem.HybridizationType.SP3,
                  Chem.rdchem.HybridizationType.SP3D,
                  Chem.rdchem.HybridizationType.SP3D2,
                  Chem.rdchem.HybridizationType.OTHER]
CHIRALITY = [Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
             Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
             Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
             Chem.rdchem.ChiralType.CHI_OTHER]
STEREO = [Chem.rdchem.BondStereo.STEREONONE,
          Chem.rdchem.BondStereo.STEREOE,
          Chem.rdchem.BondStereo.STEREOZ]


def featurize_smile(smi: str) -> Tuple[torch.Tensor,
                                       torch.Tensor,
                                       torch.Tensor,
                                       torch.Tensor]:
    """
    [summary]

    Parameters
    ----------
    smi : str
        [description]

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        [description]
    """
    mol = MolGraph(smi)
    node_feat = getattr(mol, "atoms_features")
    edge_feat = getattr(mol, "edges_features")
    edge_idx = getattr(mol, "edges_indices")
    feats = (torch.FloatTensor(node_feat),
             torch.FloatTensor(edge_feat),
             torch.LongTensor(edge_idx))
    nans = [torch.isnan(feat).any() for feat in feats]
    assert sum(nans) == 0
    return feats


def featurize(smiles: Iterable[str]) -> numpy.array:
    """
    [summary]

    Parameters
    ----------
    smiles : Iterable[str]
        [description]

    Returns
    -------
    numpy.array
        [description]
    """
    out = numpy.empty(len(smiles), dtype=object)
    for idx, smi in enumerate(smiles):
        try:
            feats = featurize_smile(smi)
            out[idx] = feats
        except Exception:
            continue
    return out


def onek(x: Any,
         allowable_set: Iterable
         ) -> List[int]:
    """
    [summary]

    Parameters
    ----------
    x : Any
        [description]
    allowable_set : Iterable
        [description]

    Returns
    -------
    List[int]
        [description]
    """
    if x not in allowable_set:
        return [0, ] * len(allowable_set)
    else:
        return list(map(lambda s: x == s, allowable_set))


class Edges(object):
    """Edges class for RDKit MolGraph.
    Edges in this case refers to both the edge features and
    the edge indices wrt the nodes of the graph. thus it is
    sensitive to index reordering.

    The features are well understood chemical bond descriptors,
    found in the rdkit.Chem.rdchem.Bond object methods, such as:
    * bond order
    * bond stereochemistry
    * whether the bond is in a ring
    * etc.
    """
    def __init__(self,
                 mol: rdkit.Chem.rdchem.Mol
                 ) -> None:
        """
        [summary]

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            [description]

        Returns
        -------
        [type]
            [description]
        """
        self._receivers = []
        self._senders = []
        self._edges_features = []
        for bond in mol.GetBonds():
            bgn_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            edge_feats = self._featurize_edge(bond, mol)
            self._receivers += [bgn_idx, end_idx]
            self._senders += [end_idx, bgn_idx]
            self._edges_features += [edge_feats, edge_feats]
        return None

    @staticmethod
    def _featurize_edge(bond: rdkit.Chem.rdchem.Bond,
                        mol: rdkit.Chem.rdchem.Mol
                        ) -> numpy.array:
        """
        [summary]

        Parameters
        ----------
        bond : rdkit.Chem.rdchem.Bond
            [description]
        mol : rdkit.Chem.rdchem.Mol
            [description]

        Returns
        -------
        numpy.array
            [description]
        """
        #            first the bond types
        results = (onek(bond.GetBondTypeAsDouble(), BOND_ORDERS) +
                   # is the bond aromatic?
                   [bond.GetIsAromatic()] +
                   # ist it conjugated? (nb: not the same as aromatic)
                   [bond.GetIsConjugated()] +
                   # is it part of a ring
                   [bond.IsInRing()] +
                   # does it have defined stereochemistry
                   onek(bond.GetStereo(), STEREO)
                   )
        return numpy.array(results, dtype=numpy.float32)

    @property
    def edges_indices(self) -> numpy.array:
        """
        [summary]

        Returns
        -------
        numpy.array
            [description]
        """
        r = numpy.array(self._receivers, dtype=numpy.int16)
        s = numpy.array(self._senders, dtype=numpy.int16)
        return numpy.asarray([r, s], dtype=numpy.int16)

    @property
    def edges_features(self) -> numpy.array:
        """
        [summary]

        Returns
        -------
        numpy.array
            [description]
        """
        return numpy.asarray(self._edges_features, dtype=numpy.float32)


class Nodes(object):
    """Nodes class for RDKit MolGraph
    Most of the data passed around in graph convolutional
    networks is node centered. Indeed, in chemical graphs,
    we instinctively grasp that properties are the result
    of the interaction between atoms with individual properties:
    * mass
    * chirality
    * hybridization
    * etc.

    thus generating features that reflect well the behaviour
    of atoms in a molecule is one of the main areas of improvement
    for the performance of a GCN. so far, featurization can be based on
    the full element type or a simplified version. Work is underway to
    use specifically developped categories such as E-States or UFF atom types
    """
    def __init__(self,
                 mol: rdkit.Chem.rdchem.Mol
                 ) -> None:
        """
        [summary]

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            [description]

        Returns
        -------
        [type]
            [description]
        """
        self._atoms_features = numpy.array([self._featurize_atom(a)
                                            for a in mol.GetAtoms()])
        return None

    @staticmethod
    def _featurize_atom(atom: rdkit.Chem.rdchem.Atom) -> list:
        """
        [summary]

        Parameters
        ----------
        atom : rdkit.Chem.rdchem.Atom
            [description]

        Returns
        -------
        list
            [description]
        """
        #            hybridization state for symmetry
        results = (onek(atom.GetHybridization(), HYBRIDIZATIONS) +
                   # degree of connection for the node
                   onek(atom.GetTotalDegree(), NODE_DEGREES) +
                   # is the atom part of an aromatic chain?
                   [atom.GetIsAromatic()] +
                   # is the atom in a ring?
                   [atom.IsInRing()] +
                   # number of hydrogens, both implicit and explicit
                   onek(atom.GetTotalNumHs(includeNeighbors=True), H_DEGREES) +
                   # chirality of the atom if specified
                   onek(atom.GetChiralTag(), CHIRALITY) +
                   # formal charge of atom
                   onek(atom.GetFormalCharge(), CHARGES) +
                   # element type
                   onek(atom.GetSymbol(), ATOMIC_SYMBOLS)
                   )
        return results

    @property
    def atoms_features(self) -> numpy.array:
        """
        [summary]

        Returns
        -------
        numpy.array
            [description]
        """
        return numpy.asarray(self._atoms_features, dtype=numpy.float32)


class MolGraph(Nodes, Edges):
    """
    The graph object corresponding to a molecule
    including features for the nodes, edges and globals.
    """
    def __init__(self,
                 smiles: str) -> None:
        """
        [summary]

        Parameters
        ----------
        smiles : str
            [description]

        Returns
        -------
        [type]
            [description]
        """        
        self.smiles = smiles
        self.mol = rdkit.Chem.RemoveHs(rdkit.Chem.MolFromSmiles(smiles))
        Nodes.__init__(self, self.mol)
        Edges.__init__(self, self.mol)
        return None

class GraphsDataset(torch.utils.data.Dataset):
    """
    """
    def __init__(self, 
                 graph_tuples: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
                 targets: torch.Tensor,
                 noise: float = 0.0) -> None:
        """
        [summary]

        Parameters
        ----------
        graph_tuples : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            [description]
        targets : torch.Tensor
            [description]
        """    
        super().__init__()
        self.graphs = graph_tuples
        self.targets = targets
        self.noise = noise
    
    def __len__(self) -> int:
        """
        [summary]

        Returns
        -------
        int
            [description]
        """
        return len(self.graphs)

    def __getitem__(self, 
                    idx: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        [summary]

        Parameters
        ----------
        idx : int
            [description]

        Returns
        -------
        Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
            [description]
        """        
        nodes, edges, edges_indices = self.graphs[idx]
        targets = self.targets[idx]
        noise = numpy.random.normal(targets) * self.noise 
        targets += noise
        return (nodes, edges, edges_indices), targets


def collate_fn(batch_data: List[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    [summary]

    Parameters
    ----------
    batch_data : List[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]
        [description]

    Returns
    -------
    Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
        [description]
    """
    X, y = zip(*batch_data)
    num_atoms = [len(_x[0]) for _x in X]
    num_edges = [len(_x[1]) for _x in X]
    offsets = numpy.repeat(numpy.cumsum(numpy.hstack([0, num_atoms[:-1]])), num_edges)
    offsets = torch.LongTensor(offsets)
    node_feat = torch.cat([_x[0] for _x in X], dim=0)
    edge_feat = torch.cat([_x[1] for _x in X], dim=0)
    senders, receivers = zip(*[_x[2] for _x in X])
    senders = torch.cat(senders, dim=0)
    receivers = torch.cat(receivers, dim=0)
    senders += offsets
    receivers += offsets
    edge_idx = torch.stack([senders, receivers])
    batch = torch.cat([torch.full((nat, ), 0, dtype=torch.long) + i for i, nat in enumerate(num_atoms)])
    y = torch.FloatTensor(y)
    return (node_feat, edge_feat, edge_idx, batch), y


class GraphsDataModule(pytorch_lightning.LightningDataModule):
    """
    """
    def __init__(self, 
                 hparams: argparse.Namespace,
                 cv: Optional[int] = None) -> None:
        """
        [summary]

        Parameters
        ----------
        hparams : argparse.Namespace
            [description]
        cv : Optional[int], optional
            [description], by default None

        Returns
        -------
        [type]
            [description]
        """        
        super().__init__()
        self.hparams = hparams
        if cv is not None: 
            self.cv = f".{cv}" 
        else: 
            self.cv = ""
        self.dataset_train = None
        self.dataset_valid = None
        self.dataset_test = None
        self.train_data_name = f"{self.hparams.study_name}/{self.hparams.data_file}.train" + self.cv
        self.valid_data_name = f"{self.hparams.study_name}/{self.hparams.data_file}.valid" + self.cv
        self.test_data_name = f"{self.hparams.study_name}/{self.hparams.data_file}.test"
        self.train_ckpt_name = f"{self.hparams.study_name}/train.ckpt" + self.cv
        self.valid_ckpt_name = f"{self.hparams.study_name}/valid.ckpt" + self.cv
        self.test_ckpt_name = f"{self.hparams.study_name}/test.ckpt"
        if hasattr(self.hparams, "batch_size"):
            batch_size = self.hparams.batch_size
        else:
            batch_size = 4
        self.dl_kwargs = {"collate_fn":collate_fn, 
                          "num_workers": self.hparams.workers, 
                          "pin_memory": torch.cuda.is_available(),
                          "batch_size": batch_size}
        return None

    def get_train_len(self) -> Optional[int]:
        return len(pandas.read_csv(self.train_data_name))

    def prepare_data(self) -> None:
        """
        [summary]
        """        
        if not os.path.isfile(self.test_ckpt_name):
            # same with the holdout
            test_data = pandas.read_csv(self.test_data_name)
            X_test = featurize(test_data[self.hparams.smiles_col])
            y_test = test_data[self.hparams.target_col].values
            if len(self.hparams.target_col) == 1:
                y_test = y_test.reshape(-1,1)
            dataset_test = GraphsDataset(X_test, y_test, noise=self.hparams.noise)
            torch.save(dataset_test, self.test_ckpt_name)
        if not os.path.isfile(self.train_ckpt_name):
            # featurize and save the training data
            train_data = pandas.read_csv(self.train_data_name)
            X_train = featurize(train_data[self.hparams.smiles_col])
            y_train = train_data[self.hparams.target_col].values
            if len(self.hparams.target_col) == 1:
                y_train = y_train.reshape(-1,1)
            dataset_train = GraphsDataset(X_train, y_train, noise=self.hparams.noise)
            torch.save(dataset_train, self.train_ckpt_name)
        if not os.path.isfile(self.valid_ckpt_name):
            # same with the holdout
            valid_data = pandas.read_csv(self.valid_data_name)
            X_valid = featurize(valid_data[self.hparams.smiles_col])
            y_valid = valid_data[self.hparams.target_col].values
            if len(self.hparams.target_col) == 1:
                y_valid = y_valid.reshape(-1,1)
            dataset_valid = GraphsDataset(X_valid, y_valid)
            torch.save(dataset_valid, self.valid_ckpt_name)

    def setup(self, stage: Optional[str] = None):
        """
        [summary]

        Parameters
        ----------
        stage : Optional[str], optional
            [description], by default None
        """        
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.dataset_train = torch.load(self.train_ckpt_name)
            self.dataset_valid = torch.load(self.valid_ckpt_name)
        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.dataset_test = torch.load(self.test_ckpt_name)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        [summary]

        Returns
        -------
        torch.utils.data.DataLoader
            [description]
        """        
        return  torch.utils.data.DataLoader(self.dataset_train, **self.dl_kwargs)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        [summary]

        Returns
        -------
        torch.utils.data.DataLoader
            [description]
        """        
        return torch.utils.data.DataLoader(self.dataset_valid,  **self.dl_kwargs)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """
        [summary]

        Returns
        -------
        torch.utils.data.DataLoader
            [description]
        """        
        return torch.utils.data.DataLoader(self.dataset_test,  **self.dl_kwargs)

    @staticmethod
    def add_data_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        [summary]

        Parameters
        ----------
        parent_parser : argparse.ArgumentParser
            [description]

        Returns
        -------
        argparse.ArgumentParser
            [description]
        """
        # Dataset specific
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_file", default="data.csv", type=str, help="The csv file containing the input data and input SMILES.")
        parser.add_argument("--smiles_col", default="SMILES", type=str, help="The columns name for the SMILES strings in the input data table.")
        parser.add_argument("--target_col", nargs = '+', type=str, help="A list of the target column names in the input data table.")
        parser.add_argument("--workers", default=4, type=int, help="Number of CPUs to use for each dataloader collating function.")
        parser.add_argument("--noise", default=0.1, type=float, help="Gaussian noise stddev to apply to the outputs during training.")
        return parser
