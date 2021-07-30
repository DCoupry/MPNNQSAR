#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import torch
from rdkit.Chem.rdMolDescriptors import GetHashedMorganFingerprint
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import DataStructs
from rdkit import Chem
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from k_means_constrained import KMeansConstrained
from pytorch_lightning import seed_everything
from tqdm.auto import tqdm
from itertools import chain
from typing import *
import logging
__author__ = "Damien Coupry"
__copyright__ = "GSK"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Damien Coupry"
__email__ = "damien.x.coupry@gsk.com"
__status__ = "Dev"

# setting up loggers and seeds
logging.getLogger("lightning").setLevel(logging.ERROR)
logger = logging.getLogger("optuna")
seed_everything(42)


def ECFP4(smi: str) -> numpy.ndarray:
    """
    Returns the ECFP4 fingerprint of a smiles

    Parameters
    ----------
    smi : str
        the input smiles

    Returns
    -------
    [type]
        [description]
    """
    mol = Chem.MolFromSmiles(s)
    fp_arr = numpy.zeros((0,), dtype=np.int8)
    fp = GetHashedMorganFingerprint(mol, 2, nBits=2048)
    DataStructs.ConvertToNumpyArray(fp, fp_arr)
    return fp_arr



class ScaffoldSplitter(object):
    """Class for chemically aware splits.

    It is known
    that the random splitting of datasets fro ML purposes
    does not work to assess true variability in chemistry:
    the molecular space is too varied for any dataset to capture
    it entirely. A better way is to train and evaluate models
    on different chemical series entirely. In pharma, this can be
    approximated by  splitting along time of molecule submission. Yet
    this lacks both systematism and repeatability.
    For more detailed justification, see the paper by Yang, Barzilay et al.
    `"Analyzing Learned Molecular Representations for Property Prediction"
    <https://doi.org/10.1021/acs.jcim.9b00237>`_.

    This class clusters Murcko scaffolds using a k-means algorithm,
    then creates a leave-one-out sequence of splits, thus enabling a
    cross validation scheme to appropriately model out-of-chemistry
    performance.
    """

    def __init__(self,
                 smiles: Iterable[str],
                 kx: int = 5,
                 embedder: Optional[Callable] = None
                 ) -> None:
        """
        [summary]

        Parameters
        ----------
        smiles : Iterable[str]
            The input smiles to split.
        k : int, optional
            number of splits to generate, by default 5
        embedder : Optional[Callable], optional
            function taking a SMILES string and returning an embedding vector
            by default None
        """
        self.smiles = smiles
        self.k = k
        self.embedder = embedder if embedder is not None else ECFP4
        return None

    def get_splits(self) -> Tuple[Dict[str, numpy.array], numpy.array]:
        """
        [summary]

        Returns
        -------
        Tuple[Dict[str, numpy.array], numpy.array]
            [description]
        """
        # the output is a list of dictionaries
        splits = []
        # get the scaffold information
        scaffold_dict = self._generate_scaffolds(self.smiles)
        # cluster into k+1 scaffold families
        scaffold_dict = self._cluster_scaffolds(scaffold_dict)
        # sort from smallest to largest families
        scaffold_sets = sorted(scaffold_dict.values(), key=len, reverse=True)
        # first take out the test set
        # this takes the smallest scaffold sets out first
        test_idx = scaffold_sets.pop()
        # now we generate the folds
        set_indices = list(range(len(scaffold_sets)))
        for set_idx in set_indices:
            valid = scaffold_sets[set_idx]
            train = [scaffold_sets[other_idx]
                     for other_idx in set_indices if other_idx != set_idx]
            train = list(chain.from_iterable(train))
            splits.append({"train": numpy.array(train, dtype=numpy.int16),
                           "valid": numpy.array(valid, dtype=numpy.int16)})
        return splits, numpy.array(test_idx, dtype=numpy.int16)

    def _generate_scaffolds(self,
                            smiles: List[str]
                            ) -> Dict[str, List[int]]:
        """
        [summary]

        Parameters
        ----------
        smiles : List[str]
            [description]

        Returns
        -------
        Dict[str, List[int]]
            [description]
        """
        scaffolds = {}
        for idx, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                # invalid smiles, somehow
                continue
            # Compute the Bemis-Murcko scaffold for a SMILES string
            s_kwargs = {"mol": mol,
                        "includeChirality": False}
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(**s_kwargs)
            if scaffold not in scaffolds.keys():
                scaffolds[scaffold] = [idx]
            else:
                scaffolds[scaffold].append(idx)
        return scaffolds

    def _cluster_scaffolds(self,
                           scaffolds: dict
                           ) -> dict:
        """
        Cluster scaffolds in large groups of similarity
        This will reduce the number of scaffold families while retaining
        the differenciation between chemical series.

        Returns
        -------
        [type]
            [description]
        """        
        # embed the scaffolds
        fps, s_keys = zip(*[(self.embedder(s), s) for s in tqdm(scaffolds)])
        fps = numpy.asarray(fps)
        # use PCA on the fingerprintS
        pca = PCA(n_components=8, whiten=True)
        fps = pca.fit_transform(fps)
        # create and fit the k-means clusterer
        size_min = int(0.75 * len(fps) / float(self.k + 1))
        kmeans = KMeansConstrained(n_clusters=self.k + 1, size_min=size_min)
        kmeans.fit(fps)
        clusters = kmeans.labels_
        # store the results  by cluster
        clustered = {i: [] for i in range(self.k + 1)}
        for idx, cluster in enumerate(clusters):
            clustered[cluster] += scaffolds[s_keys[idx]]
        return clustered

