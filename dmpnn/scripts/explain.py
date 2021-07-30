#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
from ..data import featurize_smile

import argparse
import torch
import numpy
import pandas
import matplotlib.pyplot
from rdkit.Chem import RemoveHs, MolFromSmiles
from rdkit.Chem.Draw import SimilarityMaps
# from captum.attr import IntegratedGradients, NoiseTunnel

__author__ = "Damien Coupry"
__copyright__ = "GSK"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Damien Coupry"
__email__ = "damien.x.coupry@gsk.com"
__status__ = "Dev"


def plot_rationale(args):
    attributions = get_attributions_GE(args)
    mol = RemoveHs(MolFromSmiles(args.smiles))
    fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, weights=attributions)
    fig.savefig(f"{args.property}.png", bbox_inches="tight")
    return

def get_attributions_GE(args):
    model = torch.load(f"{args.project}/graph_{args.property}_explainer.ckpt")
    n, e, e_i = featurize_smile(args.smiles)
    batch = torch.zeros(len(n), dtype=torch.long)
    attributions = model((n, e, e_i, batch)).detach().numpy()
    return attributions

# def get_attributions_IG(args):
#     """
#     [summary]

#     Parameters
#     ----------
#     args : argparse.Namespace
#         [description]

#     Returns
#     -------
#     [type]
#         [description]
#     """    
#     model = torch.load(args.model)
#     n, e, e_i = featurize_smile(args.smiles)
#     batch = torch.zeros(len(n), dtype=torch.long)
#     n = n.unsqueeze(0)
#     e = e.unsqueeze(0)
#     # print("0 :", n.shape, e.shape, e_i.shape, batch.shape)
#     baseline_n = torch.zeros_like(n)
#     baseline_e = torch.zeros_like(e)
#     baseline = (baseline_n, baseline_e)
#     def forward_fn(n, e):
#         n = n.squeeze(0)
#         e = e.squeeze(0)
#         # print("1 :", n.shape, e.shape)
#         out, std = model((n, e, e_i, batch))
#         return out
#     ig = IntegratedGradients(forward_fn)
#     # nt = NoiseTunnel(ig)
#     attributions_n, attributions_e = ig.attribute((n, e), 
#                                                 #   nt_type='smoothgrad',
#                                                 #   stdevs=0.02, 
#                                                 #   n_samples=10, 
#                                                   baselines=baseline, 
#                                                   internal_batch_size=1, 
#                                                   target=0)
#     attributions_n = (attributions_n ** 2.0).sum(dim=-1).detach().numpy()
#     attributions_e = (attributions_e ** 2.0).sum(dim=-1).detach().numpy()
#     return attributions_n, attributions_e

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--project", type=str)
    parser.add_argument("--property", type=str)
    parser.add_argument("--smiles", type=str)
    args = parser.parse_args()
    # run the code
    plot_rationale(args)
