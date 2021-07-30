#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
from dmpnn.data import featurize

import argparse
import torch
import numpy
import pandas
import matplotlib.pyplot    
from tqdm.auto import tqdm
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, kendalltau

__author__ = "Damien Coupry"
__copyright__ = "GSK"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Damien Coupry"
__email__ = "damien.x.coupry@gsk.com"
__status__ = "Dev"


def plot_regression(y_true, y_pred, std, name="output"):
    y_true = y_true.values.ravel()
    y_pred = y_pred.values.ravel()
    r2 = r2_score(y_true, y_pred)
    kt, _ = kendalltau(y_true, y_pred)
    pr, _ = pearsonr(y_true, y_pred)
    label = f"R2    = {r2:.2f}\nKTau = {kt:.2f}\nCorr  = {pr:.2f}"
    axlims = (min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max()))
    yeqx = numpy.linspace(axlims[0], axlims[1], 100)
    ax = matplotlib.pyplot.hexbin(y_true, y_pred, mincnt=1, gridsize=30, cmap="YlOrRd", bins='log')
    cbar = matplotlib.pyplot.colorbar()
    cbar.set_label("Bincount [log].")
    ax = matplotlib.pyplot.gca()
    ax.plot(yeqx, yeqx, ":k", label=label)
    ax.set_xlim(*axlims)
    ax.set_ylim(*axlims)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(f"Actual {name}")
    ax.set_ylabel(f"Predicted {name}")
    ax.legend(handlelength=0, handletextpad=0, frameon=False)
    matplotlib.pyplot.savefig(f"{name}.svg")
    matplotlib.pyplot.clf()


def main(args: argparse.Namespace) -> None:
    """
    [summary]

    Parameters
    ----------
    args : argparse.Namespace
        [description]

    Returns
    -------
    [type]
        [description]
    """    
    model = torch.load(args.model)
    if hasattr(model, "names"):
        names = model.names
    else:
        names = [str(i) for i in range(model.output_layer.weight.size(1))]
    data = pandas.read_csv(args.input_file, index_col=args.smiles_col)
    smiles = data.index.values
    data["graphs"] = featurize(smiles)
    data = data.dropna()
    graphs = [(x[0], x[1], x[2], torch.zeros(len(x[0]), dtype=torch.long)) for x in data.graphs.values]
    if torch.cuda.is_available():
        model = model.cuda().eval()
    predicted = []
    for x in tqdm(graphs):
        try:
            if torch.cuda.is_available():
                x = tuple([xx.cuda() for xx in x])
            p = model(x)
            if not isinstance(p, tuple):
                p = (p, torch.zeros(1,1, dtype=torch.float))
        except Exception as e:
            nan = torch.FloatTensor([[float("nan")]])
            p = (nan, nan)
        predicted.append(p)
    y_pred, y_std = zip(*predicted)
    if args.task == "classification":
        y_pred = [torch.sigmoid(x) for x in y_pred]
    y_pred = numpy.concatenate([x.cpu().detach().numpy() for x in y_pred], axis=0)
    y_std = numpy.concatenate([x.cpu().detach().numpy() for x in y_std], axis=0)
    for i in range(y_pred.shape[-1]):
        data[f"Prediction_{names[i]}"] = y_pred[:,i]
        data[f"Std_{names[i]}"] = y_std[:,i]
    data = data.dropna()
    data.drop("graphs", axis=1).to_csv(args.output_file)
    if args.task == "regression" and args.true_label_col is not None :
        for i, label in zip(range(y_pred.shape[-1]), args.true_label_col):
            plot_regression(y_true=data[label], 
                            y_pred=data[f"Prediction_{names[i]}"], 
                            std=data[f"Std_{names[i]}"],
                            name=names[i])
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--model", type=str)
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--smiles_col", default="SMILES", type=str)
    parser.add_argument("--true_label_col", nargs="+", default=None, type=str)
    parser.add_argument('--task', choices=["regression", "classification"], type=str)
    args = parser.parse_args()
    # run the code
    main(args)
