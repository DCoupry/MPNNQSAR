#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import pytorch_lightning
import argparse
import torch
import numpy
from typing import Tuple
from typing import List
from typing import Dict
from typing import Callable
from typing import Optional

__author__ = "Damien Coupry"
__copyright__ = "GSK"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Damien Coupry"
__email__ = "damien.x.coupry@gsk.com"
__status__ = "Dev"

NODE_FEATURES = 39
EDGE_FEATURES = 10
pytorch_lightning.seed_everything(42)


def mse_loss_with_nans(input, target):
    # Missing data are nan's
    mask = torch.isnan(target)
    out = (input[~mask]-target[~mask])**2
    loss = out.mean()
    return loss


class GraphAttentionLayer(torch.nn.Module):
    def __init__(self, 
                 hidden_size: int,
                 num_heads: int) -> None:
        """
        [summary]

        Parameters
        ----------
        hidden_size : int
            [description]
        depth : int
            [description]
        dropout : float
            [description]
        """
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(embed_dim=hidden_size, 
                                                num_heads=num_heads)
    
    def forward(self, 
                node_features: torch.Tensor, 
                batch_indices: torch.Tensor
                ) -> torch.Tensor:
        """
        [summary]

        Parameters
        ----------
        node_features : torch.Tensor
            [description]
        batch_indices : torch.Tensor
            [description]

        Returns
        -------
        torch.Tensor
            [description]
        """
        _, seq_lengths = torch.unique(batch_indices, sorted=False, return_counts=True)
        sequences = torch.split(node_features, seq_lengths.tolist())
        src = torch.nn.utils.rnn.pad_sequence(sequences)
        mask = torch.zeros((src.size(1), src.size(0)), dtype=torch.bool, device=src.device)
        for batch_idx, seq_length in enumerate(seq_lengths):
            mask[batch_idx, seq_length:] = True
        return self.attn(src, src, src, key_padding_mask=mask)

class GraphLayer(torch.nn.Module):
    """
    """
    def __init__(self,
                 node_features_size: int,
                 edge_features_size: int, 
                 hidden_size: int,
                 ) -> None:
        """
        [summary]

        Parameters
        ----------
        node_features_size : int
            [description]
        edge_features_size : int
            [description]
        hidden_size : int
            [description]

        Returns
        -------
        [type]
            [description]
        """        
        super().__init__()
        edge_input_size = edge_features_size + node_features_size
        node_input_size = hidden_size + node_features_size
        self.edge_model = torch.nn.Sequential(torch.nn.Linear(edge_input_size, hidden_size, bias=False),
                                              torch.nn.PReLU(hidden_size))
        self.node_model = torch.nn.Sequential(torch.nn.Linear(node_input_size, hidden_size, bias=False),
                                              torch.nn.PReLU(hidden_size))
        self.dropedges = torch.nn.Dropout2d(p=0.1)
        return None

    def forward(self,
                node_features: torch.Tensor,
                edge_index: torch.Tensor,
                edge_features: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        [summary]

        Parameters
        ----------
        node_features : torch.Tensor
            [description]
        edge_index : torch.Tensor
            [description]
        edge_features : torch.Tensor
            [description]

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            [description]
        """        
        receivers, senders = edge_index[0], edge_index[1]
        updated_edge_features = self.edge_model(torch.cat([node_features[senders],
                                                           edge_features], dim=-1))
        updated_edge_features = self.dropedges(updated_edge_features.unsqueeze(0)).squeeze(0)
        node_features_update = torch.zeros(node_features.size(0),
                                           updated_edge_features.size(-1),
                                           dtype=edge_features.dtype,
                                           device=edge_features.device)
        node_features_update = node_features_update.index_add_(0, receivers, updated_edge_features)
        updated_node_features = self.node_model(torch.cat([node_features_update,
                                                           node_features], dim=-1))
        return updated_node_features, updated_edge_features 
        
class DenseGraphBlock(torch.nn.Module):
    def __init__(self, 
                 node_features_size: int,
                 edge_features_size: int, 
                 hidden_size: int,
                 num_heads: int = 4,
                 depth: int = 3) -> None:
        super().__init__()
        node_dims = [node_features_size, ] + [hidden_size for _ in range(depth -1)]
        edge_dims = [edge_features_size, ] + [hidden_size for _ in range(depth -1)]
        graph_dims = list(zip(numpy.cumsum(node_dims), numpy.cumsum(edge_dims)))
        graph_layers = []
        for i, (node_features_size,  edge_features_size) in enumerate(graph_dims):
            gl = GraphLayer(node_features_size=node_features_size, 
                            edge_features_size=edge_features_size, 
                            hidden_size=hidden_size)
            graph_layers.append(gl)
        self.graph_layers = torch.nn.ModuleList(graph_layers)
        self.pool_node = torch.nn.Sequential(torch.nn.Linear(graph_dims[-1][0] + hidden_size, hidden_size * num_heads),
                                             torch.nn.PReLU(hidden_size * num_heads))
        self.pool_edge = torch.nn.Sequential(torch.nn.Linear(graph_dims[-1][1] + hidden_size, hidden_size),
                                             torch.nn.PReLU(hidden_size))
        self.attend_node = GraphAttentionLayer(hidden_size, num_heads)

    def forward(self, 
                n: torch.Tensor, 
                e: torch.Tensor,
                e_i: torch.Tensor,
                batch: torch.tensor,
                ) -> torch.Tensor:
        Ns = [n, ]
        Es = [e, ]
        for gl in self.graph_layers:
            ni, ei = gl(n, e_i, e)
            Ns.append(ni)
            Es.append(ei)
            n = torch.cat(Ns, dim=-1)
            e = torch.cat(Es, dim=-1)
        n = self.pool_node(n)
        e = self.pool_edge(e)
        u = self.attend_node(n, batch)
        return n, e, u

class DenseGraphNetwork(torch.nn.Module):
    def __init__(self,
                 hidden_size: int,
                 output_size: int = 1,
                 graph_depth: int = 3,
                 fc_depth: int = 3,
                 dropout: float = 0.1,
                 extra_features: Optional[List[torch.nn.Module]] = None,
                 ) -> None:
        """
        [summary]

        Parameters
        ----------
        hidden_size : int
            [description]
        output_size : int, optional
            [description], by default 1
        graph_depth : int, optional
            [description], by default 3
        fc_depth : int, optional
            [description], by default 5
        dropout : float, optional
            [description], by default 0.1

        Returns
        -------
        [type]
            [description]
        """        
        super().__init__()
        self.embed_size = hidden_size
        graph_dims = [(NODE_FEATURES, EDGE_FEATURES)] + [(hidden_size, hidden_size) for _ in range(graph_depth - 1)]
        graph_layers = []
        for i, (node_features_size,  edge_features_size) in enumerate(graph_dims):
            gl = DenseGraphBlock(node_features_size=node_features_size, 
                                edge_features_size=edge_features_size, 
                                hidden_size=hidden_size,
                                num_heads=4,
                                depth=3)
            graph_layers.append(gl)
        self.graph_layers = torch.nn.ModuleList(graph_layers)
        fc_dims = [NODE_FEATURES + graph_depth * hidden_size] + [hidden_size for _ in range(fc_depth - 1)]
        fc_layers = []    
        for fc_dim in fc_dims:
            fc_layers += [torch.nn.Linear(fc_dim, hidden_size, bias=True),
                          torch.nn.PReLU(hidden_size),
                          torch.nn.Dropout(p=dropout),]
        self.fc_layers = torch.nn.Sequential(*fc_layers)
        self.output_layer = torch.nn.Linear(hidden_size, output_size, bias=True)
        return None

    @torch.jit.export
    def encode(self,
                x: Tuple[torch.Tensor,
                         torch.Tensor,
                         torch.Tensor,
                         torch.Tensor]
                ) -> torch.Tensor:
        """
        [summary]

        Parameters
        ----------
        x : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            [description]

        Returns
        -------
        torch.Tensor
            [description]
        """        
        n, e, e_i, batch = x
        u = torch.zeros(batch.max()+1, n.size(1), dtype=n.dtype, device=n.device)
        u = u.index_add_(0, batch, n)
        densify_u = [u, ]
        for gl in self.graph_layers:
            n, e, u = gl(n, e, e_i, batch)
            Us.append(u)
        u = torch.cat(Us, dim=-1)
        u = self.fc_layers(u)
        return u

    def forward(self,
                x: Tuple[torch.Tensor,
                         torch.Tensor,
                         torch.Tensor,
                         torch.Tensor]
                ) -> torch.Tensor:
        """
        [summary]

        Parameters
        ----------
        x : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            [description]

        Returns
        -------
        torch.Tensor
            [description]
        """        
        latent = self.encode(x)
        logits = self.output_layer(latent)
        return logits

class GraphNetwork(torch.nn.Module):
    def __init__(self,
                 hidden_size: int,
                 output_size: int = 1,
                 graph_depth: int = 3,
                 fc_depth: int = 3,
                 dropout: float = 0.1,
                 extra_features: Optional[List[torch.nn.Module]] = None,
                 ) -> None:
        """
        [summary]

        Parameters
        ----------
        hidden_size : int
            [description]
        output_size : int, optional
            [description], by default 1
        graph_depth : int, optional
            [description], by default 3
        fc_depth : int, optional
            [description], by default 5
        dropout : float, optional
            [description], by default 0.1

        Returns
        -------
        [type]
            [description]
        """        
        super().__init__()
        self.embed_size = hidden_size
        node_dims = [NODE_FEATURES, ] + [hidden_size for _ in range(graph_depth - 1)]
        edge_dims = [EDGE_FEATURES, ] + [hidden_size for _ in range(graph_depth - 1)]
        graph_dims = list(zip(node_dims, edge_dims))
        graph_layers = []
        for i, (node_features_size,  edge_features_size) in enumerate(graph_dims):
            gl = GraphLayer(node_features_size=node_features_size, 
                            edge_features_size=edge_features_size, 
                            hidden_size=hidden_size)
            graph_layers.append(gl)
        self.graph_layers = torch.nn.ModuleList(graph_layers)
        fc_dims = [hidden_size for _ in range(fc_depth)]
        fc_layers = []    
        for fc_dim in fc_dims:
            fc_layers += [torch.nn.Linear(fc_dim, hidden_size, bias=True),
                          torch.nn.PReLU(hidden_size),
                          torch.nn.Dropout(p=dropout),]
        self.fc_layers = torch.nn.Sequential(*fc_layers)
        self.output_layer = torch.nn.Linear(hidden_size, output_size, bias=True)
        return None

    @torch.jit.export
    def encode(self,
                x: Tuple[torch.Tensor,
                         torch.Tensor,
                         torch.Tensor,
                         torch.Tensor]
                ) -> torch.Tensor:
        """
        [summary]

        Parameters
        ----------
        x : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            [description]

        Returns
        -------
        torch.Tensor
            [description]
        """        
        n, e, e_i, batch = x
        for gl in self.graph_layers:
            n, e = gl(n, e_i, e)
        u = torch.zeros(batch.max()+1, n.size(1), dtype=n.dtype, device=n.device)
        u = u.index_add_(0, batch, n)
        u = self.fc_layers(u)
        return u

    def forward(self,
                x: Tuple[torch.Tensor,
                         torch.Tensor,
                         torch.Tensor,
                         torch.Tensor]
                ) -> torch.Tensor:
        """
        [summary]

        Parameters
        ----------
        x : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            [description]

        Returns
        -------
        torch.Tensor
            [description]
        """        
        latent = self.encode(x)
        logits = self.output_layer(latent)
        return logits


class GraphNetworkLigthning(pytorch_lightning.LightningModule):
    """
    """
    def __init__(self,
                 hparams: argparse.Namespace,
                 ) -> None:
        """
        [summary]

        Parameters
        ----------
        hparams : argparse.Namespace
            [description]

        Returns
        -------
        [type]
            [description]
        """        
        super().__init__()
        self.hparams = hparams
        extra_features = None
        if self.hparams.extra is not None:
            extra_features = [torch.load(xfeat) for xfeat in self.hparams.extra]
        self._net = GraphNetwork(hidden_size=self.hparams.hidden_size,
                                 output_size=self.hparams.output_size,
                                 graph_depth=self.hparams.graph_depth,
                                 fc_depth=self.hparams.fc_depth,
                                 dropout=self.hparams.dropout,
                                 extra_features=extra_features)
        if self.hparams.task == "classification":
            self._loss = torch.nn.BCEWithLogitsLoss()
        elif self.hparams.task == "regression":
            self._loss = mse_loss_with_nans
        return None

    def forward(self,
                x: Tuple[torch.Tensor,
                         torch.Tensor,
                         torch.Tensor,
                         torch.Tensor]) -> torch.Tensor:
        """
        [summary]

        Parameters
        ----------
        x : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            [description]

        Returns
        -------
        torch.Tensor
            [description]
        """        
        logits = self._net(x)
        return logits

    def training_step(self,
                      batch: Tuple[Tuple[torch.Tensor,
                                         torch.Tensor,
                                         torch.Tensor,
                                         torch.Tensor],
                                   torch.Tensor],
                      batch_nb: int) -> float:
        """
        [summary]

        Parameters
        ----------
        batch : Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
            [description]
        batch_nb : int
            [description]

        Returns
        -------
        float
            [description]
        """        
        x, y = batch
        yhats = self.forward(x)
        loss = self._loss(yhats, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self,
                        batch: Tuple[Tuple[torch.Tensor,
                                           torch.Tensor,
                                           torch.Tensor,
                                           torch.Tensor],
                                     torch.Tensor],
                        batch_nb: int) -> float:
        """
        [summary]

        Parameters
        ----------
        batch : Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
            [description]
        batch_nb : int
            [description]

        Returns
        -------
        float
            [description]
        """        
        x, y = batch
        yhats = self.forward(x)
        loss = self._loss(yhats, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        [summary]

        Returns
        -------
        torch.optim.Optimizer
            [description]
        """
        return torch.optim.Adam(self._net.parameters(), lr=1e-4)
        # 
        # optimizer = torch.optim.AdamW(self._net.parameters(), 
        #                         lr=self.hparams.learning_rate,
        #                         weight_decay=self.hparams.weight_decay,
        #                         betas=(self.hparams.beta1, self.hparams.beta2))
        # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
        #                                               base_lr=1e-6, max_lr=self.hparams.learning_rate, 
        #                                               step_size_up=self.hparams.lr_cycle_size, 
        #                                               mode='exp_range', gamma=self.hparams.gamma, 
        #                                               cycle_momentum=False)
        # return [optimizer], [scheduler]



class GraphNetworkEnsemble(torch.nn.Module):
    """
    """    
    def __init__(self, 
                 models: List[torch.nn.Module],
                 prediction_names: List[str]) -> None:
        """
        [summary]

        Parameters
        ----------
        models : List[torch.nn.Module]
            [description]
        """        
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.names = prediction_names
        self.embed_size = len(self.models) * models[0].output_layer.weight.size(1)

    @torch.jit.export
    def encode(self, 
               x: Tuple[torch.Tensor,
                        torch.Tensor,
                        torch.Tensor,
                        torch.Tensor]
               ) -> torch.Tensor:
        """
        [summary]

        Parameters
        ----------
        x : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            [description]

        Returns
        -------
        torch.Tensor
            [description]
        """        
        outs = []
        for model in self.models:
            this_out = model.encode(x)
            outs.append(this_out)
        return torch.cat(outs, dim=-1)

    def forward(self, 
                x: Tuple[torch.Tensor,
                         torch.Tensor,
                         torch.Tensor,
                         torch.Tensor]
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        [summary]

        Parameters
        ----------
        x : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            [description]

        TODO: bugs when models being ensembled have not the same output dim

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            [description]
        """        
        # first enable dropout
        for m in self.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()
        outs = []
        for model in self.models:
            for bayes_iter in range(10):
                this_out = model(x)
                outs.append(this_out)
        outs = torch.stack(outs, dim=0)
        logits = outs.mean(dim=0)
        std = outs.std(dim=0)
        return logits, std


class GraphNetworkEnsembleLightning(pytorch_lightning.LightningModule):
    """
    """    
    def __init__(self, 
                 models: List[torch.nn.Module], 
                 loss: torch.nn.Module,
                 prediction_names: List[str]
                 ) -> None:
        """
        [summary]

        Parameters
        ----------
        models : List[torch.nn.Module]
            [description]
        loss : torch.nn.Module
            [description]
        """        
        super().__init__()        
        self._net = GraphNetworkEnsemble(models, prediction_names)
        self._loss = loss

    def forward(self, 
                x: Tuple[torch.Tensor,
                         torch.Tensor,
                         torch.Tensor,
                         torch.Tensor]
               ) -> torch.Tensor:
        """
        [summary]

        Parameters
        ----------
        x : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            [description]

        Returns
        -------
        torch.Tensor
            [description]
        """        
        logits, std = self._net(x)
        return logits
    
    def training_step(self,
                      batch: Tuple[Tuple[torch.Tensor,
                                         torch.Tensor,
                                         torch.Tensor,
                                         torch.Tensor],
                                   torch.Tensor],
                      batch_nb: int) -> None:
        """
        [summary]

        Parameters
        ----------
        batch : Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
            [description]
        batch_nb : int
            [description]
        """        
        return 

    def test_step(self,
                  batch: Tuple[Tuple[torch.Tensor,
                                     torch.Tensor,
                                     torch.Tensor,
                                     torch.Tensor],
                               torch.Tensor],
                  batch_nb: int) -> float:
        """
        [summary]

        Parameters
        ----------
        batch : Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
            [description]
        batch_nb : int
            [description]

        Returns
        -------
        float
            [description]
        """        
        x, y = batch
        yhats = self.forward(x)
        loss = self._loss(yhats, y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self) -> None:
        """
        [summary]
        """        
        return 


class ExplainerNetwork(torch.nn.Module):
    def __init__(self) -> None:
        """
        """
        super().__init__()
        HIDDEN_SIZE = 32
        EDGE_INPUT_SIZE = EDGE_FEATURES + 2 * NODE_FEATURES
        NODE_INPUT_SIZE = HIDDEN_SIZE + NODE_FEATURES
        self.edge_model = torch.nn.Sequential(torch.nn.Linear(EDGE_INPUT_SIZE, HIDDEN_SIZE),
                                              torch.nn.Tanh(),
                                              torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                                              torch.nn.Tanh())
        self.node_model = torch.nn.Sequential(torch.nn.Linear(NODE_INPUT_SIZE, HIDDEN_SIZE),
                                              torch.nn.Tanh(),
                                              torch.nn.Linear(HIDDEN_SIZE, 1))
        return None

    def forward(self,
                x: Tuple[torch.Tensor,
                         torch.Tensor,
                         torch.Tensor,
                         torch.Tensor]
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        """
        n, e, e_i, batch = x
        e_up = self.edge_model(torch.cat([n[e_i[0]], e, n[e_i[1]]], dim=-1))
        n_up = torch.zeros(n.size(0), e_up.size(-1), device=e.device, dtype=e.dtype)
        n_up = n_up.index_add_(0, e_i[0], e_up)
        atomic_embed = self.node_model(torch.cat([n_up, n], dim=-1))
        return atomic_embed


class ExplainerNetworkLightning(pytorch_lightning.LightningModule):
    """
    """
    def __init__(self, task : str = "regression", property_index : int = 0) -> None:
        """
        [summary]

        Parameters
        ----------
        task : str, optional
            [description], by default "regression"

        Returns
        -------
        [type]
            [description]
        """
        super().__init__()
        self._idx = property_index
        self._net = ExplainerNetwork()
        if task == "classification":
            self._loss = torch.nn.BCEWithLogitsLoss()
        elif task == "regression":
            self._loss = mse_loss_with_nans
        return None

    def forward(self,
                x: Tuple[torch.Tensor,
                         torch.Tensor,
                         torch.Tensor,
                         torch.Tensor]) -> torch.Tensor:
        """
        [summary]

        Parameters
        ----------
        x : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            [description]

        Returns
        -------
        torch.Tensor
            [description]
        """        
        batch = x[-1]
        atomic_embed = self._net(x)
        logits = torch.zeros(batch.max()+1, 
                             atomic_embed.size(1), 
                             dtype=atomic_embed.dtype, 
                             device=atomic_embed.device)
        return logits.index_add_(0, batch, atomic_embed)

    def training_step(self,
                      batch: Tuple[Tuple[torch.Tensor,
                                         torch.Tensor,
                                         torch.Tensor,
                                         torch.Tensor],
                                   torch.Tensor],
                      batch_nb: int) -> float:
        """
        [summary]

        Parameters
        ----------
        batch : Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
            [description]
        batch_nb : int
            [description]

        Returns
        -------
        float
            [description]
        """        
        x, y = batch
        y = y[:, self._idx].reshape(-1, 1)
        yhats = self.forward(x)
        loss = self._loss(yhats, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        return loss

    def validation_step(self,
                        batch: Tuple[Tuple[torch.Tensor,
                                           torch.Tensor,
                                           torch.Tensor,
                                           torch.Tensor],
                                     torch.Tensor],
                        batch_nb: int) -> float:
        """
        [summary]

        Parameters
        ----------
        batch : Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
            [description]
        batch_nb : int
            [description]

        Returns
        -------
        float
            [description]
        """        
        x, y = batch
        y = y[:, self._idx].reshape(-1, 1)
        yhats = self.forward(x)
        loss = self._loss(yhats, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        [summary]

        Returns
        -------
        torch.optim.Optimizer
            [description]
        """        
        opt = torch.optim.AdamW(self.parameters(), lr=3e-4)
        return opt
