##############################################################################
# Copyright (c) 2021, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################

from typing import Optional

from math import pi as PI

import torch
from torch import Tensor
from torch.nn import Identity, Linear, Sequential
from torch_geometric.nn import GraphNorm, LayerNorm
from torch_geometric.nn.models.schnet import (
    CFConv,
    GaussianSmearing,
    RadiusInteractionGraph,
    ShiftedSoftplus,
)

from .Base import Base


class IEQStack(Base):
    def __init__(
        self,
        num_filters: int,
        num_gaussians: list,
        radius: float,
        *args,
        max_neighbours: Optional[int] = None,
        **kwargs,
    ):
        self.radius = radius
        self.max_neighbours = max_neighbours
        self.num_filters = num_filters
        self.num_gaussians = num_gaussians

        super().__init__(*args, **kwargs)

        self.distance_expansion = GaussianSmearing(0.0, radius, num_gaussians)
        self.interaction_graph = RadiusInteractionGraph(radius, max_neighbours)

        pass

    def _init_conv(self):
        self.graph_convs.append(self.get_conv(self.input_dim, self.hidden_dim))
        self.feature_layers.append(Identity())
        for _ in range(self.num_conv_layers - 1):
            conv = self.get_conv(self.hidden_dim, self.hidden_dim)
            self.graph_convs.append(conv)
            self.feature_layers.append(Identity())

    def get_conv(self, input_dim, output_dim):
        mlp = Sequential(
            Linear(self.num_gaussians, self.num_filters),
            ShiftedSoftplus(),
            Linear(self.num_filters, self.num_filters),
        )

        return IEQConv(
            in_channels=input_dim,
            out_channels=output_dim,
            nn=mlp,
            num_filters=self.num_filters,
            cutoff=self.radius,
        )

    def _conv_args(self, data):
        if (data.edge_attr is not None) and (self.use_edge_attr):
            edge_index = data.edge_index
            edge_weight = data.edge_attr.norm(dim=-1)
        else:
            edge_index, edge_weight = self.interaction_graph(data.pos, data.batch)

        conv_args = {
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "edge_attr": self.distance_expansion(edge_weight),
            "batch": data.batch,
        }

        return conv_args

    def __str__(self):
        return "IEQStack"



class IEQConv(CFConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        nn: Sequential,
        cutoff: float,
    ):
        super().__init__(in_channels, out_channels, num_filters, nn, cutoff)

        self.norm = GraphNorm(in_channels)
        # self.norm = LayerNorm(in_channels)
        # self.lin_layer = Linear(in_channels, out_channels, bias=False)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor,
                edge_attr: Tensor, batch: Tensor) -> Tensor:
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * C.view(-1, 1)

        x_in = self.norm(x, batch)
        x_in = self.lin1(x_in)
        x_in = self.propagate(edge_index, x=x_in, W=W)
        x = self.lin2(x_in) #* self.lin_layer(x)
        return x