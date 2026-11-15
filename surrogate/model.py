"""Graph neural network power flow surrogate for the Kentucky cut.

The topology is fixed apart from the outaged branch, so the model runs on a
batch of node-feature tensors [B, N, F] over one shared edge list, with a
per-sample edge mask [B, E] that switches the outaged branch off. Messages
use scatter-add (index_add_) so no PyG kernels are needed and the same code
runs on CPU and GPU.

Architecture (edge-conditioned message passing, GNS/PowerFlowNet style):
  node encoder -> L x [edge message MLP(h_i, h_j, e_ij) * mask, summed in both
  directions, + global mean context] with residual + LayerNorm ->
  node head (Vm, Va) and edge head (P_from, Q_from, loading/100).

`physics_flows` turns predicted voltages into branch flows with the branch
pi-model; it is used as an optional consistency loss and for evaluation.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def mlp(i, h, o, n=2):
    layers, d = [], i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class MPLayer(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.msg_fwd = mlp(3 * h, h, h)
        self.msg_bwd = mlp(3 * h, h, h)
        self.upd = mlp(3 * h, h, h)
        self.edge_upd = mlp(3 * h, h, h)
        self.norm = nn.LayerNorm(h)
        self.enorm = nn.LayerNorm(h)

    def forward(self, h, e, src, dst, mask):
        # h [B,N,H]  e [B,E,H]  mask [B,E,1]
        hs, hd = h[:, src], h[:, dst]
        m_f = self.msg_fwd(torch.cat([hs, hd, e], -1)) * mask    # src -> dst
        m_b = self.msg_bwd(torch.cat([hd, hs, e], -1)) * mask    # dst -> src
        agg = torch.zeros_like(h)
        agg.index_add_(1, dst, m_f)
        agg.index_add_(1, src, m_b)
        ctx = h.mean(1, keepdim=True).expand_as(h)
        h = self.norm(h + self.upd(torch.cat([h, agg, ctx], -1)))
        e = self.enorm(e + self.edge_upd(torch.cat([h[:, src], h[:, dst], e], -1)))
        return h, e


class PFSurrogate(nn.Module):
    def __init__(self, n_node_feat, n_edge_feat, hidden=128, layers=12):
        super().__init__()
        self.node_enc = mlp(n_node_feat, hidden, hidden)
        self.edge_enc = mlp(n_edge_feat + 1, hidden, hidden)
        self.layers = nn.ModuleList([MPLayer(hidden) for _ in range(layers)])
        self.node_head = mlp(hidden, hidden, 2)          # dVm (from setpoint/1.0), Va
        self.edge_head = mlp(3 * hidden, hidden, 3)      # P_from, Q_from, loading/100

    def forward(self, x, edge_attr, edge_index, mask):
        """x [B,N,F] (normalised), edge_attr [E,Fe] (normalised), mask [B,E] (1 = in service)."""
        src, dst = edge_index
        B = x.shape[0]
        m = mask.unsqueeze(-1)
        e = self.edge_enc(torch.cat([edge_attr.unsqueeze(0).expand(B, -1, -1), m], -1))
        h = self.node_enc(x)
        for layer in self.layers:
            h, e = layer(h, e, src, dst, m)
        node = self.node_head(h)
        edge = self.edge_head(torch.cat([h[:, src], h[:, dst], e], -1)) * m
        return node, edge


def physics_flows(vm, va, br, src, dst, mask):
    """Branch from/to complex power (pu) from predicted voltages.

    br: dict of tensors yff, yft, ytf, ytt as (real, imag) pairs, shape [E].
    """
    V = torch.polar(vm, va)                     # [B,N] complex
    Vf, Vt = V[:, src], V[:, dst]
    yff = torch.complex(*br["yff"]); yft = torch.complex(*br["yft"])
    ytf = torch.complex(*br["ytf"]); ytt = torch.complex(*br["ytt"])
    Sf = Vf * torch.conj(yff * Vf + yft * Vt)
    St = Vt * torch.conj(ytf * Vf + ytt * Vt)
    return Sf * mask, St * mask
