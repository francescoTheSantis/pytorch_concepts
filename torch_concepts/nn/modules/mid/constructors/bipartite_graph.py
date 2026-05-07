"""Spec §7: :class:`BipartiteGraph` — bipartite input ↔ concept directed graph.

Kept in its own module to avoid the circular-import chain that
``constructors/bipartite.py`` (which holds the legacy ``BipartiteModel``)
participates in.
"""
from typing import List

import torch

from .concept_graph import ConceptGraph


class BipartiteGraph(ConceptGraph):
    """Bipartite directed graph for input → concept edges (spec §7).

    A thin :class:`ConceptGraph` subclass whose adjacency is fully
    determined by two disjoint node groups: ``input_nodes`` (sources)
    and ``concept_nodes`` (targets). Every input node is connected to
    every concept node; concept→concept and input→input edges are zero.
    """

    def __init__(
            self,
            input_nodes: List[str],
            concept_nodes: List[str],
    ):
        if not input_nodes:
            raise ValueError("`input_nodes` must be non-empty.")
        if not concept_nodes:
            raise ValueError("`concept_nodes` must be non-empty.")
        overlap = set(input_nodes) & set(concept_nodes)
        if overlap:
            raise ValueError(
                f"`input_nodes` and `concept_nodes` must be disjoint; "
                f"overlap: {sorted(overlap)}"
            )
        all_nodes = list(input_nodes) + list(concept_nodes)
        n = len(all_nodes)
        adj = torch.zeros(n, n)
        n_in = len(input_nodes)
        adj[:n_in, n_in:] = 1.0
        super().__init__(adj, node_names=all_nodes)
        self.input_nodes = list(input_nodes)
        self.concept_nodes = list(concept_nodes)


__all__ = ["BipartiteGraph"]
