"""Correct two-segment featurization and scoring for ATOMICA.

The original pipeline (`scripts/process_expert_atomica.py`) fed ATOMICA a single
segment with the whole pocket collapsed into one `UNK` block, which engages none
of the model's interaction semantics and erases its block vocabulary. This package
builds inputs the way ATOMICA's own dataset pipeline does: chemically-typed blocks,
two segments, one per interacting entity.

Submodules are imported on demand rather than eagerly: `featurize` pulls in ATOMICA
and therefore torch, while `scoring`'s metrics are numpy-only and must stay usable
(and testable) without a torch environment.

See docs/experiment-plan.md.
"""

__all__ = [
    "atom_segment_ids",
    "interface_data",
    "ligand_blocks_from_arrays",
    "ligand_blocks_from_mol",
    "pocket_blocks_from_arrays",
    "pocket_blocks_from_pdb",
    "summarize",
    "to_batch",
]

_FEATURIZE_EXPORTS = frozenset(__all__)


def __getattr__(name):
    if name in _FEATURIZE_EXPORTS:
        from atomica_interface import featurize

        return getattr(featurize, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
