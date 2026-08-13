# Third-party notices

This repository vendors two upstream research codebases in place, under
`ATOMICA/` and `DiffSBDD/`. They are **not** unmodified dependencies pulled from
a package index: `DiffSBDD/` carries substantial local changes, which are the
contribution of this project and are itemised in [MODIFICATIONS.md](MODIFICATIONS.md).
`ATOMICA/` is used as-is.

Both upstream projects are MIT licensed, which permits redistribution with
modification provided the original copyright and permission notices are
retained. The full upstream licenses remain in place at `ATOMICA/LICENSE` and
`DiffSBDD/LICENSE`; the notices are reproduced below.

They are vendored rather than pinned as dependencies because the conditioning
mechanism required editing DiffSBDD's denoiser internals — `EGNNDynamics` has no
extension point for an external conditioning signal, so the change cannot be
expressed as a subclass or a plugin.

---

## ATOMICA

Universal representations of intermolecular interactions.
Upstream: <https://github.com/mims-harvard/ATOMICA>
Used **unmodified** as a frozen pocket encoder.

```
MIT License

Copyright (c) 2024 Artificial Intelligence for Medicine and Science @ Harvard
```

```bibtex
@article{fang2025atomica,
  title={Learning Universal Representations of Intermolecular Interactions with ATOMICA},
  author={Fang, Ada and Desgagn\'{e}, Micha\"{e}l and Zhang, Zaixi and Zhou, Andrew and
          Loscalzo, Joseph and Pentelute, Bradley L and Zitnik, Marinka},
  journal={In review},
  year={2025},
  url={https://www.biorxiv.org/content/10.1101/2025.04.02.646906}
}
```

## DiffSBDD

SE(3)-equivariant diffusion for structure-based drug design.
Upstream: <https://github.com/arneschneuing/DiffSBDD>
**Modified** — see [MODIFICATIONS.md](MODIFICATIONS.md).

```
MIT License

Copyright (c) 2022 Arne Schneuing, Yuanqi Du, Charles Harris
```

```bibtex
@article{schneuing2024diffsbdd,
  title={Structure-based drug design with equivariant diffusion models},
  author={Schneuing, Arne and Harris, Charles and Du, Yuanqi and Didi, Kieran and
          Jamasb, Arian and Igashov, Ilia and Du, Weitao and Gomes, Carla and
          Blundell, Tom L and Lio, Pietro and Welling, Max and Bronstein, Michael
          and Correia, Bruno},
  journal={Nature Computational Science},
  year={2024}, volume={4}, number={12}, pages={899--909},
  doi={10.1038/s43588-024-00737-x}
}
```

## Other components

Invoked as external tools or libraries, not vendored: **AutoDock Vina** /
**smina** (docking), **Boltz-2** (affinity prediction), **ADMET-AI** (property
prediction), **RDKit** (cheminformatics). Each carries its own license; consult
the respective projects.

Training complexes come from the **CrossDocked2020** dataset, which has its own
terms of use.
