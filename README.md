# Retinal Input to the Mouse Superior Colliculus — Digital Twin Model & Analyses

**Paper:** *The functional organization of retinal input to the mouse superior colliculus*
**Authors:** Katrin Franke et al.
**Journal submission:** Cell Press multi-journal (Neuron / Cell Reports / Current Biology)

---

## Overview

This repository accompanies the paper and contains two main components:

1. **Digital twin model** — a deep convolutional neural network trained on natural movie responses of retinal ganglion cell (RGC) axon terminals recorded *in vivo* in the mouse superior colliculus (SC). The model generalizes to parametric stimuli and can be used as an *in silico* platform for probing how retinal input types drive SC computations.

2. **Analyses** — notebooks and scripts for analyses performed both on digital twin model outputs and on directly recorded responses to parametric stimuli. The folder structure mirrors the figures in the paper (see [Analyses](#analyses) below).

> **Data availability:** Experimental data (calcium imaging recordings, retinal datasets) are stored externally. Links to the data will be provided here upon publication.

---

## Repository Structure

```
retina-axon-model/
├── models/                         # Trained digital twin model weights and metadata
└── Analyses/
    ├── Clustering/                 # RGC type clustering & functional diversity analyses
    ├── Aligning_exVivo_inVivo/     # Alignment of in vivo SC bouton responses with
    │                               #   ex vivo mouse retinal datasets
    └── Looming/                    # Collision-detection analyses using the digital twin
```

---

## Digital Twin Model

The `models/` folder contains released model weights for *in silico* experiments:

| File | Description |
|------|-------------|
| `data_retina_sc_multiple_v4a_fluor_model_fac3d_seed_111_v05aaa.pt` | Model trained on 10 scans |
| `data_retina_sc_multiple_v4a_fluor_meta_info_10scans.pkl` | Metadata for the 10-scan model |
| `data_retina_sc_multiple_v4a_fluor_model_fac3d_seed_111_v3aa.pt` | Base model |
| `data_retina_sc_multiple_v4a_fluor_meta_info.pkl` | Metadata for the base model |

**Quick start with Google Colab:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1k9411tLWNcDlUX7nYDsU_grMwafiI3qw?usp=sharing)

---

## Analyses

Analysis code is organized into subfolders that mirror the figures in the paper. Each subfolder contains Jupyter notebooks and/or Python scripts.

### `Analyses/Clustering/`
Functional clustering of RGC response types recorded across superficial SC layers. Analyses characterize the near-complete sampling of RGC functional diversity received by the SC and assess how response types are distributed relative to known *ex vivo* retinal cell-type atlases.

### `Analyses/Aligning_exVivo_inVivo/`
Domain-adversarial representation learning pipeline for aligning *in vivo* SC bouton responses with large-scale *ex vivo* mouse retinal datasets. Used to demonstrate that the SC receives the full complement of functional RGC types without early selection at the retino-collicular synapse.

### `Analyses/Looming/`
Application of the digital twin model to looming stimuli to identify collision-sensitive retinal response types. Demonstrates how a discrete subset of RGC types tuned for collision detection at low angular thresholds is embedded within a largely unselective majority.

---

## Citation

If you use this code or model, please cite our paper (citation will be updated upon publication).

---

## License

See [LICENSE](LICENSE).
