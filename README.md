# AI in Sciences and Engineering (AISE)

<div align="center">

**[Overview](#overview)** |
**[Installation](#installation)** |
**[Experimental Setup](#experimental-setup)**
</div>

<div align="center">
  <table>
    <tr>
      <td><strong>Author:</strong></td>
      <td>Wu, You</td>
    </tr>
    <tr>
      <td><strong>Duration:</strong></td>
      <td>December 17th, 2024 - January 21st, 2025</td>
    </tr>
  </table>
</div>


## Overview

The repository contains implementation of state-of-the-art machine learning architectures for scientific and engineering tasks. It consists of three projects, serving as the final submission for the course [AI in Sciences and Engineering 401-4656-21L](https://github.com/camlab-ethz/AI_Science_Engineering) at ETH Zürich.



| Project | Topic | Task | Report |
| :---: | :--- | :--- | :---: |
| [1](project_1/) | [Fourier Neural Operator (FNO)](https://arxiv.org/abs/2010.08895) | [<img src="assets/done_icon.png" alt="View Task" width="20"/>](assets/project-1-fno-on-1d-wave.pdf) | [<img src="assets/pdf_icon.png" alt="Download Report" width="20"/>](project_1/report.pdf) |
| [2](project_2) | [Data-driven discovery of partial differential equations (PDE-Find)](https://www.science.org/doi/10.1126/sciadv.1602614) | [<img src="assets/done_icon.png" alt="View Task" width="20"/>](assets/project-2-pde-find.pdf) | [<img src="assets/pdf_icon.png" alt="Download Report" width="20"/>](project_2/report.pdf) |
| [3](project_3) | [FNO for Transfer Learning](https://arxiv.org/abs/2306.00258v1) | [<img src="assets/done_icon.png" alt="View Task" width="20"/>](project-3-phase-field-dynamics.pdf) | [<img src="assets/pdf_icon.png" alt="Download Report" width="20"/>](assets/project_3/report.pdf) |

In the projects listed above, we cover a wide range of SciML topics including **operator learning** between infinite-dimensional function spaces, **data-driven system discovery** where we focus on symbolic regression of unknown PDE systems. In the last project, we leverage the technique of transfer learning and aim to use the FNO architecture and build a simple **foundation model for phase-field dynamics**.


## Installation

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

2. Install Git LFS (for fetching trained models under `project_X/checkpoints`):

If you are working on your local machine, the git LFS for large file storage can be installed and used by:

```bash
pip install git-lfs
git lfs install
git lfs pull
```

otherwise you could install git-lfs via some package manager on the cluster like spack with:

```bash
spack install git-lfs
spack load git-lfs # then you can load the module and proceed with pulling LFS files
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Experimental Setup

- Original project handouts locates under the `assets/` directory
- Each project `X ∈ {1, 2, 3}` contains:
    - an associated report `report.pdf` located under `project_X/` folder
    - For each task, we reported our results using maximal two pages
    - Visualizations and results can be found in the `results/` directory


For project reports, experiments were conducted on the `piora` cluster from the student cluster competition team RACKlette @ ETHZ using the hardware & software configuration as follows:


| **CPU Model** | **CPU Core Info** | **Memory** |
|-------------|------------------|------------|
| AMD EPYC 7773X 64-Core Processor | 128 Cores (2 Sockets × 64 Cores), 3.5 GHz max | 1.0 TiB RAM + 15 GiB Swap |
| **Python** | **Deep Learning Framework** | **OS** |
| Python 3.11.9 | PyTorch 2.5.1+cu124 | Rocky Linux 9.4 (Blue Onyx) |
| **GPU** | **CUDA** | **CPU Architecture** |
| NVIDIA A100 80GB PCIe | CUDA 12.4 | x86_64 |