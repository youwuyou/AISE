# AI in Sciences and Engineering - Final Projects

This repository contains the implementation of the final projects for the course AI in Sciences and Engineering 401-4656-21L at ETH Zürich.

## Installation

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

2. Install Git LFS (for fetching trained models under `project_X/checkpoints`):

```bash
pip install git-lfs
git lfs install
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Repository Structure

The repository consists of three main projects, we try to resolve all projects using both AI-based and traditional numerical approaches:

1. Training FNO to Solve the 1D Wave Equation
2. PDE-Find: Reconstructing PDEs from Data
3. Foundation Models for Phase-Field Dynamics

## Progress Overview

| Task  | ML-based Implementation | Numerical Solver Implementation | Documentation |
|-------------------------------|-------------------|------------------|---------------|
| 1. 1D Wave Equation (FNO) | ✅  | N/A   | ✅  |
| Bonus 1: All2all Training     | ✅    | N/A | ⏳  |
| 2. PDE-Find   | ⏳ Not Started    | ⏳ Not Started   | ⏳ |
| 3. Phase-Field Dynamics    | ⏳ Not Started    | ⏳ Not Started   | ⏳  |
| Bonus 2: Stability Analysis    | ⏳ Not Started    | N/A    | ⏳            |

### Legend:
- ✅ Complete
- 🚧 In Progress
- ⏳ Not Started
- ❌ Blocked
- N/A Not Applicable


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