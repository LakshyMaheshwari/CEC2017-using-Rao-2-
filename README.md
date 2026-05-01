<div align="center">
  <h1>Unified CEC2013 & CEC2017 Benchmark Suite</h1>
  <p><b>A Research-Grade Implementation of 58 Metaheuristic Optimization Functions</b></p>

  [![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
  [![Mathematical Parity](https://img.shields.io/badge/mathematics-C_Source_Verified-success.svg)](#)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 📖 Abstract

This repository provides a highly-optimized, mathematically rigorous Python implementation of the **CEC2013** and **CEC2017** Benchmark Suites for Single-Objective Real-Parameter Numerical Optimization. 

Designed for researchers and practitioners, this framework abandons black-box dependencies in favor of **1-to-1 mathematical parity** with the original competition C source codes. It comes pre-packaged with four state-of-the-art parameter-less metaheuristics (**Rao-1, Rao-2, Rao-3, and FISA**) and features heavily accelerated multiprocessing execution.

<div align="center">
  <img src="assets/rao2_F5_3D.png" width="32%" alt="3D Surface of Function 5">
  <img src="assets/rao2_F5_2D.png" width="32%" alt="2D Contour of Function 5">
  <img src="assets/rao2_convergence_D2.png" width="32%" alt="Convergence Graph">
  <p><i>Left: 3D Surface. Middle: 2D Contour Map. Right: Algorithm Convergence Tracking.</i></p>
</div>

---

## ✨ Key Features

- **Pristine Mathematical Fidelity**: Shift vectors ($O_s$) and orthogonal rotation matrices ($M$) are rigorously audited against the official Suganthan `test_func.cpp` files to guarantee competition-standard validation.
- **High-Performance Execution**: Bypasses Python's GIL utilizing `ProcessPoolExecutor` for hardware-accelerated batch testing (e.g., executing 30 independent runs of $2.5 \times 10^5$ FES in seconds).
- **Automated Visualization**: Built-in 2D topological contours, 3D surface mapping, and iteration-by-iteration convergence plotting.
- **Robust IO**: Automated summary CSV generation, graceful error handling for locked files, and cached matrix-loading (`@lru_cache`).

---

## 🧠 Algorithms Implemented

| Algorithm | Description | Characteristic |
|-----------|-------------|----------------|
| **Rao-1** | Best–worst directional perturbation | Highly exploitative |
| **Rao-2** | Rao-1 + random partner interaction term | Balanced exploration |
| **Rao-3** | Rao-2 + second random partner interaction term | Highly explorative |
| **FISA**  | Fitness-based Individual-Step Algorithm | Parameter-free adaptability |

---

## 🚀 Quick Start

The framework is strictly script-based, keeping the workspace transparent and free of heavy Python package clutter.

### 1. Clone the Repository
```bash
git clone https://github.com/LakshyMaheshwari/CEC13-CEC17.git
cd CEC13-CEC17
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Run Interactive Mode
Launch the interactive CLI to choose your algorithm and function visually:
```bash
# For CEC2013:
cd CEC2013
python main.py

# For CEC2017:
cd ../CEC2017
python main.py
```

### 4. Batch Automation (Run All)
Need to compile results for an entire suite overnight?
```bash
cd CEC2013
python run_all.py
```
*This command orchestrates $30$ statistical runs across all algorithms, functions, and dimensions.*

---

## 📂 Architecture

```text
CEC17/
├── CEC2013/
│   ├── algorithms/          # Rao & FISA heuristics
│   ├── functions/           # CEC2013 F1-F28 & original data mappings
│   ├── visualization/       # Graph generation modules
│   ├── main.py              # Interactive CLI
│   └── run_all.py           # Parallelized batch processor
├── CEC2017/
│   └── (Mirrored architecture for CEC2017 F1, F3-F30)
└── tests/                   
    └── test_all_comprehensive.py # Comprehensive 67-test CI validation suite
```

---

## 📈 Output Structure

All execution metrics are automatically structured into publishable formats within the `results/` directory:

```
results/
├── rao2/
│   ├── F1/
│   │   ├── rao2_F1_D10_solution.txt   # Final optimal vector
│   │   ├── rao2_convergence_D2.png    # Plotted graphs
│   │   └── rao2_F1_3D.png
│   └── summary_rao2_D2.csv            # Per-algorithm analytical stats
└── comparison_summary.csv             # Global Wilcoxon-ready cross-algorithm data
```

---

## 🤝 Contributing

Contributions to expand the suite (e.g., CEC2022) or add new metaheuristics (e.g., DE, PSO) are highly encouraged!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/NewAlgorithm`)
3. Ensure the test suite passes (`pytest tests/`)
4. Commit your changes (`git commit -m 'Add New Algorithm'`)
5. Push and open a Pull Request

---

## 📜 Citation

If you utilize this highly-audited framework in your academic research or papers, please cite:

```bibtex
@software{cec_benchmark_rao,
  author = {Lakshya Maheshwari},
  title = {Unified CEC2013 & CEC2017 Benchmark Optimization Suite},
  url = {https://github.com/LakshyMaheshwari/CEC13-CEC17},
  year = {2026},
  note = {Features mathematically verified 1-to-1 C parity.}
}
```