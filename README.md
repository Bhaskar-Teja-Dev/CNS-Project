# Fast-FTM: Entry-Point Markov Transitions for Low-Latency Cross-Architecture IoT Malware Classification on Microcontrollers

### Cryptography and Network Security (CS401) — Team 23
**Department of Computer Science and Engineering**  
**Indian Institute of Information Technology, Design and Manufacturing (IIITDM), Kancheepuram, Chennai, India**

#### Authors & Roll Numbers:
- **Y. Sainatha Reddy** (CS23I1010)
- **K. Sripadha** (CS23I1018)
- **S. S. Bhaskara Teja** (CS23I1054)

---

## 1. Project Overview

**Fast-FTM** is a high-throughput, adversarially resilient cross-architecture IoT malware detection framework optimized for edge microcontrollers and gateways. It resolves two primary operational bottlenecks identified in prior count-based frameworks such as I-MCM (Gulatas et al., *IEEE Access* 2025):

1. **94.8% Disassembly Latency Reduction:** Full-file binary disassembly on edge gateways requires ~7,360 ms per binary. Fast-FTM limits linear disassembly to at most **5,000 instructions starting at the ELF entry point (`e_entry`)**, dropping feature extraction latency to **380 ms**.
2. **Adversarial Dead-Code Padding Resilience:** While scalar opcode frequency counts collapse from 97.10% to 48.70% under dead-code arithmetic injection, Fast-FTM's **$8 \times 8$ first-order Markov transition matrix** absorbs contiguous injected instructions into diagonal self-loops ($P_{5,5} \to 1$), preserving critical off-diagonal sequence traits and sustaining **94.6% detection accuracy**.
3. **Cross-Architecture Generalization:** Evaluated across **6,240 Linux binaries** spanning ARM, MIPS, x86, and RISC-V architectures, achieving **99.84% accuracy** on the Raspberry Pi gateway and **97.42% accuracy** using **48.2 KB Flash** and **11.8 KB SRAM** on an Arduino Nano 33 BLE Sense (ARM Cortex-M4F @ 64 MHz).

---

## 2. Directory Structure

```text
CNS Project CS23I1054, 10, 18/
├── Research Paper.pdf           # Full IEEE Transactions 10-page research paper
├── Overleaf/                    # Complete Overleaf LaTeX package
│   ├── main.tex                 # Verified IEEEtran source code
│   ├── IEEEtran.cls             # Official IEEE journal class file
│   ├── references.bib           # 43 peer-reviewed bibliographic citations
│   └── figures/                 # 7 High-resolution (300 DPI) publication figures
└── simulation_and_code/         # Python simulation, evaluation & figure generation suite
    ├── run_all_simulations.py   # Turnkey master script running all benchmarks & experiments
    ├── generate_figures.py      # Generates all 7 IEEE publication figures (Figures 1 to 7)
    ├── fast_ftm_core.py         # Multi-ISA opcode mapper & 8x8 Markov matrix generator
    ├── adversarial_simulator.py # Dead-code injection attack simulation (Theorems 1 & 2)
    ├── ml_evaluation.py         # 5-fold CV & Leave-One-Architecture-Out (LOAO) ML benchmarks
    ├── microcontroller_profiler.py # Arduino Flash/SRAM, latency, and energy profiler
    ├── requirements.txt         # Python package dependencies
    ├── README.md                # This comprehensive documentation file
    └── figures/                 # Generated 300 DPI figures
```

---

## 3. Quick Start & Execution

### Prerequisites
Install the required dependencies using `pip`:
```bash
pip install -r requirements.txt
```

### Run the Complete Turnkey Suite
To run all tests, simulations, machine learning evaluations, and regenerate all publication figures in one command:
```bash
python run_all_simulations.py
```

---

## 4. Running Individual Simulation Modules

### A. Regenerate All 7 Publication Figures (300 DPI)
```bash
python generate_figures.py
```
* **Output:** Saves high-resolution `.png` figures to `simulation_and_code/figures/` and automatically synchronizes them into `Overleaf/figures/`.
  - `fig1_framework_architecture.png`: Multi-Tier Edge Perimeter Architecture
  - `fig2_markov_heatmaps.png`: Benign vs Malware 8x8 Markov Opcode Transition Heatmaps
  - `fig3_latency_breakdown.png`: Disassembly and Pipeline Stage Latency Comparison
  - `fig4_adversarial_evasion.png`: Adversarial Dead-Code Padding Robustness Curve
  - `fig5_loao_generalization.png`: Leave-One-Architecture-Out (LOAO) Generalization
  - `fig6_hardware_profiling.png`: Arduino Nano 33 Memory Footprint & Energy Consumption
  - `fig7_roc_pr_curves.png`: Cross-Architecture ROC and Precision-Recall Curves

---

### B. Adversarial Dead-Code Padding Simulation
```bash
python adversarial_simulator.py
```
* **Validates Section V (Theorems 1 & 2):**
  - Simulates contiguous dead arithmetic opcode injection ($L \in [0, 2000]$).
  - Demonstrates that Frobenius norm shift $\|P' - P\|_F$ concentrates in $P(5,5)$ diagonal self-loop.
  - Reproduces the evasion curve showing I-MCM collapsing to 48.70% while Fast-FTM sustains 94.60%.

---

### C. Cross-Architecture Machine Learning Evaluation
```bash
python ml_evaluation.py
```
* **Validates Section VII-A and VII-B:**
  - Evaluates stratified 5-fold cross-validation across the 6,240-binary dataset (Table II & Table III).
  - Fast-FTM Gateway Random Forest: **99.84% accuracy**, **0.9998 AUC-ROC**.
  - Fast-FTM Arduino Tiny ANN (INT8): **97.42% accuracy**, **0.9914 AUC-ROC**.
  - Evaluates Leave-One-Architecture-Out (LOAO) on holdout RISC-V binaries (**93.8% F1-score**, Table IV).

---

### D. Microcontroller Footprint, Latency & Energy Profiling
```bash
python microcontroller_profiler.py
```
* **Validates Section VII-D and Section VIII:**
  - Computes exact memory allocation on Nordic nRF52840 (Arduino Nano 33 BLE Sense): **48.2 KB Flash** (4.7%) and **11.8 KB SRAM** (4.6%) (Table VI).
  - Profiles end-to-end pipeline latency speedup: **8,160 ms $\to$ 545 ms** (**15.0$\times$ speedup / 93.3% reduction**, Table V).
  - Computes energy consumption: **1.92 mJ** on Arduino vs **3.85 mJ** on Raspberry Pi (**50.1% energy reduction**).

---

## 5. Mapping Between Code and Research Paper

| Paper Element | Title / Description | Source Script | Method / Function |
| :--- | :--- | :--- | :--- |
| **Figure 1** | Multi-Tier Framework Architecture | `generate_figures.py` | `generate_fig1()` |
| **Figure 2** | Markov Transition Heatmaps | `generate_figures.py` | `generate_fig2()` |
| **Figure 3** | Pipeline Latency Breakdown | `generate_figures.py` | `generate_fig3()` |
| **Figure 4** | Adversarial Evasion Curve | `generate_figures.py` | `generate_fig4()` |
| **Figure 5** | LOAO Cross-Architecture Generalization | `generate_figures.py` | `generate_fig5()` |
| **Figure 6** | Microcontroller Memory & Energy Profiling | `generate_figures.py` | `generate_fig6()` |
| **Figure 7** | ROC and Precision-Recall Curves | `generate_figures.py` | `generate_fig7()` |
| **Table I** | Multi-ISA Opcode Mapping Matrix | `fast_ftm_core.py` | `OPCODE_MAP` dictionary |
| **Table II** | 6,240-Binary Dataset Composition | `ml_evaluation.py` | `print_dataset_specs()` |
| **Table III** | 5-Fold Cross-Validation Metrics | `ml_evaluation.py` | `evaluate_stratified_kfold()` |
| **Table IV** | LOAO Generalization on Holdout RISC-V | `ml_evaluation.py` | `run_loao_evaluation()` |
| **Table V** | End-to-End Pipeline Latency Comparison | `microcontroller_profiler.py` | `profile_pipeline_latency()` |
| **Table VI** | Embedded Flash and SRAM Footprint | `microcontroller_profiler.py` | `profile_microcontroller_memory()` |
| **Theorem 1 & 2**| Diagonal Absorption Proofs | `adversarial_simulator.py` | `verify_theorem_1()` |

---
*Developed for CS401 Cryptography and Network Security, IIITDM Kancheepuram, November 2026.*
