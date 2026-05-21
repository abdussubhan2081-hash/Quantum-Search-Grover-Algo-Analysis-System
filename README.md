# Quantum-Search-Grover-Algo-Analysis-System

# ⚛️ Quantum Search Analysis System

A Streamlit web application that compares **Classical Linear Search $O(N)$** with **Grover's Quantum Search $O(\sqrt{N})$**, visualised interactively with circuit diagrams, complexity scaling curves, and measurement probability histograms.

---

## 🖥️ Features

- **Manual Entry or File Upload** — supports `.txt`, `.pdf`, and `.docx` datasets.
- **Classical Linear Search** — sequential scan with step counting and precise timing.
- **Grover's Algorithm** — simulated on Qiskit's `AerSimulator` with dynamic circuit construction.
- **Quantum Result Verification** — decodes the most probable state and computes the success probability.
- **Complexity Chart** — visual comparison showing $O(N)$ vs $O(\sqrt{N})$ scaling with the current dataset size marked.
- **Quantum Measurement Histogram** — shows measurement counts distribution with the target state highlighted in green.
- **Compiled Circuit Diagram** — draws the Grover circuit as a clean matplotlib figure (with ASCII fallback).
- **Educational Explanations** — explains qubits, superposition, oracle, and the diffusion operator in details.

---

## 🛠️ Setup & Installation

### 1. Place the project files in a folder
Ensure your folder structure contains:
```
Quantum Project/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

### 2. Create a virtual environment (recommended)
Open your terminal inside the project directory and run:
```bash
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows
```

### 3. Install dependencies
Install the required packages using pip:
```bash
pip install -r requirements.txt
```

### 4. Run the app
Start the Streamlit dashboard:
```bash
streamlit run app.py
```
The app will automatically open at `http://localhost:8501`.

---

## ⚙️ Configuration Constants

You can customize the simulation constraints at the top of `app.py` (around lines 26-29):

| Constant | Default | Description |
|---|---|---|
| `SHOTS` | `1024` | Number of quantum measurement shots. |
| `MAX_QUBITS` | `20` | Hard cap to prevent RAM exhaustion (fits up to $2^{20} = 1,048,576$ elements). |
| `MIN_ELEMENTS` | `4` | Minimum dataset size for a meaningful search comparison. |
| `PROGRESS_STEP_DELAY` | `0.008` | Animation delay in seconds for the loading state. |

---

## 🧠 Technical Notes

### Qubit Endianness & Key Mapping
- **Oracle Construction**: Qiskit uses **little-endian** qubit ordering (qubit `0` is the Least Significant Bit). Therefore, the binary representation of the target index is reversed (`target_bin_oracle`) before being passed into the oracle so that the loop maps X-gates to the correct qubits.
- **Counts Lookup**: Qiskit's `result.get_counts()` returns output bitstrings formatted as standard **big-endian** (MSB-first). The success probability lookup (`counts.get(target_bin_be)`) and top predicted index decoding (`int(predicted_state, 2)`) are processed directly in big-endian representation without reversal.

### Circuit Compilation (Transpilation)
All circuits are compiled using `transpile(qc, simulator)` before execution. This ensures that high-level multi-controlled gates (`mcx`) are correctly decomposed into simulator-native instructions, preventing layout or synthesis warnings.

### No-Ancilla Mode
To support simulation sizes $> 5$ qubits without error, the multi-controlled NOT gates (`qc.mcx`) in both the `oracle` and `diffusion` functions are run with `mode='noancilla'`. This avoids requiring helper/ancilla qubits, preventing simulation failures for larger qubit registers.

### Simulation Overhead
`AerSimulator` is a classical simulation of quantum mechanics. Its compute and memory requirements scale exponentially with the qubit count. As a result, the wall-clock time will always favor classical search. The true algorithmic advantage is measured by comparing **Classical Steps** versus **Grover Iterations**.

---

## 📚 References

- Grover, L.K. (1996). *A fast quantum mechanical algorithm for database search.* [arXiv:quant-ph/9605043](https://arxiv.org/abs/quant-ph/9605043)
- Nielsen, M.A. & Chuang, I.L. — *Quantum Computation and Quantum Information* (Cambridge, 2010)
- [Qiskit Grover's Algorithm Tutorial](https://learn.qiskit.org/course/ch-algorithms/grovers-algorithm)
- [IBM Quantum Platform](https://quantum.ibm.com)
- [Quantum Algorithm Zoo](https://quantumalgorithmzoo.org)
