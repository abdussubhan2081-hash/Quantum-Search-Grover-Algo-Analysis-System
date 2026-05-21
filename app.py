"""
Quantum Search Analysis System
================================
Compares Classical Linear Search (O(N)) with Grover's Quantum Search (O(√N)).
Built with Streamlit + Qiskit (Aer simulator).

Run with:  streamlit run app.py
"""

import io
import re
import time

import docx
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import PyPDF2
import streamlit as st
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SHOTS: int = 1024          # Number of quantum measurement shots
MAX_QUBITS: int = 20       # Hard cap to prevent RAM exhaustion (2^20 = 4096 elements)
MIN_ELEMENTS: int = 4      # Minimum dataset size for a meaningful comparison
PROGRESS_STEP_DELAY: float = 0.008   # Seconds per progress bar tick


# ─────────────────────────────────────────────────────────────────────────────
# CORE QUANTUM LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def linear_search(arr: list[str], target: str) -> tuple[int, int]:
    """
    Perform a classical linear (sequential) search.

    Args:
        arr:    The list to search through (original, unpadded).
        target: The element to find.

    Returns:
        A tuple of (found_index, steps_taken).
        Returns (-1, len(arr)) if the target is not found.
    """
    for i, val in enumerate(arr):
        if val == target:
            return i, i + 1
    return -1, len(arr)


def oracle(n: int, target_bin: str) -> QuantumCircuit:
    """
    Build the Grover phase oracle that marks the target state.

    The oracle flips the phase (sign) of the target basis state |t⟩
    without disturbing any other state, using the following approach:
      1. Apply X gates to qubits where target_bin[i] == '0' → maps |t⟩ to |11…1⟩.
      2. Apply multi-controlled-Z (via H · MCX · H sandwich) to flip |11…1⟩.
      3. Uncompute (re-apply X gates) to restore all non-target states.

    Args:
        n:          Number of qubits.
        target_bin: Little-endian binary string of the target index
                    (qubit 0 = LSB, matching Qiskit's convention).

    Returns:
        A QuantumCircuit implementing the oracle gate.
    """
    qc = QuantumCircuit(n, name="Oracle")
    # Step 1: Map target to |11…1⟩
    for i in range(n):
        if target_bin[i] == '0':
            qc.x(i)
    # Step 2: Controlled-Z on all qubits (H·MCX·H = MCZ)
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1, mode='noancilla')
    qc.h(n - 1)
    # Step 3: Uncompute
    for i in range(n):
        if target_bin[i] == '0':
            qc.x(i)
    return qc


def diffusion(n: int) -> QuantumCircuit:
    """
    Build the Grover diffusion operator (inversion about the mean).

    Reflects all amplitude values about their average, which constructively
    amplifies the marked state and suppresses all others. Applied after each
    oracle call.

    Args:
        n: Number of qubits.

    Returns:
        A QuantumCircuit implementing the diffusion operator.
    """
    qc = QuantumCircuit(n, name="Diffusion")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1, mode='noancilla')
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


def grover_search(n: int, target_bin: str) -> tuple[QuantumCircuit, int]:
    """
    Assemble the full Grover's algorithm circuit.

    Structure:
      |0⟩^n  →  H^⊗n  →  [Oracle · Diffusion] × iterations  →  Measure

    The optimal number of iterations is floor(π/4 × √(2^n)).

    Args:
        n:          Number of qubits.
        target_bin: Little-endian binary string of the target index.

    Returns:
        A tuple of (QuantumCircuit ready to simulate, number_of_iterations).
    """
    iterations = max(1, int(np.floor(np.pi / 4 * np.sqrt(2 ** n))))

    # Build oracle and diffusion once, reuse across iterations
    oracle_gate = oracle(n, target_bin)
    diffusion_gate = diffusion(n)

    qc = QuantumCircuit(n, n)
    qc.h(range(n))           # Initialize uniform superposition

    for _ in range(iterations):
        qc.compose(oracle_gate, inplace=True)
        qc.compose(diffusion_gate, inplace=True)

    qc.measure(range(n), range(n))
    return qc, iterations


def run_grover(n: int, target_bin: str) -> tuple[dict, int, QuantumCircuit]:
    """
    Compile and simulate the Grover circuit on AerSimulator.

    Args:
        n:          Number of qubits.
        target_bin: Little-endian binary string of the target index.

    Returns:
        A tuple of (measurement_counts_dict, iterations, circuit).

    Raises:
        MemoryError:  If statevector simulation exhausts available RAM.
        RuntimeError: If Qiskit simulation fails for any other reason.
    """
    qc, iterations = grover_search(n, target_bin)
    simulator = AerSimulator()
    # Transpile converts all gates to Aer-native basis gates (fixes version compatibility)
    qc_compiled = transpile(qc, simulator)
    result = simulator.run(qc_compiled, shots=SHOTS).result()
    counts = result.get_counts()
    return counts, iterations, qc


# ─────────────────────────────────────────────────────────────────────────────
# DATA EXTRACTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(file) -> str:
    """
    Extract plain text from a PDF file.

    Handles image-based pages gracefully — they are silently skipped.

    Args:
        file: A file-like object (BytesIO or uploaded file).

    Returns:
        Concatenated text from all readable pages.

    Raises:
        ValueError: If the PDF cannot be read or parsed.
    """
    try:
        reader = PyPDF2.PdfReader(file)
        pages_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:  # page.extract_text() returns None for image-only pages
                pages_text.append(text)
        return " ".join(pages_text)
    except Exception as e:
        raise ValueError(f"Could not read PDF file: {e}") from e


def extract_text_from_docx(file) -> str:
    """
    Extract plain text from a .docx Word document.

    Args:
        file: A file-like object (BytesIO or uploaded file).

    Returns:
        Concatenated paragraph text from the document.

    Raises:
        ValueError: If the DOCX file cannot be read or parsed.
    """
    try:
        doc = docx.Document(file)
        return " ".join([p.text for p in doc.paragraphs if p.text.strip()])
    except Exception as e:
        raise ValueError(f"Could not read DOCX file: {e}") from e


def parse_data(raw_text: str) -> list[str]:
    """
    Parse a raw string into a list of elements.

    Splits on any combination of commas, spaces, tabs, or newlines,
    and strips leading/trailing whitespace from each token.

    Args:
        raw_text: Raw input string from manual entry or file upload.

    Returns:
        A list of non-empty string tokens.
    """
    return [x.strip() for x in re.split(r'[\s,]+', raw_text) if x.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quantum Search Analysis System",
    layout="wide",
    page_icon="⚛️",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  (Matrix / cyberpunk theme)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Background ── */
[data-testid="stAppViewContainer"] { background-color: #000000; }
[data-testid="stHeader"]           { background-color: rgba(0,0,0,0); }
[data-testid="stSidebar"]          { background-color: #050505; }

/* ── Headings: Matrix green + glow ── */
h1, h2, h3, h4, h5, h6 {
    color: #00ff00 !important;
    font-family: 'Courier New', Courier, monospace !important;
    text-shadow: 0px 0px 6px rgba(0, 255, 0, 0.5) !important;
    border: 1px solid #00ff00 !important;
    padding: 10px 14px !important;
    border-radius: 4px !important;
    background-color: #050505 !important;
}
h1 { text-align: center !important; }

/* ── Body text ── */
p, span, label, li, .stMarkdown {
    color: #e0e0e0 !important;
    font-family: 'Georgia', 'Times New Roman', serif !important;
}

/* ── Metric cards ── */
[data-testid="stMetricValue"] { color: #00ff00 !important; font-family: 'Courier New', monospace !important; }
[data-testid="stMetricLabel"] { color: #aaaaaa !important; }
[data-testid="metric-container"] {
    background-color: #050505 !important;
    border: 1px solid #00ff00 !important;
    border-radius: 6px !important;
    padding: 14px !important;
    box-shadow: 0 0 10px rgba(0, 255, 0, 0.15) !important;
}

/* ── Primary button (orange glow) ── */
button[kind="primary"] {
    background-color: #000000 !important;
    border: 1px solid #ff6e00 !important;
    color: #00ff00 !important;
    text-transform: uppercase !important;
    font-family: 'Courier New', monospace !important;
    font-weight: bold !important;
    box-shadow: 0 0 10px #ff6e00 !important;
    transition: all 0.25s ease !important;
}
button[kind="primary"]:hover {
    background-color: #00ff00 !important;
    color: #000000 !important;
    box-shadow: 0 0 20px #00ff00 !important;
}

/* ── Secondary button ── */
button[kind="secondary"] {
    background-color: #050505 !important;
    border: 1px solid #555555 !important;
    color: #aaaaaa !important;
    transition: all 0.2s ease !important;
}
button[kind="secondary"]:hover {
    border-color: #00ff00 !important;
    color: #00ff00 !important;
}

/* ── Text inputs / selects ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div > div {
    background-color: #080808 !important;
    color: #03a9f4 !important;
    border: 1px solid #00ff00 !important;
    font-family: 'Courier New', monospace !important;
}

/* ── Radio buttons ── */
.stRadio > label { color: #e0e0e0 !important; }

/* ── Expanders ── */
[data-testid="stExpander"] {
    border: 1px solid #00ff00 !important;
    background-color: rgba(0, 255, 0, 0.03) !important;
    border-radius: 4px !important;
}
[data-testid="stExpanderToggle"] { color: #00ff00 !important; }

/* ── Alert / info boxes ── */
[data-testid="stAlert"] {
    background-color: #050505 !important;
    border-left: 4px solid #00ff00 !important;
}

/* ── Images: green border + glow ── */
img {
    border: 1px solid #00ff00;
    border-radius: 4px;
    box-shadow: 0 0 12px rgba(0, 255, 0, 0.25);
}

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div > div { background-color: #00ff00 !important; }

/* ── Dividers ── */
hr { border-color: #1a1a1a !important; }

/* ── Caption text ── */
.stCaption, small { color: #888888 !important; font-style: italic; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER & INTRODUCTION
# ─────────────────────────────────────────────────────────────────────────────
st.title("⚛️ Quantum Search Analysis System")

st.markdown("""
### 📖 Fundamentals

* **Quantum Computing:** A computational paradigm exploiting quantum mechanics to solve
  problems that are intractable on classical hardware.
* **Qubits:** The quantum analogue of a classical bit. Unlike a strict `0` or `1`, a qubit
  can exist in a *superposition* — a weighted combination of both states simultaneously.
* **Superposition:** A system of $n$ qubits can represent $2^n$ basis states at once,
  enabling massively parallel amplitude manipulation.

---

### 🔍 Search Algorithm Comparison

| Property | Classical Linear Search | Grover's Quantum Search |
|---|---|---|
| **Time Complexity** | $O(N)$ | $O(\\sqrt{N})$ |
| **Strategy** | Checks elements one-by-one | Amplitude amplification via constructive interference |
| **Iterations at N = 256** | ≤ 256 steps | ≈ 12 iterations |
| **Iterations at N = 1024** | ≤ 1024 steps | ≈ 25 iterations |

> **⚠️ Simulation Note:** Grover's advantage comes from *constructive interference* —
> not from checking elements in parallel (a common misconception). Because we are
> simulating quantum states on classical hardware, **raw wall-clock time always favours
> classical Python**. Focus on the **Steps / Iterations** column for the true algorithmic advantage.
""")

# ── Oracle explanation expander ──
with st.expander("📘 How Does Grover's Algorithm Work? (Click to expand)"):
    st.markdown("""
    #### The Oracle (Phase Kickback)
    The Oracle is a quantum circuit that **marks** the target state by flipping its amplitude sign
    (phase), without revealing which state it marked to any external observer.

    **Step-by-step:**
    1. Apply **X (NOT) gates** to qubits where the target binary string has a `0`.
       This temporarily maps the target state `|t⟩` to the all-ones state `|11…1⟩`.
    2. Apply a **multi-controlled-Z gate** (implemented as H·MCX·H) — this flips the
       phase of `|11…1⟩` only, applying a factor of −1 to that amplitude.
    3. **Uncompute** by re-applying the same X gates, restoring all other states.

    #### The Diffusion Operator (Inversion About the Mean)
    After each oracle call, the diffusion operator reflects all amplitude values about
    their average. Because the marked state has a *negative* amplitude, it ends up
    *above* the mean after reflection — boosted relative to everything else.

    #### Iterations
    Repeating Oracle + Diffusion for $\\lfloor \\frac{\\pi}{4}\\sqrt{N} \\rfloor$ rounds
    maximises the probability of measuring the target state. Measuring the circuit then
    collapses the superposition to the correct answer with high probability.

    #### Oracle Caveat (Real Hardware)
    In practice, constructing an oracle for an *unstructured* database on real quantum
    hardware requires knowing the answer in advance (to encode it into the circuit).
    Grover's true advantage shines in structured problems where the oracle can be
    efficiently implemented (e.g., SAT solving, cryptography).
    """)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 1. INPUT SECTION
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📥 1. Load Dataset")
col_input1, col_input2 = st.columns([1, 1])

with col_input1:
    input_method = st.radio(
        "Choose input method:",
        ("Manual Entry", "File Upload (.txt, .pdf, .docx)"),
    )

    raw_data = ""
    if input_method == "Manual Entry":
        raw_data = st.text_area(
            "Enter elements (comma or space separated):",
            "apple, banana, cherry, date, elderberry, fig, grape, honeydew",
        )
    else:
        uploaded_file = st.file_uploader("Upload Document", type=["txt", "pdf", "docx"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".pdf"):
                    raw_data = extract_text_from_pdf(uploaded_file)
                elif uploaded_file.name.endswith(".docx"):
                    raw_data = extract_text_from_docx(uploaded_file)
                else:
                    raw_data = uploaded_file.getvalue().decode("utf-8")
                st.success("✅ File loaded successfully.")
            except ValueError as e:
                st.error(f"❌ File Error: {e}")
                st.stop()
            except Exception as e:
                st.error(f"❌ Unexpected error reading file: {e}")
                st.stop()

arr = parse_data(raw_data)

# ── Input validation ──
if not arr:
    st.warning("⚠️ Dataset is empty. Please provide some data.")
    st.stop()

if len(arr) < MIN_ELEMENTS:
    st.warning(
        f"⚠️ Please provide at least **{MIN_ELEMENTS} elements** for a meaningful comparison. "
        f"You currently have {len(arr)}."
    )
    st.stop()

N = len(arr)
n_qubits = max(2, int(np.ceil(np.log2(N))))

if n_qubits > MAX_QUBITS:
    st.error(
        f"❌ Your dataset of **{N} elements** requires **{n_qubits} qubits**, which exceeds the "
        f"safe simulation limit of **{MAX_QUBITS} qubits** ({2**MAX_QUBITS} elements). "
        f"Please reduce your dataset size to ≤ {2**MAX_QUBITS} elements."
    )
    st.stop()

if N > 256:
    st.warning(
        f"⚠️ Your dataset has **{N} items** ({n_qubits} qubits). Simulation will be slower. "
        f"Consider reducing to ≤ 256 elements for the best interactive experience."
    )

padded_size = 2 ** n_qubits
arr_padded = arr + ["<PAD>"] * (padded_size - N)

with col_input2:
    st.write(f"**Valid Elements Detected:** `{N}`")
    st.write(f"**Padded Size for Quantum Register:** `{padded_size}` (fits {n_qubits} qubits)")
    st.write(f"**Max Safe Qubits:** `{MAX_QUBITS}` ({2**MAX_QUBITS} elements)")

    target = st.selectbox("Select Target to Search:", options=arr)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🚀 Execute Search Comparison", use_container_width=True, type="primary"):
            st.session_state.run_analysis = True
    with col_btn2:
        if st.button("🔄 Reset Analysis", use_container_width=True):
            st.session_state.run_analysis = False
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# 2. ANALYSIS & RESULTS
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.run_analysis:
    st.divider()
    st.header("📊 Analysis & Results")

    try:
        target_index = arr.index(target)
    except ValueError:
        st.error(f"❌ Target element '{target}' was not found in the dataset.")
        st.stop()

    # ── Target bitstrings for circuit representation and counts lookup ──
    # target_bin_be: big-endian string (how Qiskit reports counts, e.g., '100' for index 4)
    target_bin_be = format(target_index, f"0{n_qubits}b")
    # target_bin_oracle: reversed to match Qiskit's qubit ordering in oracle (qubit 0 = target_bin_oracle[0])
    target_bin_oracle = target_bin_be[::-1]

    # ── Classical Linear Search ──
    with st.spinner("🔍 Running Classical Linear Search..."):
        c_start = time.perf_counter()
        c_idx, c_steps = linear_search(arr, target)   # search original arr, not padded
        c_time = time.perf_counter() - c_start

    # ── Grover's Quantum Search ──
    with st.spinner("🔬 Initializing Quantum Register & Simulating Circuit..."):
        prog = st.progress(0, text="Building oracle & diffusion gates…")
        for i in range(35):
            time.sleep(PROGRESS_STEP_DELAY)
            prog.progress(i + 1, text="Building oracle & diffusion gates…")

        try:
            q_start = time.perf_counter()
            counts, q_steps, qc = run_grover(n_qubits, target_bin_oracle)
            q_time = time.perf_counter() - q_start
        except MemoryError:
            prog.empty()
            st.error(
                "❌ **MemoryError:** The quantum simulation exhausted available RAM. "
                "Please reduce your dataset size and try again."
            )
            st.stop()
        except Exception as e:
            prog.empty()
            st.error(f"❌ **Quantum simulation failed:** {e}")
            st.stop()

        for i in range(35, 100):
            time.sleep(PROGRESS_STEP_DELAY)
            prog.progress(i + 1, text="Measuring quantum state…")
        prog.empty()

    # ── Verify quantum result ──
    total_shots = sum(counts.values())

    # Most probable measured state (MSB-first)
    predicted_state = max(counts, key=counts.get)
    # Convert back to index integer
    predicted_index = int(predicted_state, 2)

    # Success probability = fraction of shots that gave the correct answer
    success_shots = counts.get(target_bin_be, 0)
    success_probability = (success_shots / total_shots) * 100
    grover_correct = (predicted_index == target_index)

    # ── Top-level Metrics ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset Size (N)", N)
    col2.metric("Quantum Register Size", padded_size)
    col3.metric("Qubits Required", n_qubits)
    col4.metric("Target Index", target_index)

    st.divider()

    # ── Side-by-side Algorithm Results ──
    col_c, col_q = st.columns(2)

    with col_c:
        st.subheader("🔍 Classical Linear Search")
        st.metric("Steps Taken", c_steps)
        st.metric("Execution Time", f"{c_time:.6f} sec")
        if c_idx != -1:
            st.success(f"✅ Found **'{target}'** at index **{c_idx}**")
        else:
            st.error("❌ Target not found")
        st.caption(
            f"Scanned elements sequentially. "
            f"Found target after checking {c_steps} element(s). "
            f"Worst case = {N} steps."
        )

    with col_q:
        st.subheader("⚛️ Grover's Quantum Search")
        st.metric("Grover Iterations", q_steps)
        st.metric("Simulator Time (classical overhead)", f"{q_time:.4f} sec")
        st.metric("Success Probability", f"{success_probability:.1f}%")

        if grover_correct:
            st.success(
                f"✅ Grover's predicted index **{predicted_index}** → **Correct!** "
                f"({success_probability:.1f}% of {SHOTS} shots collapsed to the right state)"
            )
        else:
            st.warning(
                f"⚠️ Grover's top prediction was index **{predicted_index}**, "
                f"but the true target is at index **{target_index}**. "
                f"Correct state probability: {success_probability:.1f}%. "
                f"This can occur at very small N due to statistical noise — try more shots."
            )
        st.caption(
            f"Required only **{q_steps}** oracle + diffusion iteration(s) "
            f"(≈ π/4 × √{padded_size} ≈ {q_steps}). "
            "Simulator wall-clock time reflects classical overhead, NOT real quantum speed."
        )

    st.divider()

    # ── Algorithmic Speedup Summary ──
    st.subheader("⚡ Algorithmic Advantage Summary")
    speedup = c_steps / q_steps if q_steps > 0 else 1
    theoretical_c = padded_size
    theoretical_q = int(np.floor(np.pi / 4 * np.sqrt(padded_size)))

    st.info(
        f"**For your dataset (N = {N}, padded to {padded_size}):**\n\n"
        f"- Classical worst case: **{theoretical_c} steps**\n"
        f"- Grover's optimal iterations: **{theoretical_q} iterations**\n"
        f"- Theoretical speedup: **{theoretical_c / theoretical_q:.1f}×** fewer operations\n\n"
        f"In your specific run: Classical took **{c_steps}** step(s), "
        f"Grover's used **{q_steps}** iteration(s) — a **{speedup:.1f}×** reduction."
    )

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # VISUALIZATIONS
    # ─────────────────────────────────────────────────────────────────────────
    plt.style.use("dark_background")
    col_hist, col_plot = st.columns(2)

    # ── Chart 1: Quantum State Measurement Distribution ──
    with col_hist:
        st.markdown("### 📉 Quantum Measurement Distribution")
        st.caption(
            f"Each bar shows how many times (out of {SHOTS} shots) the circuit collapsed "
            f"to that quantum state. The **green bar** at `{target_bin_be}` (binary "
            f"for index {target_index}) shows that Grover's amplified the correct "
            f"answer — it dominates with {success_probability:.1f}% of all measurements."
        )

        fig1, ax1 = plt.subplots(figsize=(6, 4))
        fig1.patch.set_facecolor("#000000")
        ax1.set_facecolor("#000000")

        sorted_counts = dict(sorted(counts.items()))
        bar_colors = [
            "#00ff00" if state == target_bin_be else "#fba300"
            for state in sorted_counts.keys()
        ]

        ax1.bar(
            list(sorted_counts.keys()),
            list(sorted_counts.values()),
            color=bar_colors,
            edgecolor="#1a1a1a",
            linewidth=0.5,
        )
        ax1.set_xlabel("Quantum States (Binary)", color="#cccccc", fontsize=9)
        ax1.set_ylabel("Shot Count", color="#cccccc", fontsize=9)
        ax1.set_title(f"Grover Measurement Outcomes ({SHOTS} shots)", color="#00ff00", fontsize=10)
        ax1.tick_params(colors="#aaaaaa", labelsize=8)
        plt.xticks(rotation=45, ha="right")

        legend_handles = [
            mpatches.Patch(facecolor="#00ff00", label=f"Target state → index {target_index}"),
            mpatches.Patch(facecolor="#fba300", label="Other states"),
        ]
        ax1.legend(handles=legend_handles, facecolor="#0a0a0a", edgecolor="#00ff00",
                   labelcolor="#ffffff", fontsize=8)
        ax1.grid(axis="y", linestyle="--", alpha=0.2, color="#333333")

        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    # ── Chart 2: Complexity Scaling Comparison ──
    with col_plot:
        st.markdown("### 📈 Search Complexity Comparison")
        st.caption(
            "Classical search scales as O(N) — steps grow linearly with dataset size. "
            "Grover's scales as O(√N) — the gap widens dramatically as N increases. "
            "The green dashed line marks your current dataset size."
        )

        sizes = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
        c_scale = sizes
        q_scale = np.floor(np.pi / 4 * np.sqrt(sizes))

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        fig2.patch.set_facecolor("#000000")
        ax2.set_facecolor("#000000")

        ax2.plot(sizes, c_scale, marker="o", markersize=4,
                 label="Classical O(N)", color="#168cf3", linewidth=2)
        ax2.plot(sizes, q_scale, marker="o", markersize=4,
                 label="Grover O(√N)", color="#ff4000", linewidth=2)
        ax2.axvline(x=N, color="#00ff00", linestyle="--", alpha=0.75,
                    label=f"Your dataset (N={N})", linewidth=1.5)

        ax2.set_xlabel("Dataset Size (N)", color="#cccccc", fontsize=9)
        ax2.set_ylabel("Steps / Iterations", color="#cccccc", fontsize=9)
        ax2.set_title("O(N) vs O(√N) Complexity Scaling", color="#00ff00", fontsize=10)
        ax2.tick_params(colors="#aaaaaa", labelsize=8)
        ax2.legend(facecolor="#0a0a0a", edgecolor="#00ff00", labelcolor="#ffffff", fontsize=8)
        ax2.grid(True, linestyle="--", alpha=0.2, color="#333333")
        ax2.set_xscale("log", base=2)

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    st.divider()

    # ── Quantum Circuit Diagram ──
    st.subheader("🔬 Compiled Quantum Circuit")
    st.caption(
        f"The actual Grover circuit executed by the Aer simulator: "
        f"**{n_qubits} qubits**, **{q_steps}** Oracle + Diffusion iteration(s), measured at the end."
    )

    try:
        fig_circ = qc.draw(
            output="mpl",
            style={"backgroundcolor": "#000000", "textcolor": "#00ff00",
                   "gatetextcolor": "#00ff00", "barrierfill": "#222222"},
            fold=60,
        )
        fig_circ.patch.set_facecolor("#000000")
        st.pyplot(fig_circ)
        plt.close(fig_circ)
    except Exception:
        # Graceful fallback to text if mpl circuit draw fails
        st.code(str(qc.draw(output="text")), language="")

    # ── Further Reading ──
    st.divider()
    with st.expander("📚 Further Reading & References"):
        st.markdown("""
        | Resource | Link |
        |---|---|
        | **Original Grover Paper (1996)** | [arXiv:quant-ph/9605043](https://arxiv.org/abs/quant-ph/9605043) |
        | **Textbook** — Nielsen & Chuang | *Quantum Computation and Quantum Information*, Cambridge University Press, 2010 |
        | **Qiskit Grover Tutorial** | [learn.qiskit.org — Grover's Algorithm](https://learn.qiskit.org/course/ch-algorithms/grovers-algorithm) |
        | **IBM Quantum** | [quantum.ibm.com](https://quantum.ibm.com) |
        | **Quantum Algorithm Zoo** | [quantumalgorithmzoo.org](https://quantumalgorithmzoo.org) |
        """)
