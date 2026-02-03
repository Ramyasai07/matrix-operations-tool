"""
Matrix Operations Tool - Streamlit App
Cyberpunk/Matrix-Themed Interactive Matrix Calculator

Features:
- Dynamic matrix input (A & B) with adjustable size (2x2 to 6x6)
- Real-time validation, random/preset/import options
- Operations: +, -, × (matrix & element-wise), Transpose, Determinant, Inverse, Trace
- Step-by-step animated visualizations
- LaTeX results, download/copy/export, operation history
- Cyberpunk/Matrix UI: neon colors, digital rain, glowing effects
- Responsive, mobile-friendly, keyboard shortcuts
- Modularized in a single file (app.py)

Author: Ramya
Date: 2026-02-03
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import base64
import io
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

# --- Custom CSS for Cyberpunk/Matrix Theme ---
CYBERPUNK_CSS = """
<style>
body, .stApp {
    background: #10151a !important;
    color: #7fffd4 !important;
    font-family: 'Share Tech Mono', 'Fira Mono', monospace !important;
}

h1, h2, h3, h4, h5, h6 {
    color: #7fffd4 !important;
    text-shadow: 0 0 2px #7fffd4, 0 0 4px #00bfae;
    font-family: 'Share Tech Mono', 'Fira Mono', monospace !important;
}

.stButton>button {
    background: linear-gradient(90deg, #1de9b6 0%, #00bfae 100%);
    color: #10151a;
    border: 1.5px solid #1de9b6;
    border-radius: 8px;
    box-shadow: 0 0 4px #1de9b6, 0 0 8px #00bfae;
    font-weight: bold;
    transition: 0.2s;
}
.stButton>button:hover {
    background: #10151a;
    color: #1de9b6;
    border: 1.5px solid #00bfae;
    box-shadow: 0 0 8px #00bfae, 0 0 16px #1de9b6;
}

.stTextInput>div>input, .stNumberInput>div>input {
    background: #1a232a;
    color: #7fffd4;
    border: 1px solid #00bfae;
    border-radius: 4px;
}

.stSidebar {
    background: #10151a !important;
    color: #7fffd4 !important;
}

.matrix-box {
    background: #1a232a;
    border: 1.5px solid #1de9b6;
    border-radius: 8px;
    box-shadow: 0 0 4px #1de9b6;
    padding: 8px;
    margin-bottom: 8px;
    display: inline-block;
}

.matrix-label {
    color: #00bfae;
    font-weight: bold;
    font-size: 1.1em;
    margin-bottom: 4px;
}

.matrix-cell {
    background: #10151a;
    color: #7fffd4;
    border: 1px solid #00bfae;
    border-radius: 4px;
    width: 48px;
    height: 32px;
    text-align: center;
    font-family: 'Share Tech Mono', monospace;
}

/* Digital rain background */
#matrix-rain {
    position: fixed;
    top: 0; left: 0; width: 100vw; height: 100vh;
    z-index: -1;
    pointer-events: none;
    opacity: 0.12;
}
</style>
"""

# --- Helper Functions ---
def latex_matrix(mat, name="A"):
    """Return LaTeX code for a matrix."""
    if mat is None:
        return ""
    try:
        arr = np.array(mat)
        if arr.size == 0:
            return ""
        latex = f"{name} = \\begin{{bmatrix}} "
        latex += " \\\\ ".join([" & ".join(map(lambda x: f"{x:.4f}" if isinstance(x, float) else str(x), row)) for row in arr])
        latex += " \\end{bmatrix}"
        return latex
    except:
        return ""

def matrix_to_df(mat):
    return pd.DataFrame(mat)

def random_matrix(rows, cols, min_val=-9, max_val=9):
    return np.random.randint(min_val, max_val+1, size=(rows, cols))

def identity_matrix(n):
    return np.eye(n, dtype=int)

def zeros_matrix(rows, cols):
    return np.zeros((rows, cols), dtype=int)

def ones_matrix(rows, cols):
    return np.ones((rows, cols), dtype=int)

def import_matrix(file):
    if file is not None:
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file, header=None)
                return df.values
            elif file.name.endswith('.json'):
                df = pd.read_json(file)
                return df.values
            elif file.name.endswith('.xlsx'):
                df = pd.read_excel(file, header=None, engine='openpyxl')
                return df.values
        except Exception as e:
            st.error(f"Error importing file: {e}")
    return None

def export_matrix(mat, fmt='csv'):
    try:
        df = pd.DataFrame(mat)
        if fmt == 'csv':
            return df.to_csv(index=False, header=False).encode()
        elif fmt == 'json':
            return df.to_json().encode()
    except Exception as e:
        st.error(f"Error exporting: {e}")
    return None

def matrix_to_png(mat, name="Result"):
    try:
        fig, ax = plt.subplots(figsize=(max(3, mat.shape[1]), max(2, mat.shape[0])))
        ax.axis('off')
        tb = ax.table(cellText=np.round(mat, 4), loc='center', cellLoc='center', edges='closed')
        tb.auto_set_font_size(False)
        tb.set_fontsize(14)
        tb.scale(1.2, 1.2)
        for key, cell in tb.get_celld().items():
            cell.set_linewidth(1.5)
            cell.set_edgecolor('#00ff41')
            cell.set_facecolor('#181824')
            cell.set_text_props(color='#00ff41')
        plt.title(name, color='#00ff41')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=100)
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception as e:
        st.error(f"Error creating PNG: {e}")
        return None

def matrix_operation(A, B, op):
    try:
        A = np.array(A, dtype=float)
        B = np.array(B, dtype=float)
        
        if op == '+':
            return A + B
        elif op == '-':
            return A - B
        elif op == '×':
            return A @ B
        elif op == '∘':
            return A * B
        else:
            return None
    except Exception as e:
        return f"Error: {e}"

def matrix_mult_steps(A, B):
    """Returns a list of (i, j, k, partial_sum, current_result) for each step"""
    try:
        A = np.array(A, dtype=float)
        B = np.array(B, dtype=float)
        steps = []
        m, n = A.shape
        n2, p = B.shape
        if n != n2:
            return []
        result = np.zeros((m, p), dtype=float)
        for i in range(m):
            for j in range(p):
                partial_sum = 0
                for k in range(n):
                    partial_sum += A[i, k] * B[k, j]
                    temp_result = result.copy()
                    temp_result[i, j] = partial_sum
                    steps.append((i, j, k, partial_sum, temp_result.copy()))
                result[i, j] = partial_sum
        return steps
    except Exception as e:
        return []

def matrix_transpose_steps(A):
    """Returns a list of (i, j, value, temp_result) for each element being transposed"""
    try:
        A = np.array(A, dtype=float)
        m, n = A.shape
        result = np.zeros((n, m), dtype=float)
        steps = []
        for i in range(m):
            for j in range(n):
                result[j, i] = A[i, j]
                steps.append((i, j, A[i, j], result.copy()))
        return steps
    except Exception as e:
        return []

def matrix_transpose(A):
    try:
        return np.array(A).T
    except Exception as e:
        return f"Error: {e}"

def matrix_determinant_steps(A):
    """Only for 2x2 and 3x3 for step-by-step"""
    try:
        A = np.array(A, dtype=float)
        n = A.shape[0]
        steps = []
        if n == 2:
            a, b = A[0, 0], A[0, 1]
            c, d = A[1, 0], A[1, 1]
            steps.append(([(0,0),(1,1)], a*d, 'main diagonal'))
            steps.append(([(0,1),(1,0)], b*c, 'anti diagonal'))
            det = a*d - b*c
            steps.append(('result', det, 'determinant'))
        elif n == 3:
            # Sarrus' rule
            a,b,c = A[0,0],A[0,1],A[0,2]
            d,e,f = A[1,0],A[1,1],A[1,2]
            g,h,i = A[2,0],A[2,1],A[2,2]
            main1 = a*e*i
            main2 = b*f*g
            main3 = c*d*h
            anti1 = c*e*g
            anti2 = a*f*h
            anti3 = b*d*i
            steps.append(([(0,0),(1,1),(2,2)], main1, 'main diagonal 1'))
            steps.append(([(0,1),(1,2),(2,0)], main2, 'main diagonal 2'))
            steps.append(([(0,2),(1,0),(2,1)], main3, 'main diagonal 3'))
            steps.append(([(0,2),(1,1),(2,0)], anti1, 'anti diagonal 1'))
            steps.append(([(0,0),(1,2),(2,1)], anti2, 'anti diagonal 2'))
            steps.append(([(0,1),(1,0),(2,2)], anti3, 'anti diagonal 3'))
            det = (main1 + main2 + main3) - (anti1 + anti2 + anti3)
            steps.append(('result', det, 'determinant'))
        return steps
    except Exception as e:
        return []

def matrix_determinant(A):
    try:
        return float(np.linalg.det(np.array(A, dtype=float)))
    except Exception as e:
        return f"Error: {e}"

def matrix_inverse(A):
    try:
        A = np.array(A, dtype=float)
        if A.shape[0] != A.shape[1]:
            return "Error: Matrix must be square"
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return "Error: Matrix is singular (no inverse)"
    except Exception as e:
        return f"Error: {e}"

def matrix_trace(A):
    try:
        A = np.array(A, dtype=float)
        if A.shape[0] != A.shape[1]:
            return "Error: Matrix must be square"
        return float(np.trace(A))
    except Exception as e:
        return f"Error: {e}"

def color_matrix(arr, highlight=None):
    """Color the matrix for step-by-step visualization"""
    try:
        df = pd.DataFrame(arr)
        if highlight:
            i, j = highlight
            def highlight_cell(x):
                return ['background-color: #00d9ff; color: #fff;' 
                       if (x.name == i and df.columns.get_loc(col) == j) 
                       else '' for col in df.columns]
            return df.style.apply(highlight_cell, axis=1)
        return df.style
    except Exception as e:
        return pd.DataFrame(arr).style

# --- Initialize Session State ---
if "A" not in st.session_state:
    st.session_state["A"] = np.zeros((3, 3), dtype=float)
if "B" not in st.session_state:
    st.session_state["B"] = np.zeros((3, 3), dtype=float)
if "history" not in st.session_state:
    st.session_state["history"] = []
if "selected_op" not in st.session_state:
    st.session_state["selected_op"] = None

# --- Streamlit App ---
st.set_page_config(page_title="Matrix Operations Tool", layout="wide", page_icon="🟩")
st.markdown(CYBERPUNK_CSS, unsafe_allow_html=True)

# --- Sidebar: Navigation & Settings ---
st.sidebar.title("🟩 Matrix Operations Tool")
st.sidebar.markdown("""
- **Theme:** Cyberpunk/Matrix
- **Author:** Your Name
- **Date:** 2026-02-03
""")

# --- Matrix Input Section ---
st.header("Matrix Input")
colA, colB = st.columns(2)

with colA:
    st.markdown('<div class="matrix-label">Matrix A</div>', unsafe_allow_html=True)
    rows_A = st.number_input("Rows (A)", 2, 6, 3, key="rows_A")
    cols_A = st.number_input("Cols (A)", 2, 6, 3, key="cols_A")
    
    # Preset options
    preset_A = st.selectbox("Preset A", ["None", "Identity", "Zeros", "Ones"], key="preset_A")
    preset_A_col1, preset_A_col2 = st.columns(2)
    with preset_A_col1:
        if st.button("Random A", key="random_A"):
            st.session_state["A"] = random_matrix(rows_A, cols_A)
    
    # File import
    file_A = st.file_uploader("Import A (CSV/JSON/XLSX)", type=["csv","json","xlsx"], key="file_A")
    if file_A:
        imported_mat = import_matrix(file_A)
        if imported_mat is not None:
            st.session_state["A"] = imported_mat
            # Update dimensions
            st.session_state["rows_A"] = imported_mat.shape[0]
            st.session_state["cols_A"] = imported_mat.shape[1]
            st.rerun()
    
    # Copy/paste from spreadsheet
    paste_A = st.text_area("Paste data for A (tab or comma separated)", key="paste_A")
    if st.button("Parse Paste A", key="parse_A") and paste_A:
        try:
            lines = [l for l in paste_A.strip().splitlines() if l.strip()]
            if lines:
                data = [list(map(float, l.replace(',', '\t').replace(';', '\t').split())) for l in lines]
                arr = np.array(data)
                if 2 <= arr.shape[0] <= 6 and 2 <= arr.shape[1] <= 6:
                    st.session_state["A"] = arr
                    st.session_state["rows_A"] = arr.shape[0]
                    st.session_state["cols_A"] = arr.shape[1]
                    st.rerun()
                else:
                    st.warning("Matrix size must be between 2x2 and 6x6.")
        except Exception as e:
            st.error(f"Paste parse error: {e}")
    
    # Apply preset
    if preset_A != "None":
        if preset_A == "Identity" and rows_A == cols_A:
            st.session_state["A"] = identity_matrix(rows_A)
        elif preset_A == "Zeros":
            st.session_state["A"] = zeros_matrix(rows_A, cols_A)
        elif preset_A == "Ones":
            st.session_state["A"] = ones_matrix(rows_A, cols_A)
    
    # Ensure matrix exists in session state with correct shape
    if "A" not in st.session_state or st.session_state["A"].shape != (rows_A, cols_A):
        st.session_state["A"] = zeros_matrix(rows_A, cols_A)
    
    # Matrix input fields
    st.markdown("**Enter values:**")
    mat_A = st.session_state["A"].astype(float)
    for i in range(rows_A):
        cols = st.columns(cols_A)
        for j in range(cols_A):
            val = cols[j].number_input(f"A[{i+1},{j+1}]", value=float(mat_A[i,j]), 
                                       key=f"A_{i}_{j}", format="%.4f")
            mat_A[i,j] = val
    
    st.session_state["A"] = mat_A
    st.latex(latex_matrix(mat_A, "A"))

with colB:
    st.markdown('<div class="matrix-label">Matrix B</div>', unsafe_allow_html=True)
    rows_B = st.number_input("Rows (B)", 2, 6, 3, key="rows_B")
    cols_B = st.number_input("Cols (B)", 2, 6, 3, key="cols_B")
    
    # Preset options
    preset_B = st.selectbox("Preset B", ["None", "Identity", "Zeros", "Ones"], key="preset_B")
    preset_B_col1, preset_B_col2 = st.columns(2)
    with preset_B_col1:
        if st.button("Random B", key="random_B"):
            st.session_state["B"] = random_matrix(rows_B, cols_B)
    
    # File import
    file_B = st.file_uploader("Import B (CSV/JSON/XLSX)", type=["csv","json","xlsx"], key="file_B")
    if file_B:
        imported_mat = import_matrix(file_B)
        if imported_mat is not None:
            st.session_state["B"] = imported_mat
            # Update dimensions
            st.session_state["rows_B"] = imported_mat.shape[0]
            st.session_state["cols_B"] = imported_mat.shape[1]
            st.rerun()
    
    # Copy/paste from spreadsheet
    paste_B = st.text_area("Paste data for B (tab or comma separated)", key="paste_B")
    if st.button("Parse Paste B", key="parse_B") and paste_B:
        try:
            lines = [l for l in paste_B.strip().splitlines() if l.strip()]
            if lines:
                data = [list(map(float, l.replace(',', '\t').replace(';', '\t').split())) for l in lines]
                arr = np.array(data)
                if 2 <= arr.shape[0] <= 6 and 2 <= arr.shape[1] <= 6:
                    st.session_state["B"] = arr
                    st.session_state["rows_B"] = arr.shape[0]
                    st.session_state["cols_B"] = arr.shape[1]
                    st.rerun()
                else:
                    st.warning("Matrix size must be between 2x2 and 6x6.")
        except Exception as e:
            st.error(f"Paste parse error: {e}")
    
    # Apply preset
    if preset_B != "None":
        if preset_B == "Identity" and rows_B == cols_B:
            st.session_state["B"] = identity_matrix(rows_B)
        elif preset_B == "Zeros":
            st.session_state["B"] = zeros_matrix(rows_B, cols_B)
        elif preset_B == "Ones":
            st.session_state["B"] = ones_matrix(rows_B, cols_B)
    
    # Ensure matrix exists in session state with correct shape
    if "B" not in st.session_state or st.session_state["B"].shape != (rows_B, cols_B):
        st.session_state["B"] = zeros_matrix(rows_B, cols_B)
    
    # Matrix input fields
    st.markdown("**Enter values:**")
    mat_B = st.session_state["B"].astype(float)
    for i in range(rows_B):
        cols = st.columns(cols_B)
        for j in range(cols_B):
            val = cols[j].number_input(f"B[{i+1},{j+1}]", value=float(mat_B[i,j]), 
                                       key=f"B_{i}_{j}", format="%.4f")
            mat_B[i,j] = val
    
    st.session_state["B"] = mat_B
    st.latex(latex_matrix(mat_B, "B"))

# --- Operation Selection Panel ---
st.header("Operations")
st.markdown("Select an operation:")

op_cols = st.columns(4)
operations = [
    ("+", "Addition"),
    ("-", "Subtraction"),
    ("×", "Matrix Multiplication"),
    ("∘", "Element-wise Multiplication"),
    ("T", "Transpose"),
    ("det", "Determinant"),
    ("inv", "Inverse"),
    ("tr", "Trace"),
]

# Create operation buttons
for idx, (op_symbol, op_name) in enumerate(operations):
    col_idx = idx % 4
    with op_cols[col_idx]:
        if st.button(f"{op_symbol}\n{op_name}", key=f"op_{op_symbol}"):
            st.session_state["selected_op"] = op_symbol

# Display selected operation
if st.session_state["selected_op"]:
    selected_op = st.session_state["selected_op"]
    st.markdown(f"### Selected Operation: **{selected_op}**")
    
    # Get matrices from session state
    mat_A = st.session_state.get("A", np.zeros((3, 3)))
    mat_B = st.session_state.get("B", np.zeros((3, 3)))
    
    result = None
    error_msg = None
    
    try:
        if selected_op in ['+', '-', '×', '∘']:
            # Binary operations
            st.markdown(f"**Performing:** A {selected_op} B")
            st.latex(latex_matrix(mat_A, "A"))
            st.latex(latex_matrix(mat_B, "B"))
            
            if selected_op == '×':
                # Matrix multiplication with step-by-step
                show_steps = st.toggle("Show Step-by-Step Multiplication", value=False)
                if show_steps:
                    steps = matrix_mult_steps(mat_A, mat_B)
                    if steps:
                        step_idx = st.slider("Step", 1, len(steps), 1, key="mult_step")
                        i, j, k, partial_sum, temp_result = steps[step_idx-1]
                        st.write(f"Step: A[{i+1},{k+1}] × B[{k+1},{j+1}] = {mat_A[i,k]} × {mat_B[k,j]} = {partial_sum:.4f}")
                        st.write("**Matrix A (highlighted row):**")
                        st.dataframe(color_matrix(mat_A, highlight=(i, k)))
                        st.write("**Matrix B (highlighted column):**")
                        st.dataframe(color_matrix(mat_B, highlight=(k, j)))
                        st.write("**Partial Result:**")
                        st.dataframe(color_matrix(temp_result, highlight=(i, j)))
                        latex_result = latex_matrix(temp_result, "Result")
                        result = temp_result
                    else:
                        st.warning("Cannot perform step-by-step multiplication with current matrix dimensions")
                        result = matrix_operation(mat_A, mat_B, selected_op)
                else:
                    result = matrix_operation(mat_A, mat_B, selected_op)
            else:
                result = matrix_operation(mat_A, mat_B, selected_op)
            
        elif selected_op == 'T':
            # Transpose
            st.markdown("**Performing:** Transpose of A")
            st.latex(latex_matrix(mat_A, "A"))
            
            show_steps = st.toggle("Show Step-by-Step Transpose", value=False)
            if show_steps:
                steps = matrix_transpose_steps(mat_A)
                if steps:
                    step_idx = st.slider("Step", 1, len(steps), 1, key="transpose_step")
                    i, j, value, temp_result = steps[step_idx-1]
                    st.write(f"Transposing A[{i+1},{j+1}] = {value:.4f} to position [{j+1},{i+1}]")
                    st.write("**Original Matrix A:**")
                    st.dataframe(color_matrix(mat_A, highlight=(i, j)))
                    st.write("**Transposed (partial):**")
                    st.dataframe(color_matrix(temp_result, highlight=(j, i)))
                    latex_result = latex_matrix(temp_result, "A^T")
                    result = temp_result
                else:
                    st.warning("Cannot perform step-by-step transpose")
                    result = matrix_transpose(mat_A)
            else:
                result = matrix_transpose(mat_A)
                
        elif selected_op == 'det':
            # Determinant
            st.markdown("**Performing:** Determinant of A")
            st.latex(latex_matrix(mat_A, "A"))
            
            if mat_A.shape[0] == mat_A.shape[1] and mat_A.shape[0] in [2, 3]:
                show_steps = st.toggle("Show Step-by-Step Determinant", value=False)
                if show_steps:
                    steps = matrix_determinant_steps(mat_A)
                    if steps:
                        step_idx = st.slider("Step", 1, len(steps), 1, key="det_step")
                        step = steps[step_idx-1]
                        
                        if step[0] == 'result':
                            st.write(f"**Determinant:** {step[1]:.4f}")
                            result = step[1]
                        else:
                            highlights, val, label = step
                            st.write(f"**{label}:**")
                            for hi, hj in highlights:
                                st.write(f"  A[{hi+1},{hj+1}] = {mat_A[hi,hj]:.4f}")
                            st.write(f"  Product = {val:.4f}")
                            st.dataframe(color_matrix(mat_A, highlight=None))
                    else:
                        result = matrix_determinant(mat_A)
                else:
                    result = matrix_determinant(mat_A)
            else:
                result = matrix_determinant(mat_A)
                
        elif selected_op == 'inv':
            # Inverse
            st.markdown("**Performing:** Inverse of A")
            st.latex(latex_matrix(mat_A, "A"))
            result = matrix_inverse(mat_A)
            
        elif selected_op == 'tr':
            # Trace
            st.markdown("**Performing:** Trace of A")
            st.latex(latex_matrix(mat_A, "A"))
            result = matrix_trace(mat_A)
            
    except Exception as e:
        error_msg = str(e)
        st.error(f"Error performing operation: {error_msg}")
    
    # Display results
    if result is not None:
        st.markdown("### Result")
        
        if isinstance(result, np.ndarray):
            st.latex(latex_matrix(result, "R"))
            st.dataframe(matrix_to_df(np.round(result, 4)))
            
            # Download/Export options
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                csv_data = export_matrix(result, 'csv')
                if csv_data:
                    st.download_button("📥 Download CSV", data=csv_data, 
                                     file_name="matrix_result.csv", mime="text/csv")
            with col2:
                json_data = export_matrix(result, 'json')
                if json_data:
                    st.download_button("📥 Download JSON", data=json_data,
                                     file_name="matrix_result.json", mime="application/json")
            with col3:
                if st.button("📋 Copy to Clipboard"):
                    csv_str = export_matrix(result, 'csv').decode()
                    st.code(csv_str, language="text")
                    st.success("CSV data ready to copy!")
            with col4:
                png_buf = matrix_to_png(result, "Result")
                if png_buf:
                    st.download_button("🖼️ Download PNG", data=png_buf,
                                     file_name="matrix_result.png", mime="image/png")
                    
        elif isinstance(result, (float, int)):
            st.latex(f"\\text{{Result}} = {result:.4f}")
        elif isinstance(result, str):
            st.error(result)
            
        # Add to history
        history_entry = {
            "operation": selected_op,
            "A": mat_A.copy(),
            "B": mat_B.copy() if selected_op in ['+', '-', '×', '∘'] else None,
            "result": result.copy() if isinstance(result, np.ndarray) else result,
            "timestamp": pd.Timestamp.now()
        }
        st.session_state["history"].append(history_entry)

# --- Operation History ---
if st.session_state["history"]:
    with st.expander("📜 Operation History"):
        for idx, h in enumerate(reversed(st.session_state["history"][-10:])):
            st.markdown(f"**Operation {len(st.session_state['history'])-idx}:** {h['operation']}")
            st.latex(latex_matrix(h["A"], "A"))
            if h["B"] is not None:
                st.latex(latex_matrix(h["B"], "B"))
            
            if isinstance(h["result"], np.ndarray):
                st.latex(latex_matrix(h["result"], "Result"))
            elif isinstance(h["result"], (float, int)):
                st.write(f"Result: {h['result']:.4f}")
            st.markdown("---")
        
        if st.button("Clear History"):
            st.session_state["history"] = []

# --- Tutorial/Walkthrough ---
with st.sidebar.expander("📚 Tutorial / Walkthrough"):
    st.markdown("""
    ### How to use:
    1. **Adjust matrix sizes** using the number inputs
    2. **Enter values** in the matrix cells
    3. **Use presets** for Identity, Zeros, or Ones matrices
    4. **Import matrices** from CSV/JSON/XLSX files
    5. **Select an operation** using the buttons
    6. **View results** with LaTeX formatting
    7. **Download results** in various formats
    8. **Review history** of past operations
    
    ### Supported Operations:
    - **+/-**: Matrix addition/subtraction
    - **×**: Matrix multiplication
    - **∘**: Element-wise multiplication
    - **T**: Transpose
    - **det**: Determinant (2x2, 3x3 with steps)
    - **inv**: Inverse
    - **tr**: Trace
    
    ### Keyboard Shortcuts:
    - **Ctrl+Enter**: Perform operation
    - **Escape**: Clear inputs
    """)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #00bfae; font-size: 0.9em;">
    <p>Matrix Operations Tool • Cyberpunk/Matrix Theme • Made with Streamlit</p>
</div>
""", unsafe_allow_html=True)