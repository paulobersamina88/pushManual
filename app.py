
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Manual RSA–Pushover Reconciliation", layout="wide")

st.title("Manual RSA–Pushover Reconciliation Tool")
st.caption("Offline teaching app: simplified MDOF shear-building pushover + RSA comparison")

# -----------------------------
# Helper functions
# -----------------------------
def assemble_K(k_storey):
    """
    Assemble shear-building stiffness matrix.
    k_storey[0] = storey 1 stiffness, k_storey[1] = storey 2 stiffness, etc.
    """
    n = len(k_storey)
    K = np.zeros((n, n), dtype=float)
    for i in range(n):
        if i == 0:
            K[i, i] += k_storey[i]
        else:
            K[i, i] += k_storey[i]
            K[i-1, i-1] += k_storey[i]
            K[i, i-1] -= k_storey[i]
            K[i-1, i] -= k_storey[i]
    return K

def modal_properties(weights_kN, k_storey_kN_m):
    """
    Solve generalized eigenproblem K phi = omega^2 M phi.
    Uses mass = W/g in kN-s^2/m.
    """
    g = 9.80665
    W = np.array(weights_kN, dtype=float)
    Mdiag = W / g
    M = np.diag(Mdiag)
    K = assemble_K(np.array(k_storey_kN_m, dtype=float))
    A = np.linalg.inv(M) @ K
    eigvals, eigvecs = np.linalg.eig(A)
    idx = np.argsort(np.real(eigvals))
    eigvals = np.real(eigvals[idx])
    eigvecs = np.real(eigvecs[:, idx])
    omegas = np.sqrt(np.maximum(eigvals, 0))
    periods = 2*np.pi/omegas

    # normalize each mode to roof = 1
    phis = eigvecs.copy()
    for j in range(phis.shape[1]):
        if abs(phis[-1, j]) > 1e-12:
            phis[:, j] = phis[:, j] / phis[-1, j]

    rows = []
    total_mass = np.sum(Mdiag)
    one = np.ones(len(W))
    for j in range(len(W)):
        phi = phis[:, j]
        num = phi @ M @ one
        den = phi @ M @ phi
        gamma = num / den
        m_eff = (num**2) / den
        ratio = m_eff / total_mass * 100
        rows.append({
            "Mode": j+1,
            "Omega_rad_per_s": omegas[j],
            "Period_s": periods[j],
            "Gamma": gamma,
            "Effective_Mass_%": ratio,
            "Mode_shape_roof_1": phi[-1]
        })
    df = pd.DataFrame(rows)
    df["Cumulative_%"] = df["Effective_Mass_%"].cumsum()
    return df, phis, K, M

def bilinear_drift(V, Vy, k, alpha):
    """
    Return storey drift in m from storey shear V.
    Elastic-perfectly/post-yield bilinear model.
    """
    V = float(V)
    Vy = float(Vy)
    k = float(k)
    if k <= 0:
        return np.nan
    dy = Vy / k
    if V <= Vy:
        return V / k
    kp = max(alpha * k, 1e-9)
    return dy + (V - Vy) / kp

def state_label(storey_shears, Vy):
    yielded = [i+1 for i, (v, y) in enumerate(zip(storey_shears, Vy)) if v >= y - 1e-9]
    if not yielded:
        return "Elastic"
    if len(yielded) == 1:
        return f"Storey {yielded[0]} yielded"
    return "Storey " + " + ".join(map(str, yielded)) + " yielded"

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Main Inputs")

n = st.sidebar.number_input("Number of storeys", min_value=1, max_value=10, value=2, step=1)

default_weights = [422.53, 382.30] + [350.0]*(n-2)
default_k = [56960.0, 16720.0] + [15000.0]*(n-2)

input_rows = []
for i in range(n):
    input_rows.append({
        "Storey": i+1,
        "Floor weight W_i (kN)": default_weights[i] if i < len(default_weights) else 350.0,
        "Storey stiffness k_i (kN/m)": default_k[i] if i < len(default_k) else 15000.0,
        "Height h_i (m)": 3.0,
        "Columns per frame": 3,
        "Bays per frame": 2,
        "Column Mp each (kN-m)": 300.0,
        "Beam Mp per end (kN-m)": 150.0,
    })

st.subheader("1. Floor / Storey Input Table")
df_in = st.data_editor(
    pd.DataFrame(input_rows),
    use_container_width=True,
    num_rows="fixed"
)

colA, colB, colC = st.columns(3)
with colA:
    number_of_frames = st.number_input(
        "Number of similar frames in this axis",
        min_value=1,
        max_value=20,
        value=3,
        help="Use 1 if stiffness/mass and plastic capacities are already for the whole axis."
    )
with colB:
    input_basis = st.selectbox(
        "Plastic moment input basis",
        ["Per frame - multiply by number of frames", "Whole axis - do not multiply"],
        index=0
    )
with colC:
    alpha = st.number_input("Post-yield stiffness ratio α", min_value=0.001, max_value=0.50, value=0.05, step=0.01)

st.info(
    "Important: Mass basis = stiffness basis = plastic moment basis. "
    "If STAAD stiffness/mass are for 3 frames in one axis, your yield capacity should also represent those 3 frames."
)

# -----------------------------
# Extract input arrays
# -----------------------------
weights = df_in["Floor weight W_i (kN)"].to_numpy(dtype=float)
k_storey = df_in["Storey stiffness k_i (kN/m)"].to_numpy(dtype=float)
h = df_in["Height h_i (m)"].to_numpy(dtype=float)
cols = df_in["Columns per frame"].to_numpy(dtype=float)
bays = df_in["Bays per frame"].to_numpy(dtype=float)
mp_col = df_in["Column Mp each (kN-m)"].to_numpy(dtype=float)
mp_beam = df_in["Beam Mp per end (kN-m)"].to_numpy(dtype=float)

frame_multiplier = number_of_frames if input_basis.startswith("Per frame") else 1

# -----------------------------
# Modal properties
# -----------------------------
st.subheader("2. Modal Properties from Extracted STAAD Mass and Stiffness")

try:
    modal_df, phis, K_elastic, M = modal_properties(weights, k_storey)
    c1, c2, c3 = st.columns(3)
    c1.metric("Fundamental period T1 (s)", f"{modal_df.loc[0,'Period_s']:.4f}")
    c2.metric("1st mode mass participation", f"{modal_df.loc[0,'Effective_Mass_%']:.2f}%")
    c3.metric("Cumulative mass participation", f"{modal_df['Cumulative_%'].iloc[-1]:.2f}%")
    st.dataframe(modal_df, use_container_width=True)

    phi1 = phis[:, 0]
    # Make all signs positive for practical pushover pattern
    if np.sum(phi1) < 0:
        phi1 = -phi1
except Exception as e:
    st.error(f"Modal calculation failed: {e}")
    st.stop()

# -----------------------------
# Force pattern
# -----------------------------
pattern_raw = weights * phi1
if np.any(pattern_raw < 0):
    pattern_raw = np.abs(pattern_raw)
force_ratio = pattern_raw / np.sum(pattern_raw)

pattern_df = pd.DataFrame({
    "Floor": np.arange(1, n+1),
    "Weight W_i kN": weights,
    "Mode shape phi_i": phi1,
    "W_i phi_i": pattern_raw,
    "Floor force ratio": force_ratio
})

st.subheader("3. First-Mode Pushover Force Pattern")
st.dataframe(pattern_df, use_container_width=True)

# -----------------------------
# Yield capacity calculation
# -----------------------------
st.subheader("4. Yield Capacity per Storey from Plastic Moments")

# Column mechanism: 2*sum(Mp columns)/h
Vy_col_per_frame = 2.0 * cols * mp_col / h

# Beam mechanism: bays * 2 ends * Mp_beam / h
Vy_beam_per_frame = bays * 2.0 * mp_beam / h

Vy_col_total = Vy_col_per_frame * frame_multiplier
Vy_beam_total = Vy_beam_per_frame * frame_multiplier
Vy_governing = np.minimum(Vy_col_total, Vy_beam_total)

yield_df = pd.DataFrame({
    "Storey": np.arange(1, n+1),
    "Column-controlled Vy per frame (kN)": Vy_col_per_frame,
    "Beam-controlled Vy per frame (kN)": Vy_beam_per_frame,
    "Frame multiplier": frame_multiplier,
    "Column-controlled Vy total (kN)": Vy_col_total,
    "Beam-controlled Vy total (kN)": Vy_beam_total,
    "Governing Vy total (kN)": Vy_governing,
    "Yield drift dy = Vy/k (mm)": Vy_governing / k_storey * 1000
})
st.dataframe(yield_df, use_container_width=True)

# -----------------------------
# Pushover calculation
# -----------------------------
st.subheader("5. Simplified MDOF Pushover Curve")

# Storey shear influence from floor force ratios:
# Storey i shear = sum of floor forces from i to top.
storey_shear_ratio = np.array([np.sum(force_ratio[i:]) for i in range(n)])

max_base_shear = st.slider(
    "Maximum base shear for curve (kN)",
    min_value=50.0,
    max_value=max(2000.0, float(np.max(Vy_governing)*3)),
    value=float(max(800.0, np.max(Vy_governing)*1.5)),
    step=10.0
)
num_steps = st.slider("Number of pushover points", min_value=10, max_value=200, value=41, step=1)

base_shears = np.linspace(0, max_base_shear, num_steps)
rows = []

for Vb in base_shears:
    storey_shears = storey_shear_ratio * Vb
    drifts_m = np.array([
        bilinear_drift(storey_shears[i], Vy_governing[i], k_storey[i], alpha)
        for i in range(n)
    ])
    floor_disp_m = np.cumsum(drifts_m)
    rows.append({
        "Base shear kN": Vb,
        "Roof displacement mm": floor_disp_m[-1] * 1000,
        "State": state_label(storey_shears, Vy_governing),
        **{f"Storey {i+1} shear kN": storey_shears[i] for i in range(n)},
        **{f"Storey {i+1} drift mm": drifts_m[i]*1000 for i in range(n)},
    })

pushover_df = pd.DataFrame(rows)

# Add exact yield points
yield_points = []
for i in range(n):
    if storey_shear_ratio[i] > 0:
        Vb_y = Vy_governing[i] / storey_shear_ratio[i]
        if 0 <= Vb_y <= max_base_shear:
            storey_shears = storey_shear_ratio * Vb_y
            drifts_m = np.array([
                bilinear_drift(storey_shears[j], Vy_governing[j], k_storey[j], alpha)
                for j in range(n)
            ])
            yield_points.append((i+1, Vb_y, np.sum(drifts_m)*1000))

c1, c2 = st.columns([1.2, 1])
with c1:
    fig, ax = plt.subplots()
    ax.plot(pushover_df["Roof displacement mm"], pushover_df["Base shear kN"], marker="o", markersize=3)
    for storey, Vb_y, roof_y in yield_points:
        ax.scatter([roof_y], [Vb_y])
        ax.annotate(f"S{storey} yield", (roof_y, Vb_y), textcoords="offset points", xytext=(5,5))
    ax.set_xlabel("Roof displacement, mm")
    ax.set_ylabel("Base shear, kN")
    ax.set_title("Simplified Pushover Curve")
    ax.grid(True)
    st.pyplot(fig)

with c2:
    st.write("Yield points:")
    if yield_points:
        st.dataframe(pd.DataFrame(yield_points, columns=["Storey", "Base shear at yield kN", "Roof displacement at yield mm"]), use_container_width=True)
    else:
        st.write("No yield point within selected maximum base shear.")

st.dataframe(pushover_df, use_container_width=True)

# -----------------------------
# RSA comparison
# -----------------------------
st.subheader("6. RSA / Static Base Shear Comparison")

cc1, cc2, cc3 = st.columns(3)
with cc1:
    static_base_shear = st.number_input("Static / code base shear to compare (kN)", min_value=0.0, value=104.15, step=1.0)
with cc2:
    dynamic_base_shear = st.number_input("Dynamic RSA base shear to compare (kN)", min_value=0.0, value=101.17, step=1.0)
with cc3:
    demand_choice = st.selectbox("Use demand for check", ["Static base shear", "Dynamic RSA base shear"], index=0)

demand = static_base_shear if demand_choice == "Static base shear" else dynamic_base_shear
storey_shears = storey_shear_ratio * demand
drifts_m = np.array([bilinear_drift(storey_shears[i], Vy_governing[i], k_storey[i], alpha) for i in range(n)])
roof_demand = np.sum(drifts_m)*1000
min_yield_vb = min([yp[1] for yp in yield_points], default=np.nan)

r1, r2, r3 = st.columns(3)
r1.metric("Selected demand base shear", f"{demand:.2f} kN")
r2.metric("Estimated roof displacement at demand", f"{roof_demand:.2f} mm")
if not np.isnan(min_yield_vb):
    r3.metric("First yield base shear", f"{min_yield_vb:.2f} kN")

if not np.isnan(min_yield_vb):
    if demand < min_yield_vb:
        st.success("Demand is below first yield in this simplified model.")
    else:
        st.warning("Demand exceeds first yield in this simplified model. Nonlinear behavior is expected.")

# -----------------------------
# Formula notes
# -----------------------------
with st.expander("Show manual formulas used"):
    st.markdown(r"""
### Shear-building stiffness matrix

For 2 storeys:

\[
K=
\begin{bmatrix}
k_1+k_2 & -k_2 \\
-k_2 & k_2
\end{bmatrix}
\]

### First-mode lateral force pattern

\[
F_i = V_b \frac{W_i\phi_i}{\sum W_i\phi_i}
\]

### Storey shear

\[
V_{storey,i}=\sum_{j=i}^{n} F_j
\]

### Yield shear from plastic moments

Column-controlled:

\[
V_{y,col}=\frac{2\sum M_{p,col}}{h}
\]

Beam-controlled:

\[
V_{y,beam}=\frac{\sum M_{p,beam}}{h}
\]

Governing:

\[
V_y=\min(V_{y,col},V_{y,beam})
\]

### Bilinear drift

Elastic:

\[
\delta = \frac{V}{k}
\]

Post-yield:

\[
\delta = \delta_y+\frac{V-V_y}{\alpha k}
\]

where:

\[
\delta_y = \frac{V_y}{k}
\]

### Roof displacement

\[
\Delta_{roof}=\sum \delta_i
\]
""")

# -----------------------------
# Downloads
# -----------------------------
st.subheader("7. Downloads")
csv1 = pushover_df.to_csv(index=False).encode("utf-8")
csv2 = yield_df.to_csv(index=False).encode("utf-8")
st.download_button("Download pushover table CSV", csv1, "pushover_curve.csv", "text/csv")
st.download_button("Download yield capacity table CSV", csv2, "yield_capacity.csv", "text/csv")
