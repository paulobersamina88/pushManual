
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="RSA–Pushover Reconciliation Tool",
    layout="wide"
)

st.title("RSA–Pushover Reconciliation Tool")
st.caption("Online Streamlit version: simplified manual MDOF pushover curve + response spectrum/base shear comparison")

# ----------------------------------------------------
# FUNCTIONS
# ----------------------------------------------------
def assemble_K(k_storey):
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
    periods = 2 * np.pi / omegas

    phis = eigvecs.copy()

    # normalize each mode to roof = 1
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
        effective_mass = (num ** 2) / den
        ratio = effective_mass / total_mass * 100

        rows.append({
            "Mode": j + 1,
            "Omega_rad_per_s": omegas[j],
            "Period_s": periods[j],
            "Gamma": gamma,
            "Modal_Mass": den,
            "Effective_Modal_Mass": effective_mass,
            "Eff_Mass_Ratio_%": ratio,
        })

    modal_df = pd.DataFrame(rows)
    modal_df["Cumulative_%"] = modal_df["Eff_Mass_Ratio_%"].cumsum()

    return modal_df, phis, K, M


def bilinear_drift(storey_shear, Vy, k, alpha):
    if k <= 0:
        return 0.0

    dy = Vy / k

    if storey_shear <= Vy:
        return storey_shear / k

    kp = max(alpha * k, 1e-9)
    return dy + (storey_shear - Vy) / kp


def state_label(storey_shears, Vy):
    yielded = [i + 1 for i, (v, y) in enumerate(zip(storey_shears, Vy)) if v >= y - 1e-9]

    if len(yielded) == 0:
        return "Elastic"

    if len(yielded) == 1:
        return f"Storey {yielded[0]} yielded"

    return "Storey " + " + ".join(map(str, yielded)) + " yielded"


def compute_yield_capacity(df, frame_multiplier):
    h = df["Storey height h_i (m)"].to_numpy(dtype=float)
    cols = df["Columns per frame"].to_numpy(dtype=float)
    bays = df["Bays per frame"].to_numpy(dtype=float)
    mp_col = df["Column Mp each (kN-m)"].to_numpy(dtype=float)
    mp_beam = df["Beam Mp per end (kN-m)"].to_numpy(dtype=float)

    # Column mechanism: hinges at both ends of columns
    Vy_col_per_frame = 2.0 * cols * mp_col / h

    # Beam mechanism: 2 plastic ends per bay
    Vy_beam_per_frame = bays * 2.0 * mp_beam / h

    Vy_col_total = Vy_col_per_frame * frame_multiplier
    Vy_beam_total = Vy_beam_per_frame * frame_multiplier
    Vy_gov = np.minimum(Vy_col_total, Vy_beam_total)

    return Vy_col_per_frame, Vy_beam_per_frame, Vy_col_total, Vy_beam_total, Vy_gov


# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------
st.sidebar.header("Model Settings")

n = st.sidebar.number_input("Number of storeys", min_value=1, max_value=10, value=2, step=1)

alpha = st.sidebar.selectbox(
    "Post-yield stiffness ratio α",
    [0.03, 0.05, 0.10, 0.15],
    index=1
)

frame_count = st.sidebar.number_input(
    "Number of similar frames in this axis",
    min_value=1,
    max_value=20,
    value=3,
    step=1
)

basis = st.sidebar.radio(
    "Plastic moment input basis",
    ["Per frame - multiply by number of frames", "Whole axis - do not multiply"],
    index=0
)

frame_multiplier = frame_count if basis.startswith("Per frame") else 1

st.sidebar.markdown("---")
st.sidebar.info("Use consistent basis: mass = stiffness = plastic moment capacity basis.")

# ----------------------------------------------------
# INPUT TABLE
# ----------------------------------------------------
st.subheader("1. Floor / Storey Input")

default_rows = []
for i in range(n):
    if i == 0:
        W = 422.53
        k = 56960.0
    elif i == 1:
        W = 382.30
        k = 16720.0
    else:
        W = 350.0
        k = 15000.0

    default_rows.append({
        "Storey": i + 1,
        "Floor weight W_i (kN)": W,
        "Storey stiffness k_i (kN/m)": k,
        "Storey height h_i (m)": 3.0,
        "Columns per frame": 3,
        "Bays per frame": 2,
        "Column Mp each (kN-m)": 300.0,
        "Beam Mp per end (kN-m)": 150.0,
    })

df_in = st.data_editor(
    pd.DataFrame(default_rows),
    use_container_width=True,
    num_rows="fixed",
    key="input_table"
)

weights = df_in["Floor weight W_i (kN)"].to_numpy(dtype=float)
k_storey = df_in["Storey stiffness k_i (kN/m)"].to_numpy(dtype=float)

# ----------------------------------------------------
# MODAL PROPERTIES
# ----------------------------------------------------
st.subheader("2. Modal Properties")

try:
    modal_df, phis, K, M = modal_properties(weights, k_storey)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("T1 (s)", f"{modal_df.loc[0, 'Period_s']:.4f}")
    c2.metric("1st Mode Mass Participation", f"{modal_df.loc[0, 'Eff_Mass_Ratio_%']:.2f}%")
    c3.metric("Cumulative Mass", f"{modal_df['Cumulative_%'].iloc[-1]:.2f}%")
    c4.metric("Total Weight", f"{weights.sum():.2f} kN")

    st.dataframe(modal_df.round(4), use_container_width=True)

except Exception as e:
    st.error(f"Modal analysis failed. Check inputs. Error: {e}")
    st.stop()

# ----------------------------------------------------
# FORCE PATTERN
# ----------------------------------------------------
st.subheader("3. First Mode Pushover Force Pattern")

phi1 = phis[:, 0]

# Use positive mode shape for pushover
if np.sum(phi1) < 0:
    phi1 = -phi1

pattern_raw = weights * phi1

if np.any(pattern_raw < 0):
    pattern_raw = np.abs(pattern_raw)

force_ratio = pattern_raw / pattern_raw.sum()

pattern_df = pd.DataFrame({
    "Floor": np.arange(1, n + 1),
    "Weight W_i (kN)": weights,
    "Mode shape phi_i": phi1,
    "W_i phi_i": pattern_raw,
    "Floor force ratio": force_ratio
})

st.dataframe(pattern_df.round(5), use_container_width=True)

fig_pattern = go.Figure()
fig_pattern.add_trace(go.Bar(
    x=pattern_df["Floor"],
    y=pattern_df["Floor force ratio"],
    text=[f"{x*100:.1f}%" for x in pattern_df["Floor force ratio"]],
    textposition="auto"
))
fig_pattern.update_layout(
    title="Lateral Force Distribution",
    xaxis_title="Floor",
    yaxis_title="Force Ratio",
    height=350
)
st.plotly_chart(fig_pattern, use_container_width=True)

# ----------------------------------------------------
# YIELD CAPACITY
# ----------------------------------------------------
st.subheader("4. Yield Capacity per Storey")

Vy_col_pf, Vy_beam_pf, Vy_col_total, Vy_beam_total, Vy_gov = compute_yield_capacity(df_in, frame_multiplier)

yield_df = pd.DataFrame({
    "Storey": np.arange(1, n + 1),
    "Column Vy per frame (kN)": Vy_col_pf,
    "Beam Vy per frame (kN)": Vy_beam_pf,
    "Frame multiplier": frame_multiplier,
    "Column Vy total (kN)": Vy_col_total,
    "Beam Vy total (kN)": Vy_beam_total,
    "Governing Vy total (kN)": Vy_gov,
    "Yield drift dy (mm)": Vy_gov / k_storey * 1000
})

st.dataframe(yield_df.round(3), use_container_width=True)

# ----------------------------------------------------
# PUSHOVER CURVE
# ----------------------------------------------------
st.subheader("5. Simplified MDOF Pushover Curve")

# Storey shear ratio = sum of lateral forces above and at the storey
storey_shear_ratio = np.array([force_ratio[i:].sum() for i in range(n)])

col1, col2 = st.columns(2)
with col1:
    max_vb = st.number_input(
        "Maximum base shear to plot (kN)",
        min_value=50.0,
        value=float(max(800.0, np.max(Vy_gov) * 1.5)),
        step=50.0
    )
with col2:
    n_steps = st.number_input(
        "Number of calculation steps",
        min_value=10,
        max_value=300,
        value=61,
        step=1
    )

base_shears = np.linspace(0, max_vb, int(n_steps))
rows = []

for Vb in base_shears:
    storey_shears = storey_shear_ratio * Vb

    drifts_m = np.array([
        bilinear_drift(storey_shears[i], Vy_gov[i], k_storey[i], alpha)
        for i in range(n)
    ])

    floor_disp_m = np.cumsum(drifts_m)

    row = {
        "Base shear kN": Vb,
        "Roof displacement mm": floor_disp_m[-1] * 1000,
        "State": state_label(storey_shears, Vy_gov)
    }

    for i in range(n):
        row[f"Storey {i+1} shear kN"] = storey_shears[i]
        row[f"Storey {i+1} drift mm"] = drifts_m[i] * 1000
        row[f"Floor {i+1} displacement mm"] = floor_disp_m[i] * 1000

    rows.append(row)

pushover_df = pd.DataFrame(rows)

yield_points = []
for i in range(n):
    if storey_shear_ratio[i] > 0:
        Vb_y = Vy_gov[i] / storey_shear_ratio[i]

        if 0 <= Vb_y <= max_vb:
            storey_shears_y = storey_shear_ratio * Vb_y
            drifts_y = np.array([
                bilinear_drift(storey_shears_y[j], Vy_gov[j], k_storey[j], alpha)
                for j in range(n)
            ])
            roof_y = drifts_y.sum() * 1000
            yield_points.append({
                "Storey": i + 1,
                "Base shear at yield kN": Vb_y,
                "Roof displacement at yield mm": roof_y
            })

fig_push = go.Figure()
fig_push.add_trace(go.Scatter(
    x=pushover_df["Roof displacement mm"],
    y=pushover_df["Base shear kN"],
    mode="lines+markers",
    name="Pushover curve"
))

for yp in yield_points:
    fig_push.add_trace(go.Scatter(
        x=[yp["Roof displacement at yield mm"]],
        y=[yp["Base shear at yield kN"]],
        mode="markers+text",
        text=[f"S{int(yp['Storey'])} yield"],
        textposition="top center",
        name=f"Storey {int(yp['Storey'])} yield"
    ))

fig_push.update_layout(
    title="Base Shear vs Roof Displacement",
    xaxis_title="Roof displacement (mm)",
    yaxis_title="Base shear (kN)",
    height=500
)

st.plotly_chart(fig_push, use_container_width=True)

if yield_points:
    st.write("Yield points:")
    st.dataframe(pd.DataFrame(yield_points).round(3), use_container_width=True)

st.dataframe(pushover_df.round(3), use_container_width=True)

# ----------------------------------------------------
# DEMAND COMPARISON
# ----------------------------------------------------
st.subheader("6. Demand Comparison")

c1, c2, c3 = st.columns(3)
with c1:
    static_v = st.number_input("Static base shear demand (kN)", min_value=0.0, value=104.15, step=1.0)
with c2:
    dynamic_v = st.number_input("Dynamic RSA base shear demand (kN)", min_value=0.0, value=101.17, step=1.0)
with c3:
    selected = st.selectbox("Demand to check", ["Static", "Dynamic RSA"])

demand_v = static_v if selected == "Static" else dynamic_v
storey_shears_d = storey_shear_ratio * demand_v
drifts_d = np.array([
    bilinear_drift(storey_shears_d[i], Vy_gov[i], k_storey[i], alpha)
    for i in range(n)
])
roof_d = drifts_d.sum() * 1000

if yield_points:
    first_yield_v = min([p["Base shear at yield kN"] for p in yield_points])
else:
    first_yield_v = np.nan

d1, d2, d3 = st.columns(3)
d1.metric("Demand base shear", f"{demand_v:.2f} kN")
d2.metric("Estimated roof displacement", f"{roof_d:.2f} mm")
if not np.isnan(first_yield_v):
    d3.metric("First yield base shear", f"{first_yield_v:.2f} kN")

if not np.isnan(first_yield_v):
    if demand_v < first_yield_v:
        st.success("Demand is below first yield. Structure remains elastic in this simplified pushover model.")
    else:
        st.warning("Demand exceeds first yield. Nonlinear behavior is expected in this simplified model.")

# ----------------------------------------------------
# MANUAL FORMULAS
# ----------------------------------------------------
with st.expander("Manual calculation formulas used"):
    st.markdown(r"""
### First-mode pushover force pattern

\[
F_i = V_b \frac{W_i \phi_i}{\sum W_i \phi_i}
\]

### Storey shear

\[
V_i = \sum_{j=i}^{n} F_j
\]

### Column-controlled storey yield

\[
V_{y,col} = \frac{2 \sum M_{p,col}}{h}
\]

### Beam-controlled storey yield

\[
V_{y,beam} = \frac{\sum M_{p,beam}}{h}
\]

### Governing storey yield

\[
V_y = \min(V_{y,col}, V_{y,beam})
\]

### Elastic drift

\[
\delta = \frac{V}{k}
\]

### Post-yield bilinear drift

\[
\delta = \delta_y + \frac{V - V_y}{\alpha k}
\]

### Roof displacement

\[
\Delta_{roof} = \sum \delta_i
\]
""")

# ----------------------------------------------------
# DOWNLOADS
# ----------------------------------------------------
st.subheader("7. Download Results")

st.download_button(
    "Download pushover curve CSV",
    pushover_df.to_csv(index=False).encode("utf-8"),
    file_name="pushover_curve.csv",
    mime="text/csv"
)

st.download_button(
    "Download yield capacity CSV",
    yield_df.to_csv(index=False).encode("utf-8"),
    file_name="yield_capacity.csv",
    mime="text/csv"
)

st.download_button(
    "Download modal properties CSV",
    modal_df.to_csv(index=False).encode("utf-8"),
    file_name="modal_properties.csv",
    mime="text/csv"
)
