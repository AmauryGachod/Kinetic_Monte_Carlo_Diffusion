import streamlit as st
import numpy as np
import plotly.graph_objs as go
import json
# numba is optional; left imported at the top for future acceleration but not used with @njit here
from numba import njit
st.set_page_config(
    page_title="Monte Carlo Diffusion Dashboard",
    page_icon="🧪",
    layout="wide"
)


# ----------------------
# Utility functions
# ----------------------

def prepare_json_download(positions):
    """Prepare a JSON string for downloading a trajectory.

    positions: ndarray of shape (N,2) with columns x,y
    Returns a pretty-printed JSON string with time, x and y arrays.
    """
    t_vals = np.arange(len(positions))
    data_dict = {"time": t_vals.tolist(),
                 "x": positions[:, 0].tolist(),
                 "y": positions[:, 1].tolist()}
    return json.dumps(data_dict, indent=4)

# ----------------------
# KMC simulations
# ----------------------

def make_rng(seed=None):
    """Create a numpy random generator (harmonized with notebook)."""
    return np.random.default_rng(seed)

def kmc_classical(num_steps, Gamma1, Gamma2, a=1.0, seed=None):
    """Classical KMC simulation (harmonized with notebook)."""
    rng = make_rng(seed)
    Gamma = float(Gamma1) + float(Gamma2)
    if Gamma <= 0:
        raise ValueError("Γ₁ + Γ₂ must be > 0")
    dt = 1.0 / Gamma

    pos = np.zeros((num_steps + 1, 2))
    times = np.zeros(num_steps + 1)

    moves_10 = np.array([[1,0],[-1,0],[0,1],[0,-1]], dtype=float)
    moves_11 = np.array([[1,1],[-1,-1],[1,-1],[-1,1]], dtype=float)

    for n in range(num_steps):
        r = rng.random()
        move = moves_10[rng.integers(0,4)] if r < (Gamma1/Gamma) else moves_11[rng.integers(0,4)]
        pos[n+1] = pos[n] + move
        times[n+1] = times[n] + dt

    return pos * a, times

def kmc_modified_with_O(num_steps, Gamma_trans, Gamma_rot, a=1.0, b=0.25, seed=None):
    """Modified KMC with O atom tracking (harmonized with notebook)."""
    rng = make_rng(seed)
    Gamma = Gamma_trans + Gamma_rot
    if Gamma <= 0:
        raise ValueError("Gamma_trans + Gamma_rot must be > 0")
    dt = 1.0 / Gamma

    state_unit = {
        0: np.array([1.0, 0.0]),   # right
        1: np.array([0.0, 1.0]),   # up
        2: np.array([-1.0, 0.0]),  # left
        3: np.array([0.0, -1.0])   # down
    }
    neighbor_offset = {
        0: np.array([1, 0]),   # right neighbor
        1: np.array([0, 1]),   # up neighbor
        2: np.array([-1, 0]),  # left neighbor
        3: np.array([0, -1])   # down neighbor
    }

    pos_H = np.zeros((num_steps + 1, 2))
    pos_O = np.zeros((num_steps + 1, 2))
    times = np.zeros(num_steps + 1)

    ix, iy = 0, 0
    state = 0

    O_pos = np.array([ix * a, iy * a], dtype=float)
    H_pos = O_pos + state_unit[state] * b
    pos_H[0] = H_pos
    pos_O[0] = O_pos

    prob_trans = Gamma_trans / Gamma

    for n in range(num_steps):
        u = rng.random()
        if u < prob_trans and Gamma_trans > 0:
            off = neighbor_offset[state]
            ix += int(off[0])
            iy += int(off[1])
            state = (state + 2) % 4
            O_pos = np.array([ix * a, iy * a], dtype=float)
            H_pos = O_pos + state_unit[state] * b
        else:
            if rng.random() < 0.5:
                state = (state - 1) % 4
            else:
                state = (state + 1) % 4
            O_pos = np.array([ix * a, iy * a], dtype=float)
            H_pos = O_pos + state_unit[state] * b
        pos_H[n+1] = H_pos
        pos_O[n+1] = O_pos
        times[n+1] = times[n] + dt
    return pos_H, times, pos_O

# ----------------------
# MSD and diffusion utilities
# ----------------------

# def compute_msd_per_component(positions):
#     N = positions.shape[0]
#     max_lag = N//4
#     msd_x = np.zeros(max_lag)
#     msd_y = np.zeros(max_lag)
#     x = positions[:,0]
#     y = positions[:,1]
#     for lag in range(1, max_lag):
#         diffs_x = x[lag:] - x[:-lag]
#         diffs_y = y[lag:] - y[:-lag]
#         msd_x[lag] = np.mean(diffs_x**2)
#         msd_y[lag] = np.mean(diffs_y**2)
#     return np.arange(1,max_lag), msd_x[1:], msd_y[1:]

# (top-level imports already include numba and numpy)

def compute_msd_zero_offset(positions, max_lag=None):
    """Compute MSD using zero-offset method (harmonized with notebook)."""
    N = len(positions)
    if max_lag is None:
        max_lag = N // 4
    lags = np.arange(1, max_lag+1)
    msd = np.zeros_like(lags, dtype=float)
    for i, lag in enumerate(lags):
        diffs = positions[lag:] - positions[:-lag]
        msd[i] = np.mean(np.sum(diffs**2, axis=1))
    return lags, msd




def fit_diffusion_coefficient(lags, msd, Gamma1, Gamma2, fit_range=(200,2000)):
    """Fit D using notebook's least squares approach."""
    dt = 1.0 / (Gamma1 + Gamma2)
    lo, hi = fit_range
    if hi > lags.max():
        hi = int(lags.max())
    sel = (lags >= lo) & (lags <= hi)
    A = np.vstack([lags[sel]*dt, np.ones(sel.sum())]).T
    slope, intercept = np.linalg.lstsq(A, msd[sel], rcond=None)[0]
    return slope/4.0, slope, intercept

def analytic_D(Gamma1, Gamma2, d1, d2):
    """Analytical D (harmonized with notebook)."""
    return 0.25 * (Gamma1*d1**2 + Gamma2*d2**2)

# ----------------------
# Streamlit application
# ----------------------

def main():
    # Streamlit UI setup
    st.title("Monte Carlo Diffusion: Classical & Modified")
    model_type = st.radio("Choose diffusion model:", ["Classical","Modified"], horizontal=True)
    num_steps = st.slider("Number of steps", 1000, 500000, 10000, 1000)

    # Number of simulations to compare
    num_sim = st.slider("Number of simulations", 1, 3, 2, 1)

    # dynamically generate Γ₁ sliders for each simulation case
    ratios = []
    cols = st.columns(num_sim)
    for i in range(num_sim):
        with cols[i]:
            r = st.slider(f"Γ₁ case {i+1}", 0.0, 1.0, 0.25 + 0.25*i, 0.01)
            ratios.append(r)

    # Model parameters
    a = 1.0
    b = None
    if model_type=="Classical":
        a = st.number_input("Parameter a (Classical)", value=1.0)
    else:
        a = st.number_input("Parameter a", value=3.0)
        b = st.number_input("Parameter b", value=1.0)

    # Run simulations when requested
    if st.button("Run simulation"):
        results = []
        for idx, ratio in enumerate(ratios, start=1):
            Gamma1 = ratio
            Gamma2 = 1 - ratio
            seed = idx  # ensure reproducibility for each simulation
            if model_type == "Classical":
                traj, sim_times = kmc_classical(num_steps, Gamma1, Gamma2, a, seed=seed)
                # For analytic D, use d1=a, d2=sqrt(2)*a
                d1, d2 = a, np.sqrt(2)*a
            else:
                pos_H, sim_times, pos_O = kmc_modified_with_O(num_steps, Gamma1, Gamma2, a, b, seed=seed)
                traj = pos_H
                # For analytic D, use d1=a-2*b, d2=sqrt(2)*b
                d1, d2 = a-2*b, np.sqrt(2)*b

            lags, msd = compute_msd_zero_offset(traj, 4000)
            D_num, slope, intercept = fit_diffusion_coefficient(lags, msd, Gamma1, Gamma2)
            D_ana = analytic_D(Gamma1, Gamma2, d1, d2)

            json_data = prepare_json_download(traj)
            st.download_button(
                label=f"Download trajectory (Γ₁={ratio:.2f})",
                data=json_data,
                file_name=f"traj_Gamma1_{ratio:.2f}.json",
                mime="application/json",
                key=idx
            )
            results.append((ratio, traj, sim_times, lags, msd, D_num, D_ana))

        st.session_state['results'] = results
    # --- Display ---
    if 'results' in st.session_state:
        results = st.session_state['results']
        # number of simulations to display based on generated results
        num_sim = len(results)

        # create columns dynamically based on current number of simulations
        cols = st.columns(num_sim)

        for idx, res in enumerate(results):
            # limit to current selection if necessary
            ratio, traj, sim_times, lags, msd, D_num, D_ana = res
            col = cols[idx % num_sim]
            with col:
                st.markdown(f"### Γ₁={ratio:.2f} | Γ₂={1-ratio:.2f}")
                st.metric("Estimated D", f"{D_num:.6f}")
                st.metric("Analytical D", f"{D_ana:.6f}")

                x_vals, y_vals = traj[:,0], traj[:,1]
                # prefer real simulation times if available and same length as trajectory
                if sim_times is not None and len(sim_times) == len(x_vals):
                    z_vals = sim_times
                else:
                    z_vals = np.arange(len(x_vals))

                min_val = min(np.min(x_vals), np.min(y_vals))
                max_val = max(np.max(x_vals), np.max(y_vals))

                fig_traj = go.Figure()
                fig_traj.add_trace(go.Scatter3d(
                    x=x_vals, y=y_vals, z=z_vals, mode='lines+markers',
                    line=dict(color='lightgray'), marker=dict(size=3), name='Trajectory'
                ))
                fig_traj.add_trace(go.Scatter3d(
                    x=[x_vals[0]], y=[y_vals[0]], z=[z_vals[0]],
                    mode='markers', marker=dict(color='green', size=10), name='Start'
                ))
                fig_traj.add_trace(go.Scatter3d(
                    x=[x_vals[-1]], y=[y_vals[-1]], z=[z_vals[-1]],
                    mode='markers', marker=dict(color='red', size=10), name='End'
                ))

                fig_traj.update_layout(
                    scene=dict(
                        xaxis=dict(title='x', range=[min_val,max_val]),
                        yaxis=dict(title='y', range=[min_val,max_val]),
                        zaxis=dict(title='t'),
                        aspectmode='manual',
                        aspectratio=dict(x=1,y=1,z=0.5)
                    ),
                    margin=dict(l=20,r=20,t=40,b=20)
                )
                st.plotly_chart(fig_traj, use_container_width=True)

                # MSD plot
                fig_msd = go.Figure()
                fig_msd.add_trace(go.Scatter(x=lags, y=msd, mode='lines', name='MSD total'))
                # add linear fit only when there are enough points
                if len(lags) >= 2:
                    fit_line = D_num * 4 * lags + (msd[0] if len(msd) > 0 else 0)
                    fig_msd.add_trace(go.Scatter(x=lags, y=fit_line,
                                                mode='lines', line=dict(dash='dash'), name=f'Linear fit D={D_num:.6f}'))
                fig_msd.update_layout(title="MSD(t)", xaxis_title="t (steps)", yaxis_title="MSD(t)")
                st.plotly_chart(fig_msd, use_container_width=True)



if __name__=="__main__":
    main()
