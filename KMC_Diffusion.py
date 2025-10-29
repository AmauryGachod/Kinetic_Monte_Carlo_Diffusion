import streamlit as st
import numpy as np
import plotly.graph_objs as go
import json
from numba import njit  # pour accélérer les boucles KMC

st.set_page_config(
    page_title="Monte Carlo Diffusion Dashboard",
    page_icon="🧪",
    layout="wide"
)

# ----------------------
# Fonctions utilitaires
# ----------------------

def prepare_json_download(positions):
    t_vals = np.arange(len(positions))
    data_dict = {"time": t_vals.tolist(),
                 "x": positions[:, 0].tolist(),
                 "y": positions[:, 1].tolist()}
    return json.dumps(data_dict, indent=4)

# ----------------------
# KMC Simulation
# ----------------------

@njit
def kmc_classical(num_steps, Gamma1, Gamma2, a):
    positions = np.zeros((num_steps+1, 2))
    d1 = a
    d2 = np.sqrt(2) * a
    directions_type1 = np.array([[d1, 0], [-d1, 0], [0, d1], [0, -d1]])
    directions_type2 = np.array([[d2, d2], [d2, -d2], [-d2, d2], [-d2, -d2]])
    Gamma_tot = Gamma1 + Gamma2
    prob_gamma1 = Gamma1 / Gamma_tot

    for i in range(1, num_steps+1):
        r = np.random.rand()
        if r < prob_gamma1:
            step = directions_type1[np.random.randint(0, 4)]
        else:
            step = directions_type2[np.random.randint(0, 4)]
        positions[i] = positions[i-1] + step
    return positions

def kmc_modified(num_steps, Gamma1, Gamma2, a, b):
    positions = np.zeros((num_steps+1, 2))
    d1 = a - 2*b
    d_step = b  # pour rotation, pas x=y

    O_position = np.array([0.0, 0.0])
    H_position = O_position + np.array([b, 0.0])
    positions[0] = H_position.copy()
    Gamma_tot = Gamma1 + Gamma2
    prob_trans = Gamma1 / Gamma_tot

    for i in range(1, num_steps+1):
        r = np.random.rand()
        step = np.zeros(2)

        if r < prob_trans and Gamma1 > 0:
            # Translation
            directions = np.array([[1,0], [-1,0], [0,1], [0,-1]])
            dir_vec = directions[np.random.randint(0,4)]
            step = dir_vec * d1
            O_position += dir_vec * a
            H_position = O_position - dir_vec * b
        else:
            # Rotation autour du même O
            
                rot_dir = np.random.choice(['cw', 'ccw'])
                angle = np.pi/2 if rot_dir=='ccw' else -np.pi/2
                rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                       [np.sin(angle), np.cos(angle)]])
                rel_vec = H_position - O_position
                new_H_pos = O_position + rot_matrix @ rel_vec
                step = new_H_pos - H_position
                H_position = new_H_pos

        positions[i] = positions[i-1] + step
    return positions

# ----------------------
# MSD et diffusion
# ----------------------

def compute_msd_per_component(positions):
    N = positions.shape[0]
    max_lag = N//4
    msd_x = np.zeros(max_lag)
    msd_y = np.zeros(max_lag)
    x = positions[:,0]
    y = positions[:,1]
    for lag in range(1, max_lag):
        diffs_x = x[lag:] - x[:-lag]
        diffs_y = y[lag:] - y[:-lag]
        msd_x[lag] = np.mean(diffs_x**2)
        msd_y[lag] = np.mean(diffs_y**2)
    return np.arange(1,max_lag), msd_x[1:], msd_y[1:]

def estimate_diffusion_coefficient(t_vals, msd_x, msd_y, fit_ratio=1):
    max_fit = int(len(t_vals)*fit_ratio)
    t_fit = t_vals[:max_fit]
    msd_total = msd_x[:max_fit]+msd_y[:max_fit]
    slope = np.polyfit(t_fit, msd_total,1)[0]
    return slope/4  # 2D

def analytical_diffusion_coefficient(Gamma1, Gamma2, a=1.0, b=None):
    if b is None:
        d1 = a
        d2 = np.sqrt(2)*a
    else:
        d1 = a-2*b
        d2 = np.sqrt(2)*b
    return 0.25*(Gamma1*d1**2 + Gamma2*d2**2)

# ----------------------
# Streamlit App
# ----------------------

def main():
    # ----------------------
# Paramètres Streamlit
# ----------------------
    st.title("Monte Carlo Diffusion: Classical & Modified")
    model_type = st.radio("Choose diffusion model:", ["Classical","Modified"], horizontal=True)
    num_steps = st.slider("Number of steps", 1000, 500000, 10000, 1000)

    # Choix du nombre de simulations à comparer
    num_sim = st.slider("Number of simulations", 1, 3, 2, 1)

    # Générer dynamiquement les Γ₁
    ratios = []
    cols = st.columns(num_sim)
    for i in range(num_sim):
        with cols[i]:
            r = st.slider(f"Γ₁ case {i+1}", 0.0, 1.0, 0.25 + 0.25*i, 0.01)
            ratios.append(r)

    # Paramètres du modèle
    a = 1.0
    b = None
    if model_type=="Classical":
        a = st.number_input("Parameter a (Classical)", value=1.0)
    else:
        a = st.number_input("Parameter a", value=3.0)
        b = st.number_input("Parameter b", value=1.0)

    # ----------------------
    # Lancer simulation
    # ----------------------
    if st.button("Run simulation"):
        results = []
        for idx, ratio in enumerate(ratios, start=1):
            Gamma1 = ratio
            Gamma2 = 1 - ratio
            if model_type=="Classical":
                traj = kmc_classical(num_steps, Gamma1, Gamma2, a)
            else:
                traj = kmc_modified(num_steps, Gamma1, Gamma2, a, b)
            t_vals, msd_x, msd_y = compute_msd_per_component(traj)
            D_num = estimate_diffusion_coefficient(t_vals, msd_x, msd_y)
            D_ana = analytical_diffusion_coefficient(Gamma1, Gamma2, a, b)

            json_data = prepare_json_download(traj)
            st.download_button(
                label=f"Download trajectory (Γ₁={ratio:.2f})",
                data=json_data,
                file_name=f"traj_Gamma1_{ratio:.2f}.json",
                mime="application/json",
                key=idx
            )
            results.append((ratio, traj, t_vals, msd_x, msd_y, D_num, D_ana))

        st.session_state['results'] = results
   # --- Affichage ---
    if 'results' in st.session_state:
        results = st.session_state['results']

        # Nombre de simulations à afficher, selon le nombre de résultats générés
        num_sim = len(results)

    # Créer dynamiquement les colonnes selon le nombre de simulations actuelles
        cols = st.columns(num_sim)

        for idx, res in enumerate(results):
            # limiter à la sélection actuelle si nécessaire
            ratio, traj, t_vals, msd_x, msd_y, D_num, D_ana = res
            col = cols[idx % num_sim]  # assignation colonne correcte
            with col:
                st.markdown(f"### Γ₁={ratio:.2f} | Γ₂={1-ratio:.2f}")
                st.metric("Estimated D", f"{D_num:.6f}")
                st.metric("Analytical D", f"{D_ana:.6f}")

                x_vals, y_vals = traj[:,0], traj[:,1]
                t_vals_triplet = np.arange(len(x_vals))

                min_val = min(np.min(x_vals), np.min(y_vals))
                max_val = max(np.max(x_vals), np.max(y_vals))

                fig_traj = go.Figure()
                fig_traj.add_trace(go.Scatter3d(
                    x=x_vals, y=y_vals, z=t_vals_triplet, mode='lines+markers',
                    line=dict(color='lightgray'), marker=dict(size=3), name='Trajectory'
                ))
                fig_traj.add_trace(go.Scatter3d(
                    x=[x_vals[0]], y=[y_vals[0]], z=[t_vals_triplet[0]],
                    mode='markers', marker=dict(color='green', size=10), name='Start'
                ))
                fig_traj.add_trace(go.Scatter3d(
                    x=[x_vals[-1]], y=[y_vals[-1]], z=[t_vals_triplet[-1]],
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
                fig_msd.add_trace(go.Scatter(x=t_vals, y=msd_x, mode='lines', name='MSD x'))
                fig_msd.add_trace(go.Scatter(x=t_vals, y=msd_y, mode='lines', name='MSD y'))
                msd_total = msd_x + msd_y
                fig_msd.add_trace(go.Scatter(x=t_vals, y=msd_total, mode='lines', name='MSD x+y', line=dict(color='purple')))
                coeffs_total = np.polyfit(t_vals, msd_total,1)
                fig_msd.add_trace(go.Scatter(x=t_vals, y=coeffs_total[0]*t_vals+coeffs_total[1],
                                            mode='lines', line=dict(dash='dash'), name=f'Linear fit D={D_num:.6f}'))
                fig_msd.update_layout(title="MSD(t)", xaxis_title="t (steps)", yaxis_title="MSD(t)")
                st.plotly_chart(fig_msd, use_container_width=True)



if __name__=="__main__":
    main()
