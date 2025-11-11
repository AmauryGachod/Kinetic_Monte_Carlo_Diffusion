# Kinetic Monte Carlo Simulation of Hydrogen Diffusion

Simulation of hydrogen atom diffusion on a 2D square lattice of oxygen atoms using Kinetic Monte Carlo (KMC) methods.

## Overview

This project compares two diffusion models:

1. **Classical model**: Independent lattice jumps (nearest-neighbor and diagonal)
2. **Modified model**: Hydrogen covalently bound to oxygen, capable of translation and rotation (±90°)

The modified model exhibits temporal correlations between rotations and translations, leading to nonlinear diffusion behavior.

## Features

- Interactive Streamlit dashboard
- 3D trajectory visualization
- Mean squared displacement (MSD) analysis with zero-offset averaging
- Numerical vs analytical comparison of diffusion coefficients

## Installation

    git clone https://github.com/AmauryGachod/Kinetic_Monte_Carlo_Diffusion.git
    cd Kinetic_Monte_Carlo_Diffusion
    pip install -r requirements.txt
    streamlit run kmc_hydrogen_diffusion.py

## Usage

1. Select model (Classical or Modified)
2. Set parameters: number of steps, transition rates Γ₁ and Γ₂
3. Run simulation and visualize trajectories and MSD
4. Compare numerical results with analytical predictions

## Project Structure

    Kinetic_Monte_Carlo_Diffusion/
    ├── KMC_Diffusion.py      # Main dashboard
    ├── KMC_Analysis.ipynb   # Theoretical analysis
    ├── requirements.txt
    └── README.md

## Key Results

- **Classical model**: Numerical D matches analytical predictions (< 2% error)
- **Modified model**: Deviations due to rotation-translation coupling
- **Optimal diffusion**: Maximum D at Γ₁ ≈ 0.41 (balance between processes)

## Scientific Background

The modified model's diffusion coefficient follows:

    D(Γ₁) = L² · Γ₁(1-Γ₁) / [4(1+Γ₁)]

where L = a - 2b (effective translation distance).

## Dependencies

- numpy, streamlit, plotly, pandas, scipy


