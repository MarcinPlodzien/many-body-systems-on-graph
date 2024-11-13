#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 2024

@author: Macin Płodzień
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time

# Parameters
U = 1.0       # On-site interaction strength
n_max = 5     # Maximum occupation number per site
L = 10         # Number of sites in the cycle graph

# Create a cycle graph (ring) with L nodes

L = 259
G = nx.cycle_graph(L)

r = 6
h = 3


node_size = 100
# G = nx.balanced_tree(r, h)


L = G.number_of_nodes()
# pos = nx.nx_agraph.graphviz_layout(G, prog="twopi", args="")
pos = nx.circular_layout(G)
nx.draw(G, pos, with_labels=False, node_color='darkblue', edge_color='black', node_size=node_size)
#%%
# Function definitions
def compute_phi_i(f_i_n_i):
    """Compute the local order parameter phi_i from f_i_n at site i."""
    phi = 0.0
    for n in range(1, len(f_i_n_i)):
        phi += np.sqrt(n) * np.conj(f_i_n_i[n-1]) * f_i_n_i[n]
    return phi

def compute_n_i(f_i_n_i):
    """Compute the average number of particles at site i."""
    n_i = 0.0
    for n in range(len(f_i_n_i)):
        n_i += n * abs(f_i_n_i[n])**2
    return n_i

def construct_H_i(U, mu, t, Phi_i, n_max, z_i, phi_i):
    """Construct the local Hamiltonian H_i at site i."""
    H = np.zeros((n_max+1, n_max+1), dtype=np.complex128)
    for n in range(n_max+1):
        # Diagonal elements
        H[n, n] = (U/2) * n * (n - 1) - mu * n - z_i * t * abs(phi_i)**2
        # Off-diagonal elements
        if n < n_max:
            H[n, n+1] = - t * Phi_i * np.sqrt(n+1)
            H[n+1, n] = - t * np.conj(Phi_i) * np.sqrt(n+1)
    return H

# Parameter ranges for mu/U and t/U
mu_values = np.linspace(0, 2, 50)
t_over_U_values = np.linspace(0, 0.4, 50)  # Adjusted upper limit

# Initialize array to store average |phi_i| values
phi_avg_values = np.zeros((len(t_over_U_values), len(mu_values)))

start_time = time.time()
# Loop over t/U and mu
for idx_t, t_over_U in enumerate(t_over_U_values):
    t = t_over_U * U  # Compute t from t/U and U
    for idx_mu, mu in enumerate(mu_values * U):  # Convert mu/U to mu
        # Initialize the local wavefunction coefficients f_i_n and order parameters phi_i
        f_i_n = {}
        phi_i = {}

        for i in G.nodes():
            # Initialize f_i_n with uniform coefficients and normalize
            f_i_n[i] = np.ones(n_max+1, dtype=np.complex128) / np.sqrt(n_max+1)
            # Initialize phi_i with a small non-zero value
            phi_i[i] = 1e-2  # Small non-zero initial value

        # Self-consistency loop
        max_iterations = 500
        tolerance = 1e-6
        converged = False
        for iteration in range(max_iterations):
            phi_i_old = phi_i.copy()
            new_f_i_n = {}
            for i in G.nodes():
                # Compute Phi_i = sum over neighbors of phi_j
                Phi_i = sum(phi_i[j] for j in G.neighbors(i))
                # Get coordination number z_i
                z_i = len(list(G.neighbors(i)))
                # Construct H_i
                H_i = construct_H_i(U, mu, t, Phi_i, n_max, z_i, phi_i[i])
                # Diagonalize H_i
                eigenvalues, eigenvectors = np.linalg.eigh(H_i)
                # Get ground state (lowest eigenvalue)
                ground_state = eigenvectors[:, 0]
                # Normalize ground_state
                ground_state /= np.linalg.norm(ground_state)
                # Store new f_i_n without updating phi_i yet
                new_f_i_n[i] = ground_state
            # Update f_i_n and phi_i after all sites have been processed
            for i in G.nodes():
                f_i_n[i] = new_f_i_n[i]
                phi_i[i] = compute_phi_i(f_i_n[i])
            # Check convergence
            phi_diff = np.array([abs(phi_i[i] - phi_i_old[i]) for i in G.nodes()])
            max_phi_diff = np.max(phi_diff)
            if max_phi_diff < tolerance:
                converged = True
                break
        # Compute average |phi_i| over all sites
        phi_values = np.array([abs(phi_i[i]) for i in G.nodes()])
        phi_avg = np.mean(phi_values)
        # Store the average |phi_i| value
        phi_avg_values[idx_t, idx_mu] = phi_avg
end_time = time.time()
print(f"Calculation completed in {end_time - start_time:.2f} seconds")

#%%
# Plotting
t_over_U_mesh, mu_mesh = np.meshgrid(t_over_U_values, mu_values, indexing='ij')

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Left panel: Plot the graph
ax = axs[0]
# pos = nx.circular_layout(G)
node_size = 100
nx.draw(G, pos, with_labels=False, node_color='darkblue', edge_color='black', node_size=node_size, ax=ax)
ax.set_title(r"Cycle Graph with L={:d} Nodes | r = {:d} | h = {:d}".format(L, r,h))
ax.axis('equal')

# Right panel: Plot the phi density heatmap
ax = axs[1]
c = ax.contourf(mu_mesh / U, t_over_U_mesh, phi_avg_values.T, levels=100, cmap='gnuplot')
fig.colorbar(c, ax=ax, label=r'Average $|\phi_i|$')
ax.set_ylabel(r'$\mu/U$')
ax.set_xlabel(r'$t/U$')
ax.set_title(f'Average $|\phi_i|$ on a graph G')

plt.tight_layout()
plt.show()
