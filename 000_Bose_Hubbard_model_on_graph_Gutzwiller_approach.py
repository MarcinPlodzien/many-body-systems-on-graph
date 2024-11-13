import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Parameters of the Bose-Hubbard model
t = 2       # Hopping amplitude (modified as per your parameters)
U = 1.0       # On-site interaction strength (modified as per your parameters)
mu = 0.5      # Chemical potential
n_max = 5     # Maximum occupation number per site

# Create a ring graph G with periodic boundary conditions
L = 5  # Number of sites
G = nx.cycle_graph(L)  # A ring graph


r = 6
h = 3


G = nx.balanced_tree(r, h)

L = G.number_of_nodes()
pos = nx.nx_agraph.graphviz_layout(G, prog="twopi", args="")
# pos = nx.circular_layout(G)
# Initialize the local wavefunction coefficients f_i_n and order parameters phi_i
f_i_n = {}  # Dictionary to hold f_i_n arrays for each site
phi_i = {}  # Dictionary to hold phi_i for each site

for i in G.nodes():
    # Initialize f_i_n with uniform coefficients and normalize
    f_i_n[i] = np.ones(n_max+1, dtype=np.complex128) / np.sqrt(n_max+1)
    # Initialize phi_i with the same small value
    phi_i[i] = 1e-2  # Uniform initialization

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

def compute_Phi_i(i, phi_i, G):
    """Compute Phi_i = sum over neighbors of phi_j."""
    Phi = 0.0
    for j in G.neighbors(i):
        Phi += phi_i[j]
    return Phi

def construct_H_i(U, mu, t, Phi_i, n_max):
    """Construct the local Hamiltonian H_i at site i."""
    H = np.zeros((n_max+1, n_max+1), dtype=np.complex128)
    for n in range(n_max+1):
        # Diagonal elements
        H[n, n] = (U/2) * n * (n - 1) - mu * n
        # Off-diagonal elements
        if n < n_max:
            H[n, n+1] = - t * Phi_i * np.sqrt(n+1)
            H[n+1, n] = - t * np.conj(Phi_i) * np.sqrt(n+1)
    return H

# Self-consistency loop
max_iterations = 5000
tolerance = 1e-10
converged = False
for iteration in range(max_iterations):
    phi_i_old = phi_i.copy()
    new_f_i_n = {}
    for i in G.nodes():
        # Compute Phi_i
        Phi_i = compute_Phi_i(i, phi_i, G)
        # Construct H_i
        H_i = construct_H_i(U, mu, t, Phi_i, n_max)
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
        print(f'Converged after {iteration+1} iterations')
        converged = True
        break
if not converged:
    print('Did not converge within the maximum number of iterations')

# Compute the average number of particles at each site
n_i = {}  # Dictionary to hold n_i for each site
for i in G.nodes():
    n_i[i] = compute_n_i(f_i_n[i])

# Output the results
print('Local order parameters phi_i and average particle numbers n_i:')
for i in G.nodes():
    print(f'Site {i}: phi = {phi_i[i].real:.15f}, n = {n_i[i]:.15f}')

# Compute the total number of particles
total_particles = sum(n_i.values())
print(f'Total number of particles: {total_particles:.15f}')

# Plot the local order parameters
phi_values = [abs(phi_i[i]) for i in G.nodes()]
vmin = min(phi_values)
vmax = max(phi_values)
plt.figure()

node_size = 100
nx.draw(G, pos, with_labels=False, node_color=phi_values, cmap='viridis', node_size = node_size)
# v_min = min(phi_values)
# v_max = max(phi_values)

v_min = 0
v_max = 1
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=v_min, vmax=v_max))
sm.set_array([])
plt.colorbar(sm, label=r'|$\phi_i$|')
title_string = "Local Superfluid Order Parameters $|\phi_i|$ \n"
title_string = title_string + r"t = {:2.2f} | U = {:2.2f} | t/U = {:2.2f} | $\mu = {:2.2f}$".format(t, U, t/U, mu)
plt.title(title_string)
plt.show()

