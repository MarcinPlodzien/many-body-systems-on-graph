#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 2024

@author: Macin Płodzień
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy.linalg as LA

def generate_bosonic_basis_recursive(N, L, current_site = 0, current_state=None, basis=None):
    """
    Recursively generates the Fock basis for bosonic particles.

    Parameters:
    N (int): Number of bosons.
    L (int): Number of lattice sites.
    current_site (int): Current lattice site being filled (for recursion).
    current_state (np.ndarray): Current configuration of particles across sites.
    basis (list): List to store the generated basis vectors.

    Returns:
    np.ndarray: Array of all valid bosonic basis states.
    """
    if basis is None:
        basis = []
    if current_state is None:
        current_state = np.zeros(L, dtype=int)

    # Base case: if we are at the last site, fill it with the remaining particles
    if current_site == L - 1:
        current_state[current_site] = N
        basis.append(current_state.copy())
        return basis

    # Recursive case: distribute particles among the sites
    for particles in range(N + 1):
        current_state[current_site] = particles
        generate_bosonic_basis_recursive(N - particles, L, current_site + 1, current_state, basis)

    return basis

def get_Fock_basis_bosons(N, L):
    """
    Wrapper function to generate the Fock basis for bosonic particles using recursion.

    Parameters:
    N (int): Number of bosons.
    L (int): Number of lattice sites.

    Returns:
    np.ndarray: Array of all valid bosonic basis states.
    """
    print("Number of particles N: ", N)
    print("Lattice size        L: ", L)
    print("Basis generation via recursion...")

    # Initialize recursive basis generation
    basis = generate_bosonic_basis_recursive(N, L)
    basis = np.array(basis)
    basis = np.flipud(basis)
    print("Number of basis states D: ", len(basis))
    print("............ done")
    basis_Fock_idx = {}
    idx = 0
    for v_idx, v_Fock in enumerate(basis):       
        basis_Fock_idx[tuple(v_Fock)] = idx
        idx += 1
    return basis, basis_Fock_idx    
 
def aDag_a(i, j, fock_state):
    """
    Apply the bosonic operator a_i^† a_j on the Fock state.
    
    Parameters:
        i (int): Index where a creation operator will act.
        j (int): Index where an annihilation operator will act.
        fock_state (np.array): Array representing the Fock state.
        
    Returns:
        factor (float): Coefficient resulting from the action of the operator.
        result (np.array): New Fock state after applying the operator.        
    """
    # Copy the state to avoid modifying the input
    new_state = np.array(fock_state, dtype=int)
    
    # Occupations at sites i and j
    n_i = fock_state[i]
    n_j = fock_state[j]
    
    # If there's no particle at site j, the result is zero
    if n_j == 0:
        factor = 0
        return factor, new_state
    
    # Calculate the prefactor from annihilation and creation
    factor = np.sqrt(n_i + 1)*np.sqrt(n_j)
    
    # Apply the operators: decrease n_j by 1, increase n_i by 1
    new_state[j] -= 1
    new_state[i] += 1
    return factor, new_state
 
 
def get_H_kinetic_elements(basis, basis_Fock_idx, J_hopping):
    H_vw = []
    print("Generate kinetic hamiltonian for component 'a' ")
    rows, cols = np.nonzero(J_hopping)
    for v_idx, v_ket in enumerate(basis):
        for site_k, site_l in zip(rows, cols):
            J = J_hopping[site_k, site_l]
            factor, w_ket = aDag_a(site_k, site_l, v_ket)
            w_ket = tuple(w_ket)         
            w_idx = basis_Fock_idx[w_ket]                                    
            H_vw.append([v_idx, w_idx, factor*J])
    return np.array(H_vw)



def get_H_interactions_elements(basis, basis_Fock_idx):
    print("Generate interaction hamiltonian for component 'a' ")
    H_vw = []
    for v_idx, v_ket in enumerate(basis):
        for site_k in range(0, L):            
            n_site_k = v_ket[site_k]
            factor = n_site_k*(n_site_k-1)
            H_vw.append([v_idx, v_idx, factor])
    return np.array(H_vw)

def get_matrix_representations(data, rows, cols, D):
    return coo_matrix((data, (rows, cols)), shape = (D, D))
#%%
r = 6
h = 2


G = nx.balanced_tree(r, h)
# G = nx.path_graph(h)

J_hopping = nx.adjacency_matrix(G).toarray()
L = G.number_of_nodes()
print("Number of nodes: {:d}".format(L))
pos = nx.nx_agraph.graphviz_layout(G, prog="twopi", args="")
# pos = nx.circular_layout(G)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
node_size = 200
nx.draw(G, pos, node_size=node_size, alpha=1, node_color="blue", edge_color="black", with_labels=False)

ax.set_aspect('equal')
ax.set_title(r"$L = {:d}$ nodes".format(L))
plt.show()

#%%

N_a = 2


basis_a, basis_a_Fock_idx = get_Fock_basis_bosons(N_a, L)
D_a = basis_a.shape[0]
print("D_a = {:d} ".format(D_a))
 

H_a_kin_elements =  get_H_kinetic_elements(basis_a, basis_a_Fock_idx, J_hopping)

H_ = H_a_kin_elements
data = H_[:,2]
rows = H_[:, 0]
cols = H_[:, 1]
H_a_kin = get_matrix_representations(data, rows, cols, D_a)

 
H_a_int_elements = get_H_interactions_elements(basis_a, basis_a_Fock_idx)
H_ = H_a_int_elements
data = H_[:,2]
rows = H_[:, 0]
cols = H_[:, 1]
H_a_int = get_matrix_representations(data, rows, cols, D_a)

 
#%%

data_all = []

U_a_vec = np.linspace(0, 20, 50)
for U_a in U_a_vec:
    
    H_full = -H_a_kin + U_a/2.*H_a_int
    eigenvalues, eigenvectors = eigsh(H_full, k=3, which = 'SA')
    E_GS = eigenvalues[0]
  
    
    energy_gap =  eigenvalues[1] - eigenvalues[0]
    psi_GS = eigenvectors[:,0]
    rho_OBDM = np.zeros((L, L))                 # one-body densitym atrix rho = <GS| aDag_k a_l |GS>
    for site_k in range(0, L):
        for site_l in range(0, L):
            for v_ket_idx, v_ket in enumerate(basis_a):
                factor, w_ket = aDag_a(site_k, site_l, v_ket)
                w_ket = tuple(w_ket)
                if(w_ket in basis_a_Fock_idx):
                    w_ket_idx = basis_a_Fock_idx[tuple(w_ket)]
                    rho_OBDM[site_k, site_l] += factor*np.conj(psi_GS[w_ket_idx])*psi_GS[v_ket_idx]
    rho_OBDM = rho_OBDM/np.trace(rho_OBDM)
    evals_rho_OBDM, evecs_rho_OBDM = LA.eigh(rho_OBDM)


    n_k = np.zeros((L,))
    n_k_n_k = np.zeros((L,))
    for site_k in range(0, L):        
        for v_ket_idx, v_ket in enumerate(basis_a):
            factor = v_ket[site_k]
            n_k[site_k] += factor*np.abs(psi_GS[v_ket_idx])**2
            n_k_n_k[site_k] += factor**2*np.abs(psi_GS[v_ket_idx])**2

    n_k_variance = n_k_n_k - n_k**2  
    
    mean_n_k_std= np.mean(np.sqrt(n_k_variance))
    f_SF = np.max(evals_rho_OBDM)
    print(U_a)
    dict_tmp = {
                "L"                 : L,
                "N_a"               : N_a,
                "D_a"               : D_a,
                "U_a"               : U_a,
                "E_GS"              : E_GS,
                "energy_gap"        : energy_gap,
                "f_SF"              : f_SF, # superfluid fraction
                "mean_n_k_std"      : mean_n_k_std,
                "n_k_variance"      : n_k_variance,
                "psi_GS"            : psi_GS,
                "rho_single_body"   : rho_OBDM,
                "J_couplings"       : J_hopping,
                }
    data_all.append(dict_tmp)

df = pd.DataFrame(data_all)

#%%
FontSize = 17
fig, ax = plt.subplots(2, 2, figsize = (14, 8) )
title_string = r"$\hat{H} = -\sum_{kl}J_{kl}\hat{a}^\dagger_k\hat{a}_l + h.c. + \frac{U}{2}\sum_k\hat{n}_k$"
plt.suptitle(title_string, fontsize = FontSize)

print("r = {:d}, h = {:d} | Number of nodes: {:d}".format(r, h, L))
node_size = 200
nx.draw(G, pos, node_size=node_size, alpha=1, node_color="blue", edge_color="black", with_labels=False, ax = ax[0,0])

ax[0,0].set_aspect('equal')
ax[0,0].set_title(r"$r = {:d}, h = {:d} | L = {:d}$ nodes, N = {:d} bosons".format(r, h, L, N_a), fontsize = FontSize)

ax[0,1].plot(df["U_a"], df["energy_gap"],'-o')
ax[0,1].set_ylabel("energy gap", fontsize = FontSize)
ax[0,1].set_xlabel(r"$U$", fontsize = FontSize)

ax[1,0].plot(df["U_a"], df["mean_n_k_std"],'-o')
ax[1,0].set_ylabel(r"mean  $\langle\Delta \hat{n}_k\rangle$", fontsize = FontSize)
ax[1,0].set_xlabel(r"$U$", fontsize = FontSize)


ax[1,1].plot(df["U_a"], df["f_SF"],'-o')
ax[1,1].set_ylabel("condensate fraction", fontsize = FontSize)
ax[1,1].set_xlabel(r"$U/J$", fontsize = FontSize)


for i_j in [[0,1], [1,0], [1,1]]:
    i,j = i_j[0], i_j[1]
    ax[i,j].tick_params(axis='both', which='major', labelsize = FontSize)
    ax[i,j].tick_params(axis='both', which='minor', labelsize = FontSize)
    ax[i,j].tick_params(axis='both', which='major', labelsize = FontSize)
    ax[i,j].tick_params(axis='both', which='minor', labelsize = FontSize)

filename_params = "_r." + str(r) + "_h." + str(h)
filename = "fig" + filename_params + ".png"
plt.savefig(filename, dpi = 400, format = "png")
plt.show()