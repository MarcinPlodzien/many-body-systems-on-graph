#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 2024

@author: Macin Płodzień
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import kron
from scipy.sparse.linalg import eigsh
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

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


def generate_fermionic_basis_recursive(N, L, current_position=0, current_state=None, basis=None):
    """
    Recursively generates the Fock basis for fermionic particles starting from [1, 1, ..., 0, 0].

    Parameters:
    N (int): Number of fermions.
    L (int): Number of lattice sites.
    current_position (int): Current lattice position to fill (for recursion).
    current_state (np.ndarray): Current configuration of particles across sites.
    basis (list): List to store the generated basis vectors.

    Returns:
    list: List of all valid fermionic basis states.
    """
    if basis is None:
        basis = []
    if current_state is None:
        # Start with the initial configuration: [1, 1, ..., 0, 0] with N ones at the start
        current_state = np.array([1] * N + [0] * (L - N), dtype=int)

    # Base case: if we are at the end, add the configuration to the basis
    if current_position >= L:
        basis.append(current_state.copy())
        return basis

    # Recursive case: generate all unique permutations of N ones across L sites
    for i in range(current_position, L):
        # If there's a 1 at the current position, try moving it to the next positions
        if current_state[i] == 1:
            # Find the next position to move the particle
            for j in range(i + 1, L):
                if current_state[j] == 0:
                    # Move the particle from position i to j
                    current_state[i], current_state[j] = 0, 1
                    # Recur with the updated state and current position
                    generate_fermionic_basis_recursive(N, L, i + 1, current_state, basis)
                    # Backtrack to restore the state
                    current_state[i], current_state[j] = 1, 0

    # If this configuration hasn't been added yet, add it now
    basis.append(current_state.copy())
    return basis

def get_Fock_basis_fermions(N, L):
    """
    Wrapper function to generate the Fock basis for fermionic particles using recursion, 
    starting from the configuration [1, 1, ..., 0, 0].

    Parameters:
    N (int): Number of fermions.
    L (int): Number of lattice sites.

    Returns:
    np.ndarray: Array of all valid fermionic basis states.
    """
    print("Number of particles N: ", N)
    print("Lattice size        L: ", L)
 
    # Initialize recursive basis generation
    basis = generate_fermionic_basis_recursive(N, L)
    basis = np.unique(np.array(basis), axis=0)  # Remove any duplicate configurations
    basis = np.flipud(basis)
    print("Number of basis states D: ", len(basis))
    print("............ done")
    
    basis_Fock_idx = {}
    idx = 0
    for v_idx, v_Fock in enumerate(basis):       
        basis_Fock_idx[tuple(v_Fock)] = idx
        idx += 1
    
    return basis, basis_Fock_idx


def a_op(site_idx, ket, statistic):      
    ket_tmp = np.copy(ket)
    # print(ket_tmp)
    if(statistic=="boson"):        
        if(ket_tmp[site_idx] == 0):
            factor = 0
        else:
            factor = np.sqrt(ket_tmp[site_idx])
            ket_tmp[site_idx] = ket_tmp[site_idx]-1
        return factor, ket_tmp

    if(statistic=="fermion"):
        # We use the following convention: |v> = |11010> = aDag_0 aDag_1 aDag_3 |0>
        # where sites are enumerated from left to right starting with site_idx = 0
        if(ket_tmp[site_idx] == 0):
            factor = 0
        else:
            factor = (-1.0)**np.sum(ket[:site_idx])
            ket_tmp[site_idx] = ket_tmp[site_idx]-1
        return factor, ket_tmp


def aDag_op(site_idx, ket, statistic):      
    ket_tmp = np.copy(ket)
    
    if(statistic == "boson"):        
        factor = np.sqrt(ket_tmp[site_idx]+1)     
        ket_tmp[site_idx] = ket_tmp[site_idx]+1
        return factor, ket_tmp

    if(statistic == "fermion"):
        # We use the following convention: |v> = |11010> = aDag_0 aDag_1 aDag_3 |0>,
        # where sites are enumerated from left to right starting with site_idx = 0
        if(ket_tmp[site_idx] == 1):
            factor = 0
        else:
            factor = (-1.0)**np.sum(ket[0:site_idx]) 
            ket_tmp[site_idx] = ket_tmp[site_idx]+1
        return factor, ket_tmp
    
 

def get_H_kinetic_term(basis, basis_Fock_idx, J_couplings, statistic):
    print("Generate kinetic hamiltonian for component 'a' ")
    H_vw = []
    for v_idx, v_ket in enumerate(basis):
        for site_k in range(0, L):
            for site_l in range(0, L):
                [factor_1, w_ket] = a_op(site_l, v_ket, statistic)      # act on "a" component 
                [factor_2, w_ket] = aDag_op(site_k, w_ket, statistic)   # act on "a" component                   
                factor = factor_1*factor_2*J_couplings[site_k, site_l]
                w_ket = tuple(w_ket)
                if((w_ket in basis_Fock_idx) and factor!=0):                    
                    w_idx = basis_Fock_idx[w_ket]                                    
                    H_vw.append((v_idx, w_idx, factor))                    
    H_vw =  np.array(H_vw)    
    return H_vw

def get_H_interactions_term(basis, basis_Fock_idx, U_bare_interactions, statistic):
    print("Generate interaction hamiltonian for component 'a' ")
    H_vw = []
    for v_idx, v_ket in enumerate(basis):
        for site_k in range(0, L):            
            [factor_1, w_ket] = a_op(site_k, v_ket, statistic)      # act on "a" component 
            [factor_2, w_ket] = aDag_op(site_k, w_ket, statistic)   # act on "a" component   
            w_ket = tuple(w_ket)                
            if(w_ket in basis_Fock_idx):                    
                w_idx = basis_Fock_idx[w_ket]    
                n_site_k = factor_1*factor_2 # number of particles on site_k
                factor = U_bare_interactions[site_k]*n_site_k*(n_site_k-1)
                H_vw.append((v_idx, w_idx, factor))                    
    H_vw =  np.array(H_vw)    
    return H_vw



def get_H_ab_interactions_term(basis_ab, basis_ab_Fock_idx, U_ab_bare_interactions, statistic_a, statistic_b):
    D = len(basis_ab)
    H_vw = []
    print(D)
    for v_ab_idx, v_ab_ket in enumerate(basis_ab):
        #split combined basis Fock vector into "a" and "b" component
        v_a_ket = v_ab_ket[:L]
        v_b_ket = v_ab_ket[L:]
        for site_k in range(0, L):             
            [factor_a_1, w_a_ket] = a_op(site_k, v_a_ket, statistic_a)      # act on "a" component 
            [factor_a_2, w_a_ket] = aDag_op(site_k, w_a_ket, statistic_a)   # act on "a" component  
            
            [factor_b_1, w_b_ket] = a_op(site_k, v_b_ket, statistic_b)      # act on "b" component 
            [factor_b_2, w_b_ket] = aDag_op(site_k, w_b_ket, statistic_b)   # act on "b" component                   
            w_ab_ket = tuple(np.concatenate((w_a_ket, w_b_ket)))
            if(w_ab_ket in basis_ab_Fock_idx):                    
                w_ab_idx = basis_ab_Fock_idx[w_ab_ket]  
                n_a_site_k = factor_a_1*factor_a_2
                n_b_site_k = factor_b_1*factor_b_2
                factor = U_ab_bare_interactions[site_k]*n_a_site_k*n_b_site_k
                H_vw.append((v_ab_idx, w_ab_idx, factor))                    
    H_vw =  np.array(H_vw)    
    return H_vw

def get_matrix_representations(data, rows, cols, D):
    return coo_matrix((data, (rows, cols)), shape = (D, D))

#%%
r = 5
h = 2

 
G = nx.balanced_tree(r, h)
pos = nx.circular_layout(G)

L = G.number_of_nodes()
print("Number of nodes: {:d}".format(L))
pos = nx.nx_agraph.graphviz_layout(G, prog="twopi", args="")
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
node_size = 200
nx.draw(G, pos, node_size=node_size, alpha=1, node_color="blue", edge_color="black", with_labels=False)

ax.set_aspect('equal')
ax.set_title(r"$L = {:d}$ nodes".format(L))
plt.show()



#%%
N_a = 2
statistic_a = "boson"
 

if(statistic_a == "boson"):
    basis_a, basis_a_Fock_idx = get_Fock_basis_bosons(N_a, L)
else:
    basis_a, basis_a_Fock_idx = get_Fock_basis_fermions(N_a, L)
 
D_a = basis_a.shape[0]
 
print("D_a = {:d} ".format(D_a))

J_couplings = nx.adjacency_matrix(G).toarray()
U_a_bare = np.ones((L,))
 
H_kinetic_a_elements = get_H_kinetic_term(basis_a, basis_a_Fock_idx, J_couplings, statistic_a)
H_interactions_a_elements = get_H_interactions_term(basis_a, basis_a_Fock_idx, U_a_bare, statistic_a)

#%% Get matrix representations
H_ = H_kinetic_a_elements
D = D_a
rows = H_[:,0]
cols = H_[:,1]
data = H_[:,2]
H_a_kin = coo_matrix((data, (rows, cols)), shape = (D, D))

H_ = H_interactions_a_elements
D = D_a
rows = H_[:,0]
cols = H_[:,1]
data = H_[:,2]
H_a_int = coo_matrix((data, (rows, cols)), shape = (D, D))

 
#%%

data_all = []

U_a_vec = np.linspace(0, 10, 11)
for U_a in U_a_vec:
    H_full = -H_a_kin + U_a/2.*H_a_int
    eigenvalues, eigenvectors = eigsh(H_full, k=2)
    E_GS = eigenvalues[0]
    psi_GS = eigenvectors[:,0]
    print(U_a)
    dict_tmp = {
                "L"             : L,
                "N_a"           : N_a,
                "statistic_a"   : statistic_a,
                "D_a"           : D_a,
                "U_a"           : U_a,
                "J_couplings"   : J_couplings,
                "E_GS"          : E_GS,
                "psi_GS"        : psi_GS                                        
                }
    data_all.append(dict_tmp)
#%%
df = pd.DataFrame(data_all)