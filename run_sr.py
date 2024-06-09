#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__docformat__ = 'reStructuredText'

###################################
#%% Imports
###################################

import dr as dr
import plotting as myplt
import error_estimation as err

import numpy as np
import pickle
import matplotlib.pyplot as plt
plt.style.use('mystyle.mplstyle')

###################################
#%% Functions
###################################

###############################################################################
def compute_sr(dr, vals, key, mode, max_iter=10, estimator=None, precond = False, full_grad=True,\
               prior_std = None, prior_samples = None, G = None, gradG = None, V = None, Xbar = None,\
               load = True, filename = ''):
###############################################################################
    if filename == '':
        filename = f'{dr.bayes.cache_folder}/M{dr.M}maxiter{max_iter}precond{precond}_sr{key}_{mode}'   
    res = {}
    
    # Load existing errors ####################
    if load:
        try:
            with open(f'{filename}', 'rb') as f:
                res = pickle.load(f)
            if set(vals.keys()).issubset(res.keys()):
                vals = {}
                print('Loaded sr')
            else:
                for idx in res.keys():
                    vals.pop(idx, None)
                print(f'Incomplete sr, proceed to computation')
        except FileNotFoundError:
            print('No existing sr, proceed to computation')
        
   # Compute sr errors ##########################
    if estimator is None:
        estimator = err.ErrorEstimator(dr=dr, precond=precond, prior_std=prior_std,\
                                       prior_samples=prior_samples, G=G, gradG=gradG)
    if mode == 'Joint':
        HX = dr.compute_HX(precond)
        HY = dr.compute_HY(precond)
        U, _,_ = np.linalg.svd(HX)
        V, _,_ = np.linalg.svd(HY)
    elif mode == 'PCA':
        V = dr.compute_PCA_out()[0]
        if precond:
            U = np.eye(dr.d_red)
        else:
            U = dr.compute_PCA_in()[0]
    elif mode == 'CCA':
        if precond:
            U, _ , V= np.linalg.svd(np.cov(dr.prior_std.T, dr.G.T)[:dr.d_red,dr.d_red:])
        else:
            U, _ , V= np.linalg.svd(np.cov(dr.prior_samples.T, dr.G.T)[:dr.d,dr.d:])
        V = V.T
        
    for idx, (r,s) in vals.items(): 
        print(f'Computing r = {r}, s = {s} ----------------------------')
        if mode == 'Coupled':
            _,_,iter_U, iter_V = dr.compute_coupled(r=r, s=s, V=V, max_iter=max_iter, precond = precond)
            U_r = iter_U[-1]
            V_s = iter_V[-1]
        else:
            U_r = U[:,:r]
            V_s = V[:,:s]
            
        if key[0] == 'u':
            res[idx] = estimator.compute_upper(U_r,V_s)
        elif key[0] =='e':
            res[idx] = estimator.compute_err(U_r,V_s, Xbar)
        elif key[0] == 'l':
            res[idx] = estimator.compute_lower(U_r,V_s)

    with open(f'{filename}', 'wb') as f:
         pickle.dump(res, f)
    return res

##################################
#%% Main
##################################

if __name__ == '__main__':

    problem = dr.init_dr('conddiff', obs_init = True, M=10000, load=True)


###################################
#%%% Setup sr1D
###################################

    # s_vals = [1,5,10]
    # r_vals = s_vals

    s_vals = np.arange(0,problem.d_red,5)
    s_vals[0] = 1
    r_vals = s_vals
    
    precond =  True # because size d_red not for joint/PCA
    load = True
    max_iter = 4
    vals1D = {f'{r},{s}':(r,s) for r,s in zip(r_vals, s_vals)}

    keys = ['upper', 'err']
    modes = ['Coupled','Joint','PCA','CCA']
    
    results_sr1D_path = f'{problem.results_folder}/M{problem.M}maxiter{max_iter}precond{int(precond)}_sr1D'
    plots_sr1D_path = f'{problem.plots_folder}/M{problem.M}maxiter{max_iter}precond{int(precond)}_sr1D'

###################################
#%%% Run sr 1D
###################################
    res1D = {}
    for mode in modes:
        for key in keys:
            sr = compute_sr(problem, vals1D, key, mode=mode, precond=precond, load=True)
            res1D[f'{mode}_{key}'] = np.array([sr[idx] for idx in vals1D.keys()])
    
    with open(f'{results_sr1D_path}', 'wb') as f: 
        pickle.dump(res1D, f)
        
    
#%%% Plot sr 1D

    with open(f'{results_sr1D_path}', 'rb') as f:
        res1D = pickle.load(f)
    
    norm = np.mean(np.linalg.norm(problem.G, axis=1)**2)
    if precond:
        factor = 1 / norm
    else:
        factor = problem.poincare / norm
      
    width = 0.6*myplt.textwidth
    fig, ax = plt.subplots(figsize=(width,0.3*myplt.textwidth))
    ax.set_xlabel(fr'$r,s$')   
    ax.xaxis.set_major_locator(myplt.ticker.MaxNLocator(integer=True)) 
    ax.set_yscale('log')
    for i, mode in enumerate(modes):
        for key in keys:
            if key[0] == 'e':
                ax.plot(s_vals, res1D[f'{mode}_{key}']/norm, linestyle='--',color=myplt.colours[i], label=fr'{mode} $L^2$-Error')
            elif key[0] == 'u':
                ax.plot(s_vals, res1D[f'{mode}_{key}']*factor, color=myplt.colours[i], label=fr'{mode} Bound')
    ax.legend(loc='center left', bbox_to_anchor=(1.04, 0.5))
    fig.tight_layout()
    fig.savefig(f'{plots_sr1D_path}.pdf')
    fig.savefig(f'{plots_sr1D_path}.eps')
    fig.savefig(f'{plots_sr1D_path}.svg')

