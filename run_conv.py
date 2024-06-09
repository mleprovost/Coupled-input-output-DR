#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__docformat__ = 'reStructuredText'

###################################
#%% Imports
###################################

import dr 
import plotting as myplt
import error_estimation as err

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('mystyle.mplstyle')
import pickle
import os

###################################
#%% Functions
###################################

###############################################################################
def compute_conv(iter_U, iter_V, key, estimator = None, dr=None, precond=False, full_grad=True,\
                 prior_std = None, prior_samples = None, G = None, gradG = None, Xbar = None):
###############################################################################
    """ Computes L2/Upper/Lower Error over Alternating SVD iterations
    
    Parameters
    ----------
    key: str
        Which error estimator is used ('u'pper, 'e'rror, 'l'ower)
    estimator: err.ErrorEstimator
        If estimator not given, initialise with dr,precond,prior_std,prior_samples
        G and gradG from kwargs
    Xbar:
        Provide for testing
        Samples for pick-freeze in L2-error

    Returns
    -------
    conv: array(maxiter)
        Error over AlternEIG iterations

    """
    r = np.shape(iter_U[0])[1]
    s = np.shape(iter_V[0])[1]
    max_iter = len(iter_U)
    conv = np.empty(max_iter)
    
    if estimator is None:
        estimator = err.ErrorEstimator(dr=dr, precond=precond, prior_std=prior_std,\
                                       prior_samples=prior_samples, G=G, gradG=gradG)

    for i in range(max_iter):
        U = iter_U[i]
        V = iter_V[i]
        
        if key[0]=='u':
            res = estimator.compute_upper(U,V)
        elif key[0]=='e':
            res  = estimator.compute_err(U,V, Xbar)
        elif key[0]=='l':
            res = estimator.compute_lower(U,V)
        conv[i] = res
            
    return conv

###############################################################################
def compute_conv_randinit(dr, r, s, max_iter, key, nb = 10, precond=False,\
                          Vs = None, Xbar = None, load = True, filename = ''):
###############################################################################
    if filename == '':
        filename = f'{dr.bayes.cache_folder}/M{dr.M}r{r}s{s}maxiter{max_iter}precond{int(precond)}fullgrad{int(dr.full_grad)}_conv{key}_randinit.npy'    
    conv = np.empty((nb, max_iter+1)) #iter_U has len maxiter+1 due to init
    nb_computed = 0
        
    # Load existing conv #########################
    if load:
        if os.path.isfile(filename):
            computed_conv = np.load(filename)
            nb_computed = min(np.shape(computed_conv)[0], nb)
            conv[:nb_computed] = computed_conv[:nb_computed]
            if nb_computed < nb:
                print(f'Incomplete conv diagnostics, proceed to computation')
            else: 
                print('Loaded conv')
        else:
            print('No existing conv, proceed to computation')
        
    # Compute conv ###############################
    estimator = err.ErrorEstimator(dr=dr, precond=precond)
    for i in range(nb_computed,nb):
        if Vs is None:
            V = np.random.randn(dr.m,s) 
            V,_,_ = np.linalg.svd(V, full_matrices=False)
        else:
            V = Vs[i]
        _,_, iter_U, iter_V = dr.compute_coupled(r=r, s=s, V=V,\
                                                 max_iter=max_iter, precond=precond, load = False) 
        conv[i] = compute_conv(iter_U, iter_V, key, estimator, Xbar = Xbar)
    np.save(filename, conv)
            
    return conv

###############################################################################
def compute_conv_keys(dr, keys, mode, nb=10, precond=False,\
                          load = True, filename = '', **kwargs):
###############################################################################
    conv = {}
    if mode == 'randinit':
        Vs = np.empty((nb, dr.m, kwargs['s']))
        for i in range(nb):
            V = np.random.randn(dr.m,kwargs['s']) 
            Vs[i],_,_ = np.linalg.svd(V, full_matrices=False)
            
    for key in keys:
        if mode == 'randinit':
            conv[key] = compute_conv_randinit(dr=dr, r=kwargs['r'], s=kwargs['s'], max_iter=kwargs['max_iter'],\
                                              key=key, nb=nb, precond=precond, Vs = Vs, load=load, filename=filename)
        elif mode == 'diagnostics':
            conv[key] = compute_conv_diagnostics(dr=dr, iter_U=kwargs['iter_U'], iter_V=kwargs['iter_V'],\
                                                 key=key, nb = nb, precond=precond, load=load, filename=filename)
            
    return conv


###############################################################################
def compute_stats(conv, G, precond=True):    
###############################################################################
    norm = np.mean(np.linalg.norm(G, axis=1)**2)
    if precond:
        const_upper = 1
        const_lower = 1
    else: 
        const_upper = problem.poincare
        const_lower = problem.cramerrao
        
    if isinstance(conv, np.ndarray):
        if len(np.shape(conv))==1:
            conv = conv/norm
            mean = conv.copy()
            std = np.zeros(len(conv))
        else:
            conv = conv/norm
            mean = np.mean(conv, axis=0)
            std = np.std(conv, axis=0)
    elif isinstance(conv, dict):
        mean = {}
        std  = {}
        for key, val in conv.items():
            if len(np.shape(val))==1:
                conv[key] = val/norm
                mean[key]= conv[key].copy()
                std[key] = np.zeros(len(val))
            else:
                if key[0] == 'u':
                    conv[key] = const_upper * conv[key]/norm
                elif key[0] == 'e':
                    conv[key] = conv[key]/norm
                elif key[0] == 'l':
                    conv[key] = const_lower * conv[key]/norm  
                mean[key] = np.mean(conv[key], axis=0)
                std[key] = np.std(conv[key], axis=0)
    return conv, mean, std


###############################################################################
def plot_conv(mean, labels, conv=None, std=None, filename = 'convergence', log_scale = False, **kwargs):      
###############################################################################
    width = kwargs.get('width',0.55*myplt.textwidth)
    fig, ax = plt.subplots(figsize=(width, 0.25*myplt.textwidth)) 
    ax.set_xlabel(r'Iteration')
    ax.xaxis.set_major_locator(myplt.ticker.MaxNLocator(integer=True))
    if log_scale:
        ax.set_yscale('log')
        
    for i, [key,label] in enumerate(labels.items()): 
        if std is None or np.all(std[key]==0):
            if key[0] == 'e':
                ax.plot(mean[key],linestyle='--', label = label)
            else:
                ax.plot(mean[key], label = label)
        else:
            if key[0] == 'e':
                ax.errorbar(np.arange(len(mean[key])), mean[key], yerr=std[key],linestyle='--', label = label)
                ax.plot(conv[key].T, color = myplt.colours[i],linestyle='--', alpha=0.2)
            else:
                ax.errorbar(np.arange(len(mean[key])), mean[key], yerr=std[key], label = label)
                ax.plot(conv[key].T, color = myplt.colours[i], alpha=0.2)
    ax.legend(loc='center left', bbox_to_anchor=(1.04,0.5))
    fig.tight_layout()
    
    fig.savefig(f'{filename}.pdf')
    fig.savefig(f'{filename}.eps')
    fig.savefig(f'{filename}.svg')

###################################
#%% Main
###################################

if __name__ == "__main__":

    T = 0.1
    dt = 0.001
    problem = dr.init_dr('conddiff', obs_init = False, M=10000, load=True, save = True)
    

#%%% Setup Variables

    r = 10#<=n
    s = 10#<=m
    precond = True
    max_iter = 10
    nb_diagnostics = 5
    nb_randinit = 10
    load_coupled = False
    
    estimators = ['error', 'upper', 'lower'] 
    labels = {'upper':'Upper Bound','error': r'$L^2$-error', 'lower' : 'Lower Bound'}
    
    results_coupled_path = f'{problem.results_folder}/r{r}s{s}M{problem.M}maxiter{max_iter}precond{int(precond)}fullgrad{int(problem.full_grad)}_coupled'
    randinit_conv_path = f'{problem.results_folder}/r{r}s{s}M{problem.M}maxiter{max_iter}precond{int(precond)}fullgrad{int(problem.full_grad)}randinit{nb_randinit}_conv'
    plots_conv_coupled_path = f'{problem.plots_folder}/r{r}s{s}M{problem.M}maxiter{max_iter}precond{int(precond)}fullgrad{int(problem.full_grad)}_conv'

    HX,HY, iter_U, iter_V = problem.compute_coupled(r=r, s=s, max_iter=max_iter, precond=precond,\
                                                load = load_coupled, filename = results_coupled_path)      

#%%% Run randinit
    conv = compute_conv_keys(problem, keys = labels.keys(), mode = 'randinit', nb=1, r=r, s=s, max_iter = max_iter, precond=precond, load=False)
    conv, mean, std = compute_stats(conv, G=problem.G, precond=precond)
    
    with open(f'{randinit_conv_path}', 'wb') as f: 
        pickle.dump([conv, mean, std],f)
   
#%%% Plot conv alternating SVD
    with open(f'{randinit_conv_path}', 'rb') as f: 
    # with open(f'{diagnostics_conv_path}', 'rb') as f: 
        conv, mean, std = pickle.load(f)
    
    plot_conv(mean = mean, conv = conv, std=std, labels=labels, filename = plots_conv_coupled_path, log_scale = True)

