#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__docformat__ = 'reStructuredText'

###################################
#%% Imports
###################################

import dr
import plotting as myplt
import goals_setup as gs
import utils

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('mystyle.mplstyle')
import pickle


    
###########################################
#%% Main
###########################################

if __name__ == '__main__':
    
    M = 10000
    problem = dr.init_dr('conddiff',M=M, load=True, save = True)
    
    r = 10
    s = 10
    precond = True
    load=True
    save = True
    
    results_sa_path = f'{problem.results_folder}/M{problem.M}precond{int(precond)}r{r}s{s}_sa'
    plots_sa_path = f'{problem.plots_folder}/M{problem.M}precond{int(precond)}r{r}s{s}_sa'
    
    #Setup goals
    goals_setter = gs.GoalsSetter(problem.model, r=r, s=s)
    goals = goals_setter.setup_goals(mode='out', precond=precond, Sigm12 = problem.bayes.prior.Sigm12)
    
    res = []
    for goal in goals.values():
        tmp = problem.compute_coupled(r=r, s=s, U = goal.U, V=goal.V, \
                                max_iter=1, precond=precond, load = load, filename = f'{problem.results_folder}/M{M}_goal_{goal.name}_r{r}s{s}')
        res.append(tmp)
        
#%% Compute Bounds

    for i, goal in enumerate(goals.values()):
        HX, HY, iter_U, iter_V = res[i]
        
        normal=np.trace(np.cov((problem.G@goal.V).T))
        D = np.diag(HX)
        mean_gradG = np.dot(np.mean(problem.gradG, axis=0), problem.bayes.prior.Sig12)
        goal_mean_gradG = mean_gradG.T @ goal.V @ goal.V.T @ mean_gradG

        upper_cl = np.empty(problem.d)
        lower_cl = np.empty(problem.d)
        for i in range(problem.d):
            upper_cl[i] = 1 - np.sum(np.delete(np.diag(goal_mean_gradG),i))/normal
            lower_cl[i] = 1 - np.sum(np.delete(D,i))/normal
        
        upper_tot = D/normal
        lower_tot = np.diag(goal_mean_gradG)/normal
        
#%% Load/Compute samples for Sobol estimation
    
    new_prior_samples = np.empty((problem.M,problem.d))
    G_old_i = np.empty((M, problem.m, problem.d)) # X = prior_samples, X_i = new_prior samples
    nb_loaded = 0
    
    if load: 
        nb_loaded_prior, new_prior_samples = utils.load_array(new_prior_samples, M, filename=f'{problem.results_folder}/goal{goal.name}_new_prior_samples.npy', name = 'new_prior_samples')
        nb_loaded_old_i, G_old_i = utils.load_array(G_old_i, M, filename=f'{problem.results_folder}/goal{goal.name}_G_old_i.npy', name = 'G_old_i samples')
        nb_loaded = np.min([M, nb_loaded_prior, nb_loaded_old_i])#, nb_loaded_new_i])
            
    if nb_loaded < M:
        print(f'Generating {M-nb_loaded} G_old_i and G_new_i samples for each input i')
        if isinstance(problem.bayes.prior.Sig, (int,float)):
            C = problem.bayes.prior.Sig * np.eye(problem.d)
        else:
            C = problem.bayes.prior.Sig
        new_prior_samples[nb_loaded:] = np.random.multivariate_normal(problem.bayes.prior.mu,C, problem.M-nb_loaded)
        for i in range(problem.d):
            X = problem.prior_samples[nb_loaded:].copy()
            X[:,i] = new_prior_samples[nb_loaded:,i]
            G_old_i[nb_loaded:,:,i] = problem.bayes.generate_G_samples(X)
        if save:
            np.save(f'{problem.results_folder}/goal{goal.name}_new_prior_samples', new_prior_samples)
            np.save(f'{problem.results_folder}/goal{goal.name}_G_old_i', G_old_i)


#%% Compute Sobol estimation

    S_tot = np.empty(problem.d)            
    VG = problem.G @ goal.V
    for i in range(problem.d):    
        cov_cond_exp = np.cov(VG.T, (G_old_i[:,:,i] @ goal.V).T)[:s, -s:]
        tmp = np.trace(cov_cond_exp)
        S_tot[i] = 1- tmp/normal 
        
    with open(f'{results_sa_path}', 'wb') as f: 
        pickle.dump([upper_tot, S_tot, lower_tot], f)
            
#%% Plot Sobol

    with open(f'{results_sa_path}', 'rb') as f: 
        upper_tot, S_tot, lower_tot = pickle.load(f)

    width = 0.7*myplt.textwidth
    fig, ax = plt.subplots(figsize=(width, 0.24*myplt.textwidth)) 
    ax.set_xlabel(r'$i$')
    ax.set_yscale('log')
    ax.set_ylim([1e-8, 10])
    
    ax.plot(upper_tot, label='Upper Bound')
    ax.plot(S_tot, linestyle='--', label=r'$S^{\textup{tot}}(U_{\{i\}}|V_s)$')
    ax.plot(lower_tot, label='Lower Bound')
    ax.margins(x=0)
    ax.legend(loc='center left', bbox_to_anchor=(1.04,0.5))
    fig.tight_layout()
    
    fig.savefig(f'{plots_sa_path}_Sobol_tot.pdf')
    fig.savefig(f'{plots_sa_path}_Sobol_tot.eps')
    fig.savefig(f'{plots_sa_path}_Sobol_tot.svg')