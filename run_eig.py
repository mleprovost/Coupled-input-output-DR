#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__docformat__ = 'reStructuredText'

###################################
#%% Imports
###################################

import dr
import plotting as myplt
import goals_setup as gs
import sensor_placement as sp
import eig_estimation as eg
import utils


import numpy as np
from scipy.stats import multivariate_normal
import math #for calculation number of combinations
import matplotlib.pyplot as plt
plt.style.use('mystyle.mplstyle')
from dataclasses import dataclass
import itertools
import copy
import pickle



###################################
#%% Functions
###################################

###############################################################################
def setup_random_sensors(m, nb_sensors, nb_random):
###############################################################################
    """ Array or random sensor locations

    Returns
    -------
    sensors : array(nb_random, nb_sensors)

    """
    sensors = np.empty((nb_random, nb_sensors), dtype=int)
    for i in range(nb_random):
        sensors[i] = np.random.choice(m, nb_sensors, replace=False)
    return sensors


def plot_eig(eigs, xticklabels=None, fig=None, ax=None, positions = None, markers=None,colours=None, legend = None, filename=''):

    if ax is None:
        width = 0.7*myplt.textwidth
        fig, ax = plt.subplots(figsize=(width, 0.24*myplt.textwidth))
    if markers is None:
        markers = myplt.markers[0]
    if colours is None:
        colours = myplt.colours[0]
    if positions is None:
        positions = np.arange(len(eigs)) + 0.5
    if xticklabels is not None:
        ax.set_xticks(positions)
        ax.set_xticklabels(xticklabels)
    # else:
    #     ax.set_xticks([])
    
    if isinstance(eigs, (np.ndarray)):
        ims = ax.violinplot(eigs.T, positions = positions, showmeans =True) 
    else:
        ims = ax.scatter(positions, eigs.values(), color=colours, marker = markers,s=20, alpha=0.7, zorder=100)
    # fig.tight_layout()
    if filename != '':
        fig.savefig(filename)
    return fig, ax, ims


###################################
#%% Main
###################################

if __name__ == "__main__":

    M = 1000
    T = 0.1
    dt = 0.001
    problem = dr.init_dr('burgers', obs_init = False, M=M,T=T,dt=0.001,ks=1,kt=int(T/0.001), load=True, save = True)

#%% Setup EIG

    r = 5
    s = 10
    nb_sensors = s
    EIM_fct = 'EIM_ordered'
    methods_sensor = ['maxdiag', EIM_fct]

    precond=False
    load_goals=True
    load_sensors=True
    load_eig = True
    
    PCA = True
    optimal = False
    random = 100
    
    estimator= 'DLIS' 
    if problem.model.name[:7]=='advdiff':
        estimator = 'Gauss'
    nb_inner = 30
    estimator_name = estimator
    if estimator in ['NMC','DLIS']:
        estimator_name += f'{nb_inner}'
    
    results_eig_path = f'{problem.results_folder}/M{problem.M}precond{int(precond)}r{r}s{s}{estimator_name}_eig'
    plots_eig_path = f'{problem.plots_folder}/M{problem.M}precond{int(precond)}r{r}s{s}{estimator_name}_eig'


#%% Run sensor-placement

    #Setup goals
    goals_setter = gs.GoalsSetter(problem.model, r=r, s=s)
    goals = goals_setter.setup_goals(mode='in', precond=precond, Sigm12 = problem.bayes.prior.Sigm12)

#%%
    #Compute goals
    results = {}
    for goal in goals.values():
        results[goal.name] = problem.compute_coupled(r=r, s=nb_sensors, U = goal.U, \
                                max_iter=1, precond=precond, load = load_goals, filename = f'{results_eig_path}_goal{goal.name}_r{r}s{nb_sensors}')
    if PCA: 
        res_PCA = problem.compute_PCA_out()
        

    #Compute sensors
    sensors = {}
    for goal in goals.values():
        sensors[goal.name] = sp.compute_sensor(dr=problem,res=results[goal.name], nb_sensors=nb_sensors, methods=methods_sensor,precond=precond, load=load_sensors)
    if PCA: 
        sensors_PCA = sp.compute_sensor(res=res_PCA, nb_sensors=nb_sensors, methods = [EIM_fct])

    #Setup random sensors
    sensors_random = setup_random_sensors(problem.m, nb_sensors=s, nb_random=random)
    
    with open(f'{results_eig_path}', 'wb') as f: 
        pickle.dump([results, sensors, sensors_PCA, sensors_random], f)

#%%% Illustrate goals for burgers

    width = 0.35*myplt.textwidth
    fig, ax = plt.subplots(figsize = (width, 0.7*0.4*myplt.textwidth))
    ax.plot(problem.bayes.prior.mu, color = myplt.colours[0])
    ax.set_xlabel('Space')
    
    u0_pert = np.load('advdiff_seeds.npy')
    ax.plot(u0_pert.T, color = myplt.colours[0], alpha = 0.2)   
    
    xlim = ax.get_xlim() 
    ylim = ax.get_ylim() 
    for goal in goals.values():
        x,_,width,_ = goal.marker
        ax.add_patch(myplt.patches.Rectangle((x,-3), width=width,height=10,\
                    color=myplt.beamer_colours['UGAOrange'], alpha=0.3, fill = True, zorder=100, clip_on=True))
    fig.tight_layout()
    fig.savefig(f'plots/{problem.model.name}/goals.pdf')
    fig.savefig(f'plots/{problem.model.name}/goals.svg')

#%%% Plot sensors

    with open(f'{results_eig_path}', 'rb') as f: 
        results, sensors, sensors_PCA, sensors_random = pickle.load(f)
        
    markers = [myplt.markers[1],myplt.markers[0],myplt.markers[2]]
    colours = [myplt.colours[2],myplt.colours[1],myplt.colours[4]]

    for goal in goals.values():
        taus = [sensors[goal.name].maxdiag, sensors[goal.name].EIM, sensors_PCA.EIM]
        V = results[goal.name][3][-1]
        # For without legend
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(myplt.textwidth*0.4, myplt.textwidth*0.25), sharex = True,gridspec_kw={'height_ratios': [2.5,1.5]})
        myplt.plot_V(problem.model, V, ax=ax[0], goal=goal, xlabel=False, legend=False)
        myplt.plot_sensors(problem.model, taus=taus,ax=ax[1],\
                            markers= markers,alpha=0.7, colours=colours, filename=f'{plots_eig_path}_{goal.name}')
        # For with legend:
        # fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(myplt.textwidth*0.4, myplt.textwidth*0.26), sharex = True,gridspec_kw={'height_ratios': [2.5,1.5]})
        # myplt.plot_V(problem.model, V, ax=ax[0], goal=goal, xlabel=False, legend=False)
        # myplt.plot_sensors(problem.model, taus=taus,ax=ax[1], sensor_labels=[r'$\tau^*$', r'$\widetilde\tau$', r'$\widetilde\tau^{\textup{PCA}}$'],\
        #                     markers= markers,alpha=0.7, colours=colours, filename=f'{plots_eig_path}_{goal.name}')
       
       
        
#%% Compute EIG
    
    eigs_maxdiag = {}
    eigs_EIM = {}
    eigs_PCA = {}
    eigs_random = np.zeros((len(goals), random))
    
    for i, (key, goal) in enumerate(goals.items()):
        eig_estimator = eg.EIGEstimator(goal=goal, precond=precond, M=M, estimator=estimator, bayes=problem.bayes, nb_inner=nb_inner, load=load_eig, filepath=problem.results_folder,\
                                         prior_samples=problem.prior_samples, G_samples=problem.G, gradG_samples=problem.gradG)
                                         
            
        eigs_maxdiag[goal.name] = eig_estimator.compute_EIG(tau=sensors[key].maxdiag)
        eigs_EIM[goal.name] = eig_estimator.compute_EIG(tau=sensors[key].EIM)
        if PCA:
            eigs_PCA[goal.name] = eig_estimator.compute_EIG(tau=sensors_PCA.EIM)
        if random:
            for j, tau in enumerate(sensors_random):
                eigs_random[i,j] = eig_estimator.compute_EIG(tau=tau)
        
    with open(f'{results_eig_path}_{estimator}', 'wb') as f: 
        pickle.dump([eigs_maxdiag, eigs_EIM, eigs_PCA, eigs_random], f)
        
#%% Plot EIG 

    with open(f'{results_eig_path}_{estimator}', 'rb') as f: 
        eigs_maxdiag, eigs_EIM, eigs_PCA, eigs_random = pickle.load(f)
        
    positions = np.arange(len(goals))
    xticklabels = [f'{i*(r+5)+5}' for i in positions]
    fig, ax,_ = plot_eig(eigs_random, positions=positions, xticklabels = xticklabels, legend=None)
    fig, ax, im1 = plot_eig(eigs_maxdiag, fig=fig, ax=ax, positions=positions, markers=myplt.markers[1], colours=myplt.colours[2],legend=None)
    fig, ax, im2 = plot_eig(eigs_EIM, fig=fig, ax=ax, positions=positions, markers=myplt.markers[0], colours=myplt.colours[1],legend=None)
    ims = [im1,im2]
    if PCA:
        fig, ax, im_PCA = plot_eig(eigs_PCA, fig=fig, ax=ax, positions=positions, markers=myplt.markers[2],colours=[myplt.colours[4]],legend=None)
        ims.append(im_PCA)
    ax.get_xaxis().set_visible(True)
    ax.set_xlabel(r'$i$')    
    ax.legend(handles=ims, labels = [r'$\Phi(V_{\tau^*}|U_r^{(i)})$',r'$\Phi(V_{\widetilde\tau}|U_r^{(i)})$',r'$\Phi(V_{\widetilde\tau^{\textup{PCA}}}|U_r^{(i)})$'],\
              loc='center left', bbox_to_anchor=(1.04,0.5))
    fig.tight_layout()
    fig.savefig(f'{plots_eig_path}.pdf')
    fig.savefig(f'{plots_eig_path}.svg')