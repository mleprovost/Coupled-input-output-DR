#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Plotting functions """

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import patches
from matplotlib import cm
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable


#####################################
#%% Plotting Variables
#####################################

textwidth = 6.32283569445
colours =  plt.rcParams['axes.prop_cycle'].by_key()['color']
markers = ['o', '^', 's','P', 'X']
beamer_colours = {'UGAOrange': np.array([[232,78,15]])/255}
###################################
#%% Auxiliary Functions
###################################

def add_colorbar(im, ax, fig, loc = 'right', size = "5%", pad = 0.05, shrink = None, nbins = None):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(loc, size= size, pad=pad)
    if loc == 'right' or loc == 'left':
        orientation = 'vertical'
    else:
        orientation = 'horizontal'
    cbar = fig.colorbar(im, ax = ax, cax=cax, orientation=orientation)
    cbar.outline.set_visible(False)
    if nbins is not None:
        tick_locator = ticker.MaxNLocator(nbins=nbins)
        cbar.locator = tick_locator
        cbar.update_ticks()
        
def add_colorbar_axes(im, axes, fig, loc = 'right', pad = 0.02, shrink = 0.4, fraction=0.05, nbins = None):
    if loc == 'right' or loc == 'left':
        orientation = 'vertical'
    else:
        orientation = 'horizontal'
    cbar = fig.colorbar(im, ax = axes, orientation=orientation, shrink=shrink, pad=pad, fraction=fraction)
    cbar.outline.set_visible(False)
    if nbins is not None:
        tick_locator = ticker.MaxNLocator(nbins=nbins)
        cbar.locator = tick_locator
        cbar.update_ticks()
      

###########################################
#%% Plot out
###########################################

def _plot_out_1Dspacetime(model, out, ax=None, width = 0.4*textwidth, norm = None, alpha = 1,\
                          xlabel=True, ylabel=True, colorbar = True,\
                          ax_title = None, title = None, filename = ''):
    """ Plot 1D space-time output (G=H*M or M) as 2D Space-time plot
    
    Defaults to stand-alone figure, but can also only create ax.
    Supply norm to have same color scale if only create ax.
    
    Parameters
    ----------
    norm   : norm, optional
        Give norm to use same colormap as other plot. 
    alpha : float, optional
        Used for plotting background state for sensors
    xlabel : bool, optional
        Set xlabel or not. 
    ylabel : bool, optional
        Set ylabel or not. 
    colorbar : bool, optional
        Set colorbar or not. 
    
    """
    try:                # Assume G = Hobs * M(X)
        out = out.reshape((model.nb_obs_time, model.nb_obs_space)).T 
        n = model.n
    except ValueError:  # Otherwise G = M(X) with arbitrary nb time steps
        out = out.reshape((-1, model.N)).T 
        n = np.shape(out)[1]
        
    if ax is None:
        fig, ax = plt.subplots(figsize=(width,0.7*0.4*textwidth))
        fig.suptitle(title)
    else: # No title nor save if only create subplot 
        fig = plt.gcf()
        filename = ''
        
    im = ax.imshow(out,origin='lower',extent=[0,model.dt*n, 0, model.N], aspect= 'auto', norm = norm,  interpolation='none', alpha= alpha)
    
    ax.invert_yaxis()
    ax.set_xlim([0,model.dt*n])
    ax.grid(False)
    if ylabel: ax.set_ylabel('Space')
    if xlabel: ax.set_xlabel('Time')
    if colorbar: add_colorbar(im, ax, fig, size='2%')
    ax.set_title(ax_title)
    fig.tight_layout()
    
    if filename != '':
        fig.savefig(filename)    
    return fig, ax, im

#########################################
#%% Plot U 
#########################################


def plot_U(model, U, ax=None, nb_ev=5, goal=None, \
           ax_title=None, title=None, filename='', **kwargs):
    """ Plot several EV from U 

    Defaults to stand-alone plot. 
    Allow only plot ax with no legend, no x/ylabel etc. for series of plots.
    
    Parameters
    ----------
    ax    : axes, optional
        Only make subplot in given axes. The default is creating separate figure.
    legend : bool, optional
        Plot legend or not. 
    xlabel : bool, optional
        Plot xlabel or not. Only bool since different models have differnt labels.
        
    """
    if model.name == 'conddiff':
        fig, ax = _plot_U_1Dparticle(model, U, ax=ax, nb_ev=nb_ev, width=kwargs.get('width',0.4*textwidth),\
                                   legend=kwargs.get('legend', True), xlabel=kwargs.get('xlabel',True),\
                                   ax_title=ax_title, title=title)
    elif model.name in ['advdiff_static', 'burgers_static']:
        fig, ax = _plot_U_1Dspace(model, U, ax=ax, nb_ev=nb_ev, width=kwargs.get('width',0.4*textwidth),\
                                             legend=kwargs.get('legend', True), xlabel=kwargs.get('xlabel',True),\
                                             ax_title=ax_title, title=title)
    elif model.name in ['advdiff', 'burgers']:
        fig, ax = _plot_U_1Dspacetime(model, U, ax=ax, nb_ev=nb_ev, width=kwargs.get('width',0.4*textwidth),\
                                      legend=kwargs.get('legend', True), xlabel=kwargs.get('xlabel',True),\
                                      ax_title=ax_title, title=title)
    if goal:
        addgoal(model, goal, ax)
        
    if filename != '':
        fig.tight_layout()
        fig.savefig(filename)  
    return fig, ax



def _plot_U_1Dparticle(model, U, ax = None, nb_ev = 5, width = 0.4*textwidth,\
                    legend = True, xlabel = True, \
                    ax_title = None, title = None,filename = ''):
    """ Plot several EV from U as lines over [0,T]   """
    if ax is None:
        fig, ax = plt.subplots(figsize=(width,0.55*width))
        fig.suptitle(title)
    else: # No title if only create subplot 
        fig = plt.gcf()
        filename = ''
        
    nb_ev = min(nb_ev, np.shape(U)[1])
    xgrid = np.arange(0, model.T, model.dt)
    
    for j in range(nb_ev):
        ax.plot(xgrid, U[:,j], color = colours[j], label=fr'$u_{j}$')
        
    ax.set_xlim([0, model.T])
    if legend: ax.legend()
    if xlabel: ax.set_xlabel('Time')
    ax.set_title(ax_title)
    fig.tight_layout()
    if filename != '':
        fig.savefig(filename) 
    return fig, ax



def _plot_U_1Dspace(model, U, ax = None, nb_ev = 5, width = 0.5*textwidth,\
                               xlabel=True, legend = True,\
                               ax_title = None, title = None, filename = ''):
    """ Plot several EV from U as lines over [0,N]  """
    if ax is None:
        fig, ax = plt.subplots(figsize=(width,0.6*width))
        fig.suptitle(title)
    else: # No title if only create subplot 
        fig = plt.gcf()
        filename = ''
        
    nb_ev = min(nb_ev,np.shape(U)[1])
    
    for j in range(nb_ev):
        ax.plot(U[:,j], color = colours[j], label=fr'$u_{j}$')
      
    if legend: ax.legend()
    if xlabel: ax.set_xlabel('Space')
    ax.set_title(ax_title)
    fig.tight_layout()
    if filename != '':
        fig.savefig(filename) 
    return fig, ax


def _plot_U_1Dspacetime(model, U, ax = None, nb_ev = 5, width = 0.5*textwidth, \
                        legend = True, xlabel = True, \
                        ax_title = None, title = None, filename = ''):
    """ Plot several EV from U as lines over [0,N], same as static   """
    fig, ax = _plot_U_1Dspace(model, U, ax=ax, nb_ev=nb_ev, width=width, legend=legend, xlabel=xlabel, ax_title=ax_title, title=title, filename=filename) 
    return fig, ax



###########################################
#%% Plot V
###########################################

def plot_V(model, V, ax=None, nb_ev=5, goal=None, \
           ax_title=None, title=None, filename='', **kwargs):
    """ Plot several EV from V

    Defaults to stand-alone plot. 
    Allow only plot ax with no legend, no x/ylabel etc. for series of plots.
    
    Parameters
    ----------
    ax    : axes, optional
        Only make subplot in given axes. The default is creating separate figure.
    legend : bool, optional
        Plot legend or not. 
    xlabel : bool, optional
        Plot xlabel or not.

    """
    if model.name == 'conddiff':
        fig, ax = _plot_V_1Dparticle(model, V, ax=ax, nb_ev=nb_ev, width=kwargs.get('width',0.4*textwidth),\
                                   legend=kwargs.get('legend', True), xlabel=kwargs.get('xlabel',True),\
                                   ax_title=ax_title, title=title)
    elif model.name in ['advdiff_static', 'burgers_static']:
        fig, ax = _plot_V_1Dspace(model, V, ax=ax, nb_ev=nb_ev, width=kwargs.get('width',0.4*textwidth),\
                                             legend=kwargs.get('legend', True), xlabel=kwargs.get('xlabel',True),\
                                             ax_title=ax_title, title=title)
    elif model.name in ['advdiff', 'burgers']:
        assert ax is None
        fig, ax = _plot_V_1Dspacetime(model, V, nb_ev=nb_ev, alpha=kwargs.get('alpha',1),\
                                      xlabel=kwargs.get('xlabel',True), ylabel=kwargs.get('ylabel',True),\
                                      norm=kwargs.get('norm',None), colorbar=kwargs.get('colorbar',True),\
                                      ax_title=ax_title, title=title)

    if goal:
        addgoal(model, goal, ax)
    if filename != '':
        fig.tight_layout()
        fig.savefig(filename) 
    return fig, ax

        

def _plot_V_1Dparticle(model, V, ax = None, nb_ev = 5, width = 0.4*textwidth,\
                    legend = True, xlabel = True, ylabel= True, \
                    ax_title = None, title = None,filename = ''):
    """ Plot several EV from V as lines over [0,T]     """
    if ax is None:
        fig, ax = plt.subplots(figsize=(width,0.55*width))
        fig.suptitle(title)
    else: # No title if only create subplot 
        fig = plt.gcf()
        filename = ''
        
    nb_ev = min(nb_ev, np.shape(V)[1])
    ygrid = np.arange(0,model.T + model.dt, model.dt*model.k)[1:]
    
    for j in range(nb_ev):
        ax.plot(ygrid, V[:,j], color = colours[j], label=fr'$v_{j}$')
        
    ax.set_xlim([0, model.T])
    if legend: ax.legend()
    if xlabel: ax.set_xlabel('Time')
    ax.set_title(ax_title)
    fig.tight_layout()
    if filename != '':
        fig.savefig(filename) 
    return fig, ax


def _plot_V_1Dspace(model, V, nb_ev = 5, ax=None, xlabel=True, legend = True,\
                    width = 0.5*textwidth, ax_title = None, title = None, filename = ''):
    """ Plot several EV from U as lines over [0,N]  """
    if ax is None:
        fig, ax = plt.subplots(figsize=(width,0.6*width))
        fig.suptitle(title)
    else: # No title if only create subplot 
        fig = plt.gcf()
        filename = ''
        
    nb_ev = min(nb_ev,np.shape(V)[1])
    
    for j in range(nb_ev):
        ax.plot(V[:,j], color = colours[j], label=fr'$v_{j}$')
      
    if legend: ax.legend()
    if xlabel: ax.set_xlabel('Space')
    ax.set_title(ax_title)
    fig.tight_layout()
    if filename != '':
        fig.savefig(filename) 
    return fig, ax


def _plot_V_1Dspacetime(model, V, nb_ev = 3, norm = None, alpha = 1,\
                        xlabel=True, ylabel=None, colorbar = True,\
                        ax_title = None, title = None, filename = ''):
    """ Plots several EV from V as 2D subplot in one row with same colormap
    
    Stand-alone figure (since itself consists of multiple subplots) but allow color norm
    to match with other plot_V
    
    """
    nb_ev = min(nb_ev, np.shape(V)[1], 3)    
    if norm is None:
        vmax = np.max(V[:,:nb_ev])
        vmin = np.min(V[:,:nb_ev])
        norm = colors.Normalize(vmin,vmax)
    im=cm.ScalarMappable(norm=norm)
    
    fig, ax = plt.subplots(nrows=1,ncols=nb_ev, figsize=(textwidth,0.25*textwidth), sharex=True, sharey=True)    
    _plot_out_1Dspacetime(model, out = V[:,0], ax=ax[0], xlabel=xlabel, ylabel=ylabel, norm = norm, alpha=alpha, colorbar = False)
    for i in range(nb_ev):
        _plot_out_1Dspacetime(model, out = V[:,i], ax=ax[i], xlabel=xlabel, ylabel=False, norm = norm,alpha=alpha, colorbar = False)
        
    fig.suptitle(title)
    if nb_ev%2==0: 
        # ax.invert_yaxis called for each ax but sharey means each call inverts all axes anew
        ax[0].invert_yaxis()
    if colorbar:
        # Not good, will reduce last ax but not even with rest
        add_colorbar_axes(im, ax.ravel().tolist(), fig, pad = 0.02, shrink = 1, fraction=0.05)
    if filename != '':
        fig.savefig(filename) 
    return fig, ax

#######################################
#%% addgoals and addsensors
#######################################

def addgoal(model, goal, ax):
    """ Convert indices tau to match observation times/locations for plotting """
    if model.name == 'conddiff':
        _addgoal_1Dparticle(ax, *goal.marker)
    elif model.name in ['advdiff_static', 'burgers_static']:
        _addgoal_1Dspace(ax, *goal.marker)
    elif model.name in ['advdiff', 'burgers']:
        if goal.V is not None:
            _addgoal_1Dspacetime(ax, *goal.marker)


def _addgoal_1Dparticle(ax, x, y, width, height, alpha=0.3):
    ax.add_patch(patches.Rectangle((x,y), width=width,height=height,color=beamer_colours['UGAOrange'], alpha=alpha, fill = True, zorder=10))
    return ax

def _addgoal_1Dspace(ax,x,y,width,height, alpha=0.3):
    ylim = ax.get_ylim() # For whatever reason need this here so ylim doesn't get redrawn
    ax.add_patch(patches.Rectangle((x,y), width=width,height=height,color=beamer_colours['UGAOrange'], alpha=alpha, fill = True, zorder=10))
    return ax

def _addgoal_1Dspacetime(ax, x, y, width, height,centered=True, alpha=1, linestyle='-', linewidth=1):
    #TODO: something wrong with centering
    ax.add_patch(patches.Rectangle((x,y-0.5*centered), width=width, height=height, alpha=alpha, fill = False, \
                                   edgecolor=beamer_colours['UGAOrange'], linestyle=linestyle,linewidth=linewidth,\
                                   clip_on=False, zorder=100)) 
    return ax


def _idx2sensor_1Dparticle(model, tau):
    times = tau*model.k/model.m*model.T
    return times

def _idx2sensor_1Dspace(model, tau):
    locations = tau
    return locations
    
def _idx2sensor_1Dspacetime(model, tau, centered=False):
    times, locations = np.unravel_index(tau, (model.nb_obs_time, model.nb_obs_space))
    times = (times + int(not model.obs_init) + centered*0.5) * (model.kt*model.dt)
    locations = (locations+centered*0.5) * model.ks
    return times, locations


# Add sensor as separate function since can be added on top of EV or separate ax etc
# Returns im for legend
def _addsensor_1Dparticle(model, tau, ax,\
                          s = 20, marker = markers[0], colour= colours[0], position=1, alpha=1):
    # position = y value, allows different sets of sensors to be plotted above each other
    sensors = _idx2sensor_1Dparticle(model, tau)
    im = ax.scatter(sensors, position * np.ones(len(sensors)), marker=marker, color=colour, s=s, alpha=alpha)
    return im

def _addsensor_1Dspace(model, tau, ax,\
                       s = 20, marker = markers[0], colour= colours[0], position=1, alpha=1):
    sensors = _idx2sensor_1Dspace(model, tau)
    im = ax.scatter(sensors, position * np.ones(len(sensors)), marker=marker, color=colour, s=s, alpha=alpha)
    return im

def _addsensor_1Dspacetime(model, tau, ax,\
                           s=20, marker='s', facecolors='none', edgecolors='black', linewidths=0.5, alpha=1):
    time, space = _idx2sensor_1Dspacetime(model,tau, centered=True)
    im = ax.scatter(time, space, s=s, marker = marker, facecolors=facecolors, edgecolors=edgecolors,\
                  linewidths=linewidths, alpha=alpha, clip_on=False)
    return im


#######################################
#%% Plot sensors
#######################################

def plot_sensors(model, taus, goal=None, ax=None, sensor_labels=None, \
                 s=20, alpha=1, markers=markers, colours=colours,\
                 title=None, filename='', **kwargs):
    """ Plots different sets of sensors slightly shifted 
    
    Defaults to stand-alone plot
    But can also be axis with e.g. goal-oriented EV on axis above

    """    
    if model.name =='conddiff':
        fig, ax = _plot_sensors_1Dparticle(model, taus, ax=ax, goal=goal, sensor_labels=sensor_labels,\
                                         xlabel=kwargs.get('xlabel',True), width=kwargs.get('width',0.5*textwidth), positions=kwargs.get('positions',None), \
                                         s=s, alpha=alpha, markers=markers, colours=colours)
    elif model.name in ['advdiff_static', 'burgers_static']:
        fig, ax = _plot_sensors_1Dspace(model, taus, ax=ax, goal=goal, sensor_labels=sensor_labels,\
                                                   xlabel=kwargs.get('xlabel',True), width=kwargs.get('width',0.5*textwidth), positions=kwargs.get('positions',None), \
                                                   s=s, alpha=alpha, markers=markers, colours=colours)
    elif model.name in ['advdiff', 'burgers']:
        fig, ax = _plot_sensors_1Dspacetime(model, taus, ax=ax, goal=goal, bg = kwargs.get('bg',None), sensor_labels=sensor_labels,\
                                           xlabel=kwargs.get('xlabel',True), ylabel=kwargs.get('ylabel',True), width=kwargs.get('width',0.5*textwidth), \
                                           s=s, alpha=alpha, markers=markers, colours=colours, linewidths=kwargs.get('linewidths',0.5))
    if filename!='':
        fig.savefig(f'{filename}.pdf')
        fig.savefig(f'{filename}.svg')
    return fig, ax
    
    
def _plot_sensors_1Dparticle(model, taus, goal = None, ax=None, sensor_labels=None, \
                           xlabel = True, width = 0.5*textwidth, positions=None, \
                           s=20, alpha=1, markers=markers, colours=colours):
    if ax is None:
        fig, ax = plt.subplots(figsize=(width,0.6*width))
    else:
        fig = plt.gcf()
    if positions is None:
        positions = np.arange(len(taus))
    if goal is not None:
        _addgoal_1Dparticle(ax, *goal.marker)
        
    ims = [None]*len(taus)
    for i in range(len(taus)):
        ims[i] = _addsensor_1Dparticle(model, taus[i], ax, position=positions[i], marker=markers[i], colour=colours[i], s=s, alpha=alpha)
        
    ax.set_xlim([0,model.T])
    ax.get_yaxis().set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.tick_params(axis='x', colors='black')
    ax.set_ylim([np.min(positions)-1.5, np.max(positions) + 0.2])
    ax.set_facecolor('white')
    if xlabel: ax.set_xlabel('Time')
    if sensor_labels:
        ax.legend(handles=ims, labels = sensor_labels, loc='upper left', mode='expand', ncols=len(taus),\
                   bbox_to_anchor=(0, -0.8, 1, 0.2))
    fig.tight_layout()
    
    return fig, ax


def _plot_sensors_1Dspace(model, taus, goal=None, ax=None, sensor_labels=None, \
                           xlabel = True, width = 0.5*textwidth, positions=None, \
                           s=20, alpha=1, markers=markers, colours=colours):
    if ax is None:
        fig, ax = plt.subplots(figsize=(width,0.6*width))
    else:
        fig = plt.gcf()
    if positions is None:
        positions = np.arange(len(taus))
    if goal is not None:
        _addgoal_1Dspace(ax,*goal.marker)
        
    ims = [None]*len(taus)
    for i in range(len(taus)):
        ims[i] = _addsensor_1Dspace(model=model, tau=taus[i], ax=ax, position=positions[i], marker=markers[i], colour=colours[i], s=s, alpha=alpha)
        
    ax.set_xlim([0,model.N])
    ax.get_yaxis().set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.tick_params(axis='x', colors='black')
    ax.set_ylim([np.min(positions)-0.5, np.max(positions) + 0.2])
    ax.set_facecolor('white')
    if xlabel: ax.set_xlabel('Space')
    if sensor_labels:
        fig.legend(handles=ims, labels = sensor_labels, loc='lower left', mode='expand', ncols=len(taus),\
                    bbox_to_anchor=(0.2,0,0.7,0.1), borderaxespad=0)
    fig.tight_layout(rect=[0,0.04,1,1])
    
    return fig, ax


def _plot_sensors_1Dspacetime(model, taus, goal=None, ax=None, bg = None, sensor_labels = None,\
                              xlabel = True, ylabel= True, width = 0.5*textwidth, \
                              s=20, markers=markers, colours= colours, alpha=1, linewidths=1):
    """ Plot bg state in background, caondidate locations as boxes and computed sensors on top
    
    Defaults to standalone figure.
    But can also creat only ax with e.g. other sensors on ax next to it

    Parameters
    ----------
    bg : array(d), optional
        For computing background state in background. The default is None.

    """
    # Plot longrun from bg in background
    states = model.integrate_model(bg, model.T)
    fig, ax, _ = _plot_out_1Dspacetime(model, states, ax = ax, alpha=0.1, colorbar=False, ylabel=ylabel, xlabel=xlabel, width = width)
    # Adding goal markers
    if goal: _addgoal_1Dspacetime(ax,*goal.marker) 
    # Adding candidate locations as grid
    gridy = np.arange(model.ks*0.5, model.N,model.ks)
    gridx = np.arange((model.dt*model.kt)*(int(not model.obs_init) + 0.5), model.T, model.dt*model.kt)
    ax.set_yticks(gridy)   
    ax.set_xticks(gridx)  
    ax.set_xlim([0,model.T])
    ax.grid(True)
    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
    for tick in ax.yaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
    # # Adding candidate locations as boxes
    # _addsensor_1Dspacetime(model,np.arange(model.m), ax, s=s, marker = 's', facecolors='none', edgecolors='black', linewidths=linewidths)
    
    # Add computed sensors
    ims = [None]*len(taus)
    for i in range(len(taus)):
        ims[i] = _addsensor_1Dspacetime(model, taus[i], ax, s=s, marker = markers[i], alpha=alpha, facecolors='none', edgecolors=colours[i], linewidths=linewidths)
    
    if sensor_labels:
        ax.legend(handles=ims, labels = sensor_labels, loc='upper left', mode='expand', ncols=len(taus),\
                   bbox_to_anchor=(0, -0.8, 1, 0.2))
    fig.tight_layout()
    return fig, ax
