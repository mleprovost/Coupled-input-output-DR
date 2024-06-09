#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__docformat__ = 'reStructuredText'

###################################
#%% Imports
###################################

import numpy as np
from dataclasses import dataclass

###########################################
#%% Classes
###########################################

@dataclass
class Goal:
    name: str
    U: np.ndarray = None
    V: np.ndarray = None
    marker: list = None

class GoalsSetter:    
    
    def __init__(self, model, r=None, s=None):
        self.model = model
        self.d = model.d
        self.m = model.m
        self.r = r
        self.s = s
    
    
    def setup_goals(self, mode='all', precond=False,Sigm12=None, **kwargs):
        """ Setup dict with goals, index by their goal names
    
        Parameters
        ----------
        mode : 'in', 'out' or 'all'
            Only create input/output goals or all
        r,s : int
            Size of goal, optional if only in/out or some models fixed
        precond:
            Input goals need to be preconditioned with Sigm12
        **kwargs : 
            To specify certain goals
    
        Returns
        -------
        goals: dict
            keys corresponding to goal names
    
        """

        if self.model.name == 'conddiff':
            goals = self._setup_goals_conddiff(mode=mode, **kwargs)
        elif self.model.name in ['advdiff_static','burgers_static']:
            goals = self._setup_goals_1Dspacetime_static(mode=mode, **kwargs)
        elif self.model.name in ['advdiff','burgers']:
            goals = self._setup_goals_1Dspacetime(mode=mode, **kwargs)
            
        if precond:
            for key, goal in goals.items():
                if goal.U is not None:
                    goals[key].U = np.dot(goal.U.T, Sigm12).T
        return goals


    def _setup_goals_conddiff(self, times=None, mode='all'):
        early = 20
        late = 70
        goals = {}
        if mode in ['in', 'all']:
            # # Fix U to projection onto early Noise
            U = np.zeros((self.model.d, self.r))
            U[early:early+ self.r,:self.r] = np.eye(self.r)
            goals['Uearly'] = Goal('Uearly', U=U, marker = [early/self.model.d,-10000,self.r/self.model.d,20000])
        if mode in ['out', 'all']:
            V = np.zeros((self.model.m,self.s))
            V[late:late+self.s,:self.s]=np.eye(self.s)
            goals['Vlate'] = Goal('Vlate', V=V, marker = [late/self.model.d,-10000,self.s/self.model.d,20000])
            
        # if times is None:
        #     # times is used for both input and output space
        #     # assumes that self.d=self.m
        #     times = np.arange(5,self.d+5,10)
        #     if times[-1]+self.r >self.d:
        #         times = times[:-1]
        # goals = {}
        # if mode in ['in', 'all']:
        #     # Fix U to driving forces during time interval [i,i+r]
        #     for i in times:
        #         U = np.zeros((self.d, self.r))
        #         U[i:i+ self.r,:self.r] = np.eye(self.r)
        #         goals[f'U{i}'] = Goal(f'U{i}', U=U, marker = [i/self.d,-10000,self.r/self.d,20000])
        # if mode in ['out', 'all']:
        #     # Fix V to particla path observations during time interval [i,i+r]
        #     for i in times:
        #         V = np.zeros((self.m, self.s))
        #         V[i:i+ self.s,:self.s] = np.eye(self.s)
        #         goals[f'V{i}'] = Goal(f'V{i}', V=V, marker = [i/self.m,-10000,self.s/self.m,20000])
        return goals
    
    def _setup_goals_1Dspacetime_static(self,locations=None, mode='all'):
        if locations is None:
            # locations used for both input and output space
            # assumes that self.d = self.m and self.r=self.s
            locations = np.arange(5,self.d+5,self.r+5)
            if locations[-1]+self.r >self.d:
                locations = locations[:-1]
            
        goals = {}
        if mode in ['in', 'all']:
            # Fix U to initial conditions at nodes [i:i+r]
            for i in locations:
                U = np.zeros((self.d, self.r))
                U[i:i+self.r,:self.r] = np.eye(self.r)
                goals[f'U{i}'] = Goal(f'U{i}', U=U, marker = [i,-10000, self.r, 20000])
        if mode in ['out', 'all']:
            # Fix V to final states at the nodes [i:i+r]
            for i in locations:
                V = np.zeros((self.m, self.s))
                V[i:i+self.s,:self.s] = np.eye(self.s)
                goals[f'V{i}'] = Goal(f'V{i}', V=V, marker = [i,-10000, self.s, 20000])
        return goals
            
    def _setup_goals_1Dspacetime(self, locations=None, obs_width=None, mode='all'):
        if locations is None:
            # For input space model.ks irrelevant
            locations = np.array([1/3*self.model.N, 1/2*(self.model.N-self.r), 2/3*self.model.N]).astype(int)
        if obs_width is None:
            # Instead of specifying s, we give obs_width and compute s
            obs_width = int((0.1*self.model.N)/self.model.ks)
            
        goals = {}
        if mode in ['in', 'all']:
            # Fix U to initial conditions at nodes [i:i+r]
            for i in locations: 
                U = np.zeros((self.d, self.r))
                U[i:i+self.r,:self.r] = np.eye(self.r)
                goals[f'U{i}'] = Goal(f'U{i}', U=U, marker = [0, i, self.model.T, self.r])
        if mode in ['out', 'all']:            
            # Fix V to observation at nodes [i:i+r] over all times
            s = obs_width * self.model.nb_obs_time
            #Attention: for output space ks relevant in locations
            for i in (locations/self.model.ks).astype(int):
                V = np.zeros((self.m,s))
                k = 0
                for t in range(self.model.nb_obs_time):
                    for j in range(obs_width):
                        V[t*self.model.nb_obs_space + i + j,k]=1
                        k += 1
                goals[f'V{i}'] = Goal(f'V{i}', V=V, marker = [0, i, self.model.T, obs_width])   
        return goals
    