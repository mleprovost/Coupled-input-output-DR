#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Class for Bayes Problem definition 

TODO: ugly self.prior.mu as dynamic attribute of Covariance instance self.prior


"""
###########################################
#%% Imports
###########################################

from bayes import gaussian
import utils #for loading

import numpy as np
import os #to check if saved file exists


###########################################
#%% Classes
###########################################

class BIP:
    
    def __init__(self, model, R, bg, C, cache_tag = ''):
        self.model = model
        self.d = model.d
        self.m = model.m
        self.prior = gaussian.Gaussian(mu=bg, Sig=C, d=self.d)
        self.d_red = self.prior.d_red
        self.like = gaussian.Gaussian(mu=0, Sig=R, d=self.m)
        # determines if compute_G and gradG in one function or two
        self.G_gradG = callable(getattr(self.model, 'compute_G_gradG', None))
        self._cache_folder = f'cache{cache_tag}/{self.model.name}/{self}'
        
    def __str__(self):
        out = f'{self.model}R{self.like}bg{self.prior.mu_name}C{self.prior.Sig_name}'
        return out
    
    @property
    def cache_folder(self):
        os.makedirs(self._cache_folder, exist_ok=True)
        return self._cache_folder
    
            
    ###########################################
    def generate_prior_samples(self, M,\
                               filename = '', load = False, save = False):
    ###########################################
        """ Generate M prior_samples 
        
        For loranz96 use ergodic prior samples
        For conddiff,advdiff,burgers sample from bg Gaussian
        
        Each prior sample not large, 
        depending on M actually faster to save in one npy and load all
        than to save each prior sample separately like for gradG
        
        """
        if (load or save) and not filename:
            filename = f'{self.cache_folder}/prior'
        
        samples_precond = np.empty((M,self.d_red))
        samples = np.empty((M,self.d)) 
        nb_loaded = 0
        
        if load:
            nb_loaded, samples = utils.load_array(samples, M, filename = f'{filename}_samples.npy', name = 'prior samples')
            if isinstance(self.prior, gaussian.Gaussian):
                nb_loaded_precond, samples_precond = utils.load_array(samples_precond, M, filename =  f'{filename}_std.npy', name = 'prior samples precond')
            elif self.prior is None:
                nb_loaded_precond = M
            nb_loaded = min(M, nb_loaded_precond, nb_loaded)
            
        if nb_loaded < M:
            print(f'Generating {M-nb_loaded} prior samples')
            if isinstance(self.prior, gaussian.Gaussian): # Gaussian prior
                samples_precond[nb_loaded:] = np.random.randn(M-nb_loaded, self.d_red)
                samples[nb_loaded:]  = self.prior.mu + np.dot(self.prior.Sig12, samples_precond[nb_loaded:,].T).T
                if save: 
                    np.save(f'{filename}_std.npy', samples_precond)
                    np.save(f'{filename}_samples.npy', samples)
            elif self.prior is None: # Log-uniform prior
                samples[nb_loaded:] = np.exp(np.random.uniform(np.log(0.1), np.log(1), (M-nb_loaded,self.d)))
                if save: 
                    np.save(f'{filename}_samples.npy', samples)
        
        return samples_precond, samples
        
    ###########################################    
    def generate_state_samples(self, prior_samples,\
                               filename='', load = False, save=False):
    ###########################################
        """ Generate M samples of states, Returns function to load states """
        if (load or save) and not filename:
            filename = f'{self.cache_folder}/states'
        
        if self.model.name == 'conddiff':
            return None
            
        M = np.shape(prior_samples)[0]
        nb_computed = 0
        
        if load: 
            while os.path.isfile(f'{filename}_{nb_computed}.npy') and nb_computed < M:
                nb_computed += 1
            if nb_computed > 0:
                print(f'Loaded {nb_computed} states samples from {filename}_i.npy')
            else:
                print(f'No existing states samples in {filename}_i.npy')
                
        if nb_computed < M:
            print(f'Generating {M-nb_computed} states samples')
            for i in range(nb_computed, M):
                states = self.model.integrate_model(prior_samples[i], tf=self.model.T)
                if save: 
                    np.save(f'{filename}_{i}.npy', states)
                else:
                    np.save(f'{filename}_tmp{i}.npy', states)
               
        if save == True:
            def states(i):
                return np.load(f'{filename}_{i}.npy')
        else:
            def states(i):
                return np.load(f'{filename}_tmp{i}.npy')
        return states
    
    
    ###########################################    
    def generate_G_samples(self, input_samples, M=None,\
                           filename = '', load = False, save=False):
    ###########################################
        """ Generate M samples of G, Returns array(M,m) 
        
        ATTENTION: G already preconditioned with obs covariance
        
        input_samples: array(d) or array(nb_dt,d)
            Can prior_samples or states_samples for advdiff and burgers
        M : int
            In case input_samples are states, the loading functions have no M
            
        """
        
        if (load or save) and not filename:
            filename = f'{self.cache_folder}/G.npy'
        if not M:
            M = np.shape(input_samples)[0]
        samples = np.empty((M,self.m))
        nb_loaded = 0
        
        if load: 
            nb_loaded, samples = utils.load_array(samples, M, filename=filename, name = 'G samples')
            nb_loaded = min(M, nb_loaded)
                
        if nb_loaded < M:
            print(f'Generating {M-nb_loaded} G samples')
            for i in range(nb_loaded, M):
                if callable(input_samples): 
                    samples[i] = np.dot(self.model.compute_G(input_samples(i)),self.like.Sigm12).T
                else:
                    samples[i] = np.dot(self.model.compute_G(input_samples[i]),self.like.Sigm12).T
            if save: 
                np.save(filename, samples)
        return samples
    
    ###########################################    
    def generate_gradG_samples(self, input_samples, M=None,\
                               filename = '', load = False, save=False):
    ###########################################
        """ Generate M samples of gradG, Returns array(M,m,d)  
        
        input_samples: array(d) or array(nb_dt,d)
            Can be prior_samples or states_samples for advdiff and burgers
        M : int
            In case input_samples are states, the loading functions have no M
            
        """  
        
        if (load or save) and not filename:
            filename = f'{self.cache_folder}/gradG'
        if not M:
            M = np.shape(input_samples)[0]
        samples = np.empty((M,self.m, self.d))
        nb_loaded = 0
        
        if load: 
            nb_loaded, samples = utils.load_multiple_array(samples, M, filename=filename, name = 'gradG samples')
            nb_loaded = min(M, nb_loaded)

        if nb_loaded < M:
            print(f'Generating {M-nb_loaded} gradG samples')
            for i in range(nb_loaded, M):
                if callable(input_samples): 
                    samples[i] = np.dot(self.model.compute_gradG(input_samples(i)).T,self.like.Sigm12).T
                else:
                    samples[i] = np.dot(self.model.compute_gradG(input_samples[i]).T,self.like.Sigm12).T
                if save: 
                    np.save(f'{filename}_{i}.npy', samples[i])
        return samples
    
    ###########################################    
    def generate_G_gradG_samples(self, input_samples, \
                      filename_G = '', filename_gradG='', load = False, save = False):
    ###########################################
        """ Generate M samples of G_gradG 
        
        input_samples: array(d) or array(nb_dt,d)
            Can be prior_samples or states_samples for advdiff and burgers
        M : int
            In case input_samples are states, the loading functions have no M
        """
        
        if (load or save) and not filename_G:
            filename_G= f'{self.cache_folder}/G.py'
        if (load or save) and not filename_gradG:
            filename_gradG = f'{self.cache_folder}/gradG'
        
        M = np.shape(input_samples)[0]
        nb_loaded_G = 0
        nb_loaded_gradG = 0
        G = np.empty((M,self.m))
        gradG = np.empty((M,self.m, self.d))
        
        if load: 
            nb_loaded_G, G = utils.load_array(G, M, filename= filename_G, name = 'G samples')
            nb_loaded_G = min(nb_loaded_G, M)
            nb_loaded_gradG, gradG = utils.load_multiple_array(gradG, M, filename= filename_gradG, name = 'gradG samples')
            nb_loaded_gradG = min(M, nb_loaded_gradG)
        
        nb_loaded = min(nb_loaded_G, nb_loaded_gradG)
        if nb_loaded < M:
            print(f'Generating {M-nb_loaded} G_gradG samples')
            for i in range(nb_loaded, M):
                if callable(input_samples): 
                    G_tmp, gradG_tmp = self.model.compute_G(input_samples(i))
                else:
                    G_tmp, gradG_tmp = self.model.compute_G_gradG(input_samples[i]) 
                G[i] = np.dot(G_tmp,self.like.Sigm12).T
                gradG[i] = np.dot(gradG_tmp.T,self.like.Sigm12).T
                if save: 
                    np.save(filename_G, G)
                    np.save(f'{filename_gradG}_{i}.npy', gradG[i])
        return G, gradG
    
    ###########################################    
    def generate_G_lin(self, filename = '', load = False, save = False):
    ###########################################
        """ Generate G_lin, Returns array(m,d)"""
        if (load or save) and not filename:
            filename = f'{self.cache_folder}/G_lin.npy'
            
        loaded = False
        if load: 
            try: 
                G_lin = np.load(filename)
                loaded = True
            except FileNotFoundError:
                pass        
        if not loaded:
            G_lin = np.dot(self.model.compute_gradG(self.prior.mu).T,self.like.Sigm12).T
            if save: 
                np.save(filename, G_lin)
        return G_lin
    
    
    
###################################
#%% Initiate BIP Instance
###################################

def init_bayes(model, **kwargs):
    if model.name == 'conddiff':
        bayes = BIP(model,\
                    R=kwargs.get('R',0.1),\
                    bg=kwargs.get('bg',0),\
                    C=kwargs.get('C',1),\
                    cache_tag=kwargs.get('cache_tag',''))
    elif model.name[:7] in ['advdiff', 'burgers']:        
        bayes = BIP(model,\
                    R=kwargs.get('R',0.01),\
                    bg=kwargs.get('bg','gauss'),\
                    C=kwargs.get('C','Matern'),\
                    cache_tag=kwargs.get('cache_tag',''))
    return bayes