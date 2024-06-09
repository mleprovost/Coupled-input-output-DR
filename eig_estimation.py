#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__docformat__ = 'reStructuredText'

###########################################
# Imports
###########################################
import utils
from bayes import gaussian

import numpy as np
from scipy.stats import multivariate_normal
  
###########################################
#%% Functions
###########################################

class EIGEstimator:
    
    def __init__(self, goal, precond, M, bayes, estimator='Gauss', nb_inner=None, load=True, filepath='', **kwargs):
        self.U = goal.U
        self.goal_name = goal.name
        self.precond = precond #Wwhether U and G_lin are precond
        self.estimator = estimator
        
        self.bayes = bayes
        self.prior = self.bayes.prior
        self._priorU = None 
        self._G_lin = None
        
        self.M = M
        self.d, self.r = np.shape(self.U)
        self.m = self.bayes.m
        self.nb_inner = nb_inner
        
        self.filepath = filepath
        self.load = load
        
        if self.estimator=='NMC':
            priorU_samples, G_samples = self.setup_prior_G_samples(self.M*(self.nb_inner+1))
            self.priorU_samples = priorU_samples.reshape((self.M, self.nb_inner+1, self.r))
            self.G_samples = G_samples.reshape((self.M, self.nb_inner+1, self.m))
            std_normal = multivariate_normal(mean=np.zeros(self.m),cov=np.eye(self.m))
            self.data_samples = self.G_samples[:,0,:] + std_normal.rvs(size=self.M) # always std_normal since our G conditioned by Sig_like
        elif self.estimator == 'MCLA':
            self.priorU_samples = kwargs['prior_samples'] @ self.U
            self.gradGU_samples = kwargs['gradG_samples'] @ self.U
        elif self.estimator == 'DLIS':
            self.prior_samples = kwargs['prior_samples']
            self.priorU_samples = kwargs['prior_samples'] @ self.U
            self.G_samples = kwargs['G_samples']
            self.gradGU_samples = kwargs['gradG_samples'] @ self.U
            std_normal = multivariate_normal(mean=np.zeros(self.m),cov=np.eye(self.m))
            self.data_samples = self.G_samples + std_normal.rvs(size=self.M) 

    @property
    def priorU(self):
        if self._priorU is None: 
            self._priorU = gaussian.Gaussian(mu = self.U.T @ self.prior.mu, Sig=self.U.T @ np.dot(self.prior.Sig, self.U), d=self.r, d_red=self.r)
        return self._priorU
    
    @property
    def G_lin(self):
        if self._G_lin is None:
            self._G_lin = self.bayes.generate_G_lin()
            if self.precond:
                self._G_lin = np.dot(self._G_lin,self.prior.Sig12)
        return self._G_lin
            
    def setup_likelihood_pdf(self,W):
        s = np.shape(W)[1]
        Z_s =  1/np.sqrt((2*np.pi)**s)    
        def likelihood_pdf(y,x=None, G=None):
            if G is not None:
                res = 1/Z_s *  np.exp(-1/2 * np.linalg.norm( (W.T @ y - (W.T @ G.T).T).T, axis=0)**2)
            elif x is not None: 
                res = 1/Z_s *  np.exp(-1/2 * np.linalg.norm( (W.T @ y - (W.T @ self.bayes.generate_G_samples(x).T).T).T, axis=0)**2)
            return res
        return likelihood_pdf     
    
    def setup_prior_G_samples(self, nb):
        # Note that we precondition G_samples so that Sig_obs = 1
        prior_samples = np.empty((nb,self.d))
        G_samples = np.empty((nb, self.m))
        nb_loaded = 0
        if self.load:
            nb_prior, prior_samples = utils.load_array(prior_samples, nb, filename=f'{self.filepath}/eig_prior_samples.npy', name = 'prior_samples')
            nb_G, G_samples = utils.load_array(G_samples, nb, filename=f'{self.filepath}/eig_G_samples.npy', name = 'G_samples')
            nb_loaded = np.min([nb, nb_prior, nb_G])
        if nb_loaded < nb:
            print(f'Generating {nb-nb_loaded} prior, G samples for EIG estimator')
            prior_samples[nb_loaded:] = self.prior.rvs(size = nb-nb_loaded).reshape((nb-nb_loaded,self.d))
            G_samples[nb_loaded:] = self.bayes.generate_G_samples(prior_samples[nb_loaded:]) 
            np.save(f'{self.filepath}/eig_prior_samples.npy', prior_samples)
            np.save(f'{self.filepath}/eig_G_samples.npy', G_samples)
        prior_samples = prior_samples @ self.U
        return prior_samples, G_samples
    
    def compute_EIG(self, tau):
        W = np.zeros((self.m,len(tau)))
        W[tau,np.arange(len(tau))] = 1
        
        if self.estimator=='Gauss':
            res = self.Gauss(W)
        elif self.estimator =='NMC':
            res = self.NMC(W)
        elif self.estimator=='MCLA': 
            res = self.MCLA(W)
        elif self.estimator == 'DLIS':
            res = self.DLIS(W)
        return res

    def Gauss(self,V):
        """    Computes goal-oriented EIG
        
        requires preconditioned U and G_lin!
        Phi(V|U) = 1/2 * logdet(I_r+ H)
        where
        H = U.T @ G_lin.T @ V @ ( I_s + V.T @ G_lin.T @ (I_d-U@U.T) @ G_lin.T @ V)^-1 @ V.T @ G_lin @ U
        
        U : array(d_bg, r)
            U needs to be preconditioned
        G_lin: array(m, d_bg)
            G_lin also needs to be preconditioned
    
        """
        s = np.shape(V)[1]
        if self.precond:
            Sig_pos = np.eye(s) + np.linalg.multi_dot((V.T, self.G_lin, (np.eye(self.d)-self.U@self.U.T), self.G_lin.T, V))
            Sigm1_pos = np.linalg.inv(Sig_pos)
            H = np.linalg.multi_dot((self.U.T, self.G_lin.T, V, Sigm1_pos, V.T, self.G_lin,self.U))
            EIG = 1/2 * np.log(np.linalg.det(np.eye(self.r) +  H))
        else: # Need precond for implementation
            Sig_pos = np.eye(s) + np.linalg.multi_dot((V.T, self.G_lin, (np.eye(self.d)-self.U@self.U.T), self.G_lin.T, V))
            Sigm1_pos = np.linalg.inv(Sig_pos)
            H_tilde = np.linalg.multi_dot((self.priorU.Sig12, self.U.T, self.G_lin.T, V, Sigm1_pos, V.T, self.G_lin,self.U, self.priorU.Sig12.T))
            EIG = 1/2 * np.log(np.linalg.det(np.eye(self.r) +  H_tilde))
        return EIG
        
    def NMC(self, V):
        """ Computes EIG using the Nested Monte Carlo estimator
        
        data[i] depending on G_samples[i,0,:]
        Original from Ryan
        Goda2019 Multilevel MC for EIG explains it well,
        
        #inner samples determines bias
        #outer samples determines variance
        
        Parameters
        ----------
        G/prior_samples : array(M,nb_inner+1,m/d)
            M for outer loop, nb_inner for inner loop.
            G/prior_samples depends on what likeligood_fct takes 
        data_samples : array(M,m)
            N data samples in R^m depending on G/prior_samples[:,0]
        
        Source
        -----
        K.J. Ryan: Estimating Expected Information Gains for Experimental Designs 
            With Application to the Random Fatigue-Limit Model
        """
        likelihood_pdf = self.setup_likelihood_pdf(V)
        EIG = 0
        for i in range(self.M):
            try:
                term1 = likelihood_pdf(self.data_samples[i],G=self.G_samples[i,0,:])
                term2 = np.mean(likelihood_pdf(self.data_samples[i],G=self.G_samples[i,1:,:]))
            except:
                term1 = likelihood_pdf(self.data_samples[i],x=self.prior_samples[i,0,:])
                term2 = np.mean(likelihood_pdf(self.data_samples[i],x=self.prior_samples[i,1:,:]))
            EIG += 1/self.M * (np.log(term1) - np.log(term2))
        return EIG
    
    
    def MCLA(self,V):
        """ Computes EIG using the Laplace-based Monte Carlo estimator
        
        Implementation requires prior to be Gaussian
        prior_samples and data_samples coming from joint distribution, i.e.
        in practice samples data[i] depending on prior [i]
        Beck2017 explains it well, especially sum of gradients part
        Original from Long2012
        
        
        Parameters
        ----------
        prior_samples : array(M,d)
        WgradG_samples : array(M,s,d)
        
        Source
        -----
        Long et al. : Fast estimation of expected information gains for 
            Bayesian experimental designs based on Laplace approximations
        """            
        VgradGU_samples = np.array([V.T @ self.gradGU_samples[i] for i in range(self.M)])
        
        EIG = 0
        for i in range(self.M):
            #Attention our G multiplied with noise cov so Sig_like = I_m
            Sigm1_LA = VgradGU_samples[i].T @  VgradGU_samples[i] + self.priorU.Sigm1
            term1 = np.linalg.det(Sigm1_LA)/(2*np.pi)**self.r
            term2 = np.log(self.priorU.pdf(self.priorU_samples[i]))
            EIG += 1/self.M * (1/2*np.log(term1) - self.r/2 - term2)
        return EIG
    
    
    def DLIS(self,V):
        """ Computes EIG using the Nested Monte Carlo estimator with Lapalce importance sampling
        
        Implementation requires prior to be Gaussian
        Original Beck2017 
        
        Important to not just use the Laplace posterior samples in R^r as G(U@X_r)
        But add the X_perp part!!! otherwise results quite bad
        
        Source
        ------
        Beck et al.: Fast Bayesian experimental design: Laplace-based 
            importance sampling for the expected information gain
        """
        likelihood_pdf = self.setup_likelihood_pdf(V)
        VgradGU_samples = np.array([V.T @ self.gradGU_samples[i] for i in range(self.M)])
        
        EIG = 0
        for i in range(self.M):
            term1 = likelihood_pdf(y=self.data_samples[i],G = self.G_samples[i])
            Sigm1_LA = VgradGU_samples[i].T @  VgradGU_samples[i] + self.priorU.Sigm1
            ev,ew,_ = np.linalg.svd(Sigm1_LA)
            Sig_LA = ev @ np.diag(1/ew) @ ev.T
            Laplace = multivariate_normal(mean = self.priorU_samples[i], cov = Sig_LA)
            Laplace_samples = Laplace.rvs(size=self.nb_inner).reshape((self.nb_inner, self.r))
            prior_perp = self.prior_samples[i] @ (np.eye(self.d) - self.U@self.U.T)
            term2 = likelihood_pdf(y=self.data_samples[i],x = Laplace_samples @ self.U.T + prior_perp )
            IS_weight_nominator = self.priorU.pdf(Laplace_samples)
            IS_weight_denominator = Laplace.pdf(Laplace_samples)
            term2_IS = np.mean(term2 * IS_weight_nominator/IS_weight_denominator)
            EIG += 1/self.M* (np.log(term1) - np.log(term2_IS))
        return EIG
