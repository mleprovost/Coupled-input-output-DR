#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__docformat__ = 'reStructuredText'

###########################################
# Imports
###########################################
import numpy as np


###########################################
# Classes
###########################################
class ErrorEstimator:
    
    def __init__(self, dr, precond = False, nb_inner= 5, prior_std = None, prior_samples = None, G=None, gradG = None):
        self.dr = dr
        self.d = dr.d
        self.d_red = dr.d_red
        self.m = dr.m
        self.M = dr.M
        self.precond = precond
        self.full_grad = dr.full_grad
        self.nb_inner = nb_inner
        
        if prior_samples is None:
            self.prior_std = dr.prior_std
            self.prior_samples = dr.prior_samples
            self.G = dr.G
            if self.full_grad:
                self.gradG = dr.gradG
        else: 
            if precond: assert prior_std is not None
            self.prior_std = prior_std
            self.prior_samples = prior_samples
            if G is None:
                self.G = dr.bayes.generate_G_samples(input_samples = prior_samples)
            else: 
                self.G = G
            if self.full_grad:
                if gradG is None:
                    if dr.G_gradG:
                        _, self.gradG = dr.gener_G_gradG(input_samples = prior_samples)
                    else:
                        self.gradG = dr.bayes.generate_gradG_samples(input_samples = prior_samples)
                else:
                    self.gradG = gradG
        
        
    def compute_err(self, U,V, Xbar=None):
        """  Computes L2-error estimate
        
        E[\|G(X) - G^*(X)\|^2]  = 1/2 * E_{X,X'}[\|G(X) - Ps @ G(X_r, X_perp') - Ps_perp @ G(X') \|^2 ] 
                                                                                             
        """
        # First iteration has U or V being 0
        if not np.any(V):
              V = np.eye(self.m)
        elif not np.any(U):
            if self.precond:
                U = np.eye(self.d_red)
            else:
                U = np.eye(self.d)
            
        Ps = V @ V.T
        Ps_perp =  (np.eye(self.m) - Ps)
        if Xbar is None:
            Xbar = np.random.randn(self.M, self.d_red)
        X = np.dot(self.dr.bayes.prior.Sig12 , Xbar.T).T + self.dr.bayes.prior.mu
        G2 = self.dr.bayes.generate_G_samples(X)
        Pr = U @ U.T 
        if not self.precond:
            G1 = self.dr.bayes.generate_G_samples((Pr @ self.prior_samples.T + (np.eye(self.d) - Pr) @ X.T).T)
        else:
            Xmixed = np.dot(self.dr.bayes.prior.Sig12 , (Pr @ self.prior_std.T + (np.eye(self.d_red) - Pr) @ Xbar.T)).T + self.dr.bayes.prior.mu
            G1 = self.dr.bayes.generate_G_samples(Xmixed)

        res = 1/2* np.mean(np.linalg.norm(self.G - (Ps @ G1.T).T - (Ps_perp @ G2.T).T, axis = 1)**2)
        return res
    
    
    def compute_upper(self,U,V):
        """    Computes Upper Error Bound estimate
        
        E[||(1 - Vs @ Vs.T) @ gradG(X)||^2   +   ||Vs @ Vs.T @ gradG(X) (1-Ur @ Ur.T)||^2]

        """
        if self.full_grad:
            # First iteration has U or V being 0
            if not np.any(V):
                  V = np.eye(self.m)
            if not np.any(U):
                if self.precond:
                    U = np.eye(self.d_red)
                else:
                    U = np.eye(self.d)
                 
            if not self.precond:
                res = np.mean([np.linalg.norm((np.eye(self.m) - V @ V.T) @ self.gradG[k])**2 \
                                  + np.linalg.norm(V @ V.T @ self.gradG[k] @ (np.eye(self.d) - U @ U.T))**2 for k in range(self.M)])
            else: 
                # G(X) with X~N(mu,C) becomes G(C12 @ Xbar + mu) with Xbar~N(0,1)
                # gradG becomes gradG @ C12
                d_red = np.shape(U)[0]
                res = np.mean([np.linalg.norm((np.eye(self.m) - V @ V.T) @ np.dot(self.gradG[k] , self.dr.bayes.prior.Sig12))**2 \
                                  + np.linalg.norm(V @ V.T @ np.dot(self.gradG[k] , self.dr.bayes.prior.Sig12) @ (np.eye(d_red) - U @ U.T))**2 for k in range(self.M)])
          
        else:
            """    Computes Upper Error Bound estimate
            
            E[||(1 - Vs @ Vs.T) @ gradG(X)||^2   +   ||Vs @ Vs.T @ gradG(X) (1-Ur @ Ur.T)||^2]

            """
            # First iteration has U or V being 0
            if not np.any(V):
                Z_out = np.random.randn(self.nb_inner,self.m)
                if not self.precond:
                    def term2(k):
                        return np.mean([np.linalg.norm(self.dr.compute_gradGT_w(Z_out[i], states = self.G[k]) - \
                              U @ (U.T @ self.dr.compute_gradGT_w(Z_out[i], states = self.G[k])))**2 for i in range(self.nb_inner)])
                else:
                    def term2(k):
                        return np.mean([np.linalg.norm( self.dr.bayes.prior.Sig12.T @ self.dr.compute_gradGT_w(Z_out[i], states = self.G[k]) - \
                               U @ (U.T @ (self.dr.bayes.prior.Sig12.T @ self.dr.compute_gradGT_w(Z_out[i], states = self.G[k]))))**2 for i in range(self.nb_inner)])
                res = np.mean([term2(k) for k in range(self.M)])
                
            elif not np.any(U):
                if not self.precond:
                    Z_in = np.random.randn(self.nb_inner,self.d)
                    def term1(k):
                        return np.mean([np.linalg.norm(self.dr.compute_gradG_w(Z_in[i], states = self.G[k]) - V @ (V.T @ self.dr.compute_gradG_w(Z_in[i], states = self.G[k])))**2 for i in range(self.nb_inner)])
                    res = np.mean([term1(k) for k in range(self.M)])
                else: 
                    # G(X) with X~N(mu,C) becomes G(C12 @ Xbar + mu) with Xbar~N(0,1)
                    # gradG becomes gradG @ C12
                    d_red = np.shape(U)[0]
                    Z_in = (self.dr.bayes.prior.Sig12 @ np.random.randn(self.d_red, self.nb_inner)).T
                    def term1(k):
                        return np.mean([np.linalg.norm(self.dr.compute_gradG_w(Z_in[i], states = self.G[k]) - \
                               V @ (V.T @ self.dr.compute_gradG_w(Z_in[i], states = self.G[k])))**2 for i in range(self.nb_inner)])
                    res = np.mean([term1(k) for k in range(self.M)])
              
            else:
                if not self.precond:
                    Z_in = np.random.randn(self.nb_inner,self.d)
                    Z_out = np.random.randn(self.nb_inner,self.m)
                    def term1(k):
                        return np.mean([np.linalg.norm(self.dr.compute_gradG_w(Z_in[i], states = self.G[k]) - V @ (V.T @ self.dr.compute_gradG_w(Z_in[i], states = self.G[k])))**2 for i in range(self.nb_inner)])
                    def term2(k):
                        np.mean([np.linalg.norm(self.dr.compute_gradGT_w(V @ (V.T @ Z_out[i]), states = self.G[k]) - U @ (U.T @ self.dr.compute_gradGT_w(V @ (V.T @ Z_out[i]), states = self.G[k])))**2 for i in range(self.nb_inner)])
                    res = np.mean([term1(k) + term2(k) for k in range(self.M)])
                else: 
                    # G(X) with X~N(mu,C) becomes G(C12 @ Xbar + mu) with Xbar~N(0,1)
                    # gradG becomes gradG @ C12
                    d_red = np.shape(U)[0]
                    Z_in = (self.dr.bayes.prior.Sig12 @ np.random.randn(self.d_red, self.nb_inner)).T
                    Z_out = np.random.randn(self.nb_inner,self.m)
                    def term1(k):
                        return np.mean([np.linalg.norm(self.dr.compute_gradG_w(Z_in[i], states = self.G[k]) - \
                               V @ (V.T @ self.dr.compute_gradG_w(Z_in[i], states = self.G[k])))**2 for i in range(self.nb_inner)])
                    def term2(k):
                        return np.mean([np.linalg.norm( self.dr.bayes.prior.Sig12.T @ self.dr.compute_gradGT_w(V @ (V.T @ Z_out[i]), states = self.G[k]) - \
                               U @ (U.T @ (self.dr.bayes.prior.Sig12.T @ self.dr.compute_gradGT_w(V @ (V.T @ Z_out[i]), states = self.G[k]))))**2 for i in range(self.nb_inner)])
                    res = np.mean([term1(k) + term2(k) for k in range(self.M)])
                    
        return res
        
    
    def compute_lower(self, U,V):
        """    Computes Lower Error Bound estimate
        
        ||(1 - Vs @ Vs.T)@ E[gradG(X)] ||^2   +   ||Vs @ Vs.T @ E[gradG(X)] (1-Ur @ Ur.T)||^2 
        
        """   
        assert self.gradG is not None
        
        # First iteration has U or V being 0
        if not np.any(V):
              V = np.eye(self.m)
        if not np.any(U):
            if self.precond:
                U = np.eye(self.d_red)
            else:
                U = np.eye(self.d)
       
        mean_grad = np.mean(self.gradG, axis = 0)
        if not self.precond:
            res = np.linalg.norm((np.eye(self.m) - V @ V.T) @ mean_grad)**2 \
                              + np.linalg.norm(V @ V.T @ mean_grad @ (np.eye(self.d) - U @ U.T))
        else: 
            # G(X) with X~N(mu,C) becomes G(C12 @ Xbar + mu) with Xbar~N(0,1)
            # gradG becomes gradG @ C12
            d_red = np.shape(U)[0]
            res = np.linalg.norm((np.eye(self.m) - V @ V.T) @ np.dot(mean_grad , self.dr.bayes.prior.Sig12))**2 \
                              + np.linalg.norm(V @ V.T @ np.dot(mean_grad , self.dr.bayes.prior.Sig12) @ (np.eye(d_red) - U @ U.T))**2     
                              
        return res