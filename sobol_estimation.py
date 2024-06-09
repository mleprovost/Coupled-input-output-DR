#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__docformat__ = 'reStructuredText'

###########################################
# Imports
###########################################
import numpy as np

###########################################
#%% Classes
###########################################

class SobolEstimator:
    
    def __init__(self, dr, U, V, HX):
        self.mean_gradG = np.dot(np.mean(dr.gradG, axis=0), dr.bayes.prior.Sig12)
        self.d, self.r = np.shape(U)
        self.m, self.s  = np.shape(V)
        self.U = U
        self.V = V
        self.HX = HX
        _,self.ew,_ = np.linalg.svd(self.HX)
        self.D = np.diag(self.HX)
        self.normal = np.trace(np.cov((dr.G@V).T))
        
    def closed_lower(self):
        # res = 1 - np.trace(self.HX @ (np.eye(self.d)-self.U@self.U.T)) / self.normal
        res = 1-np.sum(self.ew[self.r:])/self.normal
        return res
    
    def closed_upper(self):
        res = 1 - np.linalg.norm(self.V.T @ self.mean_gradG @ (np.eye(self.d) - self.U@self.U.T))**2/ self.normal
        return res
    
    def total_lower(self):
        return np.linalg.norm(self.V.T @ self.mean_gradG @ self.U)**2 / self.normal
    
    def total_upper(self):
        # res = np.trace(self.U.T @ self.HX @ self.U) / self.normal
        res = np.sum(self.ew[:self.r])/self.normal
        return res
    
    