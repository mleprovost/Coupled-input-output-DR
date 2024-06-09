#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Abstract base class for models """

from abc import ABC, abstractmethod

class Model(ABC):
    
    @property
    @abstractmethod
    def name(self):
        pass
    
    @property
    @abstractmethod
    def d(self):
        pass
    
    @property
    @abstractmethod
    def m(self):
        pass

    @abstractmethod
    def __str__(self):
        pass
    
    @abstractmethod
    def integrate_model(self):
        pass

    @abstractmethod
    def compute_G(self):
        pass
    
    @abstractmethod
    def compute_gradG(self):
        pass
    