"""
@author: Hasan Albinsaid
@site: https://github.com/hasanabs
"""

import numpy as np

def db2lin(db):
    return np.power(10,db/10)

def H(Nr, Nt):
    return (np.random.randn(Nr,Nt)+np.random.randn(Nr,Nt)*1j)/np.sqrt(2)

def herm(matrix):
    return np.transpose(np.conjugate(matrix))
