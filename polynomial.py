from math import comb
import numpy as np

def compositions(d, q):
    if d == 1:
        return [[q]]                            
    result = []                                     
    for i in range(q + 1):
        for tail in compositions(d - 1, q - i):       
            result.append([i] + tail)                
    return result                                      

def forms(d,p):
    result = []
    constant = np.zeros(d)
    for q in range(p):
        result = result + compositions(d, q+1)
    return result

class polynomial:
    def __init__ (self, dim, order):
        self.dim = dim
        self.order = order
        self.coeffs = np.random.randn(comb(dim+order, order)) # polynomial coefficients can take any real

    def eval(self, v):
        result = (self.coeffs)[0]
        counter = 0
        if len(v) == self.dim:
            for term in forms(self.dim,self.order):
                prescaled = 1
                for i in range(self.dim):
                    prescaled *= (v[i]**term[i])
                result += (self.coeffs)[counter+1]*prescaled # counter + 1, because we're shifting , starting from linear term (const. accounted for at start)  
                counter += 1
            return result
        else:
            return "error"
