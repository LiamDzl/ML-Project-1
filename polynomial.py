from math import comb
import numpy as np


# ----- DYNAMIC PROGRAMMING, ALGORITHMS 101 -----

def compositions(d, q):
    if d == 1:
        return [[q]]                                   # necessary base case, function is recursive
    result = []                                        # we're aiming to produce a VECTOR of VECTORS, all combinations that satisfy constraints. We initialise an empty array
    for i in range(q + 1):
        for tail in compositions(d - 1, q - i):        # start running thru previous LISTS (not just prev.), since they satisfied q - i constraint they'll satisfy this one... 
            result.append([i] + tail)                  # we start with [0, ...]. now this actually needs same q since i = 0, but DIMENSION was done before. So can reuse
    return result                                      

# ----- SUMMARY -----
# ----- think of initialise for loop as asking 'OK, start ALL with vectors of the form [i,....]. 
# We need all the subvectors of 1dim less that sum to q... wait shit ..since we already have a leading i, 
# only need to sum up to q - i !'


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