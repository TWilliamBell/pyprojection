import numpy as np
import scipy.stats

np.random.seed(1)

def projectionCommute(A, phi, eps = 1e-10):
    """"This is my attempt to code up the algorithm for finding a projection that includes phi in its
    image and which commutes with A.  A is a n*n matrix, and phi is an n-dimensional vector, eps
    is the required tolerance on the commutator, which tells you whether A and P approximately commute."""
    n = A.shape[0] ## Assumes square matrix
    Amoorepenrose = np.linalg.pinv(A).transpose() ## The transpose is used to find the directions that
    ## would contribute but to our phi but are in the kernel
    normPhi = np.linalg.norm(phi, 2)
    if np.abs(normPhi-1.) > 1e-4: ## Normalize Phi
        phi = phi/normPhi
    P = np.outer(phi, phi) ## Construct first iterate for P
    i = 1 ## Count dim(im(P))
    j = 1
    phiItBack = phi
    phiItFwd = phi
    commutatorNorm = np.linalg.norm(np.matmul(P, A)-np.matmul(A, P), 2) ## Compute the size of the
    ## commutator
    fwdDoesntMatter = False
    bwdDoesntMatter = False
    while commutatorNorm > eps and not i == n and not j == n:
        ## First do the backwards direction of adding vectors to the invariant space.
        if not bwdDoesntMatter:
            phiItBack = np.matmul(Amoorepenrose, phiItBack)
            normPhi = np.linalg.norm(phiItBack)
            if normPhi < 1e-10:
                bwdDoesntMatter = True
            else:
                phiItBack = phiItBack/normPhi
                phiItBack = phiItBack-np.matmul(P, phiItBack)
                normPhi = np.linalg.norm(phiItBack, 2)
            if normPhi < 1e-10: ## Backward direction is now entirely within im(P)
                bwdDoesntMatter = True
            else:
                phiItBack = phiItBack/normPhi
                P = P+np.outer(phiItBack, phiItBack)
                commutatorNorm = np.linalg.norm(np.matmul(P, A)-np.matmul(A, P), 2)
                i = i+1
            if commutatorNorm < eps or i == n:
                break
        ## Second, do the forward direction of adding vectors to the invariant space.
        if not fwdDoesntMatter:
            phiItFwd = np.matmul(A, phiItFwd)
            phiItFwd = phiItFwd/np.linalg.norm(phiItFwd)
            phiItFwd = phiItFwd-np.matmul(P, phiItFwd)
            normPhi = np.linalg.norm(phiItFwd, 2)
            if normPhi < 1e-10: ## Forward direction is now entirely within im(P)
                fwdDoesntMatter = True
            else:
                phiItFwd = phiItFwd/normPhi
                P = P+np.outer(phiItFwd, phiItFwd)
                commutatorNorm = np.linalg.norm(np.matmul(P, A)-np.matmul(A, P), 2)
                i = i + 1
            if commutatorNorm < eps or i == n:
                break
        j = j+1
        if fwdDoesntMatter and bwdDoesntMatter:
            print("Projection may not commute well, norm of the commutator is "+
                  str(commutatorNorm)+".")
            commutator = np.matmul(P, A) - np.matmul(A, P)
            return P, commutatorNorm, commutator
    if i == n or j == n:
        print("No non-trivial subspaces found containing vector of interest.")
        return np.identity(n), 0., np.zeros([n, n])
    commutator = np.matmul(P, A) - np.matmul(A, P)
    return P, commutatorNorm, commutator

# A = np.array([[2, 0, 0, 0, 0],
#               [0, 2, 0.5, 0, 0],
#               [0, -1, 2, 0, 0],
#               [0, 0, 0, 1, 1],
#               [0, 0, 0, 0, 1]])
# U = np.array(scipy.stats.ortho_group.rvs(5))
# Ut = U.transpose()
# A = np.matmul(U, np.matmul(A, Ut))
#
# print(A)
# #print(np.linalg.pinv(A).transpose())
#
# P, commutatorNorm, commutator = projectionCommute(A, phi = np.array([0, 1, 0, 0, 0]))
# print(P)
# print(commutatorNorm)
# #print(commutator)
#
# P, commutatorNorm, commutator = projectionCommute(A, phi = np.array([1, 0, 0, 0, 0]))
# print(P.round())
# print(commutatorNorm)
#
# P, commutatorNorm, commutator = projectionCommute(A, phi = np.array([0, 0, 0, 1, 0]))
# print(P.round())
# print(commutatorNorm)
#
# P, commutatorNorm, commutator = projectionCommute(A, phi = np.array([0, 0, 1, 1, 0]))
# print(P.round())
# print(commutatorNorm)
#
# P, commutatorNorm, commutator = projectionCommute(A, phi = np.array([1, 1, 1, 1, 1]))
# print(P.round())
# print(commutatorNorm)
