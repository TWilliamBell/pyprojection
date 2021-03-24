import numpy as np
import scipy.linalg as sci

def backwardInvariantProjection(A, phi, diff = 0.01):
    """Find a projection P such that phi is in the image of P and the preimage of A^i for all i is
    in the image of P.  Record the projection P, the norm of the commutator PA-AP, and the basis for
    the projection P.  The argument diff is a tolerance for various quantities that should be zero."""
    n = A.shape[0]
    normPhi = sci.norm(phi)
    if np.abs(normPhi-1) > diff:
        phi = phi/normPhi
    P = np.outer(phi, phi)
    i = 0
    while True:
        i+=1
        ## Find the matrix P*A, the nullspace of P*A
        ## is the complement to the set of vectors that
        ## partially map onto image(P).
        ## On the first iteration, image(P) = span(phi).
        PA = np.matmul(P, A)
        nullPA = sci.null_space(PA)
        nullPA = np.transpose(nullPA)
        ## Q is the projection onto the nullspace of P*A.
        Q = np.zeros([n, n])
        for j in range(len(nullPA)):
            Q = Q+np.outer(nullPA[j], nullPA[j])
        ## newP is the projection complementary to Q, which
        ## is everything that A will map into image(P).
        newP = np.identity(n)-Q
        ## If image(newP) doesn't include phi, then add it to
        ## image(newP).
        Qx = np.matmul(Q, phi)
        normQx = sci.norm(Qx)
        if normQx > diff:
            Qx = Qx/normQx
            newP = newP+np.outer(Qx, Qx)
        ## Is newP a larger subspace than P?  If not then the
        ## iteration has arrived at a fixed point.
        if np.abs(np.trace(newP)-np.trace(P)) < 0.5:
            #print(np.linalg.norm(P-newP))
            break
        else:
            ## If not then update P.
            P = newP
        if i > n:
            P = np.identity(n)
            break
    ## Find a new basis for image(P).
    basis = np.transpose(sci.orth(P))
    ## Return results.
    return P, np.linalg.norm(np.matmul(P, A)-np.matmul(A, P)), basis

if __name__ == "__main__":
    A = np.array([[2, 0, 0, 0, 0],
              [0, 2, 0.5, 0, 0],
              [0, -1, 2, 0, 0],
              [0, 0, 0, 1, 1],
              [0, 0, 0, 0, 1]])

    print(A)

    P, commutatorNorm, basis = backwardInvariantProjection(A, phi = np.array([0, 1, 0, 0, 0]))
    print(P)
    print(commutatorNorm)
    print(basis)

    # P, commutatorNorm, basis = backwardInvariantProjection(A, phi = np.array([1, 0, 0, 0, 0]))
    # print(P.round())
    # print(commutatorNorm)
    #
    # P, commutatorNorm, basis = backwardInvariantProjection(A, phi = np.array([0, 0, 0, 1, 0]))
    # print(P.round())
    # print(commutatorNorm)
    #
    # P, commutatorNorm, basis = backwardInvariantProjection(A, phi = np.array([0, 0, 1, 1, 0]))
    # print(P.round())
    # print(commutatorNorm)
    #
    # P, commutatorNorm, basis = backwardInvariantProjection(A, phi = np.array([1, 1, 1, 1, 1]))
    # print(P.round())
    # print(commutatorNorm)
