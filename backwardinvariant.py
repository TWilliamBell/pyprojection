import numpy as np
import scipy.linalg as sci

def backwardInvariantProjection(A, phi, diff = 0.01):
    n = A.shape[0]
    normPhi = sci.norm(phi)
    if np.abs(normPhi-1) > diff:
        phi = phi/normPhi
    P = np.outer(phi, phi)
    i = 0
    while True:
        i+=1
        PA = np.matmul(P, A)
        nullPA = sci.null_space(PA)
        nullPA = np.transpose(nullPA)
        Q = np.zeros([n, n])
        for j in range(len(nullPA)):
            Q = Q+np.outer(nullPA[j], nullPA[j])
        newP = np.identity(n)-Q
        Qx = np.matmul(Q, phi)
        normQx = sci.norm(Qx)
        if normQx > diff:
            Qx = Qx/normQx
            newP = newP+np.outer(Qx, Qx)
        if np.abs(np.trace(newP)-np.trace(P)) < 0.5:
            #print(np.linalg.norm(P-newP))
            break
        else:
            P = newP
        if i > n:
            P = np.identity(n)
            break
    basis = np.transpose(sci.orth(P))
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