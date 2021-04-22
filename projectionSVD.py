import numpy as np
import scipy.linalg as sci

def projectionSVD(A, phi, eps1 = 1e-3, eps2 = 1e-3, eps3 = 1e-3):
    """Construct a projection that includes most of the stuff
    being sent to phi."""
    normPhi = np.linalg.norm(phi)
    if np.abs(normPhi-1) > eps2:
        phi = phi/normPhi
    U, S, Vh = np.linalg.svd(A)
    P = np.outer(phi, phi)
    n = len(phi)
    tol = 1.-eps1
    for k in range(n):
        PU = np.matmul(P, U)
        sizePU = [np.linalg.norm(PU[:, i]) for i in range(n)]
        ## The key idea
        weightedSize = np.abs([sizePU[i]*S[i] for i in range(n)])
        total = np.sum(weightedSize)
        if total == 0.:
            return np.outer(phi, phi), phi
        sortedWeightedSize = np.sort(weightedSize)[::-1]
        subtotal = 0.
        j = 0
        smallest = 0.0
        while subtotal/total < tol:
            subtotal = subtotal+sortedWeightedSize[j]
            smallest = sortedWeightedSize[j]
            j+=1
        includedVecs = weightedSize >= smallest
        vStart = Vh[includedVecs, :]
        sizeVStart = np.sum(includedVecs)
        newP = np.zeros((n, n))
        for i in range(sizeVStart):
            newP = newP+np.outer(vStart[i, :], vStart[i, :])
        extraPhi = phi-np.matmul(P, phi)
        normExtraPhi = np.linalg.norm(extraPhi)
        if normExtraPhi > eps2:
            extraPhi = extraPhi/normExtraPhi
            newP = newP+np.outer(extraPhi, extraPhi)
        if np.linalg.norm(newP-P) < eps3:
            P = newP
            break
        else:
            P = newP
    return P, np.transpose(sci.orth(P))

if __name__ == "__main__":
    A = np.array([[2, 0, 0, 0, 0],
              [0, 2, 0.5, 0, 0],
              [0, -1, 2, 0, 0],
              [0, 0, 0, 1, 1],
              [0, 0, 0, 0, 1]])

    print(A)

    P, basis = projectionSVD(A, phi = np.array([0, 0, 1, 0, 0]))
    print(basis)