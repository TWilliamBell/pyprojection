import numpy as np
import scipy.linalg as sci

def projectSchur(A, phi, tol = 0.9):
    """Find a subspace that commutes with A, and which gets close to including
    phi in the subspace.  Return a projector onto the subspace, P, and return P*A.
    tol is a parameter representing how close phi gets to being in the subspace,
    closer to 1 requires more vectors to include."""
    T, Z = sci.schur(A)
    Zt = Z.transpose()
    n = len(Zt)
    phi = phi/np.linalg.norm(phi)
    similarity = [0 for i in range(n)]
    for i in range(n):
        possPhi = Zt[i]
        similarity[i] = abs(np.inner(possPhi, phi))
    proportion = similarity/np.sum(similarity)
    val = 0.
    Zvaluable = list()
    while val < tol:
        i = np.argmax(proportion)
        val+=proportion[i]
        Zvaluable.append(Zt[i])
        proportion[i] = 0.
    Zvaluable = np.array(Zvaluable)
    if len(Zvaluable) == n:
        print("No non-trivial choices found with Schur decomposition.")
        return A
    else:
        PAt = np.zeros([n, n])
        At = np.transpose(A)
        m = len(Zvaluable)
        for i in range(n):
            for j in range(m):
                PAt[i] = PAt[i] + np.inner(Zvaluable[j], At[i])*Zvaluable[j]
        PA = PAt.transpose()
        return PA, m

if __name__ == "__main__":
    A = np.array([[1, 2, 3], [4, 1, 0], [5, -1, 90]])
    phi = [0, 0, 1]
    PA, m = projectSchur(A, phi)
    print(PA)
    print(m)
