import numpy as np
import argparse
import math
def read_matrix(matrix_file):
    matrix = []
    with open(matrix_file, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            row = [float(x) for x in line.split()]
            matrix.append(row)
    return np.array(matrix, dtype=np.float64)

# interchange ith row and jth row of A
def interchange(A, i, j):
    n = A.shape[0]
    assert i < n and i >= 0 and j < n and j >= 0, "i or j is not in [0, n)"
    for k in range(n):
        t = A[i, k]
        A[i, k] = A[j, k]
        A[j, k] = t

# find the max abs's index of ith column of A
def find_max(A, i): 
    n = A.shape[0]
    max_n = abs(A[i, i])
    idx = i
    for k in range(i, n):
        if abs(A[k, i]) > max_n:
            max_n = abs(A[k, i])
            idx = k
    return idx

# TypeIII operation
def operate(A, i):
    n = A.shape[0]
    for k in range(i+1, n):
        t = A[k, i]*1.0/A[i, i]
        A[k, i] = t
        for m in range(i+1, n):
            A[k, m] = A[k, m] - t*A[i, m]

def getL(A):
    L = np.array(A)
    n = A.shape[0]
    # set 1s
    for i in range(n):
        L[i, i] = 1
    for i in range(n):
        for j in range(i+1, n):
            L[i, j] = 0
    return L

def getU(A):
    U = np.array(A)
    n = A.shape[0]
    for i in range(n):
        for j in range(i):
            U[i][j] = 0
    return U


def LU_decomposition(A):
    
    # judge if A is square matrix
    assert A.shape[0] == A.shape[1], "A is not square matrix."
    # judge if A is nonsingular
    assert np.linalg.det(A) != 0, "A is singular"
    # n is the dim of matrix A
    n = A.shape[0]
    P = np.eye(n)
    for i in range(n):
        # for ith column, find max index
        idx = find_max(A, i)
        if idx != i:
            # interchange idx-row and ith row
            interchange(A, idx, i)
            # do same to P
            interchange(P, idx, i)
        # then do TypeIII operation
        operate(A, i)
    return getL(A), getU(A), P


def Gram_Schmidt(A):



    Q = np.zeros(A.shape, dtype=np.float64)
    R = np.zeros(A.shape, dtype=np.float64)

    for i in range(A.shape[1]):
        if i == 0:
            v = np.linalg.norm(A[:, i])
            Q[:, i] = A[:, i]/v
            R[i, i] = v
        else:
            q = A[:, i]
            for j in range(i):
                qa = Q[:, j] @ A[:, i]
                R[j, i] = qa 
                q = q - qa * Q[:, j]
            v = np.linalg.norm(q)
            Q[:, i] = q / v 
            R[i, i] = v
    return Q, R

def Householder(A):
    

    
    Rs = []
    B = A
    if A.shape[0] > A.shape[1]:
        it = A.shape[1]
        for i in range(it):
            e1 = np.zeros((A.shape[1]-i, 1))
            e1[0] = 1
            u = B[i:, i] - np.linalg.norm(B[i:, i]) * e1 
            R = np.eye(B.shape[1]-i) - 2 * u @ u.T/(u.T@u)

            _R = np.eye(B.shape[1])
            _R[i:, i:] = R 
            Rs.append(_R)
            B[i:, i:] = R @ B[i:, i:]
    else:
        it = A.shape[0] - 1
        for i in range(it):
            e1 = np.zeros((A.shape[1]-i, 1))
            e1[0] = 1
            u = np.array([B[i:, i]]).T - np.linalg.norm(B[i:, i]) * e1 
            # print('u:')
            # print(u)
            R = np.eye(B.shape[1]-i) - 2 * u @ u.T/(u.T@u)
            # print('R:')
            # print(R)
            _R = np.eye(B.shape[1])
            _R[i:, i:] = R 
            Rs.append(_R)
            B[i:, i:] = R @ B[i:, i:]
            # print('Bii')
            # print(B[i:, i:])
    # for i in range(len(Rs)):
    #     print(Rs[i])
    Q = Rs[0]
    # print('Rs:')
    for i in range(len(Rs) - 1):
        Q = Rs[i+1] @ Q 
    return Q.T, B
    
    

def Givens(A):
    Ps = []
    # P12 P1...n P23
    for j in range(A.shape[1]):
        for i in range(A.shape[0]):
            if i <= j:
                continue
            # Pij
            P = np.eye(A.shape[1])
            c = A[j][j]/math.sqrt(A[j][j]*A[j][j] + A[i][j]*A[i][j])
            s = A[i][j]/math.sqrt(A[j][j]*A[j][j] + A[i][j]*A[i][j])
            P[j][j] = c
            P[i][i] = c
            P[j][i] = s 
            P[i][j] = -s

            Ps.append(P)
            A = P @ A 
    Q = Ps[0]
    for i in range(len(Ps)-1):
        Q = Ps[i+1] @ Q 
    return Q.T, A




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,  choices={'LU', 'QR', 'Householder', 'Givens'},
        dest="mode", help="Matrix decomposition mode: LU/QR/Householder/Givens")
    parser.add_argument("--matrix_file", dest="matrix_file", 
        required=True, help="File that contains matrix values.")
    args = parser.parse_args()
    
    matrix = read_matrix(args.matrix_file)

    if args.mode == "LU":
        L, U, P = LU_decomposition(matrix)
        print('L:')
        print(L)
        print('U:')
        print(U)
        print('P:')
        print(P)
    elif args.mode == 'QR':
        # Gram-Schmit
        Q, R = Gram_Schmidt(matrix)
        print('Q:')
        print(Q)
        print('R:')
        print(R)
    elif args.mode == 'Householder':
        Q, R = Householder(matrix)
        print('Q:')
        print(Q)
        print('R:')
        print(R)
    else:
        # Givens
        Q, R = Givens(matrix)
        print('Q:')
        print(Q)
        print('R:')
        print(R)

