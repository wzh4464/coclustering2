import numpy as np
import h5py
import scipy.io
from scipy.linalg import svd

def mycluster1(Data, NumKind, maxstep, load_svd=False, svd_file=None):
    DataRow, DataColumn = Data.shape
    Center = np.zeros((DataRow, NumKind))
    B = np.zeros((NumKind, DataColumn))
    COV = np.zeros((DataRow, DataRow, NumKind))
    Y = np.zeros(DataColumn)
    Ynew = np.zeros_like(Y)

    N = DataColumn // NumKind
    if load_svd and svd_file is not None:
        with h5py.File(svd_file, 'r') as file:
            U = np.array(file['U'])
            S = np.array(file['S'])
            V = np.array(file['V'])
    else:
        U, S, V = np.linalg.svd(Data)
    enga = U[:, 0].dot(Data)

    Yenga, I = np.sort(enga), np.argsort(enga)
    for i in range(NumKind):
        Ynew[I[N*i:N*(i+1)]] = i
    Ynew[I[N*(NumKind-1):]] = NumKind - 1

    for step in range(maxstep):
        if np.sum(Ynew != Y) == 0:
            break
        else:
            Y = Ynew.copy()
            for i in range(NumKind):
                Center[:, i] = np.mean(Data[:, Y == i], axis=1)
            for i in range(NumKind):
                COV[:, :, i] = np.cov(Data[:, Y == i])
            B = np.zeros((NumKind, DataColumn))
            for i in range(NumKind):
                for j in range(DataColumn):
                    dist = Data[:, j] - Center[:, i]
                    B[i, j] = multivariate_normal.pdf(Data[:, j], mean=Center[:, i], cov=COV[:, :, i])
            Ynew = np.argmax(B, axis=0)

    return Center, Y, step

def findindex(I, Ngroup):
    Nout = []
    Iout = []
    Jout = []
    N = len(I)
    Nc = np.zeros(N, dtype=int)
    Ic = []
    Jc = []
    a = np.arange(2, Ngroup+1)
    aa = a**2
    Nc[:aa[0]] = a[0]
    for i in range(1, len(a)):
        Ntemp = np.sum(aa[:i])
        Nc[Ntemp:Ntemp+aa[i]] = a[i]
    for i in range(2, len(a)+2):
        for j in range(1, i+1):
            Ic.extend([j]*i)
            Jc.extend(list(range(1, i+1)))
    for i in range(N):
        Nout.append(Nc[I[i]-1])
        Iout.append(Ic[I[i]-1])
        Jout.append(Jc[I[i]-1])
    return np.array(Nout), np.array(Iout), np.array(Jout)

def svdbicluster(data, dim, Num_cluster, read_uv=False):
    scaledata = data
    # dim = 10
    # Num_cluster = 50
    mf, nf = scaledata.shape
    U, S, V = svd(scaledata)
    r = np.linalg.matrix_rank(scaledata)
    min_scale = 10
    uicell = {}
    vicell = {}
    for d in dim:
        if read_uv:
            # read from 'UV.mat'
            with h5py.File('UV.mat', 'r') as f:
                u = f['u'][:]
                v = f['v'][:]
        else:
            u = U[:, :d]
            v = V[:, :d]
        Normdata = []
        # Ngroup = min(np.floor(r/d), np.floor(r/min_scale))
        Ngroup = 5
        indexu = np.zeros((Ngroup, mf), dtype=int)
        indexv = np.zeros((Ngroup, nf), dtype=int)
        for n in range(2, Ngroup):
            pointeru, _ = mycluster1(u.T, n, 100)
            pointerv, _ = mycluster1(v.T, n, 100)
            indexu[n, :] = pointeru
            indexv[n, :] = pointerv
            for i in range(n):
                for j in range(n):
                    Cdata = scaledata[np.ix_(pointeru == i, pointerv == j)]
                    Normdata.append(np.linalg.norm(Cdata - np.mean(Cdata)))
        I = np.argsort(Normdata)
        Nc, Ic, Jc = findindex(I, Ngroup)
        for i in range(Num_cluster):
            ui = np.where(indexu[Nc[i], :] == Ic[i])[0]
            vi = np.where(indexv[Nc[i], :] == Jc[i])[0]
            uicell[i] = ui
            vicell[i] = vi
        # plt.figure()
        # plt.stem(Normdata)
    rowcluster = uicell
    columcluster = vicell
    return rowcluster, columcluster

def main():
    mat = scipy.io.loadmat('A.mat')
    dim = 5
    Num_co_cluster = 8
    data = mat['A']
    rowcluster, columcluster = svdbicluster(data, [dim], Num_co_cluster, read_uv=True)

    print(rowcluster)
    print(columcluster)


if __name__ == '__main__':
    main()