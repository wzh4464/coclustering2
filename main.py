import numpy as np
import h5py
import scipy.io
from scipy.linalg import svd


def mycluster1(Data, NumKind, maxstep):
    DataRow, DataColumn = Data.shape
    Center = np.zeros((DataRow, NumKind))
    B = np.zeros((NumKind, DataColumn))
    COV = np.zeros((DataRow, DataRow, NumKind))
    Y = np.zeros(DataColumn, dtype=int)
    Ynew = Y.copy()

    # N = DataColumn // NumKind calculates the number of data points to assign to each cluster.
    N = DataColumn // NumKind

    U, _, _ = np.linalg.svd(Data)

    # Computes the projection of the input data matrix onto the first principal component.
    enga = U[:, 0].T @ Data

    # Sorts the projection of the input data matrix onto the first principal component in ascending order and returns the sorted indices.
    I = np.argsort(enga)

    # This loop assigns the remaining data points to the clusters based on the sorted indices of the projection of the input data matrix onto the first principal component.
    # It iterates over the number of clusters and assigns the next N data points to each cluster.
    # 这个循环根据将输入数据矩阵投影到第一个主成分的排序索引，将剩余的数据点分配给聚类。
    # 它遍历聚类数量，并将接下来的N个数据点分配给每个聚类。
    for i in range(NumKind):
        Ynew[I[N * i : N * (i + 1)]] = i + 1

    # Assign the last few data points to the last cluster.
    # Avoiding the case where the number of data points is not divisible by the number of clusters.
    Ynew[I[N * (NumKind - 1) :]] = NumKind

    for step in range(maxstep):
        if np.sum(Ynew != Y) == 0:
            # If the cluster assignments do not change, then
            break
            # stop the iteration.
        else:
            # Otherwise, update the cluster assignments.
            Y = Ynew.copy()
            # Update the cluster centers.
            for i in range(NumKind):
                NewCenter = np.mean(Data[:, Y == i + 1], axis=1)
                # Update the cluster centers with the mean of the data points assigned to each cluster.
                Center[:, i] = NewCenter

            for i in range(NumKind):
                COV[:, :, i] = np.cov(Data[:, Y == i + 1], rowvar=True)
                # Update the covariance matrix of each cluster.

            B = np.zeros((NumKind, DataColumn))
            # B is the probability of each data point belonging to each cluster.

            for i in range(NumKind):
                for j in range(DataColumn):
                    d_ij = Data[:, j] - Center[:, i]
                    B[i, j] = (
                        1
                        / (2 * np.pi * np.sqrt(np.linalg.det(COV[:, :, i])))
                        * np.exp(-0.5 * d_ij.T @ np.linalg.inv(COV[:, :, i]) @ d_ij)
                    )
                    # Update the probability of each data point belonging to each cluster.
                    # Note Data[:,j]-Center[:,i] as d_ij.
                    # B(i,j)=\frac{1}{\sqrt{2\pi}\sqrt{det(Cov_i)}}exp(-\frac{1}{2}d_{ij}^T*Cov_i^{-1}*d_{ij})

            # Assign each data point to the cluster with the highest probability.
            Ynew = np.argmax(B, axis=0) + 1

    return Center, Y, step + 1


def findindex(I, Ngroup):
    Nout = []
    Iout = []
    Jout = []
    N = len(I)
    Nc = np.zeros(N, dtype=int)
    Ic = []
    Jc = []
    a = np.arange(2, Ngroup + 1)
    aa = a**2
    Nc[: aa[0]] = a[0]
    for i in range(1, len(a)):
        Ntemp = np.sum(aa[:i])
        Nc[Ntemp : Ntemp + aa[i]] = a[i]
    for i in range(2, len(a) + 2):
        for j in range(1, i + 1):
            Ic.extend([j] * i)
            Jc.extend(list(range(1, i + 1)))
    for i in range(N):
        Nout.append(Nc[I[i] - 1])
        Iout.append(Ic[I[i] - 1])
        Jout.append(Jc[I[i] - 1])
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
            with h5py.File("UV.mat", "r") as f:
                u = f["u"][:]
                v = f["v"][:]
        else:
            u = U[:, :d]
            v = V[:, :d]
        Normdata = []
        # Ngroup = min(np.floor(r/d), np.floor(r/min_scale))
        Ngroup = 5
        indexu = np.zeros((Ngroup, mf), dtype=int)
        indexv = np.zeros((Ngroup, nf), dtype=int)
        for n in range(2, Ngroup):
            pointeru, _, _ = mycluster1(u.T, n, 100)
            pointerv, _, _ = mycluster1(v.T, n, 100)
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
    mat = scipy.io.loadmat("A.mat")
    dim = 5
    Num_co_cluster = 8
    data = mat["A"]
    rowcluster, columcluster = svdbicluster(data, [dim], Num_co_cluster, read_uv=True)

    print(rowcluster)
    print(columcluster)


if __name__ == "__main__":
    main()
