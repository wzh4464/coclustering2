import numpy as np

def mycluster1(Data, NumKind):
    DataRow, DataColumn = Data.shape
    Center = np.random.rand(DataRow, NumKind)
    COV = np.zeros((DataRow, DataRow, NumKind))
    for i in range(NumKind):
        COV[:, :, i] = np.eye(DataRow)
    for k in range(10):
        D = np.zeros((NumKind, DataColumn))
        for i in range(NumKind):
            for j in range(DataColumn):
                D[i, j] = np.linalg.norm(Data[:, j] - Center[:, i])
        _, Y = np.min(D, axis=0), np.argmin(D, axis=0)
        for i in range(NumKind):
            Center[:, i] = np.mean(Data[:, Y == i], axis=1)
            COV[:, :, i] = np.cov(Data[:, Y == i])
        B = np.zeros((NumKind, DataColumn))
        for i in range(NumKind):
            for j in range(DataColumn):
                B[i, j] = 1 / (2 * np.pi * np.sqrt(np.linalg.det(COV[:, :, i]))) * \
                          np.exp(-0.5 * (Data[:, j] - Center[:, i]).T @ np.linalg.inv(COV[:, :, i]) @ \
                                 (Data[:, j] - Center[:, i]))
        Ynew = np.argmax(B, axis=0)
    return Ynew

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

def main():
    # 生成一些随机数据
    np.random.seed(0)
    Data = np.random.rand(3, 100)

    # 聚类
    Y = mycluster1(Data, 3)

    # 打印结果
    print(Y)

    # 生成一些随机数据
    np.random.seed(0)
    I = np.array([11, 14, 15, 16, 18, 19, 20, 22, 23, 24, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 12, 3, 9, 8, 6, 5, 4, 13, 10, 25, 17, 29, 21, 1, 7, 2, 39])

    # 查找索引
    Ngroup = 5
    Nout, Iout, Jout = findindex(I, Ngroup)

    # 打印结果
    print("Nout:", Nout)
    print("Iout:", Iout)
    print("Jout:", Jout)

if __name__ == '__main__':
    main()