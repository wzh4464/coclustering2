import numpy as np
import h5py
import scipy.io
from scipy.linalg import svd
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from readjson import process_json_file
# Word2Vec
from gensim.models import Word2Vec
import time

def read_weibo(file_path="微博情感分析评测/样例数据/labelled dataset/ipad.xml"):
    """
    Read the XML file and return a list of dictionaries.
    读取XML文件并返回字典列表。
    """
    # 解析 XML 文件
    tree = ET.parse(file_path)
    root = tree.getroot()

    # 遍历微博元素
    weibos = []
    for weibo in root.findall("weibo"):
        weibo_id = weibo.get("id")
        sentences = []
        # 遍历句子元素
        for sentence in weibo.findall("sentence"):
            sentence_id = sentence.get("id")
            opinionated = sentence.get("opinionated")
            polarity = sentence.get("polarity")
            target_word_1 = sentence.get("target_word_1")
            target_begin_1 = sentence.get("target_begin_1")
            target_end_1 = sentence.get("target_end_1")
            target_polarity_1 = sentence.get("target_polarity_1")
            text = sentence.text.strip()
            # 将句子信息存储到字典中
            sentence_dict = {
                "id": sentence_id,
                "opinionated": opinionated,
                "polarity": polarity,
                "target_word_1": target_word_1,
                "target_begin_1": target_begin_1,
                "target_end_1": target_end_1,
                "target_polarity_1": target_polarity_1,
                "text": text,
            }
            sentences.append(sentence_dict)
        # 将微博信息存储到字典中
        weibo_dict = {"id": weibo_id, "sentences": sentences}
        weibos.append(weibo_dict)
    return weibos


def mycluster1(Data, NumKind, maxstep, read_uv_from=None):
    """
    ! Output Y is labeled from 0 to NumKind-1, not from 1 to NumKind.
    Given an array of data points, the number of clusters, and the maximum number of iterations,
    returns the cluster centers, the cluster assignments, and the number of iterations.

    Args:
    - Data (numpy.ndarray): Array of data points.
    - NumKind (int): Number of clusters.
    - maxstep (int): Maximum number of iterations.
    - read_uv_from (str): Path to the file to read the first principal component from.

    Returns:
    - Center (numpy.ndarray): Array of cluster centers.
    - Y (numpy.ndarray): Array of cluster assignments.
    - step (int): Number of iterations.
    """
    DataRow, DataColumn = Data.shape
    Center = np.zeros((DataRow, NumKind))
    B = np.zeros((NumKind, DataColumn))
    COV = np.zeros((DataRow, DataRow, NumKind))
    Y = np.zeros(DataColumn, dtype=int)
    Ynew = Y.copy()

    # N = DataColumn // NumKind calculates the number of data points to assign to each cluster.
    N = DataColumn // NumKind

    if read_uv_from is not None:
        with h5py.File(read_uv_from, "r") as f:
            U = f["U"][:].T
    else:
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
        Ynew[I[N * i : N * (i + 1)]] = i

    # Assign the last few data points to the last cluster.
    # Avoiding the case where the number of data points is not divisible by the number of clusters.
    Ynew[I[N * (NumKind - 1) :]] = NumKind - 1

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
                NewCenter = np.mean(Data[:, Y == i], axis=1)
                # Update the cluster centers with the mean of the data points assigned to each cluster.
                Center[:, i] = NewCenter

            for i in range(NumKind):
                COV[:, :, i] = np.cov(Data[:, Y == i], rowvar=True)
                # Update the covariance matrix of each cluster.

            B = np.zeros((NumKind, DataColumn))
            # B is the probability of each data point belonging to each cluster.

            for i in range(NumKind):
                for j in range(DataColumn):
                    d_ij = Data[:, j] - Center[:, i]
                    try:
                        B[i, j] = (
                            1
                            / (2 * np.pi * np.sqrt(np.linalg.det(COV[:, :, i])))
                            * np.exp(-0.5 * d_ij.T @ np.linalg.inv(COV[:, :, i]) @ d_ij)
                        )
                    except np.linalg.LinAlgError:
                        B[i, j] = np.nan
                    # Update the probability of each data point belonging to each cluster.
                    # Note Data[:,j]-Center[:,i] as d_ij.
                    # B(i,j)=\frac{1}{\sqrt{2\pi}\sqrt{det(Cov_i)}}exp(-\frac{1}{2}d_{ij}^T*Cov_i^{-1}*d_{ij})

            # Assign each data point to the cluster with the highest probability.
            Ynew = np.argmax(B, axis=0)

    return Center, Y, step + 1


def findindex(I, Ngroup):
    """
    Given an array of indices I and the number of groups Ngroup, returns the number of elements in each group,
    the row index of each element, and the column index of each element. The number of groups is determined by
    the formula a^2, where a is an array of integers from 2 to Ngroup.
    给定索引数组I和组数Ngroup，返回每个组中元素的数量，每个元素的行索引和每个元素的列索引。组数由公式a^2确定，其中a是从2到Ngroup的整数数组。

    Args:
    - I (numpy.ndarray): Array of indices.
    - Ngroup (int): Number of groups.

    Returns:
    - Nout (numpy.ndarray): Array of the number of elements in each group.
    - Iout (numpy.ndarray): Array of the row index of each element.
    - Jout (numpy.ndarray): Array of the column index of each element.
    """
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
    # 计算每个簇的大小
    for i in range(1, len(a)):
        Ntemp = np.sum(aa[:i])
        Nc[Ntemp : Ntemp + aa[i]] = a[i]
    # 计算每个簇的行和列
    for i in range(2, len(a) + 2):
        for j in range(1, i + 1):
            Ic.extend([j] * i)
            Jc.extend(list(range(1, i + 1)))
    # 将簇的大小、行和列映射到输出向量中
    for i in range(N):
        Nout.append(Nc[I[i]])
        Iout.append(Ic[I[i]])
        Jout.append(Jc[I[i]])
    return np.array(Nout), np.array(Iout), np.array(Jout)


def svdbicluster(data, dim, Num_cluster, read_uv=False):
    scaledata = data
    # dim = 10
    # Num_cluster = 50
    mf, nf = scaledata.shape
    U, _, V = svd(scaledata)
    uicell = {}
    vicell = {}
    for d in dim:
        if read_uv:
            # read from 'UV.mat'
            with h5py.File("UV.mat", "r") as f:
                # 用h5py读取mat文件时，得到的矩阵维度顺序相反
                # u = f["u"][:]
                # v = f["v"][:]
                u = f["u"][:].T
                v = f["v"][:].T
        else:
            u = U[:, :d]
            v = V.T[:, :d]
        Normdata = []
        # Ngroup = min(np.floor(r/d), np.floor(r/min_scale))
        Ngroup = 5
        indexu = np.zeros((Ngroup, mf), dtype=int)
        indexv = np.zeros((Ngroup, nf), dtype=int)
        for n in range(2, Ngroup + 1):
            # if n == 2:
            #     _, pointeru, _ = mycluster1(u.T, n, 100, read_uv_from="u2.mat")
            # else:
            _, pointeru, _ = mycluster1(u.T, n, 100)
            _, pointerv, _ = mycluster1(v.T, n, 100)
            indexu[n - 1, :] = pointeru
            indexv[n - 1, :] = pointerv
            for i in range(n):
                for j in range(n):
                    Cdata = scaledata[np.ix_(pointeru == i, pointerv == j)]
                    if Cdata.size == 0:
                        Normdata.append(0)
                    else:
                        Normdata.append(np.linalg.norm(Cdata - np.mean(Cdata)))

        # ascending sort
        I = np.argsort(Normdata)
        Nc, Ic, Jc = findindex(I, Ngroup)
        for i in range(Num_cluster):
            ui = np.where(indexu[Nc[i] - 1, :] == Ic[i])[0]
            vi = np.where(indexv[Nc[i] - 1, :] == Jc[i])[0]
            uicell[i] = ui
            vicell[i] = vi
        plt.figure()
        plt.stem(Normdata)
    rowcluster = uicell
    columcluster = vicell
    return rowcluster, columcluster


def test_cocluster():
    ## test whole
    mat = scipy.io.loadmat("A.mat")
    dim = 5
    Num_co_cluster = 8
    data = mat["A"]
    rowcluster, columcluster = svdbicluster(data, [dim], Num_co_cluster)

    print(rowcluster)
    print(columcluster)


def test_findindex():
    Ngroup = 5
    Nc, Ic, Jc = findindex(I, Ngroup)
    # show result: (Nc, Ic, Jc) formatly and their size
    print("Nc: ", Nc)
    print("Nc size: ", Nc.size)
    print("Ic: ", Ic)
    print("Ic size: ", Ic.size)
    print("Jc: ", Jc)
    print("Jc size: ", Jc.size)
    
def generate_ngrams(text, n = 2):
    """
    生成n-gram序列
    参数:
        text (str): 输入文本
        n (int): n-gram中的n值
    返回:
        ngrams (list): 包含n-gram序列的列表
    """
    words = text.split()
    ngrams = []
    for i in range(len(words) - n + 1):
        ngrams.append(words[i:i+n])
    return ngrams

def main():
    start = time.time()
    # sentences = process_json_file("raw/ultrachat_release_230407.json")
    sentences = process_json_file("raw/first_1000.json")
    print("process_json_file time: ", time.time() - start)
    print(len(sentences))

    # n-gram
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    print("vectorizer time: ", time.time() - start)
    
    vectorized_sentences = vectorizer.fit_transform(sentences)
    print("vectorized_sentences time: ", time.time() - start)
    print("vectorized_sentences.shape: ", vectorized_sentences.shape)
    
    # count non zero elements
    print(vectorized_sentences.nnz)
    
    # extract sparse indices
    sentences_indices = vectorized_sentences.tocoo()
    print("sentences_indices time: ", time.time() - start)
    print("sentences_indices", sentences_indices)
    
    # extract first 2000 columns
    first_2000_indices = sentences_indices.col < 2000
    
    # to dense vectors
    dense_vectors = np.zeros((len(sentences), 2000))
    for i, j, v in zip(sentences_indices.row, sentences_indices.col, sentences_indices.data):
        if j < 2000:
            dense_vectors[i, j] = v
    
    # dense_vectors = vectorized_sentences.toarray()
    # dense_vectors = vectorized_sentences[:, first_2000_indices].toarray()
    print("dense_vectors time: ", time.time() - start)
    print("dense_vectors shape: ", dense_vectors.shape)
    
    # export dense_vectors to mat file that MATLAB can read
    scipy.io.savemat("dense_vectors.mat", {"dense_vectors": dense_vectors})
    print("savemat time: ", time.time() - start)
    sentences = process_json_file("raw/ultrachat_release_230407.json")
    # sentences = process_json_file("raw/first_100.json")
    # print(sentences)

    # n-gram vectorizer
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    vectorized_sentences = vectorizer.fit_transform(sentences)
    dence_vectors = vectorized_sentences.toarray()
    
    print(dence_vectors.shape)

    # TF vectorizer
    # vectorizer = TfidfVectorizer()
    # vectorized_sentences = vectorizer.fit_transform(sentences)
    # dense_vectors = vectorized_sentences.toarray()
    
    U, S, V = svd(dense_vectors)
    print("svd time: ", time.time() - start)
    
    plt.plot(S)
    plt.show()
    
    
    # word embedding
    # word2vec = Word2Vec(sentences, window=5, min_count=1, workers=4)
    # # get embedding matrix
    # embedding_matrix = word2vec.wv.vectors
    # # show embedding matrix
    # print(embedding_matrix)
    
    
    
    
    # similarity_matrix = cosine_similarity(dense_vectors)
    # rowcluster, columcluster = svdbicluster(similarity_matrix, [2:8], 50)
    # rowcluster, columcluster = svdbicluster(similarity_matrix, [2, 3, 4, 5, 6, 7], 50)
    # rowcluster, columcluster = svdbicluster(dense_vectors, [2, 3, 4, 5, 6, 7], 50)
    # print(rowcluster)
    # print(columcluster)
    
    # print(sentences)
    # print sentences line by line
    # i = 0
    # for sentence in sentences:
    #     # with line number
    #     print(i, sentence)
    #     i += 1
        
    # print these sentences: [  1,   8,  21,  65,  74, 107, 110, 131, 132, 134, 136, 150, 169, 184, 192, 197, 198, 200, 218] with a star in the front, and print the others normally
    # for sentence in sentences:
    #     if i in [  1,   8,  21,  65,  74, 107, 110, 131, 132, 134, 136, 150, 169, 184, 192, 197, 198, 200, 218]:
    #         print("*", sentence)
    #     else:
    #         print(sentence)
    #     i += 1
    
    # print these sentences: [  1,   8,  21,  65,  74, 107, 110, 131, 132, 134, 136, 150, 169, 184, 192, 197, 198, 200, 218] in red, and print the others normally 
    # for sentence in sentences:
    #     if i in [  1,   8,  21,  65,  74, 107, 110, 131, 132, 134, 136, 150, 169, 184, 192, 197, 198, 200, 218]:
    #         print("\033[1;31m", sentence, "\033[0m")
    #     else:
    #         print(sentence)
    #     i += 1
        
    # print to a file with color
    # with open("sentences.md", "w") as f:
    #     for sentence in sentences:
    #         if i in [  1,   8,  21,  65,  74, 107, 110, 131, 132, 134, 136, 150, 169, 184, 192, 197, 198, 200, 218]:
    #             f.write("**" + sentence + "**\r\n\n")
    #         else:
    #             f.write(sentence + "\r\n\n")
    #         i += 1
    print("Done!")


if __name__ == "__main__":
    main()
    
