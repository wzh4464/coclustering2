% FILEPATH: /Users/zihanwu/codes/coclustering2/mycluster1.m
% MYCLUSTER1 performs co-clustering on the input data using the specified number of clusters.
%
% INPUTS:
% Data - input data matrix
% NumKind - number of clusters to form
% maxstep - maximum number of iterations to perform
%
% OUTPUTS:
% Center - final cluster centers
% Y - final cluster assignments for each data point
% step - number of iterations performed

function [Center, Y, step] = mycluster1(Data, NumKind, maxstep)

    [DataRow, DataColumn] = size(Data);
    Center = zeros(DataRow, NumKind);
    B = zeros(NumKind, DataColumn);
    COV = zeros(DataRow, DataRow, NumKind);
    Y = zeros(1, DataColumn);
    Ynew = Y;

    % N = floor(DataColumn/NumKind) calculates the number of data points to assign to each cluster.
    N = floor(DataColumn / NumKind);

    [U, ~, ~] = svd(Data);

    % Computes the projection of the input data matrix onto the first principal component.
    enga = U(:, 1)' * Data;

    % Sorts the projection of the input data matrix onto the first principal component in ascending order and returns the sorted indices.
    [~, I] = sort(enga);

    % This loop assigns the remaining data points to the clusters based on the sorted indices of the projection of the input data matrix onto the first principal component.
    % It iterates over the number of clusters and assigns the next N data points to each cluster.
    % 这个循环根据将输入数据矩阵投影到第一个主成分的排序索引，将剩余的数据点分配给聚类。
    % 它遍历聚类数量，并将接下来的N个数据点分配给每个聚类。
    for i = 1:NumKind
        Ynew(I([1:N] + N * (i - 1))) = i;
    end

    % Assign the last few data points to the last cluster.
    % Avoiding the case where the number of data points is not divisible by the number of clusters.
    Ynew(I((1 + N * (NumKind - 1)):DataColumn)) = NumKind;

    for step = 1:maxstep

        if sum(Ynew ~= Y) == 0
            % If the cluster assignments do not change, then
            break
            % stop the iteration.
        else
            % Otherwise, update the cluster assignments.
            Y = Ynew;
            % Update the cluster centers.
            for i = 1:NumKind
                NewCenter(:, i) = mean(Data(:, Y == i), 2);
                % Update the cluster centers with the mean of the data points assigned to each cluster.
            end

            Center = NewCenter;

            for i = 1:NumKind
                COV(:, :, i) = cov(Data(:, Y == i)');
                % Update the covariance matrix of each cluster.
            end

            B = zeros(NumKind, DataColumn);
            % B is the probability of each data point belonging to each cluster.

            for i = 1:NumKind
                for j = 1:DataColumn
                    B(i, j) = 1 / (2 * pi * (det(COV(:, :, i))) ^ 0.5) * exp(-0.5 * (Data(:, j) ... ,
                        -Center(:, i))' * COV(:, :, i) ^ -1 * (Data(:, j) - Center(:, i)));
                    % Update the probability of each data point belonging to each cluster.
                    % Note Data(:,j)-Center(:,i) as d_ij.
                    % B(i,j)=\frac{1}{\sqrt{2\pi}\sqrt{det(Cov_i)}}exp(-\frac{1}{2}d_{ij}^T*Cov_i^{-1}*d_{ij})
                end
            end
            % Assign each data point to the cluster with the highest probability.
            [X, Ynew] = max(B);
        end
    end
