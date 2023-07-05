
load('A.mat');
dim = 5;
Num_co_cluster = 8;
data = A;
[rowcluster,columcluster] = svdbicluster(data,dim,Num_co_cluster);

%%% first cocluster is rowcluster{1}columcluster{1}
%%% second cocluster is rowcluster{2}columcluster{2}
% 11and so on
