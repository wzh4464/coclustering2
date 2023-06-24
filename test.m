
load('A.mat');
dim = 5;
Num_co_cluster = 8;
data = A;
[rowcluster,columcluster] = svdbicluster(data,dim,Num_co_cluster);

%%% first cocluster is rowcluster{1}columcluster{1}
%%% second cocluster is rowcluster{2}columcluster{2}
% 11and so on

%% 
% count the number of rowcluster for each cell and add up to know how many elements in all coclusters
lines = 0;
for i = 1:Num_co_cluster
    lines = lines + length(rowcluster{i});
end
print(lines);