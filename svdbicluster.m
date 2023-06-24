
function [rowcluster,columcluster] = svdbicluster(data,dim,Num_cluster)

scaledata = data;
%dim = 10;
%Num_cluster = 50;

[mf nf] = size(scaledata);
[U S V]=svd(scaledata);
r = rank(scaledata);

min_scale = 10;
uicell = {};
vicell = {};

for d = dim  
    u = U(:,1:d);
    v = V(:,1:d);
    Normdata = [];
    %Ngroup = min(floor(r/d), floor(r/min_scale));
    Ngroup=5;
    indexu = zeros(Ngroup,mf);
    indexv = zeros(Ngroup,nf);
    for n = 2:Ngroup
        [Centeru, pointeru, stepu]=mycluster1(u',n,100);
        [Centerv, pointerv, stepv]=mycluster1(v',n,100);
        indexu(n,:) = pointeru;
        indexv(n,:) = pointerv;
        for i = 1:n
            for j = 1:n
                Cdata = scaledata(find(pointeru==i), find(pointerv==j));
                Normdata = [Normdata norm(Cdata - mean(mean(Cdata)))];
            end
        end
    end
    [Yenga,I] = sort(Normdata,'ascend');
    [Nc Ic Jc] = findindex(I,Ngroup);
    for i = 1:Num_cluster
        ui = find(indexu(Nc(i),:) == Ic(i));
        vi = find(indexv(Nc(i),:) == Jc(i));
        uicell{i} = ui;% biclusters row
        vicell{i} = vi;% biclusters column
    end
    figure
    stem(Normdata)
end

rowcluster = uicell;
columcluster = vicell;

end



