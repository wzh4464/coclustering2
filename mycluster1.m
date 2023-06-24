function [Center, Y, step]=mycluster1(Data,NumKind,maxstep)

[DataRow,DataColumn]=size(Data);
Center = zeros(DataRow,NumKind);%
B=zeros(NumKind,DataColumn); %
COV=zeros(DataRow,DataRow,NumKind);%
Y=zeros(1,DataColumn);%
Ynew=Y;%

N = floor(DataColumn/NumKind);
[U S V] = svd(Data);
enga = U(:,1)'*Data;

[Yenga,I] = sort(enga);
for i=1:NumKind
    Ynew(I([1:N]+N*(i-1)))=i;
end
Ynew(I((1+N*(NumKind-1)):DataColumn))=NumKind;

for step = 1:maxstep
    if sum(Ynew~=Y)==0 break
    else
        Y = Ynew;
        for i=1:NumKind
            NewCenter(:,i)=mean(Data(:,Y==i),2);
        end
        Center=NewCenter;
        for i=1:NumKind
                COV(:,:,i)=cov(Data(:,Y==i)');
        end
        B=zeros(NumKind,DataColumn); 
        for i = 1:NumKind
            for j = 1:DataColumn
                B(i,j)=1/(2*pi*(det(COV(:,:,i)))^0.5)*exp(-0.5*(Data(:,j) ...,
                    -Center(:,i))'*COV(:,:,i)^-1*(Data(:,j)-Center(:,i)));
            end
        end

        [X,Ynew]= max(B);  

    end
end
% 

