clear
clc
close all
addpath data utilities
load('AR_DAT.mat');

par.nClass        =   length(unique(trainlabels));
dim=[600];
Tr_DAT   =   double(NewTrain_DAT(:,trainlabels<=par.nClass));
trls     =   trainlabels(trainlabels<=par.nClass);
Tt_DAT   =   double(NewTest_DAT(:,testlabels<=par.nClass));
ttls     =   testlabels(testlabels<=par.nClass);
clear NewTest_DAT NewTrain_DAT testlabels trainlabels

train_tol= size(Tr_DAT,2);
test_tol = size(Tt_DAT,2);
ClassNum = par.nClass;
% diff_lambda=[0 0.00001]
diff_lambda = [0.00001] % ²ÎÊýlambda
reg_rate = zeros(1,length(diff_lambda));

for eigen_num=dim
    kk = 1;
    eigen_num
    %eigenface extracting
    [disc_set,disc_value,Mean_Image]  =  Eigenface_f(Tr_DAT,eigen_num);
    tr_dat  =  disc_set'*Tr_DAT;
    tt_dat  =  disc_set'*Tt_DAT;
    tr_dat = normc(tr_dat);
    tt_dat = normc(tt_dat);

    for lambda=diff_lambda
        lambda
        ID = zeros(1,test_tol);
        X = tr_dat;
        [~,n] = size(X);
        XTX = X'*X;

        M = [];
        for i = 1 : ClassNum
            index = find(trls == i);
            Xi = X(:, index);
            temp = Xi'* Xi;
            M = blkdiag(M, temp);
        end
        mu= 1e-1;
        c_tmp=XTX+1/2*mu*eye(n)+2*lambda*(ClassNum*M+XTX);
        tic;
        for i=1:test_tol
            y = tt_dat(:,i);
            XTy = X'*y;
            [z,c] = DNRC(mu,n, XTy,c_tmp);
            W = sparse([],[],[],train_tol,ClassNum,length(c));

            for j=1:ClassNum
                ind = (j==trls);
                W(ind,j) = c(ind);
            end

            temp = X*W-repmat(y,1,ClassNum);
            residual = sqrt(sum(temp.^2));

            [~,index]=min(residual);
            ID(i)=index;
        end
        
        cornum      =   sum(ID==ttls);
        reg_rate(kk)         =   cornum/length(ttls); % recognition rate
        kk = kk+1;
        toc
    end
    reg_rate
end