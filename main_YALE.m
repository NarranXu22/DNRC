clear
clc
close all
addpath data utilities
load('Yale_54_48.mat');
par.nClass        =   length(unique(trainlabels));
dim = [540]
Tr_DAT   =   double(NewTrain_DAT(:,trainlabels<=par.nClass ));
trls     =   trainlabels(trainlabels<=par.nClass );
Tt_DAT   =   double(NewTest_DAT(:,testlabels<=par.nClass   ));
ttls     =   testlabels(testlabels<=par.nClass );
clear NewTrain_DAT trainlabels NewTest_DAT testlabels

train_tol= size(Tr_DAT,2);%训练样本数
test_tol = size(Tt_DAT,2);%测试样本数
ClassNum = par.nClass ;%样本的种类数
%不同lamba
% diff_lambda=[10000 1000 100 10 1 0.1 0.01 0.001 0.0001 0.00001 0.000001 0]
% diff_lambda=0;
% diff_lambda=[10000 1000 100 10 1 0.1 0.01 ]
diff_lambda=[ 0.001 0.0001 0.00001 0.000001 0]

reg_rate = zeros(1,length(diff_lambda));

for eigen_num=dim
    kk = 1;
    eigen_num

    [disc_set,disc_value,Mean_Image]  =  Eigenface_f(Tr_DAT,eigen_num);
    tr_dat  =  disc_set'*Tr_DAT;
    tt_dat  =  disc_set'*Tt_DAT;

    for lambda=diff_lambda
        lambda
        ID = zeros(1,test_tol);
        X = tr_dat;
        [~,n] = size(X);%训练样本数
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
        invc_tmp=inv(c_tmp);
        tic;
        for i=1:test_tol
            y = tt_dat(:,i);
            XTy = X'*y;            
            [z,c] = DNRC_YALE(mu,n, XTy,invc_tmp);            
          
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
    kk

    reg_rate

end

