clear
clc
close all
addpath data utilities
load('GeorgiaTech.mat');
ClassNum      =   class_num; 
dim = length(gnd)-1;
% diff_lambda=[0 100]
diff_lambda = [100]
for lambda=diff_lambda
    kk = 1;
    for train_num=8
        acc=zeros(10,1);
        for iter=1:10
      
            [train_data, train_label, test_data, test_label] = separate_data_rand(fea',gnd',train_num);
            train_data=train_data';
            train_label=train_label';
            test_data=test_data';
            test_label=test_label';
            train_tol= size(train_data,2);
            test_tol=size(test_data,2);
           
            ID = zeros(1,test_tol);
            X = train_data;
            [~,n] = size(X);
            XTX = X'*X;

            M = [];
            for i = 1 : ClassNum
                index = find(train_label == i);
                Xi = X(:, index);
                temp = Xi'* Xi;
                M = blkdiag(M, temp);
            end
       
            mu= 1e-1;
            c_tmp=XTX+1/2*mu*eye(n)+2*lambda*(ClassNum*M+XTX);
            invc_tmp=inv(c_tmp);
       
            tic;
            for i=1:test_tol
                y = test_data(:,i);
                XTy = X'*y;
                [z,c] = DNRC_GT(mu,n, XTy,invc_tmp);
                
                W = sparse([],[],[],train_tol,ClassNum,length(c));
                
          
                for j=1:ClassNum
                    ind = (j==train_label);
                    W(ind,j) = c(ind);
                end
             
                temp = X*W-repmat(y,1,ClassNum);
                residual = sqrt(sum(temp.^2));
                
               
                [~,index]=min(residual);
                ID(i)=index;
            end
            cornum      =   sum(ID==test_label);
            acc(iter,1) =   cornum/length(test_label); 
%             toc
        end
        reg_rate(1,kk)=mean(acc)
        kk=kk+1;       
    end
    
end
