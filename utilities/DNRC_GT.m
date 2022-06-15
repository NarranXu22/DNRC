% function [z,c] = NRC(X, y,L,M,lamba)
function [z,c] = NRC(mu,n, XTy,c_tmp)
% [~,n] = size(X);%样本数
tol = 1e-5;
maxIter =5;
% ss=zeros(1,maxIter);
% labe=[1:maxIter];
% mu= 1e-1;
z = zeros(n,1);
c = zeros(n,1);
delta = zeros(n,1); 

% XTX = X'*X;
% XTy = X'*y;
iter = 0;
%准备工作
% c_tmp=XTX+1/2*mu*eye(n)+2*lambda*(L*M+XTX);
while iter<maxIter
    iter = iter + 1;
    zk = z;
    ck = c;
    
    % update c
    c=c_tmp*(XTy+mu/2*z+delta/2);
    % update z
    z_temp = c-delta/mu;
    z = max(0,z_temp);
    
    leq1 = z-c;
    leq2 = z-zk;
    leq3 = c-ck;
    stopC1 = max(norm(leq1),norm(leq2));
    stopC = max(stopC1,norm(leq3));
    
    if stopC<tol || iter>=maxIter
        break;
    else
        delta = delta + mu*leq1;
    end
    
end
