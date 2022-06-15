function [z,c] = DNRC(mu,n, XTy,c_tmp)
tol = 1e-5;
maxIter =25;
z = zeros(n,1);
c = zeros(n,1);
delta = zeros(n,1); 


% y = 1:1:25;
y = linspace(1,25,25);
x1=[];
x2=[];
x3=[];
x4=[];
x5=[];
x6=[];
iter = 0;
while iter<maxIter
    iter = iter + 1;
    zk = z;
    ck = c;
    % update c
    c=c_tmp\(XTy+mu/2*z+delta/2);
    % update z
    z_temp = c-delta/mu;
    z = max(0,z_temp);
    
    leq1 = z-c;
    leq2 = z-zk;
    leq3 = c-ck;
    stopC1 = max(norm(leq1),norm(leq2));
    stopC = max(stopC1,norm(leq3));
    x1(iter)=norm(leq1);
    x2(iter)=norm(leq2);
    x3(iter)=norm(leq3);
    x4(iter) = stopC1;
    x5(iter) = stopC;
%     
%     if stopC<tol || iter>=maxIter
%         break;
%     else
%         delta = delta + mu*leq1;
%     end
    
end
figure;
plot(y,x1,y,x2,y,x3)
% plot(x4,y,x5,y)
figure;
plot(y,x4,y,x5)
