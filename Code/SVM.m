
function [alpha,Ker,beta0]=SVM(X,Y,kernel)
% X is N*p, Y is N*1,{-1,1}
global  precision Cost

switch kernel
    case 'linear'
        Ker=Ker_Linear(X,X);
    case 'polynomial'
        Ker=Ker_Polynomial(X,X);
    case 'RBF'
        Ker=Ker_RBF(X,X);
end

N= size(X,1);
size(X)
H= diag(Y)*Ker*diag(Y);
% disp('H');
disp(size(H));
f= - ones(N,1);
% disp('f');
disp(size(f));
Aeq=[Y';zeros(size(X,1)-1,size(X,1))];
% disp('Aeq');
disp(size(Aeq));
beq=zeros(size(X,1),1);
% disp('Beq');
disp(size(beq));
A=[];
b=[];
lb = zeros(N,1);
ub = Cost*ones(size(lb));
size(lb)
size(ub)
alpha=quadprog(H,f,[],[],Aeq,beq,lb,ub);
% disp('alpha');
disp(size(alpha));


serial_num=(1:size(X,1))';
serial_sv=serial_num(alpha>precision&alpha<Cost);
% disp('ser_sv');
disp(serial_sv);

temp_beta0=0;
for i=1:size(serial_sv,1)
    temp_beta0=temp_beta0+Y(serial_sv(i));
    temp_beta0=temp_beta0-sum(alpha(serial_sv(i))*...
        Y(serial_sv(i))*Ker(serial_sv,serial_sv(i)));
end
beta0=temp_beta0/size(serial_sv,1);

return










