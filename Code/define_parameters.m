% define_parameters
% Constant=Inf for Hard Margin
global poly_con gamma kappa1 kappa2 precision Cost
poly_con=0.9; % For Polynomial Kernel
gamma=10000;% For RBF
kappa1=1/size(X,1);kappa2=kappa1; % For Sigmoid

precision=10^-8;Cost=0.5;