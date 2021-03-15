clc;
clear all;
close all;

total_size = 200;
training_size = 100;
data_set = 'data.mat';
training_data = get_training_data(data_set, training_size);
testing_data = get_testing_data(data_set, total_size, training_size);

class_neutral = [];
class_expression = [];
for n = 1: training_size
    %3*n-2 is how neutral faces are indexed.
    class_neutral = [class_neutral training_data(:,3*n-2)];
    %3*n-1 is how expression faces are indexed.
    class_expression = [class_expression training_data(:,3*n-1)];
end
size(class_neutral)
% partitioning testing_set as well so it is easier to determine
% accuracy
testing_set_N = [];
testing_set_E = [];
for n = 1: total_size-training_size
    % ignoring the illumination class, appending only the neutral and
    % expression class. The illumination class is ignored (discarded).
    testing_set_N = [testing_set_N testing_data(:,3*n-2)];
    testing_set_E = [testing_set_E testing_data(:,3*n-1)];
end
% joining the two test set so 1st half are neutral points and 2nd half are
% expression points.
testing_set = [testing_set_N testing_set_E];


% SVM 
% concatenating both the classes in the following way
X = [class_neutral class_expression]';

%% 
pos_label = ones(1,size(class_neutral,2))';
neg_label = -ones(1,size(class_expression,2))';
Y = [pos_label;neg_label];

%%
Kernel_Cell={'linear';'polynomial';'RBF'};

%%
define_parameters;

%%
% Step 3: Fit the model
% Choose the kernel
kernel=char(Kernel_Cell(2));
%
[alpha,Ker,beta0]=SVM(X,Y,kernel);

alpha_ = [];
for i = 1:size(alpha,1)
    if alpha(i) <= 10^-8
        alpha_(i) = 0;
    else
        alpha_(i) = alpha(i);
    end
end
% mu_ is an appropriate vector with small values reduced to zeros.
alpha_ = alpha_';

% obtaining the values of wt. vector and bias term for linear
% classification
theta = ((alpha_.*Y)'*X)';
%%

accuracy = SVMtesting(theta,beta0,testing_set);
disp('base accuracy:');
disp(accuracy);


