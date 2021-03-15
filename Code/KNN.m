clc;
clear all;
close all;

%the value of number of nearest neighbors.
K = 8;

%this is the total size of the subjects.
total_size = 200;
%this is the size of subjects again, 150 points in each class.
training_size = 100;
data_set = 'data.mat';

%this simple partitions the .mat file into training_size and testing_size.
%But the samples are still interleaved.
training_data = get_training_data(data_set, training_size);
testing_data = get_testing_data(data_set, total_size, training_size);

class_neutral = [];
class_expression = [];
for n = 1: training_size
    %3xn-2 is how neutral faces are indexed.
    class_neutral = [class_neutral training_data(:,3*n-2)];
    %3xn-1 is how expression faces are indexed.
    class_expression = [class_expression training_data(:,3*n-1)];
end

testing_set_N = [];
testing_set_E = [];
for n = 1: total_size-training_size
    %ignoring the illumination class, appending only the neutral and
    %expression class. The illumination class is ignored (discarded).
    testing_set_N = [testing_set_N testing_data(:,3*n-2)];
    testing_set_E = [testing_set_E testing_data(:,3*n-1)];
end
%joining the two test set so 1st half are neutral data points and 2nd half
%are expression data points.
testing_set = [testing_set_N testing_set_E];

%testing without LDA -->
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%following steps are performed for every test_image.
accuracy = 0;
for n = 1: size(testing_set, 2)
    distance_vector = [];
    if n <= size(testing_set, 2)/2
        true_label = 1;
    else
        true_label = -1;
    end
    %computing L2norm or a testing image to all images in class_neutral
    %with appending label = +1.
    for m = 1: size(class_neutral, 2)
        %distance = L2_norm(testing_set(:,n), class_neutral(:,m));
        distance = norm(testing_set(:,n)-class_neutral(:,m));
        distance_vector = [distance_vector;[distance 1]];
    end
    %computing L2norm or a testing image to all images in class_expression
    %with appending label = -1.
    for m = 1: size(class_expression, 2)
        %distance = L2_norm(testing_set(:,n), class_expression(:,m));
        distance = norm(testing_set(:,n)-class_expression(:,m));
        distance_vector = [distance_vector;[distance -1]];
    end
    %find the computed label using the value of K from distance_vector
    computed_label = get_label(distance_vector, K);
    
    if true_label*computed_label == 1
        accuracy = accuracy + 1;
    end
end
disp('Base acccuracy: ');
disp(accuracy/size(testing_set, 2)*100);

%LDA starts here -->
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%getting theta from LDA function
theta = LDA('data.mat');
accuracy = LDA_KNN(theta,K,class_neutral,class_expression,testing_set);
disp('LDA acccuracy: ');
disp((accuracy/size(testing_set,2))*100);

%PCA starts here --> on 57
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
comp = myPCA([class_neutral class_expression],23);
accuracy = PCA_KNN(comp,K,class_neutral,class_expression,testing_set);
disp('PCA acccuracy: ');
disp((accuracy/size(testing_set,2))*100);


function training_data = get_training_data(filename, size)
    images = load(filename);
    faces = images.face;
    data = [];
    for n = 1:size*3
        image = faces(:,:,n);
        image = image(:);
        data = [data image];
    end
    training_data = data;
end

function testing_data = get_testing_data(filename, total_size, training_size)
    images = load(filename);
    faces = images.face;
    data = [];
    for n = (training_size*3)+1:total_size*3
        image = faces(:,:,n);
        image = image(:);
        data = [data image];
    end
    testing_data = data;
end

function computed_label = get_label(distance_vector, K)
F = distance_vector(:,1);
    [B,I] = sort(F,1);
    X = [];
    for m = 1:K
    X = [X distance_vector(I(m),2)];
    end
    computed_label = mode(X);
end

function theta = LDA(data_set)
    total_size = 200;
    training_size = 150;

    training_data = get_training_data(data_set, training_size);
    testing_data = get_testing_data(data_set, total_size, training_size);

    class_neutral = [];
    class_expression = [];
    for n = 1: training_size
        class_neutral = [class_neutral training_data(:,3*n-2)];
        class_expression = [class_expression training_data(:,3*n-1)];
    end

    %MDA begins:
    %calculating covariance of class1 and class2 (neutral and expression)
    cov_neutral = cov(class_neutral');
    cov_expression = cov(class_expression');
    cov_matrix = cov_neutral + cov_expression;
    inv_cov_matrix = pinv(cov_matrix);
    mean_neutral = sum(class_neutral, 2)/training_size;
    mean_expression = sum(class_expression, 2)/training_size;
    final_mean = mean_neutral - mean_expression;
    theta = inv_cov_matrix * final_mean; 
end

function accuracy = LDA_KNN(theta,K,class1,class2,testing_set)
    
    %projecting both classes on theta
    proj_classNeutr = theta' * class1;
    proj_classExprss = theta' * class2;

    acc = 0;
    for n = 1: size(testing_set, 2)
        distance_vector = [];
        if n <= size(testing_set, 2)/2
            true_label = 1;
        else
            true_label = -1;
        end
        %computing L2norm of a testing image to all images in class_neutral
        %with appending label = +1.
        for m = 1: size(proj_classNeutr, 2)
            %note that testing point is projected onto theta for testing.
            distance = norm(theta'*testing_set(:,n)-proj_classNeutr(:,m));
            distance_vector = [distance_vector;[distance 1]];
        end
        %computing L2norm of a testing image to all images in class_expression
        %with appending label = -1.
        for m = 1: size(proj_classExprss, 2)
            %note that testing point is projected onto theta for testing.
            distance = norm(theta'*testing_set(:,n)-proj_classExprss(:,m));
            distance_vector = [distance_vector;[distance -1]];
        end
        %find the computed label using the value of K from distance_vector
        computed_label = get_label(distance_vector, K);

        if true_label*computed_label == 1
            acc = acc + 1;
        end
    end
    accuracy = acc;
end

function something = myPCA(training_data,number)
    coeff = pca(training_data');
    coeff = coeff(:,1:number);
    something = coeff;
end

function accuracy = PCA_KNN(comp,K,class1,class2,testing_set)
    classN = comp'*class1;
    classE = comp'*class2;
    
    acc = 0;
    for n = 1: size(testing_set, 2)
        distance_vector = [];
        if n <= size(testing_set, 2)/2
            true_label = 1;
        else
            true_label = -1;
        end
        %computing L2norm of a testing image to all images in class_neutral
        %with appending label = +1.
        for m = 1: size(classN, 2)
            %note that testing point is projected onto theta for testing.
            distance = norm(comp'*testing_set(:,n)-classN(:,m));
            distance_vector = [distance_vector;[distance 1]];
        end
        %computing L2norm of a testing image to all images in class_expression
        %with appending label = -1.
        for m = 1: size(classE, 2)
            %note that testing point is projected onto theta for testing.
            distance = norm(comp'*testing_set(:,n)-classE(:,m));
            distance_vector = [distance_vector;[distance -1]];
        end
        %find the computed label using the value of K from distance_vector
        computed_label = get_label(distance_vector, K);

        if true_label*computed_label == 1
            acc = acc + 1;
        end
    end
    accuracy = acc;
end


