clear variables
close all
clc

addpath 'Subspaces' 'Plotters' 'ResponseSurfaces' 'Grad_Free'

%% Test functions - Choose 1

% Linear function
m = 5; n = 1;
M = 100;
X = 2*rand(M, m) - 1;
[f, df] = test_function_1(X);
df = df./repmat(sqrt(sum(df.^2, 2)), 1, m);

% Data Set
% [X, f, df] = test_function_2();
% [M, m] = size(X);
% n = 2;
% df = df./repmat(sqrt(sum(df.^2, 2)), 1, m);

%% Compute active subspace

sub = compute(df);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);

% Plot results
eigenvalues(sub.eigenvalues, sub.e_br)
eigenvectors(sub.W1)
subspace_errors(sub.sub_br)
sufficient_summary(X*sub.eigenvectors(:, 1:2), f)

%% Compute Grad Free Active Subspace

gf_sub = data_projections(X,@(x) test_function_1(x),1);

% Plot results
eigenvalues(gf_sub.eigenvalues, gf_sub.e_br)
eigenvectors(gf_sub.W1)
subspace_errors(gf_sub.sub_br)
sufficient_summary(X*gf_sub.eigenvectors(:, 1:2), f)

%% Compute Subspace Error Metric
sub_er = norm(sub.W1'*gf_sub.W1,'inf');
disp('Subpsace Error Metric: ')
disp(sub_er)

%% Approximate Active Subspace Demo

% 1D
m = 1;
N = 4;
y = X*gf_sub.eigenvectors(:,1:m);
y_new = zeros(50,m);
for i=1:m
    y_new(:,i) = linspace(min(y(:,i)),max(y(:,i)),50)';
end

% Poly Approximation
[Coef,B,~,~,f_hat,res] = poly_train(y,f,N);
[f_poly] = poly_predict(y_new,Coef,N);

% RBF Approximation
[net] = rbf_train(y,f);
f_rbf = rbf_predict(y_new,net);

figure(9)
hold on
plot(y_new,f_poly);
plot(y_new,f_rbf);

% 2D
m = 2;
N = 2;
y = X*gf_sub.eigenvectors(:,1:m);
y_new = zeros(25,m);
for i=1:m
    y_new(:,i) = linspace(min(y(:,i)),max(y(:,i)),25)';
end
[X,Y] = meshgrid(y_new(:,1),y_new(:,2),25);
[J,~] = size(X);

% Poly Approximation
[Coef,B,~,~,f_hat,res] = poly_train(y,f,N);
[f_poly] = poly_predict([reshape(X,J*J,1), reshape(Y,J*J,1)],Coef,N);

% RBF Approximation
[net] = rbf_train(y,f,3);
f_rbf = rbf_predict([reshape(X,J*J,1), reshape(Y,J*J,1)],net);

figure
surf(X,Y,reshape(f_poly,J,J));
hold on
scatter3(y(:,1),y(:,2),f,'k','MarkerFaceColor','k');
xlabel 'Active Variable 1'
ylabel 'Active Variable 2'
zlabel 'Output'
title 'Polynomial Active Subspace Approximation'

figure
surf(X,Y,reshape(f_rbf,J,J));
hold on
scatter3(y(:,1),y(:,2),f,'k','MarkerFaceColor','k');
xlabel 'Active Variable 1'
ylabel 'Active Variable 2'
zlabel 'Output'
title 'RBF Active Subspace Approximation'
