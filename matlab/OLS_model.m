function [sub,surr,fig] = OLS_model(X,F,deg,Nsurr)
%% Global linear model (strictly reduces to 1-dimensional subspace)
% inputs:
%      X: a N by m matrix of parameter values from the [-1,1]^m hypercube
%      F: a N by 1 vector of paired function evaluations to the rows of X
%    deg: an integer specifying the degree of the monomial surrogate
%  Nsurr: an integer specifying the number of uniform samples of the surrogate
%
% outputs:
%    sub: a structure array containing the subspace information
%   surr: a surrogate strucutre array containing the polynomial model
%    fig: the figure object for the plots

%% Active subspace
% Approximate the active subspace by a global linear model
% the number of random samples and parameters
N = size(X,1); m = size(X,2);
% fit a global linear model (make sure to use the transformed parameters!)
sub.W = [ones(N,1), X] \ F;
% make this an orthonormal basis
sub.W = sub.W(2:end)/norm(sub.W(2:end));

%% Domain
% uniformly sample over extent of projected samples
Y = X*sub.W;
surr.dom.Y = linspace(min(Y),max(Y),Nsurr)';
% Find the extents of the 1-dimensional active variable domain
% determine the extent of the original domain projected to the new 1d-domain
options = optimset('Display','off');
ext1 = linprog(sub.W,[],[],[],[],-ones(m,1),ones(m,1),options);
ext2 = linprog(-sub.W,[],[],[],[],-ones(m,1),ones(m,1),options);
% assign the extent of the domain
surr.dom.ext = [min([ext1'; ext2']*sub.W), max([ext1'; ext2']*sub.W)];
% uniformly sample the surrogate domain over the full extent
surr.dom.uni = linspace(surr.dom.ext(1),surr.dom.ext(2),Nsurr)';

%% Surrogate model
% train a polynomial surrogate
[surr.Coef, ~, ~, ~, surr.Hx, surr.res] = poly_train(X*sub.W, F, deg);
surr.Rsqd = 1 - cov(surr.res)./cov(F);

% obtain the surrogate response over the full domain
[surr.H, surr.dH] = poly_predict(surr.dom.uni,surr.Coef,deg);
% obtain the surrogate response over the data projection
[surr.Hy,surr.dHy] = poly_predict(surr.dom.Y,surr.Coef,deg);

%% Visualize
fig = figure;
% make the shadow plot
scatter(X*sub.W,F,50,'filled'); alpha(0.5); hold on;
% plot the resulting approximation
plot(surr.dom.Y,surr.Hy,'k','linewidth',2); 
plot(surr.dom.uni,surr.H,'k--','linewidth',2); 
ax = gca; Ylim = ax.YLim; ax.YLim(1) = Ylim(1);
% plot the extent of the original domain
scatter(surr.dom.ext,min(ax.YLim)*ones(1,2),50,'filled','k');
% plot the extent of the projected sample domain
scatter([min(Y),max(Y)],min(ax.YLim)*ones(1,2),50,'k');
% label stuff
title(['Shadow Plot - OLS Model, R^2 = ', num2str(surr.Rsqd)]);
xlabel 'y = w_1^Tx'; ylabel 'f';
legend('Original smpls',['order-',num2str(deg), ' poly.'])