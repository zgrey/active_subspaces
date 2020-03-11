% Active Subspace approximation demo...
% Zach Grey, 02/12/2020
%
% The following implementations are based on the code and work developed by
% Paul Constantine and his research group including but not limited to:
% Zach Grey
% Andrew Glaws
% Jeffery Hokanson
% Izabel Aguiar
% Kerrek Stinson
% Paul Diaz
%
% Instructions: assuming you have pairs of inputs/outputs (x0,f(x0)) and a 
% uniform domain over a hypercube you'll want to specify the following:
% ub: a row vector of upper bounds
% lb: a row vector of lower bounds
% X0: a N by m matrix of inputs (in physical scales) 
% F:  a N by 1 column vector of outputs
% m:  the total number of parameters being varried (can be inferred from
%     the size of X0)
% N:  the total number of samples from the domain (can be inferred from the
%     size of X0)
clc; close all; clearvars; rng(42);

%%%%%%%%%%%%%%%%%%%%%%%%%%% THINGS TO MODIFY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% assign this to your local directory containing the matlab scripts
% codes available at GitHub, run the following:
% >>git clone https://github.com/zgrey/active_subspaces.git
% Linux
% AS_HOME = '/local/tmp/active_subspaces/matlab/';
% Windows
% AS_HOME = 'C:\Users\zgrey\Documents\GitHub\active_subspaces\matlab\';
% AS_HOME = '.\matlab\';
AS_HOME = './';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% add routines for surrogates and domains
addpath([AS_HOME,'ResponseSurfaces'])
addpath([AS_HOME,'Domains'])

%% Generate samples from your domain
%%%%%%%%%%%%%%%%%%%%%%%%%%% THINGS TO MODIFY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% select subspace approximation type ('QPHD' or 'FD')
sstype = 'FD';
% select differencing type ('fwd','bwd','cen')
FDtype = 'bwd';
% rank for activity score computation (pick this based on eigvalue gap, Figure 3)
r = 2;
% total number of parameters
m = 18;
% number of random samples 
N = 1000;
% finite difference step size (isotropic coordinate perturbations)
h = 1e-6;
% try a deg-order polynomial surrogate
deg = 5;
% The number of ~Nnew^r active coordinates for improving the r-dim. surrogate
Nnew = 100;
% set the number of inactive samples to use in stretch sampling
Nz = 10;
% given upper and lower bounds (replace these with your own domain def.):
% labels
par_lbl = {'SNR1' 'SNR2' 'SNR3' 'SNR4' 'SNR5' 'SNR6' 'SNR7'...
           'SNR8' 'SNR9' 'SNR10' 'SNR11' 'SNR12' 'SNR13' 'SNR14' 'SNR15' 'SNR16' 'W' 'Mnw'};
% ub = 2*ones(1,m); lb = ones(1,m);
SNRu = 22*10^5*ones(1,16); SNRl = 0.0001*ones(1,16);
ub = [SNRu 32 3]; lb = [SNRl 16 1];
% try log-scales since your parameters appear as powers in the objective
log_scl = true;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% log-scaling
if log_scl == 1
    ub = log([SNRu 32 3]); lb = log([SNRl 16 1]);
end
% define an affine transformation x = M*x0 + b so that x in [-1,1]^m
M = diag(2./(ub - lb)); b = -M*mean([ub; lb],1)'; Minv = diag((ub-lb)/2);

if log_scl == 1
    % original-scale samples
    X0 = exp(repmat(lb,N,1) + rand(N,m)*diag(ub - lb));
    % the log-scaled transformed samples become
    X = log(X0)*M + repmat(b',N,1);
    % and the inverse scaling becomes
    inv = @(X) exp((X - repmat(b',size(X,1),1))*Minv);
else
    % generate uniform random samples from box domain (replace with your data)
    X0 = repmat(lb,N,1) + rand(N,m)*diag(ub - lb);
    % the transformed samples become
    X = X0*M + repmat(b',N,1);
    % and the inverse scaling becomes
    inv = @(X) (X - repmat(b',size(X,1),1))*Minv;
end

%% Collect function response
%%%%%%%%%%%%%%%%%%%%%%%%%%% THINGS TO MODIFY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% test function (QPHD and Finite Differences should be exact up to machine precision)
% random linear function
% a = 2*rand(m,1)-1; Pr_func = @(SNR,W,Mnw) [SNR,W,Mnw]*a; dF.true = repmat(a',N,1);
% rank-2 quadratic coordinate aligned
% a = [1 1 zeros(1,m-2)]; Pr_func =@(SNR,W,Mnw) sum([SNR,W,Mnw]*(diag(a)).*[SNR,W,Mnw],2); dF.true = 2*X0*diag(a);
% rank-3 quadratic random
% a = (2*rand(m,3)-1); Pr_func =@(SNR,W,Mnw) sum([SNR,W,Mnw]*(a*a').*[SNR,W,Mnw],2); dF.true = 2*X0*(a*a');
% evaluate the function (replace this with your vector of responses)
% F = Pr_func(X0(:,1:m-2),X0(:,m-1),X0(:,m));

% dummy example model with "vectorized" Pr_func
% partition into model parameters
SNR = X0(:,1:16); W = X0(:,17); Mnw = X0(:,18);
p_cnw = 0.15;
P_trnw = @(W,Mnw) 2*(1-2*p_cnw)./( (1-2*p_cnw).*(1 + W) + p_cnw*W.*(1 - (2*p_cnw).^Mnw ) );
Pr_func = @(SNR,W,Mnw) sum(P_trnw(W,Mnw).*log2(1 + SNR),2);
F = Pr_func(SNR,W,Mnw);

% make Pr_func an ambiguous function handle f
f = @(X) Pr_func(X(:,1:m-2),X(:,m-1),X(:,m));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Finite difference designs (using normalized scales)
% perturb all entries
fwd = inv(X + h); bwd = inv(X - h);
% preallocate
dF.fwd = zeros(N,m); dF.bwd = zeros(N,m); dF.cen = zeros(N,m);
% loop through entries of gradient
for j = 1:m
    % set entries to nominal random samples
    Xfwd = inv(X); Xbwd = inv(X);
    
    if strcmpi(FDtype,'fwd') || strcmpi(FDtype,'cen')
        % forward differencing
        Xfwd(:,j) = fwd(:,j);
        dF.fwd(:,j) = 1/h*(f(Xfwd) - f(inv(X)));
        dF.eval(:,j) = dF.fwd(:,j);
    end
    
    if strcmpi(FDtype,'bwd') || strcmpi(FDtype,'cen')
        % backward differencing
        Xbwd(:,j) = bwd(:,j);
        dF.bwd(:,j) = 1/h*(f(inv(X)) - f(Xbwd));
        dF.eval(:,j) = dF.bwd(:,j);
    end
end
if strcmpi(FDtype,'cen')
    % central differencing
    dF.cen = 0.5*(dF.fwd + dF.bwd);
    dF.eval = dF.cen;
end

if strcmpi(sstype,'QPHD')
%% Global quadratic model (arbitrary rank "r" approximation, potentially biased)
% run the ordinary linear least-squares (OLS or global linear model)
% original scale
[sub,surr1D,surr2D,fig] = QPHD_model(X,F,deg,Nnew,Nz);
elseif strcmpi(sstype,'FD')
%% Finite differences model (arbitrary rank "r" approximation, unbiased but requires differentiability)
% dont forget the chain rule!
[sub,surr1D,surr2D,fig] = FD_model(X,F,dF.eval,deg,Nnew,Nz);
end

%% Sensitivity analysis
sub.act_scr = sub.W(:,1:r).^2*sub.eigs(1:r);
psort = sortrows([sub.act_scr,(1:m)']);
% make a pareto
figure; subplot(1,2,1), stem(psort(:,1),'filled'); alpha(0.5);
axis([1,m,0,max(sub.act_scr)]); xlabel('param. index');
ylabel('activity score'); 
ax = subplot(1,2,2); pareto(psort(:,1),par_lbl(psort(:,2))); 
gca; ylabel('activity score'); alpha(0.5); xtickangle(ax, 45);

%% Stretch sampling
% rescale stretch samples
surr1D.dom.newX0 = inv(surr1D.dom.newX);
surr2D.dom.newX0 = inv(surr2D.dom.newX);

% evaluate 1D stretch samples
surr1D.Fnew = f(surr1D.dom.newX0);
% overlay new function evaluations
figure(fig.shdw1D);
scatter(surr1D.dom.newX*sub.W(:,1),surr1D.Fnew,'filled'); alpha(0.5);
% retrain surrogate
[surr1D.newCoef,~,~,~,surr1D.newHx,surr1D.newres] = poly_train([X; surr1D.dom.newX]*sub.W(:,1), [F; surr1D.Fnew], deg);
% update the coefficient of determination
surr1D.newRsqd = 1-cov(surr1D.newres)/cov([F; surr1D.Fnew]);
% evaluate surrogate
surr1D.newH = poly_predict(surr1D.dom.uniViz,surr1D.newCoef,deg);
% plot updated surrogate
plot(surr1D.dom.uniViz,surr1D.newH,'linewidth',2);
fig.shdw1D.CurrentAxes.YLim = [min(fig.shdw1D.CurrentAxes.YLim),...
                                    max([F; surr1D.Fnew; surr1D.H])];
legend off;
title(['1D Shadow Plot - ',sstype,' Model, R^2 = ', num2str(surr1D.newRsqd), ' (old ',num2str(surr1D.Rsqd),')']);

% evaluate 2D stretch samples
surr2D.Fnew = f(surr2D.dom.newX0);
% overlay new function evaluations
figure(fig.shdw2D);
scatter(surr2D.dom.newX*sub.W(:,1),surr2D.dom.newX*sub.W(:,2),50,'filled','cdata',surr2D.Fnew); alpha(0.5);
% retrain surrogate with new stretch samples
[surr2D.newCoef,~,~,~,surr2D.newHx,surr2D.newres] = poly_train([X; surr2D.dom.newX]*sub.W(:,1:2), [F; surr2D.Fnew], deg);
% update the coefficient of determination
surr2D.newRsqd = 1-cov(surr2D.newres)/cov([F; surr2D.Fnew]);
% evaluate surrogate
surr2D.newH = poly_predict(surr2D.dom.uniViz,surr2D.newCoef,deg);
% throw out values outside of domain
surr2D.newH(~in2DZ(surr2D.dom.ext,surr2D.dom.uniViz)) = NaN;

% update the surrogate plot
figure;
% plot projected samples
scatter([X; surr2D.dom.newX]*sub.W(:,1),[X; surr2D.dom.newX]*sub.W(:,2),50,'filled','cdata',[F; surr2D.Fnew]); alpha(0.5);
hold on; axis square;
% plot extent of 2-dim. domain vertices
scatter(surr2D.dom.ext(:,1),surr2D.dom.ext(:,2),50,'k','filled'); 
% plot the edges of the domain zonotope vertices
plot([surr2D.dom.ext(:,1); surr2D.dom.ext(1,1)],...
     [surr2D.dom.ext(:,2); surr2D.dom.ext(1,2)],'k','linewidth',2);
contour(reshape(surr2D.dom.uniViz(:,1),100,100), reshape(surr2D.dom.uniViz(:,2),100,100), reshape(surr2D.newH,100,100),50);
colorbar;
% plot the edges of the projected data convex hull
plot([sub.conH(:,1); sub.conH(1,1)],...
     [sub.conH(:,2); sub.conH(1,2)],'k--','linewidth',1);
% plot new active variable samples
scatter(surr2D.dom.newY(:,1),surr2D.dom.newY(:,2),55,'ko');
% format
title(['2D Shadow Plot - ',sstype,' Model, R^2 = ', num2str(surr2D.newRsqd), ' (old ',num2str(surr2D.Rsqd),')']);
xlabel 'y_1 = w_1^Tx'; ylabel 'y_2 = w_2^Tx';
