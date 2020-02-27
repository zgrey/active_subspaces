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
AS_HOME = '/local/tmp/active_subspaces/matlab/';
% Windows
% AS_HOME = 'C:\Users\zgrey\Documents\GitHub\active_subspaces\matlab\';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% add routines for surrogates and domains
addpath([AS_HOME,'ResponseSurfaces'])
addpath([AS_HOME,'Domains'])

%% Generate samples from your domain
%%%%%%%%%%%%%%%%%%%%%%%%%%% THINGS TO MODIFY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% total number of parameters
m = 18;
% number of random samples for OLS and QPHD
N = 5000;
% given upper and lower bounds (replace these with your own domain def.):
% ub = 2*ones(1,m); lb = ones(1,m);
SNRu = 22*10^5*ones(1,16); SNRl = 0.001*ones(1,16);
ub = [SNRu 32 3]; lb = [SNRl 16 1];
% try log-scales since your parameters appear as powers in the objective
log_scl = 'false';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Input scaling
% log-scaling
if strcmpi(log_scl,'true')
    ub = log([SNRu 32 3]); lb = log([SNRl 16 1]);
end
% define an affine transformation x = M*x0 + b so that x in [-1,1]^m
M = diag(2./(ub - lb)); b = -M*mean([ub; lb],1)'; Minv = diag((ub - lb)/2);

if strcmpi(log_scl,'true')
    % log-scale version
    X0 = exp(repmat(lb,N,1) + rand(N,m)*diag(ub - lb));
    % the log-scaled transformed samples become
    X = log(X0)*M + repmat(b',N,1);
else
    % generate uniform random samples from box domain (replace with your data)
    X0 = repmat(lb,N,1) + rand(N,m)*diag(ub - lb);
    % the transformed samples become
    X = X0*M + repmat(b',N,1);
end

%% Collect function response
%%%%%%%%%%%%%%%%%%%%%%%%%%% THINGS TO MODIFY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% test function (QPHD and finite differences should be exact up to machine precision)
a = rand(m,2); Pr_func =@(SNR,W,Mnw) sum([SNR,W,Mnw]*(a*a').*[SNR,W,Mnw],2);
% evaluate the function (replace this with your vector of responses)
F = Pr_func(X0(:,1:16),X0(:,17),X0(:,18));
% or make this a new ambiguous function
f = @(X) Pr_func(X(:,1:16),X(:,17),X(:,18));

% dummy example model with "vectorized" Pr_func
% partition into model parameters
% SNR = X0(:,1:16); W = X0(:,17); Mnw = X0(:,18);
% p_cnw = 0.75;
% P_trnw = @(W,Mnw) 2*(1-2*p_cnw)./( (1-2*p_cnw).*(1 + W) + p_cnw*W.*(1 - (2*p_cnw).^Mnw ) );
% Pr_func = @(SNR,W,Mnw) sum(P_trnw(W,Mnw).*log2(1 + SNR),2);
% F = Pr_func(SNR,W,Mnw);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Global linear model (strictly reduces to 1-dimensional subspace)
%%%%%%%%%%%%%%%%%%%%%%%%%%% THINGS TO MODIFY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% let's try a 5th order polynomial surrogate
% deg = 5;
% and let's resample the domain of the surrogate uniformly over 100 points
% Nsurr = 100;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% run the ordinary linear least-squares (OLS or global linear model)
% original scale
% [OLS_sub,OLS_surr,OLS_fig] = OLS_model(X,F,deg,Nsurr);
% log-scale response
% [OLS_sub,OLS_surr,OLS_fig] = OLS_model(X,log(F),deg,Nsurr);
% reassign plot lables
% figure(OLS_fig); ylabel 'Log(f)';

%% Global quadratic model (arbitrary rank "r" approximation, potentially biased)
%%%%%%%%%%%%%%%%%%%%%%%%%%% THINGS TO MODIFY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% let's try a 5th order polynomial surrogate
deg = 3;
% Nnew^2 grid of active coordinates for improving the 2D surrogate
Nnew = 10;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% run the ordinary linear least-squares (OLS or global linear model)
% original scale
% [QPHD_sub,QPHD_surr1D,QPHD_surr2D,QPHD_fig] = QPHD_model(X,F,deg,Nsurr);
% log-scale response
[QPHD_sub,QPHD_surr1D,QPHD_surr2D,QPHD_fig] = QPHD_model(X,log(F),deg,Nnew);
% reassign plot lables
figure(QPHD_fig.shdw1D); ylabel 'Log(f)';

% evaluate new "stretched" samples
Fnew = f((QPHD_surr2D.dom.newX - repmat(b',size(QPHD_surr2D.dom.newX,1),1))*Minv);
[QPHD_sub,QPHD_surr1D,QPHD_surr2D,QPHD_fig] = QPHD_model([X;QPHD_surr2D.dom.newX],log([F; Fnew]),deg,0);
% reassign plot lables
figure(QPHD_fig.shdw1D); ylabel 'Log(f)';