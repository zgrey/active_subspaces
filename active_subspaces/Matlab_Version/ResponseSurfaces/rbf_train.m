function [net,f_hat,r] = rbf_train(X, f, varargin)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Train the radial basis approximation with standard Guassian transfer
%   function
%
%   Inputs:
%          X: The M-by-m array of training points for the polynomial
%             approximation such that m is the number of dimensions
%          f: The M-by-1 vector of function values paired with the M
%             observations of the training points in X
%          s: radial basis funtion spreading
%
%   Outputs:
%        net: Radial Basis Neural Network stucture
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if max(size(varargin)) <= 1

    % Check variable inputs
    if isempty(varargin)
        s = 1;
    else
        s = varargin{1};
    end
    
    % Train network using internal Matlab routine
    net = newrb(X',f',0,s);

    % Approximate training function evaluations
    f_hat = sim(net,X');
    f_hat = f_hat';

    r = f_hat-f;

    close('NEWRB')
else 
    disp('Error: Too many inputs');
end