function out = kernel_ridge_reg(K, t, lambda)
%
%   Kernel Ridge Regression. It assumes that t are normalised
%
%   Inputs:
%           K: Kernel matrix
%           t: targets
%           lambda: regulatization hyper-parameter (default: 1)
%
%
%   Version: 2016-08-03
%__________________________________________________________________________

% Written by Joao Matos Monteiro
% Email: joao.monteiro@ucl.ac.uk



%--- Read options
%--------------------------------------------------------------------------
if ~exist('lambda', 'var')
    lambda = 1;
end


%--- Kernel Ridge Regression
%--------------------------------------------------------------------------

%     m = mean(t);                     % mean of the training data
%     tr = t - m;                             % mean centre targets
tr = t;
alpha = (K+lambda*eye(size(K)))\tr;

%     out.m = m;
out = alpha;

end