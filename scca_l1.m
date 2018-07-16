function [u, v, diffs, corrs, fus, fvs] = scca_l1(X, Y, opt)
%
% Function to perform Sparse Canonical Correlation Analysis (CCA) in a
% regression framework. The function takes as input 2 data matrices, and 2
% regression functions. It then subjects each view to the constraints
% contained by the corresponding regression function.
%
% Requirements - glmnet
%
%   Inputs:
%           X, Y - Data matrices (samples x features), or kernels
%           (samples x samples). They do not have to be on the same space,
%           i.e. one can be on input space, and the other on kernel space.
%
%           opt - Struct with options for algorithm
%                   - opt.plot: plot convergence (default: false)
%                   - opt.itr_lim: maximum number of iterations
%                   (default: 1000)
%                   - opt.diff: norm different between iterations, after
%                   which the algorithm will stop (default: 1E-5)
%
%
%   Outputs:
%           u, v - Weight vectors for X and Y, respectively
%
%
%   Version: 2017-03-16
%__________________________________________________________________________

% Written by Joao Matos Monteiro
% Email: joao.monteiro@ucl.ac.uk


%--- Check general options from opt struct
%--------------------------------------------------------------------------
if ~isfield(opt,'itr_lim')
    opt.itr_lim = 1000;
end

if ~isfield(opt,'diff')
    opt.diff = 1E-5;
end

if ~isfield(opt, 'plot')
    opt.plot = false;
end

if ~isfield(opt, 'update_alpha')
    opt.update_alpha = true;
end

if ~isfield(opt, 'lasso')
    opt.lasso = 1;
end

if ~isfield(opt, 'rescale_type')
    opt.rescale_type = 'cca';
end

if ~isfield(opt, 'update_step')
    opt.update_step = true;
end


if isfield(opt,'regX') % Use alternative function provided by the user
    opt.regX = regexprep(opt.regX,'<input>','X');
    opt.regX = regexprep(opt.regX,'<target>','b');
    warning('Using alternative function for X.')
end
if isfield(opt,'regY') % Use alternative function provided by the user
    opt.regY = regexprep(opt.regY,'<input>','Y');
    opt.regY = regexprep(opt.regY,'<target>','b');
    warning('Using alternative function for Y.')
end


%--- Checks to options
if isfield(opt,'fx')
    if opt.fx*size(X,2) < 1
        warning(['Fraction of features in X too low, changing it to :' num2str(1/size(X,2))]);
        opt.fx = 1/size(X,2);
    end
end
if isfield(opt,'fy')
    if opt.fy*size(Y,2) < 1
        warning(['Fraction of features in Y too low, changing it to :' num2str(1/size(Y,2))]);
        opt.fy = 1/size(Y,2);
    end
end


%--- Make progress plots
%--------------------------------------------------------------------------
if opt.plot
    figure('Position', [0 0 1000 800])
    title('Progress')
    subplot(2,2,1)
    hu = animatedline('Color','r', 'Linewidth', 1.5);
    hv = animatedline('Color','b', 'Linewidth', 1.5);
    
    xlabel('Iteration')
    ylabel('$\left\|\mathbf{w}_i - \mathbf{w}_{i-1}\right\|_2$',...
        'Interpreter','latex',...
        'FontSize', 14)
    ax = gca;
    ax.YScale = 'log';
    
    legend('u','v')
    
    subplot(2,2,2)
    hc = animatedline('Color','k', 'Linewidth', 1.5);
    xlabel('Iteration')
    ylabel('$\mathrm{Crr}(\mathbf{Xu},\mathbf{Yv})$',...
        'Interpreter','latex',...
        'FontSize', 14)
    
    subplot(2,2,3)
    hfu = animatedline('Color','r', 'Linewidth', 1.5);
    hfv = animatedline('Color','b', 'Linewidth', 1.5);
    xlabel('Iteration')
    ylabel('Fraction of features', 'FontSize', 14)
    legend('u','v')
    
    
end


%--- Main Loop
%--------------------------------------------------------------------------


[u,~,v] = svd(X'*Y);
u = [u(:,1),u(:,1)];
v = [v(:,1),v(:,1)];



stop = false;
itr = 1;
alpha = 0.5;
epsilon = 1E-9;

diffs = [];
corrs = [];
fus = [];
fvs = [];

while ~stop
    
    %--- Get u
    b = Y*v(:,2);
    if isfield(opt,'regX') % Use alternative function provided by the user
        eval(['u(:,2) = ' opt.regX ';']);
        if strcmp(opt.rescale_type, 'cca')
            u(:,2) = u(:,2)./norm(X*u(:,2));
        elseif strcmp(opt.rescale_type, 'pls')
            u(:,2) = u(:,2)./norm(u(:,2));
        else
            error('The rescaling type is not recognised.')
        end
    else
        u(:,2) = glmnet_wrapper(X,b,opt.fx,opt.lasso);
        if opt.update_step
            % Threshold
            u(:,2) = threshold(u(:,2),u(:,1),X, alpha, opt.rescale_type);
        else
            if strcmp(opt.rescale_type, 'cca')
                u(:,2) = u(:,2)./norm(X*u(:,2));
            elseif strcmp(rescale_type, 'pls')
                u(:,2) = u(:,2)./norm(u(:,2));
            else
                error('The rescaling type is not recognised.')
            end
        end
    end
    
    
    
    %--- Get v
    b = X*u(:,2);
    if isfield(opt,'regY') % Use alternative function provided by the user
        eval(['v(:,2) = ' opt.regY ';']);
        if strcmp(opt.rescale_type, 'cca')
            v(:,2) = v(:,2)./norm(Y*v(:,2));
        elseif strcmp(opt.rescale_type, 'pls')
            v(:,2) = v(:,2)./norm(v(:,2));
        else
            error('The rescaling type is not recognised.')
        end
    else
        v(:,2) = glmnet_wrapper(Y,b,opt.fy,opt.lasso);
        if opt.update_step
            % Threshold
            v(:,2) = threshold(v(:,2),v(:,1),Y, alpha, opt.rescale_type);
        else
            if strcmp(opt.rescale_type, 'cca')
                v(:,2) = v(:,2)./norm(Y*v(:,2));
            elseif strcmp(rescale_type, 'pls')
                v(:,2) = v(:,2)./norm(v(:,2));
            else
                error('The rescaling type is not recognised.')
            end
        end
    end
    
    
    
    %--- Evaluate convergence
    diff_u = norm(u(:,1) - u(:,2));
    diff_v = norm(v(:,1) - v(:,2));
    fu = sum(u(:,2)~=0)./length(u(:,2));
    fv = sum(v(:,2)~=0)./length(v(:,2));
    c = corr(X*u(:,2),Y*v(:,2));
    
    
    if itr >= opt.itr_lim
        stop = true;
    else
        if diff_u < opt.diff && diff_v < opt.diff
            if isfield(opt,'fx') && isfield(opt,'fy')
                if fu <= opt.fx && fv <= opt.fy
                    stop = true;
                end
            elseif isfield(opt,'fx')
                if fu <= opt.fx
                    stop = true;
                end
            elseif isfield(opt,'fy')
                if fv <= opt.fy
                    stop = true;
                end
            else % No Sparsity
                stop = true;
            end
        end
    end
    
    
    
    % If there is a chance that it's oscillating, reduce learning rate
    %---
    if opt.update_alpha
        % Has this corr value appeared before multiple times?
        if min(abs(corrs-c)) < 1E-6
            % Has the algorithm reached the desired number of features?
            if isfield(opt,'fx') && isfield(opt,'fy')
                if fu <= opt.fx && fv <= opt.fy
                    % Has the algorithm converged enough?
                    if diffs(1,end) < 1E-3 && diffs(2,end) < 1E-3
                        alpha = alpha/2;
                    end
                end
            elseif isfield(opt,'fx')
                % Has the algorithm converged enough?
                if fu <= opt.fx && diffs(1,end) < 1E-3
                    alpha = alpha/2;
                end
            elseif isfield(opt,'fy')
                % Has the algorithm converged enough?
                if fv <= opt.fy&& diffs(2,end) < 1E-3
                    alpha = alpha/2;
                end
            end
        end
    end
    
    
    % Save
    diffs = [diffs, [diff_u;diff_v]];
    corrs = [corrs, c];
    fus = [fus, fu];
    fvs = [fvs, fv];
    
    
    %--- Update
    v(:,1) = v(:,2);
    u(:,1) = u(:,2);
    itr = itr + 1;
    
    % Update plots
    if opt.plot
        addpoints(hu,itr,diff_u);
        addpoints(hv,itr,diff_v);
        addpoints(hc,itr,c);
        addpoints(hfu,itr,fu);
        addpoints(hfv,itr,fv);
        drawnow
    end
    
    
    
end

u = u(:,end);
v = v(:,end);


end


%--------------------------------------------------------------------------
%--- Private functions
%--------------------------------------------------------------------------

function b = glmnet_wrapper(x,y,ratio_feat, alpha)

% Get upper and lower boundaries for the number of features
nfeat = floor(size(x,2)*ratio_feat);
nfeat_upper = nfeat;

opt.alpha = alpha;

sucessfull_fit = false;
nfeat_upper_temp = nfeat_upper;
while ~sucessfull_fit
    
    %     opt.pmax = nfeat_upper_temp;
    opt.df = nfeat_upper_temp;
    
    % fit
    fit = glmnet(x,y, 'gaussian', opt);
    
    
    % choose only solutions with nfeat <= nfeat_upper and that select at least
    % 1 feature
    ind = sum(fit.beta~=0,1);
    ind = ind<=nfeat_upper_temp & ind > 0;
    
    % If there is no solution, increase nfeat_upper and check again
    flag = false;
    while sum(ind) == 0 && ~flag
        %         warning('glmnet couldl not find solution, increasing nfeat_upper by 1');
        nfeat_upper_temp = nfeat_upper_temp+1;
        ind = sum(fit.beta~=0, 1);
        if max(ind) < nfeat_upper_temp
            flag = true; % There are no solutions, glmnet has to be refit
        end
        ind = ind<=nfeat_upper_temp & ind > 0;
    end
    
    if flag
        sucessfull_fit = false;
    else
        sucessfull_fit = true;
    end
end

% remove beta == 0 solution
ind = sum(fit.beta~=0,1);
ind = ind > 0;
beta = fit.beta(:,ind);

% Make sure the betas are correct, i.e. there are no features that are
% removed after being added. In other words, the number of features is
% always increasing
% In theory should not happen, but I noticed that sometimes glmnet dos this
max_ind = 1;
flag = false;
f = sum(beta~=0,1);
while ~flag
    if max_ind <= length(f)
        if f(max_ind) <= nfeat_upper
            max_ind = max_ind + 1;
        else
            flag = true;
            max_ind = max_ind-1;
        end
    else
        flag = true;
        max_ind = length(f);
    end
end

if max_ind == 0;
    warning('Cannot fit to the number of features asked. Output extra features.')
    max_ind = 1;
end
beta = beta(:,1:max_ind);

if nfeat_upper_temp == nfeat_upper
    b = beta(:,end);
else
    ind = sum(beta~=0, 1);
    % The upper bound was increased, choose the most constrained solution
    [~,i] = min(abs(ind-nfeat_upper));
    ind = ind==ind(i);
    beta = beta(:,ind);
    b = beta(:,1);
end


end

function w = threshold(w,w_previous,X, alpha, rescale_type)
epsilon = 1E-9;

if strcmp(rescale_type, 'cca')
    w = w./norm(X*w);
elseif strcmp(rescale_type, 'pls')
    w = w./norm(w);
else
    error('The rescaling type is not recognised.')
end

% Check variables to be removed after the update
rm = (abs(w-w_previous) < epsilon) & (abs(w) == 0);

% Compute step
step = alpha.*(w-w_previous);
w = w_previous + step;

% Remove features that did not pass threshold
w(rm) = 0;

% normalise
if strcmp(rescale_type, 'cca')
    w = w./norm(X*w);
elseif strcmp(rescale_type, 'pls')
    w = w./norm(w);
else
    error('The rescaling type is not recognised.')
end

end