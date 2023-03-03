% GEDClassifier - classifies samples in 2D space using GED
%
% INPUTS:
% X - Nx2 matrix - points to classify
% mu_cell - cell array - entries in array are 2x1 matrix of cluster means
% cov_cell - cell array - entries in array are 2x2 matrix of cluster
% covariances
%
% OUTPUT:
% labels - Nx1 matrix - the class label of every point to classify
function labels = GEDClassifier(X, mu_cell, cov_cell)
    discrim = @(x, zk, Sk) ((x-zk)'*inv(Sk)*(x-zk));
    labels = genericClassifier(X, discrim, @min, mu_cell, cov_cell);
end
