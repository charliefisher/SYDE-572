% MAPClassifier - classifies samples in 2D space using MAP
%
% INPUTS:
% X - Nx2 matrix - points to classify
% mu_cell - cell array - entries in array are cluster means (2x1 matrix)
% cov_cell - cell array - entries in array are cluster covariances (2x2
% matrix)
% N_cell - cell array - entries in array are number of points in cluster
% (integer)
%
% OUTPUT:
% labels - Nx1 matrix - the class label of every point to classify
function labels = MAPClassifier(X, mu_cell, cov_cell, N_cell)
    discrim = @(x, zk, Sk, Nk) (2*log(Nk) - log(det(Sk)) - (x-zk)'*inv(Sk)*(x-zk));
    labels = genericClassifier(X, discrim, @max, mu_cell, cov_cell, N_cell);
end
