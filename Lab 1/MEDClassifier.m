% MEDClassifier - classifies samples in 2D space using MED
%
% INPUTS:
% X - Nx2 matrix - points to classify
% prototypes - cell array - entries in array are Nx2 or 2x1 matrix
% containg prototype to use for each point or single prototype to use for 
% all points
%
% OUTPUT:
% labels - Nx1 matrix - the class label of every point to classify
function labels = MEDClassifier(X, prototypes)
    discrim = @(x, zk) (-zk'*x + 0.5*zk'*zk);
    labels = genericClassifier(X, discrim, @min, prototypes);
end
