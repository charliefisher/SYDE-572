% kNNClassifier - classifies samples in 2D space using kNN
%
% INPUTS:
% X - Nx2 matrix - points to classify
% clusters - cell array - list of points in cluster to use when computing
% kNN prototype
% k - integer - k used in kNN prototype selection
%
% OUTPUT:
% labels - Nx1 matrix - the class label of every point to classify
function labels = kNNClassifier(X, clusters, k)
    n_clusters = length(clusters);
    nn_prototypes = cell(1, n_clusters);

    % compute prototypes for each point
    for i = 1:n_clusters
        c = cell2mat(clusters(i));
        nn_prototypes{i} = knnPrototype(X, c, k);
    end

    % classify points using MED and kNN prototype
    labels = MEDClassifier(X, nn_prototypes);
end
