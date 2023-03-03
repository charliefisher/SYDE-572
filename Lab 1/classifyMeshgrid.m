% classifyMeshgrid - wrapper that allows you to call a classifier with a 
%                    meshgrid instead of a Nx2 matrix of points
%
% INPUTS:
% X1 - MxM matrix - meshgrid matrix
% X2 - MxM matrix - meshgrid matrix
% classifier_func - function handle - classifies N_samplesx2 matrix of
% points
%
% OUTPUT:
% class_label - MxM matrix - the class label for each point in meshgrid
function class_label = classifyMeshgrid(X1, X2, classifier_func)
    % check function argument preconditions
    assert(isequal(size(X1), size(X2)));

    % convert meshgrid into Nx2 matrix where each row is a point
    % this just makes looping slightly clearer
    x1_vec = reshape(X1, numel(X1), 1);
    x2_vec = reshape(X2, numel(X2), 1);
    X = [x1_vec, x2_vec];
    assert(isequal(size(X,2), 2));  % second dimension should be 2

    % call genericClassifier with converted meshgrid
    class_label = classifier_func(X);
    
    % reshape class label into dimension of meshgrid
    class_label = reshape(class_label, size(X1));
end
