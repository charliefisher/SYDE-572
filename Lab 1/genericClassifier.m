% genericClassifier - classifies samples in 2D space given a discriminant 
%                     function and comparison method
%
% INPUTS:
% X1 - MxM matrix - meshgrid matrix
% X2 - MxM matrix - meshgrid matrix
% discriminant_func - function handle - computes discriminant of sample;
% arguments to discriminant_func are sample (2x1 matrix), prototype (2x1 
% matrix), and optionally covariance (2x2 matrix)
% min_or_max - function handle - @min or @max; choice depends on
% discriminant_func
% prototypes - K cell array - prototypes for each class; prototype can be a
% 2x1 matrix or a N_samplesx2 matrix
% covariances - K cell array - covariances for each class; entries are 2x2
% matrices
%
% OUTPUT:
% class_label - MxM matrix - the class label for each point in meshgrid
function class_label = genericClassifier(X1, X2, discriminant_func, min_or_max, prototypes, covariances)
    % covariances are optional argument since MED does not use them
    use_covariance = exist('covariances', 'var');

    % check function argument preconditions
    assert(~use_covariance || isequal(length(prototypes), length(covariances)));
    assert(isequal(size(X1), size(X2)));

    % convert meshgrid into Nx2 matrix where each row is a point
    % this just makes looping slightly clearer
    x1_vec = reshape(X1, numel(X1), 1);
    x2_vec = reshape(X2, numel(X2), 1);
    X = [x1_vec, x2_vec];
    assert(isequal(size(X,2), 2));  % second dimension should be 2

    n_classes = length(prototypes);
    n_points = size(X,1);
    discriminants = zeros(n_points, n_classes);

    for k = 1:n_classes
        zk = cell2mat(prototypes(k));

        % if zk is not a 2x1 matrix, we have a different prototype for each
        % point
        % if zk is a single vector, duplicate it so function can be called 
        % in same way as multi-prototype case
        if (isequal(size(zk), [2 1]))
            zk = repmat(zk', n_points, 1);
        end

        % only use covariance if it was provided
        if (use_covariance)
            Sk = cell2mat(covariances(k));
        end

        % compute discriminant for each point
        for i = 1:n_points
            x = X(i,:)';
            z = zk(i,:)';

            if (use_covariance)
                discrim = discriminant_func(x, z, Sk);
            else
                discrim = discriminant_func(x, z);
            end

            discriminants(i,k) = discrim;
        end
    end

    % find minimum or maximum discriminant
    % look along 2nd axis (which corresponds to classes
    [~, compare_idx] = min_or_max(discriminants, [], 2);
    assert(isequal(size(compare_idx), [n_points 1]));
    
    % reshape class label into dimension of meshgrid
    class_label = reshape(compare_idx, size(X1));
end