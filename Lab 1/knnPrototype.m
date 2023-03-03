% knnPrototype - computes the kNN prototype for every point in a 2d space
%
% INPUTS:
% X1 - MxM matrix - meshgrid matrix
% X2 - MxM matrix - meshgrid matrix
% cluster - Nx2 matrix - the points in a cluster
% k - integer - how many neighbors to use when calculating the mean
% prototype
%
% OUTPUT:
% prototypes - (MxM)x2 matrix - the prototype for every point in the 
% meshgrid
function prototypes = knnPrototype(X1, X2, cluster, k)
    % check function argument preconditions
    assert(isequal(size(X1), size(X2)));

    % convert meshgrid into Nx2 matrix where each row is a point
    % this just makes looping slightly clearer
    x1_vec = reshape(X1, numel(X1), 1);
    x2_vec = reshape(X2, numel(X2), 1);
    X = [x1_vec, x2_vec];
    assert(isequal(size(X,2), 2));  % second dimension should be 2

    n_points = size(X,1);
    prototypes = zeros(n_points, 2);
    n_points_in_cluster = size(cluster, 1);

    % compute discriminant for each point
    for i = 1:n_points
        % repeat point so we can compute distance to every point in
        % cluster once
        x = repmat(X(i,:), n_points_in_cluster, 1);
        % compute squared distance to every point in cluster
        dist_2 = sum((cluster - x).^2, 2);
        assert(isequal(size(dist_2), [n_points_in_cluster 1]));

        % sort distances to find closest points
        [~, cluster_point_indices] = sortrows(dist_2);
        % k closest point indices
        k_closest_pts = cluster_point_indices(1:k,:);
        assert(isequal(size(k_closest_pts, 1), k));

        % take mean of k closest points
        closest_pts = cluster(k_closest_pts,:);
        prototype = mean(closest_pts, 1);
        assert(isequal(size(prototype), [1 2]));

        % store prototype
        prototypes(i, :) = prototype;
    end
end
