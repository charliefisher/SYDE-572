% genericClassifier - classifies samples in 2D space given a discriminant 
%                     function and comparison method
%
% INPUTS:
% X - N_samplesx2 matrix - points to classify
% discriminant_func - function handle - computes discriminant of sample;
% arguments to discriminant_func are sample (2x1 matrix), prototype (2x1 
% matrix), and optionally covariance (2x2 matrix)
% min_or_max - function handle - @min or @max; choice depends on
% discriminant_func
% prototypes - K cell array - prototypes for each class; prototype can be a
% 2x1 matrix or a N_samplesx2 matrix
% covariances - K cell array - (optional) covariances for each class; entries are 2x2
% matrices; if not specified, discriminant_func does not receive 
% covariance
% cluster_sizes - K cell array - (optional) number of elements in each 
% cluster; if not specified, discriminant_func does not receive Nk
%
% OUTPUT:
% class_label - N_samplesx1 - the class label for each point in X
function class_label = genericClassifier(X, discriminant_func, min_or_max, prototypes, covariances, cluster_sizes)
    % covariances are optional argument since MED does not use them
    use_covariance = exist('covariances', 'var');
    % Nk is optional argument since MED and GED do not use them
    use_Nk = exist('cluster_sizes', 'var');

    % check function argument preconditions
    assert(~use_covariance || isequal(length(prototypes), length(covariances)));
    assert(~use_Nk || isequal(length(prototypes), length(cluster_sizes)));
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

        % only use cluster size if it was provided
        if (use_Nk)
            Nk = cell2mat(cluster_sizes(k));
        end

        % compute discriminant for each point
        for i = 1:n_points
            x = X(i,:)';
            z = zk(i,:)';

            args = {x, z};
            if (use_covariance)
                args{end+1} = Sk;
            end
            if (use_Nk)
                args{end+1} = Nk;
            end   

            discriminants(i,k) = discriminant_func(args{:});
        end
    end

    % find minimum or maximum discriminant
    % look along 2nd axis (which corresponds to classes)
    [~, class_label] = min_or_max(discriminants, [], 2);
    assert(isequal(size(class_label), [n_points 1]));
end
