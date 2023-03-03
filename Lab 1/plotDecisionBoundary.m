% plotDecisionBoundary - plots the decision boundary for a set of samples
%
% INPUTS:
% X1 - MxM matrix - meshgrid matrix
% X2 - MxM matrix - meshgrid matrix
% boundaries - cell array - list of MxM matrix of emperically determined
% class regions
% labels - cell array - label to use for boundary on legend
function plotDecisionBoundary(X1, X2, boundaries, labels)
    % check function argument preconditions
    assert(isequal(size(X1), size(X2)));
    n_boundaries = length(boundaries);
    assert(n_boundaries <= 3);
    
    colors = ["#000000"; "#FF8000"; "#7E2F8E"]; % list of colors to use

    for i_b = 1:n_boundaries
        boundary = cell2mat(boundaries(i_b));
        assert(isequal(min(boundary, [], 'all'), 1));
        n_classes = max(boundary, [], 'all');
        
        for k=1:n_classes
            [~, c] = contour(X1, X2, boundary == k, 1, LineWidth=2, EdgeColor=colors(i_b));
            % only set display name once so we do not get duplicate entries
            % of boundary
            if (k == 1)
                c.DisplayName = cell2mat(labels(i_b));
            else
                c.HandleVisibility = 'off';
            end
            
        end

        % only color regions if there is a single boundary
        % otherwise, this gets confusion to visualize
        if (isequal(n_boundaries, 1))
            cmap = [1 0 0; 0 0 1]; % r and b for 2 classes
            % add g for 3rd class
            if (isequal(n_classes, 3))
                cmap(end+1,:) = [0 1 0];
            end
            colormap(cmap);
            surf(X1, X2, boundary, FaceAlpha=0.1, EdgeColor='none', HandleVisibility='off');
        end
    end
end
