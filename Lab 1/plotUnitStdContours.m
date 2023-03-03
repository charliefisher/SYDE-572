% plotUnitStdContour - plots the unit standard deviation contour for a
%                      class
%
% INPUTS:
% mu_cell - 2x1 matrix - mean of class
% cov_cell - 2x2 matrix - covariance matrix of class
function plotUnitStdContours(mu_cell, cov_cell)
    % check function argument preconditions
    assert(isequal(length(mu_cell), length(cov_cell)));
    n_countours = length(mu_cell);
    assert(n_countours <= 3);
    
    colors = ['r'; 'b'; 'g']; % list of colors to use

    for i = 1:n_countours
        mu = cell2mat(mu_cell(i));
        cov = cell2mat(cov_cell(i));

        [eigVec, eigVal] = eig(cov);

        theta = atan2(eigVec(1,2), eigVec(1,1));
        a = sqrt(eigVal(1,1));
        b = sqrt(eigVal(2,2));
    
        plot_ellipse(mu(1), mu(2), theta, a, b, colors(i))
    end
end
