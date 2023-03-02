% plotUnitStdContour - plots the unit standard deviation contour for a
%                      class
%
% INPUTS:
% mu - 2x1 matrix - mean of class
% S - 2x2 matrix - covariance matrix of class
% color - color name
function plotUnitStdContour(mu, S, color)
    [eigVec, eigVal] = eig(S);

    theta = atan2(eigVec(1,2), eigVec(1,1));
    a = sqrt(eigVal(1,1));
    b = sqrt(eigVal(2,2));

    plot_ellipse(mu(1), mu(2), theta, a, b, color)
end