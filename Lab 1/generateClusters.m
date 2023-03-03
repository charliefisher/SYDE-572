% generateClusters - generates random points in cluster with specific
%                    statistical properties
%
% INPUTS:
% N - integer - number of points to generate
% mu - 2x1 matrix - mean of cluster
% cov - 2x2 matrix - covariance of cluster
%
% OUTPUT:
% cluster - Nx2 matrix - the generated points in the cluster
function cluster = generateClusters(N, mu, cov)    
    % Inverse of Orthonormal Covariance Transform and Whitening Transfrom
    [eigenvectors, eigenvalues] = eig(cov);
    x = randn(N,2);
    cluster = x*inv(eigenvectors')*sqrt(eigenvalues) + repmat(mu', N, 1);    
end
