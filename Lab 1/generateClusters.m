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
    x = randn(2, N);
    mu = repmat(mu, 1, N);
    cluster = inv(eigenvectors')*sqrt(eigenvalues)*x + mu; 
    % convert cluster back to Nx2 matrix
    cluster = cluster';
    assert(isequal(size(cluster), [N 2]));  % output should be Nx2
end
