function cluster = generateClusters(N, mu, cov)
    % TWO WAYS OF MAKING THE RANDOM POINTS, THEY WOULD PROBABLY WANT
    % REVERSE WHITENING TRANSFROM?

    % Bivariate Normal Random Numbers Example from randn 
    % points = repmat(mu', N, 1) + randn(N,2)*chol(cov)
    
    % Reverse Whitening Transfrom
    [eigenvectors, eigenvalues] = eig(cov);
    cluster = repmat(mu', N, 1) + randn(N,2)*sqrt(eigenvalues)*inv(eigenvectors');    
end
