% Case 1 Data
N_A = 200;
mu_A = [5 10]';
cov_A = [8 0;
        0 4];

N_B = 200;
mu_B = [10 15]';
cov_B = [8 0;
        0 4];

% Case 2 Data
N_C = 100;
mu_C = [5 10]';
cov_C = [8 4;
        4 40];

N_D = 200;
mu_D = [15 10]';
cov_D = [8 0;
        0 8];

N_E = 150;
mu_E = [10 5]';
cov_E = [10 -5
        -5 20];

% 2 Generating Clusters

cluster_A = generateClusters(N_A, mu_A, cov_A);
cluster_B = generateClusters(N_B, mu_B, cov_B);

cluster_C = generateClusters(N_C, mu_C, cov_C);
cluster_D = generateClusters(N_D, mu_D, cov_D);
cluster_E = generateClusters(N_E, mu_E, cov_E);

cluster1_fig = figure();
title('Case 1 Clusters and Standard Deviation Contours')
xlabel('Feature 1')
ylabel('Feature 2')
hold on
plotCluster(cluster_A, mu_A, cov_A, 'red')
plotCluster(cluster_B, mu_B, cov_B, 'blue')
hold off

cluster2_fig = figure();
title('Case 2 Clusters and Standard Deviation Contours')
xlabel('Feature 1')
ylabel('Feature 2')
hold on
plotCluster(cluster_C, mu_C, cov_C, 'red')
plotCluster(cluster_D, mu_D, cov_D, 'blue')
plotCluster(cluster_E, mu_E, cov_E, 'green')
hold off




function cluster = generateClusters(N, mu, cov)
    % TWO WAYS OF MAKING THE RANDOM POINTS, THEY WOULD PROBABLY WANT
    % REVERSE WHITENING TRANSFROM?

    % Bivariate Normal Random Numbers Example from randn 
    % points = repmat(mu', N, 1) + randn(N,2)*chol(cov)
    
    % Reverse Whitening Transfrom
    [eigenvectors, eigenvalues] = eig(cov);
    cluster = repmat(mu', N, 1) + randn(N,2)*sqrt(eigenvalues)*inv(eigenvectors');    
end

function plotCluster(points, mu, cov, c)
    scatter(points(:,1),points(:,2),c)
    [eigenvectors, eigenvalues] = eig(cov);
    plot_ellipse(mu(1), mu(2), atan2(eigenvectors(1,2), eigenvectors(1,1)),sqrt(eigenvalues(1,1)),sqrt(eigenvalues(2,2)),c)
end




