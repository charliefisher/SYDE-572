close all

STEP = 0.05;

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

% Generating Clusters
cluster_A = generateClusters(N_A, mu_A, cov_A);
cluster_B = generateClusters(N_B, mu_B, cov_B);
cluster_C = generateClusters(N_C, mu_C, cov_C);
cluster_D = generateClusters(N_D, mu_D, cov_D);
cluster_E = generateClusters(N_E, mu_E, cov_E);

% Plot Clusters and Standard Deviation Countours for Case 1
cluster1_fig = figure();
title('Case 1 Clusters and Standard Deviation Contours')
xlabel('Feature 1')
ylabel('Feature 2')
hold on
plotClusterAndContour(cluster_A, mu_A, cov_A, 'red')
plotClusterAndContour(cluster_B, mu_B, cov_B, 'blue')
hold off

% Plot Clusters and Standard Deviation Countours for Case 2
cluster2_fig = figure();
title('Case 2 Clusters and Standard Deviation Contours')
xlabel('Feature 1')
ylabel('Feature 2')
hold on
plotClusterAndContour(cluster_C, mu_C, cov_C, 'red')
plotClusterAndContour(cluster_D, mu_D, cov_D, 'blue')
plotClusterAndContour(cluster_E, mu_E, cov_E, 'green')
hold off

% Calculate True Means and Covariances
mu_A_true = mean(cluster_A, 1)';
mu_B_true = mean(cluster_B, 1)';
mu_C_true = mean(cluster_C, 1)';
mu_D_true = mean(cluster_D, 1)';
mu_E_true = mean(cluster_E, 1)';

cov_A_true = cov(cluster_A);
cov_B_true = cov(cluster_B);
cov_C_true = cov(cluster_C);
cov_D_true = cov(cluster_D);
cov_E_true = cov(cluster_E);

% Classifier Case 1

figure()
title('Case 1 MED GED MAP')
xlabel('Feature 1')
ylabel('Feature 2')
hold on

x1 = -5:STEP:20;
x2 = -5:STEP:20;

[X1, X2] = meshgrid(x1, x2);

classes = MEDClassifier([mu_A_true mu_B_true], X1, X2);

contour(x1, x2, classes, 'Color', 'k');
xlabel('x_1');
ylabel('x_2');

plotClusterAndContour(cluster_A, mu_A_true, cov_A, 'red')
plotClusterAndContour(cluster_B, mu_B_true, cov_B, 'blue')
hold off

% Classifer Case 2

figure()
title('Case 1 MED GED MAP')
xlabel('Feature 1')
ylabel('Feature 2')
hold on

x1 = -5:STEP:25;
x2 = -5:STEP:25;

[X1, X2] = meshgrid(x1, x2);

classes = MEDClassifier([mu_C_true mu_D_true mu_E_true], X1, X2);

contour(x1, x2, classes, 'Color', 'k');
xlabel('x_1');
ylabel('x_2');

plotClusterAndContour(cluster_C, mu_C_true, cov_C_true, 'red')
plotClusterAndContour(cluster_D, mu_D_true, cov_D_true, 'blue')
plotClusterAndContour(cluster_E, mu_E_true, cov_E_true, 'green')
hold off

function classes = MEDClassifier(mus, x, y)
    gridsize = size(x);
    classes = zeros(gridsize(1), gridsize(2));

    for i = 1:gridsize(1)
        for j = 1:gridsize(2)
            d = [];
            for k = 1:size(mus,2)
                d(end+1) = -mus(:,k)'*[x(i,j); y(i,j)] + 0.5*mus(:,k)'*mus(:,k);
            end
            [min_val, min_idx] = min(d);

            classes(i,j) = min_idx;
        end
    end
end


function cluster = generateClusters(N, mu, cov)
    % TWO WAYS OF MAKING THE RANDOM POINTS, THEY WOULD PROBABLY WANT
    % REVERSE WHITENING TRANSFROM?

    % Bivariate Normal Random Numbers Example from randn 
    % points = repmat(mu', N, 1) + randn(N,2)*chol(cov)
    
    % Reverse Whitening Transfrom
    [eigenvectors, eigenvalues] = eig(cov);
    cluster = repmat(mu', N, 1) + randn(N,2)*sqrt(eigenvalues)*inv(eigenvectors');    
end

function plotClusterAndContour(points, mu, cov, c)
    scatter(points(:,1),points(:,2),c)
    [eigenvectors, eigenvalues] = eig(cov);
    plot_ellipse(mu(1), mu(2), atan2(eigenvectors(1,2), eigenvectors(1,1)),sqrt(eigenvalues(1,1)),sqrt(eigenvalues(2,2)),c)
end




