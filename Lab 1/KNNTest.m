clc
clear
close all

% Case 1 Data
N_A = 200;
mu_A = [5 10]';
cov_A = [8 0; 0 4];

N_B = 200;
mu_B = [10 15]';
cov_B = [8 0; 0 4];

% Case 2 Data
N_C = 100;
mu_C = [5 10]';
cov_C = [8 4; 4 40];

N_D = 200;
mu_D = [15 10]';
cov_D = [8 0; 0 8];

N_E = 150;
mu_E = [10 5]';
cov_E = [10 -5; -5 20];

% Generating Clusters
cluster_A = generateClusters(N_A, mu_A, cov_A);
cluster_B = generateClusters(N_B, mu_B, cov_B);
cluster_C = generateClusters(N_C, mu_C, cov_C);
cluster_D = generateClusters(N_D, mu_D, cov_D);
cluster_E = generateClusters(N_E, mu_E, cov_E);

case1 = {cluster_A, cluster_B};
case2 = {cluster_C, cluster_D, cluster_E};

mu_case1 = {mu_A, mu_B};  % store cases as cell arrays
mu_case2 = {mu_C, mu_D, mu_E};  % store cases as cell arrays

cov_case1 = {cov_A, cov_B};  % store cases as cell arrays
cov_case2 = {cov_C, cov_D, cov_E};  % store cases as cell arrays

x1 = -10:0.05:25;
x2 = -10:0.05:25;

[X1, X2] = meshgrid(x1,x2);

[~, knnMap] = KNNClassifier(case2, 5, X1, X2);

%% plots
figure(1)
hold on
surf(X1,X2,knnMap,EdgeColor='none')

figure(2)
hold on
scatter(cluster_C(:,1), cluster_C(:,2), 'b')
scatter(cluster_D(:,1), cluster_D(:,2), 'r')
scatter(cluster_E(:,1), cluster_E(:,2), 'g')

contour(X1,X2,knnMap,'k')