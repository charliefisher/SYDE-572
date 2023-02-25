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

mu_case1 = {mu_A, mu_B};  % store cases as cell arrays
mu_case2 = {mu_C, mu_D, mu_E};  % store cases as cell arrays

cov_case1 = {cov_A, cov_B};  % store cases as cell arrays
cov_case2 = {cov_C, cov_D, cov_E};  % store cases as cell arrays

sizes1 = [N_A N_B];
sizes2 = [N_C N_D N_E];

x1 = -30:0.5:60;
x2 = -30:0.5:60;

[X1, X2] = meshgrid(x1,x2);

[g, h] = MAPClassifier(mu_case2, cov_case2, sizes2, X1, X2);

%% plots
figure(1)
hold on
surf(X1,X2,h,EdgeColor='none')


figure(2)
hold on
% surf(X1, X2, g(:,:,1), FaceColor='b', EdgeColor='k')
% surf(X1, X2, g(:,:,2), FaceColor='r', EdgeColor='k')
% surf(X1, X2, g(:,:,3), FaceColor='g', EdgeColor='k')

contour(X1,X2,h,'k')

% c1 = double(abs(cls(:,:,1) - cls(:,:,2)) < 0.1);
% c2 = abs(cls(:,:,1) - cls(:,:,3)) < 0.1;
% c3 = abs(cls(:,:,2) - cls(:,:,3)) < 0.1;
% 
% figure(3)
% surf(X1,X2,c1)



