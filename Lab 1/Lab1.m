close all
clc
clear

STEP = 0.1;
x1 = -5:STEP:25;
x2 = -5:STEP:25;

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

%% Plot Clusters and Standard Deviation Countours for Case 1
cluster1_fig = figure(1);
hold on

title('Case 1 Clusters and Standard Deviation Contours')
xlabel('x_1')
ylabel('x_2')
xlim([-5,25])
ylim([-5,25])

scatter(cluster_A(:,1), cluster_A(:,2), 'r')
scatter(cluster_B(:,1), cluster_B(:,2), 'b')

plotEllipse(mu_A, cov_A, 'r')
plotEllipse(mu_B, cov_B, 'b')
hold off

%% Plot Clusters and Standard Deviation Countours for Case 2
cluster2_fig = figure(2);
hold on

title('Case 2 Clusters and Standard Deviation Contours')
xlabel('x_1')
ylabel('x_2')
xlim([-5,25])
ylim([-5,25])

scatter(cluster_C(:,1), cluster_C(:,2), 'r')
scatter(cluster_D(:,1), cluster_D(:,2), 'b')
scatter(cluster_E(:,1), cluster_E(:,2), 'g')

plotEllipse(mu_C, cov_C, 'r')
plotEllipse(mu_D, cov_D, 'b')
plotEllipse(mu_E, cov_E, 'g')

hold off

% %% Calculate True Means and Covariances
% mu_A_true = mean(cluster_A, 1)';
% mu_B_true = mean(cluster_B, 1)';
% mu_C_true = mean(cluster_C, 1)';
% mu_D_true = mean(cluster_D, 1)';
% mu_E_true = mean(cluster_E, 1)';
% 
% cov_A_true = cov(cluster_A);
% cov_B_true = cov(cluster_B);
% cov_C_true = cov(cluster_C);
% cov_D_true = cov(cluster_D);
% cov_E_true = cov(cluster_E);

%% Classifier Case 1
[X1, X2] = meshgrid(x1, x2);

med1 = MEDClassifier([mu_A mu_B], X1, X2);
[~, ged1] = GEDClassifier(mu_case1, cov_case1, X1, X2);

figure(3)
hold on

title('Case 1 MED GED MAP')
xlabel('x_1');
ylabel('x_2');
xlim([-5,25])
ylim([-5,25])

contour(x1, x2, med1, EdgeColor='k');
contour(X1, X2, ged1, EdgeColor='#FF8000');

scatter(cluster_A(:,1), cluster_A(:,2), 'r')
scatter(cluster_B(:,1), cluster_B(:,2), 'b')

plotEllipse(mu_A, cov_A, 'r')
plotEllipse(mu_B, cov_B, 'b')
hold off

%% Classifer Case 2
[X1, X2] = meshgrid(x1, x2);

med2 = MEDClassifier([mu_C mu_D mu_E], X1, X2);
[~, ged2] = GEDClassifier(mu_case2, cov_case2, X1, X2);

figure(4)
hold on

title('Case 2 MED GED MAP')
xlabel('x_1');
ylabel('x_2');
xlim([-5,25])
ylim([-5,25])

contour(x1, x2, med2, EdgeColor='k');
contour(X1, X2, ged2, EdgeColor='#FF8000');

scatter(cluster_C(:,1), cluster_C(:,2), 'r')
scatter(cluster_D(:,1), cluster_D(:,2), 'b')
scatter(cluster_E(:,1), cluster_E(:,2), 'g')

plotEllipse(mu_C, cov_C, 'r')
plotEllipse(mu_D, cov_D, 'b')
plotEllipse(mu_E, cov_E, 'g')

hold off