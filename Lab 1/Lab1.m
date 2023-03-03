close all; clear; clc;  % cleanup workspace

% set seed for random numbers to make results reproduciple
rng(1);

%% Class Data %%

% case 1 Data
N_A = 200;
mu_A = [5 10]';
cov_A = [8 0; 0 4];

N_B = 200;
mu_B = [10 15]';
cov_B = [8 0; 0 4];

% case 2 Data
N_C = 100;
mu_C = [5 10]';
cov_C = [8 4; 4 40];

N_D = 200;
mu_D = [15 10]';
cov_D = [8 0; 0 8];

N_E = 150;
mu_E = [10 5]';
cov_E = [10 -5; -5 20];

%% Generate Clusters %%

cluster_A = generateClusters(N_A, mu_A, cov_A);
cluster_B = generateClusters(N_B, mu_B, cov_B);
cluster_C = generateClusters(N_C, mu_C, cov_C);
cluster_D = generateClusters(N_D, mu_D, cov_D);
cluster_E = generateClusters(N_E, mu_E, cov_E);

% store cases as cell arrays
mu_case1 = {mu_A, mu_B};
mu_case2 = {mu_C, mu_D, mu_E};

cov_case1 = {cov_A, cov_B};
cov_case2 = {cov_C, cov_D, cov_E};

clusters_case1 = {cluster_A, cluster_B};
clusters_case2 = {cluster_C, cluster_D, cluster_E};

%% Plot Clusters and Standard Deviation Countours %%

% case 1 clusters
clusters1_fig = plotCluster(clusters_case1, 1);
plotUnitStdContours(mu_case1, cov_case1);

% case 2 clusters
clusters2_fig = plotCluster(clusters_case2, 2);
plotUnitStdContours(mu_case2, cov_case2);

saveas(clusters1_fig, 'clusters_case1.png');
saveas(clusters2_fig, 'clusters_case2.png');

%% Create Classifiers  %%

% setup meshgrid for classification
STEP = 0.1;
x1 = -2:STEP:18;
x2 = 2:STEP:25;
[X1_case1, X2_case1] = meshgrid(x1, x2);
x1 = -7:STEP:25;
x2 = -10:STEP:25;
[X1_case2, X2_case2] = meshgrid(x1, x2);

% descriminant functions
GED = @(x, zk, Sk) ((x-zk)'*inv(Sk)*(x-zk));
MAP = @(x, zk, Sk, Nk) (2*log(Nk) - log(det(Sk)) - (x-zk)'*inv(Sk)*(x-zk));

% classifiers that work on Nx2 matrix of points
GEDClassifier = @(X, mu_cell, cov_cell) genericClassifier(X, GED, @min, mu_cell, cov_cell);
MAPClassifier = @(X, mu_cell, cov_cell, N_cell) genericClassifier(X, MAP, @max, mu_cell, cov_cell, N_cell);

% build MED, GED, MAP, NN, 5NN classifiers for each case
MED_1 = @(X) MEDClassifier(X, mu_case1);
MED_2 = @(X) MEDClassifier(X, mu_case2);

GED_1 = @(X) GEDClassifier(X, mu_case1, cov_case1);
GED_2 = @(X) GEDClassifier(X, mu_case2, cov_case2);

MAP_1 = @(X) MAPClassifier(X, mu_case1, cov_case1, {N_A, N_B});
MAP_2 = @(X) MAPClassifier(X, mu_case2, cov_case2, {N_C, N_D, N_E});

NN_1 = @(X) kNNClassifier(X, clusters_case1, 1);
NN_2 = @(X) kNNClassifier(X, clusters_case2, 1);

NN5_1 = @(X) kNNClassifier(X, clusters_case1, 5);
NN5_2 = @(X) kNNClassifier(X, clusters_case2, 5);

%% Determine Decision Boundaries %%

% MED, GED, and MAP
med1 = classifyMeshgrid(X1_case1, X2_case1, MED_1);
ged1 = classifyMeshgrid(X1_case1, X2_case1, GED_1);
map1 = classifyMeshgrid(X1_case1, X2_case1, MAP_1);
med2 = classifyMeshgrid(X1_case2, X2_case2, MED_2);
ged2 = classifyMeshgrid(X1_case2, X2_case2, GED_2);
map2 = classifyMeshgrid(X1_case2, X2_case2, MAP_2);

% NN and 5NN
med1_nn = classifyMeshgrid(X1_case1, X2_case1, NN_1);
med1_5nn = classifyMeshgrid(X1_case1, X2_case1, NN5_1);
med2_nn = classifyMeshgrid(X1_case2, X2_case2, NN_2);
med2_5nn = classifyMeshgrid(X1_case2, X2_case2, NN5_2);

%% Plot Decision Boundaries %%

% case 1 boundaries for MED, GED, and MAP
decision1_fig = plotCluster(clusters_case1, 1);
plotUnitStdContours(mu_case1, cov_case1);
plotDecisionBoundary(X1_case1, X2_case1, {med1, ged1, map1}, {'MED', 'GED', 'MAP'});

% case 1 boundaries for NN and 5NN
decision1_nn_fig = plotCluster(clusters_case1, 1);
plotDecisionBoundary(X1_case1, X2_case1, {med1_nn}, {'NN'});

decision1_5nn_fig = plotCluster(clusters_case1, 1);
plotDecisionBoundary(X1_case1, X2_case1, {med1_5nn}, {'5NN'});

% case 2 boundaries for MED, GED, and MAP
decision2_fig = plotCluster(clusters_case2, 2);
plotUnitStdContours(mu_case2, cov_case2);
plotDecisionBoundary(X1_case2, X2_case2, {med2, ged2, map2}, {'MED', 'GED', 'MAP'});

% case 2 boundaries for NN and 5NN
decision2_nn_fig = plotCluster(clusters_case2, 2);
plotDecisionBoundary(X1_case2, X2_case2, {med2_nn}, {'NN'});

decision2_5nn_fig = plotCluster(clusters_case2, 2);
plotDecisionBoundary(X1_case2, X2_case2, {med2_5nn}, {'5NN'});

saveas(decision1_fig, 'decision_case1.png');
saveas(decision1_nn_fig, 'decision_nn_case1.png');
saveas(decision1_5nn_fig, 'decision_5nn_case1.png');
saveas(decision2_fig, 'decision_case2.png');
saveas(decision2_nn_fig, 'decision_nn_case2.png');
saveas(decision2_5nn_fig, 'decision_5nn_case2.png');

%% Error Analysis %%

case1_train = [cluster_A; cluster_B];
case1_labels = [
    ones(N_A, 1);
    ones(N_B, 1)*2;
];

case2_train = [cluster_C; cluster_D; cluster_E];
case2_labels = [
    ones(N_C, 1);
    ones(N_D, 1)*2;
    ones(N_E, 1)*3;
];

% run each classifier on training set
[~, P_e] = testClassifier(case1_train, case1_labels, MED_1)
[~, P_e] = testClassifier(case1_train, case1_labels, GED_1)
[~, P_e] = testClassifier(case1_train, case1_labels, MAP_1)
[~, P_e] = testClassifier(case1_train, case1_labels, NN_1)
[~, P_e] = testClassifier(case1_train, case1_labels, NN5_1)
[~, P_e] = testClassifier(case2_train, case2_labels, MED_2)
[~, P_e] = testClassifier(case2_train, case2_labels, GED_2)
[~, P_e] = testClassifier(case2_train, case2_labels, MAP_2)
[~, P_e] = testClassifier(case2_train, case2_labels, NN_2)
[~, P_e] = testClassifier(case2_train, case2_labels, NN5_2)

% generate new cluster for test sets
A_test = generateClusters(N_A, mu_A, cov_A);
B_test = generateClusters(N_B, mu_B, cov_B);
C_test = generateClusters(N_C, mu_C, cov_C);
D_test = generateClusters(N_D, mu_D, cov_D);
E_test = generateClusters(N_E, mu_E, cov_E);

% combine clusters for each case and add true labels
case1_test = [A_test; B_test];
case1_labels = [
    ones(N_A, 1);
    ones(N_B, 1)*2;
];

case2_test = [C_test; D_test; E_test];
case2_labels = [
    ones(N_C, 1);
    ones(N_D, 1)*2;
    ones(N_E, 1)*3;
];

% run each classifier on test set
[confusion, P_e] = testClassifier(case1_test, case1_labels, MED_1)
[confusion, P_e] = testClassifier(case1_test, case1_labels, GED_1)
[confusion, P_e] = testClassifier(case1_test, case1_labels, MAP_1)
[confusion, P_e] = testClassifier(case1_test, case1_labels, NN_1)
[confusion, P_e] = testClassifier(case1_test, case1_labels, NN5_1)
[confusion, P_e] = testClassifier(case2_test, case2_labels, MED_2)
[confusion, P_e] = testClassifier(case2_test, case2_labels, GED_2)
[confusion, P_e] = testClassifier(case2_test, case2_labels, MAP_2)
[confusion, P_e] = testClassifier(case2_test, case2_labels, NN_2)
[confusion, P_e] = testClassifier(case2_test, case2_labels, NN5_2)
