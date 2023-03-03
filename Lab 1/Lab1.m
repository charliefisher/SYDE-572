close all; clear; clc;  % cleanup workspace

% set seed for random numbers to make results reproduciple
% TODO: remove me
rng(1);

%% Class Data %%

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

%% Plot Clusters and Standard Deviation Countours %%

% Case 1
clusters1_fig = figure;
xlabel('x_1');
ylabel('x_2');
xlim([-5 25]);
ylim([-5 25]);
hold on;

scatter(cluster_A(:,1), cluster_A(:,2), 'r')
scatter(cluster_B(:,1), cluster_B(:,2), 'b')
plotUnitStdContour(mu_A, cov_A, 'r');
plotUnitStdContour(mu_B, cov_B, 'b');
legend('Class A', 'Class B');

% Case 2
clusters2_fig = figure;
xlabel('x_1');
ylabel('x_2');
xlim([-5 25]);
ylim([-5 25]);
hold on;

scatter(cluster_C(:,1), cluster_C(:,2), 'r');
scatter(cluster_D(:,1), cluster_D(:,2), 'b');
scatter(cluster_E(:,1), cluster_E(:,2), 'g');
plotUnitStdContour(mu_C, cov_C, 'r');
plotUnitStdContour(mu_D, cov_D, 'b');
plotUnitStdContour(mu_E, cov_E, 'g');
legend('Class C', 'Class D', 'Class E');

saveas(clusters1_fig, 'clusters_case1.png');
saveas(clusters2_fig, 'clusters_case2.png');

%% Create Classifiers  %%

% setup meshgrid for classification
STEP = 0.1;
x1 = -5:STEP:25;
x2 = -5:STEP:25;
[X1, X2] = meshgrid(x1, x2);

% get prototypes for NN
nn_A = knnPrototype(X1, X2, cluster_A, 1);
nn_B = knnPrototype(X1, X2, cluster_B, 1);
nn_C = knnPrototype(X1, X2, cluster_C, 1);
nn_D = knnPrototype(X1, X2, cluster_D, 1);
nn_E = knnPrototype(X1, X2, cluster_E, 1);

% get prototypes for 5NN
nn5_A = knnPrototype(X1, X2, cluster_A, 200);
nn5_B = knnPrototype(X1, X2, cluster_B, 200);
nn5_C = knnPrototype(X1, X2, cluster_C, 100);
nn5_D = knnPrototype(X1, X2, cluster_D, 200);
nn5_E = knnPrototype(X1, X2, cluster_E, 150);

% descriminant functions
MED = @(x, zk, ~) (-zk'*x + 0.5*zk'*zk);
GED = @(x, zk, Sk) ((x-zk)'*inv(Sk)*(x-zk));
MAP = @(x, zk, Sk, Nk) (2*log(Nk) - log(det(Sk)) - (x-zk)'*inv(Sk)*(x-zk));

% classifiers that work on Nx2 matrix of points
MEDClassifier = @(X, mu_cell) genericClassifier(X, MED, @min, mu_cell);
GEDClassifier = @(X, mu_cell, cov_cell) genericClassifier(X, GED, @min, mu_cell, cov_cell);
MAPClassifier = @(X, mu_cell, cov_cell, N_cell) genericClassifier(X, MAP, @max, mu_cell, cov_cell, N_cell);

% build MED, GED, MAP, NN, 5NN classifiers for each case
MED_1 = @(X) MEDClassifier(X, mu_case1);
MED_2 = @(X) MEDClassifier(X, mu_case2);

GED_1 = @(X) GEDClassifier(X, mu_case1, cov_case1);
GED_2 = @(X) GEDClassifier(X, mu_case2, cov_case2);

MAP_1 = @(X) MAPClassifier(X, mu_case1, cov_case1, {N_A, N_B});
MAP_2 = @(X) MAPClassifier(X, mu_case2, cov_case2, {N_C, N_D, N_E});

NN_1 = @(X) MEDClassifier(X, {nn_A, nn_B});
NN_2 = @(X) MEDClassifier(X, {nn_C, nn_D, nn_E});

NN5_1 = @(X) MEDClassifier(X, {nn5_A, nn5_B});
NN5_2 = @(X) MEDClassifier(X, {nn5_C, nn5_D, nn5_E});

%% Determine Decision Boundaries %%

% MED, GED, and MAP
med1 = classifyMeshgrid(X1, X2, MED_1);
ged1 = classifyMeshgrid(X1, X2, GED_1);
map1 = classifyMeshgrid(X1, X2, MAP_1);
med2 = classifyMeshgrid(X1, X2, MED_2);
ged2 = classifyMeshgrid(X1, X2, GED_2);
map2 = classifyMeshgrid(X1, X2, MAP_2);

% NN and 5NN
med1_nn = classifyMeshgrid(X1, X2, NN_1);
med2_nn = classifyMeshgrid(X1, X2, NN_2);
med1_5nn = classifyMeshgrid(X1, X2, NN5_1);
med2_5nn = classifyMeshgrid(X1, X2, NN5_2);

%% Plot Decision Boundaries %%

% plot case 1 with cluster, standard deviation contour, and decision boundaries
decision1_fig = figure;
xlabel('x_1');
ylabel('x_2');
xlim([-5 25]);
ylim([-5 25]);
hold on;

scatter(cluster_A(:,1), cluster_A(:,2), 'r')
scatter(cluster_B(:,1), cluster_B(:,2), 'b')

plotUnitStdContour(mu_A, cov_A, 'r')
plotUnitStdContour(mu_B, cov_B, 'b')

contour(X1, X2, med1, 1, LineWidth=2, EdgeColor='k');
contour(X1, X2, ged1, 1, LineWidth=2, EdgeColor='#FF8000');
contour(X1, X2, map1, 1, LineWidth=2, EdgeColor='y');
legend('Class A', 'Class B');  % TODO: fix me

nn5_1_fig = figure;
xlabel('x_1');
ylabel('x_2');
xlim([-5 25]);
ylim([-5 25]);
hold on;

scatter(cluster_A(:,1), cluster_A(:,2), 'r')
scatter(cluster_B(:,1), cluster_B(:,2), 'b')

contour(X1, X2, med1_5nn, 1, LineWidth=2, EdgeColor='k');
legend('Class A', 'Class B');  % TODO: fix me

% plot case 2 with cluster, standard deviation contour, and decision boundaries
decision2_fig = figure;
xlabel('x_1');
ylabel('x_2');
xlim([-5 25]);
ylim([-5 25]);
hold on;

scatter(cluster_C(:,1), cluster_C(:,2), 'r')
scatter(cluster_D(:,1), cluster_D(:,2), 'b')
scatter(cluster_E(:,1), cluster_E(:,2), 'g')

plotUnitStdContour(mu_C, cov_C, 'r')
plotUnitStdContour(mu_D, cov_D, 'b')
plotUnitStdContour(mu_E, cov_E, 'g')

for k=1:3
    contour(X1, X2, med2 == k, 1, LineWidth=2, EdgeColor='k');
    contour(X1, X2, ged2 == k, 1, LineWidth=2, EdgeColor='#FF8000');
    contour(X1, X2, map2 == k, 1, LineWidth=2, EdgeColor='y');
end
legend('Class C', 'Class D', 'Class E'); % TODO: fix me

nn_2_fig = figure;
xlabel('x_1');
ylabel('x_2');
xlim([-5 25]);
ylim([-5 25]);
hold on;

scatter(cluster_C(:,1), cluster_C(:,2), 'r')
scatter(cluster_D(:,1), cluster_D(:,2), 'b')
scatter(cluster_E(:,1), cluster_E(:,2), 'g')

for k=1:3
    contour(X1, X2, med2_nn == k, 1, LineWidth=2, EdgeColor='k');
end
legend('Class C', 'Class D', 'Class E'); % TODO: fix me

nn5_2_fig = figure;
xlabel('x_1');
ylabel('x_2');
xlim([-5 25]);
ylim([-5 25]);
hold on;

scatter(cluster_C(:,1), cluster_C(:,2), 'r')
% scatter(cluster_D(:,1), cluster_D(:,2), 'b')
% scatter(cluster_E(:,1), cluster_E(:,2), 'g')

for k=1:3
    contour(X1, X2, med2_5nn == k, 1, LineWidth=2, EdgeColor='k');
end
% legend('Class C', 'Class D', 'Class E'); % TODO: fix me

saveas(decision1_fig, 'decision_boundaries_case1.png');
saveas(decision2_fig, 'decision_boundaries_case2.png');
saveas(nn_2_fig, 'nn_case2.png');
saveas(nn5_1_fig, '5nn_case1.png');
saveas(nn5_2_fig, '5nn_case2.png');


%% Error Analysis %%

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
