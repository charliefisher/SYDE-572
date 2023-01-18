% SYDE 572 Lab 0 - Matlab Introduction
% Name: Charlie Fisher
% Date: Jan 17, 2023

clear all % clear all variables from memory
close all % close all open figures

mu = [5 0]'; % the mean of the pdf
sigma = [1 0; 0 4]; % the variance matrix of the pdf

x1 = (mu(1)-2.5*sigma(1,1)):sigma(1,1)/10:(mu(1)+2.5*sigma(1,1)); % range of the random variable x1
x2 = (mu(2)-2.5*sigma(2,2)):sigma(2,2)/10:(mu(2)+2.5*sigma(2,2)); % range of the random variable x2

% Calculate the pdf
y = Gauss2d(x1,x2,mu,sigma);

% Show a 3-D plot of the pdf
figure;
subplot(2,1,1);
surf(x1,x2,y);
xlabel("x_1");
ylabel("x_2");

% Show contours of the pdf
subplot(2,1,2);
contour(x1,x2,y);
xlabel("x_{1}");
ylabel("x_{2}");
axis square;

% Show a colour map of the pdf
figure;
imagesc(x1,x2,y);
xlabel("x_{1}");
ylabel("x_{2}");

% Plot region where pdf is greater than 0.1
z = (y>0.05);

figure;
colormap summer;
imagesc(x1,x2,z);
hold on; % allow us to plot more on the same figure
plot(mu(1,1), mu(2,1), "gx"); % plot the mean
xlabel("x_{1}");
ylabel("x_{2}");
