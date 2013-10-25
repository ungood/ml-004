clear; close all; clc;

% load data
data = load('data.txt');
X = data(:, 1); y = data(:, 2);
m = length(y);

% plot data
figure;

X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1);
iterations = 1500;
alpha = 0.01;
J = computeCost(X, y, theta);

for i = [0:5]
    plotData(X, y, theta, J, i);
    [theta, J] = gradientDescentStep(X, y, theta, alpha);
    pause;
end

for i = [0:iterations]
    [theta, J] = gradientDescentStep(X, y, theta, alpha, i);
end

plotData(X, y, theta, J, 1500);