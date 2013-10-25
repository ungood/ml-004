function [theta, J] = gradientDescentStep(X, y, theta, alpha)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples

X_Theta = X * theta - y;
delta = alpha * X_Theta / m;

theta = theta - (X' * delta);
J = computeCost(X, y, theta);

end
