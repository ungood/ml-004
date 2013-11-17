function [theta, costHistory] = gradientDescent(costFunction, initialTheta, alpha, iterations)
%gradientDescent Performs gradient descent to learn theta
%   costFunction should be a function pointer that returns cost(theta) as well as the gradient of cost(theta).

theta = initialTheta;
costHistory = zeros(iterations, 1);

for i = 1:iterations
    [cost, gradient] = costFunction(theta);
    costHistory(i) = cost;
    theta = theta - (alpha * gradient);
end

end
