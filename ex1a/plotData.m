function plotData(X, y, theta, J, step)
    plot(X(:,2), y, 'rx', 'MarkerSize', 10);
    hold on;

    plot(X(:,2), X*theta, 'g-');
    ylabel('Profit in $10,000s');
    xlabel('Population of City in 10,000s');
    legend(sprintf('Step: %d, Cost: %f', step, J));
        
    hold off;
end