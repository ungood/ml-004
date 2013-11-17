function [h, display_array] = displayPredictions(predictions, example_width)
%DISPLAYDATA Display 2D data in a nice grid
%   [h, display_array] = DISPLAYDATA(predictions)

% Set example_width automatically if not passed in
if ~exist('example_width', 'var') || isempty(example_width) 
	example_width = 20;
end

example_height = example_width;

% Compute rows, cols
m = size(predictions, 1);

% Compute number of items to display
rows = floor(sqrt(m));
cols = ceil(m / rows);
reshaped = reshape(predictions, rows, cols);
pad = 1;

% Copy each example into a patch on the display array
curr_ex = 1;
for j = 1:rows
	for i = 1:cols
        x = pad + (j - 1) * (example_width + pad) + (2);
        y = pad + (i - 1) * (example_height + pad) + (4);
        digit = sprintf("%d", reshaped(j, i));
		text(x, y, digit);
	end
end

end
