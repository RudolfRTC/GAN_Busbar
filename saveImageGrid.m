function saveImageGrid(images, filename, params)
%SAVEIMAGEGRID Save a grid of generated images to file
%
% Inputs:
%   images - 4D array [H x W x C x N] in range [-1, 1]
%   filename - output filename (PNG)
%   params - struct with .numChannels
%
% Output:
%   Saves image grid to file

    % Extract parameters
    [H, W, C, N] = size(images);

    % Denormalize from [-1, 1] to [0, 1]
    images = (images + 1) / 2;
    images = max(min(images, 1), 0); % Clamp to [0, 1]

    % Determine grid size (square grid)
    gridSize = ceil(sqrt(N));

    % Create empty grid
    gridH = gridSize * H;
    gridW = gridSize * W;

    if C == 1
        grid = ones(gridH, gridW);
    else
        grid = ones(gridH, gridW, C);
    end

    % Fill grid with images
    idx = 1;
    for row = 1:gridSize
        for col = 1:gridSize
            if idx > N
                break;
            end

            % Calculate position
            rowStart = (row - 1) * H + 1;
            rowEnd = row * H;
            colStart = (col - 1) * W + 1;
            colEnd = col * W;

            % Place image
            if C == 1
                grid(rowStart:rowEnd, colStart:colEnd) = images(:, :, 1, idx);
            else
                grid(rowStart:rowEnd, colStart:colEnd, :) = images(:, :, :, idx);
            end

            idx = idx + 1;
        end
        if idx > N
            break;
        end
    end

    % Save to file
    imwrite(grid, filename);
end
