%% BRIGHTEN IMAGES - Post-processing fix for dark/weak colors
% This is a QUICK FIX for already-generated images
% For BEST results, retrain with 600 epochs instead!
%
% Usage:
%   brighten_images        % Process ./outputs/synthetic/
%   brighten_images(1.5)   % Custom brightness factor

function brighten_images(brightenFactor, colorBoost)
    if nargin < 1
        brightenFactor = 1.95;  % Default: 95% brighter
    end
    if nargin < 2
        colorBoost = 1.3;       % Default: 30% more saturated
    end

    fprintf('========================================\n');
    fprintf('Brightening Synthetic Images\n');
    fprintf('========================================\n\n');
    fprintf('Brightness factor: %.2fx\n', brightenFactor);
    fprintf('Color boost factor: %.2fx\n\n', colorBoost);

    % Folders
    inputFolder = './outputs/synthetic';
    outputFolder = './outputs/synthetic_brightened';

    if ~exist(inputFolder, 'dir')
        error('Input folder not found: %s', inputFolder);
    end

    if ~exist(outputFolder, 'dir')
        mkdir(outputFolder);
    end

    % Find images
    imageFiles = dir(fullfile(inputFolder, '*.png'));
    if isempty(imageFiles)
        imageFiles = dir(fullfile(inputFolder, '*.jpg'));
    end

    if isempty(imageFiles)
        error('No images found in %s', inputFolder);
    end

    fprintf('Processing %d images...\n\n', numel(imageFiles));

    % Process
    for i = 1:numel(imageFiles)
        % Load
        imgPath = fullfile(imageFiles(i).folder, imageFiles(i).name);
        img = imread(imgPath);

        % Convert to HSV
        img_hsv = rgb2hsv(img);

        % Increase Value (brightness)
        img_hsv(:,:,3) = min(1, img_hsv(:,:,3) * brightenFactor);

        % Increase Saturation (color strength)
        img_hsv(:,:,2) = min(1, img_hsv(:,:,2) * colorBoost);

        % Convert back to RGB
        img_bright = hsv2rgb(img_hsv);
        img_bright = uint8(img_bright * 255);

        % Save
        [~, name, ext] = fileparts(imageFiles(i).name);
        outputPath = fullfile(outputFolder, [name ext]);
        imwrite(img_bright, outputPath);

        % Progress
        if mod(i, 100) == 0 || i == numel(imageFiles)
            fprintf('  Progress: %d/%d (%.1f%%)\n', i, numel(imageFiles), 100*i/numel(imageFiles));
        end
    end

    fprintf('\nDone!\n');
    fprintf('Brightened images saved to: %s\n', outputFolder);
    fprintf('\n');
    fprintf('Compare:\n');
    fprintf('  Original: %s\n', inputFolder);
    fprintf('  Brightened: %s\n', outputFolder);
    fprintf('\n');
    fprintf('NOTE: This is a post-processing fix, not a true solution.\n');
    fprintf('For BEST results, retrain model with 600 epochs!\n');
    fprintf('See: SOLUTION_WEAK_COLORS.md\n');
end
