%% DEBUG PREPROCESSING - Find where colors are lost
% This will show you EXACTLY where RGB becomes grayscale
%
% Run on your computer:
% cd 'G:\Fax\Prijava Teme\Clanek za busbare\Code'
% debug_preprocessing

function debug_preprocessing()
    %% Setup
    imgFolder = './data/images';

    % Get first image
    imageFiles = dir(fullfile(imgFolder, '*.png'));
    if isempty(imageFiles)
        imageFiles = dir(fullfile(imgFolder, '*.jpg'));
    end

    if isempty(imageFiles)
        error('No images found!');
    end

    imgPath = fullfile(imageFiles(1).folder, imageFiles(1).name);
    fprintf('========================================\n');
    fprintf('DEBUG: Testing preprocessing pipeline\n');
    fprintf('========================================\n\n');
    fprintf('Test image: %s\n\n', imageFiles(1).name);

    %% Step 1: Load original
    fprintf('STEP 1: Load original image\n');
    img_original = imread(imgPath);
    checkRGB(img_original, 'Original loaded');

    %% Step 2: Convert to uint8
    fprintf('\nSTEP 2: Convert to uint8\n');
    if ~isa(img_original, 'uint8')
        img_uint8 = im2uint8(img_original);
    else
        img_uint8 = img_original;
    end
    checkRGB(img_uint8, 'After uint8 conversion');

    %% Step 3: Force RGB (if grayscale)
    fprintf('\nSTEP 3: Force RGB (convert grayscale to RGB if needed)\n');
    if size(img_uint8, 3) == 1
        fprintf('  Converting grayscale (1 channel) to RGB (3 channels)...\n');
        img_rgb = repmat(img_uint8, [1 1 3]);
    else
        fprintf('  Already RGB (3 channels), no conversion needed\n');
        img_rgb = img_uint8;
    end
    checkRGB(img_rgb, 'After RGB forcing');

    %% Step 4: Auto-crop (if enabled)
    fprintf('\nSTEP 4: Auto-crop white background\n');
    cropThreshold = 0.85;
    img_cropped = autoCropWhiteBackground(img_rgb, cropThreshold);
    if isequal(img_cropped, img_rgb)
        fprintf('  No cropping needed\n');
    else
        fprintf('  Cropped from %dx%d to %dx%d\n', ...
            size(img_rgb,2), size(img_rgb,1), ...
            size(img_cropped,2), size(img_cropped,1));
    end
    checkRGB(img_cropped, 'After auto-crop');

    %% Step 5: Resize
    fprintf('\nSTEP 5: Resize to training size\n');
    trainSize = [64 128];  % [height width]
    img_resized = imresize(img_cropped, [trainSize(1), trainSize(2)]);
    fprintf('  Resized to %dx%d\n', size(img_resized,2), size(img_resized,1));
    checkRGB(img_resized, 'After resize');

    %% Step 6: Normalize to [-1, 1]
    fprintf('\nSTEP 6: Normalize to [-1, 1]\n');
    img_normalized = single(img_resized) / 255.0;  % [0, 1]
    img_normalized = img_normalized * 2 - 1;        % [-1, 1]
    fprintf('  Value range: [%.3f, %.3f]\n', min(img_normalized(:)), max(img_normalized(:)));
    checkRGB(img_normalized, 'After normalization');

    %% Step 7: Augmentation (simulate)
    fprintf('\nSTEP 7: Data augmentation (horizontal flip)\n');
    img_augmented = fliplr(img_normalized);
    checkRGB(img_augmented, 'After augmentation');

    %% Summary
    fprintf('\n========================================\n');
    fprintf('SUMMARY\n');
    fprintf('========================================\n');
    fprintf('If all steps show "TRUE RGB", then preprocessing is correct.\n');
    fprintf('Problem is likely in:\n');
    fprintf('  1. Generator architecture\n');
    fprintf('  2. Training loop\n');
    fprintf('  3. Image saving (generateSynthetic.m)\n');
    fprintf('\n');

    %% Save debug images
    fprintf('Saving debug images...\n');
    debugFolder = './debug_preprocessing';
    if ~exist(debugFolder, 'dir')
        mkdir(debugFolder);
    end

    % Denormalize for saving
    img_to_save = (img_normalized + 1) / 2 * 255;
    img_to_save = uint8(img_to_save);

    imwrite(img_original, fullfile(debugFolder, '1_original.png'));
    imwrite(img_rgb, fullfile(debugFolder, '2_rgb.png'));
    imwrite(img_cropped, fullfile(debugFolder, '3_cropped.png'));
    imwrite(img_resized, fullfile(debugFolder, '4_resized.png'));
    imwrite(img_to_save, fullfile(debugFolder, '5_normalized.png'));

    fprintf('Debug images saved to: %s\n', debugFolder);
    fprintf('Check these images to see if colors are preserved!\n\n');
end

function checkRGB(img, stage)
    % Check if image is truly RGB or grayscale
    if size(img, 3) ~= 3
        fprintf('  [%s] ❌ GRAYSCALE (%d channel)\n', stage, size(img, 3));
        return;
    end

    % Sample pixels
    R = double(img(:,:,1));
    G = double(img(:,:,2));
    B = double(img(:,:,3));

    % Sample 100 random pixels
    numPixels = numel(R);
    sampleSize = min(100, numPixels);
    idx = randperm(numPixels, sampleSize);

    R_sample = R(idx);
    G_sample = G(idx);
    B_sample = B(idx);

    % Calculate MAD
    mad_RG = mean(abs(R_sample - G_sample));
    mad_GB = mean(abs(G_sample - B_sample));
    mad_RB = mean(abs(R_sample - B_sample));

    % For normalized images [-1, 1], scale threshold
    if min(img(:)) < 0
        threshold = 0.02;  % Normalized threshold
    else
        threshold = 5.0;   % uint8 threshold
    end

    if mad_RG < threshold && mad_GB < threshold && mad_RB < threshold
        fprintf('  [%s] ❌ FAKE RGB (R≈G≈B) - MAD: RG=%.3f, GB=%.3f, RB=%.3f\n', ...
            stage, mad_RG, mad_GB, mad_RB);
    else
        fprintf('  [%s] ✅ TRUE RGB - MAD: RG=%.3f, GB=%.3f, RB=%.3f\n', ...
            stage, mad_RG, mad_GB, mad_RB);
    end
end

function imgOut = autoCropWhiteBackground(img, threshold)
    % Auto-crop white background
    % (Same as in preprocessAndLoadDatastore.m)

    % Convert to grayscale for thresholding
    if size(img, 3) == 3
        grayImg = rgb2gray(img);
    else
        grayImg = img;
    end

    % Normalize to [0, 1]
    grayImg = double(grayImg) / 255.0;

    % Threshold: pixels below threshold are "object"
    binaryMask = grayImg < threshold;

    % Find bounding box
    [rows, cols] = find(binaryMask);

    if isempty(rows)
        % No object detected, return original
        imgOut = img;
        return;
    end

    yMin = max(1, min(rows) - 5);
    yMax = min(size(img, 1), max(rows) + 5);
    xMin = max(1, min(cols) - 5);
    xMax = min(size(img, 2), max(cols) + 5);

    % Crop ORIGINAL RGB image
    imgOut = img(yMin:yMax, xMin:xMax, :);
end
