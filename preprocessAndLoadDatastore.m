function [mbq, params] = preprocessAndLoadDatastore(params)
%PREPROCESSANDLOADDATASTORE Load and preprocess images with auto-crop and augmentation
%
% Inputs:
%   params - struct with fields:
%     .dataFolder - path to image folder
%     .imageSize - target image size
%     .autoCrop - whether to auto-crop white background
%     .cropThreshold - threshold for white detection
%     .miniBatchSize - batch size
%     .numChannels - will be auto-detected
%
% Outputs:
%   mbq - minibatchqueue for training
%   params - updated params struct with detected numChannels

    %% Check if data folder exists
    if ~exist(params.dataFolder, 'dir')
        error('Data folder not found: %s\nPlease create it and add images.', params.dataFolder);
    end

    %% Load image filenames
    fprintf('  Scanning for images in: %s\n', params.dataFolder);
    imageFiles = [
        dir(fullfile(params.dataFolder, '*.jpg'));
        dir(fullfile(params.dataFolder, '*.jpeg'));
        dir(fullfile(params.dataFolder, '*.png'));
        dir(fullfile(params.dataFolder, '*.bmp'))
    ];

    if isempty(imageFiles)
        error('No images found in: %s\nSupported formats: .jpg, .png, .bmp', params.dataFolder);
    end

    fprintf('  Found %d images\n', numel(imageFiles));

    %% Auto-detect RGB vs Grayscale
    fprintf('  Auto-detecting color format...\n');
    numRGB = 0;
    numGray = 0;

    % Sample first 10 images to detect
    numSamples = min(10, numel(imageFiles));
    for i = 1:numSamples
        imgPath = fullfile(imageFiles(i).folder, imageFiles(i).name);
        img = imread(imgPath);
        if size(img, 3) == 3
            numRGB = numRGB + 1;
        else
            numGray = numGray + 1;
        end
    end

    % Decide: if majority is RGB, convert all to RGB; else grayscale
    if numRGB >= numGray
        params.numChannels = 3;
        fprintf('  Detected: RGB images (will convert grayscale to RGB if any)\n');
    else
        params.numChannels = 1;
        fprintf('  Detected: Grayscale images (will convert RGB to grayscale if any)\n');
    end

    %% Create image datastore
    fullPaths = cellfun(@(folder, name) fullfile(folder, name), ...
        {imageFiles.folder}', {imageFiles.name}', 'UniformOutput', false);
    imds = imageDatastore(fullPaths);

    %% Preprocessing function
    function imgOut = preprocessImage(img)
        % Read if path
        if ischar(img) || isstring(img)
            img = imread(img);
        end

        % Convert to uint8 if needed
        if ~isa(img, 'uint8')
            img = im2uint8(img);
        end

        % Handle RGB vs Grayscale
        if params.numChannels == 3
            if size(img, 3) == 1
                % Convert grayscale to RGB
                img = repmat(img, [1 1 3]);
            end
        else
            if size(img, 3) == 3
                % Convert RGB to grayscale
                img = rgb2gray(img);
            end
        end

        % Auto-crop white background (optional)
        if params.autoCrop
            img = autoCropWhiteBackground(img, params.cropThreshold);
        end

        % Resize to target size
        img = imresize(img, [params.imageSize, params.imageSize]);

        % Convert to single and normalize to [-1, 1]
        imgOut = single(img) / 255.0;  % [0, 1]
        imgOut = imgOut * 2 - 1;        % [-1, 1]
    end

    %% Apply preprocessing with transform
    imds.ReadFcn = @(x) preprocessImage(x);

    %% Data Augmentation (strong augmentation for small dataset)
    augmenter = imageDataAugmenter( ...
        'RandXReflection', true, ...                    % Random horizontal flip
        'RandRotation', [-10 10], ...                   % Small rotation
        'RandXTranslation', [-10 10], ...               % Horizontal shift
        'RandYTranslation', [-10 10], ...               % Vertical shift
        'RandXScale', [0.9 1.1], ...                    % Slight zoom
        'RandYScale', [0.9 1.1]);

    % For augmentedImageDatastore, we need to apply it differently
    % Since our images are already preprocessed to [-1,1], we'll create
    % a custom augmentation function

    function imgAug = augmentImage(img)
        % Apply augmentation to image in [-1, 1] range

        % Convert back to [0, 1] for augmenter
        imgTemp = (img + 1) / 2;

        % Random horizontal flip
        if rand > 0.5
            imgTemp = fliplr(imgTemp);
        end

        % Random small rotation
        angle = (rand - 0.5) * 20; % -10 to +10 degrees
        imgTemp = imrotate(imgTemp, angle, 'bilinear', 'crop');

        % Random translation (simple shift)
        shiftX = randi([-5, 5]);
        shiftY = randi([-5, 5]);
        imgTemp = imtranslate(imgTemp, [shiftX, shiftY], 'FillValues', 1);

        % Random brightness/contrast jitter
        brightnessJitter = (rand - 0.5) * 0.2; % -0.1 to +0.1
        contrastJitter = 0.9 + rand * 0.2;      % 0.9 to 1.1
        imgTemp = imgTemp * contrastJitter + brightnessJitter;
        imgTemp = max(min(imgTemp, 1), 0); % Clamp to [0, 1]

        % Small Gaussian noise
        noiseStd = 0.01;
        imgTemp = imgTemp + noiseStd * randn(size(imgTemp), 'like', imgTemp);
        imgTemp = max(min(imgTemp, 1), 0);

        % Convert back to [-1, 1]
        imgAug = imgTemp * 2 - 1;
    end

    %% Create transformed datastore with augmentation
    tds = transform(imds, @(x) augmentImage(x));

    %% Adjust batch size if needed
    numImages = numel(imageFiles);

    % Save number of images to params for later use
    params.numImages = numImages;

    if params.miniBatchSize > numImages
        fprintf('  WARNING: Batch size (%d) > number of images (%d)\n', ...
            params.miniBatchSize, numImages);
        params.miniBatchSize = max(1, floor(numImages / 2));
        fprintf('  Adjusted batch size to: %d\n', params.miniBatchSize);
    end

    %% Create minibatchqueue
    if params.numChannels == 3
        outputFormat = 'SSCB'; % Spatial x Spatial x Channel x Batch
    else
        outputFormat = 'SSCB';
    end

    mbq = minibatchqueue(tds, ...
        'MiniBatchSize', params.miniBatchSize, ...
        'MiniBatchFormat', outputFormat, ...
        'OutputEnvironment', params.executionEnvironment);

    fprintf('  Preprocessing complete!\n');
end

%% Auto-crop white background
function imgCropped = autoCropWhiteBackground(img, threshold)
    % Auto-crop white background around object
    %
    % Inputs:
    %   img - input image (uint8)
    %   threshold - white threshold (0-1), e.g., 0.85 means > 0.85 is white
    %
    % Output:
    %   imgCropped - cropped image

    try
        % Convert to grayscale for thresholding
        if size(img, 3) == 3
            imgGray = rgb2gray(img);
        else
            imgGray = img;
        end

        % Normalize to [0, 1]
        imgGray = double(imgGray) / 255.0;

        % Create mask: white background = 1, object = 0
        whiteMask = imgGray > threshold;

        % Invert to get object mask
        objectMask = ~whiteMask;

        % Morphological operations to clean up mask
        se = strel('disk', 3);
        objectMask = imopen(objectMask, se);
        objectMask = imclose(objectMask, se);

        % Find largest connected component (the main object)
        cc = bwconncomp(objectMask);
        if cc.NumObjects == 0
            % No object found, return original
            imgCropped = img;
            return;
        end

        numPixels = cellfun(@numel, cc.PixelIdxList);
        [~, idx] = max(numPixels);
        objectMask = false(size(objectMask));
        objectMask(cc.PixelIdxList{idx}) = true;

        % Find bounding box
        props = regionprops(objectMask, 'BoundingBox');
        if isempty(props)
            imgCropped = img;
            return;
        end

        bbox = props(1).BoundingBox;

        % Add some padding (10% on each side)
        [h, w, ~] = size(img);
        padding = 0.1;
        x1 = max(1, floor(bbox(1) - bbox(3) * padding));
        y1 = max(1, floor(bbox(2) - bbox(4) * padding));
        x2 = min(w, ceil(bbox(1) + bbox(3) * (1 + padding)));
        y2 = min(h, ceil(bbox(2) + bbox(4) * (1 + padding)));

        % Crop
        imgCropped = img(y1:y2, x1:x2, :);

        % If cropped region is too small, return original
        if size(imgCropped, 1) < 50 || size(imgCropped, 2) < 50
            imgCropped = img;
        end

    catch ME
        % If any error, return original image
        warning('Auto-crop failed: %s. Using original image.', ME.message);
        imgCropped = img;
    end
end
