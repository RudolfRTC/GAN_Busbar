%% CHECK SYNTHETIC IMAGES - Are they truly grayscale or fake RGB?
% Check the 2000 synthetic images you already generated
%
% cd 'G:\Fax\Prijava Teme\Clanek za busbare\Code'
% check_synthetic_images

function check_synthetic_images()
    fprintf('========================================\n');
    fprintf('Checking Synthetic Images\n');
    fprintf('========================================\n\n');

    % Find synthetic images
    syntheticFolder = './outputs/synthetic';
    imageFiles = dir(fullfile(syntheticFolder, '*.png'));

    if isempty(imageFiles)
        imageFiles = dir(fullfile(syntheticFolder, '*.jpg'));
    end

    if isempty(imageFiles)
        error('No synthetic images found in %s', syntheticFolder);
    end

    fprintf('Found %d synthetic images\n\n', numel(imageFiles));

    % Check first 10 images
    numTest = min(10, numel(imageFiles));
    numTrueRGB = 0;
    numFakeRGB = 0;

    for i = 1:numTest
        imgPath = fullfile(imageFiles(i).folder, imageFiles(i).name);
        fprintf('[%d/%d] %s\n', i, numTest, imageFiles(i).name);

        img = imread(imgPath);

        % Check if RGB or grayscale
        isRGB = checkImageRGB(img);

        if isRGB
            numTrueRGB = numTrueRGB + 1;
        else
            numFakeRGB = numFakeRGB + 1;
        end
    end

    % Summary
    fprintf('\n========================================\n');
    fprintf('SUMMARY\n');
    fprintf('========================================\n');
    fprintf('True RGB images:  %d / %d\n', numTrueRGB, numTest);
    fprintf('Fake RGB images:  %d / %d\n', numFakeRGB, numTest);
    fprintf('\n');

    if numFakeRGB == numTest
        fprintf('🔴 CONFIRMED: All synthetic images are FAKE RGB!\n');
        fprintf('\n');
        fprintf('Images have 3 channels but R=G=B (grayscale).\n');
        fprintf('\n');
        fprintf('ROOT CAUSE: Generator outputs identical values for all 3 channels.\n');
        fprintf('\n');
        fprintf('SOLUTION NEEDED:\n');
        fprintf('I need to fix buildGenerator.m so it outputs 3 INDEPENDENT channels.\n');
        fprintf('\n');
        fprintf('The problem is likely:\n');
        fprintf('1. Final transposedConv2dLayer is not properly configured for 3 channels\n');
        fprintf('2. OR: All 3 output channels are constrained to be equal somehow\n');
        fprintf('3. OR: Network architecture forces grayscale output\n');
        fprintf('\n');
        fprintf('I will investigate and fix the generator architecture!\n');
    elseif numTrueRGB > 0
        fprintf('✅ Good: Found %d RGB images!\n', numTrueRGB);
        fprintf('\n');
        fprintf('Some images have color. If they look gray to you:\n');
        fprintf('1. Color might be very subtle\n');
        fprintf('2. Training needs more epochs to learn stronger colors\n');
        fprintf('3. Check if monitor/viewer is showing colors correctly\n');
    end
end

function isRGB = checkImageRGB(img)
    % Check if image is true RGB or fake RGB

    [h, w, c] = size(img);
    fprintf('  Size: %dx%d, Channels: %d\n', w, h, c);

    if c ~= 3
        fprintf('  ❌ Not RGB (%d channels)\n\n', c);
        isRGB = false;
        return;
    end

    % Extract channels
    R = double(img(:,:,1));
    G = double(img(:,:,2));
    B = double(img(:,:,3));

    % Sample pixels
    numPixels = numel(R);
    sampleSize = min(1000, numPixels);
    idx = randperm(numPixels, sampleSize);

    R_sample = R(idx);
    G_sample = G(idx);
    B_sample = B(idx);

    % Statistics
    R_mean = mean(R_sample);
    G_mean = mean(G_sample);
    B_mean = mean(B_sample);

    % MAD
    mad_RG = mean(abs(R_sample - G_sample));
    mad_GB = mean(abs(G_sample - B_sample));
    mad_RB = mean(abs(R_sample - B_sample));

    fprintf('  Channel means: R=%.1f, G=%.1f, B=%.1f\n', R_mean, G_mean, B_mean);
    fprintf('  Channel diffs (MAD): RG=%.2f, GB=%.2f, RB=%.2f\n', mad_RG, mad_GB, mad_RB);

    % Threshold for uint8 [0-255]
    THRESHOLD = 2.0;

    if mad_RG < THRESHOLD && mad_GB < THRESHOLD && mad_RB < THRESHOLD
        fprintf('  ❌ FAKE RGB (R≈G≈B) - Actually grayscale!\n\n');
        isRGB = false;
    else
        fprintf('  ✅ TRUE RGB - Has color!\n\n');
        isRGB = true;
    end
end
