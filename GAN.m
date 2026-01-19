%% GAN - Main entry point for DCGAN training
%
% This is the main script to train a DCGAN for generating synthetic images
% of industrial parts (busbars). This wrapper provides a simplified interface
% to the full training pipeline.
%
% QUICK START:
% 1. Add your training images (minimum 30, recommended 50-100) to:
%    ./data/images/
%    Supported formats: .jpg, .jpeg, .png, .bmp
%
% 2. Run this script:
%    >> GAN
%
% 3. Wait for training to complete (2-4 hours with GPU)
%
% 4. Find results in:
%    ./outputs/preview/     - Training preview images
%    ./outputs/models/      - Trained generator and discriminator
%    ./outputs/synthetic/   - 2000 generated synthetic images
%
% CUSTOMIZATION:
% To customize training parameters, edit train_gan.m instead of this file.
%
% For advanced usage and troubleshooting, see README.md

%% Clear workspace
clear; close all; clc;

fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║                                                            ║\n');
fprintf('║        DCGAN for Industrial Part Image Generation         ║\n');
fprintf('║                                                            ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n');
fprintf('\n');

%% Pre-flight checks
fprintf('Running pre-flight checks...\n\n');

% 1. Check MATLAB version
fprintf('[1/4] Checking MATLAB version...\n');
matlabVersion = version('-release');
matlabYear = str2double(matlabVersion(1:4));

if matlabYear < 2020
    fprintf('  ✗ ERROR: MATLAB R%s detected\n', matlabVersion);
    fprintf('  → This code requires MATLAB R2020b or newer\n');
    fprintf('  → Please upgrade your MATLAB installation\n');
    error('MATLAB version too old. Requires R2020b or newer.');
else
    fprintf('  ✓ MATLAB R%s detected (OK)\n', matlabVersion);
end

% 2. Check Deep Learning Toolbox
fprintf('\n[2/4] Checking Deep Learning Toolbox...\n');
v = ver('deeplearning');
if isempty(v)
    fprintf('  ✗ ERROR: Deep Learning Toolbox NOT installed\n');
    fprintf('  → This toolbox is required for GAN training\n');
    fprintf('  → Install it from MATLAB Add-Ons Manager\n');
    error('Deep Learning Toolbox not found. Please install it.');
else
    fprintf('  ✓ Deep Learning Toolbox installed (v%s)\n', v.Version);
end

% 3. Check for training images
fprintf('\n[3/4] Checking for training images...\n');
dataFolder = './data/images';

if ~exist(dataFolder, 'dir')
    fprintf('  ✗ ERROR: Data folder not found: %s\n', dataFolder);
    fprintf('  → Creating folder...\n');
    mkdir(dataFolder);
    fprintf('  ✓ Folder created\n');
    fprintf('\n');
    fprintf('  ⚠ WARNING: No training images found!\n');
    fprintf('  → Please add images to: %s\n', dataFolder);
    fprintf('  → Supported formats: .jpg, .jpeg, .png, .bmp\n');
    fprintf('  → Minimum: 30 images\n');
    fprintf('  → Recommended: 50-100 images\n');
    error('No training images. Add images to ./data/images/ and run again.');
end

imageFiles = [
    dir(fullfile(dataFolder, '*.jpg'));
    dir(fullfile(dataFolder, '*.jpeg'));
    dir(fullfile(dataFolder, '*.png'));
    dir(fullfile(dataFolder, '*.bmp'))
];

numImages = numel(imageFiles);

if numImages == 0
    fprintf('  ✗ ERROR: No images found in: %s\n', dataFolder);
    fprintf('  → Please add images (supported: .jpg, .jpeg, .png, .bmp)\n');
    fprintf('  → Minimum: 30 images\n');
    fprintf('  → Recommended: 50-100 images\n');
    error('No training images found. Add images and run again.');
elseif numImages < 30
    fprintf('  ⚠ WARNING: Only %d images found\n', numImages);
    fprintf('  → This is very few for GAN training\n');
    fprintf('  → Recommended: at least 50-100 images\n');
    fprintf('  → Training may produce poor results\n');
    fprintf('\n');

    % Ask user if they want to continue
    response = input('  Continue anyway? (y/n): ', 's');
    if ~strcmpi(response, 'y')
        fprintf('  Training cancelled. Please add more images.\n');
        return;
    end
    fprintf('  ✓ Continuing with %d images (results may vary)\n', numImages);
elseif numImages < 50
    fprintf('  ⚠ Found %d images (minimum met, but more recommended)\n', numImages);
    fprintf('  → For best results, use 50-100+ images\n');
else
    fprintf('  ✓ Found %d training images (excellent!)\n', numImages);
end

% 4. Check GPU availability
fprintf('\n[4/4] Checking GPU availability...\n');
try
    v_parallel = ver('parallel');
    if ~isempty(v_parallel)
        numGPUs = gpuDeviceCount;
        if numGPUs > 0
            g = gpuDevice();
            fprintf('  ✓ GPU available: %s\n', g.Name);
            fprintf('    - Memory: %.2f GB\n', g.TotalMemory / 1e9);
            fprintf('    - Training will use GPU (fast)\n');
        else
            fprintf('  ⚠ No GPU detected\n');
            fprintf('    - Training will use CPU (slower)\n');
            fprintf('    - Expect 10-20 hours for 300 epochs\n');
        end
    else
        fprintf('  ⚠ Parallel Computing Toolbox not installed\n');
        fprintf('    - Cannot use GPU without this toolbox\n');
        fprintf('    - Training will use CPU (slower)\n');
    end
catch
    fprintf('  ⚠ GPU check failed\n');
    fprintf('    - Training will use CPU\n');
end

%% Summary
fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║  Pre-flight checks completed!                             ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n');
fprintf('\n');

fprintf('Training configuration:\n');
fprintf('  - Number of images: %d\n', numImages);
fprintf('  - Image size: 64x128 pixels (2:1 aspect ratio)\n');
fprintf('  - Training epochs: 300\n');
fprintf('  - Batch size: 8 (auto-adjusted if needed)\n');
fprintf('  - Synthetic images to generate: 2000\n');
fprintf('\n');

fprintf('Estimated time:\n');
try
    if gpuDeviceCount > 0
        fprintf('  - With GPU: ~2-4 hours\n');
    else
        fprintf('  - With CPU: ~10-20 hours\n');
    end
catch
    fprintf('  - With CPU: ~10-20 hours\n');
end
fprintf('\n');

fprintf('Output folders:\n');
fprintf('  - ./outputs/preview/    - Training preview images\n');
fprintf('  - ./outputs/models/     - Trained models\n');
fprintf('  - ./outputs/synthetic/  - Generated synthetic images\n');
fprintf('\n');

%% Confirm start
fprintf('════════════════════════════════════════════════════════════\n');
response = input('Start training? (y/n): ', 's');
if ~strcmpi(response, 'y')
    fprintf('\nTraining cancelled.\n');
    fprintf('To start training later, run: GAN\n\n');
    return;
end

fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║  Starting GAN Training...                                 ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n');
fprintf('\n');
fprintf('Note: Training will take several hours. You can monitor progress\n');
fprintf('      by checking the preview images in ./outputs/preview/\n');
fprintf('\n');

%% Launch training
try
    train_gan;

    fprintf('\n');
    fprintf('╔════════════════════════════════════════════════════════════╗\n');
    fprintf('║  Training completed successfully!                         ║\n');
    fprintf('╚════════════════════════════════════════════════════════════╝\n');
    fprintf('\n');
    fprintf('Your results are ready:\n');
    fprintf('  ✓ Preview images: ./outputs/preview/\n');
    fprintf('  ✓ Trained models: ./outputs/models/\n');
    fprintf('  ✓ Synthetic images: ./outputs/synthetic/\n');
    fprintf('\n');
    fprintf('Next steps:\n');
    fprintf('  1. Review the generated images in ./outputs/synthetic/\n');
    fprintf('  2. If quality is poor, see README.md troubleshooting section\n');
    fprintf('  3. To generate more images, use: generateSynthetic(netG, params)\n');
    fprintf('\n');

catch ME
    fprintf('\n');
    fprintf('╔════════════════════════════════════════════════════════════╗\n');
    fprintf('║  Training failed with error                               ║\n');
    fprintf('╚════════════════════════════════════════════════════════════╝\n');
    fprintf('\n');
    fprintf('Error message:\n');
    fprintf('  %s\n', ME.message);
    fprintf('\n');
    fprintf('Stack trace:\n');
    for i = 1:length(ME.stack)
        fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
    fprintf('\n');
    fprintf('Troubleshooting:\n');
    fprintf('  1. Check README.md for common issues\n');
    fprintf('  2. Verify your images are valid (not corrupted)\n');
    fprintf('  3. Try reducing batch size in train_gan.m if out of memory\n');
    fprintf('\n');
    rethrow(ME);
end
