%% TRAIN_GAN - Train DCGAN for Industrial Part Synthetic Image Generation
%
% HOW TO RUN:
% 1. Place your ~70 images in: ./data/images/
%    (Supported formats: .jpg, .png, .bmp)
% 2. Run this script: train_gan
% 3. Results:
%    - Preview images: ./outputs/preview/
%    - Trained models: ./outputs/models/
%    - Synthetic images: ./outputs/synthetic/
%
% PARAMETERS (edit below as needed):
% - trainSize: training resolution [height width] (default: [64 128] for 2:1 aspect ratio)
%   Supported: [64 128], [128 256], [64 64], [128 128], [256 256]
% - latentDim: latent vector size (default: 100)
% - numEpochs: training epochs (default: 300)
% - miniBatchSize: batch size (default: 8-16, auto-adjusted if needed)
% - learnRate: Adam learning rate (default: 0.0002)
% - numSynthetic: number of synthetic images to generate (default: 2000)

%% Clear workspace
clear; close all; clc;

fprintf('==============================================\n');
fprintf('  DCGAN Training for Industrial Parts\n');
fprintf('==============================================\n\n');

%% ===== PARAMETERS (MODIFY HERE) =====
params = struct();

% Data parameters
% UPDATED: Using cropped images from new path
params.dataFolder = 'G:\Fax\Prijava Teme\Clanek za busbare\Code\data\images';
params.trainSize = [64 128];         % Training size [height width]
                                     % NOTE: Current generator architecture supports [64 128]
                                     % To use [128 128], need to update buildGenerator.m
params.autoCrop = false;             % Disabled - images are already cropped
params.cropThreshold = 0.85;         % Threshold for white detection (0-1)

% Network parameters
params.latentDim = 100;              % Latent vector dimension
params.numChannels = 3;              % Will be auto-detected (1=grayscale, 3=RGB)

% Training parameters
params.numEpochs = 500;              % Number of epochs (increased for better convergence)
params.miniBatchSize = 32;           % Batch size INCREASED from 8 to 32
                                     % Larger batches prevent mode collapse
                                     % (64 may cause OOM on RTX A2000 4GB, use 32 for safety)
params.learnRate = 0.0002;           % Learning rate (standard DCGAN)
params.beta1 = 0.5;                  % Adam beta1
params.executionEnvironment = 'auto'; % 'auto', 'gpu', or 'cpu'

% Stabilization parameters - ENHANCED to prevent mode collapse
params.labelSmoothing = 0.9;         % Real labels = 0.9 instead of 1.0
params.instanceNoise = 0.1;          % Initial instance noise std (INCREASED from 0.05)
params.noiseDecay = 0.998;           % Instance noise decay per epoch (SLOWER decay)

% Output parameters
params.outputFolder = './outputs';
params.previewEvery = 50;            % Save preview every N iterations
params.numPreviewImages = 16;        % Number of images in preview grid
params.numSynthetic = 2000;          % Number of synthetic images to generate

fprintf('Parameters:\n');
fprintf('  Training Size: %dx%d (HxW)\n', params.trainSize(1), params.trainSize(2));
fprintf('  Aspect Ratio: %.2f:1\n', params.trainSize(2) / params.trainSize(1));
fprintf('  Latent Dim: %d\n', params.latentDim);
fprintf('  Epochs: %d\n', params.numEpochs);
fprintf('  Batch Size: %d\n', params.miniBatchSize);
fprintf('  Learn Rate: %.4f\n', params.learnRate);
fprintf('  Auto-crop: %s\n', mat2str(params.autoCrop));
fprintf('\n');

%% Create output directories
if ~exist(params.outputFolder, 'dir')
    mkdir(params.outputFolder);
end
if ~exist(fullfile(params.outputFolder, 'preview'), 'dir')
    mkdir(fullfile(params.outputFolder, 'preview'));
end
if ~exist(fullfile(params.outputFolder, 'models'), 'dir')
    mkdir(fullfile(params.outputFolder, 'models'));
end
if ~exist(fullfile(params.outputFolder, 'synthetic'), 'dir')
    mkdir(fullfile(params.outputFolder, 'synthetic'));
end

%% Load and preprocess data
fprintf('Loading and preprocessing images...\n');
[mbqTrain, params] = preprocessAndLoadDatastore(params);
fprintf('Dataset ready: %d images\n', params.numImages);
fprintf('Using %d channels (1=grayscale, 3=RGB)\n', params.numChannels);
fprintf('\n');

%% Build Generator
fprintf('Building Generator...\n');
netG = buildGenerator(params);
fprintf('Generator parameters: %d\n', sum(cellfun(@numel, netG.Learnables.Value)));
fprintf('\n');

%% Build Discriminator
fprintf('Building Discriminator...\n');
netD = buildDiscriminator(params);
fprintf('Discriminator parameters: %d\n', sum(cellfun(@numel, netD.Learnables.Value)));
fprintf('\n');

%% GPU Diagnostics and Setup
fprintf('==============================================\n');
fprintf('  GPU Diagnostics\n');
fprintf('==============================================\n');
[gpuAvailable, gpuInfo] = checkAndReportGPU();
fprintf('\n');

%% Sanity Check: Test Generator
fprintf('Sanity check: Testing generator...\n');
try
    % Create test input with correct shape for featureInputLayer: [latentDim x N]
    if gpuAvailable && (strcmp(params.executionEnvironment, 'auto') || strcmp(params.executionEnvironment, 'gpu'))
        Z_test = dlarray(randn(params.latentDim, 2, 'single', 'gpuArray'), 'CB');
    else
        Z_test = dlarray(randn(params.latentDim, 2, 'single'), 'CB');
    end

    % Forward pass
    X_test = predict(netG, Z_test);
    X_test = extractdata(X_test);

    % Check output shape
    [outH, outW, outC, outN] = size(X_test);
    fprintf('  Generator output shape: [%d x %d x %d x %d]\n', outH, outW, outC, outN);
    fprintf('  Expected shape:         [%d x %d x %d x %d]\n', ...
        params.trainSize(1), params.trainSize(2), params.numChannels, 2);

    % Assert correct shape
    assert(outH == params.trainSize(1), ...
        'ERROR: Generator output height %d != expected %d', outH, params.trainSize(1));
    assert(outW == params.trainSize(2), ...
        'ERROR: Generator output width %d != expected %d', outW, params.trainSize(2));
    assert(outC == params.numChannels, ...
        'ERROR: Generator output channels %d != expected %d', outC, params.numChannels);

    fprintf('  ✓ Generator test PASSED!\n');
catch ME
    fprintf('  ✗ Generator test FAILED!\n');
    fprintf('  Error: %s\n', ME.message);
    error('Generator sanity check failed. Please check buildGenerator.m architecture.');
end
fprintf('\n');

%% Initialize training
avgG = [];
avgGS = [];
avgD = [];
avgDS = [];

% Fixed latent vector for preview (format: [latentDim x N] for featureInputLayer)
ZPreview = randn(params.latentDim, params.numPreviewImages, 'single');

iteration = 0;
startTime = tic;

fprintf('==============================================\n');
fprintf('  Starting Training\n');
fprintf('==============================================\n');
fprintf('Execution Environment: %s\n', params.executionEnvironment);
if gpuAvailable && (strcmp(params.executionEnvironment, 'auto') || strcmp(params.executionEnvironment, 'gpu'))
    fprintf('Training on GPU: %s\n', gpuInfo.Name);
    fprintf('Available VRAM: %.2f GB\n', gpuInfo.AvailableMemory / 1e9);
else
    fprintf('Training on CPU\n');
end
fprintf('\n');

%% Training Loop
for epoch = 1:params.numEpochs

    % Reset minibatchqueue at the start of each epoch
    reset(mbqTrain);

    % Decay instance noise
    currentInstanceNoise = params.instanceNoise * (params.noiseDecay ^ epoch);

    while hasdata(mbqTrain)
        iteration = iteration + 1;

        % Get next mini-batch from minibatchqueue
        % minibatchqueue already returns dlarray in SSCB format and on GPU if configured
        XReal = next(mbqTrain);

        % Ensure XReal is a dlarray (in case of empty batch or other edge cases)
        if ~isa(XReal, 'dlarray')
            XReal = dlarray(XReal, 'SSCB');
        end

        % Check for empty or undersized batch
        batchSize = size(XReal, 4);
        if batchSize == 0
            warning('Empty batch encountered at iteration %d, skipping...', iteration);
            continue;
        end

        % Generate random latent vectors (format: [latentDim x N] for featureInputLayer)
        % CRITICAL: Must be dlarray with format 'CB' (Channel x Batch)
        % If GPU is available, create Z directly on GPU
        if gpuAvailable && (strcmp(params.executionEnvironment, 'auto') || strcmp(params.executionEnvironment, 'gpu'))
            Z = randn(params.latentDim, batchSize, 'single', 'gpuArray');
        else
            Z = randn(params.latentDim, batchSize, 'single');
        end
        Z = dlarray(Z, 'CB');  % Explicitly label: C=latentDim (100), B=batchSize (8)

        % Sanity check: Verify Z dimensions are correct
        zSize = size(Z);
        assert(zSize(1) == params.latentDim, ...
            'ERROR: Z channel dimension is %d, expected %d. Z must be [latentDim x batchSize].', ...
            zSize(1), params.latentDim);
        assert(zSize(2) == batchSize, ...
            'ERROR: Z batch dimension is %d, expected %d. Z must be [latentDim x batchSize].', ...
            zSize(2), batchSize);

        % Add instance noise to real images (stabilization)
        if currentInstanceNoise > 0
            XReal = XReal + currentInstanceNoise * randn(size(XReal), 'like', XReal);
            XReal = max(min(XReal, 1), -1); % Clamp to [-1, 1]
        end

        % ===== Train Discriminator =====
        [gradD, lossD] = dlfeval(@modelGradientsD, netD, netG, XReal, Z, params);
        [netD, avgD, avgDS] = adamupdate(netD, gradD, avgD, avgDS, iteration, ...
            params.learnRate, params.beta1);

        % ===== Train Generator =====
        [gradG, lossG] = dlfeval(@modelGradientsG, netD, netG, Z);
        [netG, avgG, avgGS] = adamupdate(netG, gradG, avgG, avgGS, iteration, ...
            params.learnRate, params.beta1);

        % ===== Display Progress =====
        if mod(iteration, 10) == 0 || iteration == 1
            elapsedTime = toc(startTime);
            fprintf('[Epoch %d/%d, Iter %d] Time: %.1fs | D Loss: %.4f | G Loss: %.4f | Noise: %.4f\n', ...
                epoch, params.numEpochs, iteration, elapsedTime, ...
                extractdata(lossD), extractdata(lossG), currentInstanceNoise);
        end

        % ===== Save Preview =====
        if mod(iteration, params.previewEvery) == 0
            % Generate preview images (format: 'CB' for featureInputLayer)
            ZPreviewDL = dlarray(ZPreview, 'CB');
            if gpuAvailable && (strcmp(params.executionEnvironment, 'auto') || strcmp(params.executionEnvironment, 'gpu'))
                ZPreviewDL = gpuArray(ZPreviewDL);
            end

            XGenerated = predict(netG, ZPreviewDL);
            XGenerated = extractdata(gather(XGenerated));

            % Save preview grid
            previewFilename = fullfile(params.outputFolder, 'preview', ...
                sprintf('preview_epoch%03d_iter%05d.png', epoch, iteration));
            saveImageGrid(XGenerated, previewFilename, params);
            fprintf('  --> Saved preview: %s\n', previewFilename);
        end
    end

    % End of epoch summary
    fprintf('--- Epoch %d/%d completed ---\n\n', epoch, params.numEpochs);
end

totalTime = toc(startTime);
fprintf('==============================================\n');
fprintf('  Training Completed!\n');
fprintf('  Total Time: %.1f minutes\n', totalTime/60);
fprintf('==============================================\n\n');

%% Save trained models
fprintf('Saving trained models...\n');
modelPath = fullfile(params.outputFolder, 'models');
save(fullfile(modelPath, 'generator.mat'), 'netG', 'params');
save(fullfile(modelPath, 'discriminator.mat'), 'netD');
fprintf('Models saved to: %s\n', modelPath);
fprintf('\n');

%% Generate synthetic images
fprintf('==============================================\n');
fprintf('  Generating %d Synthetic Images\n', params.numSynthetic);
fprintf('==============================================\n');
generateSynthetic(netG, params);

fprintf('\n==============================================\n');
fprintf('  ALL DONE!\n');
fprintf('==============================================\n');
fprintf('Check outputs:\n');
fprintf('  - Preview images: %s\n', fullfile(params.outputFolder, 'preview'));
fprintf('  - Trained models: %s\n', fullfile(params.outputFolder, 'models'));
fprintf('  - Synthetic images: %s\n', fullfile(params.outputFolder, 'synthetic'));
fprintf('\n');

%% ========== HELPER FUNCTIONS ==========

function [gpuAvailable, gpuInfo] = checkAndReportGPU()
    % Check GPU availability and print diagnostics
    %
    % Returns:
    %   gpuAvailable - boolean, true if GPU is available
    %   gpuInfo - struct with GPU information (Name, ComputeCapability, AvailableMemory)

    gpuAvailable = false;
    gpuInfo = struct('Name', 'N/A', 'ComputeCapability', 'N/A', 'AvailableMemory', 0);

    % Check if Parallel Computing Toolbox is installed
    fprintf('Checking Parallel Computing Toolbox...\n');
    v = ver('parallel');
    if isempty(v)
        fprintf('  ✗ Parallel Computing Toolbox NOT installed\n');
        fprintf('  → MATLAB cannot use GPU without this toolbox\n');
        fprintf('\n');
        printGPUFixInstructions();
        return;
    else
        fprintf('  ✓ Parallel Computing Toolbox installed (v%s)\n', v.Version);
    end

    % Check GPU device count
    fprintf('\nChecking GPU devices...\n');
    try
        numGPUs = gpuDeviceCount;
        fprintf('  GPU Device Count: %d\n', numGPUs);

        if numGPUs == 0
            fprintf('  ✗ No GPU devices detected by MATLAB\n');
            fprintf('\n');
            printGPUFixInstructions();
            return;
        end

        % Get GPU device information
        fprintf('\nGPU Device Information:\n');
        try
            g = gpuDevice();
            gpuAvailable = true;
            gpuInfo.Name = g.Name;
            gpuInfo.ComputeCapability = g.ComputeCapability;
            gpuInfo.AvailableMemory = g.AvailableMemory;
            gpuInfo.TotalMemory = g.TotalMemory;

            fprintf('  ✓ GPU Available: %s\n', g.Name);
            fprintf('    - Compute Capability: %s\n', g.ComputeCapability);
            fprintf('    - Total Memory: %.2f GB\n', g.TotalMemory / 1e9);
            fprintf('    - Available Memory: %.2f GB\n', g.AvailableMemory / 1e9);
            fprintf('    - Device Index: %d\n', g.Index);
            fprintf('    - CUDA Version: %s\n', g.ToolkitVersion);

            % Check if memory is sufficient
            minRequiredMemory = 2e9; % 2 GB minimum
            if g.AvailableMemory < minRequiredMemory
                fprintf('\n  ⚠ WARNING: Low GPU memory (%.2f GB available)\n', g.AvailableMemory / 1e9);
                fprintf('    Recommended: At least 2 GB for batch size 8\n');
                fprintf('    Consider reducing batch size if OOM errors occur\n');
            end

        catch ME
            fprintf('  ✗ Error accessing GPU device: %s\n', ME.message);
            gpuAvailable = false;
        end

    catch ME
        fprintf('  ✗ Error checking GPU devices: %s\n', ME.message);
        fprintf('\n');
        printGPUFixInstructions();
        return;
    end
end

function printGPUFixInstructions()
    % Print instructions on how to enable GPU support
    fprintf('==============================================\n');
    fprintf('  How to Enable GPU Support in MATLAB\n');
    fprintf('==============================================\n');
    fprintf('1. Install NVIDIA GPU drivers:\n');
    fprintf('   - Download from: https://www.nvidia.com/drivers\n');
    fprintf('   - For RTX A2000: Use "Quadro/RTX Desktop" driver series\n');
    fprintf('   - Recommended: Latest Game Ready or Studio driver\n');
    fprintf('\n');
    fprintf('2. Install CUDA Toolkit (if not already installed):\n');
    fprintf('   - MATLAB R2021a+: CUDA 11.x recommended\n');
    fprintf('   - MATLAB R2023a+: CUDA 11.8 or 12.x supported\n');
    fprintf('   - Download from: https://developer.nvidia.com/cuda-downloads\n');
    fprintf('\n');
    fprintf('3. Verify MATLAB can see GPU:\n');
    fprintf('   >> gpuDeviceCount  %% Should return > 0\n');
    fprintf('   >> gpuDevice       %% Should show GPU info\n');
    fprintf('\n');
    fprintf('4. If GPU is still not detected:\n');
    fprintf('   - Restart MATLAB after driver installation\n');
    fprintf('   - Check Windows Device Manager → Display Adapters\n');
    fprintf('   - Run: nvidia-smi (in Command Prompt) to verify driver\n');
    fprintf('   - Check MATLAB GPU support: https://www.mathworks.com/help/parallel-computing/gpu-support-by-release.html\n');
    fprintf('\n');
    fprintf('5. Install Parallel Computing Toolbox (if missing):\n');
    fprintf('   - Open MATLAB Add-Ons → Get Add-Ons\n');
    fprintf('   - Search for "Parallel Computing Toolbox"\n');
    fprintf('   - Install (requires active MATLAB license)\n');
    fprintf('==============================================\n');
end

function [gradD, lossD] = modelGradientsD(netD, netG, XReal, Z, params)
    % Discriminator gradients

    % Generate fake images
    XGenerated = forward(netG, Z);

    % Discriminator outputs
    YReal = forward(netD, XReal);
    YGenerated = forward(netD, XGenerated);

    % Label smoothing: real target = 0.9 (one-sided), fake target = 0.0
    % Binary cross entropy with soft targets:
    % BCE(y, t) = -[t*log(y) + (1-t)*log(1-y)]
    % For real images with target=labelSmoothing (0.9):
    lossReal = -mean(params.labelSmoothing * log(YReal + 1e-8) + ...
                     (1 - params.labelSmoothing) * log(1 - YReal + 1e-8));
    % For fake images with target=0.0:
    lossFake = -mean(log(1 - YGenerated + 1e-8));

    % Total discriminator loss
    lossD = lossReal + lossFake;

    % Compute gradients
    gradD = dlgradient(lossD, netD.Learnables);
end

function [gradG, lossG] = modelGradientsG(netD, netG, Z)
    % Generator gradients

    % Generate fake images
    XGenerated = forward(netG, Z);

    % Discriminator output for fake images
    YGenerated = forward(netD, XGenerated);

    % Generator loss: fool discriminator
    lossG = -mean(log(YGenerated + 1e-8));

    % Compute gradients
    gradG = dlgradient(lossG, netG.Learnables);
end
