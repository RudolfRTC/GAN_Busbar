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
params.dataFolder = './data/images';
params.trainSize = [64 128];         % Training size [height width]
                                     % Options: [64 128] (2:1, memory efficient)
                                     %          [128 256] (2:1, higher quality)
                                     %          [64 64], [128 128], [256 256] (square)
params.autoCrop = true;              % Auto-crop white background
params.cropThreshold = 0.85;         % Threshold for white detection (0-1)

% Network parameters
params.latentDim = 100;              % Latent vector dimension
params.numChannels = 3;              % Will be auto-detected (1=grayscale, 3=RGB)

% Training parameters
params.numEpochs = 300;              % Number of epochs
params.miniBatchSize = 8;            % Batch size (8-16 for laptop GPU)
params.learnRate = 0.0002;           % Learning rate
params.beta1 = 0.5;                  % Adam beta1
params.executionEnvironment = 'auto'; % 'auto', 'gpu', or 'cpu'

% Stabilization parameters
params.labelSmoothing = 0.9;         % Real labels = 0.9 instead of 1.0
params.instanceNoise = 0.05;         % Initial instance noise std
params.noiseDecay = 0.995;           % Instance noise decay per epoch

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

%% Sanity Check: Test Generator
fprintf('Sanity check: Testing generator...\n');
try
    % Create test input with correct shape for featureInputLayer: [latentDim x N]
    Z_test = dlarray(randn(params.latentDim, 2, 'single'), 'CB');
    if strcmp(params.executionEnvironment, 'auto') || strcmp(params.executionEnvironment, 'gpu')
        if canUseGPU
            Z_test = gpuArray(Z_test);
        end
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

%% Build Discriminator
fprintf('Building Discriminator...\n');
netD = buildDiscriminator(params);
fprintf('Discriminator parameters: %d\n', sum(cellfun(@numel, netD.Learnables.Value)));
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
if strcmp(params.executionEnvironment, 'auto') || strcmp(params.executionEnvironment, 'gpu')
    if canUseGPU
        fprintf('GPU detected: %s\n', gpuDevice().Name);
    else
        fprintf('No GPU detected, using CPU\n');
    end
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
        Z = randn(params.latentDim, batchSize, 'like', XReal);
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
            if strcmp(params.executionEnvironment, 'auto') || strcmp(params.executionEnvironment, 'gpu')
                if canUseGPU
                    ZPreviewDL = gpuArray(ZPreviewDL);
                end
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

function [gradD, lossD] = modelGradientsD(netD, netG, XReal, Z, params)
    % Discriminator gradients

    % Generate fake images
    XGenerated = forward(netG, Z);

    % Discriminator outputs
    YReal = forward(netD, XReal);
    YGenerated = forward(netD, XGenerated);

    % Label smoothing: real = 0.9, fake = 0.0
    lossReal = -mean(log(YReal + 1e-8)) * params.labelSmoothing - mean(log(1 - YReal + 1e-8)) * (1 - params.labelSmoothing);
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
