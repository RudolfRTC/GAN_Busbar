%% TEST_GAN_FULL - Comprehensive test of entire GAN pipeline
%
% This script performs a full integration test of the GAN system:
% 1. Creates synthetic test images
% 2. Tests all network components
% 3. Runs a mini training loop (2 iterations)
% 4. Tests image generation
% 5. Validates all outputs
%
% Usage: test_gan_full
%
% This test takes ~2-5 minutes and doesn't require real training data.

clear; close all; clc;

fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║                                                            ║\n');
fprintf('║         GAN Pipeline - Full Integration Test              ║\n');
fprintf('║                                                            ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n');
fprintf('\n');

testsPassed = 0;
testsFailed = 0;
totalTests = 12;

%% ===== TEST 1: Create Test Images =====
fprintf('[Test 1/%d] Creating synthetic test images...\n', totalTests);
try
    % Create test data folder
    testDataFolder = './data/images_test';
    if ~exist(testDataFolder, 'dir')
        mkdir(testDataFolder);
    end

    % Generate 10 synthetic RGB images with random colored shapes
    numTestImages = 10;
    imgSize = [256, 512, 3]; % HxWxC

    for i = 1:numTestImages
        % Create white background
        img = ones(imgSize, 'uint8') * 255;

        % Add random colored rectangle (simulating busbar)
        rectH = randi([80, 150]);
        rectW = randi([200, 400]);
        centerY = randi([80, 170]);
        centerX = randi([150, 360]);

        y1 = max(1, centerY - rectH/2);
        y2 = min(imgSize(1), centerY + rectH/2);
        x1 = max(1, centerX - rectW/2);
        x2 = min(imgSize(2), centerX + rectW/2);

        % Random color (copper-like tones)
        r = randi([150, 200]);
        g = randi([80, 130]);
        b = randi([40, 80]);

        img(round(y1):round(y2), round(x1):round(x2), 1) = r;
        img(round(y1):round(y2), round(x1):round(x2), 2) = g;
        img(round(y1):round(y2), round(x1):round(x2), 3) = b;

        % Add some noise for realism
        noise = uint8(randn(imgSize) * 5);
        img = uint8(double(img) + double(noise));

        % Save image
        filename = fullfile(testDataFolder, sprintf('test_busbar_%02d.png', i));
        imwrite(img, filename);
    end

    fprintf('  ✓ Created %d test images in: %s\n', numTestImages, testDataFolder);
    testsPassed = testsPassed + 1;
catch ME
    fprintf('  ✗ FAILED: %s\n', ME.message);
    testsFailed = testsFailed + 1;
end
fprintf('\n');

%% ===== TEST 2: Test Parameter Setup =====
fprintf('[Test 2/%d] Testing parameter configuration...\n', totalTests);
try
    params = struct();
    params.dataFolder = testDataFolder;
    params.trainSize = [64 128];
    params.autoCrop = true;
    params.cropThreshold = 0.85;
    params.latentDim = 100;
    params.numChannels = 3;
    params.numEpochs = 2; % Only 2 for testing
    params.miniBatchSize = 4;
    params.learnRate = 0.0002;
    params.beta1 = 0.5;
    params.executionEnvironment = 'auto';
    params.labelSmoothing = 0.9;
    params.instanceNoise = 0.05;
    params.noiseDecay = 0.995;
    params.outputFolder = './outputs_test';
    params.previewEvery = 10;
    params.numPreviewImages = 4;
    params.numSynthetic = 5; % Only 5 for testing

    fprintf('  ✓ Parameters configured successfully\n');
    testsPassed = testsPassed + 1;
catch ME
    fprintf('  ✗ FAILED: %s\n', ME.message);
    testsFailed = testsFailed + 1;
end
fprintf('\n');

%% ===== TEST 3: Create Output Directories =====
fprintf('[Test 3/%d] Creating test output directories...\n', totalTests);
try
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

    fprintf('  ✓ Output directories created\n');
    testsPassed = testsPassed + 1;
catch ME
    fprintf('  ✗ FAILED: %s\n', ME.message);
    testsFailed = testsFailed + 1;
end
fprintf('\n');

%% ===== TEST 4: Test Data Preprocessing =====
fprintf('[Test 4/%d] Testing data preprocessing and loading...\n', totalTests);
try
    [mbqTrain, params] = preprocessAndLoadDatastore(params);

    fprintf('  ✓ Data loaded: %d images\n', params.numImages);
    fprintf('  ✓ Channels detected: %d\n', params.numChannels);
    fprintf('  ✓ Batch size: %d\n', params.miniBatchSize);

    % Test getting a batch
    testBatch = next(mbqTrain);
    batchShape = size(testBatch);
    fprintf('  ✓ Test batch shape: [%d %d %d %d]\n', batchShape(1), batchShape(2), batchShape(3), batchShape(4));

    % Reset for later use
    reset(mbqTrain);

    testsPassed = testsPassed + 1;
catch ME
    fprintf('  ✗ FAILED: %s\n', ME.message);
    fprintf('    Stack: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
    testsFailed = testsFailed + 1;
end
fprintf('\n');

%% ===== TEST 5: Test Generator Network =====
fprintf('[Test 5/%d] Testing Generator network...\n', totalTests);
try
    netG = buildGenerator(params);

    fprintf('  ✓ Generator created\n');
    fprintf('  ✓ Parameters: %d\n', sum(cellfun(@numel, netG.Learnables.Value)));

    % Test forward pass
    Z_test = dlarray(randn(params.latentDim, 2, 'single'), 'CB');
    X_gen = predict(netG, Z_test);
    genShape = size(X_gen);

    fprintf('  ✓ Forward pass successful\n');
    fprintf('  ✓ Output shape: [%d %d %d %d]\n', genShape(1), genShape(2), genShape(3), genShape(4));

    % Validate shape
    assert(genShape(1) == params.trainSize(1), 'Height mismatch');
    assert(genShape(2) == params.trainSize(2), 'Width mismatch');
    assert(genShape(3) == params.numChannels, 'Channels mismatch');

    fprintf('  ✓ Output shape validation passed\n');
    testsPassed = testsPassed + 1;
catch ME
    fprintf('  ✗ FAILED: %s\n', ME.message);
    if ~isempty(ME.stack)
        fprintf('    Stack: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
    end
    testsFailed = testsFailed + 1;
end
fprintf('\n');

%% ===== TEST 6: Test Discriminator Network =====
fprintf('[Test 6/%d] Testing Discriminator network...\n', totalTests);
try
    netD = buildDiscriminator(params);

    fprintf('  ✓ Discriminator created\n');
    fprintf('  ✓ Parameters: %d\n', sum(cellfun(@numel, netD.Learnables.Value)));

    % Test forward pass
    X_test = dlarray(randn(params.trainSize(1), params.trainSize(2), params.numChannels, 2, 'single'), 'SSCB');
    Y_disc = predict(netD, X_test);
    discShape = size(Y_disc);

    fprintf('  ✓ Forward pass successful\n');
    fprintf('  ✓ Output shape: [%d %d %d %d]\n', discShape(1), discShape(2), discShape(3), discShape(4));

    % Validate output is scalar per image
    assert(discShape(1) == 1 && discShape(2) == 1 && discShape(3) == 1, 'Output should be scalar');

    fprintf('  ✓ Output shape validation passed\n');
    testsPassed = testsPassed + 1;
catch ME
    fprintf('  ✗ FAILED: %s\n', ME.message);
    if ~isempty(ME.stack)
        fprintf('    Stack: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
    end
    testsFailed = testsFailed + 1;
end
fprintf('\n');

%% ===== TEST 7: Test Generator-Discriminator Integration =====
fprintf('[Test 7/%d] Testing Generator-Discriminator integration...\n', totalTests);
try
    % Generate fake images
    Z_test = dlarray(randn(params.latentDim, 2, 'single'), 'CB');
    X_fake = predict(netG, Z_test);

    % Discriminate fake images
    Y_fake = predict(netD, X_fake);

    fprintf('  ✓ Generated images from noise\n');
    fprintf('  ✓ Discriminated generated images\n');
    fprintf('  ✓ Integration test passed\n');
    testsPassed = testsPassed + 1;
catch ME
    fprintf('  ✗ FAILED: %s\n', ME.message);
    testsFailed = testsFailed + 1;
end
fprintf('\n');

%% ===== TEST 8: Test Gradient Computation =====
fprintf('[Test 8/%d] Testing gradient computation...\n', totalTests);
try
    % Get a real batch
    reset(mbqTrain);
    XReal = next(mbqTrain);
    batchSize = size(XReal, 4);

    % Generate latent vectors
    Z = dlarray(randn(params.latentDim, batchSize, 'single'), 'CB');

    % Test discriminator gradients
    [gradD, lossD] = dlfeval(@modelGradientsD, netD, netG, XReal, Z, params);
    fprintf('  ✓ Discriminator gradients computed\n');
    fprintf('    D Loss: %.4f\n', extractdata(lossD));

    % Test generator gradients
    [gradG, lossG] = dlfeval(@modelGradientsG, netD, netG, Z);
    fprintf('  ✓ Generator gradients computed\n');
    fprintf('    G Loss: %.4f\n', extractdata(lossG));

    testsPassed = testsPassed + 1;
catch ME
    fprintf('  ✗ FAILED: %s\n', ME.message);
    if ~isempty(ME.stack)
        fprintf('    Stack: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
    end
    testsFailed = testsFailed + 1;
end
fprintf('\n');

%% ===== TEST 9: Test Training Loop (2 iterations) =====
fprintf('[Test 9/%d] Testing mini training loop (2 iterations)...\n', totalTests);
try
    % Initialize Adam optimizers
    avgG = [];
    avgGS = [];
    avgD = [];
    avgDS = [];

    reset(mbqTrain);

    for iter = 1:2
        % Get batch
        XReal = next(mbqTrain);
        batchSize = size(XReal, 4);

        % Generate latent vectors
        Z = dlarray(randn(params.latentDim, batchSize, 'single'), 'CB');

        % Train Discriminator
        [gradD, lossD] = dlfeval(@modelGradientsD, netD, netG, XReal, Z, params);
        [netD, avgD, avgDS] = adamupdate(netD, gradD, avgD, avgDS, iter, ...
            params.learnRate, params.beta1);

        % Train Generator
        [gradG, lossG] = dlfeval(@modelGradientsG, netD, netG, Z);
        [netG, avgG, avgGS] = adamupdate(netG, gradG, avgG, avgGS, iter, ...
            params.learnRate, params.beta1);

        fprintf('    Iter %d: D Loss = %.4f, G Loss = %.4f\n', ...
            iter, extractdata(lossD), extractdata(lossG));
    end

    fprintf('  ✓ Training loop executed successfully\n');
    testsPassed = testsPassed + 1;
catch ME
    fprintf('  ✗ FAILED: %s\n', ME.message);
    if ~isempty(ME.stack)
        fprintf('    Stack: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
    end
    testsFailed = testsFailed + 1;
end
fprintf('\n');

%% ===== TEST 10: Test Image Grid Saving =====
fprintf('[Test 10/%d] Testing image grid saving...\n', totalTests);
try
    % Generate test images
    Z_preview = dlarray(randn(params.latentDim, 4, 'single'), 'CB');
    X_preview = predict(netG, Z_preview);
    X_preview = extractdata(X_preview);

    % Save grid
    testGridFile = fullfile(params.outputFolder, 'preview', 'test_grid.png');
    saveImageGrid(X_preview, testGridFile, params);

    % Verify file exists
    assert(exist(testGridFile, 'file') == 2, 'Grid file not created');

    fprintf('  ✓ Image grid saved: %s\n', testGridFile);
    testsPassed = testsPassed + 1;
catch ME
    fprintf('  ✗ FAILED: %s\n', ME.message);
    testsFailed = testsFailed + 1;
end
fprintf('\n');

%% ===== TEST 11: Test Synthetic Image Generation =====
fprintf('[Test 11/%d] Testing synthetic image generation...\n', totalTests);
try
    % Generate 5 synthetic images
    generateSynthetic(netG, params);

    % Verify files exist
    syntheticFiles = dir(fullfile(params.outputFolder, 'synthetic', '*.png'));
    numGenerated = length(syntheticFiles);

    assert(numGenerated == params.numSynthetic, ...
        sprintf('Expected %d images, got %d', params.numSynthetic, numGenerated));

    fprintf('  ✓ Generated %d synthetic images\n', numGenerated);
    testsPassed = testsPassed + 1;
catch ME
    fprintf('  ✗ FAILED: %s\n', ME.message);
    testsFailed = testsFailed + 1;
end
fprintf('\n');

%% ===== TEST 12: Test Model Saving/Loading =====
fprintf('[Test 12/%d] Testing model save/load...\n', totalTests);
try
    % Save models
    modelPath = fullfile(params.outputFolder, 'models');
    save(fullfile(modelPath, 'test_generator.mat'), 'netG', 'params');
    save(fullfile(modelPath, 'test_discriminator.mat'), 'netD');

    fprintf('  ✓ Models saved\n');

    % Load models
    loadedData = load(fullfile(modelPath, 'test_generator.mat'));
    netG_loaded = loadedData.netG;

    fprintf('  ✓ Models loaded\n');

    % Test loaded generator
    Z_test = dlarray(randn(params.latentDim, 1, 'single'), 'CB');
    X_test = predict(netG_loaded, Z_test);

    fprintf('  ✓ Loaded generator works\n');
    testsPassed = testsPassed + 1;
catch ME
    fprintf('  ✗ FAILED: %s\n', ME.message);
    testsFailed = testsFailed + 1;
end
fprintf('\n');

%% ===== SUMMARY =====
fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║                      Test Summary                          ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n');
fprintf('\n');
fprintf('Total Tests: %d\n', totalTests);
fprintf('Passed:      %d ✓\n', testsPassed);
fprintf('Failed:      %d ✗\n', testsFailed);
fprintf('\n');

if testsFailed == 0
    fprintf('╔════════════════════════════════════════════════════════════╗\n');
    fprintf('║  ✓✓✓  ALL TESTS PASSED!  ✓✓✓                             ║\n');
    fprintf('║                                                            ║\n');
    fprintf('║  The GAN pipeline is working correctly!                   ║\n');
    fprintf('║  You can now run: GAN                                     ║\n');
    fprintf('╚════════════════════════════════════════════════════════════╝\n');
else
    fprintf('╔════════════════════════════════════════════════════════════╗\n');
    fprintf('║  ⚠  SOME TESTS FAILED  ⚠                                  ║\n');
    fprintf('║                                                            ║\n');
    fprintf('║  Please review the errors above                           ║\n');
    fprintf('╚════════════════════════════════════════════════════════════╝\n');
end
fprintf('\n');

fprintf('Test outputs saved to: %s\n', params.outputFolder);
fprintf('  - Preview images: %s\n', fullfile(params.outputFolder, 'preview'));
fprintf('  - Synthetic images: %s\n', fullfile(params.outputFolder, 'synthetic'));
fprintf('  - Models: %s\n', fullfile(params.outputFolder, 'models'));
fprintf('\n');

%% ===== Helper Functions (copy from train_gan.m) =====

function [gradD, lossD] = modelGradientsD(netD, netG, XReal, Z, params)
    % Discriminator gradients

    % Generate fake images
    XGenerated = forward(netG, Z);

    % Discriminator outputs
    YReal = forward(netD, XReal);
    YGenerated = forward(netD, XGenerated);

    % Label smoothing: real target = 0.9, fake target = 0.0
    lossReal = -mean(params.labelSmoothing * log(YReal + 1e-8) + ...
                     (1 - params.labelSmoothing) * log(1 - YReal + 1e-8));
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
