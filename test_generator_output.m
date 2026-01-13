%% TEST GENERATOR OUTPUT - Check if generator outputs true RGB or fake RGB
% Run AFTER training completes
%
% cd 'G:\Fax\Prijava Teme\Clanek za busbare\Code'
% test_generator_output

function test_generator_output()
    fprintf('========================================\n');
    fprintf('Testing Generator Output\n');
    fprintf('========================================\n\n');

    % Check if model exists
    modelPath = './outputs/models/generator.mat';
    if ~exist(modelPath, 'file')
        error('Model not found: %s\nTrain the model first!', modelPath);
    end

    % Load generator
    fprintf('Loading generator...\n');
    loaded = load(modelPath);
    netG = loaded.netG;
    params = loaded.params;

    fprintf('  latentDim: %d\n', params.latentDim);
    fprintf('  numChannels: %d\n', params.numChannels);
    fprintf('  trainSize: [%d %d]\n\n', params.trainSize(1), params.trainSize(2));

    % Generate 10 test images
    numTest = 10;
    fprintf('Generating %d test images...\n\n', numTest);

    numTrueRGB = 0;
    numFakeRGB = 0;

    for i = 1:numTest
        % Generate random latent vector
        Z = randn(params.latentDim, 1, 'single');
        Z = dlarray(Z, 'CB');

        % Move to GPU if available
        if canUseGPU
            Z = gpuArray(Z);
        end

        % Generate image
        XGenerated = predict(netG, Z);
        XGenerated = extractdata(gather(XGenerated));

        % Denormalize from [-1, 1] to [0, 1]
        XGenerated = (XGenerated + 1) / 2;
        XGenerated = max(min(XGenerated, 1), 0);

        % Get image (first in batch)
        img = XGenerated(:, :, :, 1);

        % Check RGB
        fprintf('[%d/%d] Generated image\n', i, numTest);
        isRGB = checkGeneratedRGB(img);

        if isRGB
            numTrueRGB = numTrueRGB + 1;
        else
            numFakeRGB = numFakeRGB + 1;
        end

        % Save first 3 images for inspection
        if i <= 3
            testFolder = './test_output';
            if ~exist(testFolder, 'dir')
                mkdir(testFolder);
            end

            img_uint8 = uint8(img * 255);
            filename = sprintf('test_generated_%d.png', i);
            imwrite(img_uint8, fullfile(testFolder, filename));
        end
    end

    % Summary
    fprintf('\n========================================\n');
    fprintf('SUMMARY\n');
    fprintf('========================================\n');
    fprintf('True RGB outputs:  %d / %d\n', numTrueRGB, numTest);
    fprintf('Fake RGB outputs:  %d / %d\n', numFakeRGB, numTest);
    fprintf('\n');

    if numFakeRGB == numTest
        fprintf('🔴 PROBLEM FOUND: Generator outputs FAKE RGB!\n');
        fprintf('\n');
        fprintf('Generator produces 3 channels but R=G=B (grayscale).\n');
        fprintf('\n');
        fprintf('POSSIBLE CAUSES:\n');
        fprintf('1. Generator architecture: All 3 output channels are identical\n');
        fprintf('   - Check buildGenerator.m final layer\n');
        fprintf('   - transposedConv2dLayer should output 3 INDEPENDENT channels\n');
        fprintf('\n');
        fprintf('2. Training data was grayscale (but you verified it is RGB!)\n');
        fprintf('\n');
        fprintf('3. Discriminator cannot distinguish RGB vs grayscale\n');
        fprintf('   - Check buildDiscriminator.m input layer\n');
        fprintf('\n');
        fprintf('4. Mode collapse: Generator learned to output only gray\n');
        fprintf('   - Training might need more epochs\n');
        fprintf('   - Check loss curves\n');
        fprintf('\n');
        fprintf('SOLUTION:\n');
        fprintf('I will investigate buildGenerator.m and buildDiscriminator.m\n');
        fprintf('to find the bug causing grayscale output!\n');
    elseif numTrueRGB > 0 && numFakeRGB > 0
        fprintf('⚠️  MIXED: Generator sometimes outputs RGB, sometimes grayscale\n');
        fprintf('\n');
        fprintf('This suggests mode collapse or instability during training.\n');
        fprintf('\n');
        fprintf('Check:\n');
        fprintf('1. Loss curves (discriminator might be too strong)\n');
        fprintf('2. Training might need more epochs\n');
        fprintf('3. Learning rate might be too high\n');
    else
        fprintf('✅ GOOD: Generator outputs TRUE RGB!\n');
        fprintf('\n');
        fprintf('Generated images have different R, G, B values.\n');
        fprintf('\n');
        fprintf('If synthetic images still look gray to you:\n');
        fprintf('1. Check saved images in ./outputs/synthetic/\n');
        fprintf('2. They might have subtle color that looks gray\n');
        fprintf('3. Training might need more epochs for stronger colors\n');
        fprintf('4. Try increasing learning rate or training longer\n');
    end

    fprintf('\nTest images saved to: ./test_output/\n');
    fprintf('Open them to visually inspect colors!\n\n');
end

function isRGB = checkGeneratedRGB(img)
    % Check if generated image is true RGB or fake RGB
    % img is in [0, 1] range

    if size(img, 3) ~= 3
        fprintf('  ❌ Wrong channels: %d (expected 3)\n\n', size(img, 3));
        isRGB = false;
        return;
    end

    % Extract channels
    R = img(:,:,1);
    G = img(:,:,2);
    B = img(:,:,3);

    % Sample 500 pixels
    numPixels = numel(R);
    sampleSize = min(500, numPixels);
    idx = randperm(numPixels, sampleSize);

    R_sample = R(idx);
    G_sample = G(idx);
    B_sample = B(idx);

    % Calculate statistics
    R_mean = mean(R_sample);
    G_mean = mean(G_sample);
    B_mean = mean(B_sample);

    % MAD (mean absolute difference)
    mad_RG = mean(abs(R_sample - G_sample));
    mad_GB = mean(abs(G_sample - B_sample));
    mad_RB = mean(abs(R_sample - B_sample));

    % Threshold (for [0,1] range)
    THRESHOLD = 0.02;  % 2% difference

    fprintf('  Channel means: R=%.3f, G=%.3f, B=%.3f\n', R_mean, G_mean, B_mean);
    fprintf('  Channel diffs (MAD): RG=%.3f, GB=%.3f, RB=%.3f\n', mad_RG, mad_GB, mad_RB);

    if mad_RG < THRESHOLD && mad_GB < THRESHOLD && mad_RB < THRESHOLD
        fprintf('  ❌ FAKE RGB (R≈G≈B) - Grayscale output!\n\n');
        isRGB = false;
    else
        fprintf('  ✅ TRUE RGB - Colored output!\n\n');
        isRGB = true;
    end
end
