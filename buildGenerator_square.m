function netG = buildGenerator_square(params)
%BUILDGENERATOR_SQUARE Build DCGAN Generator for SQUARE images
%
% Supports: [64 64], [128 128], [256 256]
%
% Architecture: FeatureInputLayer -> FC -> Reshape -> Transposed Convolutions
% Input: latent vector [latentDim x N]
% Output: square image [size x size x 3 x N], tanh activation
%
% Inputs:
%   params - struct with:
%     .latentDim - latent vector dimension (e.g., 100)
%     .trainSize - [height width] MUST BE SQUARE!
%     .numChannels - number of output channels (1 or 3)
%
% Output:
%   netG - dlnetwork generator

    latentDim = params.latentDim;
    trainHeight = params.trainSize(1);
    trainWidth = params.trainSize(2);
    numChannels = params.numChannels;

    % Validate SQUARE
    if trainHeight ~= trainWidth
        error('buildGenerator_square only supports SQUARE images!\ntrainSize = [%d %d] is not square.', ...
            trainHeight, trainWidth);
    end

    size = trainHeight;

    % Determine architecture
    if size == 64
        % 64x64: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 (4 upsamples)
        startSize = 4;
        numUpsamples = 4;
    elseif size == 128
        % 128x128: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128 (5 upsamples)
        startSize = 4;
        numUpsamples = 5;
    elseif size == 256
        % 256x256: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256 (6 upsamples)
        startSize = 4;
        numUpsamples = 6;
    else
        error('Supported sizes: 64x64, 128x128, 256x256. Got: %dx%d', size, size);
    end

    startChannels = 256;

    %% Architecture: 64x64
    if size == 64
        layers = [
            featureInputLayer(latentDim, 'Name', 'input', 'Normalization', 'none')
            fullyConnectedLayer(startSize * startSize * startChannels, 'Name', 'fc', ...
                'WeightsInitializer', @(sz) 0.02*randn(sz, 'single'), 'BiasInitializer', 'zeros')
            functionLayer(@(X) reshapeTensor(X, startSize, startSize, startChannels), ...
                'Name', 'reshape', 'Formattable', true)
            batchNormalizationLayer('Name', 'bn0')
            leakyReluLayer(0.2, 'Name', 'lrelu0')

            % 4x4 -> 8x8
            transposedConv2dLayer(4, 128, 'Name', 'tconv1', 'Stride', 2, 'Cropping', 'same', ...
                'WeightsInitializer', @(sz) 0.02*randn(sz, 'single'), 'BiasInitializer', 'zeros')
            batchNormalizationLayer('Name', 'bn1')
            leakyReluLayer(0.2, 'Name', 'lrelu1')

            % 8x8 -> 16x16
            transposedConv2dLayer(4, 128, 'Name', 'tconv2', 'Stride', 2, 'Cropping', 'same', ...
                'WeightsInitializer', @(sz) 0.02*randn(sz, 'single'), 'BiasInitializer', 'zeros')
            batchNormalizationLayer('Name', 'bn2')
            leakyReluLayer(0.2, 'Name', 'lrelu2')

            % 16x16 -> 32x32
            transposedConv2dLayer(4, 64, 'Name', 'tconv3', 'Stride', 2, 'Cropping', 'same', ...
                'WeightsInitializer', @(sz) 0.02*randn(sz, 'single'), 'BiasInitializer', 'zeros')
            batchNormalizationLayer('Name', 'bn3')
            leakyReluLayer(0.2, 'Name', 'lrelu3')

            % 32x32 -> 64x64 (final)
            transposedConv2dLayer(4, numChannels, 'Name', 'tconv4', 'Stride', 2, 'Cropping', 'same', ...
                'WeightsInitializer', @(sz) 0.02*randn(sz, 'single'), 'BiasInitializer', 'zeros')
            tanhLayer('Name', 'tanh')
        ];

    %% Architecture: 128x128
    elseif size == 128
        layers = [
            featureInputLayer(latentDim, 'Name', 'input', 'Normalization', 'none')
            fullyConnectedLayer(startSize * startSize * startChannels, 'Name', 'fc', ...
                'WeightsInitializer', @(sz) 0.02*randn(sz, 'single'), 'BiasInitializer', 'zeros')
            functionLayer(@(X) reshapeTensor(X, startSize, startSize, startChannels), ...
                'Name', 'reshape', 'Formattable', true)
            batchNormalizationLayer('Name', 'bn0')
            leakyReluLayer(0.2, 'Name', 'lrelu0')

            % 4x4 -> 8x8
            transposedConv2dLayer(4, 128, 'Name', 'tconv1', 'Stride', 2, 'Cropping', 'same', ...
                'WeightsInitializer', @(sz) 0.02*randn(sz, 'single'), 'BiasInitializer', 'zeros')
            batchNormalizationLayer('Name', 'bn1')
            leakyReluLayer(0.2, 'Name', 'lrelu1')

            % 8x8 -> 16x16
            transposedConv2dLayer(4, 128, 'Name', 'tconv2', 'Stride', 2, 'Cropping', 'same', ...
                'WeightsInitializer', @(sz) 0.02*randn(sz, 'single'), 'BiasInitializer', 'zeros')
            batchNormalizationLayer('Name', 'bn2')
            leakyReluLayer(0.2, 'Name', 'lrelu2')

            % 16x16 -> 32x32
            transposedConv2dLayer(4, 128, 'Name', 'tconv3', 'Stride', 2, 'Cropping', 'same', ...
                'WeightsInitializer', @(sz) 0.02*randn(sz, 'single'), 'BiasInitializer', 'zeros')
            batchNormalizationLayer('Name', 'bn3')
            leakyReluLayer(0.2, 'Name', 'lrelu3')

            % 32x32 -> 64x64
            transposedConv2dLayer(4, 64, 'Name', 'tconv4', 'Stride', 2, 'Cropping', 'same', ...
                'WeightsInitializer', @(sz) 0.02*randn(sz, 'single'), 'BiasInitializer', 'zeros')
            batchNormalizationLayer('Name', 'bn4')
            leakyReluLayer(0.2, 'Name', 'lrelu4')

            % 64x64 -> 128x128 (final)
            transposedConv2dLayer(4, numChannels, 'Name', 'tconv5', 'Stride', 2, 'Cropping', 'same', ...
                'WeightsInitializer', @(sz) 0.02*randn(sz, 'single'), 'BiasInitializer', 'zeros')
            tanhLayer('Name', 'tanh')
        ];

    %% Architecture: 256x256
    elseif size == 256
        layers = [
            featureInputLayer(latentDim, 'Name', 'input', 'Normalization', 'none')
            fullyConnectedLayer(startSize * startSize * startChannels, 'Name', 'fc', ...
                'WeightsInitializer', @(sz) 0.02*randn(sz, 'single'), 'BiasInitializer', 'zeros')
            functionLayer(@(X) reshapeTensor(X, startSize, startSize, startChannels), ...
                'Name', 'reshape', 'Formattable', true)
            batchNormalizationLayer('Name', 'bn0')
            leakyReluLayer(0.2, 'Name', 'lrelu0')

            % 4x4 -> 8x8
            transposedConv2dLayer(4, 128, 'Name', 'tconv1', 'Stride', 2, 'Cropping', 'same', ...
                'WeightsInitializer', @(sz) 0.02*randn(sz, 'single'), 'BiasInitializer', 'zeros')
            batchNormalizationLayer('Name', 'bn1')
            leakyReluLayer(0.2, 'Name', 'lrelu1')

            % 8x8 -> 16x16
            transposedConv2dLayer(4, 128, 'Name', 'tconv2', 'Stride', 2, 'Cropping', 'same', ...
                'WeightsInitializer', @(sz) 0.02*randn(sz, 'single'), 'BiasInitializer', 'zeros')
            batchNormalizationLayer('Name', 'bn2')
            leakyReluLayer(0.2, 'Name', 'lrelu2')

            % 16x16 -> 32x32
            transposedConv2dLayer(4, 128, 'Name', 'tconv3', 'Stride', 2, 'Cropping', 'same', ...
                'WeightsInitializer', @(sz) 0.02*randn(sz, 'single'), 'BiasInitializer', 'zeros')
            batchNormalizationLayer('Name', 'bn3')
            leakyReluLayer(0.2, 'Name', 'lrelu3')

            % 32x32 -> 64x64
            transposedConv2dLayer(4, 128, 'Name', 'tconv4', 'Stride', 2, 'Cropping', 'same', ...
                'WeightsInitializer', @(sz) 0.02*randn(sz, 'single'), 'BiasInitializer', 'zeros')
            batchNormalizationLayer('Name', 'bn4')
            leakyReluLayer(0.2, 'Name', 'lrelu4')

            % 64x64 -> 128x128
            transposedConv2dLayer(4, 64, 'Name', 'tconv5', 'Stride', 2, 'Cropping', 'same', ...
                'WeightsInitializer', @(sz) 0.02*randn(sz, 'single'), 'BiasInitializer', 'zeros')
            batchNormalizationLayer('Name', 'bn5')
            leakyReluLayer(0.2, 'Name', 'lrelu5')

            % 128x128 -> 256x256 (final)
            transposedConv2dLayer(4, numChannels, 'Name', 'tconv6', 'Stride', 2, 'Cropping', 'same', ...
                'WeightsInitializer', @(sz) 0.02*randn(sz, 'single'), 'BiasInitializer', 'zeros')
            tanhLayer('Name', 'tanh')
        ];
    end

    %% Create dlnetwork
    lgraph = layerGraph(layers);
    netG = dlnetwork(lgraph);

    fprintf('  Square Generator built successfully:\n');
    fprintf('    Input shape:   [%d x N]\n', latentDim);
    fprintf('    Final output:  [%d x %d x %d x N]\n', size, size, numChannels);
    fprintf('    Total parameters: %d\n', sum(cellfun(@numel, netG.Learnables.Value)));
end

%% Helper function for reshaping
function Y = reshapeTensor(X, H, W, C)
    X = extractdata(X);
    N = size(X, 2);
    Y = reshape(X, [H W C N]);
    Y = dlarray(Y, 'SSCB');
end
