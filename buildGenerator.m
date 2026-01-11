function netG = buildGenerator(params)
%BUILDGENERATOR Build DCGAN Generator network
%
% Architecture: FeatureInputLayer -> FC -> Reshape -> Transposed Convolutions
% Input: latent vector [latentDim x N] (using featureInputLayer)
% Output: image [trainSize(1) x trainSize(2) x numChannels x N], tanh activation
%
% This architecture outputs exactly [64 x 128 x 3 x N] for params.trainSize = [64 128]
%
% Inputs:
%   params - struct with:
%     .latentDim - latent vector dimension (e.g., 100)
%     .trainSize - [height width] for training (must be [64 128])
%     .numChannels - number of output channels (1 or 3)
%
% Output:
%   netG - dlnetwork generator

    latentDim = params.latentDim;
    trainHeight = params.trainSize(1);
    trainWidth = params.trainSize(2);
    numChannels = params.numChannels;

    % Validate trainSize
    if trainHeight ~= 64 || trainWidth ~= 128
        error(['This generator architecture is designed for trainSize = [64 128].\n' ...
               'Current trainSize = [%d %d] is not supported.\n' ...
               'Please set params.trainSize = [64 128] in train_gan.m'], ...
               trainHeight, trainWidth);
    end

    %% Architecture: 4x8 -> 8x16 -> 16x32 -> 32x64 -> 64x128
    % Start size: 4x8x256
    % Upsample 4 times with stride [2 2], then final layer with stride [1 1]

    startH = 4;
    startW = 8;
    startChannels = 256;  % Reduced for VRAM constraints (RTX A1000)

    layers = [
        % Input: [latentDim x N]
        featureInputLayer(latentDim, 'Name', 'input', 'Normalization', 'none')

        % Project to 4x8x256 = 8192 features
        % [latentDim x N] -> [8192 x N]
        fullyConnectedLayer(startH * startW * startChannels, 'Name', 'fc')

        % Reshape to [4 x 8 x 256 x N]
        % This uses a custom function to reshape from [8192 x N] to [4 x 8 x 256 x N]
        functionLayer(@(X) reshapeTensor(X, startH, startW, startChannels), ...
            'Name', 'reshape', 'Formattable', true)

        batchNormalizationLayer('Name', 'bn0')
        leakyReluLayer(0.2, 'Name', 'lrelu0')  % LeakyReLU prevents dying neurons

        % Layer 1: 4x8x256 -> 8x16x128
        % Dimension calculation with stride=[2 2], filterSize=4, cropping='same':
        % outputSize = (inputSize - 1) * stride + filterSize - 2*padding
        % For 'same': padding chosen so outputSize ≈ inputSize * stride
        % H: 4 * 2 = 8 ✓
        % W: 8 * 2 = 16 ✓
        transposedConv2dLayer(4, 128, 'Name', 'tconv1', 'Stride', [2 2], 'Cropping', 'same')
        batchNormalizationLayer('Name', 'bn1')
        leakyReluLayer(0.2, 'Name', 'lrelu1')  % LeakyReLU prevents dying neurons

        % Layer 2: 8x16x128 -> 16x32x128
        % H: 8 * 2 = 16 ✓
        % W: 16 * 2 = 32 ✓
        transposedConv2dLayer(4, 128, 'Name', 'tconv2', 'Stride', [2 2], 'Cropping', 'same')
        batchNormalizationLayer('Name', 'bn2')
        leakyReluLayer(0.2, 'Name', 'lrelu2')  % LeakyReLU prevents dying neurons

        % Layer 3: 16x32x128 -> 32x64x64
        % H: 16 * 2 = 32 ✓
        % W: 32 * 2 = 64 ✓
        transposedConv2dLayer(4, 64, 'Name', 'tconv3', 'Stride', [2 2], 'Cropping', 'same')
        batchNormalizationLayer('Name', 'bn3')
        leakyReluLayer(0.2, 'Name', 'lrelu3')  % LeakyReLU prevents dying neurons

        % Layer 4: 32x64x64 -> 64x128x32
        % H: 32 * 2 = 64 ✓
        % W: 64 * 2 = 128 ✓
        transposedConv2dLayer(4, 32, 'Name', 'tconv4', 'Stride', [2 2], 'Cropping', 'same')
        batchNormalizationLayer('Name', 'bn4')
        leakyReluLayer(0.2, 'Name', 'lrelu4')  % LeakyReLU prevents dying neurons

        % Layer 5: 64x128x32 -> 64x128x3 (final output)
        % Stride=[1 1] to maintain spatial size
        % H: 64 * 1 = 64 ✓
        % W: 128 * 1 = 128 ✓
        transposedConv2dLayer(3, numChannels, 'Name', 'tconv5', 'Stride', [1 1], 'Cropping', 'same')
        tanhLayer('Name', 'tanh')
    ];

    %% Create dlnetwork
    lgraph = layerGraph(layers);
    netG = dlnetwork(lgraph);

    % Initialize weights with DCGAN standard: Normal(mean=0, std=0.02)
    netG = initializeGANWeights(netG);

    fprintf('  Generator built successfully:\n');
    fprintf('    Input shape:   [%d x N] (featureInputLayer)\n', latentDim);
    fprintf('    After FC:      [%d x N]\n', startH * startW * startChannels);
    fprintf('    After reshape: [%d x %d x %d x N]\n', startH, startW, startChannels);
    fprintf('    Final output:  [%d x %d x %d x N]\n', trainHeight, trainWidth, numChannels);
    fprintf('    Total parameters: %d\n', sum(cellfun(@numel, netG.Learnables.Value)));
end

%% Helper function for reshaping
function Y = reshapeTensor(X, H, W, C)
    % Reshape from [features x batch] to [H x W x C x batch]
    % Input X: dlarray with format 'CB' (channels/features x batch)
    % Output Y: dlarray with format 'SSCB' (spatial x spatial x channels x batch)

    X = extractdata(X);  % Extract numeric array
    N = size(X, 2);      % Batch size

    % Reshape: [H*W*C x N] -> [H x W x C x N]
    Y = reshape(X, [H W C N]);

    % Convert back to dlarray with correct format
    Y = dlarray(Y, 'SSCB');
end

%% DCGAN Weight Initialization Helper
function net = initializeGANWeights(net)
    % Initialize weights according to DCGAN paper:
    % Normal distribution with mean=0, std=0.02
    % Biases initialized to 0

    for i = 1:height(net.Learnables)
        layerName = net.Learnables.Layer{i};
        paramName = net.Learnables.Parameter{i};

        if contains(paramName, 'Weights')
            % Initialize weights with Normal(0, 0.02)
            sz = size(net.Learnables.Value{i});
            net.Learnables.Value{i} = 0.02 * randn(sz, 'single');
        elseif contains(paramName, 'Bias')
            % Initialize biases to 0
            sz = size(net.Learnables.Value{i});
            net.Learnables.Value{i} = zeros(sz, 'single');
        end
    end
end

% This architecture is sized to fit NVIDIA Quadro RTX A1000 VRAM constraints.
% Maximum channels: 256 (instead of 512 or 1024)
% Batch size: 8 (recommended for RTX A1000)
