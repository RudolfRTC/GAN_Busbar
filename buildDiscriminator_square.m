function netD = buildDiscriminator_square(params)
%BUILDDISCRIMINATOR_SQUARE Build DCGAN Discriminator for SQUARE images
%
% Supports: [64 64], [128 128], [256 256]
%
% Architecture: Convolution layers with LeakyReLU and Dropout
% Input: square image [size x size x 3]
% Output: scalar probability [0, 1] (sigmoid)
%
% Inputs:
%   params - struct with:
%     .trainSize - [height width] MUST BE SQUARE!
%     .numChannels - number of input channels (1 or 3)
%
% Output:
%   netD - dlnetwork discriminator

    trainHeight = params.trainSize(1);
    trainWidth = params.trainSize(2);
    numChannels = params.numChannels;

    % Validate SQUARE
    if trainHeight ~= trainWidth
        error('buildDiscriminator_square only supports SQUARE images!\ntrainSize = [%d %d] is not square.', ...
            trainHeight, trainWidth);
    end

    size = trainHeight;

    %% Helper for DCGAN-style conv layer
    conv2d = @(filterSize, numFilters, name, varargin) ...
        convolution2dLayer(filterSize, numFilters, 'Name', name, ...
            'WeightsInitializer', @(sz) 0.02*randn(sz, 'single'), ...
            'BiasInitializer', 'zeros', varargin{:});

    %% Architecture: 64x64
    if size == 64
        layers = [
            imageInputLayer([size size numChannels], 'Name', 'input', 'Normalization', 'none')

            % 64x64 -> 32x32
            conv2d(4, 64, 'conv1', 'Stride', 2, 'Padding', 'same')
            leakyReluLayer(0.2, 'Name', 'lrelu1')
            dropoutLayer(0.3, 'Name', 'dropout1')

            % 32x32 -> 16x16
            conv2d(4, 128, 'conv2', 'Stride', 2, 'Padding', 'same')
            batchNormalizationLayer('Name', 'bn2')
            leakyReluLayer(0.2, 'Name', 'lrelu2')
            dropoutLayer(0.3, 'Name', 'dropout2')

            % 16x16 -> 8x8
            conv2d(4, 256, 'conv3', 'Stride', 2, 'Padding', 'same')
            batchNormalizationLayer('Name', 'bn3')
            leakyReluLayer(0.2, 'Name', 'lrelu3')
            dropoutLayer(0.3, 'Name', 'dropout3')

            % 8x8 -> 4x4
            conv2d(4, 512, 'conv4', 'Stride', 2, 'Padding', 'same')
            batchNormalizationLayer('Name', 'bn4')
            leakyReluLayer(0.2, 'Name', 'lrelu4')
            dropoutLayer(0.3, 'Name', 'dropout4')

            % 4x4 -> 1x1 (FC)
            fullyConnectedLayer(1, 'Name', 'fc')
            sigmoidLayer('Name', 'sigmoid')
        ];

    %% Architecture: 128x128
    elseif size == 128
        layers = [
            imageInputLayer([size size numChannels], 'Name', 'input', 'Normalization', 'none')

            % 128x128 -> 64x64
            conv2d(4, 64, 'conv1', 'Stride', 2, 'Padding', 'same')
            leakyReluLayer(0.2, 'Name', 'lrelu1')
            dropoutLayer(0.3, 'Name', 'dropout1')

            % 64x64 -> 32x32
            conv2d(4, 128, 'conv2', 'Stride', 2, 'Padding', 'same')
            batchNormalizationLayer('Name', 'bn2')
            leakyReluLayer(0.2, 'Name', 'lrelu2')
            dropoutLayer(0.3, 'Name', 'dropout2')

            % 32x32 -> 16x16
            conv2d(4, 256, 'conv3', 'Stride', 2, 'Padding', 'same')
            batchNormalizationLayer('Name', 'bn3')
            leakyReluLayer(0.2, 'Name', 'lrelu3')
            dropoutLayer(0.3, 'Name', 'dropout3')

            % 16x16 -> 8x8
            conv2d(4, 512, 'conv4', 'Stride', 2, 'Padding', 'same')
            batchNormalizationLayer('Name', 'bn4')
            leakyReluLayer(0.2, 'Name', 'lrelu4')
            dropoutLayer(0.3, 'Name', 'dropout4')

            % 8x8 -> 4x4
            conv2d(4, 512, 'conv5', 'Stride', 2, 'Padding', 'same')
            batchNormalizationLayer('Name', 'bn5')
            leakyReluLayer(0.2, 'Name', 'lrelu5')
            dropoutLayer(0.3, 'Name', 'dropout5')

            % 4x4 -> 1x1 (FC)
            fullyConnectedLayer(1, 'Name', 'fc')
            sigmoidLayer('Name', 'sigmoid')
        ];

    %% Architecture: 256x256
    elseif size == 256
        layers = [
            imageInputLayer([size size numChannels], 'Name', 'input', 'Normalization', 'none')

            % 256x256 -> 128x128
            conv2d(4, 64, 'conv1', 'Stride', 2, 'Padding', 'same')
            leakyReluLayer(0.2, 'Name', 'lrelu1')
            dropoutLayer(0.3, 'Name', 'dropout1')

            % 128x128 -> 64x64
            conv2d(4, 128, 'conv2', 'Stride', 2, 'Padding', 'same')
            batchNormalizationLayer('Name', 'bn2')
            leakyReluLayer(0.2, 'Name', 'lrelu2')
            dropoutLayer(0.3, 'Name', 'dropout2')

            % 64x64 -> 32x32
            conv2d(4, 256, 'conv3', 'Stride', 2, 'Padding', 'same')
            batchNormalizationLayer('Name', 'bn3')
            leakyReluLayer(0.2, 'Name', 'lrelu3')
            dropoutLayer(0.3, 'Name', 'dropout3')

            % 32x32 -> 16x16
            conv2d(4, 512, 'conv4', 'Stride', 2, 'Padding', 'same')
            batchNormalizationLayer('Name', 'bn4')
            leakyReluLayer(0.2, 'Name', 'lrelu4')
            dropoutLayer(0.3, 'Name', 'dropout4')

            % 16x16 -> 8x8
            conv2d(4, 512, 'conv5', 'Stride', 2, 'Padding', 'same')
            batchNormalizationLayer('Name', 'bn5')
            leakyReluLayer(0.2, 'Name', 'lrelu5')
            dropoutLayer(0.3, 'Name', 'dropout5')

            % 8x8 -> 4x4
            conv2d(4, 512, 'conv6', 'Stride', 2, 'Padding', 'same')
            batchNormalizationLayer('Name', 'bn6')
            leakyReluLayer(0.2, 'Name', 'lrelu6')
            dropoutLayer(0.3, 'Name', 'dropout6')

            % 4x4 -> 1x1 (FC)
            fullyConnectedLayer(1, 'Name', 'fc')
            sigmoidLayer('Name', 'sigmoid')
        ];
    else
        error('Supported sizes: 64x64, 128x128, 256x256. Got: %dx%d', size, size);
    end

    %% Create dlnetwork
    lgraph = layerGraph(layers);
    netD = dlnetwork(lgraph);

    fprintf('  Square Discriminator built successfully:\n');
    fprintf('    Input shape:  [%d x %d x %d x N]\n', size, size, numChannels);
    fprintf('    Output shape: [1 x N] (probability)\n');
    fprintf('    Total parameters: %d\n', sum(cellfun(@numel, netD.Learnables.Value)));
end
