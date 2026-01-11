function netD = buildDiscriminator(params)
%BUILDDISCRIMINATOR Build DCGAN Discriminator network
%
% Architecture: Convolution layers with LeakyReLU and Dropout
% Input: image [trainSize(1) x trainSize(2) x numChannels]
% Output: scalar probability [0, 1] (sigmoid)
%
% Inputs:
%   params - struct with:
%     .trainSize - [height width] for training (e.g., [64 128])
%     .numChannels - number of input channels (1 or 3)
%
% Output:
%   netD - dlnetwork discriminator

    trainHeight = params.trainSize(1);
    trainWidth = params.trainSize(2);
    numChannels = params.numChannels;

    layers = [];

    %% Helper function for DCGAN-style convolution layer
    % DCGAN standard: Normal(0, 0.02) initialization
    conv2d = @(filterSize, numFilters, name, varargin) ...
        convolution2dLayer(filterSize, numFilters, 'Name', name, ...
            'WeightsInitializer', @(sz) 0.02*randn(sz, 'single'), ...
            'BiasInitializer', 'zeros', varargin{:});

    %% Architecture selection based on trainSize
    if trainHeight == 64 && trainWidth == 128
        % ===== 64x128 Discriminator =====
        layers = [
            imageInputLayer([trainHeight trainWidth numChannels], 'Name', 'input', ...
                'Normalization', 'none')

            % 64x128 -> 32x64
            conv2d(4, 64, 'conv1', 'Stride', 2, 'Padding', 'same')
            leakyReluLayer(0.2, 'Name', 'lrelu1')
            dropoutLayer(0.3, 'Name', 'dropout1')

            % 32x64 -> 16x32
            conv2d(4, 128, 'conv2', 'Stride', 2, 'Padding', 'same')
            batchNormalizationLayer('Name', 'bn2')
            leakyReluLayer(0.2, 'Name', 'lrelu2')
            dropoutLayer(0.3, 'Name', 'dropout2')

            % 16x32 -> 8x16
            conv2d(4, 256, 'conv3', 'Stride', 2, 'Padding', 'same')
            batchNormalizationLayer('Name', 'bn3')
            leakyReluLayer(0.2, 'Name', 'lrelu3')
            dropoutLayer(0.3, 'Name', 'dropout3')

            % 8x16 -> 4x8
            conv2d(4, 512, 'conv4', 'Stride', 2, 'Padding', 'same')
            batchNormalizationLayer('Name', 'bn4')
            leakyReluLayer(0.2, 'Name', 'lrelu4')
            dropoutLayer(0.3, 'Name', 'dropout4')

            % 4x8 -> 2x4
            conv2d(4, 1024, 'conv5', 'Stride', 2, 'Padding', 'same')
            batchNormalizationLayer('Name', 'bn5')
            leakyReluLayer(0.2, 'Name', 'lrelu5')
            dropoutLayer(0.3, 'Name', 'dropout5')

            % 2x4 -> 1x1
            conv2d([2 4], 1, 'conv6', 'Stride', 1, 'Padding', 0)
            sigmoidLayer('Name', 'sigmoid')
        ];

    elseif trainHeight == 128 && trainWidth == 256
        % ===== 128x256 Discriminator =====
        layers = [
            imageInputLayer([trainHeight trainWidth numChannels], 'Name', 'input', ...
                'Normalization', 'none')

            % 128x256 -> 64x128
            conv2d(4, 64, 'conv1', 'Stride', 2, 'Padding', 'same')
            leakyReluLayer(0.2, 'Name', 'lrelu1')
            dropoutLayer(0.3, 'Name', 'dropout1')

            % 64x128 -> 32x64
            conv2d(4, 128, 'conv2', 'Stride', 2, 'Padding', 'same')
            batchNormalizationLayer('Name', 'bn2')
            leakyReluLayer(0.2, 'Name', 'lrelu2')
            dropoutLayer(0.3, 'Name', 'dropout2')

            % 32x64 -> 16x32
            conv2d(4, 256, 'conv3', 'Stride', 2, 'Padding', 'same')
            batchNormalizationLayer('Name', 'bn3')
            leakyReluLayer(0.2, 'Name', 'lrelu3')
            dropoutLayer(0.3, 'Name', 'dropout3')

            % 16x32 -> 8x16
            conv2d(4, 512, 'conv4', 'Stride', 2, 'Padding', 'same')
            batchNormalizationLayer('Name', 'bn4')
            leakyReluLayer(0.2, 'Name', 'lrelu4')
            dropoutLayer(0.3, 'Name', 'dropout4')

            % 8x16 -> 4x8
            conv2d(4, 1024, 'conv5', 'Stride', 2, 'Padding', 'same')
            batchNormalizationLayer('Name', 'bn5')
            leakyReluLayer(0.2, 'Name', 'lrelu5')
            dropoutLayer(0.3, 'Name', 'dropout5')

            % 4x8 -> 1x1
            conv2d([4 8], 1, 'conv6', 'Stride', 1, 'Padding', 0)
            sigmoidLayer('Name', 'sigmoid')
        ];

    elseif trainHeight == 64 && trainWidth == 64
        % ===== 64x64 Discriminator =====
        layers = [
            imageInputLayer([trainHeight trainWidth numChannels], 'Name', 'input', ...
                'Normalization', 'none')

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

            % 4x4 -> 1x1
            conv2d(4, 1, 'conv5', 'Stride', 1, 'Padding', 0)
            sigmoidLayer('Name', 'sigmoid')
        ];

    elseif trainHeight == 128 && trainWidth == 128
        % ===== 128x128 Discriminator =====
        layers = [
            imageInputLayer([trainHeight trainWidth numChannels], 'Name', 'input', ...
                'Normalization', 'none')

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
            conv2d(4, 1024, 'conv5', 'Stride', 2, 'Padding', 'same')
            batchNormalizationLayer('Name', 'bn5')
            leakyReluLayer(0.2, 'Name', 'lrelu5')
            dropoutLayer(0.3, 'Name', 'dropout5')

            % 4x4 -> 1x1
            conv2d(4, 1, 'conv6', 'Stride', 1, 'Padding', 0)
            sigmoidLayer('Name', 'sigmoid')
        ];

    elseif trainHeight == 256 && trainWidth == 256
        % ===== 256x256 Discriminator =====
        layers = [
            imageInputLayer([trainHeight trainWidth numChannels], 'Name', 'input', ...
                'Normalization', 'none')

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
            conv2d(4, 1024, 'conv5', 'Stride', 2, 'Padding', 'same')
            batchNormalizationLayer('Name', 'bn5')
            leakyReluLayer(0.2, 'Name', 'lrelu5')
            dropoutLayer(0.3, 'Name', 'dropout5')

            % 8x8 -> 4x4
            conv2d(4, 1024, 'conv6', 'Stride', 2, 'Padding', 'same')
            batchNormalizationLayer('Name', 'bn6')
            leakyReluLayer(0.2, 'Name', 'lrelu6')
            dropoutLayer(0.3, 'Name', 'dropout6')

            % 4x4 -> 1x1
            conv2d(4, 1, 'conv7', 'Stride', 1, 'Padding', 0)
            sigmoidLayer('Name', 'sigmoid')
        ];

    else
        error(['Unsupported trainSize: [%d %d]. Supported sizes:\n' ...
               '  [64 128]   - aspect ratio 2:1\n' ...
               '  [128 256]  - aspect ratio 2:1\n' ...
               '  [64 64]    - square\n' ...
               '  [128 128]  - square\n' ...
               '  [256 256]  - square'], ...
               trainHeight, trainWidth);
    end

    %% Create dlnetwork
    lgraph = layerGraph(layers);
    netD = dlnetwork(lgraph);

    % NOTE: Weights are initialized at layer definition time, not post-hoc.
    % This approach is more robust across MATLAB versions and avoids issues
    % with dlnetwork.Learnables format (which requires cell arrays containing
    % dlarray objects in newer MATLAB versions, not raw numeric arrays).

    fprintf('  Discriminator built successfully:\n');
    fprintf('    Input shape:  [%d x %d x %d x N]\n', trainHeight, trainWidth, numChannels);
    fprintf('    Output shape: [1 x 1 x 1 x N]\n');
end
