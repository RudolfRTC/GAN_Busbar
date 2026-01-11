function netD = buildDiscriminator(params)
%BUILDDISCRIMINATOR Build DCGAN Discriminator network
%
% Architecture: Convolution layers with LeakyReLU and Dropout
% Input: image [imageSize x imageSize x numChannels]
% Output: scalar probability [0, 1] (sigmoid)
%
% Inputs:
%   params - struct with:
%     .imageSize - input image size (64 or 128)
%     .numChannels - number of input channels (1 or 3)
%
% Output:
%   netD - dlnetwork discriminator

    imageSize = params.imageSize;
    numChannels = params.numChannels;

    layers = [];

    %% Architecture depends on image size
    if imageSize == 64
        % ===== 64x64 Discriminator =====
        % Input: 64x64xnumChannels
        layers = [
            imageInputLayer([imageSize imageSize numChannels], 'Name', 'input', ...
                'Normalization', 'none')

            % 64x64xC -> 32x32x64
            convolution2dLayer(4, 64, 'Name', 'conv1', ...
                'Stride', 2, 'Padding', 'same')
            leakyReluLayer(0.2, 'Name', 'lrelu1')
            dropoutLayer(0.3, 'Name', 'dropout1')

            % 32x32x64 -> 16x16x128
            convolution2dLayer(4, 128, 'Name', 'conv2', ...
                'Stride', 2, 'Padding', 'same')
            batchNormalizationLayer('Name', 'bn2')
            leakyReluLayer(0.2, 'Name', 'lrelu2')
            dropoutLayer(0.3, 'Name', 'dropout2')

            % 16x16x128 -> 8x8x256
            convolution2dLayer(4, 256, 'Name', 'conv3', ...
                'Stride', 2, 'Padding', 'same')
            batchNormalizationLayer('Name', 'bn3')
            leakyReluLayer(0.2, 'Name', 'lrelu3')
            dropoutLayer(0.3, 'Name', 'dropout3')

            % 8x8x256 -> 4x4x512
            convolution2dLayer(4, 512, 'Name', 'conv4', ...
                'Stride', 2, 'Padding', 'same')
            batchNormalizationLayer('Name', 'bn4')
            leakyReluLayer(0.2, 'Name', 'lrelu4')
            dropoutLayer(0.3, 'Name', 'dropout4')

            % 4x4x512 -> 1x1x1
            convolution2dLayer(4, 1, 'Name', 'conv5', ...
                'Stride', 1, 'Padding', 0)
            sigmoidLayer('Name', 'sigmoid')
        ];

    elseif imageSize == 128
        % ===== 128x128 Discriminator =====
        % Input: 128x128xnumChannels
        layers = [
            imageInputLayer([imageSize imageSize numChannels], 'Name', 'input', ...
                'Normalization', 'none')

            % 128x128xC -> 64x64x64
            convolution2dLayer(4, 64, 'Name', 'conv1', ...
                'Stride', 2, 'Padding', 'same')
            leakyReluLayer(0.2, 'Name', 'lrelu1')
            dropoutLayer(0.3, 'Name', 'dropout1')

            % 64x64x64 -> 32x32x128
            convolution2dLayer(4, 128, 'Name', 'conv2', ...
                'Stride', 2, 'Padding', 'same')
            batchNormalizationLayer('Name', 'bn2')
            leakyReluLayer(0.2, 'Name', 'lrelu2')
            dropoutLayer(0.3, 'Name', 'dropout2')

            % 32x32x128 -> 16x16x256
            convolution2dLayer(4, 256, 'Name', 'conv3', ...
                'Stride', 2, 'Padding', 'same')
            batchNormalizationLayer('Name', 'bn3')
            leakyReluLayer(0.2, 'Name', 'lrelu3')
            dropoutLayer(0.3, 'Name', 'dropout3')

            % 16x16x256 -> 8x8x512
            convolution2dLayer(4, 512, 'Name', 'conv4', ...
                'Stride', 2, 'Padding', 'same')
            batchNormalizationLayer('Name', 'bn4')
            leakyReluLayer(0.2, 'Name', 'lrelu4')
            dropoutLayer(0.3, 'Name', 'dropout4')

            % 8x8x512 -> 4x4x1024
            convolution2dLayer(4, 1024, 'Name', 'conv5', ...
                'Stride', 2, 'Padding', 'same')
            batchNormalizationLayer('Name', 'bn5')
            leakyReluLayer(0.2, 'Name', 'lrelu5')
            dropoutLayer(0.3, 'Name', 'dropout5')

            % 4x4x1024 -> 1x1x1
            convolution2dLayer(4, 1, 'Name', 'conv6', ...
                'Stride', 1, 'Padding', 0)
            sigmoidLayer('Name', 'sigmoid')
        ];

    else
        error('Unsupported imageSize: %d. Use 64 or 128.', imageSize);
    end

    %% Create dlnetwork
    lgraph = layerGraph(layers);
    netD = dlnetwork(lgraph);

    % Initialize weights
    netD = initialize(netD);
end
