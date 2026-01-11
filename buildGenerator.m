function netG = buildGenerator(params)
%BUILDGENERATOR Build DCGAN Generator network
%
% Architecture: Transposed Convolution layers with BatchNorm and ReLU
% Input: latent vector [1 x 1 x latentDim] (using imageInputLayer)
% Output: image [trainSize(1) x trainSize(2) x numChannels], tanh activation
%
% Inputs:
%   params - struct with:
%     .latentDim - latent vector dimension (e.g., 100)
%     .trainSize - [height width] for training (e.g., [64 128])
%     .numChannels - number of output channels (1 or 3)
%
% Output:
%   netG - dlnetwork generator

    latentDim = params.latentDim;
    trainHeight = params.trainSize(1);
    trainWidth = params.trainSize(2);
    numChannels = params.numChannels;

    %% ===== ARCHITECTURE SELECTION BASED ON trainSize =====
    % Support common sizes: [64 128], [128 256], [64 64], [128 128], [256 256]

    % Use imageInputLayer to ensure proper input shape [1 x 1 x latentDim x N]
    layers = imageInputLayer([1 1 latentDim], 'Name', 'input', 'Normalization', 'none');

    %% Build architecture based on target size
    if trainHeight == 64 && trainWidth == 128
        % ===== 64x128 Generator (aspect ratio 2:1) =====
        % 1x1x100 -> 2x4x1024 -> 4x8x512 -> 8x16x256 -> 16x32x128 -> 32x64x64 -> 64x128xC

        layers = [layers
            % 1x1 -> 2x4
            transposedConv2dLayer([2 4], 1024, 'Name', 'tconv1', 'Stride', 1, 'Cropping', 'same')
            batchNormalizationLayer('Name', 'bn1')
            reluLayer('Name', 'relu1')

            % 2x4 -> 4x8
            transposedConv2dLayer(4, 512, 'Name', 'tconv2', 'Stride', 2, 'Cropping', 'same')
            batchNormalizationLayer('Name', 'bn2')
            reluLayer('Name', 'relu2')

            % 4x8 -> 8x16
            transposedConv2dLayer(4, 256, 'Name', 'tconv3', 'Stride', 2, 'Cropping', 'same')
            batchNormalizationLayer('Name', 'bn3')
            reluLayer('Name', 'relu3')

            % 8x16 -> 16x32
            transposedConv2dLayer(4, 128, 'Name', 'tconv4', 'Stride', 2, 'Cropping', 'same')
            batchNormalizationLayer('Name', 'bn4')
            reluLayer('Name', 'relu4')

            % 16x32 -> 32x64
            transposedConv2dLayer(4, 64, 'Name', 'tconv5', 'Stride', 2, 'Cropping', 'same')
            batchNormalizationLayer('Name', 'bn5')
            reluLayer('Name', 'relu5')

            % 32x64 -> 64x128
            transposedConv2dLayer(4, numChannels, 'Name', 'tconv6', 'Stride', 2, 'Cropping', 'same')
            tanhLayer('Name', 'tanh')
        ];

    elseif trainHeight == 128 && trainWidth == 256
        % ===== 128x256 Generator (aspect ratio 2:1) =====
        layers = [layers
            % 1x1 -> 4x8
            transposedConv2dLayer([4 8], 1024, 'Name', 'tconv1', 'Stride', 1, 'Cropping', 'same')
            batchNormalizationLayer('Name', 'bn1')
            reluLayer('Name', 'relu1')

            % 4x8 -> 8x16
            transposedConv2dLayer(4, 512, 'Name', 'tconv2', 'Stride', 2, 'Cropping', 'same')
            batchNormalizationLayer('Name', 'bn2')
            reluLayer('Name', 'relu2')

            % 8x16 -> 16x32
            transposedConv2dLayer(4, 512, 'Name', 'tconv3', 'Stride', 2, 'Cropping', 'same')
            batchNormalizationLayer('Name', 'bn3')
            reluLayer('Name', 'relu3')

            % 16x32 -> 32x64
            transposedConv2dLayer(4, 256, 'Name', 'tconv4', 'Stride', 2, 'Cropping', 'same')
            batchNormalizationLayer('Name', 'bn4')
            reluLayer('Name', 'relu4')

            % 32x64 -> 64x128
            transposedConv2dLayer(4, 128, 'Name', 'tconv5', 'Stride', 2, 'Cropping', 'same')
            batchNormalizationLayer('Name', 'bn5')
            reluLayer('Name', 'relu5')

            % 64x128 -> 128x256
            transposedConv2dLayer(4, numChannels, 'Name', 'tconv6', 'Stride', 2, 'Cropping', 'same')
            tanhLayer('Name', 'tanh')
        ];

    elseif trainHeight == 64 && trainWidth == 64
        % ===== 64x64 Generator (square) =====
        layers = [layers
            % 1x1 -> 4x4
            transposedConv2dLayer(4, 512, 'Name', 'tconv1', 'Stride', 1, 'Cropping', 'same')
            batchNormalizationLayer('Name', 'bn1')
            reluLayer('Name', 'relu1')

            % 4x4 -> 8x8
            transposedConv2dLayer(4, 256, 'Name', 'tconv2', 'Stride', 2, 'Cropping', 'same')
            batchNormalizationLayer('Name', 'bn2')
            reluLayer('Name', 'relu2')

            % 8x8 -> 16x16
            transposedConv2dLayer(4, 128, 'Name', 'tconv3', 'Stride', 2, 'Cropping', 'same')
            batchNormalizationLayer('Name', 'bn3')
            reluLayer('Name', 'relu3')

            % 16x16 -> 32x32
            transposedConv2dLayer(4, 64, 'Name', 'tconv4', 'Stride', 2, 'Cropping', 'same')
            batchNormalizationLayer('Name', 'bn4')
            reluLayer('Name', 'relu4')

            % 32x32 -> 64x64
            transposedConv2dLayer(4, numChannels, 'Name', 'tconv5', 'Stride', 2, 'Cropping', 'same')
            tanhLayer('Name', 'tanh')
        ];

    elseif trainHeight == 128 && trainWidth == 128
        % ===== 128x128 Generator (square) =====
        layers = [layers
            % 1x1 -> 4x4
            transposedConv2dLayer(4, 1024, 'Name', 'tconv1', 'Stride', 1, 'Cropping', 'same')
            batchNormalizationLayer('Name', 'bn1')
            reluLayer('Name', 'relu1')

            % 4x4 -> 8x8
            transposedConv2dLayer(4, 512, 'Name', 'tconv2', 'Stride', 2, 'Cropping', 'same')
            batchNormalizationLayer('Name', 'bn2')
            reluLayer('Name', 'relu2')

            % 8x8 -> 16x16
            transposedConv2dLayer(4, 256, 'Name', 'tconv3', 'Stride', 2, 'Cropping', 'same')
            batchNormalizationLayer('Name', 'bn3')
            reluLayer('Name', 'relu3')

            % 16x16 -> 32x32
            transposedConv2dLayer(4, 128, 'Name', 'tconv4', 'Stride', 2, 'Cropping', 'same')
            batchNormalizationLayer('Name', 'bn4')
            reluLayer('Name', 'relu4')

            % 32x32 -> 64x64
            transposedConv2dLayer(4, 64, 'Name', 'tconv5', 'Stride', 2, 'Cropping', 'same')
            batchNormalizationLayer('Name', 'bn5')
            reluLayer('Name', 'relu5')

            % 64x64 -> 128x128
            transposedConv2dLayer(4, numChannels, 'Name', 'tconv6', 'Stride', 2, 'Cropping', 'same')
            tanhLayer('Name', 'tanh')
        ];

    elseif trainHeight == 256 && trainWidth == 256
        % ===== 256x256 Generator (square) =====
        layers = [layers
            % 1x1 -> 4x4
            transposedConv2dLayer(4, 1024, 'Name', 'tconv1', 'Stride', 1, 'Cropping', 'same')
            batchNormalizationLayer('Name', 'bn1')
            reluLayer('Name', 'relu1')

            % 4x4 -> 8x8
            transposedConv2dLayer(4, 1024, 'Name', 'tconv2', 'Stride', 2, 'Cropping', 'same')
            batchNormalizationLayer('Name', 'bn2')
            reluLayer('Name', 'relu2')

            % 8x8 -> 16x16
            transposedConv2dLayer(4, 512, 'Name', 'tconv3', 'Stride', 2, 'Cropping', 'same')
            batchNormalizationLayer('Name', 'bn3')
            reluLayer('Name', 'relu3')

            % 16x16 -> 32x32
            transposedConv2dLayer(4, 256, 'Name', 'tconv4', 'Stride', 2, 'Cropping', 'same')
            batchNormalizationLayer('Name', 'bn4')
            reluLayer('Name', 'relu4')

            % 32x32 -> 64x64
            transposedConv2dLayer(4, 128, 'Name', 'tconv5', 'Stride', 2, 'Cropping', 'same')
            batchNormalizationLayer('Name', 'bn5')
            reluLayer('Name', 'relu5')

            % 64x64 -> 128x128
            transposedConv2dLayer(4, 64, 'Name', 'tconv6', 'Stride', 2, 'Cropping', 'same')
            batchNormalizationLayer('Name', 'bn6')
            reluLayer('Name', 'relu6')

            % 128x128 -> 256x256
            transposedConv2dLayer(4, numChannels, 'Name', 'tconv7', 'Stride', 2, 'Cropping', 'same')
            tanhLayer('Name', 'tanh')
        ];

    else
        error(['Unsupported trainSize: [%d %d]. Supported sizes:\n' ...
               '  [64 128]   - aspect ratio 2:1 (recommended for 4200x2128)\n' ...
               '  [128 256]  - aspect ratio 2:1 (higher quality)\n' ...
               '  [64 64]    - square (memory efficient)\n' ...
               '  [128 128]  - square (balanced)\n' ...
               '  [256 256]  - square (high quality, requires more memory)'], ...
               trainHeight, trainWidth);
    end

    %% Create dlnetwork
    lgraph = layerGraph(layers);
    netG = dlnetwork(lgraph);

    % Initialize weights (Xavier initialization for better convergence)
    netG = initialize(netG);

    fprintf('  Generator built successfully:\n');
    fprintf('    Input shape:  [1 x 1 x %d x N]\n', latentDim);
    fprintf('    Output shape: [%d x %d x %d x N]\n', trainHeight, trainWidth, numChannels);
end
