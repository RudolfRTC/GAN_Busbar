function netG = buildGenerator(params)
%BUILDGENERATOR Build DCGAN Generator network
%
% Architecture: Transposed Convolution layers with BatchNorm and ReLU
% Input: latent vector [1 x 1 x latentDim]
% Output: image [imageSize x imageSize x numChannels], tanh activation
%
% Inputs:
%   params - struct with:
%     .latentDim - latent vector dimension
%     .imageSize - output image size (64 or 128)
%     .numChannels - number of output channels (1 or 3)
%
% Output:
%   netG - dlnetwork generator

    latentDim = params.latentDim;
    imageSize = params.imageSize;
    numChannels = params.numChannels;

    layers = [];

    %% Architecture depends on image size
    if imageSize == 64
        % ===== 64x64 Generator =====
        % Input: 1x1x100
        % Project and reshape: 4x4x512
        filterSize = 4;
        numFilters = 512;

        layers = [
            featureInputLayer(latentDim, 'Name', 'input', 'Normalization', 'none')

            % 1x1x100 -> 4x4x512
            transposedConv2dLayer(filterSize, numFilters, 'Name', 'tconv1', ...
                'Stride', 1, 'Cropping', 'same')
            batchNormalizationLayer('Name', 'bn1')
            reluLayer('Name', 'relu1')

            % 4x4x512 -> 8x8x256
            transposedConv2dLayer(4, 256, 'Name', 'tconv2', ...
                'Stride', 2, 'Cropping', 'same')
            batchNormalizationLayer('Name', 'bn2')
            reluLayer('Name', 'relu2')

            % 8x8x256 -> 16x16x128
            transposedConv2dLayer(4, 128, 'Name', 'tconv3', ...
                'Stride', 2, 'Cropping', 'same')
            batchNormalizationLayer('Name', 'bn3')
            reluLayer('Name', 'relu3')

            % 16x16x128 -> 32x32x64
            transposedConv2dLayer(4, 64, 'Name', 'tconv4', ...
                'Stride', 2, 'Cropping', 'same')
            batchNormalizationLayer('Name', 'bn4')
            reluLayer('Name', 'relu4')

            % 32x32x64 -> 64x64xnumChannels
            transposedConv2dLayer(4, numChannels, 'Name', 'tconv5', ...
                'Stride', 2, 'Cropping', 'same')
            tanhLayer('Name', 'tanh')
        ];

    elseif imageSize == 128
        % ===== 128x128 Generator =====
        % Input: 1x1x100
        % Project and reshape: 4x4x1024
        filterSize = 4;
        numFilters = 1024;

        layers = [
            featureInputLayer(latentDim, 'Name', 'input', 'Normalization', 'none')

            % 1x1x100 -> 4x4x1024
            transposedConv2dLayer(filterSize, numFilters, 'Name', 'tconv1', ...
                'Stride', 1, 'Cropping', 'same')
            batchNormalizationLayer('Name', 'bn1')
            reluLayer('Name', 'relu1')

            % 4x4x1024 -> 8x8x512
            transposedConv2dLayer(4, 512, 'Name', 'tconv2', ...
                'Stride', 2, 'Cropping', 'same')
            batchNormalizationLayer('Name', 'bn2')
            reluLayer('Name', 'relu2')

            % 8x8x512 -> 16x16x256
            transposedConv2dLayer(4, 256, 'Name', 'tconv3', ...
                'Stride', 2, 'Cropping', 'same')
            batchNormalizationLayer('Name', 'bn3')
            reluLayer('Name', 'relu3')

            % 16x16x256 -> 32x32x128
            transposedConv2dLayer(4, 128, 'Name', 'tconv4', ...
                'Stride', 2, 'Cropping', 'same')
            batchNormalizationLayer('Name', 'bn4')
            reluLayer('Name', 'relu4')

            % 32x32x128 -> 64x64x64
            transposedConv2dLayer(4, 64, 'Name', 'tconv5', ...
                'Stride', 2, 'Cropping', 'same')
            batchNormalizationLayer('Name', 'bn5')
            reluLayer('Name', 'relu5')

            % 64x64x64 -> 128x128xnumChannels
            transposedConv2dLayer(4, numChannels, 'Name', 'tconv6', ...
                'Stride', 2, 'Cropping', 'same')
            tanhLayer('Name', 'tanh')
        ];

    else
        error('Unsupported imageSize: %d. Use 64 or 128.', imageSize);
    end

    %% Create dlnetwork
    lgraph = layerGraph(layers);
    netG = dlnetwork(lgraph);

    % Initialize weights (Xavier initialization for better convergence)
    netG = initialize(netG);
end
