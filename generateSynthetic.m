function generateSynthetic(netG, params)
%GENERATESYNTHETIC Generate synthetic images using trained generator
%
% Inputs:
%   netG - trained generator network
%   params - struct with:
%     .numSynthetic - number of images to generate
%     .latentDim - latent vector dimension
%     .outputFolder - base output folder
%     .executionEnvironment - 'auto', 'gpu', or 'cpu'
%     .numChannels - 1 or 3
%
% Output:
%   Saves synthetic images to ./outputs/synthetic/

    numSynthetic = params.numSynthetic;
    latentDim = params.latentDim;
    outputFolder = fullfile(params.outputFolder, 'synthetic');

    % Create output folder if needed
    if ~exist(outputFolder, 'dir')
        mkdir(outputFolder);
    end

    fprintf('Generating %d synthetic images...\n', numSynthetic);

    % Generate in batches to avoid memory issues
    batchSize = 64; % Generate 64 images at a time
    numBatches = ceil(numSynthetic / batchSize);

    imageCounter = 1;

    for batchIdx = 1:numBatches
        % Determine batch size for this iteration
        if batchIdx == numBatches
            currentBatchSize = numSynthetic - (batchIdx - 1) * batchSize;
        else
            currentBatchSize = batchSize;
        end

        % Generate random latent vectors
        Z = randn(1, 1, latentDim, currentBatchSize, 'single');
        Z = dlarray(Z, 'SSCB');

        % Move to GPU if available
        if strcmp(params.executionEnvironment, 'auto') || strcmp(params.executionEnvironment, 'gpu')
            if canUseGPU
                Z = gpuArray(Z);
            end
        end

        % Generate images
        XGenerated = predict(netG, Z);
        XGenerated = extractdata(gather(XGenerated));

        % Denormalize from [-1, 1] to [0, 1]
        XGenerated = (XGenerated + 1) / 2;
        XGenerated = max(min(XGenerated, 1), 0); % Clamp

        % Save each image
        for i = 1:currentBatchSize
            img = XGenerated(:, :, :, i);

            % Convert to uint8
            img = uint8(img * 255);

            % Generate filename with zero-padding
            filename = sprintf('synthetic_%06d.png', imageCounter);
            filepath = fullfile(outputFolder, filename);

            % Save image
            imwrite(img, filepath);

            imageCounter = imageCounter + 1;
        end

        % Progress update
        if mod(batchIdx, 10) == 0 || batchIdx == numBatches
            fprintf('  Progress: %d/%d images generated\n', imageCounter - 1, numSynthetic);
        end
    end

    fprintf('Done! Synthetic images saved to: %s\n', outputFolder);
end
