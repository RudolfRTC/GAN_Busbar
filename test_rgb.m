%% Test if your busbar images are truly RGB or fake RGB (grayscale)
% Run this in MATLAB on your computer:
%
% cd 'G:\Fax\Prijava Teme\Clanek za busbare\Code'
% test_rgb

function test_rgb()
    % Check if images folder exists
    imgFolder = './data/images';
    if ~exist(imgFolder, 'dir')
        fprintf('ERROR: Folder %s does not exist!\n', imgFolder);
        fprintf('Update imgFolder path in this script.\n');
        return;
    end

    % Find all images
    imageFiles = dir(fullfile(imgFolder, '*.jpg'));
    imageFiles = [imageFiles; dir(fullfile(imgFolder, '*.png'))];
    imageFiles = [imageFiles; dir(fullfile(imgFolder, '*.jpeg'))];
    imageFiles = [imageFiles; dir(fullfile(imgFolder, '*.bmp'))];

    if isempty(imageFiles)
        fprintf('ERROR: No images found in %s\n', imgFolder);
        return;
    end

    fprintf('========================================\n');
    fprintf('Testing %d images for RGB color\n', numel(imageFiles));
    fprintf('========================================\n\n');

    numTrueRGB = 0;
    numFakeRGB = 0;
    numGrayscale = 0;

    % Test first 10 images
    numTest = min(10, numel(imageFiles));

    for i = 1:numTest
        imgPath = fullfile(imageFiles(i).folder, imageFiles(i).name);
        fprintf('[%d/%d] %s\n', i, numTest, imageFiles(i).name);

        try
            img = imread(imgPath);

            % Check dimensions
            [h, w, c] = size(img);
            fprintf('  Size: %dx%d, Channels: %d\n', w, h, c);

            if c == 1
                % Single channel = grayscale
                fprintf('  ❌ GRAYSCALE (1 channel)\n\n');
                numGrayscale = numGrayscale + 1;
                continue;
            end

            % For 3-channel images, check if truly RGB or fake RGB
            R = double(img(:,:,1));
            G = double(img(:,:,2));
            B = double(img(:,:,3));

            % Sample 1000 random pixels
            numPixels = h * w;
            sampleSize = min(1000, numPixels);
            idx = randperm(numPixels, sampleSize);

            R_sample = R(idx);
            G_sample = G(idx);
            B_sample = B(idx);

            % Calculate statistics
            R_mean = mean(R_sample);
            G_mean = mean(G_sample);
            B_mean = mean(B_sample);

            % Calculate channel differences
            RG_diff = R_sample - G_sample;
            GB_diff = G_sample - B_sample;
            RB_diff = R_sample - B_sample;

            % Standard deviation of differences
            std_RG = std(RG_diff);
            std_GB = std(GB_diff);
            std_RB = std(RB_diff);

            % Mean absolute difference
            mad_RG = mean(abs(RG_diff));
            mad_GB = mean(abs(GB_diff));
            mad_RB = mean(abs(RB_diff));

            fprintf('  Channel means: R=%.1f, G=%.1f, B=%.1f\n', R_mean, G_mean, B_mean);
            fprintf('  Channel diffs (std): RG=%.2f, GB=%.2f, RB=%.2f\n', std_RG, std_GB, std_RB);
            fprintf('  Channel diffs (MAD): RG=%.2f, GB=%.2f, RB=%.2f\n', mad_RG, mad_GB, mad_RB);

            % Threshold for "fake RGB" detection
            % If all channel differences are very small, it's grayscale
            THRESHOLD = 5.0;  % MAD threshold

            if mad_RG < THRESHOLD && mad_GB < THRESHOLD && mad_RB < THRESHOLD
                fprintf('  ❌ FAKE RGB - Actually grayscale! (R≈G≈B)\n');
                fprintf('     All color channels are nearly identical.\n\n');
                numFakeRGB = numFakeRGB + 1;
            else
                fprintf('  ✅ TRUE RGB - Colored image!\n');
                if R_mean > G_mean && G_mean > B_mean
                    fprintf('     Color: Warm tones (R>G>B) - Copper/brass/gold\n');
                elseif B_mean > G_mean && G_mean > R_mean
                    fprintf('     Color: Cool tones (B>G>R)\n');
                else
                    fprintf('     Color: Mixed/neutral tones\n');
                end
                fprintf('\n');
                numTrueRGB = numTrueRGB + 1;
            end

        catch ME
            fprintf('  ERROR: %s\n\n', ME.message);
        end
    end

    % Summary
    fprintf('========================================\n');
    fprintf('SUMMARY\n');
    fprintf('========================================\n');
    fprintf('True RGB (colored):    %d / %d\n', numTrueRGB, numTest);
    fprintf('Fake RGB (grayscale):  %d / %d\n', numFakeRGB, numTest);
    fprintf('Grayscale (1-channel): %d / %d\n', numGrayscale, numTest);
    fprintf('\n');

    if numFakeRGB + numGrayscale == numTest
        fprintf('🔴 PROBLEM: All images are GRAYSCALE!\n');
        fprintf('\n');
        fprintf('Your images look like they have color, but they are actually\n');
        fprintf('grayscale images saved in RGB format (R=G=B).\n');
        fprintf('\n');
        fprintf('SOLUTION: Colorize your images!\n');
        fprintf('\n');
        fprintf('Options:\n');
        fprintf('1. DeOldify (AI colorization):\n');
        fprintf('   https://github.com/jantic/DeOldify\n');
        fprintf('\n');
        fprintf('2. Online tools:\n');
        fprintf('   - https://imagecolorizer.com/\n');
        fprintf('   - https://palette.fm/\n');
        fprintf('   - https://hotpot.ai/colorize-picture\n');
        fprintf('\n');
        fprintf('3. Manual colorization:\n');
        fprintf('   - Photoshop / GIMP\n');
        fprintf('   - Add color layers in Overlay mode\n');
        fprintf('\n');
        fprintf('Once colorized, GAN will generate COLORED outputs!\n');
    elseif numTrueRGB > 0
        fprintf('✅ GOOD: Found %d RGB colored images!\n', numTrueRGB);
        fprintf('\n');
        fprintf('Your images contain true color information.\n');
        fprintf('GAN should generate COLORED outputs.\n');
        fprintf('\n');
        fprintf('If GAN still generates gray images:\n');
        fprintf('1. Make sure you are using the latest code (commit 7af05fa)\n');
        fprintf('2. Check train_gan.m line 42: params.numChannels = 3\n');
        fprintf('3. Delete old models if they exist\n');
        fprintf('4. Retrain from scratch\n');
    end
end
