%% TEST_SETUP - Verify MATLAB environment and dependencies
%
% Run this script to check if your environment is ready for GAN training
%
% Usage: test_setup

clear; clc;

fprintf('==============================================\n');
fprintf('  GAN Setup Verification\n');
fprintf('==============================================\n\n');

allGood = true;

%% Check MATLAB Version
fprintf('1. Checking MATLAB version...\n');
matlabVersion = version('-release');
fprintf('   MATLAB version: %s\n', matlabVersion);

% Extract year
year = str2double(matlabVersion(1:4));
if year >= 2020
    fprintf('   ✓ MATLAB version OK (R2020b+ recommended)\n');
else
    fprintf('   ✗ WARNING: MATLAB version may be too old (R2020b+ recommended)\n');
    allGood = false;
end
fprintf('\n');

%% Check Deep Learning Toolbox
fprintf('2. Checking Deep Learning Toolbox...\n');
try
    v = ver('deeplearning');
    if ~isempty(v)
        fprintf('   ✓ Deep Learning Toolbox installed: %s\n', v.Version);
    else
        fprintf('   ✗ ERROR: Deep Learning Toolbox NOT found\n');
        fprintf('      Install via: MATLAB Add-Ons > Deep Learning Toolbox\n');
        allGood = false;
    end
catch
    fprintf('   ✗ ERROR: Deep Learning Toolbox NOT found\n');
    allGood = false;
end
fprintf('\n');

%% Check GPU Support
fprintf('3. Checking GPU support...\n');
try
    gpu = gpuDevice;
    fprintf('   ✓ GPU detected: %s\n', gpu.Name);
    fprintf('     GPU Memory: %.2f GB\n', gpu.AvailableMemory / 1e9);
    fprintf('     Compute Capability: %.1f\n', gpu.ComputeCapability);

    if gpu.ComputeCapability < 3.0
        fprintf('   ✗ WARNING: GPU compute capability < 3.0 (may not work)\n');
        allGood = false;
    end
catch ME
    fprintf('   ⚠ No GPU detected (will use CPU - slower but works)\n');
    fprintf('     Message: %s\n', ME.message);
end
fprintf('\n');

%% Check Data Folder
fprintf('4. Checking data folder...\n');
dataFolder = './data/images';
if exist(dataFolder, 'dir')
    fprintf('   ✓ Data folder exists: %s\n', dataFolder);

    % Count images
    imageFiles = [
        dir(fullfile(dataFolder, '*.jpg'));
        dir(fullfile(dataFolder, '*.jpeg'));
        dir(fullfile(dataFolder, '*.png'));
        dir(fullfile(dataFolder, '*.bmp'))
    ];

    numImages = numel(imageFiles);
    fprintf('     Found %d images\n', numImages);

    if numImages == 0
        fprintf('   ⚠ WARNING: No images found in data folder\n');
        fprintf('     Add your images (.jpg, .png, .bmp) to: %s\n', dataFolder);
        allGood = false;
    elseif numImages < 30
        fprintf('   ⚠ WARNING: Very few images (%d)\n', numImages);
        fprintf('     Recommended: 50+ images for good results\n');
    elseif numImages < 100
        fprintf('   ✓ OK: %d images (consider adding more for better results)\n', numImages);
    else
        fprintf('   ✓ Great: %d images\n', numImages);
    end
else
    fprintf('   ✗ ERROR: Data folder not found: %s\n', dataFolder);
    fprintf('     Creating folder...\n');
    mkdir(dataFolder);
    fprintf('     Please add your images to: %s\n', dataFolder);
    allGood = false;
end
fprintf('\n');

%% Check Required Functions
fprintf('5. Checking required files...\n');
requiredFiles = {
    'train_gan.m',
    'buildGenerator.m',
    'buildDiscriminator.m',
    'preprocessAndLoadDatastore.m',
    'saveImageGrid.m',
    'generateSynthetic.m'
};

allFilesExist = true;
for i = 1:length(requiredFiles)
    if exist(requiredFiles{i}, 'file')
        fprintf('   ✓ %s\n', requiredFiles{i});
    else
        fprintf('   ✗ MISSING: %s\n', requiredFiles{i});
        allFilesExist = false;
        allGood = false;
    end
end

if allFilesExist
    fprintf('   ✓ All required files found\n');
end
fprintf('\n');

%% Check Output Folders
fprintf('6. Checking/creating output folders...\n');
outputFolders = {
    './outputs',
    './outputs/preview',
    './outputs/models',
    './outputs/synthetic'
};

for i = 1:length(outputFolders)
    if ~exist(outputFolders{i}, 'dir')
        mkdir(outputFolders{i});
        fprintf('   ✓ Created: %s\n', outputFolders{i});
    else
        fprintf('   ✓ Exists: %s\n', outputFolders{i});
    end
end
fprintf('\n');

%% Test Basic Operations
fprintf('7. Testing basic operations...\n');
try
    % Test dlarray
    testArray = dlarray(randn(5, 5, 3, 2), 'SSCB');
    fprintf('   ✓ dlarray works\n');

    % Test layer creation
    testLayer = convolution2dLayer(3, 64);
    fprintf('   ✓ Layer creation works\n');

    fprintf('   ✓ Basic operations OK\n');
catch ME
    fprintf('   ✗ ERROR in basic operations: %s\n', ME.message);
    allGood = false;
end
fprintf('\n');

%% Summary
fprintf('==============================================\n');
if allGood
    fprintf('  ✓ ALL CHECKS PASSED!\n');
    fprintf('  You are ready to run: train_gan\n');
else
    fprintf('  ⚠ SOME ISSUES DETECTED\n');
    fprintf('  Please fix the issues above before training\n');
end
fprintf('==============================================\n\n');

%% Recommendations
fprintf('Recommendations:\n');
fprintf('  - Place 50-100 images in ./data/images/\n');
fprintf('  - Images should be on white/light background\n');
fprintf('  - Supported formats: .jpg, .png, .bmp\n');
fprintf('  - GPU recommended but not required\n');
fprintf('  - Free disk space: ~500MB for outputs\n');
fprintf('\n');

fprintf('Next steps:\n');
fprintf('  1. Add images to ./data/images/\n');
fprintf('  2. Open train_gan.m and adjust parameters if needed\n');
fprintf('  3. Run: train_gan\n');
fprintf('  4. Wait for training (2-20 hours depending on GPU/CPU)\n');
fprintf('  5. Check results in ./outputs/\n');
fprintf('\n');
