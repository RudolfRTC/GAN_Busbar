#!/usr/bin/env python3
"""
Test GAN Project Structure

This Python script validates the GAN project structure without requiring MATLAB.
It checks:
- File existence and completeness
- Code structure and syntax (basic)
- Directory structure
- Documentation
"""

import os
import re
import sys
from pathlib import Path

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_header(text):
    print(f"\n{BOLD}{BLUE}{'='*60}{RESET}")
    print(f"{BOLD}{BLUE}{text:^60}{RESET}")
    print(f"{BOLD}{BLUE}{'='*60}{RESET}\n")

def print_test(test_num, total, description):
    print(f"{BOLD}[Test {test_num}/{total}]{RESET} {description}")

def print_success(message):
    print(f"  {GREEN}✓{RESET} {message}")

def print_error(message):
    print(f"  {RED}✗{RESET} {message}")

def print_warning(message):
    print(f"  {YELLOW}⚠{RESET} {message}")

class GANStructureTest:
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.total_tests = 10
        self.project_root = Path(__file__).parent

    def test_required_files(self):
        """Test 1: Check all required MATLAB files exist"""
        print_test(1, self.total_tests, "Checking required MATLAB files...")

        required_files = {
            'GAN.m': 'Main entry point',
            'train_gan.m': 'Training script',
            'buildGenerator.m': 'Generator network',
            'buildDiscriminator.m': 'Discriminator network',
            'preprocessAndLoadDatastore.m': 'Data preprocessing',
            'saveImageGrid.m': 'Preview saving',
            'generateSynthetic.m': 'Synthetic generation',
            'test_setup.m': 'Environment test',
            'test_gan_full.m': 'Full integration test'
        }

        all_exist = True
        for filename, description in required_files.items():
            filepath = self.project_root / filename
            if filepath.exists():
                size = filepath.stat().st_size
                if size > 0:
                    print_success(f"{filename:40s} ({size:,} bytes) - {description}")
                else:
                    print_error(f"{filename:40s} is EMPTY - {description}")
                    all_exist = False
            else:
                print_error(f"{filename:40s} MISSING - {description}")
                all_exist = False

        if all_exist:
            self.tests_passed += 1
        else:
            self.tests_failed += 1
        print()

    def test_directory_structure(self):
        """Test 2: Check directory structure"""
        print_test(2, self.total_tests, "Checking directory structure...")

        required_dirs = {
            'data': 'Training data folder',
            'data/images': 'Training images',
            'outputs': 'Output folder',
            'outputs/preview': 'Preview images',
            'outputs/models': 'Trained models',
            'outputs/synthetic': 'Synthetic images'
        }

        all_exist = True
        for dirname, description in required_dirs.items():
            dirpath = self.project_root / dirname
            if dirpath.exists() and dirpath.is_dir():
                print_success(f"{dirname:30s} - {description}")
            else:
                print_warning(f"{dirname:30s} MISSING (will be created) - {description}")

        # This test always passes (directories can be created)
        self.tests_passed += 1
        print()

    def test_file_completeness(self):
        """Test 3: Check file completeness (line count, basic syntax)"""
        print_test(3, self.total_tests, "Checking file completeness...")

        file_min_lines = {
            'GAN.m': 100,
            'train_gan.m': 200,
            'buildGenerator.m': 80,
            'buildDiscriminator.m': 100,
            'preprocessAndLoadDatastore.m': 150,
            'generateSynthetic.m': 50,
            'saveImageGrid.m': 40,
            'test_gan_full.m': 350
        }

        all_complete = True
        for filename, min_lines in file_min_lines.items():
            filepath = self.project_root / filename
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    line_count = len([l for l in lines if l.strip()])  # Non-empty lines

                if line_count >= min_lines:
                    print_success(f"{filename:40s} {line_count:4d} lines (>= {min_lines})")
                else:
                    print_error(f"{filename:40s} {line_count:4d} lines (< {min_lines} expected)")
                    all_complete = False
            else:
                print_error(f"{filename:40s} MISSING")
                all_complete = False

        if all_complete:
            self.tests_passed += 1
        else:
            self.tests_failed += 1
        print()

    def test_function_definitions(self):
        """Test 4: Check function definitions exist"""
        print_test(4, self.total_tests, "Checking function definitions...")

        functions_to_check = {
            'buildGenerator.m': r'function\s+netG\s*=\s*buildGenerator',
            'buildDiscriminator.m': r'function\s+netD\s*=\s*buildDiscriminator',
            'preprocessAndLoadDatastore.m': r'function\s+\[mbq,\s*params\]\s*=\s*preprocessAndLoadDatastore',
            'saveImageGrid.m': r'function\s+saveImageGrid',
            'generateSynthetic.m': r'function\s+generateSynthetic'
        }

        all_defined = True
        for filename, pattern in functions_to_check.items():
            filepath = self.project_root / filename
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                if re.search(pattern, content):
                    print_success(f"{filename:40s} function defined")
                else:
                    print_error(f"{filename:40s} function NOT defined")
                    all_defined = False
            else:
                print_error(f"{filename:40s} MISSING")
                all_defined = False

        if all_defined:
            self.tests_passed += 1
        else:
            self.tests_failed += 1
        print()

    def test_documentation(self):
        """Test 5: Check documentation files"""
        print_test(5, self.total_tests, "Checking documentation...")

        doc_files = {
            'README.md': 'Main documentation',
            'LARGE_IMAGES_GUIDE.md': 'Large images guide',
            'RETRAIN_INSTRUCTIONS.md': 'Retraining instructions'
        }

        has_docs = True
        for filename, description in doc_files.items():
            filepath = self.project_root / filename
            if filepath.exists():
                size = filepath.stat().st_size
                print_success(f"{filename:40s} ({size:,} bytes) - {description}")
            else:
                print_warning(f"{filename:40s} MISSING - {description}")

        # Documentation is optional, so always pass
        self.tests_passed += 1
        print()

    def test_gan_main_structure(self):
        """Test 6: Check GAN.m structure"""
        print_test(6, self.total_tests, "Checking GAN.m main script structure...")

        filepath = self.project_root / 'GAN.m'
        if not filepath.exists():
            print_error("GAN.m not found")
            self.tests_failed += 1
            print()
            return

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        checks = {
            'Pre-flight checks': r'pre-flight|Pre-flight',
            'MATLAB version check': r'version.*release|matlabVersion',
            'Deep Learning Toolbox check': r'deeplearning|Deep Learning',
            'GPU check': r'gpuDevice|GPU',
            'Training images check': r'imageFiles|data.*images',
            'Calls train_gan': r'train_gan',
        }

        all_checks_passed = True
        for check_name, pattern in checks.items():
            if re.search(pattern, content, re.IGNORECASE):
                print_success(f"{check_name:40s} found")
            else:
                print_error(f"{check_name:40s} NOT found")
                all_checks_passed = False

        if all_checks_passed:
            self.tests_passed += 1
        else:
            self.tests_failed += 1
        print()

    def test_train_gan_structure(self):
        """Test 7: Check train_gan.m structure"""
        print_test(7, self.total_tests, "Checking train_gan.m structure...")

        filepath = self.project_root / 'train_gan.m'
        if not filepath.exists():
            print_error("train_gan.m not found")
            self.tests_failed += 1
            print()
            return

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        checks = {
            'Parameters section': r'params\s*=\s*struct',
            'Load data': r'preprocessAndLoadDatastore',
            'Build generator': r'buildGenerator',
            'Build discriminator': r'buildDiscriminator',
            'Training loop': r'for\s+epoch\s*=',
            'Discriminator training': r'modelGradientsD',
            'Generator training': r'modelGradientsG',
            'Save models': r'save.*generator',
            'Generate synthetic': r'generateSynthetic'
        }

        all_checks_passed = True
        for check_name, pattern in checks.items():
            if re.search(pattern, content, re.IGNORECASE):
                print_success(f"{check_name:40s} found")
            else:
                print_error(f"{check_name:40s} NOT found")
                all_checks_passed = False

        if all_checks_passed:
            self.tests_passed += 1
        else:
            self.tests_failed += 1
        print()

    def test_network_architectures(self):
        """Test 8: Check network architecture files"""
        print_test(8, self.total_tests, "Checking network architectures...")

        # Check generator
        gen_file = self.project_root / 'buildGenerator.m'
        disc_file = self.project_root / 'buildDiscriminator.m'

        checks_passed = True

        if gen_file.exists():
            with open(gen_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            gen_checks = {
                'featureInputLayer': r'featureInputLayer',
                'fullyConnectedLayer': r'fullyConnectedLayer',
                'transposedConv2dLayer': r'transposedConv2dLayer',
                'batchNormalization': r'batchNormalizationLayer',
                'tanhLayer': r'tanhLayer',
                'dlnetwork': r'dlnetwork'
            }

            for check_name, pattern in gen_checks.items():
                if re.search(pattern, content):
                    print_success(f"Generator: {check_name:35s} found")
                else:
                    print_error(f"Generator: {check_name:35s} NOT found")
                    checks_passed = False
        else:
            print_error("buildGenerator.m not found")
            checks_passed = False

        if disc_file.exists():
            with open(disc_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            disc_checks = {
                'imageInputLayer': r'imageInputLayer',
                'convolution2dLayer': r'convolution2dLayer',
                'leakyReluLayer': r'leakyReluLayer',
                'dropoutLayer': r'dropoutLayer',
                'sigmoidLayer': r'sigmoidLayer'
            }

            for check_name, pattern in disc_checks.items():
                if re.search(pattern, content):
                    print_success(f"Discriminator: {check_name:32s} found")
                else:
                    print_error(f"Discriminator: {check_name:32s} NOT found")
                    checks_passed = False
        else:
            print_error("buildDiscriminator.m not found")
            checks_passed = False

        if checks_passed:
            self.tests_passed += 1
        else:
            self.tests_failed += 1
        print()

    def test_preprocessing(self):
        """Test 9: Check preprocessing functions"""
        print_test(9, self.total_tests, "Checking preprocessing functions...")

        filepath = self.project_root / 'preprocessAndLoadDatastore.m'
        if not filepath.exists():
            print_error("preprocessAndLoadDatastore.m not found")
            self.tests_failed += 1
            print()
            return

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        checks = {
            'RGB/Grayscale detection': r'rgb2gray|numRealRGB|numGrayscale',
            'Auto-crop function': r'autoCropWhiteBackground',
            'Image resize': r'imresize',
            'Normalization': r'\[-1,\s*1\]|normalize',
            'Data augmentation': r'augment|flip|rotate',
            'minibatchqueue': r'minibatchqueue'
        }

        all_checks_passed = True
        for check_name, pattern in checks.items():
            if re.search(pattern, content, re.IGNORECASE):
                print_success(f"{check_name:40s} found")
            else:
                print_error(f"{check_name:40s} NOT found")
                all_checks_passed = False

        if all_checks_passed:
            self.tests_passed += 1
        else:
            self.tests_failed += 1
        print()

    def test_output_functions(self):
        """Test 10: Check output generation functions"""
        print_test(10, self.total_tests, "Checking output generation...")

        # Check saveImageGrid
        grid_file = self.project_root / 'saveImageGrid.m'
        synth_file = self.project_root / 'generateSynthetic.m'

        checks_passed = True

        if grid_file.exists():
            with open(grid_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            if re.search(r'imwrite', content):
                print_success("saveImageGrid: image writing found")
            else:
                print_error("saveImageGrid: image writing NOT found")
                checks_passed = False
        else:
            print_error("saveImageGrid.m not found")
            checks_passed = False

        if synth_file.exists():
            with open(synth_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            synth_checks = {
                'Batch generation': r'batchSize|numBatches',
                'Image prediction': r'predict|forward',
                'Denormalization': r'\+ 1.*/ 2|denormalize',
                'Image saving': r'imwrite'
            }

            for check_name, pattern in synth_checks.items():
                if re.search(pattern, content, re.IGNORECASE):
                    print_success(f"generateSynthetic: {check_name:25s} found")
                else:
                    print_error(f"generateSynthetic: {check_name:25s} NOT found")
                    checks_passed = False
        else:
            print_error("generateSynthetic.m not found")
            checks_passed = False

        if checks_passed:
            self.tests_passed += 1
        else:
            self.tests_failed += 1
        print()

    def print_summary(self):
        """Print test summary"""
        print_header("Test Summary")

        print(f"Total Tests:  {self.total_tests}")
        print(f"{GREEN}Passed:       {self.tests_passed} ✓{RESET}")
        print(f"{RED}Failed:       {self.tests_failed} ✗{RESET}")
        print()

        success_rate = (self.tests_passed / self.total_tests) * 100

        if self.tests_failed == 0:
            print(f"{BOLD}{GREEN}╔{'='*58}╗{RESET}")
            print(f"{BOLD}{GREEN}║  ✓✓✓  ALL TESTS PASSED!  ✓✓✓                           ║{RESET}")
            print(f"{BOLD}{GREEN}║                                                          ║{RESET}")
            print(f"{BOLD}{GREEN}║  The GAN project structure is complete!                 ║{RESET}")
            print(f"{BOLD}{GREEN}║  Ready for MATLAB testing: test_gan_full.m              ║{RESET}")
            print(f"{BOLD}{GREEN}╚{'='*58}╝{RESET}")
        elif success_rate >= 80:
            print(f"{BOLD}{YELLOW}╔{'='*58}╗{RESET}")
            print(f"{BOLD}{YELLOW}║  ⚠  MOSTLY PASSING ({success_rate:.0f}%)  ⚠                              ║{RESET}")
            print(f"{BOLD}{YELLOW}║                                                          ║{RESET}")
            print(f"{BOLD}{YELLOW}║  Minor issues detected, but should work                 ║{RESET}")
            print(f"{BOLD}{YELLOW}╚{'='*58}╝{RESET}")
        else:
            print(f"{BOLD}{RED}╔{'='*58}╗{RESET}")
            print(f"{BOLD}{RED}║  ✗  SOME TESTS FAILED ({success_rate:.0f}%)  ✗                       ║{RESET}")
            print(f"{BOLD}{RED}║                                                          ║{RESET}")
            print(f"{BOLD}{RED}║  Please review the errors above                         ║{RESET}")
            print(f"{BOLD}{RED}╚{'='*58}╝{RESET}")

        print()

    def run_all_tests(self):
        """Run all tests"""
        print_header("GAN Project Structure Validation")

        self.test_required_files()
        self.test_directory_structure()
        self.test_file_completeness()
        self.test_function_definitions()
        self.test_documentation()
        self.test_gan_main_structure()
        self.test_train_gan_structure()
        self.test_network_architectures()
        self.test_preprocessing()
        self.test_output_functions()

        self.print_summary()

        return self.tests_failed == 0


if __name__ == '__main__':
    tester = GANStructureTest()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
