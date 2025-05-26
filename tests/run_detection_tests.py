#!/usr/bin/env python3
"""
Test runner for auto-detection system tests
"""

import unittest
import sys
import os
from pathlib import Path

def run_tests():
    """Run all auto-detection tests"""
    
    # Add current directory to path for test imports
    test_dir = Path(__file__).parent
    sys.path.insert(0, str(test_dir))
    
    # Discover and run tests
    loader = unittest.TestLoader()
    
    # Load specific test modules
    test_modules = [
        'setup.test_detect_system_config',
        'setup.test_cuda_compatibility_matrix', 
        'setup.test_show_cuda_matrix',
        'setup.test_detection_integration'
    ]
    
    suite = unittest.TestSuite()
    
    for module_name in test_modules:
        try:
            module_suite = loader.loadTestsFromName(module_name)
            suite.addTest(module_suite)
            print(f"✅ Loaded tests from {module_name}")
        except Exception as e:
            print(f"❌ Failed to load {module_name}: {e}")
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*50)
    print("Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    # Return success/failure
    return len(result.failures) + len(result.errors) == 0

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)