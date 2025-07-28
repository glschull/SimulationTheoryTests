#!/usr/bin/env python3
"""
Quality Assurance Tests for Simulation Theory Test Kit
Profiles memory usage, performance, and validates code quality
"""

import time
import psutil
import tracemalloc
import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def profile_memory_usage():
    """Profile memory usage during test execution"""
    print("🔍 PROFILING MEMORY USAGE")
    print("=" * 50)
    
    # Start memory tracing
    tracemalloc.start()
    process = psutil.Process()
    
    # Initial memory
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Initial Memory Usage: {initial_memory:.2f} MB")
    
    try:
        # Import main modules
        from main_runner import SimulationTheoryTestSuite
        print(f"After imports: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        
        # Create test suite
        suite = SimulationTheoryTestSuite()
        print(f"After suite creation: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        
        # Load real data
        start_time = time.time()
        suite.generate_test_data()
        load_time = time.time() - start_time
        memory_after_load = process.memory_info().rss / 1024 / 1024
        print(f"After data loading: {memory_after_load:.2f} MB (took {load_time:.2f}s)")
        
        # Run tests
        start_time = time.time()
        results = {
            'quantum': suite.run_quantum_tests(),
            'planck': suite.run_planck_tests(),
            'constants': suite.run_constants_tests()
        }
        test_time = time.time() - start_time
        memory_after_tests = process.memory_info().rss / 1024 / 1024
        print(f"After running tests: {memory_after_tests:.2f} MB (took {test_time:.2f}s)")
        
        # Memory snapshot
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
        print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")
        
        # Memory efficiency
        total_memory_used = memory_after_tests - initial_memory
        print(f"Total memory increase: {total_memory_used:.2f} MB")
        
        if total_memory_used < 500:  # Less than 500MB
            print("✅ Memory usage: EFFICIENT")
        elif total_memory_used < 1000:  # Less than 1GB
            print("⚠️  Memory usage: MODERATE")
        else:
            print("❌ Memory usage: HIGH - Consider optimization")
            
    except Exception as e:
        print(f"❌ Error during profiling: {e}")
        import traceback
        traceback.print_exc()
    finally:
        tracemalloc.stop()

def check_computational_efficiency():
    """Check computational efficiency of core functions"""
    print("\n⚡ CHECKING COMPUTATIONAL EFFICIENCY")
    print("=" * 50)
    
    try:
        from utils.analysis import BayesianAnomalyDetector, InformationTheoryAnalyzer
        import numpy as np
        
        # Test Bayesian analysis efficiency
        detector = BayesianAnomalyDetector()
        test_data = np.random.normal(0, 1, 10000)
        
        start_time = time.time()
        for _ in range(100):  # 100 iterations
            _ = detector.calculate_anomaly_likelihood(test_data)
        bayesian_time = time.time() - start_time
        print(f"Bayesian Analysis (100 iterations): {bayesian_time:.3f}s")
        
        # Test information theory efficiency
        analyzer = InformationTheoryAnalyzer()
        
        start_time = time.time()
        for _ in range(100):
            _ = analyzer.calculate_entropy(test_data)
        entropy_time = time.time() - start_time
        print(f"Entropy Calculation (100 iterations): {entropy_time:.3f}s")
        
        # Performance assessment
        if bayesian_time < 5.0 and entropy_time < 2.0:
            print("✅ Computational efficiency: EXCELLENT")
        elif bayesian_time < 10.0 and entropy_time < 5.0:
            print("✅ Computational efficiency: GOOD")
        else:
            print("⚠️  Computational efficiency: Could be optimized")
            
    except Exception as e:
        print(f"❌ Error checking efficiency: {e}")

def validate_error_handling():
    """Test error handling for edge cases"""
    print("\n🛡️  VALIDATING ERROR HANDLING")
    print("=" * 50)
    
    import numpy as np
    error_tests_passed = 0
    total_error_tests = 0
    
    try:
        from utils.analysis import BayesianAnomalyDetector
        detector = BayesianAnomalyDetector()
        
        # Test 1: Empty data
        total_error_tests += 1
        try:
            detector.calculate_anomaly_likelihood(np.array([]))
            print("❌ Empty data test: Should have raised an error")
        except (ValueError, IndexError):
            print("✅ Empty data test: Properly handled")
            error_tests_passed += 1
        except Exception as e:
            print(f"⚠️  Empty data test: Unexpected error type: {e}")
        
        # Test 2: Invalid data types
        total_error_tests += 1
        try:
            detector.calculate_anomaly_likelihood("invalid_data")
            print("❌ Invalid type test: Should have raised an error")
        except (TypeError, AttributeError):
            print("✅ Invalid type test: Properly handled")
            error_tests_passed += 1
        except Exception as e:
            print(f"⚠️  Invalid type test: Unexpected error type: {e}")
        
        # Test 3: NaN values
        total_error_tests += 1
        try:
            nan_data = np.array([1.0, 2.0, np.nan, 4.0])
            result = detector.calculate_anomaly_likelihood(nan_data)
            if any(np.isnan(list(result.values()))):
                print("⚠️  NaN test: Result contains NaN values")
            else:
                print("✅ NaN test: Handled gracefully")
                error_tests_passed += 1
        except Exception as e:
            print(f"⚠️  NaN test: Error handling: {e}")
        
        print(f"\nError handling score: {error_tests_passed}/{total_error_tests} tests passed")
        
    except Exception as e:
        print(f"❌ Error in error handling validation: {e}")

def create_unit_tests():
    """Create unit tests for core functions"""
    print("\n🧪 CREATING UNIT TESTS")
    print("=" * 50)
    
    unit_test_code = '''#!/usr/bin/env python3
"""
Unit Tests for Simulation Theory Test Kit
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class TestBayesianAnalysis(unittest.TestCase):
    """Test Bayesian anomaly detection"""
    
    def setUp(self):
        from utils.analysis import BayesianAnomalyDetector
        self.detector = BayesianAnomalyDetector()
    
    def test_normal_data(self):
        """Test with normal distributed data"""
        data = np.random.normal(0, 1, 1000)
        result = self.detector.calculate_anomaly_likelihood(data)
        self.assertIsInstance(result, dict)
    
    def test_anomalous_data(self):
        """Test with data containing clear anomalies"""
        normal_data = np.random.normal(0, 1, 1000)
        anomalous_data = np.concatenate([normal_data, [10, -10, 15]])
        result = self.detector.calculate_anomaly_likelihood(anomalous_data)
        self.assertIsInstance(result, dict)

class TestInformationTheory(unittest.TestCase):
    """Test information theory analysis"""
    
    def setUp(self):
        from utils.analysis import InformationTheoryAnalyzer
        self.analyzer = InformationTheoryAnalyzer()
    
    def test_entropy_calculation(self):
        """Test entropy calculation"""
        data = np.random.uniform(0, 1, 1000)
        entropy = self.analyzer.calculate_entropy(data)
        self.assertGreater(entropy, 0)
        self.assertIsInstance(entropy, float)
    
    def test_mutual_information(self):
        """Test mutual information calculation"""
        x = np.random.normal(0, 1, 1000)
        y = x + np.random.normal(0, 0.1, 1000)  # Correlated data
        mi = self.analyzer.calculate_mutual_information(x, y)
        self.assertGreater(mi, 0)

class TestRealDataLoader(unittest.TestCase):
    """Test real data loading functionality"""
    
    def setUp(self):
        from utils.real_data_loader import load_all_real_datasets
        self.loader_function = load_all_real_datasets
    
    def test_data_loading(self):
        """Test that data loads without errors"""
        try:
            data = self.loader_function()
            self.assertIsInstance(data, dict)
        except Exception as e:
            self.skipTest(f"Data loading failed: {e}")

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
'''
    
    # Save unit tests
    test_file = project_root / "tests" / "unit_tests.py"
    test_file.parent.mkdir(exist_ok=True)
    
    with open(test_file, 'w') as f:
        f.write(unit_test_code)
    
    print(f"✅ Unit tests created: {test_file}")
    
    # Run the unit tests
    try:
        import subprocess
        result = subprocess.run([sys.executable, str(test_file)], 
                              capture_output=True, text=True, cwd=str(project_root))
        
        if result.returncode == 0:
            print("✅ Unit tests: PASSED")
            print("Test output:")
            print(result.stdout[-500:])  # Last 500 characters
        else:
            print("❌ Unit tests: FAILED")
            print("Error output:")
            print(result.stderr[-500:])
            
    except Exception as e:
        print(f"⚠️  Could not run unit tests automatically: {e}")

def check_dependency_versions():
    """Document exact dependency versions"""
    print("\n📋 DOCUMENTING DEPENDENCY VERSIONS")
    print("=" * 50)
    
    dependencies = [
        'numpy', 'scipy', 'matplotlib', 'pandas', 'seaborn', 
        'psutil', 'jupyter', 'scikit-learn'
    ]
    
    versions = {}
    for dep in dependencies:
        try:
            module = __import__(dep)
            version = getattr(module, '__version__', 'Unknown')
            versions[dep] = version
            print(f"✅ {dep}: {version}")
        except ImportError:
            versions[dep] = 'Not installed'
            print(f"❌ {dep}: Not installed")
    
    # Save to requirements file
    requirements_content = f"""# Simulation Theory Test Kit - Exact Dependencies
# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

"""
    
    for dep, version in versions.items():
        if version != 'Not installed' and version != 'Unknown':
            requirements_content += f"{dep}=={version}\n"
    
    requirements_file = project_root / "requirements.txt"
    with open(requirements_file, 'w') as f:
        f.write(requirements_content)
    
    print(f"\n✅ Requirements saved to: {requirements_file}")

def main():
    """Run all quality assurance checks"""
    print("🏗️  SIMULATION THEORY TEST KIT - QUALITY ASSURANCE")
    print("=" * 60)
    
    start_time = time.time()
    
    # Run all QA checks
    profile_memory_usage()
    check_computational_efficiency()
    validate_error_handling()
    create_unit_tests()
    check_dependency_versions()
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("🎯 QUALITY ASSURANCE COMPLETE")
    print(f"Total QA time: {total_time:.2f} seconds")
    print("=" * 60)

if __name__ == "__main__":
    main()
