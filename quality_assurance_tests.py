#!/usr/bin/env python3
"""
Unit tests for Simulation Theory Test Kit core functions.
Ensures code reliability and reproducibility.
"""

import unittest
import numpy as np
import json
import sys
import os
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

class TestSimulationFramework(unittest.TestCase):
    """Test core simulation theory framework components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_data_size = 1000
        self.test_data = np.random.random(self.test_data_size)
        self.results_dir = Path("results")
        self.data_dir = Path("data")
        
    def test_data_directory_exists(self):
        """Test that data directory exists and contains expected files."""
        self.assertTrue(self.data_dir.exists(), "Data directory should exist")
        
        # Check for key data files
        expected_files = [
            "auger_cosmic_ray_events.csv",
            "icecube_neutrino_events.csv", 
            "planck_cmb_temperature_map.npy",
            "bell_test_measurements.csv",
            "nist_codata_2018_constants.json"
        ]
        
        for file_name in expected_files:
            file_path = self.data_dir / file_name
            self.assertTrue(file_path.exists(), f"Expected data file {file_name} should exist")
    
    def test_results_directory_exists(self):
        """Test that results directory exists and contains analysis results."""
        self.assertTrue(self.results_dir.exists(), "Results directory should exist")
        
        # Check for key result files
        expected_files = [
            "comprehensive_analysis.json",
            "simulation_test_results.json",
            "test_summary.txt"
        ]
        
        for file_name in expected_files:
            file_path = self.results_dir / file_name
            self.assertTrue(file_path.exists(), f"Expected result file {file_name} should exist")
    
    def test_comprehensive_analysis_structure(self):
        """Test that comprehensive analysis JSON has correct structure."""
        analysis_file = self.results_dir / "comprehensive_analysis.json"
        
        with open(analysis_file, 'r') as f:
            analysis = json.load(f)
        
        # Test top-level structure
        required_keys = [
            "datasets_analyzed",
            "individual_analyses", 
            "cross_dataset_analysis",
            "overall_assessment"
        ]
        
        for key in required_keys:
            self.assertIn(key, analysis, f"Analysis should contain {key}")
        
        # Test overall assessment structure
        assessment = analysis["overall_assessment"]
        assessment_keys = [
            "overall_suspicion_score",
            "confidence_level",
            "average_simulation_probability"
        ]
        
        for key in assessment_keys:
            self.assertIn(key, assessment, f"Assessment should contain {key}")
        
        # Test score ranges
        score = assessment["overall_suspicion_score"]
        self.assertGreaterEqual(score, 0.0, "Suspicion score should be >= 0")
        self.assertLessEqual(score, 1.0, "Suspicion score should be <= 1")
    
    def test_individual_analysis_structure(self):
        """Test structure of individual dataset analyses."""
        analysis_file = self.results_dir / "comprehensive_analysis.json"
        
        with open(analysis_file, 'r') as f:
            analysis = json.load(f)
        
        individual = analysis["individual_analyses"]
        
        # Test each dataset analysis
        for dataset_name, dataset_analysis in individual.items():
            required_sections = [
                "bayesian_analysis",
                "digital_signatures", 
                "information_theory",
                "basic_statistics"
            ]
            
            for section in required_sections:
                self.assertIn(section, dataset_analysis, 
                            f"Dataset {dataset_name} should have {section}")
            
            # Test Bayesian analysis
            bayesian = dataset_analysis["bayesian_analysis"]
            self.assertIn("simulation_probability", bayesian)
            
            sim_prob = bayesian["simulation_probability"]
            self.assertGreaterEqual(sim_prob, 0.0, "Simulation probability should be >= 0")
            self.assertLessEqual(sim_prob, 1.0, "Simulation probability should be <= 1")
    
    def test_cross_dataset_analysis(self):
        """Test cross-dataset correlation analysis."""
        analysis_file = self.results_dir / "comprehensive_analysis.json"
        
        with open(analysis_file, 'r') as f:
            analysis = json.load(f)
        
        cross_analysis = analysis["cross_dataset_analysis"]
        
        # Test mutual information
        self.assertIn("mutual_information", cross_analysis)
        mi = cross_analysis["mutual_information"]
        
        # Check for expected dataset pairs
        expected_pairs = [
            "quantum_measurements_vs_planck_intervals",
            "quantum_measurements_vs_physical_constants",
            "quantum_measurements_vs_cmb_temperatures",
            "planck_intervals_vs_physical_constants",
            "planck_intervals_vs_cmb_temperatures",
            "physical_constants_vs_cmb_temperatures"
        ]
        
        for pair in expected_pairs:
            self.assertIn(pair, mi, f"Mutual information should include {pair}")
            self.assertGreaterEqual(mi[pair], 0.0, f"MI for {pair} should be >= 0")
    
    def test_statistical_calculations(self):
        """Test basic statistical calculation functions."""
        # Test with known data
        test_data = np.array([1, 2, 3, 4, 5])
        
        # Test mean calculation
        expected_mean = 3.0
        calculated_mean = np.mean(test_data)
        self.assertAlmostEqual(calculated_mean, expected_mean, places=6)
        
        # Test standard deviation
        expected_std = np.sqrt(2.0)  # Known value for this sequence
        calculated_std = np.std(test_data)
        self.assertAlmostEqual(calculated_std, expected_std, places=6)
    
    def test_data_loading_consistency(self):
        """Test that data loading produces consistent results."""
        # Check if we can load the CMB data
        cmb_file = self.data_dir / "planck_cmb_temperature_map.npy"
        if cmb_file.exists():
            cmb_data = np.load(cmb_file)
            
            # Test basic properties
            self.assertEqual(len(cmb_data.shape), 2, "CMB data should be 2D")
            self.assertGreater(cmb_data.size, 0, "CMB data should not be empty")
            
            # Test temperature range (CMB should be around 2.7K)
            mean_temp = np.mean(cmb_data)
            self.assertGreater(mean_temp, 2.0, "CMB temperature should be > 2K")
            self.assertLess(mean_temp, 4.0, "CMB temperature should be < 4K")
    
    def test_compression_ratios(self):
        """Test compression ratio calculations."""
        # Test with known compressible data
        repetitive_data = np.array([1] * 100 + [2] * 100)
        
        import zlib
        compressed = zlib.compress(repetitive_data.tobytes())
        ratio = len(repetitive_data.tobytes()) / len(compressed)
        
        # Repetitive data should compress well
        self.assertGreater(ratio, 1.0, "Repetitive data should have compression ratio > 1")
        
        # Test with random data (should compress poorly)
        random_data = np.random.random(200).astype(np.float64)
        compressed_random = zlib.compress(random_data.tobytes())
        ratio_random = len(random_data.tobytes()) / len(compressed_random)
        
        # Random data should compress worse than repetitive data
        self.assertLess(ratio_random, ratio, "Random data should compress worse than repetitive")

class TestResultsIntegrity(unittest.TestCase):
    """Test integrity and consistency of analysis results."""
    
    def test_score_consistency(self):
        """Test that scores are consistent across different result files."""
        # Load comprehensive analysis
        comp_file = Path("results/comprehensive_analysis.json")
        with open(comp_file, 'r') as f:
            comp_analysis = json.load(f)
        
        # Load simulation test results
        sim_file = Path("results/simulation_test_results.json")
        if sim_file.exists():
            with open(sim_file, 'r') as f:
                sim_results = json.load(f)
            
            # Compare overall scores if present in both files
            if "overall_assessment" in comp_analysis and "overall_score" in sim_results:
                comp_score = comp_analysis["overall_assessment"]["overall_suspicion_score"]
                sim_score = sim_results["overall_score"]
                
                # Scores should be close (allowing for small differences in calculation)
                self.assertAlmostEqual(comp_score, sim_score, places=2,
                                     msg="Overall scores should be consistent between files")
    
    def test_publication_files_exist(self):
        """Test that publication-ready files exist."""
        expected_files = [
            "EXECUTIVE_SUMMARY.md",
            "COMPREHENSIVE_RESULTS_TABLE.md",
            "METHODOLOGY.md"
        ]
        
        for file_name in expected_files:
            file_path = Path(file_name)
            self.assertTrue(file_path.exists(), f"Publication file {file_name} should exist")
    
    def test_visualization_files(self):
        """Test that visualization files are properly generated."""
        results_dir = Path("results")
        
        # Check for PNG files
        png_files = list(results_dir.glob("*.png"))
        self.assertGreater(len(png_files), 0, "Should have generated visualization files")
        
        # Check file sizes (should not be empty)
        for png_file in png_files:
            size = png_file.stat().st_size
            self.assertGreater(size, 1000, f"PNG file {png_file.name} should be > 1KB")

def run_quality_assurance_tests():
    """Run all quality assurance tests and generate report."""
    print("🧪 SIMULATION THEORY TEST KIT - QUALITY ASSURANCE")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestSimulationFramework))
    suite.addTest(unittest.makeSuite(TestResultsIntegrity))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate QA report
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total_tests - failures - errors
    
    print("\n" + "=" * 60)
    print("🎯 QUALITY ASSURANCE REPORT")
    print("=" * 60)
    print(f"Total Tests Run: {total_tests}")
    print(f"Tests Passed: {passed}")
    print(f"Tests Failed: {failures}")
    print(f"Tests with Errors: {errors}")
    print(f"Success Rate: {(passed/total_tests)*100:.1f}%")
    
    if failures == 0 and errors == 0:
        print("\n✅ ALL TESTS PASSED - SYSTEM IS READY FOR PRODUCTION")
    else:
        print(f"\n⚠️ {failures + errors} TESTS FAILED - REVIEW REQUIRED")
        
        if result.failures:
            print("\nFAILURES:")
            for test, traceback in result.failures:
                print(f"- {test}: {traceback.split(chr(10))[-2]}")
        
        if result.errors:
            print("\nERRORS:")
            for test, traceback in result.errors:
                print(f"- {test}: {traceback.split(chr(10))[-2]}")
    
    return result

if __name__ == "__main__":
    run_quality_assurance_tests()
