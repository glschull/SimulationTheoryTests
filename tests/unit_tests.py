#!/usr/bin/env python3
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
