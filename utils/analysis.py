"""
Statistical Analysis Module for Simulation Theory Tests
======================================================

Advanced statistical analysis tools for detecting computational signatures
in physical data that might indicate we're living in a simulation.

Author: AI Assistant for Garrett Schull's SimulationTheoryTests
"""

import numpy as np
import pandas as pd
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class BayesianAnomalyDetector:
    """Bayesian approach to detecting anomalies that might indicate simulation artifacts"""
    
    def __init__(self):
        self.prior_simulation_probability = 0.5  # Neutral prior
        
    def calculate_anomaly_likelihood(self, data: np.ndarray, 
                                   expected_distribution: str = 'normal') -> Dict[str, float]:
        """
        Calculate the likelihood of observing anomalies in the data
        
        Args:
            data: Input data array
            expected_distribution: Expected underlying distribution
            
        Returns:
            Dictionary with likelihood scores
        """
        likelihoods = {}
        
        # Test against expected distribution
        if expected_distribution == 'normal':
            # Shapiro-Wilk test for normality
            stat, p_value = stats.shapiro(data[:5000] if len(data) > 5000 else data)
            likelihoods['normality_p'] = p_value
            likelihoods['normality_deviation'] = 1 - p_value
            
        elif expected_distribution == 'uniform':
            # Kolmogorov-Smirnov test against uniform
            stat, p_value = stats.kstest(data, 'uniform')
            likelihoods['uniformity_p'] = p_value
            likelihoods['uniformity_deviation'] = 1 - p_value
            
        elif expected_distribution == 'exponential':
            # Test against exponential distribution
            stat, p_value = stats.kstest(data, 'expon')
            likelihoods['exponential_p'] = p_value
            likelihoods['exponential_deviation'] = 1 - p_value
        
        # Additional anomaly indicators
        # 1. Excessive periodicity
        fft_result = fft(data)
        power_spectrum = np.abs(fft_result) ** 2
        
        # Find dominant peaks
        peaks, _ = signal.find_peaks(power_spectrum, height=np.percentile(power_spectrum, 95))
        periodicity_score = len(peaks) / len(data) * 1000  # Normalized
        likelihoods['periodicity_anomaly'] = min(1.0, periodicity_score)
        
        # 2. Quantization artifacts
        # Look for evidence of discrete steps in data
        rounded_data = np.round(data, decimals=10)
        unique_values = len(np.unique(rounded_data))
        expected_unique = min(len(data), len(data) * 0.8)  # Expected for continuous data
        quantization_score = 1 - (unique_values / expected_unique)
        likelihoods['quantization_anomaly'] = max(0, quantization_score)
        
        # 3. Artificial patterns in differences
        diffs = np.diff(data)
        diff_hist, _ = np.histogram(diffs, bins=50)
        diff_entropy = stats.entropy(diff_hist + 1e-10)  # Add small value to avoid log(0)
        max_entropy = np.log(50)  # Maximum entropy for 50 bins
        entropy_anomaly = 1 - (diff_entropy / max_entropy)
        likelihoods['entropy_anomaly'] = entropy_anomaly
        
        return likelihoods
    
    def update_simulation_probability(self, likelihood_data: Dict[str, float]) -> float:
        """
        Update simulation probability using Bayesian inference
        
        Args:
            likelihood_data: Dictionary of anomaly likelihoods
            
        Returns:
            Updated probability that we're in a simulation
        """
        # Combine evidence using naive Bayes approach (assumes independence)
        # P(simulation|evidence) ∝ P(evidence|simulation) * P(simulation)
        
        # Define how likely each anomaly is under simulation vs reality
        simulation_multipliers = {
            'normality_deviation': 2.0,      # Simulations might have non-normal artifacts
            'uniformity_deviation': 1.5,     # Discrete systems might deviate from uniform
            'exponential_deviation': 1.2,    # Natural processes are often exponential
            'periodicity_anomaly': 3.0,      # High periodicity suggests computational grid
            'quantization_anomaly': 5.0,     # Quantization strongly suggests digital system
            'entropy_anomaly': 2.5           # Low entropy suggests programmed patterns
        }
        
        # Calculate likelihood ratio
        likelihood_ratio = 1.0
        for anomaly, value in likelihood_data.items():
            if anomaly in simulation_multipliers:
                # P(evidence|simulation) / P(evidence|reality)
                multiplier = simulation_multipliers[anomaly]
                likelihood_ratio *= (value * multiplier + (1 - value)) / (value + (1 - value))
        
        # Bayesian update
        posterior_odds = (self.prior_simulation_probability / (1 - self.prior_simulation_probability)) * likelihood_ratio
        posterior_probability = posterior_odds / (1 + posterior_odds)
        
        return min(0.99, max(0.01, posterior_probability))  # Clamp between 1% and 99%


class DigitalSignatureDetector:
    """Detect signatures that might indicate digital/computational origin"""
    
    def __init__(self):
        self.suspicious_patterns = [
            # Common programming constants
            'deadbeef', 'cafebabe', 'feedface', 'badc0de', '8badf00d',
            # Mathematical constants (might be hardcoded)
            '314159', '271828', '161803',  # pi, e, golden ratio
            # Powers of 2 (common in computing)
            '1024', '2048', '4096', '8192',
            # Common debug values
            '12345678', 'abcdefab', 'ffffffff', '00000000'
        ]
    
    def analyze_digital_signatures(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Analyze data for digital/computational signatures
        
        Args:
            data: Input data array
            
        Returns:
            Analysis results
        """
        results = {}
        
        # Convert data to various representations for analysis
        data_bytes = data.tobytes()
        data_hex = data_bytes.hex()
        
        # 1. Look for suspicious hex patterns
        found_patterns = []
        for pattern in self.suspicious_patterns:
            if pattern.lower() in data_hex.lower():
                found_patterns.append(pattern)
                
        results['suspicious_hex_patterns'] = found_patterns
        results['suspicious_pattern_count'] = len(found_patterns)
        
        # 2. Analyze bit patterns
        # Convert to binary representation
        data_int = np.array(data * 1e10, dtype=np.int64)  # Scale and convert to int
        binary_strings = [format(abs(x), 'b') for x in data_int]
        
        # Look for excessive regularity in binary patterns
        bit_lengths = [len(b) for b in binary_strings]
        bit_length_entropy = stats.entropy(np.bincount(bit_lengths))
        results['bit_pattern_entropy'] = bit_length_entropy
        
        # 3. Check for power-of-2 biases
        # Natural phenomena rarely align perfectly with powers of 2
        scaled_data = np.abs(data) * 1e6  # Scale to reasonable integer range
        power_of_2_proximities = []
        
        for value in scaled_data[:1000]:  # Sample first 1000 values
            # Find closest power of 2
            log2_val = np.log2(max(value, 1e-10))
            closest_power = 2 ** np.round(log2_val)
            proximity = abs(value - closest_power) / max(closest_power, 1e-10)
            power_of_2_proximities.append(proximity)
        
        avg_power2_proximity = np.mean(power_of_2_proximities)
        results['power_of_2_bias'] = 1 - avg_power2_proximity  # Higher = more biased toward powers of 2
        
        # 4. Floating point artifacts
        # Look for patterns that suggest floating point representation
        decimal_parts = np.abs(data) - np.floor(np.abs(data))
        
        # IEEE 754 floating point has specific precision limits
        # Look for clustering around representable values
        rounded_decimals = np.round(decimal_parts * 2**23) / 2**23  # 32-bit float precision
        unique_rounded = len(np.unique(rounded_decimals))
        expected_unique = min(len(decimal_parts), len(decimal_parts) * 0.9)
        floating_point_score = 1 - (unique_rounded / expected_unique)
        results['floating_point_artifacts'] = max(0, floating_point_score)
        
        # 5. Calculate overall digital signature score
        signature_components = [
            results['suspicious_pattern_count'] / 10,  # Normalize by expected max
            results['power_of_2_bias'],
            results['floating_point_artifacts'],
            1 - (bit_length_entropy / 5)  # Lower entropy = more suspicious
        ]
        
        results['digital_signature_score'] = np.mean(signature_components)
        
        return results


class InformationTheoryAnalyzer:
    """Apply information theory to detect artificial constraints"""
    
    def __init__(self):
        pass
    
    def calculate_kolmogorov_complexity_approximation(self, data: np.ndarray) -> Dict[str, float]:
        """
        Approximate Kolmogorov complexity using compression ratios
        
        Args:
            data: Input data array
            
        Returns:
            Complexity measures
        """
        import zlib
        import bz2
        import lzma
        
        # Convert data to bytes
        data_bytes = data.tobytes()
        original_size = len(data_bytes)
        
        results = {}
        
        # Try different compression algorithms
        algorithms = {
            'zlib': zlib.compress,
            'bz2': bz2.compress,
            'lzma': lzma.compress
        }
        
        for name, compress_func in algorithms.items():
            try:
                compressed = compress_func(data_bytes)
                compression_ratio = len(compressed) / original_size
                results[f'{name}_compression_ratio'] = compression_ratio
                results[f'{name}_complexity_estimate'] = -np.log2(compression_ratio)
            except Exception as e:
                results[f'{name}_compression_ratio'] = 1.0
                results[f'{name}_complexity_estimate'] = 0.0
        
        # Calculate ensemble complexity estimate
        compression_ratios = [results[f'{alg}_compression_ratio'] for alg in algorithms.keys()]
        results['average_compression_ratio'] = np.mean(compression_ratios)
        results['min_compression_ratio'] = np.min(compression_ratios)
        results['complexity_score'] = -np.log2(results['min_compression_ratio'])
        
        # High compressibility suggests low Kolmogorov complexity (more structured/artificial)
        results['artificiality_indicator'] = 1 - results['min_compression_ratio']
        
        return results
    
    def analyze_mutual_information(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """
        Calculate mutual information between two datasets
        
        Args:
            data1, data2: Input data arrays
            
        Returns:
            Mutual information score
        """
        # Discretize the data for mutual information calculation
        bins = 50
        
        # Create 2D histogram
        hist_2d, x_edges, y_edges = np.histogram2d(data1, data2, bins=bins)
        
        # Calculate marginal histograms
        hist_x = np.sum(hist_2d, axis=1)
        hist_y = np.sum(hist_2d, axis=0)
        
        # Normalize to get probabilities
        total_count = np.sum(hist_2d)
        p_xy = hist_2d / total_count
        p_x = hist_x / total_count
        p_y = hist_y / total_count
        
        # Calculate mutual information
        mutual_info = 0.0
        for i in range(bins):
            for j in range(bins):
                if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mutual_info += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
        
        return mutual_info


class SimulationHypothesisEvaluator:
    """Comprehensive evaluator that combines all analysis methods"""
    
    def __init__(self):
        self.bayesian_detector = BayesianAnomalyDetector()
        self.digital_detector = DigitalSignatureDetector()
        self.info_analyzer = InformationTheoryAnalyzer()
    
    def comprehensive_analysis(self, datasets: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Perform comprehensive simulation hypothesis analysis
        
        Args:
            datasets: Dictionary of named datasets to analyze
            
        Returns:
            Comprehensive analysis results
        """
        results = {
            'datasets_analyzed': list(datasets.keys()),
            'individual_analyses': {},
            'cross_dataset_analysis': {},
            'overall_assessment': {}
        }
        
        # Analyze each dataset individually
        for name, data in datasets.items():
            print(f"🔍 Analyzing {name}...")
            
            dataset_results = {}
            
            # Bayesian anomaly detection
            likelihoods = self.bayesian_detector.calculate_anomaly_likelihood(data)
            simulation_prob = self.bayesian_detector.update_simulation_probability(likelihoods)
            
            dataset_results['bayesian_analysis'] = {
                'anomaly_likelihoods': likelihoods,
                'simulation_probability': simulation_prob
            }
            
            # Digital signature detection
            digital_sigs = self.digital_detector.analyze_digital_signatures(data)
            dataset_results['digital_signatures'] = digital_sigs
            
            # Information theory analysis
            complexity = self.info_analyzer.calculate_kolmogorov_complexity_approximation(data)
            dataset_results['information_theory'] = complexity
            
            # Statistical summary
            dataset_results['basic_statistics'] = {
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'skewness': float(stats.skew(data)),
                'kurtosis': float(stats.kurtosis(data)),
                'range': float(np.ptp(data)),
                'entropy_estimate': float(stats.entropy(np.histogram(data, bins=50)[0] + 1))
            }
            
            results['individual_analyses'][name] = dataset_results
        
        # Cross-dataset analysis
        if len(datasets) > 1:
            dataset_list = list(datasets.items())
            
            # Mutual information between datasets
            mutual_infos = {}
            for i in range(len(dataset_list)):
                for j in range(i + 1, len(dataset_list)):
                    name1, data1 = dataset_list[i]
                    name2, data2 = dataset_list[j]
                    
                    # Take samples if datasets are too large
                    sample_size = min(len(data1), len(data2), 10000)
                    sample1 = np.random.choice(data1, sample_size, replace=False)
                    sample2 = np.random.choice(data2, sample_size, replace=False)
                    
                    mi = self.info_analyzer.analyze_mutual_information(sample1, sample2)
                    mutual_infos[f'{name1}_vs_{name2}'] = mi
            
            results['cross_dataset_analysis']['mutual_information'] = mutual_infos
            
            # Correlation analysis
            correlations = {}
            for i in range(len(dataset_list)):
                for j in range(i + 1, len(dataset_list)):
                    name1, data1 = dataset_list[i]
                    name2, data2 = dataset_list[j]
                    
                    sample_size = min(len(data1), len(data2), 10000)
                    sample1 = np.random.choice(data1, sample_size, replace=False)
                    sample2 = np.random.choice(data2, sample_size, replace=False)
                    
                    correlation = np.corrcoef(sample1, sample2)[0, 1]
                    correlations[f'{name1}_vs_{name2}'] = correlation if not np.isnan(correlation) else 0.0
            
            results['cross_dataset_analysis']['correlations'] = correlations
        
        # Overall assessment
        simulation_probabilities = []
        digital_scores = []
        compression_scores = []
        
        for analysis in results['individual_analyses'].values():
            simulation_probabilities.append(analysis['bayesian_analysis']['simulation_probability'])
            digital_scores.append(analysis['digital_signatures']['digital_signature_score'])
            compression_scores.append(analysis['information_theory']['artificiality_indicator'])
        
        # Calculate overall suspicion score first
        overall_suspicion_score = float(np.mean([
            np.mean(simulation_probabilities),
            np.mean(digital_scores),
            np.mean(compression_scores)
        ]))
        
        results['overall_assessment'] = {
            'average_simulation_probability': float(np.mean(simulation_probabilities)),
            'max_simulation_probability': float(np.max(simulation_probabilities)),
            'average_digital_signature_score': float(np.mean(digital_scores)),
            'average_compression_artificiality': float(np.mean(compression_scores)),
            'overall_suspicion_score': overall_suspicion_score,
            'confidence_level': self._calculate_confidence_level_from_score(overall_suspicion_score)
        }
        
        return results
    
    def _calculate_confidence_level_from_score(self, overall_score: float) -> str:
        """Calculate confidence level from the overall suspicion score"""        
        if overall_score > 0.8:
            return "HIGH - Strong evidence of computational signatures"
        elif overall_score > 0.6:
            return "MEDIUM-HIGH - Notable computational signatures detected"
        elif overall_score > 0.4:
            return "MEDIUM - Some computational signatures present"
        elif overall_score > 0.2:
            return "MEDIUM-LOW - Weak computational signatures"
        else:
            return "LOW - Minimal evidence of computational signatures"
    
    def generate_summary_report(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable summary report"""
        report = """
🎯 SIMULATION HYPOTHESIS ANALYSIS REPORT
========================================

OVERALL ASSESSMENT:
- Suspicion Score: {:.3f}/1.000
- Confidence Level: {}
- Datasets Analyzed: {}

INDIVIDUAL DATASET ANALYSIS:
""".format(
            results['overall_assessment']['overall_suspicion_score'],
            results['overall_assessment']['confidence_level'],
            len(results['datasets_analyzed'])
        )
        
        for dataset_name, analysis in results['individual_analyses'].items():
            report += f"""
📊 {dataset_name.upper()}:
   Simulation Probability: {analysis['bayesian_analysis']['simulation_probability']:.3f}
   Digital Signature Score: {analysis['digital_signatures']['digital_signature_score']:.3f}
   Compression Artificiality: {analysis['information_theory']['artificiality_indicator']:.3f}
   Suspicious Patterns Found: {analysis['digital_signatures']['suspicious_pattern_count']}
"""
        
        if 'mutual_information' in results['cross_dataset_analysis']:
            report += "\nCROSS-DATASET CORRELATIONS:\n"
            for pair, mi in results['cross_dataset_analysis']['mutual_information'].items():
                report += f"   {pair}: {mi:.4f} bits\n"
        
        report += f"""
INTERPRETATION:
{self._interpret_results(results)}

⚠️  DISCLAIMER: This analysis is experimental and should not be considered
    definitive evidence for or against the simulation hypothesis.
"""
        
        return report
    
    def _interpret_results(self, results: Dict[str, Any]) -> str:
        """Provide interpretation of the analysis results"""
        score = results['overall_assessment']['overall_suspicion_score']
        
        if score > 0.7:
            return """The analysis detected multiple computational signatures across datasets.
These patterns could indicate artificial constraints or digital processing artifacts.
However, they could also result from measurement limitations or natural processes."""
        
        elif score > 0.5:
            return """Some computational signatures were detected, but the evidence is mixed.
The patterns observed could have natural explanations, though some aspects
suggest possible artificial constraints."""
        
        elif score > 0.3:
            return """Minimal computational signatures detected. The data appears largely
consistent with natural physical processes, though some minor anomalies
were observed that warrant further investigation."""
        
        else:
            return """Very few computational signatures detected. The analyzed data
appears consistent with natural physical processes and shows little evidence
of artificial constraints or digital processing artifacts."""


if __name__ == "__main__":
    print("📊 Simulation Theory Statistical Analysis Tools Ready!")
    print("Import this module to perform advanced statistical analysis on your datasets.")
