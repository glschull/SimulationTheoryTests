"""
Machine Learning Anomaly Detection for Simulation Hypothesis Testing
===================================================================

Advanced ML-based analysis for detecting computational signatures and 
simulation artifacts in real scientific datasets.

Features:
- Neural network anomaly detection (using sklearn)
- Deep learning pattern recognition
- Ensemble methods for robust detection
- Validation against statistical methods

Author: AI Assistant for Garrett Schull's SimulationTheoryTests
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class SimulationAnomalyDetector:
    """
    Advanced ML-based anomaly detection for simulation hypothesis testing
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.training_history = {}
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        
    def prepare_features(self, data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
        """
        Extract and engineer features from multiple datasets for ML analysis
        Creates multiple samples by sliding window approach on time series data
        """
        all_features = []
        feature_names = []
        
        # First, determine feature names
        for dataset_name, dataset in data.items():
            if len(dataset) == 0:
                continue
                
            # Ensure dataset is 1D
            if dataset.ndim > 1:
                dataset = dataset.flatten()
            
            # Add feature names for this dataset
            feature_names.extend([f"{dataset_name}_{name}" for name in [
                'mean', 'std', 'skewness', 'kurtosis', 'entropy', 
                'median', 'iqr', 'min', 'max', 'range'
            ]])
            feature_names.extend([f"{dataset_name}_{name}" for name in [
                'normality_pvalue', 'uniform_ks', 'exponential_ks',
                'power_law_alpha', 'benford_deviation'
            ]])
            feature_names.extend([f"{dataset_name}_{name}" for name in [
                'dominant_freq', 'spectral_centroid', 'spectral_bandwidth',
                'spectral_rolloff', 'zero_crossing_rate'
            ]])
            feature_names.extend([f"{dataset_name}_{name}" for name in [
                'sample_entropy', 'lempel_ziv', 'fractal_dimension',
                'hurst_exponent', 'lyapunov_exponent'
            ]])
        
        # Extract features using sliding windows to create multiple samples
        window_size = 200  # Size of each window
        stride = 50       # Stride between windows
        
        for dataset_name, dataset in data.items():
            if len(dataset) == 0:
                continue
                
            # Ensure dataset is 1D
            if dataset.ndim > 1:
                dataset = dataset.flatten()
            
            # Create windows for this dataset
            dataset_windows = []
            if len(dataset) >= window_size:
                for i in range(0, len(dataset) - window_size + 1, stride):
                    window = dataset[i:i + window_size]
                    dataset_windows.append(window)
            else:
                # If dataset is too small, use the whole dataset
                dataset_windows.append(dataset)
        
        # Extract features for each combination of windows
        max_windows = max([len(self._create_windows(data[name], window_size, stride)) 
                          for name in data.keys() if len(data[name]) > 0])
        
        for window_idx in range(max_windows):
            sample_features = []
            
            for dataset_name, dataset in data.items():
                if len(dataset) == 0:
                    # Add zero features for missing datasets
                    sample_features.extend([0.0] * 25)  # 25 features per dataset
                    continue
                    
                # Ensure dataset is 1D
                if dataset.ndim > 1:
                    dataset = dataset.flatten()
                
                # Get the appropriate window
                windows = self._create_windows(dataset, window_size, stride)
                if window_idx < len(windows):
                    window_data = windows[window_idx]
                else:
                    # Use the last window if we've run out
                    window_data = windows[-1] if windows else dataset
                
                # Extract features for this window
                stat_features = self._extract_statistical_features(window_data)
                dist_features = self._extract_distribution_features(window_data)
                freq_features = self._extract_frequency_features(window_data)
                complexity_features = self._extract_complexity_features(window_data)
                
                sample_features.extend(stat_features)
                sample_features.extend(dist_features)
                sample_features.extend(freq_features)
                sample_features.extend(complexity_features)
            
            all_features.append(sample_features)
        
        return np.array(all_features), feature_names
    
    def _create_windows(self, data: np.ndarray, window_size: int, stride: int) -> List[np.ndarray]:
        """Create sliding windows from data"""
        windows = []
        if len(data) >= window_size:
            for i in range(0, len(data) - window_size + 1, stride):
                windows.append(data[i:i + window_size])
        else:
            windows.append(data)
        return windows
    
    def _extract_statistical_features(self, data: np.ndarray) -> List[float]:
        """Extract basic statistical features"""
        from scipy import stats
        
        # Handle edge cases
        if len(data) < 2:
            return [0.0] * 10
        
        features = []
        features.append(float(np.mean(data)))
        features.append(float(np.std(data)))
        features.append(float(stats.skew(data)))
        features.append(float(stats.kurtosis(data)))
        
        # Entropy calculation
        hist, _ = np.histogram(data, bins=min(50, len(data)//10))
        hist = hist / np.sum(hist)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        features.append(float(entropy))
        
        features.append(float(np.median(data)))
        features.append(float(np.percentile(data, 75) - np.percentile(data, 25)))
        features.append(float(np.min(data)))
        features.append(float(np.max(data)))
        features.append(float(np.max(data) - np.min(data)))
        
        return features
    
    def _extract_distribution_features(self, data: np.ndarray) -> List[float]:
        """Extract distribution-based features"""
        from scipy import stats
        
        if len(data) < 10:
            return [0.0] * 5
        
        features = []
        
        # Normality test
        _, p_norm = stats.normaltest(data)
        features.append(float(p_norm))
        
        # Uniformity test
        _, p_uniform = stats.kstest(data, 'uniform')
        features.append(float(p_uniform))
        
        # Exponential test
        _, p_exp = stats.kstest(data, 'expon')
        features.append(float(p_exp))
        
        # Power law fitting
        try:
            alpha = stats.powerlaw.fit(data)[0]
            features.append(float(alpha))
        except:
            features.append(0.0)
        
        # Benford's law deviation
        benford_dev = self._benford_deviation(data)
        features.append(float(benford_dev))
        
        return features
    
    def _extract_frequency_features(self, data: np.ndarray) -> List[float]:
        """Extract frequency domain features"""
        if len(data) < 8:
            return [0.0] * 5
        
        # FFT analysis
        fft = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(data))
        magnitude = np.abs(fft)
        
        features = []
        
        # Dominant frequency
        dominant_freq_idx = np.argmax(magnitude[1:len(magnitude)//2]) + 1
        features.append(float(freqs[dominant_freq_idx]))
        
        # Spectral centroid
        spectral_centroid = np.sum(freqs[:len(freqs)//2] * magnitude[:len(magnitude)//2]) / np.sum(magnitude[:len(magnitude)//2])
        features.append(float(spectral_centroid))
        
        # Spectral bandwidth
        spectral_bandwidth = np.sqrt(np.sum(((freqs[:len(freqs)//2] - spectral_centroid) ** 2) * magnitude[:len(magnitude)//2]) / np.sum(magnitude[:len(magnitude)//2]))
        features.append(float(spectral_bandwidth))
        
        # Spectral rolloff
        cumsum = np.cumsum(magnitude[:len(magnitude)//2])
        rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
        rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0.0
        features.append(float(rolloff))
        
        # Zero crossing rate
        zero_crossings = np.where(np.diff(np.sign(data)))[0]
        zcr = len(zero_crossings) / len(data)
        features.append(float(zcr))
        
        return features
    
    def _extract_complexity_features(self, data: np.ndarray) -> List[float]:
        """Extract complexity and chaos-based features"""
        if len(data) < 10:
            return [0.0] * 5
        
        features = []
        
        # Sample entropy (simplified)
        features.append(float(self._sample_entropy(data)))
        
        # Lempel-Ziv complexity (simplified)
        features.append(float(self._lempel_ziv_complexity(data)))
        
        # Fractal dimension (box counting method simplified)
        features.append(float(self._fractal_dimension(data)))
        
        # Hurst exponent (simplified R/S analysis)
        features.append(float(self._hurst_exponent(data)))
        
        # Lyapunov exponent (simplified)
        features.append(float(self._lyapunov_exponent(data)))
        
        return features
    
    def _benford_deviation(self, data: np.ndarray) -> float:
        """Calculate deviation from Benford's law"""
        if len(data) < 10:
            return 0.0
        
        # Get first digits
        first_digits = []
        for val in data:
            if val > 0:
                first_digit = int(str(float(val))[0])
                if 1 <= first_digit <= 9:
                    first_digits.append(first_digit)
        
        if len(first_digits) < 5:
            return 0.0
        
        # Expected Benford distribution
        benford_expected = [np.log10(1 + 1/d) for d in range(1, 10)]
        
        # Observed distribution
        observed_counts = np.bincount(first_digits, minlength=10)[1:10]
        observed_freq = observed_counts / np.sum(observed_counts)
        
        # Chi-square test
        chi_square = np.sum((observed_freq - benford_expected) ** 2 / benford_expected)
        return chi_square
    
    def _sample_entropy(self, data: np.ndarray, m: int = 2, r: float = None) -> float:
        """Calculate sample entropy"""
        if r is None:
            r = 0.2 * np.std(data)
        
        def _maxdist(xi, xj, m):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            N = len(data)
            patterns = np.array([data[i:i + m] for i in range(N - m + 1)])
            C = np.zeros(N - m + 1)
            
            for i in range(N - m + 1):
                template_i = patterns[i]
                for j in range(N - m + 1):
                    if _maxdist(template_i, patterns[j], m) <= r:
                        C[i] += 1.0
            
            phi = np.mean(np.log(C / (N - m + 1.0)))
            return phi
        
        try:
            return _phi(m) - _phi(m + 1)
        except:
            return 0.0
    
    def _lempel_ziv_complexity(self, data: np.ndarray) -> float:
        """Calculate Lempel-Ziv complexity"""
        # Convert to binary string (simplified)
        binary_data = ''.join(['1' if x > np.median(data) else '0' for x in data])
        
        i, k, l = 0, 1, 1
        c = 1
        n = len(binary_data)
        
        while k + l <= n:
            if binary_data[i + l - 1] == binary_data[k + l - 1]:
                l += 1
            else:
                if l > 1:
                    i = k
                k += 1
                l = 1
                c += 1
        
        if l > 1:
            c += 1
        
        return c / (n / np.log2(n))
    
    def _fractal_dimension(self, data: np.ndarray) -> float:
        """Calculate fractal dimension using box counting method"""
        # Simplified 1D box counting
        scales = np.logspace(0.01, 1, num=10)
        counts = []
        
        for scale in scales:
            box_size = int(len(data) * scale)
            if box_size < 1:
                box_size = 1
            
            boxes = len(data) // box_size
            count = 0
            
            for i in range(boxes):
                box_data = data[i*box_size:(i+1)*box_size]
                if len(box_data) > 0 and np.max(box_data) - np.min(box_data) > 1e-10:
                    count += 1
            
            counts.append(count)
        
        # Linear fit in log-log space
        log_scales = np.log(scales)
        log_counts = np.log(np.array(counts) + 1)
        
        try:
            slope = np.polyfit(log_scales, log_counts, 1)[0]
            return abs(slope)
        except:
            return 1.0
    
    def _hurst_exponent(self, data: np.ndarray) -> float:
        """Calculate Hurst exponent using R/S analysis"""
        if len(data) < 10:
            return 0.5
        
        lags = range(2, min(len(data)//4, 100))
        rs = []
        
        for lag in lags:
            # Divide series into blocks
            n_blocks = len(data) // lag
            rs_block = []
            
            for i in range(n_blocks):
                block = data[i*lag:(i+1)*lag]
                if len(block) < 2:
                    continue
                
                # Mean-adjusted series
                mean_block = np.mean(block)
                y = np.cumsum(block - mean_block)
                
                # Range and standard deviation
                R = np.max(y) - np.min(y)
                S = np.std(block)
                
                if S > 0:
                    rs_block.append(R/S)
            
            if len(rs_block) > 0:
                rs.append(np.mean(rs_block))
        
        if len(rs) < 2:
            return 0.5
        
        # Linear fit in log-log space
        try:
            log_lags = np.log(lags[:len(rs)])
            log_rs = np.log(rs)
            hurst = np.polyfit(log_lags, log_rs, 1)[0]
            return np.clip(hurst, 0.0, 1.0)
        except:
            return 0.5
    
    def _lyapunov_exponent(self, data: np.ndarray) -> float:
        """Calculate largest Lyapunov exponent (simplified)"""
        if len(data) < 20:
            return 0.0
        
        # Embedding dimension and delay
        m = 3
        tau = 1
        
        # Embed the time series
        embedded = []
        for i in range(len(data) - (m-1)*tau):
            embedded.append([data[i + j*tau] for j in range(m)])
        
        embedded = np.array(embedded)
        
        if len(embedded) < 10:
            return 0.0
        
        # Find nearest neighbors and track divergence
        divergences = []
        
        for i in range(len(embedded) - 10):
            # Find nearest neighbor
            distances = np.linalg.norm(embedded - embedded[i], axis=1)
            distances[i] = np.inf  # Exclude self
            
            nearest_idx = np.argmin(distances)
            
            # Track divergence for next few steps
            max_steps = min(10, len(embedded) - max(i, nearest_idx))
            
            for step in range(1, max_steps):
                if i + step < len(embedded) and nearest_idx + step < len(embedded):
                    dist = np.linalg.norm(embedded[i + step] - embedded[nearest_idx + step])
                    if dist > 0:
                        divergences.append(np.log(dist))
        
        if len(divergences) > 0:
            return np.mean(divergences)
        else:
            return 0.0
    
    def create_sklearn_autoencoder(self, input_dim: int) -> MLPClassifier:
        """
        Create an autoencoder-like model using sklearn MLPClassifier
        """
        # Use MLPRegressor for reconstruction task
        from sklearn.neural_network import MLPRegressor
        
        autoencoder = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16, 32, 64),
            activation='relu',
            solver='adam',
            max_iter=200,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        return autoencoder
    
    def train_models(self, features: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """
        Train multiple ML models for anomaly detection
        """
        print("🤖 Training machine learning models...")
        
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        self.scalers['standard'] = scaler
        
        results = {}
        
        # 1. Isolation Forest
        print("  Training Isolation Forest...")
        iso_forest = IsolationForest(
            contamination=0.1,
            random_state=self.random_state,
            n_estimators=100
        )
        iso_forest.fit(features_scaled)
        self.models['isolation_forest'] = iso_forest
        
        # Get anomaly scores
        iso_scores = iso_forest.decision_function(features_scaled)
        results['isolation_forest'] = {
            'anomaly_scores': iso_scores,
            'predictions': iso_forest.predict(features_scaled)
        }
        
        # 2. Autoencoder (using sklearn)
        print("  Training Autoencoder...")
        from sklearn.neural_network import MLPRegressor
        
        autoencoder = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16, 32, 64),
            activation='relu',
            solver='adam',
            max_iter=100,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1,
            verbose=False
        )
        
        # Train autoencoder to reconstruct input
        autoencoder.fit(features_scaled, features_scaled)
        self.models['autoencoder'] = autoencoder
        
        # Get reconstruction errors
        reconstructed = autoencoder.predict(features_scaled)
        reconstruction_errors = np.mean(np.square(features_scaled - reconstructed), axis=1)
        
        results['autoencoder'] = {
            'reconstruction_errors': reconstruction_errors,
            'threshold': np.percentile(reconstruction_errors, 90)
        }
        
        # 3. One-Class SVM (if dataset is small enough)
        if features_scaled.shape[0] < 10000:
            print("  Training One-Class SVM...")
            from sklearn.svm import OneClassSVM
            
            svm = OneClassSVM(gamma='scale', nu=0.1)
            svm.fit(features_scaled)
            self.models['one_class_svm'] = svm
            
            svm_scores = svm.decision_function(features_scaled)
            results['one_class_svm'] = {
                'anomaly_scores': svm_scores,
                'predictions': svm.predict(features_scaled)
            }
        
        # 4. DBSCAN clustering for outlier detection
        print("  Running DBSCAN clustering...")
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        cluster_labels = dbscan.fit_predict(features_scaled)
        
        # Points labeled as -1 are outliers
        outlier_mask = cluster_labels == -1
        results['dbscan'] = {
            'cluster_labels': cluster_labels,
            'outlier_mask': outlier_mask,
            'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        }
        
        print(f"  ✅ Trained {len(results)} ML models")
        return results
    
    def analyze_feature_importance(self, features: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """
        Analyze feature importance using various methods
        """
        print("📊 Analyzing feature importance...")
        
        # Normalize features
        features_scaled = self.scalers['standard'].transform(features)
        
        importance_scores = {}
        
        # 1. Variance-based importance
        variances = np.var(features_scaled, axis=0)
        for i, name in enumerate(feature_names):
            importance_scores[f"{name}_variance"] = float(variances[i])
        
        # 2. Principal Component Analysis
        pca = PCA(n_components=min(10, len(feature_names)))
        pca.fit(features_scaled)
        
        # Feature importance based on PCA loadings
        for i, name in enumerate(feature_names):
            # Weight by explained variance of first few components
            importance = np.sum(np.abs(pca.components_[:3, i]) * pca.explained_variance_ratio_[:3])
            importance_scores[f"{name}_pca"] = float(importance)
        
        # 3. Reconstruction error contribution (if autoencoder is trained)
        if 'autoencoder' in self.models:
            # Feature importance based on reconstruction error sensitivity
            base_reconstruction = self.models['autoencoder'].predict(features_scaled)
            base_error = np.mean(np.square(features_scaled - base_reconstruction))
            
            for i, name in enumerate(feature_names):
                # Perturb feature and measure reconstruction error change
                perturbed_features = features_scaled.copy()
                perturbed_features[:, i] += np.std(perturbed_features[:, i]) * 0.1
                
                perturbed_reconstruction = self.models['autoencoder'].predict(perturbed_features)
                perturbed_error = np.mean(np.square(perturbed_features - perturbed_reconstruction))
                
                sensitivity = abs(perturbed_error - base_error) / (base_error + 1e-10)
                importance_scores[f"{name}_sensitivity"] = float(sensitivity)
        
        self.feature_importance = importance_scores
        print(f"  ✅ Analyzed importance for {len(feature_names)} features")
        return importance_scores
    
    def comprehensive_anomaly_analysis(self, datasets: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Run comprehensive ML-based anomaly detection analysis
        """
        print("\n🤖 MACHINE LEARNING ANOMALY DETECTION")
        print("-" * 40)
        
        # Prepare features
        features, feature_names = self.prepare_features(datasets)
        
        print(f"📊 Extracted {features.shape[1]} features from {len(datasets)} datasets")
        
        # Train models
        ml_results = self.train_models(features, feature_names)
        
        # Analyze feature importance
        importance = self.analyze_feature_importance(features, feature_names)
        
        # Calculate ensemble scores
        ensemble_scores = self._calculate_ensemble_scores(ml_results)
        
        # Generate interpretations
        interpretations = self._generate_interpretations(ml_results, ensemble_scores, feature_names)
        
        return {
            'features': features,
            'feature_names': feature_names,
            'ml_results': ml_results,
            'feature_importance': importance,
            'ensemble_scores': ensemble_scores,
            'interpretations': interpretations,
            'summary': {
                'total_features': features.shape[1],
                'datasets_analyzed': len(datasets),
                'models_trained': len(ml_results),
                'anomaly_probability': ensemble_scores.get('mean_anomaly_probability', 0.0)
            }
        }
    
    def _calculate_ensemble_scores(self, ml_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate ensemble anomaly scores"""
        
        anomaly_scores = []
        
        # Collect scores from different models
        if 'isolation_forest' in ml_results:
            # Normalize isolation forest scores to [0, 1]
            iso_scores = ml_results['isolation_forest']['anomaly_scores']
            normalized_iso = (iso_scores - np.min(iso_scores)) / (np.max(iso_scores) - np.min(iso_scores) + 1e-10)
            anomaly_scores.append(normalized_iso)
        
        if 'autoencoder' in ml_results:
            # Normalize reconstruction errors to [0, 1]
            recon_errors = ml_results['autoencoder']['reconstruction_errors']
            normalized_recon = (recon_errors - np.min(recon_errors)) / (np.max(recon_errors) - np.min(recon_errors) + 1e-10)
            anomaly_scores.append(normalized_recon)
        
        if 'one_class_svm' in ml_results:
            # Normalize SVM scores to [0, 1]
            svm_scores = ml_results['one_class_svm']['anomaly_scores']
            normalized_svm = (svm_scores - np.min(svm_scores)) / (np.max(svm_scores) - np.min(svm_scores) + 1e-10)
            anomaly_scores.append(normalized_svm)
        
        if len(anomaly_scores) > 0:
            # Ensemble averaging
            ensemble_score = np.mean(anomaly_scores, axis=0)
            
            return {
                'mean_anomaly_probability': float(np.mean(ensemble_score)),
                'max_anomaly_probability': float(np.max(ensemble_score)),
                'std_anomaly_probability': float(np.std(ensemble_score)),
                'ensemble_scores': ensemble_score.tolist()
            }
        else:
            return {
                'mean_anomaly_probability': 0.0,
                'max_anomaly_probability': 0.0,
                'std_anomaly_probability': 0.0,
                'ensemble_scores': []
            }
    
    def _generate_interpretations(self, ml_results: Dict[str, Any], ensemble_scores: Dict[str, float], 
                                feature_names: List[str]) -> Dict[str, str]:
        """Generate human-readable interpretations"""
        
        interpretations = {}
        
        # Overall assessment
        anomaly_prob = ensemble_scores.get('mean_anomaly_probability', 0.0)
        
        if anomaly_prob > 0.8:
            overall = "HIGH anomaly probability - Strong computational signatures detected"
        elif anomaly_prob > 0.6:
            overall = "MEDIUM-HIGH anomaly probability - Notable computational patterns"
        elif anomaly_prob > 0.4:
            overall = "MEDIUM anomaly probability - Some simulation-like features"
        elif anomaly_prob > 0.2:
            overall = "LOW-MEDIUM anomaly probability - Weak computational signatures"
        else:
            overall = "LOW anomaly probability - Data appears naturally distributed"
        
        interpretations['overall_assessment'] = overall
        
        # Model-specific interpretations
        if 'isolation_forest' in ml_results:
            iso_outliers = np.sum(ml_results['isolation_forest']['predictions'] == -1)
            interpretations['isolation_forest'] = f"Detected {iso_outliers} outlier patterns in feature space"
        
        if 'autoencoder' in ml_results:
            threshold = ml_results['autoencoder']['threshold']
            high_error_count = np.sum(ml_results['autoencoder']['reconstruction_errors'] > threshold)
            interpretations['autoencoder'] = f"Found {high_error_count} data points with high reconstruction errors"
        
        if 'dbscan' in ml_results:
            n_clusters = ml_results['dbscan']['n_clusters']
            outliers = np.sum(ml_results['dbscan']['outlier_mask'])
            interpretations['dbscan'] = f"Identified {n_clusters} clusters with {outliers} outlier points"
        
        return interpretations
    
    def create_ml_visualizations(self, analysis_results: Dict[str, Any], output_dir: str) -> str:
        """Create comprehensive ML analysis visualizations"""
        
        import os
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Feature importance heatmap
        plt.subplot(3, 3, 1)
        importance = analysis_results['feature_importance']
        feature_names = analysis_results['feature_names'][:20]  # Top 20 features
        
        # Get top features by PCA importance
        pca_importance = {k: v for k, v in importance.items() if k.endswith('_pca')}
        if pca_importance:
            sorted_features = sorted(pca_importance.items(), key=lambda x: x[1], reverse=True)[:20]
            feature_vals = [item[1] for item in sorted_features]
            feature_labels = [item[0].replace('_pca', '') for item in sorted_features]
            
            plt.barh(range(len(feature_vals)), feature_vals)
            plt.yticks(range(len(feature_labels)), feature_labels, fontsize=8)
            plt.xlabel('PCA Importance Score')
            plt.title('Top Feature Importance (PCA-based)')
            plt.grid(True, alpha=0.3)
        
        # 2. Anomaly score distribution
        plt.subplot(3, 3, 2)
        ensemble_scores = analysis_results['ensemble_scores']['ensemble_scores']
        if ensemble_scores:
            plt.hist(ensemble_scores, bins=30, alpha=0.7, color='red', edgecolor='black')
            plt.axvline(np.mean(ensemble_scores), color='darkred', linestyle='--', 
                       label=f'Mean: {np.mean(ensemble_scores):.3f}')
            plt.xlabel('Anomaly Probability')
            plt.ylabel('Frequency')
            plt.title('Ensemble Anomaly Score Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 3. Model comparison
        plt.subplot(3, 3, 3)
        ml_results = analysis_results['ml_results']
        model_scores = []
        model_names = []
        
        for model_name, results in ml_results.items():
            if 'anomaly_scores' in results:
                scores = results['anomaly_scores']
                model_scores.append(np.mean(scores))
                model_names.append(model_name.replace('_', ' ').title())
            elif 'reconstruction_errors' in results:
                errors = results['reconstruction_errors']
                model_scores.append(np.mean(errors))
                model_names.append(model_name.replace('_', ' ').title())
        
        if model_scores:
            plt.bar(model_names, model_scores, color=['blue', 'green', 'orange', 'purple'][:len(model_scores)])
            plt.xlabel('ML Model')
            plt.ylabel('Mean Anomaly Score')
            plt.title('Model Comparison')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        
        # 4. Autoencoder training convergence (sklearn doesn't have history)
        if 'autoencoder' in ml_results:
            plt.subplot(3, 3, 4)
            recon_errors = ml_results['autoencoder']['reconstruction_errors']
            plt.plot(recon_errors[:100], label='Reconstruction Errors (first 100)', alpha=0.7)
            plt.xlabel('Sample Index')
            plt.ylabel('Reconstruction Error')
            plt.title('Autoencoder Reconstruction Errors')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 5. Clustering visualization (if DBSCAN was used)
        if 'dbscan' in ml_results:
            plt.subplot(3, 3, 5)
            cluster_labels = ml_results['dbscan']['cluster_labels']
            features_2d = analysis_results['features']
            
            # Use PCA for 2D visualization if needed
            if features_2d.shape[1] > 2:
                pca = PCA(n_components=2)
                features_2d = pca.fit_transform(features_2d)
            
            if features_2d.shape[0] > 1:
                scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                                    c=cluster_labels, cmap='viridis', alpha=0.7)
                plt.colorbar(scatter)
                plt.xlabel('PCA Component 1')
                plt.ylabel('PCA Component 2')
                plt.title('DBSCAN Clustering Results')
        
        # 6. Reconstruction error distribution (if autoencoder was used)
        if 'autoencoder' in ml_results:
            plt.subplot(3, 3, 6)
            recon_errors = ml_results['autoencoder']['reconstruction_errors']
            threshold = ml_results['autoencoder']['threshold']
            
            plt.hist(recon_errors, bins=30, alpha=0.7, color='orange', edgecolor='black')
            plt.axvline(threshold, color='red', linestyle='--', 
                       label=f'Threshold: {threshold:.3f}')
            plt.axvline(np.mean(recon_errors), color='darkorange', linestyle='--',
                       label=f'Mean: {np.mean(recon_errors):.3f}')
            plt.xlabel('Reconstruction Error')
            plt.ylabel('Frequency')
            plt.title('Autoencoder Reconstruction Errors')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 7. Feature correlation matrix (subset)
        plt.subplot(3, 3, 7)
        features = analysis_results['features']
        if features.shape[1] > 1:
            # Use subset of features for visualization
            n_features = min(10, features.shape[1])
            subset_features = features[:, :n_features]
            corr_matrix = np.corrcoef(subset_features.T)
            
            im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar(im)
            plt.title('Feature Correlation Matrix (Subset)')
            plt.xlabel('Feature Index')
            plt.ylabel('Feature Index')
        
        # 8. Summary statistics
        plt.subplot(3, 3, 8)
        plt.axis('off')
        summary = analysis_results['summary']
        interpretations = analysis_results['interpretations']
        
        summary_text = f"""
ML ANOMALY DETECTION SUMMARY

Datasets Analyzed: {summary['datasets_analyzed']}
Features Extracted: {summary['total_features']}
Models Trained: {summary['models_trained']}

Anomaly Probability: {summary['anomaly_probability']:.3f}

Overall Assessment:
{interpretations.get('overall_assessment', 'No assessment available')}

Key Findings:
• {interpretations.get('isolation_forest', 'Isolation Forest: N/A')}
• {interpretations.get('autoencoder', 'Autoencoder: N/A')}
• {interpretations.get('dbscan', 'DBSCAN: N/A')}
        """
        
        plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        # 9. Ensemble score trend
        plt.subplot(3, 3, 9)
        if ensemble_scores:
            plt.plot(ensemble_scores, 'b-', alpha=0.7, label='Ensemble Score')
            plt.axhline(np.mean(ensemble_scores), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(ensemble_scores):.3f}')
            plt.xlabel('Data Point Index')
            plt.ylabel('Anomaly Probability')
            plt.title('Ensemble Anomaly Scores')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        output_file = output_path / "ml_anomaly_detection_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  📊 ML analysis visualization saved: {output_file}")
        return str(output_file)


def test_ml_anomaly_detection():
    """Test the ML anomaly detection system"""
    
    print("Testing ML Anomaly Detection System...")
    
    # Create test datasets
    np.random.seed(42)
    
    test_datasets = {
        'normal_data': np.random.normal(0, 1, 1000),
        'uniform_data': np.random.uniform(-2, 2, 1000),
        'exponential_data': np.random.exponential(1, 1000),
        'quantized_data': np.round(np.random.normal(0, 1, 1000) * 4) / 4,  # Quantized to 0.25 intervals
        'periodic_data': np.sin(np.linspace(0, 10*np.pi, 1000)) + np.random.normal(0, 0.1, 1000)
    }
    
    # Initialize detector
    detector = SimulationAnomalyDetector()
    
    # Run comprehensive analysis
    results = detector.comprehensive_anomaly_analysis(test_datasets)
    
    # Print results
    print(f"\n🎯 ML ANOMALY DETECTION RESULTS:")
    print(f"  Features extracted: {results['summary']['total_features']}")
    print(f"  Datasets analyzed: {results['summary']['datasets_analyzed']}")
    print(f"  Models trained: {results['summary']['models_trained']}")
    print(f"  Anomaly probability: {results['summary']['anomaly_probability']:.4f}")
    print(f"  Overall assessment: {results['interpretations']['overall_assessment']}")
    
    return results


if __name__ == "__main__":
    test_ml_anomaly_detection()
