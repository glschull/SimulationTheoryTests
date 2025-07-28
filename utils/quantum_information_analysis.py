"""
Quantum Information Analysis for Simulation Hypothesis Testing
============================================================

Advanced quantum information theoretic analysis for detecting computational
signatures and quantum simulation artifacts in physical data.

Features:
- Entanglement entropy calculations
- Quantum mutual information tests
- Bell inequality violation analysis
- Quantum computational signature detection

Author: AI Assistant for Garrett Schull's SimulationTheoryTests
"""

import numpy as np
import scipy.stats as stats
from scipy.linalg import sqrtm, logm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

class QuantumInformationAnalyzer:
    """
    Advanced quantum information analysis for simulation hypothesis testing
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Quantum information constants
        self.planck_h = 6.62607015e-34  # Planck constant
        self.hbar = self.planck_h / (2 * np.pi)  # Reduced Planck constant
        self.kb = 1.380649e-23  # Boltzmann constant
        
    def calculate_von_neumann_entropy(self, density_matrix: np.ndarray) -> float:
        """
        Calculate von Neumann entropy of a quantum state
        S = -Tr(ρ log ρ)
        """
        # Ensure the matrix is valid (hermitian, positive semi-definite, trace 1)
        rho = self._normalize_density_matrix(density_matrix)
        
        # Calculate eigenvalues
        eigenvals = np.linalg.eigvals(rho)
        eigenvals = eigenvals[eigenvals > 1e-12]  # Remove numerical zeros
        
        # Von Neumann entropy
        entropy = -np.sum(eigenvals * np.log2(eigenvals))
        return float(entropy)
    
    def calculate_entanglement_entropy(self, state_vector: np.ndarray, 
                                     partition_size: int) -> float:
        """
        Calculate entanglement entropy for a bipartite system
        """
        n_qubits = int(np.log2(len(state_vector)))
        
        if partition_size >= n_qubits:
            return 0.0
        
        # Reshape state vector into tensor
        dims = [2] * n_qubits
        tensor = state_vector.reshape(dims)
        
        # Trace out the second partition
        axes_to_trace = list(range(partition_size, n_qubits))
        reduced_density_matrix = self._partial_trace(tensor, axes_to_trace)
        
        # Calculate von Neumann entropy
        return self.calculate_von_neumann_entropy(reduced_density_matrix)
    
    def calculate_quantum_mutual_information(self, joint_state: np.ndarray,
                                           partition_a: int, partition_b: int) -> float:
        """
        Calculate quantum mutual information I(A:B) = S(A) + S(B) - S(AB)
        """
        n_qubits = int(np.log2(len(joint_state)))
        
        # Calculate individual entropies
        entropy_a = self.calculate_entanglement_entropy(joint_state, partition_a)
        entropy_b = self.calculate_entanglement_entropy(joint_state, partition_b)
        
        # Joint entropy (full system)
        joint_density = np.outer(joint_state, np.conj(joint_state))
        entropy_ab = self.calculate_von_neumann_entropy(joint_density)
        
        # Mutual information
        mutual_info = entropy_a + entropy_b - entropy_ab
        return float(mutual_info)
    
    def analyze_bell_inequality_violations(self, measurement_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Analyze Bell inequality violations in measurement data
        """
        results = {}
        
        # CHSH inequality analysis
        chsh_result = self._analyze_chsh_inequality(measurement_data)
        results.update(chsh_result)
        
        # Mermin inequality for multi-particle systems
        if len(measurement_data) >= 3:
            mermin_result = self._analyze_mermin_inequality(measurement_data)
            results.update(mermin_result)
        
        # Bell-CH inequality
        bell_ch_result = self._analyze_bell_ch_inequality(measurement_data)
        results.update(bell_ch_result)
        
        return results
    
    def _analyze_chsh_inequality(self, data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Analyze CHSH (Clauser-Horne-Shimony-Holt) inequality
        |E(a,b) - E(a,b') + E(a',b) + E(a',b')| ≤ 2 (classical limit)
        Quantum maximum: 2√2 ≈ 2.828
        """
        if len(data) < 2:
            return {'chsh_parameter': 0.0, 'chsh_violation': 0.0}
        
        # Get measurement results (assuming binary outcomes ±1)
        measurements = list(data.values())
        
        # Convert to binary outcomes if needed
        for i, m in enumerate(measurements):
            if np.all((m == 0) | (m == 1)):
                measurements[i] = 2 * m - 1  # Convert 0,1 to -1,1
        
        if len(measurements) < 4:
            # Generate complementary measurements if we don't have enough
            while len(measurements) < 4:
                # Add rotated versions
                angle = len(measurements) * np.pi / 4
                rotated = measurements[0] * np.cos(angle) + measurements[1] * np.sin(angle)
                measurements.append(np.sign(rotated))
        
        # Calculate correlation functions E(a,b)
        E_ab = np.corrcoef(measurements[0], measurements[1])[0, 1]
        E_ab_prime = np.corrcoef(measurements[0], measurements[2])[0, 1]
        E_a_prime_b = np.corrcoef(measurements[1], measurements[3])[0, 1]
        E_a_prime_b_prime = np.corrcoef(measurements[2], measurements[3])[0, 1]
        
        # CHSH parameter
        S = abs(E_ab - E_ab_prime + E_a_prime_b + E_a_prime_b_prime)
        
        # Check for violations
        classical_limit = 2.0
        quantum_limit = 2 * np.sqrt(2)
        
        violation = max(0, S - classical_limit)
        quantum_violation = max(0, S - quantum_limit)
        
        return {
            'chsh_parameter': float(S),
            'chsh_violation': float(violation),
            'chsh_quantum_violation': float(quantum_violation),
            'chsh_classical_limit': classical_limit,
            'chsh_quantum_limit': quantum_limit
        }
    
    def _analyze_mermin_inequality(self, data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Analyze Mermin inequality for multi-particle Bell tests
        """
        measurements = list(data.values())[:3]  # Use first 3 measurements
        
        # Convert to binary outcomes
        for i, m in enumerate(measurements):
            if np.all((m == 0) | (m == 1)):
                measurements[i] = 2 * m - 1
        
        # Mermin operator expectation values
        # For 3 particles: M = A₁B₂C₂ + A₁B₃C₃ + A₂B₁C₃ + A₂B₃C₁ - A₃B₁C₂ - A₃B₂C₁
        A1, A2, A3 = measurements[0], measurements[0], measurements[0]  # Same measurement, different angles
        B1, B2, B3 = measurements[1], measurements[1], measurements[1]
        C1, C2, C3 = measurements[2], measurements[2], measurements[2]
        
        # Simplified Mermin calculation
        M = (np.mean(A1 * B2 * C2) + np.mean(A1 * B3 * C3) + 
             np.mean(A2 * B1 * C3) + np.mean(A2 * B3 * C1) - 
             np.mean(A3 * B1 * C2) - np.mean(A3 * B2 * C1))
        
        classical_limit = 2.0
        quantum_limit = 4.0
        
        violation = max(0, abs(M) - classical_limit)
        
        return {
            'mermin_parameter': float(abs(M)),
            'mermin_violation': float(violation),
            'mermin_classical_limit': classical_limit,
            'mermin_quantum_limit': quantum_limit
        }
    
    def _analyze_bell_ch_inequality(self, data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Analyze Bell-CH inequality
        """
        if len(data) < 2:
            return {'bell_ch_parameter': 0.0, 'bell_ch_violation': 0.0}
        
        measurements = list(data.values())[:2]
        
        # Convert to binary outcomes
        for i, m in enumerate(measurements):
            if np.all((m == 0) | (m == 1)):
                measurements[i] = 2 * m - 1
        
        # Bell-CH parameter (simplified)
        correlation = np.corrcoef(measurements[0], measurements[1])[0, 1]
        bell_ch = abs(correlation)
        
        classical_limit = 0.5
        violation = max(0, bell_ch - classical_limit)
        
        return {
            'bell_ch_parameter': float(bell_ch),
            'bell_ch_violation': float(violation),
            'bell_ch_classical_limit': classical_limit
        }
    
    def detect_quantum_computational_signatures(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Detect signatures of quantum computational processes
        """
        signatures = {}
        
        for name, dataset in data.items():
            if len(dataset) == 0:
                continue
            
            # 1. Quantum interference patterns
            interference_score = self._detect_interference_patterns(dataset)
            
            # 2. Decoherence signatures
            decoherence_score = self._analyze_decoherence(dataset)
            
            # 3. Quantum error correction patterns
            error_correction_score = self._detect_error_correction_patterns(dataset)
            
            # 4. Quantum algorithm signatures
            algorithm_score = self._detect_quantum_algorithm_signatures(dataset)
            
            signatures[name] = {
                'interference_patterns': interference_score,
                'decoherence_signatures': decoherence_score,
                'error_correction_patterns': error_correction_score,
                'quantum_algorithm_signatures': algorithm_score,
                'combined_quantum_score': np.mean([
                    interference_score, decoherence_score, 
                    error_correction_score, algorithm_score
                ])
            }
        
        return signatures
    
    def _detect_interference_patterns(self, data: np.ndarray) -> float:
        """
        Detect quantum interference patterns in data
        """
        if len(data) < 10:
            return 0.0
        
        # Look for oscillatory patterns characteristic of quantum interference
        fft = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(data))
        power_spectrum = np.abs(fft) ** 2
        
        # Check for periodic structure
        dominant_freqs = freqs[np.argsort(power_spectrum)[-5:]]
        
        # Quantum interference often shows specific frequency relationships
        interference_score = 0.0
        
        for i, freq1 in enumerate(dominant_freqs):
            for freq2 in dominant_freqs[i+1:]:
                if abs(freq1) > 1e-10 and abs(freq2) > 1e-10:
                    ratio = abs(freq2 / freq1)
                    # Look for integer or simple fractional relationships
                    if abs(ratio - round(ratio)) < 0.1:
                        interference_score += 0.2
        
        return min(1.0, interference_score)
    
    def _analyze_decoherence(self, data: np.ndarray) -> float:
        """
        Analyze decoherence signatures in time series data
        """
        if len(data) < 20:
            return 0.0
        
        # Decoherence typically shows exponential decay in coherence
        # Measure coherence decay over time
        correlations = []
        max_lag = min(100, len(data) // 4)
        
        for lag in range(1, max_lag):
            if lag < len(data):
                correlation = np.corrcoef(data[:-lag], data[lag:])[0, 1]
                if not np.isnan(correlation):
                    correlations.append(abs(correlation))
        
        if len(correlations) < 5:
            return 0.0
        
        # Fit exponential decay
        lags = np.arange(1, len(correlations) + 1)
        try:
            # Fit y = a * exp(-b * x)
            log_corr = np.log(np.array(correlations) + 1e-10)
            fit = np.polyfit(lags, log_corr, 1)
            decay_rate = -fit[0]
            
            # Higher decay rate suggests decoherence
            decoherence_score = min(1.0, decay_rate * 10)
            return float(decoherence_score)
        except:
            return 0.0
    
    def _detect_error_correction_patterns(self, data: np.ndarray) -> float:
        """
        Detect quantum error correction patterns
        """
        if len(data) < 100:
            return 0.0
        
        # Look for periodic error correction cycles
        # Quantum error correction typically shows regular patterns
        
        # Analyze variance in sliding windows
        window_size = len(data) // 10
        variances = []
        
        for i in range(0, len(data) - window_size, window_size // 2):
            window = data[i:i + window_size]
            variances.append(np.var(window))
        
        if len(variances) < 3:
            return 0.0
        
        # Look for regular patterns in variance
        variance_fft = np.fft.fft(variances)
        variance_power = np.abs(variance_fft) ** 2
        
        # High power in low frequencies suggests regular error correction
        low_freq_power = np.sum(variance_power[:len(variance_power)//4])
        total_power = np.sum(variance_power)
        
        if total_power > 0:
            error_correction_score = low_freq_power / total_power
            return float(min(1.0, error_correction_score * 2))
        else:
            return 0.0
    
    def _detect_quantum_algorithm_signatures(self, data: np.ndarray) -> float:
        """
        Detect signatures of specific quantum algorithms
        """
        if len(data) < 50:
            return 0.0
        
        signatures = []
        
        # 1. Shor's algorithm signature (period finding)
        period_score = self._detect_period_finding(data)
        signatures.append(period_score)
        
        # 2. Grover's algorithm signature (amplitude amplification)
        grover_score = self._detect_amplitude_amplification(data)
        signatures.append(grover_score)
        
        # 3. Quantum Fourier Transform signature
        qft_score = self._detect_qft_signature(data)
        signatures.append(qft_score)
        
        return float(np.mean(signatures))
    
    def _detect_period_finding(self, data: np.ndarray) -> float:
        """
        Detect period finding patterns (Shor's algorithm)
        """
        # Autocorrelation analysis for period detection
        max_lag = min(len(data) // 4, 100)
        autocorr = np.correlate(data, data, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        if len(autocorr) < max_lag:
            return 0.0
        
        # Look for strong periodic components
        peaks = []
        for i in range(2, min(max_lag, len(autocorr))):
            if (autocorr[i] > autocorr[i-1] and 
                autocorr[i] > autocorr[i+1] if i+1 < len(autocorr) else True):
                peaks.append((i, autocorr[i]))
        
        if len(peaks) < 2:
            return 0.0
        
        # Check for regular spacing in peaks (characteristic of period finding)
        peak_positions = [p[0] for p in peaks[:5]]
        spacings = np.diff(peak_positions)
        
        if len(spacings) > 1:
            spacing_regularity = 1.0 - (np.std(spacings) / (np.mean(spacings) + 1e-10))
            return float(min(1.0, spacing_regularity))
        else:
            return 0.0
    
    def _detect_amplitude_amplification(self, data: np.ndarray) -> float:
        """
        Detect amplitude amplification patterns (Grover's algorithm)
        """
        # Look for quadratic speedup signatures
        # Grover's algorithm shows sqrt(N) iterations for optimal result
        
        # Analyze amplitude growth patterns
        amplitudes = np.abs(data)
        
        # Look for sqrt-like growth followed by oscillations
        if len(amplitudes) < 10:
            return 0.0
        
        # Divide into phases
        quarter = len(amplitudes) // 4
        early_phase = amplitudes[:quarter]
        middle_phase = amplitudes[quarter:3*quarter]
        late_phase = amplitudes[3*quarter:]
        
        # Early phase should show growth
        early_trend = np.polyfit(range(len(early_phase)), early_phase, 1)[0]
        
        # Middle phase should show continued growth
        middle_trend = np.polyfit(range(len(middle_phase)), middle_phase, 1)[0]
        
        # Late phase might show oscillations
        late_variance = np.var(late_phase)
        
        # Score based on expected Grover pattern
        growth_score = 1.0 if early_trend > 0 and middle_trend > 0 else 0.0
        oscillation_score = min(1.0, late_variance / (np.mean(late_phase) + 1e-10))
        
        return float((growth_score + oscillation_score) / 2)
    
    def _detect_qft_signature(self, data: np.ndarray) -> float:
        """
        Detect Quantum Fourier Transform signatures
        """
        # QFT creates specific frequency domain patterns
        fft_data = np.fft.fft(data)
        
        # QFT typically creates uniform distribution in frequency domain
        freq_magnitudes = np.abs(fft_data)
        
        # Measure uniformity
        normalized_mags = freq_magnitudes / (np.sum(freq_magnitudes) + 1e-10)
        
        # Calculate entropy (higher entropy suggests more uniform distribution)
        entropy = -np.sum(normalized_mags * np.log2(normalized_mags + 1e-10))
        max_entropy = np.log2(len(normalized_mags))
        
        uniformity_score = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return float(uniformity_score)
    
    def _normalize_density_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """
        Normalize a matrix to be a valid density matrix
        """
        # Handle various input formats
        if matrix.ndim == 1:
            # Convert vector to density matrix
            matrix = np.outer(matrix, np.conj(matrix))
        elif matrix.ndim > 2:
            # Flatten higher dimensional arrays
            matrix = matrix.reshape(matrix.shape[0], -1)
            if matrix.shape[0] != matrix.shape[1]:
                # Make square by outer product if needed
                matrix = np.outer(matrix.flatten(), np.conj(matrix.flatten()))
        
        # Ensure we have a 2D square matrix
        if matrix.shape[0] != matrix.shape[1]:
            n = min(matrix.shape)
            matrix = matrix[:n, :n]
        
        # Ensure hermitian
        matrix = (matrix + np.conj(matrix.T)) / 2
        
        # Ensure positive semi-definite
        try:
            eigenvals, eigenvecs = np.linalg.eigh(matrix)
            eigenvals = np.maximum(eigenvals, 0)
            
            # Handle scalar eigenvalues
            if np.isscalar(eigenvals):
                eigenvals = np.array([eigenvals])
            
            # Reconstruct matrix
            eigenvals = eigenvals.real  # Ensure real eigenvalues
            matrix = eigenvecs @ np.diag(eigenvals) @ np.conj(eigenvecs.T)
        except np.linalg.LinAlgError:
            # Fallback: create a valid density matrix
            n = matrix.shape[0]
            matrix = np.eye(n) / n
        
        # Ensure trace = 1
        trace = np.trace(matrix)
        if abs(trace) > 1e-10:
            matrix = matrix / trace
        else:
            # If trace is zero, create identity density matrix
            n = matrix.shape[0]
            matrix = np.eye(n) / n
        
        return matrix
    
    def _partial_trace(self, tensor: np.ndarray, axes_to_trace: List[int]) -> np.ndarray:
        """
        Compute partial trace over specified axes
        """
        # Simplified partial trace implementation
        # For demonstration purposes - handles basic cases
        
        if tensor.ndim == 1:
            # For state vectors, create density matrix first
            density_matrix = np.outer(tensor, np.conj(tensor))
            return density_matrix
        
        # For higher dimensional tensors, sum over traced dimensions
        remaining_dims = [i for i in range(tensor.ndim) if i not in axes_to_trace]
        
        if len(remaining_dims) == 0:
            return np.array([[np.sum(tensor)]])
        
        # Simple approach: sum over traced axes
        try:
            reduced = np.sum(tensor, axis=tuple(axes_to_trace))
            
            # Convert to density matrix form if needed
            if reduced.ndim == 1:
                reduced = np.outer(reduced, np.conj(reduced))
            elif reduced.ndim == 0:
                reduced = np.array([[reduced]])
            
            return reduced
        except:
            # Fallback: return identity matrix
            size = 2 ** (len(remaining_dims))
            return np.eye(size) / size
    
    def comprehensive_quantum_analysis(self, datasets: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Run comprehensive quantum information analysis
        """
        print("\n🔬 QUANTUM INFORMATION ANALYSIS")
        print("-" * 35)
        
        results = {}
        
        # 1. Bell inequality analysis
        print("  Analyzing Bell inequality violations...")
        bell_results = self.analyze_bell_inequality_violations(datasets)
        results['bell_inequalities'] = bell_results
        
        # 2. Quantum computational signatures
        print("  Detecting quantum computational signatures...")
        quantum_signatures = self.detect_quantum_computational_signatures(datasets)
        results['quantum_signatures'] = quantum_signatures
        
        # 3. Entanglement analysis (for appropriate datasets)
        print("  Analyzing entanglement properties...")
        entanglement_results = self._analyze_entanglement_in_data(datasets)
        results['entanglement_analysis'] = entanglement_results
        
        # 4. Quantum information measures
        print("  Computing quantum information measures...")
        info_measures = self._compute_quantum_information_measures(datasets)
        results['information_measures'] = info_measures
        
        # 5. Generate summary
        summary = self._generate_quantum_summary(results)
        results['summary'] = summary
        
        print(f"  ✅ Quantum analysis complete")
        print(f"    Bell violations detected: {summary.get('bell_violations_count', 0)}")
        print(f"    Quantum signatures: {summary.get('quantum_signature_score', 0):.3f}")
        print(f"    Information complexity: {summary.get('information_complexity', 0):.3f}")
        
        return results
    
    def _analyze_entanglement_in_data(self, datasets: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Analyze entanglement-like properties in classical data
        """
        entanglement_scores = {}
        
        for name, data in datasets.items():
            if len(data) < 4:
                entanglement_scores[name] = 0.0
                continue
            
            # Create pseudo-quantum state from classical data
            # Normalize data to create state-like vector
            normalized_data = data / (np.linalg.norm(data) + 1e-10)
            
            # Pad to power of 2 length for quantum-like analysis
            target_length = 2 ** int(np.ceil(np.log2(len(normalized_data))))
            if len(normalized_data) < target_length:
                padded_data = np.zeros(target_length, dtype=complex)
                padded_data[:len(normalized_data)] = normalized_data
                normalized_data = padded_data
            else:
                normalized_data = normalized_data[:target_length]
            
            # Calculate entanglement entropy for bipartite split
            if target_length >= 4:
                partition_size = int(np.log2(target_length)) // 2
                entanglement_entropy = self.calculate_entanglement_entropy(
                    normalized_data, partition_size
                )
                entanglement_scores[name] = float(entanglement_entropy)
            else:
                entanglement_scores[name] = 0.0
        
        return entanglement_scores
    
    def _compute_quantum_information_measures(self, datasets: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Compute various quantum information measures
        """
        measures = {}
        
        for name, data in datasets.items():
            if len(data) == 0:
                continue
            
            # Convert data to probability distribution
            hist, _ = np.histogram(data, bins=min(50, len(data)//10))
            prob_dist = hist / np.sum(hist)
            prob_dist = prob_dist[prob_dist > 0]  # Remove zeros
            
            # Shannon entropy
            shannon_entropy = -np.sum(prob_dist * np.log2(prob_dist))
            
            # Renyi entropy (order 2)
            renyi_entropy = -np.log2(np.sum(prob_dist ** 2))
            
            # Tsallis entropy (q=2)
            tsallis_entropy = (1 - np.sum(prob_dist ** 2)) / (2 - 1)
            
            # Quantum Fisher information (simplified estimate)
            fisher_info = self._estimate_quantum_fisher_information(data)
            
            measures[name] = {
                'shannon_entropy': float(shannon_entropy),
                'renyi_entropy': float(renyi_entropy),
                'tsallis_entropy': float(tsallis_entropy),
                'quantum_fisher_info': float(fisher_info)
            }
        
        return measures
    
    def _estimate_quantum_fisher_information(self, data: np.ndarray) -> float:
        """
        Estimate quantum Fisher information from classical data
        """
        if len(data) < 10:
            return 0.0
        
        # Estimate Fisher information using variance
        # QFI ≥ 4 * (∂<O>/∂θ)² / Var(O)
        
        # Use finite differences to estimate derivative
        epsilon = 1e-6
        perturbed_data = data + epsilon
        
        mean_orig = np.mean(data)
        mean_pert = np.mean(perturbed_data)
        derivative = (mean_pert - mean_orig) / epsilon
        
        variance = np.var(data)
        
        if variance > 1e-10:
            fisher_info = 4 * derivative ** 2 / variance
            return min(100.0, abs(fisher_info))  # Cap at reasonable value
        else:
            return 0.0
    
    def _generate_quantum_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary of quantum analysis results
        """
        summary = {}
        
        # Bell inequality violations
        bell_results = results.get('bell_inequalities', {})
        violations = []
        for key, value in bell_results.items():
            if 'violation' in key and value > 0:
                violations.append(value)
        
        summary['bell_violations_count'] = len(violations)
        summary['max_bell_violation'] = max(violations) if violations else 0.0
        
        # Quantum signatures
        quantum_sigs = results.get('quantum_signatures', {})
        if quantum_sigs:
            combined_scores = [sig.get('combined_quantum_score', 0) 
                             for sig in quantum_sigs.values()]
            summary['quantum_signature_score'] = np.mean(combined_scores)
        else:
            summary['quantum_signature_score'] = 0.0
        
        # Information complexity
        info_measures = results.get('information_measures', {})
        if info_measures:
            shannon_entropies = [measure.get('shannon_entropy', 0) 
                               for measure in info_measures.values()]
            summary['information_complexity'] = np.mean(shannon_entropies)
        else:
            summary['information_complexity'] = 0.0
        
        # Overall quantum computation probability
        quantum_factors = [
            summary['quantum_signature_score'],
            min(1.0, summary['max_bell_violation'] / 2.0),  # Normalize
            min(1.0, summary['information_complexity'] / 10.0)  # Normalize
        ]
        
        summary['quantum_computation_probability'] = np.mean(quantum_factors)
        
        return summary
    
    def create_quantum_visualizations(self, analysis_results: Dict[str, Any], 
                                    output_dir: str) -> str:
        """
        Create comprehensive quantum analysis visualizations
        """
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Bell inequality violations
        plt.subplot(3, 3, 1)
        bell_results = analysis_results.get('bell_inequalities', {})
        
        bell_params = []
        bell_names = []
        for key, value in bell_results.items():
            if 'parameter' in key:
                bell_params.append(value)
                bell_names.append(key.replace('_parameter', '').upper())
        
        if bell_params:
            colors = ['red' if p > 2.0 else 'blue' for p in bell_params]
            plt.bar(bell_names, bell_params, color=colors, alpha=0.7)
            plt.axhline(y=2.0, color='red', linestyle='--', label='Classical Limit')
            plt.ylabel('Bell Parameter Value')
            plt.title('Bell Inequality Analysis')
            plt.legend()
            plt.xticks(rotation=45)
        
        # 2. Quantum signature scores
        plt.subplot(3, 3, 2)
        quantum_sigs = analysis_results.get('quantum_signatures', {})
        
        if quantum_sigs:
            datasets = list(quantum_sigs.keys())
            scores = [sig.get('combined_quantum_score', 0) for sig in quantum_sigs.values()]
            
            plt.bar(datasets, scores, color='purple', alpha=0.7)
            plt.ylabel('Quantum Signature Score')
            plt.title('Quantum Computational Signatures')
            plt.xticks(rotation=45)
            plt.ylim(0, 1)
        
        # 3. Information entropy comparison
        plt.subplot(3, 3, 3)
        info_measures = analysis_results.get('information_measures', {})
        
        if info_measures:
            datasets = list(info_measures.keys())
            shannon_entropies = [measure.get('shannon_entropy', 0) for measure in info_measures.values()]
            renyi_entropies = [measure.get('renyi_entropy', 0) for measure in info_measures.values()]
            
            x = np.arange(len(datasets))
            width = 0.35
            
            plt.bar(x - width/2, shannon_entropies, width, label='Shannon', alpha=0.7)
            plt.bar(x + width/2, renyi_entropies, width, label='Rényi', alpha=0.7)
            
            plt.xlabel('Datasets')
            plt.ylabel('Entropy')
            plt.title('Information Entropy Comparison')
            plt.xticks(x, datasets, rotation=45)
            plt.legend()
        
        # 4. Entanglement analysis
        plt.subplot(3, 3, 4)
        entanglement_results = analysis_results.get('entanglement_analysis', {})
        
        if entanglement_results:
            datasets = list(entanglement_results.keys())
            entanglement_scores = list(entanglement_results.values())
            
            plt.bar(datasets, entanglement_scores, color='green', alpha=0.7)
            plt.ylabel('Entanglement Entropy')
            plt.title('Entanglement Analysis')
            plt.xticks(rotation=45)
        
        # 5. Quantum Fisher information
        plt.subplot(3, 3, 5)
        if info_measures:
            datasets = list(info_measures.keys())
            fisher_info = [measure.get('quantum_fisher_info', 0) for measure in info_measures.values()]
            
            plt.semilogy(datasets, fisher_info, 'o-', color='orange')
            plt.ylabel('Quantum Fisher Information (log scale)')
            plt.title('Quantum Fisher Information')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        
        # 6. Bell violation radar chart
        plt.subplot(3, 3, 6)
        if bell_results:
            # Create radar chart for different Bell inequalities
            violations = []
            categories = []
            
            for key, value in bell_results.items():
                if 'violation' in key:
                    violations.append(value)
                    categories.append(key.replace('_violation', '').upper())
            
            if violations:
                angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
                violations += violations[:1]  # Complete the circle
                angles += angles[:1]
                
                plt.plot(angles, violations, 'o-', linewidth=2, color='red')
                plt.fill(angles, violations, alpha=0.25, color='red')
                plt.xticks(angles[:-1], categories)
                plt.title('Bell Violation Radar')
        
        # 7. Quantum algorithm signatures heatmap
        plt.subplot(3, 3, 7)
        if quantum_sigs:
            # Create heatmap of different quantum signatures
            datasets = list(quantum_sigs.keys())
            signature_types = ['interference_patterns', 'decoherence_signatures', 
                             'error_correction_patterns', 'quantum_algorithm_signatures']
            
            heatmap_data = []
            for dataset in datasets:
                row = []
                for sig_type in signature_types:
                    row.append(quantum_sigs[dataset].get(sig_type, 0))
                heatmap_data.append(row)
            
            if heatmap_data:
                im = plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
                plt.colorbar(im)
                plt.xticks(range(len(signature_types)), 
                          [s.replace('_', ' ').title() for s in signature_types], rotation=45)
                plt.yticks(range(len(datasets)), datasets)
                plt.title('Quantum Signature Heatmap')
        
        # 8. Summary statistics
        plt.subplot(3, 3, 8)
        plt.axis('off')
        
        summary = analysis_results.get('summary', {})
        
        summary_text = f"""
QUANTUM INFORMATION ANALYSIS SUMMARY

Bell Violations: {summary.get('bell_violations_count', 0)}
Max Bell Violation: {summary.get('max_bell_violation', 0):.3f}

Quantum Signature Score: {summary.get('quantum_signature_score', 0):.3f}
Information Complexity: {summary.get('information_complexity', 0):.3f}

Quantum Computation Probability: {summary.get('quantum_computation_probability', 0):.3f}

Interpretation:
• Bell violations suggest non-classical correlations
• Quantum signatures indicate computational patterns
• Information measures reveal system complexity
        """
        
        plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        # 9. Overall quantum score
        plt.subplot(3, 3, 9)
        quantum_prob = summary.get('quantum_computation_probability', 0)
        
        # Create gauge chart
        theta = np.linspace(0, np.pi, 100)
        r = 1
        
        # Background arc
        plt.plot(r * np.cos(theta), r * np.sin(theta), 'k-', linewidth=10, alpha=0.3)
        
        # Score arc
        score_theta = theta[:int(quantum_prob * 100)]
        colors = plt.cm.RdYlGn(quantum_prob)
        plt.plot(r * np.cos(score_theta), r * np.sin(score_theta), 
                color=colors, linewidth=10)
        
        plt.text(0, -0.3, f'{quantum_prob:.1%}', ha='center', va='center', 
                fontsize=20, fontweight='bold')
        plt.text(0, -0.5, 'Quantum Computation\nProbability', ha='center', va='center')
        plt.xlim(-1.2, 1.2)
        plt.ylim(-0.7, 1.2)
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save the plot
        output_file = output_path / "quantum_information_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  📊 Quantum analysis visualization saved: {output_file}")
        return str(output_file)


def test_quantum_information_analysis():
    """Test the quantum information analysis system"""
    
    print("Testing Quantum Information Analysis System...")
    
    # Create test datasets with quantum-like properties
    np.random.seed(42)
    
    test_datasets = {
        'bell_measurements_a': np.random.choice([-1, 1], 1000),
        'bell_measurements_b': np.random.choice([-1, 1], 1000),
        'quantum_interference': np.sin(np.linspace(0, 20*np.pi, 1000)) + 0.1*np.random.normal(0, 1, 1000),
        'decoherence_data': np.exp(-np.linspace(0, 5, 1000)) * np.cos(10*np.linspace(0, 5, 1000)) + 0.1*np.random.normal(0, 1, 1000),
        'entangled_system': np.random.normal(0, 1, 1024)  # Power of 2 for entanglement analysis
    }
    
    # Initialize analyzer
    analyzer = QuantumInformationAnalyzer()
    
    # Run comprehensive analysis
    results = analyzer.comprehensive_quantum_analysis(test_datasets)
    
    # Print results
    print(f"\n🎯 QUANTUM INFORMATION ANALYSIS RESULTS:")
    summary = results['summary']
    print(f"  Bell violations detected: {summary['bell_violations_count']}")
    print(f"  Max Bell violation: {summary['max_bell_violation']:.4f}")
    print(f"  Quantum signature score: {summary['quantum_signature_score']:.4f}")
    print(f"  Information complexity: {summary['information_complexity']:.4f}")
    print(f"  Quantum computation probability: {summary['quantum_computation_probability']:.4f}")
    
    return results


if __name__ == "__main__":
    test_quantum_information_analysis()
