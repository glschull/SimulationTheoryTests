"""
Visualization Tools for Simulation Theory Tests
===============================================

Provides plotting and visualization capabilities for analyzing test results
and displaying potential simulation signatures.

Author: AI Assistant for Garrett Schull's SimulationTheoryTests
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd


class SimulationTheoryVisualizer:
    """Main visualization class for simulation theory test results"""
    
    def __init__(self, style: str = 'dark'):
        """
        Initialize the visualizer
        
        Args:
            style: 'dark' for dark theme, 'light' for light theme
        """
        self.style = style
        self._setup_style()
    
    def _setup_style(self):
        """Setup matplotlib style for simulation theory aesthetics"""
        if self.style == 'dark':
            plt.style.use('dark_background')
            self.primary_color = '#00ff41'  # Matrix green
            self.secondary_color = '#ff0040'  # Alert red
            self.tertiary_color = '#4080ff'  # Blue
            self.background_color = '#0a0a0a'
        else:
            plt.style.use('default')
            self.primary_color = '#2E8B57'  # Sea green
            self.secondary_color = '#DC143C'  # Crimson
            self.tertiary_color = '#4169E1'  # Royal blue
            self.background_color = '#ffffff'
        
        # Custom font settings
        plt.rcParams.update({
            'font.family': 'monospace',
            'font.size': 10,
            'axes.linewidth': 1.5,
            'grid.alpha': 0.3
        })


class QuantumVisualizationModule(SimulationTheoryVisualizer):
    """Visualizations for quantum measurement and observer effect tests"""
    
    def plot_observer_effect_analysis(self, results_data: List[Dict[str, Any]], 
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot observer effect analysis results
        
        Args:
            results_data: List of quantum experiment results
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('🔬 QUANTUM OBSERVER EFFECT ANALYSIS', 
                    fontsize=16, color=self.primary_color, weight='bold')
        
        # Extract data
        observer_probs = [r['observed_ratio'] for r in results_data]
        collapse_rates = [r['collapse_rate'] for r in results_data]
        interference_means = [r['interference_mean'] for r in results_data]
        interference_stds = [r['interference_std'] for r in results_data]
        
        # Plot 1: Observer Probability vs Collapse Rate
        axes[0, 0].scatter(observer_probs, collapse_rates, 
                          color=self.primary_color, alpha=0.7, s=50)
        axes[0, 0].plot([0, 1], [0, 1], '--', color=self.secondary_color, 
                       alpha=0.5, label='Expected (no anomaly)')
        axes[0, 0].set_xlabel('Observer Probability')
        axes[0, 0].set_ylabel('Collapse Rate')
        axes[0, 0].set_title('Observer Effect Correlation')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Interference Pattern Analysis
        axes[0, 1].errorbar(observer_probs, interference_means, yerr=interference_stds,
                           fmt='o', color=self.tertiary_color, capsize=5)
        axes[0, 1].axhline(y=0, color=self.secondary_color, linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('Observer Probability')
        axes[0, 1].set_ylabel('Interference Pattern Mean')
        axes[0, 1].set_title('Interference vs Observation')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Statistical Significance
        chi_square_p_values = []
        for r in results_data:
            if r.get('chi_square_test') and r['chi_square_test'].get('p_value'):
                chi_square_p_values.append(r['chi_square_test']['p_value'])
            else:
                chi_square_p_values.append(1.0)
        
        axes[1, 0].bar(range(len(chi_square_p_values)), chi_square_p_values,
                      color=self.primary_color, alpha=0.7)
        axes[1, 0].axhline(y=0.05, color=self.secondary_color, linestyle='--', 
                          label='Significance Threshold (p=0.05)')
        axes[1, 0].set_xlabel('Experiment Index')
        axes[1, 0].set_ylabel('Chi-square p-value')
        axes[1, 0].set_title('Statistical Randomness Test')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Anomaly Detection Summary
        anomaly_scores = []
        for i, r in enumerate(results_data):
            score = 0
            if chi_square_p_values[i] < 0.05:
                score += 0.5
            if abs(interference_means[i]) > 0.5:
                score += 0.3
            if abs(collapse_rates[i] - observer_probs[i]) > 0.1:
                score += 0.2
            anomaly_scores.append(score)
        
        colors = [self.secondary_color if score > 0.5 else self.primary_color 
                 for score in anomaly_scores]
        axes[1, 1].bar(range(len(anomaly_scores)), anomaly_scores, color=colors, alpha=0.7)
        axes[1, 1].axhline(y=0.5, color=self.secondary_color, linestyle='--', 
                          label='Anomaly Threshold')
        axes[1, 1].set_xlabel('Experiment Index')
        axes[1, 1].set_ylabel('Anomaly Score')
        axes[1, 1].set_title('Anomaly Detection')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor=self.background_color)
        
        return fig


class PlanckVisualizationModule(SimulationTheoryVisualizer):
    """Visualizations for Planck-scale discreteness analysis"""
    
    def plot_discreteness_analysis(self, data: np.ndarray, analysis_result: Dict[str, Any],
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot Planck-scale discreteness analysis
        
        Args:
            data: Original measurement data
            analysis_result: Results from discreteness analysis
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('⚛️ PLANCK-SCALE DISCRETENESS ANALYSIS', 
                    fontsize=16, color=self.primary_color, weight='bold')
        
        # Plot 1: Original data distribution
        axes[0, 0].hist(data, bins=50, color=self.primary_color, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Measurement Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Original Data Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Interval differences histogram
        diffs = np.diff(np.sort(data))
        axes[0, 1].hist(diffs, bins=50, color=self.tertiary_color, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Interval Difference')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Interval Differences')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Power spectrum analysis
        fft_result = np.fft.fft(diffs)
        frequencies = np.fft.fftfreq(len(diffs))
        power_spectrum = np.abs(fft_result) ** 2
        
        # Only plot positive frequencies
        pos_freq_idx = frequencies > 0
        axes[0, 2].semilogy(frequencies[pos_freq_idx], power_spectrum[pos_freq_idx],
                           color=self.primary_color, alpha=0.8)
        axes[0, 2].set_xlabel('Frequency')
        axes[0, 2].set_ylabel('Power (log scale)')
        axes[0, 2].set_title('Power Spectrum Analysis')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Discreteness score visualization
        score = analysis_result['discreteness_score']
        score_colors = ['green', 'yellow', 'orange', 'red']
        score_labels = ['Continuous', 'Slightly Discrete', 'Discrete', 'Highly Discrete']
        score_thresholds = [0.25, 0.5, 0.75, 1.0]
        
        score_category = 0
        for i, threshold in enumerate(score_thresholds):
            if score <= threshold:
                score_category = i
                break
        
        # Create a gauge-like visualization
        theta = np.linspace(0, np.pi, 100)
        r_outer = 1
        r_inner = 0.7
        
        for i, (threshold, color, label) in enumerate(zip(score_thresholds, score_colors, score_labels)):
            start_angle = i * np.pi / 4
            end_angle = (i + 1) * np.pi / 4
            theta_section = np.linspace(start_angle, end_angle, 25)
            
            axes[1, 0].fill_between(theta_section, r_inner, r_outer, 
                                   color=color, alpha=0.6, label=label)
        
        # Add score indicator
        score_angle = score * np.pi
        axes[1, 0].plot([score_angle, score_angle], [r_inner, r_outer], 
                       color='black', linewidth=3)
        axes[1, 0].plot([score_angle, score_angle], [r_inner, r_outer], 
                       color='white', linewidth=1)
        
        axes[1, 0].set_xlim(0, np.pi)
        axes[1, 0].set_ylim(0.6, 1.1)
        axes[1, 0].set_title(f'Discreteness Score: {score:.3f}')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 5: Statistical test results
        test_names = ['KS Test', 'Anderson-Darling']
        test_scores = [1 - analysis_result['ks_test']['p_value'],
                      min(1.0, analysis_result['anderson_darling']['statistic'] / 10)]
        
        bars = axes[1, 1].bar(test_names, test_scores, 
                             color=[self.primary_color, self.tertiary_color], alpha=0.7)
        axes[1, 1].axhline(y=0.95, color=self.secondary_color, linestyle='--', 
                          label='Significance Threshold')
        axes[1, 1].set_ylabel('Test Score')
        axes[1, 1].set_title('Statistical Tests')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, test_scores):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{score:.3f}', ha='center', va='bottom')
        
        # Plot 6: Entropy analysis
        hist, bin_edges = np.histogram(diffs, bins=50)
        entropy = analysis_result['entropy']
        
        # Compare with expected entropy for random data
        uniform_entropy = np.log(len(hist))  # Maximum entropy for uniform distribution
        
        entropy_comparison = ['Observed', 'Maximum (Random)']
        entropy_values = [entropy, uniform_entropy]
        
        bars = axes[1, 2].bar(entropy_comparison, entropy_values,
                             color=[self.primary_color, self.secondary_color], alpha=0.7)
        axes[1, 2].set_ylabel('Entropy')
        axes[1, 2].set_title('Entropy Analysis')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, entropy_values):
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor=self.background_color)
        
        return fig


class CMBVisualizationModule(SimulationTheoryVisualizer):
    """Visualizations for Cosmic Microwave Background analysis"""
    
    def plot_cmb_analysis(self, cmb_data: np.ndarray, analysis_result: Dict[str, Any],
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot CMB temperature map and analysis results
        
        Args:
            cmb_data: 2D array of CMB temperature data
            analysis_result: Results from CMB analysis
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('🌌 COSMIC MICROWAVE BACKGROUND ANALYSIS', 
                    fontsize=16, color=self.primary_color, weight='bold')
        
        # Create custom layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: CMB Temperature Map (main plot)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        
        # Create custom colormap for CMB
        cmb_colors = ['#000080', '#0000FF', '#00FFFF', '#FFFF00', '#FF8000', '#FF0000']
        n_bins = 256
        cmb_cmap = LinearSegmentedColormap.from_list('cmb', cmb_colors, N=n_bins)
        
        im = ax1.imshow(cmb_data, cmap=cmb_cmap, aspect='equal')
        ax1.set_title('CMB Temperature Map')
        ax1.set_xlabel('Pixel X')
        ax1.set_ylabel('Pixel Y')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
        cbar.set_label('Temperature (K)', rotation=270, labelpad=15)
        
        # Plot 2: Temperature histogram
        ax2 = fig.add_subplot(gs[0, 2])
        flattened_temps = cmb_data.flatten()
        ax2.hist(flattened_temps, bins=50, color=self.primary_color, alpha=0.7, orientation='horizontal')
        ax2.set_ylabel('Temperature (K)')
        ax2.set_xlabel('Frequency')
        ax2.set_title('Temperature Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Power spectrum
        ax3 = fig.add_subplot(gs[1, 2])
        fft_2d = np.fft.fft2(cmb_data)
        power_spectrum_2d = np.abs(fft_2d) ** 2
        
        # Radially average the power spectrum
        center = np.array(power_spectrum_2d.shape) // 2
        y, x = np.ogrid[:power_spectrum_2d.shape[0], :power_spectrum_2d.shape[1]]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        
        # Create radial bins
        r_bins = np.linspace(0, min(center), 50)
        power_radial = []
        
        for i in range(len(r_bins) - 1):
            mask = (r >= r_bins[i]) & (r < r_bins[i + 1])
            if np.any(mask):
                power_radial.append(np.mean(power_spectrum_2d[mask]))
            else:
                power_radial.append(0)
        
        r_centers = (r_bins[:-1] + r_bins[1:]) / 2
        ax3.loglog(r_centers[1:], power_radial[1:], color=self.primary_color, linewidth=2)
        ax3.set_xlabel('Spatial Frequency')
        ax3.set_ylabel('Power')
        ax3.set_title('Angular Power Spectrum')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Anomaly detection results
        ax4 = fig.add_subplot(gs[2, 0])
        
        anomaly_score = analysis_result['anomaly_score']
        suspicious_patterns = len(analysis_result['hash_signatures']['suspicious_patterns'])
        normality_p = analysis_result['statistical_analysis']['normality_test']['p_value']
        
        metrics = ['Anomaly Score', 'Suspicious Patterns', 'Non-Gaussianity']
        values = [anomaly_score, min(1.0, suspicious_patterns / 5), 1 - normality_p]
        colors = [self.secondary_color if v > 0.5 else self.primary_color for v in values]
        
        bars = ax4.bar(metrics, values, color=colors, alpha=0.7)
        ax4.set_ylabel('Score')
        ax4.set_title('Anomaly Metrics')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 5: Statistical summary
        ax5 = fig.add_subplot(gs[2, 1])
        
        stats_text = f"""Statistical Analysis:
        
Mean Temp: {analysis_result['statistical_analysis']['mean_temperature']:.6f} K
Std Dev: {analysis_result['statistical_analysis']['std_temperature']:.2e} K
Normality p-value: {normality_p:.6f}
Max Autocorr: {analysis_result['statistical_analysis']['max_autocorrelation']:.4f}

Hash Analysis:
Suspicious Patterns: {suspicious_patterns}
Total Hash Checks: {len(analysis_result['hash_signatures']['data_hash'])}

Frequency Analysis:
Power Spectrum Peaks: {analysis_result['frequency_analysis']['peak_count']}
PS Entropy: {analysis_result['frequency_analysis']['power_spectrum_entropy']:.4f}
        """
        
        ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor=self.background_color, alpha=0.8))
        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)
        ax5.axis('off')
        ax5.set_title('Analysis Summary')
        
        # Plot 6: Suspected artifact regions (if any)
        ax6 = fig.add_subplot(gs[2, 2])
        
        if suspicious_patterns > 0:
            # Highlight potential artifact regions
            artifact_map = np.zeros_like(cmb_data)
            
            # Simple artifact detection: find regions with suspicious regularity
            from scipy import ndimage
            temp_var = ndimage.uniform_filter(cmb_data**2, size=32) - ndimage.uniform_filter(cmb_data, size=32)**2
            threshold = np.percentile(temp_var, 95)
            artifact_map[temp_var > threshold] = 1
            
            ax6.imshow(artifact_map, cmap='Reds', alpha=0.7)
            ax6.set_title(f'Potential Artifacts ({suspicious_patterns} found)')
        else:
            ax6.text(0.5, 0.5, 'No significant\nartifacts detected', 
                    transform=ax6.transAxes, ha='center', va='center',
                    fontsize=12, color=self.primary_color)
            ax6.set_title('Artifact Detection')
        
        ax6.set_xlabel('Pixel X')
        ax6.set_ylabel('Pixel Y')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor=self.background_color)
        
        return fig


def create_comprehensive_report(quantum_results: List[Dict], 
                              planck_results: List[Dict],
                              cmb_results: List[Dict],
                              constants_results: Dict,
                              save_dir: str = "./results") -> None:
    """
    Create a comprehensive visualization report of all simulation theory tests
    
    Args:
        quantum_results: Results from quantum observer effect tests
        planck_results: Results from Planck discreteness tests
        cmb_results: Results from CMB analysis
        constants_results: Results from physical constants analysis
        save_dir: Directory to save the report
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print("📊 Generating comprehensive simulation theory test report...")
    
    # Initialize visualizers
    quantum_viz = QuantumVisualizationModule(style='dark')
    planck_viz = PlanckVisualizationModule(style='dark')
    cmb_viz = CMBVisualizationModule(style='dark')
    
    # Generate quantum plots
    if quantum_results:
        fig_quantum = quantum_viz.plot_observer_effect_analysis(
            quantum_results, 
            save_path=os.path.join(save_dir, 'quantum_observer_analysis.png')
        )
        plt.close(fig_quantum)
    
    # Generate Planck plots
    for i, result in enumerate(planck_results):
        if 'data' in result and 'analysis' in result:
            fig_planck = planck_viz.plot_discreteness_analysis(
                result['data'], 
                result['analysis'],
                save_path=os.path.join(save_dir, f'planck_discreteness_{i}.png')
            )
            plt.close(fig_planck)
    
    # Generate CMB plots
    for i, result in enumerate(cmb_results):
        if 'data' in result and 'analysis' in result:
            fig_cmb = cmb_viz.plot_cmb_analysis(
                result['data'], 
                result['analysis'],
                save_path=os.path.join(save_dir, f'cmb_analysis_{i}.png')
            )
            plt.close(fig_cmb)
    
    print(f"✅ Report generated in: {os.path.abspath(save_dir)}")


if __name__ == "__main__":
    # Example usage
    print("🎨 Simulation Theory Visualization Tools Ready!")
    print("Import this module to create visualizations for your test results.")
