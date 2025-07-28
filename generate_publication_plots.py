#!/usr/bin/env python3
"""
Generate publication-ready visualizations for Simulation Theory Test results.
Creates high-resolution, professional plots suitable for scientific papers.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
import json
from pathlib import Path

# Set publication-ready style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Configure for high-quality output
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'font.family': 'serif',
    'text.usetex': False  # Set to True if LaTeX is available
})

def load_results():
    """Load comprehensive analysis results."""
    results_path = Path("results/comprehensive_analysis.json")
    if results_path.exists():
        with open(results_path, 'r') as f:
            return json.load(f)
    return None

def create_master_summary_plot():
    """Create a comprehensive summary plot of all results."""
    results = load_results()
    if not results:
        print("Results file not found!")
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, hspace=0.3, wspace=0.3)
    
    # Overall scores subplot
    ax1 = fig.add_subplot(gs[0, :])
    datasets = ['Quantum\nMeasurements', 'Planck\nIntervals', 'Physical\nConstants', 'CMB\nTemperatures']
    sim_probs = [
        results['individual_analyses']['quantum_measurements']['bayesian_analysis']['simulation_probability'],
        results['individual_analyses']['planck_intervals']['bayesian_analysis']['simulation_probability'],
        results['individual_analyses']['physical_constants']['bayesian_analysis']['simulation_probability'],
        results['individual_analyses']['cmb_temperatures']['bayesian_analysis']['simulation_probability']
    ]
    
    bars = ax1.bar(datasets, sim_probs, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], 
                   alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_ylabel('Simulation Probability')
    ax1.set_title('Simulation Hypothesis Test Results - Individual Dataset Analysis', fontweight='bold')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, prob in zip(bars, sim_probs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # Add horizontal line for overall score
    overall_score = results['overall_assessment']['overall_suspicion_score']
    ax1.axhline(y=overall_score, color='red', linestyle='--', linewidth=2, 
                label=f'Overall Score: {overall_score:.3f}')
    ax1.legend()
    
    # Digital signatures subplot
    ax2 = fig.add_subplot(gs[1, 0])
    dig_scores = [
        results['individual_analyses']['quantum_measurements']['digital_signatures']['digital_signature_score'],
        results['individual_analyses']['planck_intervals']['digital_signatures']['digital_signature_score'],
        results['individual_analyses']['physical_constants']['digital_signatures']['digital_signature_score'],
        results['individual_analyses']['cmb_temperatures']['digital_signatures']['digital_signature_score']
    ]
    
    wedges, texts, autotexts = ax2.pie(dig_scores, labels=datasets, autopct='%1.1f%%', 
                                       colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax2.set_title('Digital Signature Scores', fontweight='bold')
    
    # Mutual information heatmap
    ax3 = fig.add_subplot(gs[1, 1:])
    mi_data = results['cross_dataset_analysis']['mutual_information']
    
    # Create correlation matrix
    labels = ['Quantum', 'Planck', 'Constants', 'CMB']
    mi_matrix = np.zeros((4, 4))
    
    # Fill matrix with mutual information values
    pairs = [
        ('quantum_measurements_vs_planck_intervals', 0, 1),
        ('quantum_measurements_vs_physical_constants', 0, 2),
        ('quantum_measurements_vs_cmb_temperatures', 0, 3),
        ('planck_intervals_vs_physical_constants', 1, 2),
        ('planck_intervals_vs_cmb_temperatures', 1, 3),
        ('physical_constants_vs_cmb_temperatures', 2, 3)
    ]
    
    for key, i, j in pairs:
        value = mi_data[key]
        mi_matrix[i, j] = value
        mi_matrix[j, i] = value
    
    sns.heatmap(mi_matrix, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels,
                cmap='YlOrRd', ax=ax3, cbar_kws={'label': 'Mutual Information (bits)'})
    ax3.set_title('Cross-Dataset Information Correlations', fontweight='bold')
    
    # Compression analysis
    ax4 = fig.add_subplot(gs[2, :2])
    compression_data = []
    dataset_names = []
    
    for dataset in ['quantum_measurements', 'planck_intervals', 'physical_constants', 'cmb_temperatures']:
        info_theory = results['individual_analyses'][dataset]['information_theory']
        compression_data.append([
            info_theory['zlib_compression_ratio'],
            info_theory['bz2_compression_ratio'],
            info_theory['lzma_compression_ratio']
        ])
        dataset_names.append(dataset.replace('_', ' ').title())
    
    compression_data = np.array(compression_data)
    x = np.arange(len(dataset_names))
    width = 0.25
    
    ax4.bar(x - width, compression_data[:, 0], width, label='ZLIB', alpha=0.8)
    ax4.bar(x, compression_data[:, 1], width, label='BZ2', alpha=0.8)
    ax4.bar(x + width, compression_data[:, 2], width, label='LZMA', alpha=0.8)
    
    ax4.set_ylabel('Compression Ratio')
    ax4.set_title('Data Compression Analysis by Algorithm', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax4.legend()
    ax4.axhline(y=1, color='black', linestyle='-', alpha=0.3, label='No Compression')
    
    # Statistical summary
    ax5 = fig.add_subplot(gs[2, 2])
    stats_summary = [
        overall_score,
        results['overall_assessment']['average_simulation_probability'],
        results['overall_assessment']['average_digital_signature_score'],
        abs(results['overall_assessment']['average_compression_artificiality'])
    ]
    
    labels_stats = ['Overall\nSuspicion', 'Avg Simulation\nProbability', 'Avg Digital\nSignatures', 'Avg Compression\nArtifacts']
    
    ax5.barh(labels_stats, stats_summary, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.8)
    ax5.set_xlim(0, 1)
    ax5.set_xlabel('Score')
    ax5.set_title('Summary Statistics', fontweight='bold')
    
    # Add value labels
    for i, v in enumerate(stats_summary):
        ax5.text(v + 0.01, i, f'{v:.3f}', va='center', fontweight='bold')
    
    plt.suptitle('Simulation Theory Test Suite - Comprehensive Analysis Report\n'
                 f'Overall Suspicion Score: {overall_score:.3f} (MEDIUM Confidence)', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig('results/MASTER_ANALYSIS_REPORT.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Master analysis report saved: results/MASTER_ANALYSIS_REPORT.png")

def create_anomaly_detection_plot():
    """Create detailed anomaly detection visualization."""
    results = load_results()
    if not results:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Anomaly Detection Analysis - Detailed Breakdown', fontsize=16, fontweight='bold')
    
    datasets = ['quantum_measurements', 'planck_intervals', 'physical_constants', 'cmb_temperatures']
    dataset_labels = ['Quantum Measurements', 'Planck Intervals', 'Physical Constants', 'CMB Temperatures']
    
    for idx, (dataset, label) in enumerate(zip(datasets, dataset_labels)):
        ax = axes[idx // 2, idx % 2]
        
        # Get anomaly data
        bayesian = results['individual_analyses'][dataset]['bayesian_analysis']['anomaly_likelihoods']
        
        # Create radar-like plot for anomalies
        anomaly_types = list(bayesian.keys())
        anomaly_values = list(bayesian.values())
        
        # Convert to percentages for better visualization
        anomaly_values = [v * 100 if isinstance(v, float) else v for v in anomaly_values]
        
        bars = ax.bar(range(len(anomaly_types)), anomaly_values, 
                     color=plt.cm.viridis(np.linspace(0, 1, len(anomaly_types))), alpha=0.7)
        
        ax.set_title(f'{label}\nSimulation Probability: {results["individual_analyses"][dataset]["bayesian_analysis"]["simulation_probability"]:.1%}', 
                    fontweight='bold')
        ax.set_xticks(range(len(anomaly_types)))
        ax.set_xticklabels([t.replace('_', ' ').title() for t in anomaly_types], rotation=45, ha='right')
        ax.set_ylabel('Anomaly Score')
        
        # Add value labels
        for bar, value in zip(bars, anomaly_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{value:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/ANOMALY_DETECTION_ANALYSIS.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Anomaly detection analysis saved: results/ANOMALY_DETECTION_ANALYSIS.png")

def create_information_theory_plot():
    """Create information theory and complexity analysis plot."""
    results = load_results()
    if not results:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Information Theory & Complexity Analysis', fontsize=16, fontweight='bold')
    
    datasets = ['quantum_measurements', 'planck_intervals', 'physical_constants', 'cmb_temperatures']
    dataset_labels = ['Quantum\nMeasurements', 'Planck\nIntervals', 'Physical\nConstants', 'CMB\nTemperatures']
    
    # Compression ratios
    ax1 = axes[0, 0]
    compression_types = ['ZLIB', 'BZ2', 'LZMA']
    compression_data = np.zeros((len(datasets), len(compression_types)))
    
    for i, dataset in enumerate(datasets):
        info_theory = results['individual_analyses'][dataset]['information_theory']
        compression_data[i] = [
            info_theory['zlib_compression_ratio'],
            info_theory['bz2_compression_ratio'],
            info_theory['lzma_compression_ratio']
        ]
    
    x = np.arange(len(dataset_labels))
    width = 0.25
    
    for i, comp_type in enumerate(compression_types):
        ax1.bar(x + i * width, compression_data[:, i], width, label=comp_type, alpha=0.8)
    
    ax1.set_title('Compression Ratio Analysis', fontweight='bold')
    ax1.set_ylabel('Compression Ratio')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(dataset_labels)
    ax1.legend()
    ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='No Compression')
    
    # Complexity scores
    ax2 = axes[0, 1]
    complexity_scores = []
    for dataset in datasets:
        info_theory = results['individual_analyses'][dataset]['information_theory']
        complexity_scores.append(info_theory['complexity_score'])
    
    bars = ax2.bar(dataset_labels, complexity_scores, 
                  color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.8)
    ax2.set_title('Kolmogorov Complexity Estimates', fontweight='bold')
    ax2.set_ylabel('Complexity Score')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, score in zip(bars, complexity_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height > 0 else height - 0.03,
                f'{score:.3f}', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')
    
    # Mutual information network
    ax3 = axes[1, :]
    mi_data = results['cross_dataset_analysis']['mutual_information']
    
    # Create network-style visualization
    positions = {
        'Quantum': (0, 1),
        'Planck': (1, 1),
        'Constants': (0, 0),
        'CMB': (1, 0)
    }
    
    # Draw nodes
    for name, (x, y) in positions.items():
        ax3.scatter(x, y, s=1000, alpha=0.7, 
                   c=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'][list(positions.keys()).index(name)])
        ax3.text(x, y, name, ha='center', va='center', fontweight='bold', fontsize=10)
    
    # Draw connections based on mutual information
    connections = [
        ('Quantum', 'Planck', mi_data['quantum_measurements_vs_planck_intervals']),
        ('Quantum', 'Constants', mi_data['quantum_measurements_vs_physical_constants']),
        ('Quantum', 'CMB', mi_data['quantum_measurements_vs_cmb_temperatures']),
        ('Planck', 'Constants', mi_data['planck_intervals_vs_physical_constants']),
        ('Planck', 'CMB', mi_data['planck_intervals_vs_cmb_temperatures']),
        ('Constants', 'CMB', mi_data['physical_constants_vs_cmb_temperatures'])
    ]
    
    for node1, node2, mi_value in connections:
        x1, y1 = positions[node1]
        x2, y2 = positions[node2]
        
        # Line thickness based on mutual information
        linewidth = max(1, mi_value * 2)
        alpha = min(1, mi_value / 2)
        
        ax3.plot([x1, x2], [y1, y2], linewidth=linewidth, alpha=alpha, color='black')
        
        # Add MI value label
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax3.text(mid_x, mid_y, f'{mi_value:.2f}', ha='center', va='center', 
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8), fontsize=8)
    
    ax3.set_xlim(-0.3, 1.3)
    ax3.set_ylim(-0.3, 1.3)
    ax3.set_title('Cross-Dataset Information Network\n(Line thickness = Mutual Information strength)', fontweight='bold')
    ax3.set_aspect('equal')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/INFORMATION_THEORY_ANALYSIS.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Information theory analysis saved: results/INFORMATION_THEORY_ANALYSIS.png")

if __name__ == "__main__":
    print("🎨 Generating publication-ready visualizations...")
    print("=" * 60)
    
    create_master_summary_plot()
    create_anomaly_detection_plot()
    create_information_theory_plot()
    
    print("\n✅ All publication-ready visualizations completed!")
    print("📁 Files saved in: results/ directory")
    print("🔍 Ready for scientific publication and peer review")
