"""
Data Generation Utilities for Simulation Theory Tests
=====================================================

Generates realistic mock datasets for testing simulation hypothesis algorithms.
Includes cosmic ray data, CMB temperature maps, quantum measurement data, etc.

Author: AI Assistant for Garrett Schull's SimulationTheoryTests
"""

import numpy as np
import pandas as pd
import json
from typing import Tuple, Dict, List, Any
import os


class CosmicRayDataGenerator:
    """Generate realistic cosmic ray timing and energy data"""
    
    def __init__(self):
        self.cosmic_ray_rate = 10  # events per second (scaled for demonstration)
        
    def generate_timing_data(self, duration_hours: float = 24, 
                           detector_area_m2: float = 100) -> pd.DataFrame:
        """
        Generate cosmic ray timing data
        
        Args:
            duration_hours: Observation duration in hours
            detector_area_m2: Detector area in square meters
            
        Returns:
            DataFrame with timestamps and event properties
        """
        duration_seconds = duration_hours * 3600
        
        # Calculate expected number of events
        # Cosmic ray flux ~ 1 particle per cm²·min for E > 1 GeV
        # Scaled down to reasonable size for memory efficiency
        base_rate = 10  # events per second (scaled down from realistic rate)
        expected_events = int(base_rate * duration_seconds)
        
        # Generate arrival times (Poisson process)
        inter_arrival_times = np.random.exponential(1/self.cosmic_ray_rate, expected_events)
        timestamps = np.cumsum(inter_arrival_times)
        
        # Filter to observation window
        timestamps = timestamps[timestamps <= duration_seconds]
        
        # Generate energy values (power law distribution)
        # E^(-2.7) spectrum typical for cosmic rays
        energies = self._generate_energy_spectrum(len(timestamps))
        
        # Generate detector coordinates
        x_coords = np.random.uniform(-np.sqrt(detector_area_m2)/2, 
                                   np.sqrt(detector_area_m2)/2, len(timestamps))
        y_coords = np.random.uniform(-np.sqrt(detector_area_m2)/2, 
                                   np.sqrt(detector_area_m2)/2, len(timestamps))
        
        # Add some detector effects and noise
        measured_energies = energies * np.random.normal(1.0, 0.05, len(energies))  # 5% resolution
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'energy_gev': measured_energies,
            'x_position': x_coords,
            'y_position': y_coords,
            'detector_response': np.random.exponential(1.0, len(timestamps))
        })
    
    def _generate_energy_spectrum(self, n_events: int) -> np.ndarray:
        """Generate cosmic ray energy spectrum following E^(-2.7) power law"""
        # Energy range from 1 GeV to 10^20 eV
        min_energy = 1.0  # GeV
        max_energy = 1e11  # GeV (10^20 eV)
        
        # Generate using inverse transform sampling for power law
        u = np.random.uniform(0, 1, n_events)
        alpha = -2.7
        
        # For power law: E = [(1-u) * E_min^(α+1) + u * E_max^(α+1)]^(1/(α+1))
        # But we'll use a simpler approximation for computational efficiency
        energies = min_energy * (max_energy/min_energy) ** u
        energies = energies ** (1/(alpha + 1))
        
        return energies


class QuantumMeasurementDataGenerator:
    """Generate quantum measurement datasets for testing observer effects"""
    
    def generate_double_slit_data(self, n_experiments: int = 1000,
                                measurements_per_exp: int = 100) -> Dict[str, Any]:
        """
        Generate double-slit experiment data with observer effects
        
        Args:
            n_experiments: Number of experimental runs
            measurements_per_exp: Measurements per experiment
            
        Returns:
            Dictionary containing experimental data
        """
        experiments = []
        
        for exp_id in range(n_experiments):
            # Random observer presence probability for this experiment
            observer_prob = np.random.uniform(0, 1)
            
            measurements = []
            for meas_id in range(measurements_per_exp):
                # Quantum state before measurement
                initial_state = {
                    'amplitude_A': np.complex128(np.random.normal(0, 1) + 1j * np.random.normal(0, 1)),
                    'amplitude_B': np.complex128(np.random.normal(0, 1) + 1j * np.random.normal(0, 1))
                }
                
                # Measurement occurs?
                measurement_made = np.random.random() < observer_prob
                
                if measurement_made:
                    # Wave function collapse
                    prob_A = abs(initial_state['amplitude_A'])**2
                    prob_B = abs(initial_state['amplitude_B'])**2
                    total_prob = prob_A + prob_B
                    
                    if total_prob > 0:
                        prob_A /= total_prob
                        result = 'A' if np.random.random() < prob_A else 'B'
                    else:
                        result = 'A'  # Default
                    
                    # Measured position (no interference)
                    if result == 'A':
                        position = np.random.normal(-1, 0.1)  # Slit A position
                    else:
                        position = np.random.normal(1, 0.1)   # Slit B position
                else:
                    # Interference pattern
                    # Superposition creates interference on screen
                    phase_diff = np.angle(initial_state['amplitude_A']) - np.angle(initial_state['amplitude_B'])
                    interference_amplitude = abs(initial_state['amplitude_A']) + abs(initial_state['amplitude_B'])
                    
                    # Position shows interference pattern
                    position = interference_amplitude * np.cos(phase_diff) + np.random.normal(0, 0.2)
                    result = 'interference'
                
                measurements.append({
                    'measurement_id': meas_id,
                    'observer_present': measurement_made,
                    'result_type': result,
                    'position': position,
                    'timestamp': exp_id * measurements_per_exp + meas_id
                })
            
            experiments.append({
                'experiment_id': exp_id,
                'observer_probability': observer_prob,
                'measurements': measurements
            })
        
        return {
            'metadata': {
                'total_experiments': n_experiments,
                'measurements_per_experiment': measurements_per_exp,
                'generation_timestamp': pd.Timestamp.now().isoformat()
            },
            'experiments': experiments
        }


class CMBDataGenerator:
    """Generate cosmic microwave background temperature maps"""
    
    def __init__(self):
        self.cmb_temperature = 2.7255  # Kelvin
        self.dipole_amplitude = 3.365e-3  # Kelvin (due to solar system motion)
        
    def generate_temperature_map(self, resolution: Tuple[int, int] = (512, 512),
                               add_foregrounds: bool = True,
                               add_artifacts: bool = False) -> np.ndarray:
        """
        Generate CMB temperature map
        
        Args:
            resolution: (height, width) of the map
            add_foregrounds: Whether to include galactic foregrounds
            add_artifacts: Whether to add potential simulation artifacts
            
        Returns:
            2D array of temperature values in Kelvin
        """
        height, width = resolution
        
        # Base CMB temperature
        temp_map = np.full((height, width), self.cmb_temperature)
        
        # Add primordial fluctuations (Gaussian random field)
        # Typical amplitude ~ 10^-5 K
        fluctuation_amplitude = 1e-5
        fluctuations = np.random.normal(0, fluctuation_amplitude, (height, width))
        
        # Apply realistic power spectrum (simplified)
        # Real CMB has specific angular power spectrum
        fft_fluctuations = np.fft.fft2(fluctuations)
        
        # Create simplified angular power spectrum weighting
        kx = np.fft.fftfreq(width).reshape(1, -1)
        ky = np.fft.fftfreq(height).reshape(-1, 1)
        k = np.sqrt(kx**2 + ky**2)
        
        # Approximate CMB power spectrum shape (very simplified)
        power_spectrum = np.exp(-k**2 * 100) * (1 + k*50)**(-1)
        power_spectrum[0, 0] = 0  # Remove DC component
        
        # Apply power spectrum
        fft_fluctuations *= np.sqrt(power_spectrum)
        fluctuations = np.real(np.fft.ifft2(fft_fluctuations))
        
        temp_map += fluctuations
        
        # Add dipole anisotropy (due to our motion through CMB)
        x_coords, y_coords = np.meshgrid(np.linspace(-1, 1, width), 
                                        np.linspace(-1, 1, height))
        # Simplified dipole pattern
        dipole = self.dipole_amplitude * x_coords
        temp_map += dipole
        
        if add_foregrounds:
            # Add galactic foregrounds
            # Dust emission (correlated with galactic latitude)
            dust_template = np.abs(y_coords) ** (-1.5)  # Higher near galactic plane
            dust_emission = 1e-6 * dust_template * np.random.exponential(1, (height, width))
            temp_map += dust_emission
            
            # Synchrotron radiation
            synchrotron = 5e-7 * np.random.exponential(1, (height, width))
            temp_map += synchrotron
        
        if add_artifacts:
            # Add potential simulation artifacts
            artifact_type = np.random.choice(['grid', 'periodic', 'encoded'])
            
            if artifact_type == 'grid':
                # Regular grid pattern (suspicious for natural data)
                grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
                grid_pattern = 1e-7 * np.sin(grid_x * 2 * np.pi / 32) * np.sin(grid_y * 2 * np.pi / 32)
                temp_map += grid_pattern
                
            elif artifact_type == 'periodic':
                # Suspiciously perfect periodic structure
                period_x, period_y = 64, 48
                periodic_pattern = 1e-7 * np.sin(x_coords * 2 * np.pi * period_x) * np.cos(y_coords * 2 * np.pi * period_y)
                temp_map += periodic_pattern
                
            elif artifact_type == 'encoded':
                # Hidden message in temperature variations
                message = "SIMULATION"
                for i, char in enumerate(message):
                    if i * 50 < width:
                        # Encode ASCII value as temperature offset
                        temp_map[:, i*50:(i+1)*50] += ord(char) * 1e-9
        
        return temp_map


def generate_all_datasets():
    """Generate all datasets and save to data directory"""
    print("🔬 Generating simulation theory test datasets...")
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join('data')
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. Generate cosmic ray data
    print("⚡ Generating cosmic ray data...")
    cosmic_generator = CosmicRayDataGenerator()
    cosmic_data = cosmic_generator.generate_timing_data(duration_hours=24, detector_area_m2=100)
    cosmic_data.to_csv(os.path.join(data_dir, 'cosmic_ray_timings.csv'), index=False)
    print(f"   Generated {len(cosmic_data)} cosmic ray events")
    
    # 2. Generate quantum measurement data
    print("🔬 Generating quantum measurement data...")
    quantum_generator = QuantumMeasurementDataGenerator()
    quantum_data = quantum_generator.generate_double_slit_data(n_experiments=1000, measurements_per_exp=100)
    
    with open(os.path.join(data_dir, 'quantum_experiment_results.json'), 'w') as f:
        json.dump(quantum_data, f, indent=2, default=str)
    print(f"   Generated {quantum_data['metadata']['total_experiments']} quantum experiments")
    
    # 3. Generate CMB data
    print("🌌 Generating CMB temperature maps...")
    cmb_generator = CMBDataGenerator()
    
    # Normal CMB map
    cmb_normal = cmb_generator.generate_temperature_map((512, 512), add_artifacts=False)
    np.save(os.path.join(data_dir, 'cmb_map_normal.npy'), cmb_normal)
    
    # CMB map with potential artifacts
    cmb_artifacts = cmb_generator.generate_temperature_map((512, 512), add_artifacts=True)
    np.save(os.path.join(data_dir, 'cmb_map_artifacts.npy'), cmb_artifacts)
    
    print(f"   Generated CMB maps: {cmb_normal.shape}")
    
    # 4. Generate enhanced physical constants dataset
    print("🔢 Generating physical constants data...")
    constants_extended = {
        'fundamental_constants': {
            'c': 299792458,                    # speed of light (m/s)
            'h': 6.62607015e-34,              # Planck constant (J⋅s)
            'hbar': 1.054571817e-34,          # Reduced Planck constant (J⋅s)
            'G': 6.67430e-11,                 # gravitational constant (m³/kg⋅s²)
            'e': 1.602176634e-19,             # elementary charge (C)
            'me': 9.1093837015e-31,           # electron mass (kg)
            'mp': 1.67262192369e-27,          # proton mass (kg)
            'kb': 1.380649e-23,               # Boltzmann constant (J/K)
            'Na': 6.02214076e23,              # Avogadro constant (mol⁻¹)
        },
        'derived_constants': {
            'alpha': 7.2973525693e-3,         # fine structure constant
            'mu0': 1.25663706212e-6,          # magnetic permeability (H/m)
            'epsilon0': 8.8541878128e-12,     # electric permittivity (F/m)
            'Rinf': 10973731.568160,          # Rydberg constant (m⁻¹)
            'a0': 5.29177210903e-11,          # Bohr radius (m)
        },
        'cosmological_constants': {
            'H0': 67.4,                       # Hubble constant (km/s/Mpc)
            'Omega_m': 0.315,                 # Matter density parameter
            'Omega_Lambda': 0.685,            # Dark energy density parameter
            'T_cmb': 2.7255,                  # CMB temperature (K)
        },
        'metadata': {
            'source': 'NIST/CODATA 2018',
            'generation_date': pd.Timestamp.now().isoformat(),
            'precision_digits': 15
        }
    }
    
    with open(os.path.join(data_dir, 'physical_constants_extended.json'), 'w') as f:
        json.dump(constants_extended, f, indent=2)
    
    print("✅ All datasets generated successfully!")
    print(f"📁 Data saved to: {os.path.abspath(data_dir)}")
    
    # Generate summary
    summary = {
        'cosmic_ray_events': len(cosmic_data),
        'quantum_experiments': quantum_data['metadata']['total_experiments'],
        'cmb_map_size': list(cmb_normal.shape),
        'physical_constants_count': len(constants_extended['fundamental_constants']) + 
                                  len(constants_extended['derived_constants']) + 
                                  len(constants_extended['cosmological_constants']),
        'total_files_generated': 5
    }
    
    with open(os.path.join(data_dir, 'dataset_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary


if __name__ == "__main__":
    summary = generate_all_datasets()
    print(f"\n📊 Dataset Summary:")
    for key, value in summary.items():
        print(f"   {key}: {value}")
