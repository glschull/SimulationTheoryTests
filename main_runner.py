"""
Simulation Theory Test Suite - Main Runner
==========================================

Complete test suite for exploring the simulation hypothesis through
scientific analysis of physical data and computational signatures.

Usage:
    python main_runner.py [--generate-data] [--run-tests] [--visualize] [--all]

Author: AI Assistant for Garrett Schull's SimulationTheoryTests
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import our modules
try:
    from utils.real_data_loader import load_all_real_datasets
    from utils.analysis import SimulationHypothesisEvaluator
    from utils.visualization import create_comprehensive_report
    from tests.enhanced_simulation_tests import (
        QuantumObserverEffect, PlanckScaleDiscreteness, 
        PhysicalConstantAnalyzer, CosmicMicrowaveBackgroundAnalyzer
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all modules are properly installed and accessible.")
    sys.exit(1)

import numpy as np
import pandas as pd


class SimulationTheoryTestSuite:
    """Main test suite orchestrator"""
    
    def __init__(self):
        self.data_dir = project_root / "data"
        self.results_dir = project_root / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize test modules
        self.quantum_tester = QuantumObserverEffect()
        self.planck_tester = PlanckScaleDiscreteness()
        self.constants_analyzer = PhysicalConstantAnalyzer()
        self.cmb_analyzer = CosmicMicrowaveBackgroundAnalyzer()
        self.evaluator = SimulationHypothesisEvaluator()
        
        self.test_results = {}
    
    def generate_test_data(self):
        """Load real scientific datasets"""
        print("🔬 LOADING REAL SCIENTIFIC DATA")
        print("=" * 50)
        
        start_time = time.time()
        summary = load_all_real_datasets()
        
        print(f"✅ Real data loading completed in {time.time() - start_time:.2f} seconds")
        print(f"📁 Files saved to: {self.data_dir.absolute()}")
        
        return summary
    
    def run_quantum_tests(self):
        """Run quantum observer effect tests"""
        print("\n🔬 QUANTUM OBSERVER EFFECT TESTS")
        print("-" * 40)
        
        results = []
        
        # Test different observer probabilities
        observer_probs = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for prob in observer_probs:
            print(f"Testing observer probability: {prob:.1f}")
            result = self.quantum_tester.run_double_slit_experiment(
                trials=50000, observer_probability=prob
            )
            results.append(result)
            
            print(f"  Collapse Rate: {result['collapse_rate']:.4f}")
            print(f"  Interference Mean: {result['interference_mean']:.4f}")
        
        # Detect anomalies
        anomalies = self.quantum_tester.detect_measurement_anomalies()
        print(f"\n🚨 Anomalies: {anomalies['anomalies_found']}/{anomalies['total_experiments']}")
        
        self.test_results['quantum'] = {
            'experiments': results,
            'anomalies': anomalies
        }
        
        return results
    
    def run_planck_tests(self):
        """Run Planck-scale discreteness tests"""
        print("\n⚛️ PLANCK-SCALE DISCRETENESS TESTS")
        print("-" * 40)
        
        results = []
        
        # Test different data types
        test_cases = [
            ("Random Continuous", np.random.exponential(1e-9, 10000)),
            ("Cosmic Ray Simulation", self.planck_tester.generate_cosmic_ray_data(10000)),
            ("Quantized Test", np.arange(0, 1e-6, 1e-8) + np.random.normal(0, 1e-10, 100))
        ]
        
        for test_name, data in test_cases:
            print(f"\nAnalyzing: {test_name}")
            analysis = self.planck_tester.analyze_discreteness(data)
            
            result = {
                'name': test_name,
                'data': data,
                'analysis': analysis
            }
            results.append(result)
            
            print(f"  Discreteness Score: {analysis['discreteness_score']:.4f}")
            print(f"  Entropy: {analysis['entropy']:.4f}")
            print(f"  KS Test p-value: {analysis['ks_test']['p_value']:.6f}")
        
        self.test_results['planck'] = results
        return results
    
    def run_constants_tests(self):
        """Run physical constants compression tests"""
        print("\n🔢 PHYSICAL CONSTANTS ANALYSIS")
        print("-" * 40)
        
        results = self.constants_analyzer.comprehensive_compression_analysis()
        
        print("Individual Constants Compression:")
        for name, analysis in results.items():
            if name not in ['combined', 'relationships']:
                try:
                    best_ratio = min([
                        comp['zlib']['ratio'] for comp in analysis['compression'].values()
                        if 'zlib' in comp and 'ratio' in comp['zlib']
                    ])
                    print(f"  {name}: {best_ratio:.4f}")
                except (KeyError, ValueError):
                    print(f"  {name}: Analysis failed")
        
        print(f"\nCombined Analysis:")
        if 'combined' in results:
            for alg, comp in results['combined']['compression'].items():
                print(f"  {alg}: {comp['ratio']:.4f}")
        
        print(f"\nMathematical Relationships:")
        if 'relationships' in results:
            for name, rel in results['relationships'].items():
                print(f"  {name}: Error={rel['relative_error']:.2e}, Suspicious={rel['suspiciously_exact']}")
        
        self.test_results['constants'] = results
        return results
    
    def run_cmb_tests(self):
        """Run cosmic microwave background tests"""
        print("\n🌌 COSMIC MICROWAVE BACKGROUND TESTS")
        print("-" * 40)
        
        results = []
        
        # Test multiple CMB datasets
        for i in range(3):
            print(f"\nGenerating CMB dataset {i+1}...")
            
            # Generate with different artifact probabilities
            add_artifacts = i == 2  # Add artifacts to last dataset
            cmb_data = self.cmb_analyzer.generate_mock_cmb_data((256, 256))
            
            analysis = self.cmb_analyzer.search_for_artificial_signatures(cmb_data)
            
            result = {
                'dataset_id': i,
                'data': cmb_data,
                'analysis': analysis
            }
            results.append(result)
            
            print(f"  Anomaly Score: {analysis['anomaly_score']:.4f}")
            print(f"  Suspicious Patterns: {analysis['hash_signatures']['pattern_count']}")
            print(f"  Temperature σ: {analysis['statistical_analysis']['std_temperature']:.2e} K")
        
        self.test_results['cmb'] = results
        return results
    
    def run_ligo_tests(self):
        """Run LIGO gravitational wave tests for simulation signatures"""
        print("\n🌊 LIGO GRAVITATIONAL WAVE TESTS")
        print("-" * 40)
        
        try:
            # Import LIGO analysis
            from tests.ligo_simulation_tests import analyze_ligo_for_simulation_signatures
            from utils.ligo_data_loader import load_ligo_gravitational_wave_data
            
            # Load LIGO data if not already loaded
            ligo_data_file = self.data_dir / "ligo_gw_catalog.json"
            if not ligo_data_file.exists():
                print("Loading LIGO data...")
                ligo_data = load_ligo_gravitational_wave_data(self.data_dir)
            else:
                # Load existing LIGO data
                import json
                with open(ligo_data_file, 'r') as f:
                    catalog = json.load(f)
                
                # Reconstruct LIGO data structure
                from utils.ligo_data_loader import LIGOGravitationalWaveLoader
                loader = LIGOGravitationalWaveLoader(self.data_dir)
                strain_data = {}
                
                for event_name in loader.gw_events.keys():
                    strain_file = self.data_dir / f"ligo_strain_{event_name.lower()}.csv"
                    if strain_file.exists():
                        import pandas as pd
                        event_df = pd.read_csv(strain_file)
                        strain_data[event_name] = {
                            'time': event_df['time'].values,
                            'H1_strain': event_df['H1_strain'].values,
                            'L1_strain': event_df['L1_strain'].values,
                            'frequency': event_df['frequency'].values,
                            'event_params': loader.gw_events[event_name]
                        }
                
                ligo_data = {
                    'catalog': catalog,
                    'strain_data': strain_data
                }
            
            # Analyze LIGO data for simulation signatures
            ligo_analysis = analyze_ligo_for_simulation_signatures(ligo_data, save_visualizations=True)
            
            # Print summary
            summary = ligo_analysis.get('summary_statistics', {})
            print(f"\n🎯 LIGO Analysis Results:")
            print(f"  Events analyzed: {summary.get('total_events_analyzed', 0)}")
            print(f"  Mean discreteness score: {summary.get('mean_discreteness_score', 0):.4f}")
            print(f"  Overall spacetime signature: {summary.get('overall_discreteness_score', 0):.4f}")
            print(f"  Strain points analyzed: {summary.get('total_strain_points_analyzed', 0):,}")
            
            self.test_results['ligo'] = ligo_analysis
            return ligo_analysis
            
        except ImportError as e:
            print(f"  ❌ LIGO analysis not available: {e}")
            return None
        except Exception as e:
            print(f"  ❌ Error in LIGO analysis: {e}")
            return None
    
    def run_lhc_tests(self):
        """Run LHC particle collision tests for simulation signatures"""
        print("\n⚛️ LHC PARTICLE COLLISION TESTS")
        print("-" * 40)
        
        try:
            # Import LHC analysis modules
            from tests.lhc_simulation_tests import LHCSimulationTests
            from utils.lhc_data_loader import load_lhc_particle_collision_data
            
            # Load LHC particle collision data
            print("Loading LHC particle collision data...")
            lhc_data = load_lhc_particle_collision_data(self.data_dir)
            
            # Initialize LHC analyzer
            lhc_analyzer = LHCSimulationTests()
            
            # Run comprehensive LHC analysis
            lhc_analysis = lhc_analyzer.comprehensive_lhc_analysis(lhc_data['collision_data'])
            
            # Create visualizations
            viz_file = lhc_analyzer.create_lhc_visualizations(
                lhc_data['collision_data'], 
                lhc_analysis, 
                str(self.results_dir)
            )
            
            # Print summary
            summary = lhc_analysis.get('summary', {})
            print(f"\n🎯 LHC Analysis Results:")
            print(f"  Events analyzed: {summary.get('total_events_analyzed', 0):,}")
            print(f"  Particles analyzed: {summary.get('particles_analyzed', 0):,}")
            print(f"  Event types: {summary.get('event_types', 0)}")
            print(f"  Physics consistency: {summary.get('physics_consistency', 0):.1%}")
            print(f"  Combined LHC score: {lhc_analysis.get('combined_suspicion_score', 0):.4f}")
            print(f"  Visualization: {viz_file}")
            
            self.test_results['lhc'] = lhc_analysis
            return lhc_analysis
            
        except ImportError as e:
            print(f"  ❌ LHC analysis not available: {e}")
            return None
        except Exception as e:
            print(f"  ❌ Error in LHC analysis: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_astronomical_tests(self):
        """Run astronomical survey tests for cosmic simulation signatures"""
        print("\n🔭 ASTRONOMICAL SURVEY TESTS")
        print("-" * 30)
        
        try:
            # Import astronomical analysis modules
            from tests.astronomical_simulation_tests import AstronomicalSimulationTests
            from utils.astronomical_survey_loader import load_astronomical_survey_data
            
            # Load astronomical survey data
            print("Loading astronomical survey data...")
            survey_data = load_astronomical_survey_data(self.data_dir)
            
            # Initialize astronomical analyzer
            astro_analyzer = AstronomicalSimulationTests()
            
            # Run comprehensive astronomical analysis
            astro_analysis = astro_analyzer.comprehensive_astronomical_analysis(survey_data)
            
            # Create visualizations
            print("Creating astronomical visualizations...")
            # Note: Add visualization creation here when implemented
            
            # Print summary
            summary = astro_analysis.get('summary', {})
            print(f"\n🎯 Astronomical Analysis Results:")
            print(f"  Objects analyzed: {summary.get('total_objects_analyzed', 0):,}")
            print(f"  Stellar catalog: {summary.get('gaia_stars', 0):,} stars")
            print(f"  Hubble galaxies: {summary.get('hubble_galaxies', 0):,} objects")
            print(f"  JWST sources: {summary.get('jwst_sources', 0):,} sources")
            print(f"  Survey fields: {summary.get('survey_fields', 0)}")
            print(f"  Combined astronomy score: {astro_analysis.get('combined_suspicion_score', 0):.4f}")
            
            self.test_results['astronomical'] = astro_analysis
            return astro_analysis
            
        except ImportError as e:
            print(f"  ❌ Astronomical analysis not available: {e}")
            return None
        except Exception as e:
            print(f"  ❌ Error in astronomical analysis: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_advanced_ml_analysis(self):
        """Run advanced machine learning anomaly detection"""
        print("\n🤖 MACHINE LEARNING ANOMALY DETECTION")
        print("-" * 40)
        
        try:
            from utils.ml_anomaly_detection import SimulationAnomalyDetector
            
            # Collect data from all previous tests
            datasets = {}
            
            # Extract data from previous test results
            if 'quantum' in self.test_results:
                quantum_data = self.test_results['quantum']
                if 'measurement_data' in quantum_data:
                    datasets['quantum_measurements'] = np.array(quantum_data['measurement_data'])
            
            if 'planck' in self.test_results:
                planck_data = self.test_results['planck']
                if 'intervals' in planck_data:
                    datasets['planck_intervals'] = np.array(planck_data['intervals'])
            
            if 'constants' in self.test_results:
                constants_data = self.test_results['constants']
                if 'values' in constants_data:
                    datasets['physical_constants'] = np.array(list(constants_data['values'].values()))
            
            if 'cmb' in self.test_results:
                cmb_data = self.test_results['cmb']
                if 'temperature_data' in cmb_data:
                    datasets['cmb_temperatures'] = np.array(cmb_data['temperature_data']).flatten()
            
            if 'ligo' in self.test_results:
                ligo_data = self.test_results['ligo']
                if 'strain_data' in ligo_data:
                    datasets['gravitational_waves'] = np.array(ligo_data['strain_data']).flatten()
            
            print(f"  Collected {len(datasets)} datasets for ML analysis")
            
            # Initialize ML detector
            ml_detector = SimulationAnomalyDetector()
            
            # Run comprehensive ML analysis
            ml_analysis = ml_detector.comprehensive_anomaly_analysis(datasets)
            
            # Create visualizations
            print("  Creating ML visualizations...")
            viz_file = ml_detector.create_ml_visualizations(
                ml_analysis, str(self.results_dir)
            )
            
            # Print summary
            summary = ml_analysis.get('summary', {})
            print(f"\n🎯 ML Analysis Results:")
            print(f"  Features extracted: {summary.get('total_features', 0)}")
            print(f"  Datasets analyzed: {summary.get('datasets_analyzed', 0)}")
            print(f"  Models trained: {summary.get('models_trained', 0)}")
            print(f"  Anomaly probability: {summary.get('anomaly_probability', 0):.4f}")
            print(f"  Visualization: {viz_file}")
            
            self.test_results['ml_analysis'] = ml_analysis
            return ml_analysis
            
        except ImportError as e:
            print(f"  ❌ ML analysis not available: {e}")
            return None
        except Exception as e:
            print(f"  ❌ Error in ML analysis: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_quantum_information_analysis(self):
        """Run quantum information theoretic analysis"""
        print("\n🔬 QUANTUM INFORMATION ANALYSIS")
        print("-" * 35)
        
        try:
            from utils.quantum_information_analysis import QuantumInformationAnalyzer
            
            # Collect quantum-relevant data
            datasets = {}
            
            # Extract data from previous test results
            if 'quantum' in self.test_results:
                quantum_data = self.test_results['quantum']
                if 'measurement_data' in quantum_data:
                    datasets['bell_measurements'] = np.array(quantum_data['measurement_data'])
            
            if 'planck' in self.test_results:
                planck_data = self.test_results['planck']
                if 'intervals' in planck_data:
                    datasets['planck_intervals'] = np.array(planck_data['intervals'])
            
            if 'cmb' in self.test_results:
                cmb_data = self.test_results['cmb']
                if 'temperature_data' in cmb_data:
                    datasets['cmb_fluctuations'] = np.array(cmb_data['temperature_data']).flatten()
            
            if 'ligo' in self.test_results:
                ligo_data = self.test_results['ligo']
                if 'strain_data' in ligo_data:
                    datasets['spacetime_strain'] = np.array(ligo_data['strain_data']).flatten()
            
            print(f"  Collected {len(datasets)} datasets for quantum analysis")
            
            # Initialize quantum analyzer
            quantum_analyzer = QuantumInformationAnalyzer()
            
            # Run comprehensive quantum analysis
            quantum_analysis = quantum_analyzer.comprehensive_quantum_analysis(datasets)
            
            # Create visualizations
            print("  Creating quantum visualizations...")
            viz_file = quantum_analyzer.create_quantum_visualizations(
                quantum_analysis, str(self.results_dir)
            )
            
            # Print summary
            summary = quantum_analysis.get('summary', {})
            print(f"\n🎯 Quantum Analysis Results:")
            print(f"  Bell violations detected: {summary.get('bell_violations_count', 0)}")
            print(f"  Max Bell violation: {summary.get('max_bell_violation', 0):.4f}")
            print(f"  Quantum signature score: {summary.get('quantum_signature_score', 0):.4f}")
            print(f"  Information complexity: {summary.get('information_complexity', 0):.4f}")
            print(f"  Quantum computation probability: {summary.get('quantum_computation_probability', 0):.4f}")
            print(f"  Visualization: {viz_file}")
            
            self.test_results['quantum_information'] = quantum_analysis
            return quantum_analysis
            
        except ImportError as e:
            print(f"  ❌ Quantum information analysis not available: {e}")
            return None
        except Exception as e:
            print(f"  ❌ Error in quantum information analysis: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_comprehensive_analysis(self):
        """Run comprehensive statistical analysis across all datasets"""
        print("\n📊 COMPREHENSIVE STATISTICAL ANALYSIS")
        print("-" * 40)
        
        # Collect all numerical data for analysis
        datasets = {}
        
        # Quantum data
        if 'quantum' in self.test_results:
            quantum_positions = []
            for exp in self.test_results['quantum']['experiments']:
                # Extract meaningful numerical data
                quantum_positions.extend([
                    exp['collapse_rate'], 
                    exp['interference_mean'],
                    exp['interference_std']
                ])
            datasets['quantum_measurements'] = np.array(quantum_positions)
        
        # Planck data
        if 'planck' in self.test_results:
            planck_data = []
            for result in self.test_results['planck']:
                planck_data.extend(result['data'][:1000])  # Sample first 1000 points
            datasets['planck_intervals'] = np.array(planck_data)
        
        # Constants data
        if 'constants' in self.test_results:
            constants_values = []
            for name, analysis in self.test_results['constants'].items():
                if name not in ['combined', 'relationships'] and isinstance(analysis, dict):
                    if 'value' in analysis:
                        constants_values.append(np.log10(abs(analysis['value'])))
            if constants_values:
                datasets['physical_constants'] = np.array(constants_values)
        
        # CMB data
        if 'cmb' in self.test_results:
            cmb_temperatures = []
            for result in self.test_results['cmb']:
                cmb_flat = result['data'].flatten()
                cmb_temperatures.extend(cmb_flat[:1000])  # Sample
            datasets['cmb_temperatures'] = np.array(cmb_temperatures)
        
        # LIGO data
        if 'ligo' in self.test_results and self.test_results['ligo'] is not None:
            ligo_scores = []
            for event_result in self.test_results['ligo'].get('individual_events', {}).values():
                # Extract discreteness scores and other metrics
                h1_score = event_result['H1_analysis']['overall_discreteness_score']
                l1_score = event_result['L1_analysis']['overall_discreteness_score']
                combined_score = event_result['combined_analysis']['overall_discreteness_score']
                compression_h1 = event_result['H1_analysis']['compression_ratio']
                compression_l1 = event_result['L1_analysis']['compression_ratio']
                
                ligo_scores.extend([h1_score, l1_score, combined_score, compression_h1, compression_l1])
            
            if ligo_scores:
                datasets['gravitational_waves'] = np.array(ligo_scores)
        
        # LHC data
        if 'lhc' in self.test_results and self.test_results['lhc'] is not None:
            lhc_scores = []
            
            # Extract quantum digitization metrics
            quantum_dig = self.test_results['lhc']['quantum_digitization']
            lhc_scores.extend([
                quantum_dig['energy_discreteness'],
                quantum_dig['momentum_quantization'],
                quantum_dig['angular_uniformity'],
                quantum_dig['precision_clustering']
            ])
            
            # Extract conservation law metrics
            conservation = self.test_results['lhc']['conservation_laws']
            lhc_scores.extend([
                conservation['conservation_score'],
                conservation['violation_correlation']
            ])
            
            # Extract Standard Model consistency
            sm_analysis = self.test_results['lhc']['standard_model']
            lhc_scores.extend([
                sm_analysis['mass_deviation_avg'],
                sm_analysis['cross_section_deviation_avg'],
                sm_analysis['standard_model_score']
            ])
            
            # Add overall LHC score
            lhc_scores.append(self.test_results['lhc']['combined_suspicion_score'])
            
            datasets['particle_collisions'] = np.array(lhc_scores)
        
        # Run comprehensive analysis
        if datasets:
            print(f"Analyzing {len(datasets)} datasets...")
            comprehensive_results = self.evaluator.comprehensive_analysis(datasets)
            
            # Generate and display summary report
            report = self.evaluator.generate_summary_report(comprehensive_results)
            print(report)
            
            # Save detailed results
            results_file = self.results_dir / "comprehensive_analysis.json"
            with open(results_file, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_results = self._make_json_serializable(comprehensive_results)
                json.dump(serializable_results, f, indent=2)
            
            print(f"💾 Detailed results saved to: {results_file}")
            
            self.test_results['comprehensive'] = comprehensive_results
            return comprehensive_results
        else:
            print("❌ No datasets available for comprehensive analysis")
            return None
    
    def generate_visualizations(self):
        """Generate visualization reports"""
        print("\n🎨 GENERATING VISUALIZATIONS")
        print("-" * 40)
        
        try:
            # Prepare data for visualization
            quantum_results = self.test_results.get('quantum', {}).get('experiments', [])
            planck_results = self.test_results.get('planck', [])
            cmb_results = self.test_results.get('cmb', [])
            constants_results = self.test_results.get('constants', {})
            
            # Generate comprehensive report
            create_comprehensive_report(
                quantum_results=quantum_results,
                planck_results=planck_results,
                cmb_results=cmb_results,
                constants_results=constants_results,
                save_dir=str(self.results_dir)
            )
            
            print(f"✅ Visualizations saved to: {self.results_dir.absolute()}")
            
        except Exception as e:
            print(f"❌ Visualization generation failed: {e}")
            print("Some visualization features may require additional dependencies.")
    
    def save_all_results(self):
        """Save all test results to files"""
        print(f"\n💾 SAVING RESULTS")
        print("-" * 40)
        
        # Save main results
        results_file = self.results_dir / "simulation_test_results.json"
        with open(results_file, 'w') as f:
            serializable_results = self._make_json_serializable(self.test_results)
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to: {results_file}")
        
        # Generate summary file
        summary_file = self.results_dir / "test_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("SIMULATION THEORY TEST SUITE - SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            if 'comprehensive' in self.test_results:
                report = self.evaluator.generate_summary_report(self.test_results['comprehensive'])
                f.write(report)
            else:
                f.write("Comprehensive analysis not completed.\n")
        
        print(f"Summary saved to: {summary_file}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        else:
            return obj
    
    def run_full_suite(self, generate_data=True, visualize=True):
        """Run the complete test suite"""
        print("🚀 SIMULATION THEORY TEST SUITE")
        print("=" * 60)
        print("Exploring testable predictions of the simulation hypothesis")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Generate data if requested
            if generate_data:
                self.generate_test_data()
            
            # Run all tests
            self.run_quantum_tests()
            self.run_planck_tests()
            self.run_constants_tests()
            self.run_cmb_tests()
            self.run_ligo_tests()
            self.run_lhc_tests()
            self.run_astronomical_tests()
            
            # Advanced analysis methods
            self.run_advanced_ml_analysis()
            self.run_quantum_information_analysis()
            
            # Comprehensive analysis
            self.run_comprehensive_analysis()
            
            # Generate visualizations
            if visualize:
                self.generate_visualizations()
            
            # Save results
            self.save_all_results()
            
            total_time = time.time() - start_time
            
            print(f"\n🎯 SIMULATION HYPOTHESIS TEST COMPLETE")
            print("=" * 60)
            print(f"Total execution time: {total_time:.2f} seconds")
            print(f"Results directory: {self.results_dir.absolute()}")
            
            if 'comprehensive' in self.test_results:
                overall_score = self.test_results['comprehensive']['overall_assessment']['overall_suspicion_score']
                confidence = self.test_results['comprehensive']['overall_assessment']['confidence_level']
                
                print(f"\n🔍 FINAL ASSESSMENT:")
                print(f"Overall Suspicion Score: {overall_score:.3f}/1.000")
                print(f"Confidence Level: {confidence}")
            
            print(f"\n⚠️  Remember: These are exploratory tests designed to investigate")
            print(f"   the simulation hypothesis scientifically. Results should be")
            print(f"   interpreted carefully and are not definitive proof either way.")
            
        except Exception as e:
            print(f"\n❌ Test suite failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True


def main():
    """Main entry point with command line argument parsing"""
    parser = argparse.ArgumentParser(
        description='Simulation Theory Test Suite - Explore the simulation hypothesis scientifically'
    )
    
    parser.add_argument('--generate-data', action='store_true',
                       help='Generate test datasets')
    parser.add_argument('--run-tests', action='store_true',
                       help='Run simulation theory tests')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations')
    parser.add_argument('--all', action='store_true',
                       help='Run complete test suite (generate data, tests, and visualizations)')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Skip visualization generation')
    
    args = parser.parse_args()
    
    # Create test suite
    suite = SimulationTheoryTestSuite()
    
    # Determine what to run
    if args.all:
        suite.run_full_suite(generate_data=True, visualize=not args.no_visualize)
    elif args.generate_data:
        suite.generate_test_data()
    elif args.run_tests:
        suite.run_quantum_tests()
        suite.run_planck_tests()
        suite.run_constants_tests()
        suite.run_cmb_tests()
        suite.run_comprehensive_analysis()
        suite.save_all_results()
    elif args.visualize:
        # Load existing results if available
        results_file = suite.results_dir / "simulation_test_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                suite.test_results = json.load(f)
        suite.generate_visualizations()
    else:
        # Default: run everything
        suite.run_full_suite(generate_data=True, visualize=not args.no_visualize)


if __name__ == "__main__":
    main()
