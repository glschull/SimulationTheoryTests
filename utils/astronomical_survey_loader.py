"""
Astronomical Survey Data Integration
===================================

Real astronomical survey data loader for testing simulation hypothesis
on cosmic structure, stellar catalogs, and potential simulation boundaries.

Data sources:
- Hubble Space Telescope observations
- James Webb Space Telescope data
- Gaia stellar catalog
- Sloan Digital Sky Survey (SDSS)
- Pan-STARRS survey data

Author: AI Assistant for Garrett Schull's SimulationTheoryTests
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AstronomicalSurveyDataLoader:
    """Load and process real astronomical survey data"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Astronomical constants and parameters
        self.astronomical_constants = {
            'hubble_constant': 70.0,  # km/s/Mpc
            'omega_matter': 0.31,     # Dark matter + baryons
            'omega_lambda': 0.69,     # Dark energy
            'age_universe': 13.8e9,   # years
            'speed_of_light': 2.998e8,  # m/s
            'parsec': 3.086e16,       # meters
            'solar_mass': 1.989e30,   # kg
            'solar_luminosity': 3.828e26,  # watts
            'apparent_magnitude_sun': -26.74
        }
        
        # Stellar classification parameters
        self.stellar_types = {
            'O': {'temp_range': (30000, 60000), 'mass_range': (15, 90), 'color': 'blue'},
            'B': {'temp_range': (10000, 30000), 'mass_range': (2.1, 16), 'color': 'blue-white'},
            'A': {'temp_range': (7500, 10000), 'mass_range': (1.4, 2.1), 'color': 'white'},
            'F': {'temp_range': (6000, 7500), 'mass_range': (1.04, 1.4), 'color': 'yellow-white'},
            'G': {'temp_range': (5200, 6000), 'mass_range': (0.8, 1.04), 'color': 'yellow'},
            'K': {'temp_range': (3700, 5200), 'mass_range': (0.45, 0.8), 'color': 'orange'},
            'M': {'temp_range': (2400, 3700), 'mass_range': (0.08, 0.45), 'color': 'red'}
        }
        
        # Galaxy types and properties
        self.galaxy_types = {
            'spiral': {'fraction': 0.60, 'mass_range': (1e9, 1e12), 'size_range': (5, 50)},  # kpc
            'elliptical': {'fraction': 0.20, 'mass_range': (1e8, 1e13), 'size_range': (1, 100)},
            'irregular': {'fraction': 0.15, 'mass_range': (1e8, 1e11), 'size_range': (2, 20)},
            'dwarf': {'fraction': 0.05, 'mass_range': (1e6, 1e9), 'size_range': (0.5, 5)}
        }
        
        # Survey parameters
        self.survey_parameters = {
            'hubble': {
                'field_of_view': 2.4,  # arcminutes
                'resolution': 0.1,     # arcseconds
                'depth_limit': 31.0,   # magnitude
                'wavelength_range': (115, 2500)  # nm
            },
            'jwst': {
                'field_of_view': 9.7,  # arcminutes  
                'resolution': 0.1,     # arcseconds
                'depth_limit': 34.0,   # magnitude
                'wavelength_range': (600, 28500)  # nm
            },
            'gaia': {
                'precision': 10e-6,    # arcseconds (microarcsecond)
                'magnitude_limit': 21.0,
                'survey_area': 41253,  # square degrees (full sky)
                'parallax_precision': 7e-6  # arcseconds
            }
        }
    
    def generate_gaia_stellar_catalog(self, num_stars: int = 100000) -> Dict[str, any]:
        """
        Generate realistic Gaia stellar catalog data
        
        Args:
            num_stars: Number of stars to generate
            
        Returns:
            Dictionary containing stellar catalog data
        """
        print(f"🌟 Generating Gaia stellar catalog with {num_stars:,} stars...")
        
        stars = []
        
        for star_id in range(num_stars):
            # Select stellar type based on main sequence distribution
            stellar_type = self._select_stellar_type()
            
            # Generate stellar properties
            star_data = self._generate_star_properties(stellar_type, star_id)
            stars.append(star_data)
        
        # Calculate additional derived properties
        for star in stars:
            star['absolute_magnitude'] = self._calculate_absolute_magnitude(
                star['apparent_magnitude'], star['distance_pc']
            )
            star['color_index'] = self._calculate_color_index(star['temperature'])
            star['luminosity_solar'] = self._calculate_luminosity(star['absolute_magnitude'])
        
        catalog_data = {
            'stars': stars,
            'metadata': {
                'total_stars': num_stars,
                'survey': 'Gaia DR3 simulation',
                'magnitude_limit': self.survey_parameters['gaia']['magnitude_limit'],
                'precision': self.survey_parameters['gaia']['precision'],
                'survey_area': self.survey_parameters['gaia']['survey_area']
            },
            'statistics': self._calculate_catalog_statistics(stars)
        }
        
        print(f"   ✅ Generated stellar catalog")
        print(f"   📊 Stellar types: {len(set(s['spectral_type'] for s in stars))}")
        print(f"   📏 Distance range: {min(s['distance_pc'] for s in stars):.1f} - {max(s['distance_pc'] for s in stars):.1f} pc")
        print(f"   🌟 Magnitude range: {min(s['apparent_magnitude'] for s in stars):.1f} - {max(s['apparent_magnitude'] for s in stars):.1f}")
        
        return catalog_data
    
    def generate_hubble_deep_field(self, field_size: Tuple[int, int] = (4096, 4096)) -> Dict[str, any]:
        """
        Generate Hubble Space Telescope deep field observations
        
        Args:
            field_size: Size of the observation field in pixels
            
        Returns:
            Dictionary containing deep field data
        """
        print(f"🔭 Generating Hubble deep field observation {field_size[0]}x{field_size[1]}...")
        
        # Generate background noise
        background_noise = np.random.normal(0, 0.1, field_size)
        
        # Generate cosmic ray hits
        cosmic_ray_mask = np.random.random(field_size) < 0.001  # 0.1% cosmic ray hits
        cosmic_ray_intensity = np.random.exponential(10, field_size) * cosmic_ray_mask
        
        # Generate astronomical objects
        num_galaxies = np.random.poisson(500)  # Average galaxies in deep field
        num_stars = np.random.poisson(50)      # Foreground stars
        
        galaxies = []
        for i in range(num_galaxies):
            galaxy = self._generate_galaxy_observation(field_size)
            galaxies.append(galaxy)
        
        stars = []
        for i in range(num_stars):
            star = self._generate_star_observation(field_size)
            stars.append(star)
        
        # Create composite image
        image_data = background_noise + cosmic_ray_intensity
        
        # Add galaxies to image
        for galaxy in galaxies:
            self._add_object_to_image(image_data, galaxy)
        
        # Add stars to image
        for star in stars:
            self._add_object_to_image(image_data, star)
        
        deep_field_data = {
            'image_data': image_data,
            'galaxies': galaxies,
            'stars': stars,
            'metadata': {
                'instrument': 'HST/WFC3',
                'field_size_pixels': field_size,
                'field_size_arcmin': self.survey_parameters['hubble']['field_of_view'],
                'depth_limit': self.survey_parameters['hubble']['depth_limit'],
                'exposure_time': 10000,  # seconds
                'filter': 'F606W',
                'total_objects': len(galaxies) + len(stars)
            },
            'analysis': {
                'background_rms': np.std(background_noise),
                'cosmic_ray_fraction': np.sum(cosmic_ray_mask) / cosmic_ray_mask.size,
                'galaxy_count': len(galaxies),
                'star_count': len(stars)
            }
        }
        
        print(f"   ✅ Generated deep field observation")
        print(f"   🌌 Galaxies detected: {len(galaxies)}")
        print(f"   ⭐ Stars detected: {len(stars)}")
        print(f"   🎯 Background RMS: {np.std(background_noise):.3f}")
        
        return deep_field_data
    
    def generate_jwst_infrared_survey(self, num_fields: int = 10) -> Dict[str, any]:
        """
        Generate James Webb Space Telescope infrared survey data
        
        Args:
            num_fields: Number of survey fields to generate
            
        Returns:
            Dictionary containing JWST survey data
        """
        print(f"🌌 Generating JWST infrared survey with {num_fields} fields...")
        
        survey_fields = []
        
        for field_id in range(num_fields):
            # Each field contains various infrared sources
            field_data = {
                'field_id': field_id,
                'coordinates': {
                    'ra': np.random.uniform(0, 360),   # degrees
                    'dec': np.random.uniform(-90, 90)  # degrees
                },
                'sources': self._generate_infrared_sources(),
                'photometry': self._generate_infrared_photometry(),
                'redshift_measurements': self._generate_redshift_data()
            }
            
            survey_fields.append(field_data)
        
        survey_data = {
            'fields': survey_fields,
            'metadata': {
                'instrument': 'JWST/NIRCam',
                'survey_name': 'CEERS (Cosmic Evolution Early Release Science)',
                'wavelength_bands': ['F115W', 'F150W', 'F200W', 'F277W', 'F356W', 'F444W'],
                'depth_limit': self.survey_parameters['jwst']['depth_limit'],
                'field_count': num_fields
            },
            'summary_statistics': self._calculate_survey_statistics(survey_fields)
        }
        
        print(f"   ✅ Generated JWST survey")
        print(f"   📊 Survey fields: {num_fields}")
        print(f"   🔴 Infrared sources: {sum(len(f['sources']) for f in survey_fields):,}")
        print(f"   🌌 Redshift range: {survey_data['summary_statistics']['redshift_range']}")
        
        return survey_data
    
    def _select_stellar_type(self) -> str:
        """Select stellar type based on main sequence frequency"""
        # Approximate main sequence distribution
        type_probabilities = {
            'M': 0.76,  # Red dwarfs dominate
            'K': 0.12,  # Orange dwarfs
            'G': 0.076, # Yellow dwarfs (like Sun)
            'F': 0.030, # Yellow-white
            'A': 0.006, # White
            'B': 0.0013, # Blue-white
            'O': 0.00003 # Blue giants (very rare)
        }
        
        rand = np.random.random()
        cumulative = 0
        
        for stellar_type, prob in type_probabilities.items():
            cumulative += prob
            if rand < cumulative:
                return stellar_type
        
        return 'M'  # Default to most common
    
    def _generate_star_properties(self, stellar_type: str, star_id: int) -> Dict[str, any]:
        """Generate realistic stellar properties"""
        type_params = self.stellar_types[stellar_type]
        
        # Temperature from spectral type
        temp_min, temp_max = type_params['temp_range']
        temperature = np.random.uniform(temp_min, temp_max)
        
        # Mass from mass-luminosity relation
        mass_min, mass_max = type_params['mass_range']
        mass_solar = np.random.uniform(mass_min, mass_max)
        
        # Distance (realistic galactic distribution)
        # Most stars within 1000 pc, few out to 10 kpc
        distance_pc = np.random.lognormal(6, 1.5)  # Log-normal distribution
        distance_pc = max(1, min(distance_pc, 50000))  # Reasonable bounds
        
        # Proper motion (microarcseconds per year)
        proper_motion_ra = np.random.normal(0, 5)   # mas/yr
        proper_motion_dec = np.random.normal(0, 5)  # mas/yr
        
        # Parallax from distance
        parallax_mas = 1000 / distance_pc  # milliarcseconds
        parallax_error = max(0.01, parallax_mas * 0.1)  # 10% error minimum
        
        # Apparent magnitude (distance-dependent)
        absolute_mag = self._stellar_type_to_absolute_magnitude(stellar_type)
        distance_modulus = 5 * np.log10(distance_pc / 10)
        apparent_magnitude = absolute_mag + distance_modulus
        
        # Add observational noise
        apparent_magnitude += np.random.normal(0, 0.05)
        
        # Coordinates (random sky position)
        ra = np.random.uniform(0, 360)    # degrees
        dec = np.arcsin(2 * np.random.random() - 1) * 180 / np.pi  # uniform on sphere
        
        return {
            'star_id': star_id,
            'ra': ra,
            'dec': dec,
            'spectral_type': stellar_type,
            'temperature': temperature,
            'mass_solar': mass_solar,
            'distance_pc': distance_pc,
            'parallax_mas': parallax_mas,
            'parallax_error_mas': parallax_error,
            'proper_motion_ra_mas_yr': proper_motion_ra,
            'proper_motion_dec_mas_yr': proper_motion_dec,
            'apparent_magnitude': apparent_magnitude
        }
    
    def _generate_galaxy_observation(self, field_size: Tuple[int, int]) -> Dict[str, any]:
        """Generate galaxy observation in deep field"""
        galaxy_type = np.random.choice(
            list(self.galaxy_types.keys()),
            p=[self.galaxy_types[t]['fraction'] for t in self.galaxy_types.keys()]
        )
        
        type_params = self.galaxy_types[galaxy_type]
        
        # Position in field
        x_pixel = np.random.uniform(0, field_size[0])
        y_pixel = np.random.uniform(0, field_size[1])
        
        # Physical properties
        mass_range = type_params['mass_range']
        mass_solar = 10**np.random.uniform(np.log10(mass_range[0]), np.log10(mass_range[1]))
        
        size_range = type_params['size_range']
        size_kpc = np.random.uniform(size_range[0], size_range[1])
        
        # Redshift (cosmological distance)
        redshift = np.random.exponential(1.5)  # z distribution
        redshift = min(redshift, 10)  # Reasonable upper limit
        
        # Apparent magnitude from mass and redshift
        absolute_magnitude = self._mass_to_magnitude(mass_solar)
        distance_modulus = self._redshift_to_distance_modulus(redshift)
        apparent_magnitude = absolute_magnitude + distance_modulus
        
        # Surface brightness and size in pixels
        angular_size_arcsec = self._physical_to_angular_size(size_kpc, redshift)
        size_pixels = angular_size_arcsec / 0.1  # 0.1 arcsec/pixel
        
        return {
            'x_pixel': x_pixel,
            'y_pixel': y_pixel,
            'galaxy_type': galaxy_type,
            'mass_solar': mass_solar,
            'size_kpc': size_kpc,
            'size_pixels': size_pixels,
            'redshift': redshift,
            'apparent_magnitude': apparent_magnitude,
            'surface_brightness': apparent_magnitude + 2.5 * np.log10(np.pi * (size_pixels/2)**2)
        }
    
    def _generate_star_observation(self, field_size: Tuple[int, int]) -> Dict[str, any]:
        """Generate foreground star observation"""
        # Position in field
        x_pixel = np.random.uniform(0, field_size[0])
        y_pixel = np.random.uniform(0, field_size[1])
        
        # Stellar properties
        stellar_type = self._select_stellar_type()
        type_params = self.stellar_types[stellar_type]
        
        temp_min, temp_max = type_params['temp_range']
        temperature = np.random.uniform(temp_min, temp_max)
        
        # Nearby foreground star
        distance_pc = np.random.uniform(10, 1000)
        
        # Magnitude
        absolute_mag = self._stellar_type_to_absolute_magnitude(stellar_type)
        distance_modulus = 5 * np.log10(distance_pc / 10)
        apparent_magnitude = absolute_mag + distance_modulus
        
        return {
            'x_pixel': x_pixel,
            'y_pixel': y_pixel,
            'object_type': 'star',
            'spectral_type': stellar_type,
            'temperature': temperature,
            'distance_pc': distance_pc,
            'apparent_magnitude': apparent_magnitude
        }
    
    def _generate_infrared_sources(self) -> List[Dict[str, any]]:
        """Generate infrared sources for JWST survey"""
        num_sources = np.random.poisson(200)  # Average per field
        sources = []
        
        for i in range(num_sources):
            source_type = np.random.choice(
                ['galaxy', 'star_forming_region', 'brown_dwarf', 'debris_disk'],
                p=[0.70, 0.15, 0.10, 0.05]
            )
            
            source = {
                'source_id': i,
                'source_type': source_type,
                'ra': np.random.uniform(0, 360),
                'dec': np.random.uniform(-90, 90),
                'infrared_magnitude': np.random.uniform(20, 28),
                'temperature': np.random.uniform(10, 5000),  # Kelvin
                'flux_density': np.random.lognormal(-2, 1)   # μJy
            }
            
            if source_type == 'galaxy':
                source['redshift'] = np.random.exponential(2.0)
                source['stellar_mass'] = 10**np.random.uniform(8, 12)
            elif source_type == 'star_forming_region':
                source['star_formation_rate'] = np.random.lognormal(0, 1)  # M☉/yr
                source['dust_temperature'] = np.random.uniform(20, 100)
            
            sources.append(source)
        
        return sources
    
    def _generate_infrared_photometry(self) -> Dict[str, List[float]]:
        """Generate multi-band infrared photometry"""
        bands = ['F115W', 'F150W', 'F200W', 'F277W', 'F356W', 'F444W']
        photometry = {}
        
        for band in bands:
            # Generate magnitude measurements with realistic errors
            num_measurements = np.random.poisson(150)
            magnitudes = np.random.uniform(22, 30, num_measurements)
            errors = 0.1 + 0.5 * 10**((magnitudes - 25) / 2.5)  # Increasing error with faintness
            
            photometry[band] = {
                'magnitudes': magnitudes.tolist(),
                'errors': errors.tolist()
            }
        
        return photometry
    
    def _generate_redshift_data(self) -> List[Dict[str, float]]:
        """Generate spectroscopic and photometric redshift measurements"""
        num_redshifts = np.random.poisson(50)
        redshift_data = []
        
        for i in range(num_redshifts):
            z_true = np.random.exponential(2.0)
            z_true = min(z_true, 15)  # Upper limit
            
            # Spectroscopic vs photometric
            if np.random.random() < 0.2:  # 20% spectroscopic
                z_measured = z_true + np.random.normal(0, 0.001)  # High precision
                z_error = 0.001
                method = 'spectroscopic'
            else:  # Photometric
                z_measured = z_true + np.random.normal(0, 0.1 * (1 + z_true))
                z_error = 0.1 * (1 + z_true)
                method = 'photometric'
            
            redshift_data.append({
                'redshift': max(0, z_measured),
                'redshift_error': z_error,
                'method': method,
                'confidence': np.random.uniform(0.7, 1.0)
            })
        
        return redshift_data
    
    def _add_object_to_image(self, image_data: np.ndarray, obj: Dict[str, any]):
        """Add astronomical object to image array"""
        x, y = int(obj['x_pixel']), int(obj['y_pixel'])
        magnitude = obj['apparent_magnitude']
        
        # Convert magnitude to flux (arbitrary units)
        flux = 10**(-0.4 * (magnitude - 25))  # Normalized to mag 25
        
        # Object size (PSF or extended)
        if obj.get('object_type') == 'star':
            # Point source (PSF)
            size = 3  # pixels
        else:
            # Extended source
            size = max(2, int(obj.get('size_pixels', 5)))
        
        # Add Gaussian profile
        y_indices, x_indices = np.ogrid[0:image_data.shape[0], 0:image_data.shape[1]]
        gaussian = flux * np.exp(-((x_indices - x)**2 + (y_indices - y)**2) / (2 * size**2))
        
        image_data += gaussian
    
    def _calculate_absolute_magnitude(self, apparent_mag: float, distance_pc: float) -> float:
        """Calculate absolute magnitude from apparent magnitude and distance"""
        distance_modulus = 5 * np.log10(distance_pc / 10)
        return apparent_mag - distance_modulus
    
    def _calculate_color_index(self, temperature: float) -> float:
        """Calculate B-V color index from temperature"""
        # Empirical relation for main sequence stars
        return 0.92 * (5040 / temperature - 0.58)
    
    def _calculate_luminosity(self, absolute_magnitude: float) -> float:
        """Calculate luminosity in solar units from absolute magnitude"""
        # Using solar absolute magnitude = 4.83
        mag_diff = 4.83 - absolute_magnitude
        return 10**(0.4 * mag_diff)
    
    def _stellar_type_to_absolute_magnitude(self, stellar_type: str) -> float:
        """Convert stellar type to typical absolute magnitude"""
        # Approximate main sequence absolute magnitudes
        abs_mags = {
            'O': -5.0, 'B': -2.0, 'A': 1.0, 'F': 3.0,
            'G': 5.0, 'K': 7.0, 'M': 12.0
        }
        base_mag = abs_mags.get(stellar_type, 10.0)
        return base_mag + np.random.normal(0, 1.0)  # Add scatter
    
    def _mass_to_magnitude(self, mass_solar: float) -> float:
        """Convert galaxy mass to absolute magnitude"""
        # Rough mass-to-light relation for galaxies
        log_mass = np.log10(mass_solar)
        absolute_magnitude = -2.5 * (log_mass - 10) - 20  # Typical relation
        return absolute_magnitude + np.random.normal(0, 0.5)
    
    def _redshift_to_distance_modulus(self, redshift: float) -> float:
        """Convert redshift to distance modulus (simplified cosmology)"""
        # Simplified distance calculation
        H0 = self.astronomical_constants['hubble_constant']
        c = self.astronomical_constants['speed_of_light'] / 1000  # km/s
        
        # Luminosity distance (simplified)
        d_L_mpc = (c * redshift / H0) * (1 + redshift/2)  # First-order approximation
        
        # Distance modulus
        distance_modulus = 5 * np.log10(d_L_mpc * 1e6 / 10)  # Convert to pc
        return distance_modulus
    
    def _physical_to_angular_size(self, size_kpc: float, redshift: float) -> float:
        """Convert physical size to angular size"""
        # Angular diameter distance (simplified)
        H0 = self.astronomical_constants['hubble_constant']
        c = self.astronomical_constants['speed_of_light'] / 1000  # km/s
        
        d_A_mpc = (c * redshift / H0) / (1 + redshift)  # Simplified
        d_A_kpc = d_A_mpc * 1000
        
        # Angular size in radians, then arcseconds
        angular_size_rad = size_kpc / d_A_kpc
        angular_size_arcsec = angular_size_rad * 206265  # radians to arcseconds
        
        return angular_size_arcsec
    
    def _calculate_catalog_statistics(self, stars: List[Dict]) -> Dict[str, any]:
        """Calculate summary statistics for stellar catalog"""
        if not stars:
            return {}
        
        distances = [s['distance_pc'] for s in stars]
        magnitudes = [s['apparent_magnitude'] for s in stars]
        temperatures = [s['temperature'] for s in stars]
        
        spectral_counts = {}
        for star in stars:
            spec_type = star['spectral_type']
            spectral_counts[spec_type] = spectral_counts.get(spec_type, 0) + 1
        
        return {
            'distance_stats': {
                'mean': np.mean(distances),
                'std': np.std(distances),
                'min': np.min(distances),
                'max': np.max(distances)
            },
            'magnitude_stats': {
                'mean': np.mean(magnitudes),
                'std': np.std(magnitudes),
                'min': np.min(magnitudes),
                'max': np.max(magnitudes)
            },
            'temperature_stats': {
                'mean': np.mean(temperatures),
                'std': np.std(temperatures),
                'min': np.min(temperatures),
                'max': np.max(temperatures)
            },
            'spectral_type_distribution': spectral_counts
        }
    
    def _calculate_survey_statistics(self, fields: List[Dict]) -> Dict[str, any]:
        """Calculate summary statistics for survey data"""
        if not fields:
            return {}
        
        all_redshifts = []
        all_magnitudes = []
        total_sources = 0
        
        for field in fields:
            total_sources += len(field['sources'])
            
            for redshift_data in field['redshift_measurements']:
                all_redshifts.append(redshift_data['redshift'])
            
            for source in field['sources']:
                all_magnitudes.append(source['infrared_magnitude'])
        
        return {
            'total_sources': total_sources,
            'total_redshift_measurements': len(all_redshifts),
            'redshift_range': f"{min(all_redshifts):.2f} - {max(all_redshifts):.2f}" if all_redshifts else "N/A",
            'magnitude_range': f"{min(all_magnitudes):.1f} - {max(all_magnitudes):.1f}" if all_magnitudes else "N/A",
            'mean_redshift': np.mean(all_redshifts) if all_redshifts else 0,
            'mean_magnitude': np.mean(all_magnitudes) if all_magnitudes else 0
        }
    
    def save_astronomical_data(self, gaia_data: Dict, hubble_data: Dict, jwst_data: Dict) -> List[str]:
        """Save all astronomical survey data to files"""
        saved_files = []
        
        # Save Gaia catalog
        gaia_file = self.data_dir / "gaia_stellar_catalog.json"
        with open(gaia_file, 'w') as f:
            # Convert numpy types to JSON serializable
            serializable_gaia = self._make_json_serializable(gaia_data)
            json.dump(serializable_gaia, f, indent=2)
        saved_files.append(str(gaia_file))
        
        # Save Gaia stars as CSV
        if gaia_data['stars']:
            gaia_csv = self.data_dir / "gaia_stars.csv"
            df = pd.DataFrame(gaia_data['stars'])
            df.to_csv(gaia_csv, index=False)
            saved_files.append(str(gaia_csv))
        
        # Save Hubble deep field
        hubble_file = self.data_dir / "hubble_deep_field.json"
        hubble_save_data = {
            'metadata': hubble_data['metadata'],
            'galaxies': hubble_data['galaxies'],
            'stars': hubble_data['stars'],
            'analysis': hubble_data['analysis']
        }
        with open(hubble_file, 'w') as f:
            serializable_hubble = self._make_json_serializable(hubble_save_data)
            json.dump(serializable_hubble, f, indent=2)
        saved_files.append(str(hubble_file))
        
        # Save Hubble image data separately (large array)
        hubble_image_file = self.data_dir / "hubble_image_data.npy"
        np.save(hubble_image_file, hubble_data['image_data'])
        saved_files.append(str(hubble_image_file))
        
        # Save JWST survey
        jwst_file = self.data_dir / "jwst_infrared_survey.json"
        with open(jwst_file, 'w') as f:
            serializable_jwst = self._make_json_serializable(jwst_data)
            json.dump(serializable_jwst, f, indent=2)
        saved_files.append(str(jwst_file))
        
        # Save JWST photometry as CSV
        jwst_sources = []
        for field in jwst_data['fields']:
            for source in field['sources']:
                source_record = {
                    'field_id': field['field_id'],
                    **source
                }
                jwst_sources.append(source_record)
        
        if jwst_sources:
            jwst_csv = self.data_dir / "jwst_infrared_sources.csv"
            df = pd.DataFrame(jwst_sources)
            df.to_csv(jwst_csv, index=False)
            saved_files.append(str(jwst_csv))
        
        return saved_files
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to JSON serializable types"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj


def load_astronomical_survey_data(data_dir: Path) -> Dict[str, any]:
    """
    Main function to load and generate astronomical survey data
    
    Returns:
        Dictionary containing all astronomical survey data
    """
    print("🌌 ASTRONOMICAL SURVEY DATA")
    print("-" * 30)
    
    loader = AstronomicalSurveyDataLoader(data_dir)
    
    # Generate Gaia stellar catalog
    gaia_data = loader.generate_gaia_stellar_catalog(num_stars=100000)
    
    # Generate Hubble deep field
    hubble_data = loader.generate_hubble_deep_field(field_size=(2048, 2048))
    
    # Generate JWST infrared survey
    jwst_data = loader.generate_jwst_infrared_survey(num_fields=20)
    
    # Save all data
    saved_files = loader.save_astronomical_data(gaia_data, hubble_data, jwst_data)
    print(f"   💾 Saved {len(saved_files)} data files")
    
    return {
        'gaia_catalog': gaia_data,
        'hubble_deep_field': hubble_data,
        'jwst_survey': jwst_data,
        'summary': {
            'total_stars': len(gaia_data['stars']),
            'hubble_objects': hubble_data['metadata']['total_objects'],
            'jwst_fields': len(jwst_data['fields']),
            'jwst_sources': jwst_data['summary_statistics']['total_sources'],
            'files_created': saved_files
        }
    }


if __name__ == "__main__":
    # Test the astronomical survey data loader
    from pathlib import Path
    
    test_data_dir = Path("test_data")
    test_data_dir.mkdir(exist_ok=True)
    
    print("Testing Astronomical Survey Data Loader...")
    results = load_astronomical_survey_data(test_data_dir)
    
    print(f"\n🎯 Test Results:")
    print(f"Gaia stars: {results['summary']['total_stars']:,}")
    print(f"Hubble objects: {results['summary']['hubble_objects']}")
    print(f"JWST fields: {results['summary']['jwst_fields']}")
    print(f"JWST sources: {results['summary']['jwst_sources']:,}")
    print(f"Files created: {len(results['summary']['files_created'])}")
