"""
Signal Processing Module for EEG Data

This module provides various signal processing functions for EEG data analysis,
including filtering, artifact removal, and feature extraction.
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, welch
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class EEGSignalProcessor:
    """
    Signal processing class for EEG data analysis.
    """
    
    def __init__(self, sampling_rate: int = 250):
        """
        Initialize the signal processor.
        
        Args:
            sampling_rate: Sampling rate of the EEG data in Hz
        """
        self.sampling_rate = sampling_rate
        self.nyquist = sampling_rate / 2
    
    def butter_bandpass_filter(self, data: np.ndarray, 
                             lowcut: float, highcut: float, 
                             order: int = 4) -> np.ndarray:
        """
        Apply Butterworth bandpass filter to EEG data.
        
        Args:
            data: EEG data array (channels x samples)
            lowcut: Low cutoff frequency
            highcut: High cutoff frequency
            order: Filter order
            
        Returns:
            Filtered EEG data
        """
        try:
            # Normalize frequencies
            low = lowcut / self.nyquist
            high = highcut / self.nyquist
            
            # Design filter
            b, a = butter(order, [low, high], btype='band')
            
            # Apply filter to each channel
            filtered_data = np.zeros_like(data)
            for i in range(data.shape[0]):
                filtered_data[i, :] = filtfilt(b, a, data[i, :])
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"Error in bandpass filter: {e}")
            return data
    
    def butter_highpass_filter(self, data: np.ndarray, 
                              cutoff: float, order: int = 4) -> np.ndarray:
        """
        Apply Butterworth highpass filter to EEG data.
        
        Args:
            data: EEG data array (channels x samples)
            cutoff: Cutoff frequency
            order: Filter order
            
        Returns:
            Filtered EEG data
        """
        try:
            # Normalize frequency
            cutoff_norm = cutoff / self.nyquist
            
            # Design filter
            b, a = butter(order, cutoff_norm, btype='high')
            
            # Apply filter to each channel
            filtered_data = np.zeros_like(data)
            for i in range(data.shape[0]):
                filtered_data[i, :] = filtfilt(b, a, data[i, :])
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"Error in highpass filter: {e}")
            return data
    
    def butter_lowpass_filter(self, data: np.ndarray, 
                             cutoff: float, order: int = 4) -> np.ndarray:
        """
        Apply Butterworth lowpass filter to EEG data.
        
        Args:
            data: EEG data array (channels x samples)
            cutoff: Cutoff frequency
            order: Filter order
            
        Returns:
            Filtered EEG data
        """
        try:
            # Normalize frequency
            cutoff_norm = cutoff / self.nyquist
            
            # Design filter
            b, a = butter(order, cutoff_norm, btype='low')
            
            # Apply filter to each channel
            filtered_data = np.zeros_like(data)
            for i in range(data.shape[0]):
                filtered_data[i, :] = filtfilt(b, a, data[i, :])
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"Error in lowpass filter: {e}")
            return data
    
    def notch_filter(self, data: np.ndarray, 
                    notch_freq: float = 60.0, 
                    quality_factor: float = 30.0) -> np.ndarray:
        """
        Apply notch filter to remove power line noise.
        
        Args:
            data: EEG data array (channels x samples)
            notch_freq: Frequency to notch out
            quality_factor: Quality factor for the notch filter
            
        Returns:
            Filtered EEG data
        """
        try:
            # Normalize frequency
            notch_freq_norm = notch_freq / self.nyquist
            
            # Design notch filter
            b, a = signal.iirnotch(notch_freq_norm, quality_factor)
            
            # Apply filter to each channel
            filtered_data = np.zeros_like(data)
            for i in range(data.shape[0]):
                filtered_data[i, :] = filtfilt(b, a, data[i, :])
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"Error in notch filter: {e}")
            return data
    
    def remove_dc_offset(self, data: np.ndarray) -> np.ndarray:
        """
        Remove DC offset from EEG data.
        
        Args:
            data: EEG data array (channels x samples)
            
        Returns:
            EEG data with DC offset removed
        """
        try:
            # Remove mean from each channel
            filtered_data = data - np.mean(data, axis=1, keepdims=True)
            return filtered_data
            
        except Exception as e:
            logger.error(f"Error removing DC offset: {e}")
            return data
    
    def detect_artifacts(self, data: np.ndarray, 
                        threshold: float = 100.0) -> np.ndarray:
        """
        Detect artifacts in EEG data based on amplitude threshold.
        
        Args:
            data: EEG data array (channels x samples)
            threshold: Amplitude threshold for artifact detection
            
        Returns:
            Boolean array indicating artifact samples
        """
        try:
            # Calculate amplitude for each sample
            amplitude = np.abs(data)
            
            # Detect samples exceeding threshold
            artifacts = np.any(amplitude > threshold, axis=0)
            
            return artifacts
            
        except Exception as e:
            logger.error(f"Error detecting artifacts: {e}")
            return np.zeros(data.shape[1], dtype=bool)
    
    def interpolate_artifacts(self, data: np.ndarray, 
                             artifacts: np.ndarray) -> np.ndarray:
        """
        Interpolate artifact-contaminated samples.
        
        Args:
            data: EEG data array (channels x samples)
            artifacts: Boolean array indicating artifact samples
            
        Returns:
            EEG data with artifacts interpolated
        """
        try:
            filtered_data = data.copy()
            
            for i in range(data.shape[0]):
                if np.any(artifacts):
                    # Create time indices
                    time_indices = np.arange(data.shape[1])
                    
                    # Separate clean and artifact samples
                    clean_indices = time_indices[~artifacts]
                    artifact_indices = time_indices[artifacts]
                    
                    if len(clean_indices) > 0:
                        # Interpolate artifact samples
                        interpolated_values = np.interp(
                            artifact_indices, clean_indices, data[i, clean_indices]
                        )
                        filtered_data[i, artifact_indices] = interpolated_values
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"Error interpolating artifacts: {e}")
            return data
    
    def compute_power_spectrum(self, data: np.ndarray, 
                              window_length: int = 256,
                              overlap: int = 128) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute power spectrum of EEG data.
        
        Args:
            data: EEG data array (channels x samples)
            window_length: Length of window for FFT
            overlap: Overlap between windows
            
        Returns:
            Tuple of (frequencies, power_spectrum)
        """
        try:
            # Compute power spectrum for each channel
            frequencies = None
            power_spectra = []
            
            for i in range(data.shape[0]):
                f, psd = welch(data[i, :], 
                             fs=self.sampling_rate,
                             nperseg=window_length,
                             noverlap=overlap)
                
                if frequencies is None:
                    frequencies = f
                
                power_spectra.append(psd)
            
            return frequencies, np.array(power_spectra)
            
        except Exception as e:
            logger.error(f"Error computing power spectrum: {e}")
            return np.array([]), np.array([])
    
    def extract_band_power(self, data: np.ndarray, 
                          band: Tuple[float, float]) -> np.ndarray:
        """
        Extract power in a specific frequency band.
        
        Args:
            data: EEG data array (channels x samples)
            band: Tuple of (low_freq, high_freq)
            
        Returns:
            Band power for each channel
        """
        try:
            # Compute power spectrum
            frequencies, power_spectrum = self.compute_power_spectrum(data)
            
            if frequencies.size == 0:
                return np.zeros(data.shape[0])
            
            # Find frequency indices for the band
            band_mask = (frequencies >= band[0]) & (frequencies <= band[1])
            
            # Compute band power
            band_power = np.sum(power_spectrum[:, band_mask], axis=1)
            
            return band_power
            
        except Exception as e:
            logger.error(f"Error extracting band power: {e}")
            return np.zeros(data.shape[0])
    
    def extract_eeg_bands(self, data: np.ndarray) -> dict:
        """
        Extract power in standard EEG frequency bands.
        
        Args:
            data: EEG data array (channels x samples)
            
        Returns:
            Dictionary with band powers
        """
        try:
            bands = {
                'delta': (0.5, 4.0),
                'theta': (4.0, 8.0),
                'alpha': (8.0, 13.0),
                'beta': (13.0, 30.0),
                'gamma': (30.0, 50.0)
            }
            
            band_powers = {}
            for band_name, band_range in bands.items():
                band_powers[band_name] = self.extract_band_power(data, band_range)
            
            return band_powers
            
        except Exception as e:
            logger.error(f"Error extracting EEG bands: {e}")
            return {}
    
    def compute_spectral_features(self, data: np.ndarray) -> dict:
        """
        Compute various spectral features from EEG data.
        
        Args:
            data: EEG data array (channels x samples)
            
        Returns:
            Dictionary with spectral features
        """
        try:
            features = {}
            
            # Extract EEG bands
            band_powers = self.extract_eeg_bands(data)
            features.update(band_powers)
            
            # Compute total power
            frequencies, power_spectrum = self.compute_power_spectrum(data)
            if frequencies.size > 0:
                features['total_power'] = np.sum(power_spectrum, axis=1)
                
                # Compute spectral centroid
                features['spectral_centroid'] = np.sum(
                    frequencies.reshape(1, -1) * power_spectrum, axis=1
                ) / np.sum(power_spectrum, axis=1)
            
            return features
            
        except Exception as e:
            logger.error(f"Error computing spectral features: {e}")
            return {}
    
    def apply_complete_preprocessing(self, data: np.ndarray,
                                   remove_dc: bool = True,
                                   notch_freq: float = 60.0,
                                   bandpass_range: Tuple[float, float] = (1.0, 50.0),
                                   artifact_threshold: float = 100.0) -> np.ndarray:
        """
        Apply complete preprocessing pipeline to EEG data.
        
        Args:
            data: Raw EEG data array (channels x samples)
            remove_dc: Whether to remove DC offset
            notch_freq: Frequency for notch filter
            bandpass_range: Tuple of (low_freq, high_freq) for bandpass filter
            artifact_threshold: Threshold for artifact detection
            
        Returns:
            Preprocessed EEG data
        """
        try:
            processed_data = data.copy()
            
            # Remove DC offset
            if remove_dc:
                processed_data = self.remove_dc_offset(processed_data)
            
            # Apply notch filter
            if notch_freq > 0:
                processed_data = self.notch_filter(processed_data, notch_freq)
            
            # Apply bandpass filter
            if bandpass_range[0] > 0 and bandpass_range[1] > 0:
                processed_data = self.butter_bandpass_filter(
                    processed_data, bandpass_range[0], bandpass_range[1]
                )
            
            # Detect and interpolate artifacts
            artifacts = self.detect_artifacts(processed_data, artifact_threshold)
            if np.any(artifacts):
                processed_data = self.interpolate_artifacts(processed_data, artifacts)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error in complete preprocessing: {e}")
            return data


def test_signal_processor():
    """Test function for the signal processor."""
    # Create test data
    sampling_rate = 250
    duration = 10  # seconds
    t = np.linspace(0, duration, sampling_rate * duration)
    
    # Generate test signal with multiple frequency components
    test_signal = (np.sin(2 * np.pi * 10 * t) +  # Alpha
                  0.5 * np.sin(2 * np.pi * 25 * t) +  # Beta
                  0.3 * np.sin(2 * np.pi * 60 * t) +  # Power line noise
                  0.1 * np.random.randn(len(t)))  # Noise
    
    # Reshape for multi-channel
    test_data = np.tile(test_signal, (4, 1))  # 4 channels
    
    # Create processor
    processor = EEGSignalProcessor(sampling_rate)
    
    # Test preprocessing
    processed_data = processor.apply_complete_preprocessing(test_data)
    
    # Test feature extraction
    features = processor.compute_spectral_features(processed_data)
    
    print("Signal processor test completed successfully!")
    print(f"Extracted features: {list(features.keys())}")
    
    return processed_data, features


if __name__ == "__main__":
    test_signal_processor()


