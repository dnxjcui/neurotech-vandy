"""
Comprehensive Debug Script for OpenBCI EEG Interface

This script tests all components with NumPy to ensure everything works correctly.
"""

import sys
import os
import time
import traceback

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_numpy_import():
    """Test NumPy import and basic functionality"""
    print("=" * 60)
    print("TESTING NUMPY IMPORT AND FUNCTIONALITY")
    print("=" * 60)
    
    try:
        import numpy as np
        print(f"[OK] NumPy imported successfully - Version: {np.__version__}")
        
        # Test basic NumPy operations
        test_array = np.array([1, 2, 3, 4, 5])
        print(f"[OK] Array creation: {test_array}")
        print(f"[OK] Array shape: {test_array.shape}")
        print(f"[OK] Array dtype: {test_array.dtype}")
        
        # Test mathematical operations
        result = np.sin(test_array)
        print(f"[OK] Mathematical operations: sin({test_array}) = {result}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] NumPy test failed: {e}")
        traceback.print_exc()
        return False

def test_data_acquisition_module():
    """Test data acquisition module with NumPy"""
    print("\n" + "=" * 60)
    print("TESTING DATA ACQUISITION MODULE")
    print("=" * 60)
    
    try:
        import numpy as np
        from data_acquisition import OpenBCIDataAcquisition, get_available_ports
        
        print("[OK] Data acquisition module imported successfully")
        
        # Test port detection
        ports = get_available_ports()
        print(f"[OK] Available COM ports: {ports}")
        
        # Test class initialization (without connecting)
        acquisition = OpenBCIDataAcquisition(serial_port="COM3")
        print("[OK] OpenBCIDataAcquisition class initialized")
        
        # Test board info
        info = acquisition.get_board_info()
        print(f"[OK] Board info: {info}")
        
        # Test data callback functionality
        received_data = []
        def test_callback(data):
            received_data.append(data.copy())
            print(f"[OK] Data callback received data shape: {data.shape}")
        
        acquisition.set_data_callback(test_callback)
        print("[OK] Data callback set successfully")
        
        # Test mock data processing
        mock_data = np.random.randn(8, 100)  # 8 channels, 100 samples
        acquisition._process_data(mock_data)
        print(f"[OK] Mock data processing successful, received {len(received_data)} data packets")
        
        # Test filtering functions
        filtered_data = acquisition.apply_notch_filter(mock_data)
        print(f"[OK] Notch filter applied, output shape: {filtered_data.shape}")
        
        bandpass_data = acquisition.apply_bandpass_filter(mock_data, 1.0, 50.0)
        print(f"[OK] Bandpass filter applied, output shape: {bandpass_data.shape}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Data acquisition test failed: {e}")
        traceback.print_exc()
        return False

def test_signal_processing_module():
    """Test signal processing module with NumPy"""
    print("\n" + "=" * 60)
    print("TESTING SIGNAL PROCESSING MODULE")
    print("=" * 60)
    
    try:
        import numpy as np
        from signal_processing.eeg_processor import EEGSignalProcessor
        
        print("[OK] Signal processing module imported successfully")
        
        # Create processor
        processor = EEGSignalProcessor(sampling_rate=250)
        print("[OK] EEGSignalProcessor initialized")
        
        # Generate test EEG data
        duration = 2  # seconds
        num_samples = 250 * duration  # 500 samples
        num_channels = 4
        
        # Create realistic EEG-like signal
        t = np.linspace(0, duration, num_samples)
        eeg_data = np.zeros((num_channels, num_samples))
        
        for i in range(num_channels):
            # Generate different frequency components
            alpha = 2.0 * np.sin(2 * np.pi * 10 * t)      # Alpha waves (10 Hz)
            beta = 1.0 * np.sin(2 * np.pi * 20 * t)        # Beta waves (20 Hz)
            noise = 0.2 * np.random.randn(num_samples)     # Random noise
            
            # Combine components
            eeg_data[i, :] = alpha + beta + noise
        
        print(f"[OK] Test EEG data generated: {eeg_data.shape}")
        
        # Test preprocessing pipeline
        processed_data = processor.apply_complete_preprocessing(eeg_data)
        print(f"[OK] Complete preprocessing applied, output shape: {processed_data.shape}")
        
        # Test individual filters
        notch_filtered = processor.notch_filter(eeg_data, 60.0)
        print(f"[OK] Notch filter test passed, output shape: {notch_filtered.shape}")
        
        bandpass_filtered = processor.butter_bandpass_filter(eeg_data, 1.0, 50.0)
        print(f"[OK] Bandpass filter test passed, output shape: {bandpass_filtered.shape}")
        
        # Test feature extraction
        features = processor.compute_spectral_features(eeg_data)
        print(f"[OK] Spectral features extracted: {list(features.keys())}")
        
        # Test EEG band extraction
        band_powers = processor.extract_eeg_bands(eeg_data)
        print(f"[OK] EEG bands extracted: {list(band_powers.keys())}")
        
        # Test artifact detection
        artifacts = processor.detect_artifacts(eeg_data, threshold=100.0)
        print(f"[OK] Artifact detection completed, found {np.sum(artifacts)} artifacts")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Signal processing test failed: {e}")
        traceback.print_exc()
        return False

def test_gui_imports():
    """Test GUI module imports"""
    print("\n" + "=" * 60)
    print("TESTING GUI MODULE IMPORTS")
    print("=" * 60)
    
    try:
        # Test PyQt5
        from PyQt5.QtWidgets import QApplication, QMainWindow
        from PyQt5.QtCore import QTimer
        from PyQt5.QtGui import QFont
        print("[OK] PyQt5 imported successfully")
        
        # Test PyQtGraph
        import pyqtgraph as pg
        from pyqtgraph import PlotWidget
        print(f"[OK] PyQtGraph imported successfully - Version: {pg.__version__}")
        
        # Test our GUI module
        from gui.main_window import OpenBCIMainWindow, ControlPanel, EEGPlotWidget
        print("[OK] GUI main window classes imported successfully")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] GUI import test failed: {e}")
        traceback.print_exc()
        return False

def test_data_saving():
    """Test data saving functionality"""
    print("\n" + "=" * 60)
    print("TESTING DATA SAVING FUNCTIONALITY")
    print("=" * 60)
    
    try:
        import numpy as np
        
        # Create test data
        sampling_rate = 250
        duration = 2  # seconds
        num_samples = sampling_rate * duration
        num_channels = 4
        
        # Generate test EEG data
        t = np.linspace(0, duration, num_samples)
        eeg_data = np.zeros((num_channels, num_samples))
        
        for i in range(num_channels):
            alpha = 2.0 * np.sin(2 * np.pi * 10 * t)
            beta = 1.0 * np.sin(2 * np.pi * 20 * t)
            noise = 0.2 * np.random.randn(num_samples)
            eeg_data[i, :] = alpha + beta + noise
        
        # Test saving to .npz format
        test_filename = "test_eeg_data.npz"
        np.savez(test_filename,
                eeg_data=eeg_data,
                sampling_rate=sampling_rate,
                channel_names=['Fp1', 'Fp2', 'C3', 'C4'],
                timestamp=time.time())
        
        print(f"[OK] Data saved to {test_filename}")
        
        # Test loading the data
        loaded_data = np.load(test_filename)
        loaded_eeg = loaded_data['eeg_data']
        loaded_sampling_rate = loaded_data['sampling_rate']
        loaded_channels = loaded_data['channel_names']
        
        print(f"[OK] Data loaded successfully")
        print(f"[OK] Loaded data shape: {loaded_eeg.shape}")
        print(f"[OK] Loaded sampling rate: {loaded_sampling_rate}")
        print(f"[OK] Loaded channel names: {loaded_channels}")
        
        # Verify data integrity
        if np.array_equal(eeg_data, loaded_eeg):
            print("[OK] Data integrity verified - saved and loaded data match")
        else:
            print("[ERROR] Data integrity check failed")
            return False
        
        # Clean up test file (with error handling)
        try:
            os.remove(test_filename)
            print("[OK] Test file cleaned up")
        except PermissionError:
            print("[WARNING] Could not delete test file (may be in use)")
        except Exception as e:
            print(f"[WARNING] Could not delete test file: {e}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Data saving test failed: {e}")
        traceback.print_exc()
        return False

def test_complete_workflow():
    """Test complete workflow simulation"""
    print("\n" + "=" * 60)
    print("TESTING COMPLETE WORKFLOW SIMULATION")
    print("=" * 60)
    
    try:
        import numpy as np
        from signal_processing.eeg_processor import EEGSignalProcessor
        
        print("[OK] Starting complete workflow simulation...")
        
        # Simulate real-time EEG data acquisition
        sampling_rate = 250
        duration = 5  # seconds
        num_samples = sampling_rate * duration
        num_channels = 4
        
        # Generate realistic EEG data
        t = np.linspace(0, duration, num_samples)
        eeg_data = np.zeros((num_channels, num_samples))
        
        for i in range(num_channels):
            # Different frequency components for each channel
            alpha = 2.0 * np.sin(2 * np.pi * (8 + i) * t)      # Alpha waves
            beta = 1.0 * np.sin(2 * np.pi * (20 + i) * t)       # Beta waves
            theta = 0.5 * np.sin(2 * np.pi * (4 + i) * t)       # Theta waves
            noise = 0.1 * np.random.randn(num_samples)           # Random noise
            
            eeg_data[i, :] = alpha + beta + theta + noise
        
        print(f"[OK] Generated {duration}s of EEG data: {eeg_data.shape}")
        
        # Process the data
        processor = EEGSignalProcessor(sampling_rate)
        
        # Apply preprocessing
        processed_data = processor.apply_complete_preprocessing(eeg_data)
        print(f"[OK] Preprocessing completed: {processed_data.shape}")
        
        # Extract features
        features = processor.compute_spectral_features(processed_data)
        print(f"[OK] Features extracted: {list(features.keys())}")
        
        # Display some results
        print("\n[INFO] EEG Band Powers (average across channels):")
        for band, power in features.items():
            if isinstance(power, np.ndarray) and power.size > 0:
                avg_power = np.mean(power)
                print(f"  {band}: {avg_power:.3f}")
        
        # Simulate data recording
        recording_data = []
        batch_size = 50  # Process in batches
        
        for i in range(0, num_samples, batch_size):
            batch = eeg_data[:, i:i+batch_size]
            if batch.shape[1] > 0:  # Make sure we have data
                recording_data.append(batch)
        
        print(f"[OK] Simulated recording: {len(recording_data)} batches")
        
        # Concatenate recorded data
        if recording_data:
            recorded_data = np.concatenate(recording_data, axis=1)
            print(f"[OK] Recorded data shape: {recorded_data.shape}")
            
            # Verify data integrity
            if np.array_equal(eeg_data, recorded_data):
                print("[OK] Recording simulation successful - data integrity maintained")
            else:
                print("[ERROR] Recording simulation failed - data mismatch")
                return False
        
        print("[OK] Complete workflow simulation successful!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Complete workflow test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main debug function"""
    print("OpenBCI EEG Interface - Comprehensive Debug Test")
    print("Testing all components with NumPy")
    print("=" * 80)
    
    tests = [
        ("NumPy Import", test_numpy_import),
        ("Data Acquisition Module", test_data_acquisition_module),
        ("Signal Processing Module", test_signal_processing_module),
        ("GUI Imports", test_gui_imports),
        ("Data Saving", test_data_saving),
        ("Complete Workflow", test_complete_workflow)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
                print(f"\n[SUCCESS] {test_name} - PASSED")
            else:
                print(f"\n[FAILED] {test_name} - FAILED")
        except Exception as e:
            print(f"\n[ERROR] {test_name} - EXCEPTION: {e}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("DEBUG TEST SUMMARY")
    print("=" * 80)
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("\n[SUCCESS] ALL TESTS PASSED! The OpenBCI EEG Interface is ready to use!")
        print("\nNext steps:")
        print("1. Connect your OpenBCI Cyton board")
        print("2. Run: py -3.11 src/main.py")
        print("3. Select COM port and start acquisition")
    else:
        print(f"\n[WARNING] {total_tests - passed_tests} tests failed. Check the errors above.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
