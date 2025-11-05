"""
OpenBCI Data Acquisition Module

This module handles the connection and data acquisition from OpenBCI Cyton board
using the BrainFlow library.
"""

import time
import threading
import numpy as np
from typing import Optional, Callable, List
import logging

try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, LogLevels
    from brainflow.data_filter import DataFilter
    BRAINFLOW_AVAILABLE = True
    DEFAULT_BOARD_ID = BoardIds.CYTON_BOARD
except ImportError:
    BRAINFLOW_AVAILABLE = False
    BoardIds = None
    DEFAULT_BOARD_ID = 0  # Default fallback value
    print("Warning: BrainFlow not installed. Please install with: pip install brainflow")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenBCIDataAcquisition:
    """
    Handles data acquisition from OpenBCI Cyton board.
    """
    
    def __init__(self, serial_port: str = None, board_id: int = None):
        """
        Initialize the OpenBCI data acquisition system.
        
        Args:
            serial_port: COM port for the OpenBCI board (e.g., 'COM3')
            board_id: Board ID from BrainFlow (default: CYTON_BOARD)
        """
        if not BRAINFLOW_AVAILABLE:
            raise ImportError("BrainFlow library is required but not installed")
        
        self.board_id = board_id if board_id is not None else DEFAULT_BOARD_ID
        self.serial_port = serial_port
        self.board = None
        self.is_streaming = False
        self.data_callback = None
        self.acquisition_thread = None
        
        # EEG data configuration
        self.sampling_rate = 250  # Hz (OpenBCI Cyton default)
        self.num_channels = 8     # Up to 8 channels for Cyton
        self.channel_names = [
            'Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2'
        ]
        
        # Data buffer
        self.data_buffer = []
        self.buffer_size = 1000  # Number of samples to keep in buffer
        
    def set_data_callback(self, callback: Callable[[np.ndarray], None]):
        """
        Set callback function to receive EEG data.
        
        Args:
            callback: Function that receives EEG data as numpy array
        """
        self.data_callback = callback
    
    def connect(self) -> bool:
        """
        Connect to the OpenBCI board.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Set up BrainFlow parameters
            params = BrainFlowInputParams()
            if self.serial_port:
                params.serial_port = self.serial_port
            
            # Initialize board
            self.board = BoardShim(self.board_id, params)
            
            # Prepare session
            self.board.prepare_session()
            logger.info(f"Connected to OpenBCI board on {self.serial_port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to OpenBCI board: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the OpenBCI board."""
        try:
            if self.is_streaming:
                self.stop_acquisition()
            
            if self.board:
                self.board.release_session()
                logger.info("Disconnected from OpenBCI board")
                
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
    
    def start_acquisition(self) -> bool:
        """
        Start EEG data acquisition.
        
        Returns:
            True if acquisition started successfully, False otherwise
        """
        try:
            if not self.board:
                logger.error("Board not connected. Call connect() first.")
                return False
            
            # Start streaming
            self.board.start_stream()
            self.is_streaming = True
            
            # Start acquisition thread
            self.acquisition_thread = threading.Thread(target=self._acquisition_loop)
            self.acquisition_thread.daemon = True
            self.acquisition_thread.start()
            
            logger.info("EEG data acquisition started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start acquisition: {e}")
            return False
    
    def stop_acquisition(self):
        """Stop EEG data acquisition."""
        try:
            self.is_streaming = False
            
            if self.board:
                self.board.stop_stream()
            
            if self.acquisition_thread:
                self.acquisition_thread.join(timeout=2.0)
            
            logger.info("EEG data acquisition stopped")
            
        except Exception as e:
            logger.error(f"Error stopping acquisition: {e}")
    
    def _acquisition_loop(self):
        """Main acquisition loop running in separate thread."""
        while self.is_streaming:
            try:
                # Get data from board
                data = self.board.get_board_data()
                
                if data.size > 0:
                    # Process and send data
                    self._process_data(data)
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in acquisition loop: {e}")
                break
    
    def _process_data(self, raw_data: np.ndarray):
        """
        Process raw EEG data and send to callback.
        
        Args:
            raw_data: Raw data from BrainFlow board
        """
        try:
            # Extract EEG channels (typically channels 1-8 for Cyton)
            if raw_data.shape[0] > self.num_channels:
                eeg_data = raw_data[1:self.num_channels + 1, :]  # Skip timestamp channel
            else:
                eeg_data = raw_data
            
            # Add to buffer
            if eeg_data.size > 0:
                self.data_buffer.append(eeg_data)
                
                # Keep buffer size manageable
                if len(self.data_buffer) > self.buffer_size:
                    self.data_buffer.pop(0)
                
                # Send to callback if available
                if self.data_callback:
                    self.data_callback(eeg_data)
                    
        except Exception as e:
            logger.error(f"Error processing data: {e}")
    
    def get_latest_data(self, num_samples: int = 100) -> Optional[np.ndarray]:
        """
        Get the latest EEG data samples.
        
        Args:
            num_samples: Number of samples to retrieve
            
        Returns:
            EEG data array or None if no data available
        """
        if not self.data_buffer:
            return None
        
        # Concatenate recent data
        recent_data = np.concatenate(self.data_buffer[-num_samples:], axis=1)
        return recent_data
    
    def get_board_info(self) -> dict:
        """
        Get information about the connected board.
        
        Returns:
            Dictionary with board information
        """
        if not self.board:
            return {}
        
        try:
            info = {
                'board_id': self.board_id,
                'serial_port': self.serial_port,
                'sampling_rate': self.sampling_rate,
                'num_channels': self.num_channels,
                'channel_names': self.channel_names,
                'is_streaming': self.is_streaming
            }
            return info
        except Exception as e:
            logger.error(f"Error getting board info: {e}")
            return {}
    
    def apply_notch_filter(self, data: np.ndarray, freq: float = 60.0) -> np.ndarray:
        """
        Apply notch filter to remove power line noise.
        
        Args:
            data: EEG data array
            freq: Frequency to filter out (default: 60 Hz)
            
        Returns:
            Filtered data
        """
        try:
            filtered_data = np.copy(data)
            for i in range(data.shape[0]):
                DataFilter.remove_environmental_noise(
                    filtered_data[i], self.sampling_rate, freq, 4
                )
            return filtered_data
        except Exception as e:
            logger.error(f"Error applying notch filter: {e}")
            return data
    
    def apply_bandpass_filter(self, data: np.ndarray, 
                            low_freq: float = 1.0, 
                            high_freq: float = 50.0) -> np.ndarray:
        """
        Apply bandpass filter to EEG data.
        
        Args:
            data: EEG data array
            low_freq: Low cutoff frequency
            high_freq: High cutoff frequency
            
        Returns:
            Filtered data
        """
        try:
            filtered_data = np.copy(data)
            for i in range(data.shape[0]):
                DataFilter.perform_bandpass(
                    filtered_data[i], self.sampling_rate, low_freq, high_freq, 4, 0, 0
                )
            return filtered_data
        except Exception as e:
            logger.error(f"Error applying bandpass filter: {e}")
            return data


def get_available_ports() -> List[str]:
    """
    Get list of available COM ports.
    
    Returns:
        List of available COM port names
    """
    import serial.tools.list_ports
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]


if __name__ == "__main__":
    # Example usage
    print("OpenBCI Data Acquisition Module")
    print("Available COM ports:", get_available_ports())
    
    # Create acquisition instance
    acquisition = OpenBCIDataAcquisition(serial_port="COM3")  # Change to your port
    
    # Set up data callback
    def data_callback(data):
        print(f"Received EEG data shape: {data.shape}")
    
    acquisition.set_data_callback(data_callback)
    
    # Connect and start acquisition
    if acquisition.connect():
        print("Connected successfully!")
        acquisition.start_acquisition()
        
        # Run for 10 seconds
        time.sleep(10)
        
        acquisition.stop_acquisition()
        acquisition.disconnect()
    else:
        print("Failed to connect to OpenBCI board")
