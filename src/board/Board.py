import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter


class Board(ABC):
    """
    Abstract base class for EEG board interfaces.
    Provides a common interface for connecting, streaming, and retrieving data from EEG boards.
    Uses BrainFlow's built-in ring buffer for data storage.
    """
    
    def __init__(self, serial_port=None, mac_address=None, **kwargs):
        """
        Initialize board with connection parameters.
        
        Args:
            serial_port: Serial port for USB connection (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux)
            mac_address: MAC address for Bluetooth connection (e.g., '00:11:22:33:44:55')
            **kwargs: Additional board-specific parameters
        """
        self.serial_port = serial_port
        self.mac_address = mac_address
        self.kwargs = kwargs
        self._board = None
        self._board_id = None
        self._is_streaming = False
    
    @abstractmethod
    def connect(self):
        """
        Connect to the board and prepare the session.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def disconnect(self):
        """
        Stop streaming, release the session, and disconnect from the board.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def stream(self):
        """
        Start streaming data. Data will be stored in BrainFlow's built-in ring buffer.
        Must be implemented by subclasses.
        """
        pass
    
    def get_data(self, num_samples=None, preset=None):
        """
        Get data from BrainFlow's ring buffer. Removes data from buffer.
        
        Args:
            num_samples: Number of samples to retrieve (None for all available data)
            preset: BrainFlow preset (defaults to DEFAULT_PRESET if None)
        
        Returns:
            numpy.ndarray: Board data as a numpy array
        """
        if self._board is None:
            raise RuntimeError("Board not connected. Call connect() first.")
        
        if not self._is_streaming:
            raise RuntimeError("Board not streaming. Call stream() first.")
        
        if preset is None:
            from brainflow.board_shim import BrainFlowPresets
            preset = BrainFlowPresets.DEFAULT_PRESET
        
        return self._board.get_board_data(num_samples=num_samples, preset=preset)
    
    def get_current_data(self, num_samples, preset=None):
        """
        Get current data from BrainFlow's ring buffer without removing it.
        
        Args:
            num_samples: Number of samples to retrieve
            preset: BrainFlow preset (defaults to DEFAULT_PRESET if None)
        
        Returns:
            numpy.ndarray: Board data as a numpy array
        """
        if self._board is None:
            raise RuntimeError("Board not connected. Call connect() first.")
        
        if not self._is_streaming:
            raise RuntimeError("Board not streaming. Call stream() first.")
        
        if preset is None:
            from brainflow.board_shim import BrainFlowPresets
            preset = BrainFlowPresets.DEFAULT_PRESET
        
        return self._board.get_current_board_data(num_samples=num_samples, preset=preset)
    
    def get_data_count(self, preset=None):
        """
        Get the number of data points currently in the ring buffer.
        
        Args:
            preset: BrainFlow preset (defaults to DEFAULT_PRESET if None)
        
        Returns:
            int: Number of data points in the ring buffer
        """
        if self._board is None:
            raise RuntimeError("Board not connected. Call connect() first.")
        
        if preset is None:
            from brainflow.board_shim import BrainFlowPresets
            preset = BrainFlowPresets.DEFAULT_PRESET
        
        return self._board.get_board_data_count(preset=preset)
    
    def get_channels(self, channel_type='eeg', preset=None):
        """
        Get channel indices for a specific channel type by routing through BrainFlow's channel methods.
        
        Args:
            channel_type: Type of channels to retrieve. Supported values:
                - 'eeg': EEG channels
                - 'exg': EXG channels (general electrophysiological)
                - 'emg': EMG channels
                - 'ecg': ECG channels
                - 'eog': EOG channels
                - 'eda': EDA channels
                - 'ppg': PPG channels
                - 'accel': Accelerometer channels
                - 'gyro': Gyroscope channels
                - 'rotation': Rotation channels
                - 'analog': Analog channels
                - 'temperature': Temperature channels
                - 'resistance': Resistance channels
                - 'magnetometer': Magnetometer channels
                - 'other': Other channels
            preset: BrainFlow preset (defaults to DEFAULT_PRESET if None)
        
        Returns:
            List[int]: List of channel indices for the specified channel type
        
        Raises:
            ValueError: If channel_type is not supported
            RuntimeError: If board is not connected or board_id is not set
        """
        if self._board_id is None:
            raise RuntimeError("Board ID not set. Board must be connected first.")
        
        if preset is None:
            from brainflow.board_shim import BrainFlowPresets
            preset = BrainFlowPresets.DEFAULT_PRESET
        
        # Map channel types to BrainFlow methods
        channel_methods = {
            'eeg': BoardShim.get_eeg_channels,
            'exg': BoardShim.get_exg_channels,
            'emg': BoardShim.get_emg_channels,
            'ecg': BoardShim.get_ecg_channels,
            'eog': BoardShim.get_eog_channels,
            'eda': BoardShim.get_eda_channels,
            'ppg': BoardShim.get_ppg_channels,
            'accel': BoardShim.get_accel_channels,
            'gyro': BoardShim.get_gyro_channels,
            'rotation': BoardShim.get_rotation_channels,
            'analog': BoardShim.get_analog_channels,
            'temperature': BoardShim.get_temperature_channels,
            'resistance': BoardShim.get_resistance_channels,
            'magnetometer': BoardShim.get_magnetometer_channels,
            'other': BoardShim.get_other_channels
        }
        
        channel_type_lower = channel_type.lower()
        if channel_type_lower not in channel_methods:
            raise ValueError(f"Unsupported channel type: {channel_type}. "
                           f"Supported types: {', '.join(channel_methods.keys())}")
        
        method = channel_methods[channel_type_lower]
        return method(self._board_id, preset)