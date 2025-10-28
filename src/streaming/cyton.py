"""
OpenBCI Cyton Board Streaming Module

This module provides functionality to stream data from an OpenBCI Cyton board
using the BrainFlow library. It includes board configuration and data reception
capabilities.
"""

import time
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations


class CytonStreamer:
    """
    Handles streaming data from OpenBCI Cyton board using BrainFlow.
    
    This class provides methods to configure the board connection and receive
    real-time EEG data from all electrodes.
    """
    
    def __init__(self):
        """Initialize the Cyton streamer."""
        self.board = None
        self.board_id = BoardIds.CYTON_BOARD
        self.params = BrainFlowInputParams()
        self.is_streaming = False
        
    def configure_board(self, serial_port=None, mac_address=None):
        """
        Configure the Cyton board connection parameters.
        
        Args:
            serial_port (str, optional): Serial port for Cyton board connection.
                                       If None, BrainFlow will auto-detect.
            mac_address (str, optional): MAC address for Cyton board connection.
                                       Required for Bluetooth connections.
        
        Returns:
            bool: True if configuration was successful, False otherwise.
        """
        try:
            # Set connection parameters
            if serial_port:
                self.params.serial_port = serial_port
            if mac_address:
                self.params.mac_address = mac_address
            
            # Initialize board
            self.board = BoardShim(self.board_id, self.params)
            
            # Prepare session
            self.board.prepare_session()
            
            print(f"Cyton board configured successfully")
            print(f"Board ID: {self.board_id}")
            print(f"Serial Port: {self.params.serial_port}")
            print(f"MAC Address: {self.params.mac_address}")
            
            return True
            
        except Exception as e:
            print(f"Error configuring Cyton board: {e}")
            return False
    
    def start_streaming(self):
        """
        Start streaming data from the Cyton board.
        
        Returns:
            bool: True if streaming started successfully, False otherwise.
        """
        try:
            if not self.board:
                print("Board not configured. Call configure_board() first.")
                return False
            
            self.board.start_stream()
            self.is_streaming = True
            
            print("Started streaming from Cyton board")
            return True
            
        except Exception as e:
            print(f"Error starting stream: {e}")
            return False
    
    def receive(self, duration=None, print_data=True):
        """
        Receive data from the Cyton board.
        
        Args:
            duration (float, optional): Duration to receive data in seconds.
                                     If None, receives data indefinitely.
            print_data (bool): Whether to print electrode data to console.
        
        Returns:
            numpy.ndarray: EEG data array with shape (channels, samples)
        """
        if not self.board:
            print("Board not configured. Call configure_board() first.")
            return None
        
        if not self.is_streaming:
            print("Stream not started. Call start_streaming() first.")
            return None
        
        try:
            # Get board info
            sampling_rate = BoardShim.get_sampling_rate(self.board_id)
            eeg_channels = BoardShim.get_eeg_channels(self.board_id)
            num_channels = len(eeg_channels)
            
            print(f"Sampling rate: {sampling_rate} Hz")
            print(f"EEG channels: {eeg_channels}")
            print(f"Number of channels: {num_channels}")
            print("Receiving data... (Press Ctrl+C to stop)")
            
            start_time = time.time()
            
            while True:
                # Check if duration limit reached
                if duration and (time.time() - start_time) >= duration:
                    break
                
                # Get data from board
                data = self.board.get_board_data()
                
                if data.shape[1] > 0:  # If we have new data
                    # Extract EEG data (exclude timestamp and other channels)
                    eeg_data = data[eeg_channels, :]
                    
                    if print_data:
                        # Print latest sample from each electrode
                        latest_sample = eeg_data[:, -1] if eeg_data.shape[1] > 0 else eeg_data[:, 0]
                        print(f"Electrode data: {latest_sample}")
                    
                    # Small delay to prevent overwhelming the console
                    time.sleep(0.1)
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)
            
            return data[eeg_channels, :] if data.shape[1] > 0 else None
            
        except KeyboardInterrupt:
            print("\nStopping data reception...")
            return None
        except Exception as e:
            print(f"Error receiving data: {e}")
            return None
    
    def stop_streaming(self):
        """Stop streaming and clean up resources."""
        try:
            if self.board and self.is_streaming:
                self.board.stop_stream()
                self.is_streaming = False
                print("Stopped streaming")
            
            if self.board:
                self.board.release_session()
                print("Released board session")
                
        except Exception as e:
            print(f"Error stopping stream: {e}")
    
    def get_board_info(self):
        """
        Get information about the connected board.
        
        Returns:
            dict: Dictionary containing board information.
        """
        if not self.board:
            return {"error": "Board not configured"}
        
        try:
            info = {
                "board_id": self.board_id,
                "sampling_rate": BoardShim.get_sampling_rate(self.board_id),
                "eeg_channels": BoardShim.get_eeg_channels(self.board_id),
                "num_eeg_channels": len(BoardShim.get_eeg_channels(self.board_id)),
                "is_streaming": self.is_streaming
            }
            return info
        except Exception as e:
            return {"error": f"Failed to get board info: {e}"}


def configure_board(serial_port=None, mac_address=None):
    """
    Configure the Cyton board connection.
    
    Args:
        serial_port (str, optional): Serial port for Cyton board connection.
        mac_address (str, optional): MAC address for Cyton board connection.
    
    Returns:
        CytonStreamer: Configured CytonStreamer instance.
    """
    streamer = CytonStreamer()
    if streamer.configure_board(serial_port, mac_address):
        return streamer
    return None


def receive(streamer, duration=None, print_data=True):
    """
    Receive data from the Cyton board.
    
    Args:
        streamer (CytonStreamer): Configured CytonStreamer instance.
        duration (float, optional): Duration to receive data in seconds.
        print_data (bool): Whether to print electrode data to console.
    
    Returns:
        numpy.ndarray: EEG data array.
    """
    if not streamer:
        print("No streamer provided")
        return None
    
    # Start streaming if not already started
    if not streamer.is_streaming:
        if not streamer.start_streaming():
            return None
    
    return streamer.receive(duration, print_data)
