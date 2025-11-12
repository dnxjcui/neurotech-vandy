import numpy as np
import pandas as pd
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter
from .Board import Board


class Cyton(Board):
    """
    Implementation of Board interface for OpenBCI Cyton board using BrainFlow.
    Supports USB serial and Bluetooth connections.
    """
    
    def connect(self):
        """
        Connect to the Cyton board and prepare the session.
        Sets up BrainFlowInputParams with serial port or MAC address and initializes BoardShim.
        """
        params = BrainFlowInputParams()
        
        if self.serial_port:
            params.serial_port = self.serial_port
        if self.mac_address:
            params.mac_address = self.mac_address
        
        # Apply any additional kwargs to params
        for key, value in self.kwargs.items():
            if hasattr(params, key):
                setattr(params, key, value)
        
        self._board_id = BoardIds.CYTON_BOARD
        self._board = BoardShim(self._board_id, params)
        self._board.prepare_session()    

    def disconnect(self):
        """
        Stop streaming, release the session, and disconnect from the board.
        Uses BrainFlow's stop_stream() and release_session() methods.
        """
        if self._is_streaming:
            self._board.stop_stream()
            self._is_streaming = False
        
        if self._board:
            self._board.release_session()
            self._board = None
            self._board_id = None

    def stream(self, num_samples=450000, streamer_params=None):
        """
        Start streaming data. Data is automatically stored in BrainFlow's ring buffer.
        Uses BrainFlow's start_stream() method directly.
        
        Args:
            num_samples: Size of ring buffer to keep data (default: 450000)
            streamer_params: Optional streamer parameters (e.g., "file://data.csv:w")
        """
        if self._board is None:
            raise RuntimeError("Board not connected. Call connect() first.")
        
        if self._is_streaming:
            raise RuntimeError("Streaming already in progress.")
        
        self._board.start_stream(num_samples=num_samples, streamer_params=streamer_params)
        self._is_streaming = True

