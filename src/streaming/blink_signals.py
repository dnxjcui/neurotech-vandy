"""
blink_signals.py 
Detects blinks from EEG data and outputs a signal to Serial port when a blink is detected.
The goal is to have an arduino pick up the value and perform an action.
"""

import json
import serial
import time
import numpy as np
import pyqtgraph as pg
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from pyqtgraph.Qt import QtWidgets, QtCore
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from board.Cyton import Cyton

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'config.json')

def load_config():
    if os.path.exists(_CONFIG_PATH):
        with open(_CONFIG_PATH) as f:
            return json.load(f)
    return {}


class BlinkSignals:
    """
    Detects blinks from EEG data using delta band power changes.
    Plots EEG data and displays "Blink!" when a blink is detected.
    Can output blink signals to a serial port for Arduino communication.
    """
    
    def __init__(self, board_serial_port=None, board_mac_address=None,
                 output_serial_port=None, blink_threshold=2.0,
                 delta_low=0.5, delta_high=4.0, channels=None, refractory_period=0.2):
        """
        Initialize blink detection system.

        Args:
            board_serial_port: Serial port for Cyton board (e.g., 'COM3')
            board_mac_address: MAC address for Cyton board Bluetooth connection
            output_serial_port: Serial port for Arduino output (None to disable)
            blink_threshold: Threshold multiplier for delta band power change to detect blink
            delta_low: Lower bound for delta band (Hz)
            delta_high: Upper bound for delta band (Hz)
            channels: List of 0-based indices into the EEG channel list to use.
                      None means use all channels.
            refractory_period: Minimum seconds between blink detections (default 0.2).
        """
        self.blink_threshold = blink_threshold
        self.delta_low = delta_low
        self.delta_high = delta_high
        self.refractory_period = refractory_period

        # Initialize board
        self.board = Cyton(serial_port=board_serial_port, mac_address=board_mac_address)
        self.board.connect()

        # Get board parameters
        all_eeg_channels = self.board.get_channels('eeg')
        if channels is not None:
            try:
                self.eeg_channels = [all_eeg_channels[i] for i in channels]
            except IndexError:
                raise ValueError(
                    f"--channels index out of range. Board has {len(all_eeg_channels)} EEG channels (use 1-{len(all_eeg_channels)})."
                )
        else:
            self.eeg_channels = all_eeg_channels
        self.sampling_rate = self.board.get_sampling_rate()
        self.update_speed_ms = 50
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate
        
        # Initialize serial port for Arduino if specified
        self.arduino_serial = None
        if output_serial_port:
            try:
                self.arduino_serial = serial.Serial(output_serial_port, 9600, timeout=1)
                time.sleep(2)  # Wait for serial connection to establish
            except Exception as e:
                print(f"Warning: Could not open serial port {output_serial_port}: {e}")
        
        # Blink detection state (keyed by board channel index for multi-channel support)
        self.previous_delta_power = {}
        self.blink_detected = False
        self.blink_display_time = 0
        self.blink_display_duration_ms = 500
        self.last_blink_time = 0.0
        
        # Initialize GUI
        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(title='Blink Detection', size=(1000, 700), show=True)
        
        self._init_plots()
        
        # Start streaming
        self.board.stream()
        
        # Setup update timer
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(self.update_speed_ms)
        
        # Run application
        QtWidgets.QApplication.instance().exec()
    
    def _init_plots(self):
        """Initialize plots for EEG channels and blink indicator."""
        self.plots = list()
        self.curves = list()
        
        # Create plots for each EEG channel
        for i in range(len(self.eeg_channels)):
            p = self.win.addPlot(row=i, col=0)
            p.showAxis('left', False)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            if i == 0:
                p.setTitle('EEG TimeSeries Plot')
            self.plots.append(p)
            curve = p.plot()
            self.curves.append(curve)
        
        # Add blink indicator text widget
        self.blink_text = self.win.addLabel('', row=len(self.eeg_channels), col=0)
        self.blink_text.setText('')
        # self.blink_text.setStyleSheet("font-size: 48px; font-weight: bold; color: red;")
    
    def detect_blink(self, data, channel_idx):
        """
        Detect blink using delta band power changes.

        Args:
            data: Full board data array
            channel_idx: Board-level index of the EEG channel to analyze

        Returns:
            tuple: (blink_detected: bool, delta_power: float, power_change: float)
        """
        channel_data = data[channel_idx].copy()

        # Filter the data
        DataFilter.detrend(channel_data, DetrendOperations.CONSTANT.value)
        DataFilter.perform_bandpass(channel_data, self.sampling_rate, 3.0, 45.0, 2,
                                   FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
        DataFilter.perform_bandstop(channel_data, self.sampling_rate, 48.0, 52.0, 2,
                                   FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
        DataFilter.perform_bandstop(channel_data, self.sampling_rate, 58.0, 62.0, 2,
                                   FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)

        delta_power = self.board.get_band_power(channel_data, self.delta_low, self.delta_high)

        blink_detected = False
        prev = self.previous_delta_power.get(channel_idx)
        power_change = delta_power / prev if prev and prev > 0 else 0.0
        if prev is not None and power_change > self.blink_threshold:
            blink_detected = True

        self.previous_delta_power[channel_idx] = delta_power

        return blink_detected, delta_power, power_change
    
    def update(self):
        """Update plots and check for blinks."""
        try:
            data = self.board.get_current_data(self.num_points)
            
            if data.size == 0:
                return

            nfft = DataFilter.get_nearest_power_of_two(self.sampling_rate)
            if data.shape[1] <= nfft:
                return

            # Process each EEG channel
            for count, channel in enumerate(self.eeg_channels):
                channel_data = data[channel].copy()
                
                # Filter the data (same as realtime_plot.py)
                DataFilter.detrend(channel_data, DetrendOperations.CONSTANT.value)
                DataFilter.perform_bandpass(channel_data, self.sampling_rate, 3.0, 45.0, 2,
                                           FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
                DataFilter.perform_bandstop(channel_data, self.sampling_rate, 48.0, 52.0, 2,
                                           FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
                DataFilter.perform_bandstop(channel_data, self.sampling_rate, 58.0, 62.0, 2,
                                           FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
                
                # Update plot
                self.curves[count].setData(channel_data.tolist())
            
            # Detect blink across all selected channels; fire if any channel triggers
            now = time.time()
            in_refractory = (now - self.last_blink_time) < self.refractory_period
            any_blink = False
            for channel in self.eeg_channels:
                blink_detected, _, _ = self.detect_blink(data, channel)
                if blink_detected:
                    any_blink = True

            if any_blink and not in_refractory:
                self.last_blink_time = now
                self.blink_detected = True
                self.blink_display_time = now * 1000
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Blink detected")

                # Send signal to Arduino
                if self.arduino_serial and self.arduino_serial.is_open:
                    try:
                        self.arduino_serial.write(b'1')
                    except Exception as e:
                        print(f"Error writing to serial: {e}")
            
            # Update blink display
            current_time_ms = time.time() * 1000
            if self.blink_detected and (current_time_ms - self.blink_display_time) < self.blink_display_duration_ms:
                self.blink_text.setText('BLINK!')
            else:
                self.blink_text.setText('')
                self.blink_detected = False
            
            self.app.processEvents()
            
        except Exception as e:
            print(f"Error in update: {e}")
    
    def cleanup(self):
        """Clean up resources."""
        if self.arduino_serial and self.arduino_serial.is_open:
            self.arduino_serial.close()
        self.board.disconnect()


def main():
    """Main function to run blink detection."""
    import argparse

    config = load_config()

    parser = argparse.ArgumentParser(description='Blink Detection with EEG')
    parser.add_argument('--board-serial', type=str, help='Serial port for Cyton board',
                        default=config.get('cyton_serial_port'))
    parser.add_argument('--board-mac', type=str, help='MAC address for Cyton board', default=None)
    parser.add_argument('--arduino-serial', type=str, help='Serial port for Arduino output',
                        default=config.get('arduino_serial_port'))
    parser.add_argument('--blink-threshold', type=float, help='Blink detection threshold', default=1.6)
    parser.add_argument('--delta-low', type=float, help='Delta band lower frequency (Hz)', default=0.5)
    parser.add_argument('--delta-high', type=float, help='Delta band upper frequency (Hz)', default=4.0)
    parser.add_argument('--channels', type=str,
                        help='EEG channels to use, 1-based comma-separated (e.g. 1,2,3). Default: all.',
                        default=None)
    parser.add_argument('--refractory-period', type=float,
                        help='Minimum seconds between blink detections (default 0.2).', default=0.2)

    args = parser.parse_args()

    channels = None
    if args.channels:
        channels = [int(c.strip()) - 1 for c in args.channels.split(',')]

    try:
        blink_detector = BlinkSignals(
            board_serial_port=args.board_serial,
            board_mac_address=args.board_mac,
            output_serial_port=args.arduino_serial,
            blink_threshold=args.blink_threshold,
            delta_low=args.delta_low,
            delta_high=args.delta_high,
            channels=channels,
            refractory_period=args.refractory_period,
        )
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'blink_detector' in locals():
            blink_detector.cleanup()


if __name__ == '__main__':
    main()
