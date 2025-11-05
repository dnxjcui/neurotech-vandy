"""
OpenBCI EEG GUI Interface

This module provides a PyQt5-based GUI for real-time EEG data visualization
and control of the OpenBCI data acquisition system.
"""

import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QComboBox, 
                             QSpinBox, QDoubleSpinBox, QCheckBox, QTextEdit,
                             QGroupBox, QGridLayout, QSlider, QProgressBar,
                             QMessageBox, QFileDialog, QSplitter)
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont, QPalette, QColor
import pyqtgraph as pg
from pyqtgraph import PlotWidget, mkPen
import time
import json
import os

from data_acquisition import OpenBCIDataAcquisition, get_available_ports


class EEGPlotWidget(PlotWidget):
    """Custom plot widget for EEG data visualization."""
    
    def __init__(self, num_channels=8, sampling_rate=250):
        super().__init__()
        self.num_channels = num_channels
        self.sampling_rate = sampling_rate
        self.data_buffer = []
        self.max_samples = 1000  # Number of samples to display
        
        # Set up plot
        self.setBackground('black')
        self.setLabel('left', 'Amplitude', units='μV')
        self.setLabel('bottom', 'Time', units='s')
        self.setTitle('Real-time EEG Signals')
        
        # Create pens for different channels
        colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'white', 'orange']
        self.pens = [mkPen(color=color, width=1) for color in colors[:num_channels]]
        
        # Initialize plot curves
        self.curves = []
        for i in range(num_channels):
            curve = self.plot(pen=self.pens[i])
            self.curves.append(curve)
        
        # Set up time axis
        self.time_axis = np.linspace(0, self.max_samples / self.sampling_rate, self.max_samples)
        
    def update_data(self, eeg_data):
        """Update the plot with new EEG data."""
        if eeg_data.size == 0:
            return
        
        # Add new data to buffer
        self.data_buffer.append(eeg_data)
        
        # Keep only recent data
        if len(self.data_buffer) > self.max_samples // 10:  # Keep ~4 seconds of data
            self.data_buffer.pop(0)
        
        # Concatenate all buffered data
        if self.data_buffer:
            all_data = np.concatenate(self.data_buffer, axis=1)
            
            # Update each channel
            for i in range(min(self.num_channels, all_data.shape[0])):
                if all_data.shape[1] > 0:
                    # Create time axis for current data
                    time_points = np.linspace(0, all_data.shape[1] / self.sampling_rate, all_data.shape[1])
                    self.curves[i].setData(time_points, all_data[i, :])
    
    def clear_data(self):
        """Clear all data from the plot."""
        self.data_buffer = []
        for curve in self.curves:
            curve.setData([], [])


class ControlPanel(QWidget):
    """Control panel for OpenBCI settings and operations."""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Connection settings
        conn_group = QGroupBox("Connection Settings")
        conn_layout = QGridLayout()
        
        # COM port selection
        conn_layout.addWidget(QLabel("COM Port:"), 0, 0)
        self.port_combo = QComboBox()
        self.port_combo.addItems(get_available_ports())
        conn_layout.addWidget(self.port_combo, 0, 1)
        
        # Refresh ports button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_ports)
        conn_layout.addWidget(refresh_btn, 0, 2)
        
        # Connection buttons
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.connect_board)
        conn_layout.addWidget(self.connect_btn, 1, 0)
        
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(self.disconnect_board)
        self.disconnect_btn.setEnabled(False)
        conn_layout.addWidget(self.disconnect_btn, 1, 1)
        
        conn_group.setLayout(conn_layout)
        layout.addWidget(conn_group)
        
        # Acquisition control
        acq_group = QGroupBox("Data Acquisition")
        acq_layout = QGridLayout()
        
        self.start_btn = QPushButton("Start Acquisition")
        self.start_btn.clicked.connect(self.start_acquisition)
        self.start_btn.setEnabled(False)
        acq_layout.addWidget(self.start_btn, 0, 0)
        
        self.stop_btn = QPushButton("Stop Acquisition")
        self.stop_btn.clicked.connect(self.stop_acquisition)
        self.stop_btn.setEnabled(False)
        acq_layout.addWidget(self.stop_btn, 0, 1)
        
        acq_group.setLayout(acq_layout)
        layout.addWidget(acq_group)
        
        # Signal processing
        filter_group = QGroupBox("Signal Processing")
        filter_layout = QGridLayout()
        
        # Notch filter
        self.notch_check = QCheckBox("Notch Filter (60 Hz)")
        filter_layout.addWidget(self.notch_check, 0, 0)
        
        # Bandpass filter
        self.bandpass_check = QCheckBox("Bandpass Filter")
        filter_layout.addWidget(self.bandpass_check, 1, 0)
        
        filter_layout.addWidget(QLabel("Low Freq:"), 2, 0)
        self.low_freq_spin = QDoubleSpinBox()
        self.low_freq_spin.setRange(0.1, 50.0)
        self.low_freq_spin.setValue(1.0)
        self.low_freq_spin.setDecimals(1)
        filter_layout.addWidget(self.low_freq_spin, 2, 1)
        
        filter_layout.addWidget(QLabel("High Freq:"), 3, 0)
        self.high_freq_spin = QDoubleSpinBox()
        self.high_freq_spin.setRange(1.0, 100.0)
        self.high_freq_spin.setValue(50.0)
        self.high_freq_spin.setDecimals(1)
        filter_layout.addWidget(self.high_freq_spin, 3, 1)
        
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)
        
        # Data recording
        record_group = QGroupBox("Data Recording")
        record_layout = QGridLayout()
        
        self.record_btn = QPushButton("Start Recording")
        self.record_btn.clicked.connect(self.start_recording)
        record_layout.addWidget(self.record_btn, 0, 0)
        
        self.stop_record_btn = QPushButton("Stop Recording")
        self.stop_record_btn.clicked.connect(self.stop_recording)
        self.stop_record_btn.setEnabled(False)
        record_layout.addWidget(self.stop_record_btn, 0, 1)
        
        self.save_btn = QPushButton("Save Data")
        self.save_btn.clicked.connect(self.save_data)
        self.save_btn.setEnabled(False)
        record_layout.addWidget(self.save_btn, 1, 0)
        
        record_group.setLayout(record_layout)
        layout.addWidget(record_group)
        
        # Status display
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        
        self.status_label = QLabel("Disconnected")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        status_layout.addWidget(self.status_label)
        
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(100)
        self.info_text.setReadOnly(True)
        status_layout.addWidget(self.info_text)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def refresh_ports(self):
        """Refresh the list of available COM ports."""
        self.port_combo.clear()
        self.port_combo.addItems(get_available_ports())
    
    def connect_board(self):
        """Connect to the OpenBCI board."""
        port = self.port_combo.currentText()
        if not port:
            QMessageBox.warning(self, "Warning", "Please select a COM port")
            return
        
        # This will be connected to the main window's connect method
        self.parent().connect_to_board(port)
    
    def disconnect_board(self):
        """Disconnect from the OpenBCI board."""
        self.parent().disconnect_from_board()
    
    def start_acquisition(self):
        """Start EEG data acquisition."""
        self.parent().start_data_acquisition()
    
    def stop_acquisition(self):
        """Stop EEG data acquisition."""
        self.parent().stop_data_acquisition()
    
    def start_recording(self):
        """Start recording EEG data."""
        self.parent().start_data_recording()
    
    def stop_recording(self):
        """Stop recording EEG data."""
        self.parent().stop_data_recording()
    
    def save_data(self):
        """Save recorded EEG data."""
        self.parent().save_recorded_data()
    
    def update_status(self, status, color="black"):
        """Update the status label."""
        self.status_label.setText(status)
        self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")
    
    def update_info(self, info):
        """Update the info text area."""
        self.info_text.append(f"{time.strftime('%H:%M:%S')} - {info}")


class OpenBCIMainWindow(QMainWindow):
    """Main window for the OpenBCI EEG interface."""
    
    def __init__(self):
        super().__init__()
        self.data_acquisition = None
        self.recording_data = []
        self.is_recording = False
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("OpenBCI EEG Interface")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QHBoxLayout()
        
        # Create splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Create control panel
        self.control_panel = ControlPanel()
        self.control_panel.setMaximumWidth(300)
        
        # Create EEG plot widget
        self.eeg_plot = EEGPlotWidget()
        
        # Add widgets to splitter
        splitter.addWidget(self.control_panel)
        splitter.addWidget(self.eeg_plot)
        splitter.setSizes([300, 900])
        
        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)
        
        # Set up timer for GUI updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_gui)
        self.update_timer.start(50)  # Update every 50ms
    
    def connect_to_board(self, port):
        """Connect to the OpenBCI board."""
        try:
            self.data_acquisition = OpenBCIDataAcquisition(serial_port=port)
            
            # Set up data callback
            self.data_acquisition.set_data_callback(self.on_eeg_data)
            
            if self.data_acquisition.connect():
                self.control_panel.update_status("Connected", "green")
                self.control_panel.update_info(f"Connected to OpenBCI on {port}")
                
                # Enable/disable buttons
                self.control_panel.connect_btn.setEnabled(False)
                self.control_panel.disconnect_btn.setEnabled(True)
                self.control_panel.start_btn.setEnabled(True)
                
                # Update info
                board_info = self.data_acquisition.get_board_info()
                info_text = f"Board ID: {board_info.get('board_id', 'Unknown')}\n"
                info_text += f"Sampling Rate: {board_info.get('sampling_rate', 'Unknown')} Hz\n"
                info_text += f"Channels: {board_info.get('num_channels', 'Unknown')}"
                self.control_panel.info_text.setText(info_text)
                
            else:
                self.control_panel.update_status("Connection Failed", "red")
                self.control_panel.update_info("Failed to connect to OpenBCI board")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to connect: {str(e)}")
            self.control_panel.update_status("Connection Error", "red")
    
    def disconnect_from_board(self):
        """Disconnect from the OpenBCI board."""
        if self.data_acquisition:
            self.data_acquisition.disconnect()
            self.data_acquisition = None
            
            self.control_panel.update_status("Disconnected", "red")
            self.control_panel.update_info("Disconnected from OpenBCI board")
            
            # Enable/disable buttons
            self.control_panel.connect_btn.setEnabled(True)
            self.control_panel.disconnect_btn.setEnabled(False)
            self.control_panel.start_btn.setEnabled(False)
            self.control_panel.stop_btn.setEnabled(False)
    
    def start_data_acquisition(self):
        """Start EEG data acquisition."""
        if self.data_acquisition and self.data_acquisition.start_acquisition():
            self.control_panel.update_status("Acquiring Data", "blue")
            self.control_panel.update_info("Started EEG data acquisition")
            
            # Enable/disable buttons
            self.control_panel.start_btn.setEnabled(False)
            self.control_panel.stop_btn.setEnabled(True)
            self.control_panel.record_btn.setEnabled(True)
        else:
            QMessageBox.warning(self, "Warning", "Failed to start data acquisition")
    
    def stop_data_acquisition(self):
        """Stop EEG data acquisition."""
        if self.data_acquisition:
            self.data_acquisition.stop_acquisition()
            self.control_panel.update_status("Connected", "green")
            self.control_panel.update_info("Stopped EEG data acquisition")
            
            # Enable/disable buttons
            self.control_panel.start_btn.setEnabled(True)
            self.control_panel.stop_btn.setEnabled(False)
            self.control_panel.record_btn.setEnabled(False)
            self.control_panel.stop_record_btn.setEnabled(False)
    
    def start_data_recording(self):
        """Start recording EEG data."""
        self.is_recording = True
        self.recording_data = []
        self.control_panel.update_info("Started recording EEG data")
        
        # Enable/disable buttons
        self.control_panel.record_btn.setEnabled(False)
        self.control_panel.stop_record_btn.setEnabled(True)
    
    def stop_data_recording(self):
        """Stop recording EEG data."""
        self.is_recording = False
        self.control_panel.update_info("Stopped recording EEG data")
        
        # Enable/disable buttons
        self.control_panel.record_btn.setEnabled(True)
        self.control_panel.stop_record_btn.setEnabled(False)
        self.control_panel.save_btn.setEnabled(True)
    
    def save_recorded_data(self):
        """Save recorded EEG data to file."""
        if not self.recording_data:
            QMessageBox.warning(self, "Warning", "No data to save")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save EEG Data", "eeg_data.npz", "NumPy files (*.npz)"
        )
        
        if filename:
            try:
                # Concatenate all recorded data
                all_data = np.concatenate(self.recording_data, axis=1)
                
                # Save with metadata
                np.savez(filename, 
                        eeg_data=all_data,
                        sampling_rate=self.data_acquisition.sampling_rate,
                        channel_names=self.data_acquisition.channel_names,
                        timestamp=time.time())
                
                self.control_panel.update_info(f"Saved data to {filename}")
                QMessageBox.information(self, "Success", f"Data saved to {filename}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save data: {str(e)}")
    
    def on_eeg_data(self, data):
        """Callback function for receiving EEG data."""
        # Apply filters if enabled
        if self.control_panel.notch_check.isChecked():
            data = self.data_acquisition.apply_notch_filter(data)
        
        if self.control_panel.bandpass_check.isChecked():
            low_freq = self.control_panel.low_freq_spin.value()
            high_freq = self.control_panel.high_freq_spin.value()
            data = self.data_acquisition.apply_bandpass_filter(data, low_freq, high_freq)
        
        # Record data if recording is active
        if self.is_recording:
            self.recording_data.append(data.copy())
    
    def update_gui(self):
        """Update GUI elements."""
        if self.data_acquisition and self.data_acquisition.is_streaming:
            # Update plot with latest data
            latest_data = self.data_acquisition.get_latest_data(100)
            if latest_data is not None:
                self.eeg_plot.update_data(latest_data)
    
    def closeEvent(self, event):
        """Handle application close event."""
        if self.data_acquisition:
            self.data_acquisition.disconnect()
        event.accept()


def main():
    """Main function to run the application."""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = OpenBCIMainWindow()
    window.show()
    window.raise_()
    window.activateWindow()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

