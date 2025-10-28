# Neurotech EEG Analysis

A Python-based repository for streaming, processing, and visualizing EEG signals from the OpenBCI 8-channel Cyton board using BrainFlow.

## Overview

This project provides tools for:
- **EEG Signal Streaming**: Real-time data acquisition from OpenBCI Cyton board via BrainFlow
- **Data Processing**: Signal filtering, artifact removal, and feature extraction
- **Visualization**: Real-time and offline EEG signal visualization
- **CLI Interface**: Command-line tools for easy operation

## Hardware Requirements

- **OpenBCI Cyton Board**: 8-channel EEG acquisition system
- **Electrodes**: Compatible EEG electrodes (wet or dry)
- **Computer**: Windows/macOS/Linux with Python 3.8+
- **Connection**: USB or Bluetooth connection to Cyton board

## Software Requirements

### Python Dependencies
```bash
pip install -r requirements.txt
```

### Required Libraries
- **BrainFlow**: Cross-platform library for brain-computer interface devices
- **NumPy**: For numerical computations and data handling
- **Matplotlib**: For data visualization (optional, for future visualization features)

## Project Structure

```
neurotech/
├── src/
│   ├── streaming/
│   │   ├── arduino.py      # Arduino-based streaming utilities
│   │   └── cyton.py        # OpenBCI Cyton board interface (BrainFlow)
│   ├── visualization/
│   │   └── time_series.py  # EEG signal visualization tools
│   └── run.py              # CLI interface for running operations
├── requirements.txt        # Python dependencies
├── LICENSE
└── README.md
```

## Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd neurotech
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Connect Your Cyton Board
- Power on your OpenBCI Cyton board
- Connect via USB or pair via Bluetooth
- Ensure electrodes are properly placed and have good contact

### 4. Run Basic Streaming
```bash
# Basic data reception (auto-detect connection)
python src/run.py --receive

# Receive data for 30 seconds
python src/run.py --receive --duration 30

# Connect via specific serial port
python src/run.py --receive --serial COM3

# Connect via Bluetooth
python src/run.py --receive --mac 00:11:22:33:44:55
```

### 5. Visualize Data (Future Feature)
```python
from src.visualization.time_series import EEGVisualizer

# Create visualizer
visualizer = EEGVisualizer()

# Display real-time EEG data
visualizer.plot_realtime_data(streamer)
```

## Usage Examples

### CLI Usage
```bash
# Show help and available options
python src/run.py --help

# Basic data reception with electrode values printed to console
python src/run.py --receive

# Receive data for specific duration
python src/run.py --receive --duration 60

# Connect via USB serial port
python src/run.py --receive --serial COM3

# Connect via Bluetooth
python src/run.py --receive --mac 00:11:22:33:44:55

# Receive data without printing electrode values
python src/run.py --receive --no-print
```

### Programmatic Usage
```python
from src.streaming.cyton import configure_board, receive

# Configure board connection
streamer = configure_board(serial_port='COM3')  # or mac_address='00:11:22:33:44:55'

# Start receiving data
data = receive(streamer, duration=30, print_data=True)
```

## Configuration

### Cyton Board Settings
- **Sampling Rate**: 250 Hz (default)
- **Channels**: 8 active EEG channels
- **Resolution**: 24-bit ADC
- **Bandwidth**: 0.5 - 125 Hz
- **Connection**: USB Serial or Bluetooth

## Troubleshooting

### Common Issues

1. **Connection Problems**
   - Verify Cyton board is powered on
   - Check USB/Bluetooth connection
   - Ensure correct COM port (Windows) or device path (Linux/macOS)
   - Try auto-detection first: `python src/run.py --receive`

2. **BrainFlow Installation Issues**
   - Install BrainFlow: `pip install brainflow`
   - Verify installation: `python -c "import brainflow; print('BrainFlow installed successfully')"`

3. **Poor Signal Quality**
   - Check electrode placement and contact
   - Verify electrode impedance
   - Ensure proper grounding

### Getting Help
- Check BrainFlow documentation: https://brainflow.readthedocs.io/
- Review OpenBCI documentation: https://docs.openbci.com/
- Open an issue in this repository for project-specific problems

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the terms specified in the LICENSE file.

## Acknowledgments

- BrainFlow for the cross-platform BCI library
- OpenBCI for the Cyton board hardware
- The neurotech community for inspiration and resources
