# OpenBCI EEG Interface

A Python-based GUI application for real-time EEG data acquisition and visualization from OpenBCI hardware.

## Features

- Real-time EEG signal acquisition from OpenBCI Cyton board
- Live signal visualization with PyQtGraph
- Signal filtering and processing capabilities
- Data recording and playback functionality
- Modern GUI interface built with PyQt5

## Hardware Requirements

- OpenBCI Cyton board (up to 8 channels)
- EEG electrodes and headset
- USB connection to computer

## Installation

1. Clone or download this repository
2. Create a virtual environment:
   ```bash
   python -m venv openbci_env
   openbci_env\Scripts\activate  # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Connect your OpenBCI Cyton board to your computer via USB
2. Run the main application:
   ```bash
   python src/main.py
   ```
3. Select the correct COM port for your board
4. Click "Start Acquisition" to begin EEG data collection
5. Monitor real-time signals in the visualization window

## Project Structure

```
OpenBCI_EEG_Interface/
├── src/                    # Source code
│   ├── main.py            # Main application entry point
│   ├── data_acquisition.py # EEG data acquisition module
│   ├── gui/               # GUI components
│   └── signal_processing/ # Signal processing utilities
├── data/                  # Data storage directory
├── docs/                  # Documentation
├── tests/                 # Test files
└── requirements.txt       # Python dependencies
```

## Troubleshooting

- Ensure your OpenBCI board is properly connected and recognized
- Check COM port settings in Device Manager (Windows)
- Verify all dependencies are installed correctly
- Refer to OpenBCI documentation for hardware-specific issues

## License

This project is open source and available under the MIT License.

