# Quick Start Guide - OpenBCI EEG Interface

## 🚀 Getting Started

### Step 1: Install Dependencies
```bash
# Navigate to the project directory
cd OpenBCI_EEG_Interface

# Run the setup script
python setup.py
```

### Step 2: Connect Your OpenBCI Hardware
1. **Connect the Cyton board** to your computer via USB
2. **Note the COM port** (Windows) or device path (Linux/Mac)
   - Windows: Check Device Manager → Ports (COM & LPT)
   - Common ports: COM3, COM4, COM5, etc.

### Step 3: Run the Application
```bash
python src/main.py
```

### Step 4: Configure and Start
1. **Select COM port** from the dropdown menu
2. **Click "Connect"** to establish connection
3. **Click "Start Acquisition"** to begin EEG data collection
4. **Monitor real-time signals** in the visualization window

## 🎛️ GUI Features

### Connection Panel
- **COM Port Selection**: Choose your OpenBCI board port
- **Connect/Disconnect**: Establish or terminate connection
- **Status Display**: Shows current connection status

### Data Acquisition
- **Start/Stop Acquisition**: Control EEG data collection
- **Real-time Visualization**: Live EEG signal display
- **Multi-channel Support**: Up to 8 channels simultaneously

### Signal Processing
- **Notch Filter**: Remove 60Hz power line noise
- **Bandpass Filter**: Customizable frequency range (1-50 Hz)
- **Real-time Filtering**: Apply filters during acquisition

### Data Recording
- **Start/Stop Recording**: Save EEG data to file
- **Data Export**: Save as NumPy .npz format
- **Metadata Included**: Sampling rate, channel names, timestamps

## 📊 Understanding the Display

### EEG Plot
- **X-axis**: Time (seconds)
- **Y-axis**: Amplitude (microvolts)
- **Colors**: Each channel has a unique color
- **Channels**: Fp1, Fp2, C3, C4, P7, P8, O1, O2

### Status Information
- **Connection Status**: Green (connected), Red (disconnected)
- **Acquisition Status**: Blue (acquiring), Green (connected)
- **Info Log**: Timestamped messages about system events

## 🔧 Troubleshooting

### Connection Issues
- **Check COM port**: Ensure correct port is selected
- **Driver installation**: Install OpenBCI drivers if needed
- **USB connection**: Try different USB port or cable
- **Board power**: Ensure Cyton board is powered on

### Data Quality Issues
- **Electrode placement**: Ensure good skin contact
- **Impedance**: Check electrode impedance (< 5kΩ recommended)
- **Filtering**: Enable notch filter for power line noise
- **Artifacts**: Check for muscle or eye movement artifacts

### Performance Issues
- **Close other applications**: Free up system resources
- **Reduce buffer size**: Modify buffer_size in data_acquisition.py
- **Lower sampling rate**: If supported by your setup

## 📁 File Structure
```
OpenBCI_EEG_Interface/
├── src/
│   ├── main.py                 # Main application entry point
│   ├── data_acquisition.py     # OpenBCI communication module
│   ├── gui/
│   │   └── main_window.py      # PyQt5 GUI interface
│   └── signal_processing/
│       └── eeg_processor.py     # Signal processing utilities
├── data/                       # Data storage directory
├── requirements.txt            # Python dependencies
├── setup.py                    # Setup script
└── README.md                   # Project documentation
```

## 🧪 Testing Without Hardware

If you don't have the OpenBCI hardware yet, you can test the signal processing:

```bash
# Test signal processing module
python src/signal_processing/eeg_processor.py

# Test data acquisition module (will show available ports)
python src/data_acquisition.py
```

## 🔬 Advanced Usage

### Custom Signal Processing
```python
from src.signal_processing.eeg_processor import EEGSignalProcessor

processor = EEGSignalProcessor(sampling_rate=250)
processed_data = processor.apply_complete_preprocessing(raw_data)
features = processor.compute_spectral_features(processed_data)
```

### Data Analysis
```python
import numpy as np

# Load saved data
data = np.load('eeg_data.npz')
eeg_data = data['eeg_data']
sampling_rate = data['sampling_rate']
channel_names = data['channel_names']

# Analyze the data
print(f"Data shape: {eeg_data.shape}")
print(f"Duration: {eeg_data.shape[1] / sampling_rate:.2f} seconds")
```

## 📞 Support

- **OpenBCI Documentation**: https://docs.openbci.com/
- **BrainFlow Documentation**: https://brainflow.readthedocs.io/
- **GitHub Issues**: Report bugs or request features

## 🎯 Next Steps

1. **Familiarize yourself** with the GUI interface
2. **Practice electrode placement** for optimal signal quality
3. **Experiment with filters** to improve data quality
4. **Record sample data** for analysis
5. **Explore signal processing** features for your specific application

Happy EEG recording! 🧠⚡


