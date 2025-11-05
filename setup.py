"""
Setup script for OpenBCI EEG Interface

This script helps set up the development environment and install dependencies.
"""

import subprocess
import sys
import os
import platform

def install_requirements():
    """Install required packages from requirements.txt"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("[OK] All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error installing packages: {e}")
        return False

def check_brainflow_installation():
    """Check if BrainFlow is properly installed"""
    try:
        import brainflow
        print(f"[OK] BrainFlow version {brainflow.__version__} is installed")
        return True
    except ImportError:
        print("[ERROR] BrainFlow is not installed")
        return False

def check_pyqt5_installation():
    """Check if PyQt5 is properly installed"""
    try:
        import PyQt5
        print("[OK] PyQt5 is installed")
        return True
    except ImportError:
        print("[ERROR] PyQt5 is not installed")
        return False

def check_pyqtgraph_installation():
    """Check if PyQtGraph is properly installed"""
    try:
        import pyqtgraph
        print(f"[OK] PyQtGraph version {pyqtgraph.__version__} is installed")
        return True
    except ImportError:
        print("[ERROR] PyQtGraph is not installed")
        return False

def check_system_requirements():
    """Check system requirements"""
    print("Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 7:
        print(f"[OK] Python {python_version.major}.{python_version.minor} is compatible")
    else:
        print(f"[ERROR] Python {python_version.major}.{python_version.minor} is not compatible (requires 3.7+)")
        return False
    
    # Check operating system
    os_name = platform.system()
    print(f"[OK] Operating system: {os_name}")
    
    return True

def create_data_directories():
    """Create necessary data directories"""
    directories = ['data', 'data/recordings', 'data/exports', 'logs']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"[OK] Created directory: {directory}")
        else:
            print(f"[OK] Directory already exists: {directory}")

def test_imports():
    """Test if all required modules can be imported"""
    print("\nTesting imports...")
    
    modules = [
        ('brainflow', 'BrainFlow'),
        ('PyQt5', 'PyQt5'),
        ('pyqtgraph', 'PyQtGraph'),
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('matplotlib', 'Matplotlib'),
        ('pandas', 'Pandas')
    ]
    
    all_imports_successful = True
    
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print(f"[OK] {display_name} imported successfully")
        except ImportError as e:
            print(f"[ERROR] Failed to import {display_name}: {e}")
            all_imports_successful = False
    
    return all_imports_successful

def main():
    """Main setup function"""
    print("OpenBCI EEG Interface Setup")
    print("=" * 40)
    
    # Check system requirements
    if not check_system_requirements():
        print("\n✗ System requirements not met. Please upgrade Python to 3.7+")
        return False
    
    # Install requirements
    if not install_requirements():
        print("\n✗ Failed to install requirements")
        return False
    
    # Test imports
    if not test_imports():
        print("\n✗ Some modules failed to import")
        return False
    
    # Create directories
    create_data_directories()
    
    print("\n" + "=" * 40)
    print("[OK] Setup completed successfully!")
    print("\nNext steps:")
    print("1. Connect your OpenBCI Cyton board to your computer")
    print("2. Note the COM port (Windows) or device path (Linux/Mac)")
    print("3. Run the application: python src/main.py")
    print("4. Select the correct COM port in the GUI")
    print("5. Click 'Connect' to establish connection")
    print("6. Click 'Start Acquisition' to begin EEG data collection")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
