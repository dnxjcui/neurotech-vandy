"""
Main entry point for the OpenBCI EEG Interface application.
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

from gui.main_window import main

if __name__ == "__main__":
    main()


