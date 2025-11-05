"""Quick test to verify GUI opens"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from PyQt5.QtWidgets import QApplication, QMessageBox
from gui.main_window import OpenBCIMainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    print("Creating main window...")
    window = OpenBCIMainWindow()
    window.show()
    print("Window should be visible now!")
    
    # Show a message to confirm it's running
    QMessageBox.information(window, "GUI Test", "GUI is running! Click OK to continue.")
    
    sys.exit(app.exec_())

