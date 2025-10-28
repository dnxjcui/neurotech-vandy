"""
Neurotech CLI Runner

This module provides command-line interface for running different neurotech
operations including receiving data from OpenBCI Cyton boards.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from streaming.cyton import configure_board, receive


def setup_argument_parser():
    """
    Set up command line argument parser.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Neurotech CLI - Interface for neurotechnology operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --receive                    # Receive data from Cyton board
  python run.py --receive --duration 30      # Receive data for 30 seconds
  python run.py --receive --serial COM3      # Receive data from specific serial port
  python run.py --receive --mac 00:11:22:33:44:55  # Receive data via Bluetooth
        """
    )
    
    # Main operation arguments
    parser.add_argument(
        "--receive",
        action="store_true",
        help="Receive data from OpenBCI Cyton board"
    )
    
    # Cyton-specific arguments
    parser.add_argument(
        "--serial",
        type=str,
        help="Serial port for Cyton board connection (e.g., COM3, /dev/ttyUSB0)"
    )
    
    parser.add_argument(
        "--mac",
        type=str,
        help="MAC address for Cyton board Bluetooth connection"
    )
    
    parser.add_argument(
        "--duration",
        type=float,
        help="Duration to receive data in seconds (default: receive indefinitely)"
    )
    
    parser.add_argument(
        "--no-print",
        action="store_true",
        help="Don't print electrode data to console"
    )
    
    return parser


def run_cyton_receive(args):
    """
    Run Cyton data reception based on command line arguments.
    
    Args:
        args: Parsed command line arguments.
    """
    print("Initializing Cyton board connection...")
    
    # Configure board
    streamer = configure_board(
        serial_port=args.serial,
        mac_address=args.mac
    )
    
    if not streamer:
        print("Failed to configure Cyton board. Exiting.")
        return 1
    
    try:
        # Print board information
        board_info = streamer.get_board_info()
        if "error" not in board_info:
            print(f"Board Info:")
            print(f"  Sampling Rate: {board_info['sampling_rate']} Hz")
            print(f"  EEG Channels: {board_info['eeg_channels']}")
            print(f"  Number of Channels: {board_info['num_eeg_channels']}")
            print()
        
        # Receive data
        print("Starting data reception...")
        data = receive(
            streamer=streamer,
            duration=args.duration,
            print_data=not args.no_print
        )
        
        if data is not None:
            print(f"Received data shape: {data.shape}")
            print("Data reception completed successfully.")
        else:
            print("No data received.")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 0
    except Exception as e:
        print(f"Error during data reception: {e}")
        return 1
    finally:
        # Clean up
        streamer.stop_streaming()


def main():
    """
    Main entry point for the CLI application.
    """
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Check if any operation is specified
    if not args.receive:
        parser.print_help()
        print("\nError: No operation specified. Use --receive to start data reception.")
        return 1
    
    # Validate arguments
    if args.serial and args.mac:
        print("Error: Cannot specify both --serial and --mac. Choose one connection method.")
        return 1
    
    if args.duration and args.duration <= 0:
        print("Error: Duration must be positive.")
        return 1
    
    # Run the specified operation
    if args.receive:
        return run_cyton_receive(args)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
