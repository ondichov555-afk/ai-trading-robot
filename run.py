"""
Startup script to run the complete application
Combines CLI and web interface
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    print("\n" + "="*70)
    print(" "*15 + "AI TRADING ROBOT - APPLICATION LAUNCHER")
    print("="*70)
    print("\nSelect how you want to run the application:\n")
    print("1. CLI Mode (Interactive Command Line)")
    print("   - Full control via command-line menu")
    print("   - Direct feedback and control")
    print("")
    print("2. Web Dashboard (Browser Interface)")
    print("   - Visual interface at http://localhost:5000")
    print("   - Real-time monitoring and control")
    print("   - Better for monitoring")
    print("")
    print("3. Exit")
    print("="*70)
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        print("\nStarting CLI mode...\n")
        from main import main as cli_main
        cli_main()
    
    elif choice == "2":
        print("\nStarting Web Dashboard...")
        print("Opening http://localhost:5000 in your browser\n")
        os.system("python app.py")
    
    elif choice == "3":
        print("Exiting...")
        sys.exit(0)
    
    else:
        print("Invalid option. Exiting.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nApplication interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)
