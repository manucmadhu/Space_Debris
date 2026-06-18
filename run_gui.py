import sys
import subprocess
from pathlib import Path

def main():
    gui_script = Path(__file__).parent / "GUI" / "gui_app.py"
    
    if not gui_script.exists():
        print(f"Error: GUI script not found at {gui_script}")
        sys.exit(1)
        
    print("Launching Space Debris GUI Dashboard...")
    try:
        subprocess.run([sys.executable, str(gui_script)])
    except KeyboardInterrupt:
        print("\nGUI session closed.")
    except Exception as e:
        print(f"Error starting GUI: {e}")

if __name__ == "__main__":
    main()
