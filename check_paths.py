"""Check that all paths are correctly set to D drive."""

import os
from config import Config

def check_paths():
    config = Config()
    
    print("=== PATH VERIFICATION ===")
    print(f"Base directory: {config.base_dir}")
    print(f"Checkpoints: {config.checkpoint_dir}")
    print(f"Logs: {config.log_dir}")
    print(f"Data: {config.data_dir}")
    print(f"Temp: {config.temp_dir}")
    print(f"Expert games: {config.expert_games_dir}")
    
    print("\n=== ENVIRONMENT VARIABLES ===")
    print(f"TORCH_HOME: {os.environ.get('TORCH_HOME', 'Not set')}")
    print(f"TMPDIR: {os.environ.get('TMPDIR', 'Not set')}")
    print(f"TEMP: {os.environ.get('TEMP', 'Not set')}")
    print(f"TMP: {os.environ.get('TMP', 'Not set')}")
    
    print("\n=== DIRECTORY STATUS ===")
    directories = [
        config.checkpoint_dir,
        config.log_dir, 
        config.data_dir,
        config.temp_dir,
        config.expert_games_dir
    ]
    
    for directory in directories:
        exists = os.path.exists(directory)
        drive = os.path.splitdrive(directory)[0]
        print(f"{directory}: {'✓' if exists else '✗'} (Drive: {drive})")
    
    # Check if all paths are on D drive
    all_on_d = all(os.path.splitdrive(d)[0].upper() == 'D:' for d in directories)
    print(f"\nAll paths on D drive: {'✓ YES' if all_on_d else '✗ NO'}")

if __name__ == "__main__":
    check_paths()