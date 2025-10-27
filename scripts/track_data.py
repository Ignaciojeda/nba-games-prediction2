import subprocess
import os

def track_data_files():
    """Trackear archivos de datos con DVC"""
    
    data_files = [
        "data/01_raw/games.csv",
        "data/01_raw/games_details.csv", 
        "data/01_raw/teams.csv"
    ]
    
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"📁 Trackeando {file_path} con DVC...")
            subprocess.run(["dvc", "add", file_path], check=True)
            subprocess.run(["git", "add", f"{file_path}.dvc"], check=True)
        else:
            print(f"⚠️  {file_path} no encontrado")
    
    # Commit de datos
    subprocess.run(["git", "commit", "-m", "Add raw data files with DVC"], check=True)
    print("✅ Datos trackeados con DVC")

if __name__ == "__main__":
    track_data_files()