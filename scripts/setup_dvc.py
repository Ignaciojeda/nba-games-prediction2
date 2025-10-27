import subprocess
import os
import sys

def setup_dvc():
    """Configuración completa de DVC con verificación de Git"""
    
    print("🔧 Configurando DVC...")
    
    # Verificar si Git está inicializado
    try:
        subprocess.run(["git", "status"], check=True, capture_output=True)
        print("✅ Git repository encontrado")
    except subprocess.CalledProcessError:
        print("❌ Git no está inicializado. Inicializando...")
        subprocess.run(["git", "init"], check=True)
        print("✅ Git inicializado")
    
    # Inicializar DVC
    try:
        subprocess.run(["dvc", "init"], check=True)
        print("✅ DVC inicializado")
        
        # Configurar almacenamiento local
        storage_path = os.path.abspath("dvc_storage")
        os.makedirs(storage_path, exist_ok=True)
        
        subprocess.run([
            "dvc", "remote", "add", "-d", "local_storage", storage_path
        ], check=True)
        
        print(f"✅ Almacenamiento DVC configurado en: {storage_path}")
        
        # Commit inicial de DVC
        subprocess.run(["git", "add", ".dvc", ".dvcignore"], check=True)
        subprocess.run(["git", "commit", "-m", "Initialize DVC"], check=True)
        
        print("✅ Configuración de DVC completada")
        
    except subprocess.CalledProcessError as e:
        print(f"⚠️  DVC podría estar ya configurado: {e}")
        return False
    
    return True

if __name__ == "__main__":
    setup_dvc()