#!/usr/bin/env python3
"""
Check if all required dependencies are installed for local development.
"""

import sys
import importlib

required_packages = [
    'flask',
    'flask_cors',
    'google.auth',
    'google.oauth2',
    'google.cloud.storage',
    'google.genai',
    'langchain',
    'langchain_community',
    'langchain_chroma',
    'langchain_core',
    'fitz',  # PyMuPDF
]

def check_package(package_name):
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def main():
    print("🔍 Checking required packages for local development...")
    print("=" * 60)
    
    missing_packages = []
    
    for package in required_packages:
        if check_package(package):
            print(f"✅ {package}")
        else:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    print("=" * 60)
    
    if missing_packages:
        print(f"\n❌ {len(missing_packages)} packages are missing:")
        for package in missing_packages:
            print(f"   • {package}")
        print("\n📦 Install missing packages with:")
        print("   pip install -r requirements.txt")
        print("\n   Or install individually:")
        for package in missing_packages:
            if package == 'fitz':
                print(f"   pip install PyMuPDF")
            else:
                print(f"   pip install {package}")
        return False
    else:
        print("✅ All required packages are installed!")
        return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
