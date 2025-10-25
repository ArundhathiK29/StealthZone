"""
test_installation.py

Quick test to verify all dependencies are installed correctly.
Run this before using the main application.
"""
import sys

def test_imports():
    """Test if all required packages can be imported."""
    print("=" * 50)
    print("Testing StealthZone Dependencies")
    print("=" * 50 + "\n")
    
    tests = [
        ("OpenCV", "cv2"),
        ("MediaPipe", "mediapipe"),
        ("NumPy", "numpy"),
        ("InsightFace", "insightface"),
        ("ONNX Runtime", "onnxruntime"),
        ("SciPy", "scipy"),
    ]
    
    all_passed = True
    
    for name, module in tests:
        try:
            __import__(module)
            version = None
            try:
                mod = sys.modules[module]
                version = getattr(mod, '__version__', None)
            except:
                pass
            
            if version:
                print(f"✅ {name:15} - Installed (v{version})")
            else:
                print(f"✅ {name:15} - Installed")
        except ImportError as e:
            print(f"❌ {name:15} - MISSING ({e})")
            all_passed = False
    
    print("\n" + "=" * 50)
    
    if all_passed:
        print("✅ All dependencies installed successfully!")
        print("\nYou can now:")
        print("  1. Register faces: python register_face.py")
        print("  2. Run the app:    python face_privacy_interactive.py")
    else:
        print("❌ Some dependencies are missing.")
        print("Run: pip install -r requirements.txt")
    
    print("=" * 50 + "\n")
    
    return all_passed

def test_webcam():
    """Test if webcam is accessible."""
    print("\nTesting webcam access...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print("✅ Webcam is accessible and working!")
                return True
            else:
                print("⚠️  Webcam opened but couldn't read frame")
                return False
        else:
            print("❌ Could not open webcam (might be in use)")
            return False
    except Exception as e:
        print(f"❌ Webcam test failed: {e}")
        return False

def test_face_registry():
    """Test if face registry module works."""
    print("\nTesting face registry module...")
    try:
        from face_registry import load_registry, save_registry
        registry = load_registry()
        print(f"✅ Face registry loaded ({len(registry)} faces registered)")
        return True
    except Exception as e:
        print(f"❌ Face registry test failed: {e}")
        return False

if __name__ == "__main__":
    print("\n")
    imports_ok = test_imports()
    
    if imports_ok:
        webcam_ok = test_webcam()
        registry_ok = test_face_registry()
        
        if imports_ok and webcam_ok and registry_ok:
            print("\n🎉 All systems ready! Your StealthZone installation is complete.")
        elif not webcam_ok:
            print("\n⚠️  Setup complete but webcam issues detected.")
            print("   Make sure no other application is using the webcam.")
        else:
            print("\n⚠️  Setup mostly complete but some tests failed.")
    else:
        print("\n❌ Please install missing dependencies first.")
    
    print("\n")
