"""
Quick test to verify InsightFace loads correctly
"""
print("[TEST] Starting InsightFace test...")

try:
    from insightface.app import FaceAnalysis
    print("[TEST] ✓ InsightFace imported successfully")
    
    print("[TEST] Loading model with CPU provider...")
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    
    print("[TEST] Preparing model with ctx_id=-1 (CPU mode)...")
    app.prepare(ctx_id=-1, det_size=(640, 640))
    
    print("[TEST] ✓✓✓ SUCCESS! InsightFace is working correctly!")
    print(f"[TEST] Model info: {app}")
    
except Exception as e:
    print(f"[TEST] ✗✗✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    print("\n[HINT] If you see 'No module named insightface', run:")
    print("       pip install insightface")
    print("\n[HINT] If you see ONNX errors, run:")
    print("       pip install onnxruntime")
