# How to Test Anti-KYC Feature

## Quick Tests (Pick One)

### ✅ Test 1: Live Application Test (EASIEST)
**Currently Running!** The main app is already open.

**Steps:**
1. Press **`T`** to show settings overlay
2. Press **`K`** to enable Anti-KYC mode
   - Terminal shows: `[ANTI-KYC MODE ENABLED]`
   - Overlay shows: `Anti-KYC: ENABLED` (cyan)
3. Position an unknown face in camera
   - Should see subtle distortion instead of blur
4. Press **`D`** to cycle intensities (1-10)
   - Watch face change subtly with each level
5. Press **`K`** again to toggle back to blur
   - Notice the difference!

**What to look for:**
- ✓ Distorted face looks "mostly normal" to your eyes
- ✓ Subtle wavy/texture effect (more at higher intensities)
- ✓ Different from pixelated blur

---

### ✅ Test 2: Side-by-Side Visual Comparison
**Shows: Original | Blurred | Distorted**

**Run:**
```bash
python test_anti_kyc_visual.py
```

**Steps:**
1. Position face in camera
2. See three versions displayed side-by-side:
   - **Left**: Original (green box)
   - **Middle**: Blurred (orange box)
   - **Right**: Distorted (cyan box)
3. Press **`1-9`** to change distortion intensity
4. Compare visual differences
5. Press **`Q`** to quit

**What to look for:**
- ✓ Distorted version looks more "normal" than blurred
- ✓ Subtle warping around eyes/nose/mouth
- ✓ Slight texture noise
- ✓ Higher intensity = more visible effect

---

### ✅ Test 3: Face Recognition Effectiveness Test
**Tests if distortion actually defeats recognition**

**Run:**
```bash
python test_anti_kyc_effectiveness.py
```

**Steps:**
1. Position your face in camera
2. Press **`C`** to capture reference (original face)
3. Keep face in frame
4. Press **`T`** to test with distortion

**Results shown:**
```
Intensity  1: Distance = 0.25 → ❌ MATCHED (Anti-KYC FAILED)
Intensity  3: Distance = 0.35 → ❌ MATCHED (Anti-KYC FAILED)
Intensity  5: Distance = 0.48 → ✓ NO MATCH (Anti-KYC SUCCESS)
Intensity  7: Distance = 0.62 → ✓ NO MATCH (Anti-KYC SUCCESS)
Intensity 10: Distance = 0.85 → ✓ NO MATCH (Anti-KYC SUCCESS)
```

**What to look for:**
- ✓ Distance < 0.4 = Face MATCHED (bad for privacy)
- ✓ Distance > 0.4 = Face NOT MATCHED (good! Anti-KYC works)
- ✓ Higher intensities should have higher distances
- ✓ Intensity 5+ should defeat recognition

---

## Expected Results Summary

| Test Type | What You'll See |
|-----------|-----------------|
| **Visual** | Subtle face distortion, not pixelated blur |
| **Comparison** | Clear difference between blur/distortion |
| **Recognition** | Distance > 0.4 = recognition defeated ✓ |

## Keyboard Controls Reference

### Main Application (face_privacy_pro.py)
- **`K`** - Toggle Anti-KYC ON/OFF
- **`D`** - Cycle distortion intensity (1-10)
- **`T`** - Toggle settings display
- **`Q`** - Quit

### Visual Comparison Test (test_anti_kyc_visual.py)
- **`1-9`** - Set distortion intensity
- **`Q`** - Quit

### Effectiveness Test (test_anti_kyc_effectiveness.py)
- **`C`** - Capture reference face
- **`T`** - Test distortion effectiveness
- **`Q`** - Quit

---

## Troubleshooting

### "I don't see any difference"
- Increase intensity: Press `D` to get to 7-8
- The effect is subtle by design (defeats V-KYC but looks normal)
- Compare side-by-side (use test_anti_kyc_visual.py)

### "Face looks too distorted"
- Lower intensity: Press `D` to get to 3-4
- Default (5) is balanced for most use cases

### "Test says Anti-KYC FAILED"
- Normal at low intensities (1-3)
- Should succeed at intensity 5+
- If failing at 5+, the recognition threshold might be too lenient

### "No face detected"
- Ensure good lighting
- Face should be clearly visible
- Try moving closer to camera

---

## Quick Decision Guide

**Want to:**
- **Just see it work?** → Use Test 1 (main app, press K)
- **See visual difference?** → Use Test 2 (side-by-side)
- **Verify it defeats recognition?** → Use Test 3 (effectiveness)
- **All of the above?** → Run all three tests!

---

**Current Status:** Main application is RUNNING
- Press `K` to enable Anti-KYC mode right now!
- See settings overlay with `T`
