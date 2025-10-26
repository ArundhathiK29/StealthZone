# Normal Mode Feature

## Overview
A new privacy mode toggle that allows you to view all faces without any processing - no blur, no distortion, just the raw camera feed.

## Usage

### Toggle Normal Mode
Press **`N`** to toggle Normal Mode ON/OFF

**When ENABLED:**
- All faces (registered and unknown) are shown without blur or distortion
- Settings overlay shows: `Mode: NORMAL (no privacy)` (orange warning)
- Useful for checking camera quality or testing recognition without privacy

**When DISABLED:**
- Returns to standard privacy protection (blur or Anti-KYC)
- Unknown faces will be blurred by default
- Press 'K' to switch to Anti-KYC mode instead

## Modes Comparison

| Mode | Key | Unknown Faces | Privacy Level | Use Case |
|------|-----|---------------|---------------|----------|
| **Normal** | N | Unprocessed | ❌ None | Testing, setup |
| **Blur** | Default | Blurred | ✓ Standard | General privacy |
| **Anti-KYC** | K | Distorted | ✓✓ Advanced | Defeat V-KYC systems |

## Mode Priority

When you enable a mode, other modes are automatically disabled:

1. Press **`N`** → Normal mode ON, Anti-KYC OFF
2. Press **`K`** → Anti-KYC mode ON, Normal mode OFF
3. Press **`N`** again → Normal mode OFF, returns to Blur mode

## Visual Indicators

### Settings Overlay (Press T)
```
STEALTHZONE PRO
FPS: 30.5 | Faces: 1 | Registered: 2
Blur: PIXELATE (intensity 15)
Recognition: ON (2 enrolled)
Mode: NORMAL (no privacy)              ← Orange warning when active
Mode: BLUR (pixelate, intensity 5)     ← Gray when blur active
Mode: ANTI-KYC (intensity 5/10)        ← Cyan when Anti-KYC active

CONTROLS:
R: Register new face
F: Toggle recognition
N: Normal mode (no privacy)            ← New control!
B: Blur intensity (curr: 15)
M: Blur mode (curr: pixelate)
K: Toggle Anti-KYC distortion
D: Distortion intensity (curr: 5)
T: Toggle settings
C: Capture photo
Q/ESC: Quit
```

## Keyboard Controls Summary

| Key | Function | Result |
|-----|----------|--------|
| **N** | Toggle Normal Mode | All faces visible, no processing |
| **K** | Toggle Anti-KYC | Distort unknown faces |
| **B** | Blur Intensity | Adjust blur level (when in blur mode) |
| **M** | Blur Mode | Cycle blur types |
| **T** | Toggle Settings | Show/hide overlay |

## Workflow Examples

### Example 1: Testing Camera Setup
1. Start app: `python face_privacy_pro.py`
2. Press **`N`** → Normal mode (see raw camera)
3. Adjust lighting, position, etc.
4. Press **`N`** again → Return to privacy protection

### Example 2: Switching Between Modes
1. Default: Blur mode (unknown faces blurred)
2. Press **`K`** → Anti-KYC mode (unknown faces distorted)
3. Press **`N`** → Normal mode (all faces visible)
4. Press **`N`** → Back to blur mode
5. Press **`K`** → Back to Anti-KYC mode

### Example 3: Face Registration
1. Press **`N`** → Normal mode (see yourself clearly)
2. Press **`R`** → Start registration
3. Enter name, press **`C`** to capture
4. Press **`S`** to save
5. Press **`N`** → Return to privacy mode

## Safety Features

- **Warning Color**: Normal mode shows ORANGE in settings (indicates low privacy)
- **Mode Exclusivity**: Only one mode active at a time
- **Easy Toggle**: Single key press to enable/disable
- **Clear Feedback**: Terminal prints mode changes

## Terminal Output

```
[NORMAL MODE ENABLED] - Unknown faces will be shown without blur or distortion
[INFO] All faces visible in original form

[NORMAL MODE DISABLED] - Privacy protection active
[INFO] Unknown faces will be blurred (press 'K' for Anti-KYC mode)
```

## Use Cases

### ✓ Good Uses for Normal Mode
- Testing camera quality
- Setting up lighting
- Verifying face detection works
- Checking registration accuracy
- Demonstrations (when privacy not needed)

### ✗ Avoid Normal Mode For
- Video calls with strangers
- Public streams
- Privacy-sensitive situations
- Production use

## Technical Details

### Implementation
- Added `normal_mode` boolean to AppState
- Modified face processing logic to skip blur/distortion when enabled
- Updated settings overlay to show current mode
- Added 'N' key handler with auto-disable of other modes

### Code Flow
```python
if state.normal_mode:
    # No processing - show face as-is
    pass
elif state.anti_kyc_mode:
    # Apply distortion
    frame = apply_anti_kyc_distortion(...)
else:
    # Apply blur (default)
    frame = apply_blur(...)
```

---

**Version:** 1.0  
**Added:** October 26, 2025  
**Part of:** StealthZone Pro Privacy Protection System
