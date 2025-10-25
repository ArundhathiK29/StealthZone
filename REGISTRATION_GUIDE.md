# StealthZone - Facial Registration Feature Guide

## Overview

The facial registration feature allows influencers to register friends and colleagues so their faces remain **unblurred** automatically during video streaming, conferencing, or recording.

## How It Works

1. **Registration**: Friends are enrolled by capturing multiple images and computing face embeddings
2. **Recognition**: During real-time operation, the system identifies registered faces
3. **Privacy Protection**: Only registered faces + the primary user remain clear; all others are blurred

## Quick Start

### Step 1: Register a Friend

Run the registration script:

```bash
python register_face.py
```

**Instructions:**
1. Enter the person's name when prompted
2. Position their face in the webcam frame
3. Press `c` to capture an image (take 5-10 photos with varied expressions/angles)
4. Green rectangles confirm successful captures
5. Press `s` to save once you have enough captures
6. Press `q` to quit

**Tip:** For best results, capture images with:
- Different facial expressions (neutral, smiling, talking)
- Slight angle variations (looking left/right, up/down)
- Varied lighting conditions

### Step 2: Run the Main Application

Start the interactive privacy layer:

```bash
python face_privacy_interactive.py
```

**What Happens:**
- Registered faces are automatically identified and marked with their names
- Registered individuals are treated as "primary" and remain unblurred
- All unregistered faces are anonymized using your selected blur mode
- Face recognition runs every few frames to minimize performance impact

## Technical Details

### Face Embedding Model
- **Default Model**: Facenet (128-dimensional embeddings)
- **Alternatives**: VGG-Face, Facenet512, ArcFace, OpenFace (configurable in `face_registry.py`)

### Matching Algorithm
- **Method**: Cosine distance between embeddings
- **Threshold**: 0.4 (configurable in `face_registry.py`)
- **Lower threshold** = stricter matching = fewer false positives

### Storage
- **Format**: JSON file (`face_encodings.json`)
- **Location**: Same directory as the scripts
- **Structure**: `{"Name": [embedding array]}`
- **Security**: Local-only by default (no cloud upload)

## Configuration Options

Edit `face_registry.py` to customize:

```python
# Face recognition model (line 17)
MODEL_NAME = "Facenet"  # Options: VGG-Face, Facenet, Facenet512, OpenFace, ArcFace

# Match threshold (line 16)
MATCH_THRESHOLD = 0.4  # Lower = stricter (0.2-0.5 recommended)
```

Edit `face_privacy_interactive.py` to customize:

```python
# Recognition frequency (line 40)
FACE_REC_EVERY_N_FRAMES = 5  # Higher = better performance, less frequent checks
```

## Privacy & Consent

### Best Practices

✅ **DO:**
- Obtain explicit consent before registering someone's face
- Inform registered individuals about how their data is used
- Provide an easy way to remove registered faces
- Keep the `face_encodings.json` file secure and local

❌ **DON'T:**
- Register people without their knowledge or permission
- Share the face encodings file with third parties
- Use registered faces for purposes beyond agreed scope

### "Forget Me" Flow

To remove a registered person:

1. **Option A - Manual Delete:**
   - Open `face_encodings.json`
   - Remove the entry for that person
   - Save the file

2. **Option B - Script (coming soon):**
   ```bash
   python forget_face.py "Person Name"
   ```

## Troubleshooting

### "No face detected" during registration
- Ensure good lighting
- Face the camera directly
- Move closer to the camera
- Check that your webcam is working

### Face not being recognized
- Try registering more images (8-10 recommended)
- Adjust `MATCH_THRESHOLD` in `face_registry.py` (increase to 0.5 for looser matching)
- Ensure lighting conditions are similar to registration
- Re-register with more varied expressions/angles

### Performance issues
- Increase `FACE_REC_EVERY_N_FRAMES` (check less frequently)
- Switch to a lighter model like "OpenFace" in `face_registry.py`
- Reduce video resolution in `face_privacy_interactive.py`

### False positives (wrong people recognized)
- Decrease `MATCH_THRESHOLD` (try 0.3 or 0.2)
- Register with more diverse images
- Use a more robust model like "Facenet512" or "ArcFace"

## Advanced Usage

### Batch Registration

Register multiple people at once:

```bash
for name in "Alice" "Bob" "Charlie"
do
    echo $name | python register_face.py
done
```

### Export/Import Registry

**Backup:**
```bash
cp face_encodings.json face_encodings_backup.json
```

**Restore:**
```bash
cp face_encodings_backup.json face_encodings.json
```

### View Registered Faces

```bash
python -c "import json; data=json.load(open('face_encodings.json')); print('\\n'.join(data.keys()))"
```

## Performance Benchmarks

| Setup | FPS (No Recognition) | FPS (With Recognition) |
|-------|---------------------|------------------------|
| Basic (1 face) | ~30 FPS | ~25 FPS |
| Medium (3-5 faces) | ~28 FPS | ~20 FPS |
| Heavy (8+ faces) | ~25 FPS | ~15 FPS |

*Tested on Intel i5, 8GB RAM, integrated graphics*

## Future Enhancements

- [ ] Encryption of face embeddings at rest
- [ ] GUI for registration (no command line needed)
- [ ] Cloud sync for multiple devices (opt-in)
- [ ] Temporary guest mode (session-only recognition)
- [ ] Face clustering for group registration
- [ ] GDPR compliance tools (export, delete, audit logs)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the code comments in `face_registry.py` and `register_face.py`
3. Open an issue on the StealthZone GitHub repository

---

**Built with ❤️ for the Hashcode Hackathon**
