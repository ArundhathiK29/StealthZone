# 🎯 StealthZone - Quick Start Guide

## Facial Registration Feature is Now Live! 🎉

### ✅ Installation Complete

All dependencies have been successfully installed. You're ready to use the facial registration feature!

---

## 🚀 Quick Start (3 Simple Steps)

### Step 1: Verify Installation
```bash
python test_installation.py
```
✅ Should show: "All systems ready!"

### Step 2: Register Your First Friend
```bash
python register_face.py
```
- Enter their name (e.g., "Alice")
- Press `c` to capture (take 5-10 photos)
- Press `s` to save

### Step 3: Run the App
```bash
python face_privacy_interactive.py
```
- Follow the on-screen setup wizard
- Registered friends will appear with their names
- Everyone else will be blurred!

---

## 📖 Complete Usage

### Register Multiple People
```bash
# Run once for each person
python register_face.py    # Register Alice
python register_face.py    # Register Bob
python register_face.py    # Register Charlie
```

### View Who's Registered
```bash
python forget_face.py
# Shows list without deleting
# Press 'quit' to exit without changes
```

### Remove Someone
```bash
python forget_face.py "Alice"
# Or use interactive mode:
python forget_face.py
```

---

## 🎮 In-App Controls

| Key | Action |
|-----|--------|
| `s` | Open settings menu |
| `h` | Toggle HUD display |
| `n` | Next primary face |
| `p` | Previous primary face |
| `q` | Quit application |

---

## 💡 Tips for Best Results

### During Registration:
- ✅ Good lighting (face clearly visible)
- ✅ Look at camera directly
- ✅ Capture varied expressions (smile, neutral, talking)
- ✅ Slight angle changes (left, right, up, down)
- ❌ Don't wear sunglasses or masks
- ❌ Avoid extreme shadows

### During Real-Time Use:
- Similar lighting to registration works best
- Face should be clearly visible (not too far)
- System checks every 5 frames (low performance impact)
- Recognition confirmed by name label in green

---

## 🔧 Troubleshooting

### "Face not recognized"
1. Check lighting conditions
2. Move closer to camera
3. Re-register with more varied images
4. Adjust threshold in `face_registry.py`:
   ```python
   MATCH_THRESHOLD = 0.5  # Increase for looser matching
   ```

### Performance Issues
1. Increase check interval in `face_privacy_interactive.py`:
   ```python
   FACE_REC_EVERY_N_FRAMES = 10  # Check less frequently
   ```
2. Close other applications using webcam
3. Reduce video resolution

### "Webcam not found"
1. Check if another app is using the camera
2. Try unplugging and replugging USB camera
3. Check Windows camera permissions

---

## 📁 Files You Created

```
StealthZone/
├── face_encodings.json         ← Your registered faces (created after first registration)
├── registered_faces/           ← Saved face images (created automatically)
│   ├── Alice/
│   ├── Bob/
│   └── Charlie/
├── face_registry.py            ← Registry management
├── register_face.py            ← Registration tool
├── forget_face.py              ← Deletion tool
├── face_privacy_interactive.py ← Main app (updated)
└── requirements.txt            ← Dependencies (updated)
```

---

## 🎓 Next Steps

### Try Advanced Features:
- Change blur styles (gaussian, pixelate, mosaic)
- Adjust blur intensity
- Manage multiple primary faces

### Customize Settings:
- Edit `face_registry.py` to change the AI model
- Edit thresholds for stricter/looser matching
- Configure performance vs accuracy trade-offs

### Share & Deploy:
- Demo at your hackathon presentation
- Share `REGISTRATION_GUIDE.md` with teammates
- Backup `face_encodings.json` for safety

---

## 📞 Need Help?

1. **Check Documentation:**
   - `REGISTRATION_GUIDE.md` - Comprehensive user guide
   - `IMPLEMENTATION_SUMMARY.md` - Technical details

2. **Test Components:**
   ```bash
   python test_installation.py  # Verify all dependencies
   ```

3. **Debug Mode:**
   - Check console output for error messages
   - Look for `[INFO]`, `[WARN]`, `[ERROR]` tags

---

## ✨ Pro Tips

1. **Best Recognition**: Register in the same environment where you'll use the app
2. **Privacy First**: Always get consent before registering someone
3. **Regular Cleanup**: Use `forget_face.py` to remove old registrations
4. **Performance**: Start with 1-2 registered people, add more as needed
5. **Backup**: Copy `face_encodings.json` before making changes

---

## 🎉 You're All Set!

Your StealthZone facial registration system is ready to use.

**Time to register your first friend and see the magic happen!**

```bash
python register_face.py
```

---

**Built for the Hashcode Hackathon** 🏆
**Version**: 1.0 with Facial Registration
**Status**: ✅ Fully Operational
