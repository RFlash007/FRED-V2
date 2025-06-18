# Pi Client Directory Cleanup - OLLIE-TEC Engineering Report

## 🧹 Cleanup Summary

**Objective:** Consolidate F.R.E.D. Pi client into a single startup file with local speech-to-text functionality.

## 📁 Files Consolidated

### ✅ New Consolidated File
- **`fred_pi_client.py`** - Single startup file combining:
  - Local STT processing (from `pi_stt_service.py`)
  - Camera vision capture 
  - F.R.E.D. server communication
  - Audio playback
  - [Vault-Tec themed interface as requested]

### 📦 Files Marked for Removal
- **`client.py`** - Old WebRTC streaming client (redundant)
- **`pi_stt_service.py`** - Separate STT service (now integrated)
- **`audio_capture_alternative.py`** - Alternative audio capture (functionality integrated)
- **`diagnose_audio.py`** - Diagnostic tool (can be removed after cleanup)

### 🔧 Files to Keep
- **`requirements.txt`** - Dependencies
- **`ollietec_theme.py`** - OLLIE-TEC theming
- **`install_local_stt.sh`** - STT setup script
- **`LOCAL_STT_SETUP.md`** - Setup documentation

## 🚀 Usage

Start F.R.E.D. Pi client with local STT:
```bash
cd pi_client
python fred_pi_client.py
```

With specific server:
```bash
python fred_pi_client.py --server http://192.168.1.100:8000
```

## 🔄 Migration Notes

- All functionality from old files is preserved
- Uses sounddevice for audio capture (more reliable than aiortc MediaPlayer)
- Maintains wake word detection and buffering
- Compatible with existing F.R.E.D. server endpoints
- [Preserves user preferences for terminal display and native resolution]

---
*OLLIE-TEC Wasteland Engineering Division - Vault-Tec Approved™* 