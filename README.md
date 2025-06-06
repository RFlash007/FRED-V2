# FRED-V2 (Funny Rude Educated Droid)

A locally-hosted, personalized AI assistant system.

## WebRTC Streaming

`webrtc_server.py` runs a lightweight server that accepts WebRTC offers from the
Raspberry Pi glasses. The Pi client code lives in `pi_client/` and can be run
with:

```bash
python client.py --server http://<fred-ip>:8080
```
