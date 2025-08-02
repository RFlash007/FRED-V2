# FRED-V2 (Funny Rude Educated Droid)

A locally-hosted, personalized AI assistant system.

## WebRTC Streaming

`webrtc_server.py` runs a lightweight server that accepts WebRTC offers from the
Raspberry Pi glasses. The Pi client code lives in `pi_client/` and can be run
with:

```bash
python client.py --server http://<fred-ip>:8080
```

## Intelligent Search

FRED-V2 performs web searches using `intelligent_search`, which ranks result
links purely by semantic similarity between your query and the page title. The
relevance score is computed via embeddings in `calculate_relevance_score` and
the top results are fetched without any additional LLM analysis.
