# FRED-V2 (Funny Rude Educated Droid)

A locally-hosted, personalized AI assistant system.

## WebRTC Streaming

`webrtc_server.py` runs a lightweight server that accepts WebRTC offers from the
Raspberry Pi glasses. The Pi client code lives in `pi_client/` and can be run
with:

```bash
python client.py --server http://<fred-ip>:8080
```

## Intelligent Search Scoring

`web_search_core.intelligent_search` ranks gathered links by a semantic
relevance score.  `calculate_relevance_score` embeds the query and each link
title using the configured model and computes their cosine similarity.
Links with higher scores are assumed more related to the query and the top
results are selected directly—no link-analysis LLM call is made.
