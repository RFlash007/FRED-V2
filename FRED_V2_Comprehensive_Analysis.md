# F.R.E.D. V2 - Comprehensive Analysis

**Project:** F.R.E.D. V2 (Funny Rude Educated Droid)
**Author:** Ian The AI Engineer/Prompt Engineer for Ollie-Tech

---

## 1. Executive Summary

F.R.E.D. V2 is a sophisticated, distributed AI agent ecosystem designed for real-world interaction and continuous learning. It consists of a central "brain" (the main server), a physical interface running on a Raspberry Pi (the "body"), and a complex, multi-layered memory architecture that enables it to learn from conversations and proactively research new topics. The system is designed with a clear separation of concerns, offloading real-time sensory processing to dedicated services and gateways, while the core server focuses on higher-level reasoning, memory, and learning.

---

## 2. System Architecture & Components

The system is broken down into three main parts: the Pi Client, the WebRTC Gateway Server, and the Main F.R.E.D. Server.

### 2.1. Pi Client (`pi_client/fred_pi_client.py`)

- **Role:** The physical embodiment of F.R.E.D., running on a Raspberry Pi, likely integrated into a wearable device ("Pi Glasses"). It is responsible for all direct interaction with the user and their environment.
- **Key Functions:**
    - **Local Speech-to-Text (STT):** Uses a highly optimized `Vosk` model to perform wake-word detection ("Hey Fred") and transcribe user speech locally. This is efficient and reduces network traffic.
    - **Camera & Audio I/O:** Captures video from the PiCamera and audio from a microphone. It plays F.R.E.D.'s spoken responses through a speaker.
    - **WebRTC Communication:** Connects to the WebRTC Gateway server to send transcribed text and captured images, and to receive audio for playback.

### 2.2. WebRTC Gateway Server (`webrtc_server.py`)

- **Role:** Acts as a secure and robust bridge between the Pi Client and the Main Server. It handles the complexities of real-time communication.
- **Key Functions:**
    - **Manages WebRTC Connections:** Uses `aiortc` to handle the peer-to-peer connection with the Pi, including the crucial data channel for text and image transfer.
    - **Authentication & Rate Limiting:** Secures the connection endpoint, ensuring only authorized clients can connect.
    - **Relaying Information:**
        - Receives transcribed text from the Pi and forwards it to the Main Server via an HTTP request.
        - Receives image data from the Pi and passes it to the `VisionService`.
        - Receives generated audio from the Main Server (via Socket.IO) and sends it to the Pi for playback.

### 2.3. Main F.R.E.D. Server (`app.py`)

- **Role:** The central "brain" of the entire system. It's a Flask application that handles all cognitive tasks.
- **Key Functions:**
    - **Core Conversational Logic:** Receives user messages, orchestrates calls to the memory systems and other services to build context, and interacts with the primary Ollama LLM.
    - **Tool Use:** Provides the LLM with a set of tools (e.g., `addTaskToAgenda`, `enroll_person`) and handles their execution.
    - **Text-to-Speech (TTS):** Generates F.R.E.D.'s voice responses using a Coqui TTS model, with support for custom voice cloning.
    - **State Management:** Manages the short-term conversation history in a thread-safe manner.
    - **Socket.IO Hub:** Communicates with the WebRTC Gateway and potentially a web UI for real-time events.

### 2.4. Sensory Services

- **Vision Service (`vision_service.py`):** F.R.E.D.'s "eyes". It periodically requests images from the Pi, uses the `PersonaService` to recognize faces, and then uses a Vision-Language Model (VLM) to generate a textual description of the scene. This description provides rich visual context to the main LLM.
- **Persona Service (`persona_service.py`):** F.R.E.D.'s "face memory". It maintains a database of known individuals and their face embeddings. It includes an auto-improvement feature to refine its recognition accuracy over time.
- **STT Service (`stt_service.py`):** A server-side STT engine that can process audio captured directly from the server's microphone or streamed from the Pi. It serves as a fallback or alternative to the Pi's local STT.

---

## 3. Memory Architecture

F.R.E.D.'s ability to learn is built on a sophisticated, multi-layered memory architecture. This section provides a high-level overview, followed by a detailed deep dive into its components.

### 3.1. L1 - Working Memory (Conversation History)

- **Location:** `app.py` (`FREDState` class)
- **Description:** A simple, rolling list of the most recent user and assistant messages. This serves as the immediate context for the LLM. It is capped at a fixed number of messages to prevent context window overflow.

### 3.2. L2 - Episodic Cache (`memory/L2_memory.py`)

- **Role:** The "Recent Past" memory. It bridges the gap between short-term working memory and long-term knowledge.
- **Technology:** `DuckDB`
- **Description:** This system automatically analyzes the conversation history. Using semantic similarity, it detects when a topic of conversation has ended. It then uses an LLM to distill that conversation chunk into a structured summary, including the topic, key outcomes, and entities involved. This summary, along with its vector embedding, is stored in a `DuckDB` database. When a new user query comes in, this database is semantically searched to find relevant past conversations to inject into the LLM's prompt.

### 3.3. L3 - Long-Term Knowledge (`memory/L3_memory.py`)

- **Role:** F.R.E.D.'s permanent, structured, long-term memory.
- **Technology:** `DuckDB` (Knowledge Graph)
- **Description:** The L3 memory is a knowledge graph composed of "nodes" (pieces of information) and "edges" (the relationships between them).
    - **Ingestion:** Information enters the L3 graph either from the consolidation of L2 memories or as the output of the A.R.C.H./D.E.L.V.E. research system.
    - **Automated Edge Creation:** When a new node is added, a background task is created. This task uses an LLM to analyze the new node and compare it to existing nodes in the graph, automatically determining the correct relationship type (e.g., `causes`, `partOf`, `instanceOf`) and creating the corresponding edge. This allows F.R.E.D. to build a complex web of interconnected knowledge autonomously.

### 3.4. Memory Architecture - Deep Dive

F.R.E.D.'s cognitive abilities are rooted in a sophisticated, multi-layered memory architecture designed for both rapid contextual recall and long-term, structured learning.

**L2 Episodic Cache: The "Recent Past"**

The L2 memory (`memory/L2_memory.py`) acts as a semi-transient buffer, intelligently summarizing recent conversations to provide context without overloading the main LLM's working memory.

- **Triggering Mechanism:** An L2 summary is not created on every conversational turn. Instead, creation is event-driven, based on logic in `should_create_l2_summary`:
    - **Semantic Trigger:** A significant topic change is detected.
        - **Logic:** The system maintains a rolling average embedding of the last `L2_ROLLING_AVERAGE_WINDOW` (6) user messages. When a new user message arrives, its embedding is compared to this average. If the cosine similarity drops below `L2_SIMILARITY_THRESHOLD` (0.6), a topic shift is declared, and the preceding conversation chunk is queued for summarization.
    - **Fallback Trigger:** A single topic persists for too long.
        - **Logic:** If the number of turns since the last summary exceeds `L2_FALLBACK_TURN_LIMIT` (15), a summary is created automatically. This prevents long-running, single-topic conversations from consuming the entire L1 working memory.

- **Summarization Process:** Once triggered, the conversation chunk is passed to an LLM for analysis (`analyze_conversation_chunk`).
    - **Persona (D.R.E.A.M.):** The `L2_ANALYSIS_SYSTEM_PROMPT` casts the LLM as **D.R.E.A.M. (Data Recapitulation & Episodic Abstraction Module)**, F.R.E.D.'s "subconscious mind." Its mission is to be a clinical, objective observer of the "dream-like" conversational data.
    - **Prompt (`L2_ANALYSIS_PROMPT`):** D.R.E.A.M. is instructed to analyze the provided text and output a JSON object containing a `topic`, `key_outcomes`, `entities_mentioned`, `user_sentiment`, and a `raw_text_summary` written in its own words. It is explicitly told to use the trigger condition (`semantic_change` vs. `fallback_turn_limit`) as a hint for its analysis.

- **Storage & Retrieval:**
    - **Storage:** The resulting JSON summary and its vector embedding are stored in the `l2_episodic_summaries` table in `memory/L2_episodic_cache.db`.
    - **Retrieval:** When F.R.E.D. formulates a response, `query_l2_context` embeds the latest user message and performs a semantic search against the L2 database to find the `L2_RETRIEVAL_LIMIT` (2) most relevant recent conversation summaries to inject into the main prompt.

**L3 Knowledge Graph: Permanent, Interconnected Knowledge**

The L3 memory (`memory/L3_memory.py`) is F.R.E.D.'s permanent brain, implemented as a knowledge graph in `DuckDB`.

- **Node Creation:** A memory node is created via the `add_memory` tool, at the end of a research cycle, or during sleep cycle consolidation.
- **Automated Edge Creation:** The true power of the L3 graph lies in its ability to autonomously form connections between nodes.
    - **Trigger:** When a new node is created, a task is added to the `pending_edge_creation_tasks` table.
    - **Process:** The `process_pending_edges` function runs in the background. It finds the `AUTO_EDGE_SIMILARITY_CHECK_LIMIT` (3) most semantically similar nodes to the new node.
    - **LLM Adjudication (`determine_edge_type_llm`):** For each new-similar pair, an LLM is called with the `L3_EDGE_TYPE_PROMPT`. This prompt provides the text of both nodes and a detailed list of possible relationship types (e.g., `instanceOf`, `causes`, `partOf`, `updates`). The LLM's sole task is to return a JSON object with the single most appropriate `relationship_type`, which is then stored as a directed edge in the database.

---

## 4. The 3-Agent Research Setup (A.R.C.H./D.E.L.V.E.)

This is F.R.E.D.'s system for proactive, offline learning and deep investigation, managed by the `agenda_system.py` and executed by `arch_delve_research.py`. When the user asks a question that requires information F.R.E.D. doesn't have, the `addTaskToAgenda` tool is used. This process is then handled by a simulated research team during a "sleep cycle".

### 4.1. The Agenda (`memory/agenda_system.py`)

- A task queue stored in the `DuckDB` database. It holds research tasks that F.R.E.D. needs to work on. When a task is completed, a notification is generated for the user.

### 4.2. The Research Team (`memory/arch_delve_research.py`)

This is a multi-agent conversational framework that simulates a research team to ensure high-quality, verified results.

1.  **A.R.C.H. (Autonomous Research & Coordination Hub): The Research Lead.**
    - **Role:** Decomposes the high-level research goal from the agenda into a series of smaller, concrete steps. It creates the plan and gives instructions to D.E.L.V.E. It reviews the results and decides if the research is complete.

2.  **D.E.L.V.E. (Data-Extraction & Link-Visiting Entity): The Researcher.**
    - **Role:** Executes the plan from A.R.C.H. It uses tools like `search_web` and `read_webpage` to gather raw data from the internet. It then compiles a report of its findings.

3.  **V.E.T. (Vetting & Evaluation Triad): The Fact-Checker.**
    - **Role:** Reviews D.E.L.V.E.'s report for accuracy, bias, and completeness based on A.R.C.H.'s original instruction. It acts as a quality assurance layer.

- **Process:** The system works in a loop. A.R.C.H. gives an order, D.E.L.V.E. executes it, V.E.T. verifies the result, and the verified report is returned to A.R.C.H. A.R.C.H. then decides if the overall goal has been met or if another iteration with a new instruction is required.

- **Outcome:** The final, synthesized report is added as a new node to the L3 memory graph, permanently expanding F.R.E.D.'s knowledge base.

### 4.3. 3-Agent Research System - Deep Dive

Defined in `memory/arch_delve_research.py`, this is F.R.E.D.'s system for autonomous, in-depth investigation, simulating a hierarchical research team.

**The Agents and Their Mandates**

Each agent has a specific role defined by its system prompt in `config.py`.

*   **A.R.C.H. (The Director):**
    *   **Persona (`ARCH_SYSTEM_PROMPT`):** A strategic Research Director who plans and delegates but does not perform research.
    *   **Directives:** Must operate as a "blank slate," relying only on incoming reports. It must verify a user's core premises before investigating details. It delegates one simple, focused instruction at a time and is forbidden from suggesting *how* to research, only *what* to research. Its only tool is `complete_research`.

*   **D.E.L.V.E. (The Analyst):**
    *   **Persona (`DELVE_SYSTEM_PROMPT`):** A diligent Data Miner who believes A.R.C.H. is a human director.
    *   **Directives:** Also a "blank slate." Follows a strict **Search-Then-Read** protocol: `search_*` for sources, then `read_webpage` to extract content. It is strictly forbidden from summarizing or altering the content it fetches; its job is to provide raw data. It is also designed to be resilient to tool failures.

*   **V.E.T. (The Fact-Checker):**
    *   **Persona (`VET_SYSTEM_PROMPT`):** A skeptical, rigorous fact-checker.
    *   **Directives:** Another "blank slate" that analyzes only the raw data provided by D.E.L.V.E. It must compare and contrast sources, identify consensus and contradictions, and assess source credibility. Its output is a mandatory `VERIFIED REPORT` markdown block.

**The Iterative Research Flow**

The `conduct_iterative_research` function manages a stateful loop:

1.  **Instruction:** A.R.C.H. issues a single, focused instruction.
2.  **Execution:** D.E.L.V.E. receives the instruction and executes its Search-Then-Read loop, calling tools until it has the raw data to satisfy the request. It outputs this raw data as a JSON object.
3.  **Verification:** V.E.T. receives D.E.L.V.E.'s raw JSON output and A.R.C.H.'s original instruction. It performs its analysis and produces a structured `VERIFIED REPORT`.
4.  **Analysis:** The `VERIFIED REPORT` is returned to A.R.C.H.
5.  **Loop/Conclude:** A.R.C.H. analyzes the report and either issues a new, follow-up instruction (returning to Step 1) or, if the mission is complete, calls the `complete_research` tool to end the process.
6.  **Final Synthesis:** All `VERIFIED REPORT`s are then passed to **S.A.G.E. (Synthesis and Archive Generation Engine)**, which first creates a user-facing summary and then synthesizes that summary into a dense, permanent L3 memory node.

---
## 5. Real-time Sensory Processing & Data Flow

This section details how F.R.E.D. perceives the world and the overall flow of data through the system.

### 5.1. Overall Data Flow

1.  **Input:** User speaks to the Pi. Speech is transcribed locally and sent to the server.
2.  **Processing:** The Main Server receives the text. It gathers context from L2 memory (recent conversations) and L3 memory (long-term knowledge), and gets the current visual context from the `VisionService`.
3.  **Reasoning:** This rich, contextual prompt is sent to the primary LLM.
4.  **Action/Response:** The LLM either generates a direct text response or uses a tool.
    - If a text response, it's converted to audio and sent back to the Pi.
    - If a tool (like `addTaskToAgenda`), the relevant system (e.g., the `AgendaSystem`) is engaged.
5.  **Learning (Background):**
    - **Passively:** The L2 memory system constantly watches the conversation, creating summaries of past topics.
    - **Actively (During Sleep Cycle):** F.R.E.D. processes its research agenda using the A.R.C.H./D.E.L.V.E. system, consolidates L2 memories into L3, and expands its L3 knowledge graph by creating new connections.

This entire architecture creates a powerful feedback loop where F.R.E.D. can interact with the world, remember its interactions, and proactively learn more about topics of interest, making it a continuously evolving AI.

### 5.2. Sensory Processing - Deep Dive

F.R.E.D. perceives the world via the `webrtc_server.py` gateway, which handles low-latency data streams from the Pi client.

**WebRTC Data Handling**

*   **Data Channel (`on_datachannel`):** A bidirectional pipeline for structured data: `[HEARTBEAT]` messages for connection health, `[IMAGE_DATA:...]` messages containing captured images, and `TRANSCRIPTION:` messages with text from the Pi's local STT. Textual responses from F.R.E.D. are also sent back to the Pi over this channel.
*   **Audio Track (`on_track`):** When using server-side STT, a raw audio stream from the Pi's microphone is processed by `consume_audio_frames`, resampled, and fed directly into the server's `stt_service`.

**Vision Service**

The vision system (`vision_service.py`) is a "snapshot-based" model, not a continuous video analysis system.

*   **On-Demand Capture:** Every `VISION_PROCESSING_INTERVAL` (10) seconds, the service sends a `[CAPTURE_REQUEST]` to the Pi, which responds with a single, fresh image.
*   **Prompt Engineering (`_create_vision_prompt`):** The prompt sent to the Vision-Language Model (`qwen2.5vl:7b`) is dynamically assembled:
    1.  **Base Prompt (`VISION_USER_PROMPT`):** A general instruction to describe the scene.
    2.  **Face Recognition:** The image is first passed to the `persona_service`, and any recognized names are formatted into a "People visible: ..." string.
    3.  **Change Detection:** Critically, the description from the *previous* analysis is included in the prompt with an explicit instruction to "focus on what has changed." This makes the vision system stateful and prevents redundant descriptions of a static environment.
*   **Context Update:** The VLM's response becomes the `current_scene_description` and is made available to the main application.

---

## 6. Core Application & LLM Prompt Assembly - Deep Dive

The central logic in `app.py` dynamically assembles a rich, multi-faceted prompt before every single LLM call, which is the key to F.R.E.D.'s contextual awareness.

### 6.1. The Art of Prompt Assembly

The final prompt sent to the core LLM is constructed in a precise order inside the `/chat` endpoint's `event_stream` generator:

1.  **Base System Prompt (`FRED_SYSTEM_PROMPT`):** The foundation, defining F.R.E.D.'s core persona, operating style, and available tools.
2.  **L3 Knowledge Injection:** The user's message is used to query the L3 Knowledge Graph (`search_memory`). Relevant long-term facts are appended under a `(L3 Long-Term Memory)` heading.
3.  **L2 Context Injection:** The user's message is used to query the L2 Episodic Cache (`query_l2_context`). Relevant recent conversation summaries are appended under a `(L2 Recent Conversation Context)` heading.
4.  **Visual Context Injection:** The latest scene description from the `vision_service` is appended under a `(Current Visual Context (Pi Glasses))` heading.
5.  **System Status Injection:** Pending notifications from completed research (`get_pending_notifications`) and the current agenda status (`get_agenda_summary`) are appended under a `(SYSTEM & NOTIFICATION STATUS)` heading.
6.  **Final Assembly:** This assembled "super-prompt" is combined with the L1 conversational history and the user's latest message before being sent to the core LLM (`DEFAULT_MODEL`).

This process ensures that for every turn, F.R.E.D. has a complete, up-to-the-moment understanding of its identity, its knowledge, its recent past, its environment, and its own internal state. 