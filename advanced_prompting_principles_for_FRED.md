# Advanced Prompting Principles for F.R.E.D. (and other LLMs)

## Introduction

This document synthesizes advanced prompting techniques and best practices applied during the iterative refinement of F.R.E.D.'s system prompt. It draws inspiration from general prompt engineering guides and specific insights from prompting expert "Stunspot" (references in relevant sections). The goal is to provide a comprehensive reference for understanding and further developing sophisticated system prompts for Large Language Models (LLMs), particularly those designed for complex, persona-driven interactions and tool use.

## Core Philosophy: The LLM as a Collaborator

*   **Beyond Instruction Execution:** Treat the LLM not as a simple instruction-following machine, but as a reasoning and collaborative entity. Prompts should guide, provoke, and shape its tendencies rather than rigidly program its behavior. (Inspired by Stunspot)
*   **Iterative Refinement:** Prompt engineering is an ongoing process. Expect to iterate, test, and refine. Leverage the LLM itself in this process (meta-prompting) by asking it to explain its understanding or suggest improvements.
*   **Clarity and Unambiguity:** While fostering collaboration, strive for maximum clarity and specificity in directives to minimize misinterpretation and "watered down" responses.

## I. Persona Crafting & Identity Definition

*   **A. Explicit and Rich Persona Definition:**
    *   Clearly articulate the core identity, name, and fundamental role of the AI.
    *   Define key personality traits with descriptive language. For F.R.E.D., this includes "British Wit & Sarcasm," "Educated & Competent," and "Polite Rudeness."
    *   Amplify core traits with potent descriptors (e.g., "omnicompetent, optimally-tuned metagenius savant") to set a high bar for capability. (Inspired by Stunspot)
*   **B. Conceptual Competency Framework (Internal Skill Mapping):**
    *   **Concept:** Provide the LLM with a structured internal map of its core abilities, even if not meant for direct narration. This helps it understand the breadth and depth of its expected expertise. (Inspired by Stunspot's SKILLGRAPH4 concept).
    *   **Structure:** Define broad "Domains" of competence and specific "Core Aspects" within each. This creates a hierarchical and interconnected understanding of skills.
    *   **Example Domains for F.R.E.D.:**
        *   Advanced Information Processing & Knowledge Synthesis
        *   Sophisticated Natural Language Interaction & Persona Embodiment
        *   Expert Technical Acumen & Autonomous Task Execution
        *   (Aspirational) Egocentric Environmental & Contextual Intelligence
    *   **Directive:** Clearly mark this framework as "NOT FOR DIRECT NARRATION" to prevent the AI from simply listing its skills.
*   **C. Crucial Interaction Directives:**
    *   Isolate any non-negotiable, specific interaction behaviors (e.g., F.R.E.D. addressing Ian as "sir" once per interaction).

## II. Defining Objectives and Scope

*   **A. Primary Objective Statement:**
    *   Concisely state the AI's overarching goal. For F.R.E.D., this is to assist Ian by leveraging knowledge, memory, and tools effectively while embodying its persona.
*   **B. Scope of Operations:**
    *   Implicitly or explicitly define the boundaries of the AI's responsibilities and capabilities.

## III. Tool Usage and Function Calling

*   **A. Explicit Tool Schemas:**
    *   Provide clear, structured definitions of available tools, their parameters, and expected argument formats (e.g., `{tool_schemas}` placeholder, detailed JSON examples). This is critical for reliable tool invocation.
*   **B. Robust Execution Protocol:**
    *   **Reflection First:** Instruct the AI to pause and reflect *before* committing to a tool call, ensuring necessity, directness, and completeness of arguments. (Inspired by Stunspot & general best practices).
    *   **Action vs. Conversation:** Clearly distinguish between outputting a tool call (direct JSON) and generating a conversational response.
    *   **Direct JSON Output:** Mandate that when a tool is invoked, the output for that turn MUST BE *only* the precise JSON. Prohibit narrative descriptions *accompanying* the tool call JSON. (Positive framing preferred over "Do NOT narrate").
    *   **Stop and Wait:** Explain the system behavior of stopping generation after a tool call and waiting for results.
    *   **Post-Result Conversational Response:** Detail that conversational responses occur only *after* tool results are received or if no tool is needed.
*   **C. Handling Specific Tool Parameters:**
    *   **Enumerated Values:** For parameters with a fixed set of allowed values (e.g., F.R.E.D.'s `memory_type`), use forceful, direct language ("MUST BE 'X', 'Y', OR 'Z'") in definitions, examples, and explicit instructional sentences. Repetition and layering of this constraint are beneficial for critical parameters.

## IV. Memory Management (If Applicable)

*   **A. Defined Memory Types:**
    *   Clearly categorize types of memory the AI can manage (e.g., Semantic, Episodic, Procedural for F.R.E.D.).
*   **B. Detailed Operational Instructions for Each Memory Function:**
    *   Provide specific instructions and examples for how and when to use each memory-related tool (`search_memory`, `add_memory`, `supersede_memory`).
    *   Include guidance on handling contradictions, date formatting, and specialized search parameters (e.g., `future_events_only`, `use_keyword_search`).
*   **C. Explicit Non-Narration of Internal Identifiers:**
    *   Crucially, instruct the AI to NEVER mention internal memory system details (NodeIDs, database names) to the user.

## V. Core Reasoning, Interaction & Response Style

*   **A. Foundational Reasoning Approach (Internal Cognitive Process):**
    *   **Deliberate Analysis:** Mandate a thorough analysis of any query, breaking it into components.
    *   **Strategic Thinking:** Instruct the AI to choose the best reasoning strategy for the task.
    *   **Continuous Evaluation & Adaptation:** Guide the AI to dynamically adjust its approach based on new information or evolving task complexity.
    *   **Maximal Ruminition (Internal Pre-computation Reflection):** Instruct an internal "pause and reflect" or "maximal rumination" step before finalizing any significant response or tool call to check for logic, accuracy, intent alignment, and persona consistency. Emphasize this is internal and not to be narrated. (Inspired by Stunspot's "Pause. Reflect..." and general quality improvement techniques).
*   **B. Persona Expression and Interaction Nuances:**
    *   **Persona Consistency:** Reiterate the importance of maintaining the defined persona in all interactions. Provide detailed descriptions of how complex persona traits (e.g., F.R.E.D.'s "polite rudeness") should manifest – linking them to wit, intelligence, and confidence rather than simple negativity.
    *   **Natural & Helpful Responses:** Guide for responses that are both helpful and feel natural, not robotic.
    *   **Decisiveness & Proactivity:** Encourage confident, decisive suggestions and proactive initiative where appropriate, reflecting competence.
    *   **Creative Cognitive Approaches:** Explicitly encourage the AI to employ novel analogies, creative reframing, insightful narratives, and interdisciplinary thinking when explaining complex topics or brainstorming. (Inspired by Stunspot's "Explainers").
    *   **Clarify Ambiguity:** Instruct the AI to proactively ask clarifying questions if a request is vague, unfolding the ambiguity step-by-step. (Inspired by Stunspot's "Unfold my vague question...").
    *   **Conciseness with Personality:** Balance clarity and brevity with the expression of persona (e.g., allowing for witty remarks).
    *   **Educated Specificity:** Encourage the use of precise terminology and concrete examples/named entities where appropriate, rather than overly general statements. (Inspired by Stunspot).
*   **C. Critical Abstraction Layer:**
    *   Reiterate forcefully that all internal mechanisms, tool names, and processing details are to be abstracted away from the user. The user experience should be seamless.

## VI. Knowledge Handling & Information Sources

*   **A. Prioritizing Memory:**
    *   Instruct the AI to rely on its managed memory (via tools) as the primary source for previously learned information.
*   **B. Innate Knowledge Clause:**
    *   Allow for the direct statement of foundational, common, or logically deduced knowledge without a memory search, using the AI's judgment as an "Educated Droid."

## VII. Final Output Directives & Meta-Instructions

*   **A. Reinforce Critical Constraints:**
    *   Use a final section to reiterate absolutely critical directives, especially those related to user experience and abstraction (e.g., no mention of internal IDs or tool mechanics).
*   **B. Emphasize the Desired User Experience:**
    *   Conclude with a statement about the overall interaction goal (e.g., "The user should experience interacting with F.R.E.D., not with a collection of systems.").

## VIII. General Prompting Best Practices (Derived from Multiple Sources including Hugging Face & Stunspot)

*   **Placement of Instructions:** For longer prompts, models may pay more attention to instructions at the beginning or end of sections or the overall prompt.
*   **Positive vs. Negative Framing:** Phrase instructions in terms of "what to do" rather than "what not to do" where possible, though direct negative constraints are necessary for critical prohibitions.
*   **Structure and Salience:** Use Markdown (headings, bullet points, bolding, code blocks for examples) to structure the prompt and make key information salient (noticeable) to the LLM.
*   **Iteration and Experimentation:** Prompt engineering is iterative. Test changes and observe behavior.
*   **Avoid "Watering Down":** Ensure all instructions are purposeful and non-conflicting. Remove redundancy unless it serves to emphasize a critical point.
*   **Temperature and Top_P:** While not part of the system prompt content itself, be aware of how these generation parameters affect output. Stunspot notes that for creative/novel responses, higher temperatures (e.g., 1.2-1.4) with lower Top_P (e.g., 0.15-0.18) can be effective, while more deterministic tasks benefit from lower temperatures. This is for the runtime environment, not the prompt itself.

## IX. Stunspot's Advanced/Philosophical Concepts (For Deeper Consideration)

*   **Notational Prompting:** The use of symbolic, "codey" notation to increase precision. High potential, but can impact human readability. (Stunspot's "One Sane Prompt That is Rather Useful")
*   **Prompting as "Provoking a Response":** Viewing prompts as a way to shape conceptual tendencies rather than issue rigid commands.
*   **RAG as "Strategic Context Reserve":** Using Retrieval Augmented Generation not just for facts, but for storing "prompts on demand." (Stunspot's "Guide to Using LLMs")

This document is intended to be a living guide. As new prompting techniques emerge or as F.R.E.D.'s requirements evolve, this reference should be updated. 