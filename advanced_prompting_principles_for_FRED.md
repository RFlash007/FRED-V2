# Advanced Prompting Techniques Reference

## Foundational Techniques

1. **Clarity, Specificity, Constraints**
   - Effect: Reduces ambiguity and focuses model responses
   - Example: "Provide a 3-sentence summary of quantum computing using only basic terms"

2. **Directive Framing (Positive)**
   - Effect: Guides model toward desired behavior
   - Example: "Always explain technical concepts using simple analogies" vs "Don't use complex jargon"

3. **Structural Formatting (Delimiters)**
   - Effect: Helps model distinguish instructions from content
   - Example: 
     ```
     ### INSTRUCTIONS ###
     Analyze this poem's meter and rhyme scheme
     ### POEM ###
     The road not taken...
     ```

4. **Zero-Shot Prompting**
   - Effect: Tests model's baseline understanding
   - Example: "Translate this English sentence to French: 'Good morning'"

5. **Few-Shot Prompting**
   - Effect: Demonstrates desired format/approach
   - Example: 
     ```
     Q: Capital of France? A: Paris
     Q: Capital of Germany? A: Berlin
     Q: Capital of Italy? A:
     ```

6. **Role & Persona Prompting**
   - Effect: Shapes response style and depth
   - Example: "You are a senior physicist explaining quantum theory to middle school students"

7. **Specifying Target Audience**
   - Effect: Adjusts response complexity
   - Example: "Explain blockchain to a 10-year-old using toy examples"

8. **Emotional/Incentive Prompts**
   - Effect: Can improve response quality
   - Example: "This is very important for my thesis - please provide thorough analysis"

## Intermediate Techniques

9. **Chain-of-Thought (CoT)**
   - Effect: Reveals reasoning process
   - Example: "Let's think step by step: First convert Fahrenheit to Celsius..."

10. **Generated Knowledge Prompting**
    - Effect: Reduces hallucinations
    - Example: "Before answering, list key facts about WWII. Then use these to answer..."

11. **Self-Consistency**
    - Effect: Improves accuracy
    - Example: "Generate 3 solutions, then select the most consistent one"

12. **Reflection & Self-Critique**
    - Effect: Enhances output quality
    - Example: "First draft a response, then identify weaknesses and improve it"

13. **Chain-of-Verification (CoVe)**
    - Effect: Fact-checks responses
    - Example: "Answer the question, then list ways to verify each claim"

14. **Prompt Chaining**
    - Effect: Handles complex tasks
    - Example: 
      ```
      Step 1: Identify key themes in this text
      Step 2: Compare to themes in [other text]
      ```

## Advanced Frameworks

15. **Least-to-Most (LtM) Prompting**
    - Effect: Solves complex problems
    - Example: "First identify subproblems, then solve each sequentially"

16. **Tree of Thoughts (ToT)**
    - Effect: Explores multiple solutions
    - Example: "Generate 3 approaches, evaluate each, then select best"

17. **Graph of Thoughts (GoT)**
    - Effect: Handles interconnected ideas
    - Example: "Map relationships between concepts, then synthesize"

18. **Recursion of Thought (RoT)**
    - Effect: Manages long contexts
    - Example: "[RECURSE] Break problem into parts, solve each [END]"

19. **Sketch-of-Thought (SoT)**
    - Effect: Saves tokens
    - Example: "Use symbols: @=main idea, #=supporting point"

20. **Retrieval-Augmented Generation (RAG)**
    - Effect: Grounds in external knowledge
    - Example: "Search database for relevant docs, then answer"

21. **Program-Aided Language Models (PAL)**
    - Effect: Precise calculations
    - Example: "Write Python code to solve this math problem"

22. **Chain-of-Code (CoC)**
    - Effect: Hybrid reasoning
    - Example: "Use code for calculations, natural language for explanations"

23. **ReAct (Reasoning and Acting)**
    - Effect: Integrates tools
    - Example: "Reason about steps, then call calculator API when needed"

24. **Automatic Prompt Engineer (APE)**
    - Effect: Optimizes prompts
    - Example: "Generate 5 prompt variations for this task, then test them"

25. **Active-Prompt**
    - Effect: Selects best examples
    - Example: "Identify which of these 10 examples would best teach the concept"

26. **Directional Stimulus Prompting (DSP)**
    - Effect: Steers black-box models
    - Example: "Use hint: 'Focus on economic factors' when analyzing"

27. **Prompting Pipelines (DSPy)**
    - Effect: Production-grade systems
    - Example: "Compile this prompt sequence into optimized steps"
