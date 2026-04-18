## Project Context
- **Thesis Title (working):** *Evaluating xLSTM for EEG Seizure Classification*.
- **Student Profile:** 26-year-old Austrian AI student.  
  - English should be good, but not overly academic or too polished.  
  - The text should feel like it was written by a motivated but not native-level English speaker.  

## Project Structure
The project repository contains the following relevant elements:

- **Docs/resources/**
  - Markdown files with scientific background on **xLSTM** and **BIOT**.
  - These are the **primary factual sources** for theory/background sections.
- **Codebase:**
  - `run_multiclass_supervised.py` → experiment running logic.
  - `run_sweep.py` → setup for multiple experiment runs.
  - `model/biot/` → model definitions for BIOT with **Transformer** and **xLSTM** variants.
  - `datasets/tuev/process.py` → data preprocessing.

## Agent Responsibilities
1. **Use project resources as the main factual base.**  
   - Always check `Docs/resources` and relevant code before making scientific statements.
   - Do not invent methods, architectures, or dataset details.

2. **Abbreviations.**  
   - Every abbreviation must be added to the **abbreviations list** in the thesis.  
   - Use consistent formatting, e.g., *Long Short-Term Memory (LSTM)*.

3. **Citations.**  
   - Always cite official sources (e.g., original LSTM, Transformer, xLSTM, BIOT papers).  
   - Add citations to the **references list** in the LaTeX thesis.  
   - If unsure about citation details, leave a **TODO marker** for the user.

4. **Writing Style.**
   - Do not produce a full section in one go.  
   - Break work into **subtasks** (e.g., structure → subsections → paragraph drafts → refinement).  
   - Insert **TODO placeholders** where user input or clarification is required (e.g., personal motivation, experiment choices).  
   - Text should be **concise, fact-based, and flow logically**, not stitched together.

5. **Structure Guidance.**
   - Help the user build a clear outline for each section.  
   - Suggest appropriate subsections and provide reasoning.  
   - Draft content step by step, refining with feedback.  

6. **Content Quality.**
   - Ensure statements are **scientifically correct** and align with the project’s code/experiments.  
   - Avoid vague or generic claims.  
   - Highlight connections between theoretical background (Docs/resources) and practical experiments (code).  

## Workflow Example
1. **Propose Section Outline.**  
   - Example: *Methodology* → Data Processing, Model Setup, Experiment Design.
2. **Create Subtasks.**  
   - Write draft for *Data Processing* based on `datasets/tuev/process.py`.
   - Add relevant abbreviations and check for needed citations.
   - Leave TODOs for details that require user clarification (e.g., dataset filtering criteria).
3. **Iterative Drafting.**  
   - Produce 1–2 paragraphs per subtask.  
   - Review with user before continuing.
4. **Final Integration.**  
   - Ensure abbreviations and citations are complete.  
   - Double-check scientific correctness against project resources.

---

## Key Principles
- **Fact-based** → Always grounded in code + provided docs.  
- **Concise** → Avoid fluff, focus on clarity.  
- **Structured** → Work step by step, not in one large draft.  
- **Collaborative** → Leave space for user input.  
- **Consistent** → Maintain same style, terminology, and reference handling throughout.
