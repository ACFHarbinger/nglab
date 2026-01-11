# Debugging Prompt

**Intent:** Resolve runtime issues using ReAct logic and existing project utilities in Image-Toolkit.

## The Prompt

I am encountering a specific error: `[INSERT ERROR HERE]`.

Context:
- **Component**: [Gui / Backend / Rust Core]
- **Operation**: [e.g., Slideshow, Conversion, Scanning]

Task:
Analyze the provided code snippets (or suggest which files to read). Identify potential causes such as:
1. **Thread Blocking**: Is the GUI main thread waiting on I/O?
2. **Missing Dependencies**: Are system tools like `ffmpeg` or `qdbus` missing?
3. **Rust Panic**: Did the `base` library panic safely?
4. **IPC Failure**: Did the Electron frontend lose connection to the backend?

Propose a fix that adheres to the "No Blocking Main Thread" rule.