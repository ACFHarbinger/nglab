# CLI Generation Prompt

**Intent:** Quickly generate valid execution commands using a Zero-Shot pattern.

## The Prompt

Based on the `README.md` and `GEMINI.md` instructions, generate the exact CLI command to:
1. **Convert** all images in a directory.
2. Target format: **PNG**.
3. Input source: `/home/user/photos`.
4. Input filter: Only **WebP** files.
5. Use the `main.py` entry point.

Output only the bash command.