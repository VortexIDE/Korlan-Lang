---
trigger: always_on
---

# Korlan Language Development Rules

## Language Philosophy
- Follow "Simplicity-first" syntax: No semicolons, minimal punctuation.
- Use Arrow-driven design (`->`) for function signatures and type annotations.
- Prioritize native performance (C++/Rust level) while keeping default garbage collection.

## Project Structure
- Use a modular approach for the compiler: /lexer, /parser, /codegen, and /stdlib.
- Maintain a `specification.md` based on Korlan.txt as the "source of truth" for syntax.

## Implementation Guidelines
- Default memory management is GC, but support @nogc annotations for performance-critical blocks.
- Ensure strict null safety by default.
- Use Go-inspired lightweight constructs for concurrency.

## Syntax Standards
- No semicolons at the end of lines.
- Use `->` for function signatures and type annotations.
- Use `=>` for single-line function bodies.
- Use `mut` for mutable variables; default is immutable.
- Reference the 'Korlan Programming Language Design' document for all grammar decisions.

## Project Structure
- /compiler: The core logic (Lexer, Parser, Emitter).
- /stdlib: Standard library files.
- /examples: Sample .kor files.
- /docs: Documentation and specifications.

## Behavior
- When generating code for the compiler, prioritize native performance.
- Always check for null-safety as per the Korlan spec.

## Some general guidelines
- Maintain consistency with the Korlan language design principles.
- Keep explanations clear and aligned with the official specification.
- Don't do anything more than what users ask for.