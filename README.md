# Korlan Programming Language

A modern, compiled systems programming language designed to maximize both developer productivity and runtime performance.

## Features

- **Simplicity-first syntax** – No semicolons, no mandatory return statements, no visual clutter
- **Arrow semantics (`->`)** – Clear visual flow for functions, types, pipelines, and transformations
- **Garbage collection by default** – Automatic memory management with per-function/class opt-out
- **Type inference** – Optional type annotations; compiler infers when obvious
- **Null safety** – Explicit nullable types prevent null pointer errors
- **Immutable by default** – Variables immutable unless marked `mut`
- **Pattern matching** – Expressive control flow with `match` expressions
- **Safe concurrency** – Lightweight goroutine-style tasks with channel-based communication

## Project Structure

```
Korlan-lang/
├── src/
│   ├── main.rs              # CLI entry point
│   └── compiler/
│       ├── mod.rs            # Compiler module
│       ├── lexer/
│       │   ├── mod.rs        # Tokenizer implementation
│       │   └── token.rs      # Token definitions
│       ├── parser/
│       │   ├── mod.rs        # Recursive descent parser
│       │   └── ast.rs        # Abstract Syntax Tree
│       └── codegen.rs        # LLVM code generation
├── examples/
│   ├── hello.kor            # Hello World example
│   └── math.kor             # Math function example
└── Cargo.toml               # Rust dependencies
```

## Building and Running

### Prerequisites

- Rust (latest stable)
- LLVM 16+
- GCC (for linking)

### Commands

```bash
# Build a .kor file to native executable
cargo run -- build examples/hello.kor

# Compile and run immediately
cargo run -- run examples/hello.kor

# Generate LLVM IR for debugging
cargo run -- ir examples/hello.kor
```

## Language Examples

### Hello World

```korlan
fun main() {
    print("Hello, Korlan!")
}
```

### Functions with Arrows

```korlan
// Single-expression function
fun add(a: Int, b: Int) -> Int => a + b

// Multi-parameter function
fun greet(name: String) -> String {
    "Hello, {name}!"
}

// Arrow pipeline
fun process(input: String) -> Result {
    input
        -> trim()
        -> toLowerCase()
        -> validate()
}
```

### Variables & Type Inference

```korlan
// Immutable by default
name = "Alice"
age = 30

// Mutable when needed
mut counter = 0
counter = counter + 1

// Explicit types (optional)
greeting: String = "Hello"
total: Int = 100
```

### Control Flow

```korlan
// If expressions (no parentheses needed)
fun checkAge(age: Int) -> String {
    if age >= 18 {
        "Adult"
    } else {
        "Minor"
    }
}

// Pattern matching
fun describe(value: Any) -> String {
    match value {
        is Int -> "It's a number: {value}"
        is String -> "It's text: {value}"
        null -> "It's nothing"
        else -> "Unknown type"
    }
}
```

### Collections & Pipelines

```korlan
numbers = [1, 2, 3, 4, 5]
result = numbers
    -> filter(x => x > 2)
    -> map(x => x * 2)
    -> sum()
```

### Concurrency

```korlan
fun fetchData(url: String) {
    spawn {
        data = httpGet(url)
        print("Fetched: {data}")
    }
}
```

## Compiler Architecture

The Korlan compiler follows a traditional three-phase architecture:

1. **Lexing** - Tokenizes source code into meaningful tokens
2. **Parsing** - Builds an Abstract Syntax Tree (AST) using recursive descent
3. **Code Generation** - Translates AST to LLVM IR for native compilation

### Key Components

- **Lexer**: Handles tokenization with support for arrow operators (`->`, `=>`) and null safety (`?`)
- **Parser**: Implements recursive descent parsing with proper operator precedence
- **Codegen**: Uses LLVM via inkwell for machine code generation

## Development Status

### ✅ Completed

- Token definitions and lexer implementation
- AST structure and recursive descent parser
- Basic LLVM code generation framework
- CLI tool for building and running .kor files

### 🚧 In Progress

- Complete code generation for all expression types
- Memory management (GC vs @nogc modes)
- Standard library implementation
- Error handling and diagnostics

### 📋 Planned

- Class and interface support
- Advanced pattern matching
- Module system
- IDE integration (LSP server)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests for new functionality
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
