#!/usr/bin/env python3
"""
Korlan Language Runner
The complete pipeline: Lexer -> Parser -> Compiler -> Virtual Machine
Zero dependencies, zero ceremony, maximum simplicity.
"""

import sys
import os
from pathlib import Path

# Add core directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

from lexer import KorlanLexer, LexerError
from parser import KorlanParser, ParserError
from compiler import KorlanCompiler, CompilerError
from checker import SemanticChecker, KorlanError
from runtime.kvm import KorlanVM, KVMError

class KorlanRunner:
    """Main Korlan language runner"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
    
    def run_file(self, filepath: str) -> bool:
        """Run a Korlan file"""
        try:
            # Read source file
            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()
            
            return self.run_source(source, filepath)
            
        except FileNotFoundError:
            print(f"Error: File not found: {filepath}")
            return False
        except Exception as e:
            print(f"Error reading file: {e}")
            return False
    
    def run_source(self, source: str, filename: str = "<string>") -> bool:
        """Run Korlan source code"""
        try:
            print(f"=== Running Korlan: {filename} ===")
            
            # Phase 1: Lexing
            if self.debug:
                print("\n--- Phase 1: Lexing ---")
            
            lexer = KorlanLexer(source)
            tokens = lexer.tokenize()
            
            if self.debug:
                print(f"Generated {len(tokens)} tokens")
                lexer.print_tokens()
            
            # Phase 2: Parsing
            if self.debug:
                print("\n--- Phase 2: Parsing ---")
            
            parser = KorlanParser(tokens)
            ast = parser.parse()
            
            if self.debug:
                print("AST generated:")
                self.print_ast(ast, 0)
            
            # Phase 2.5: Semantic Analysis
            if self.debug:
                print("\n--- Phase 2.5: Semantic Analysis ---")
            
            checker = SemanticChecker()
            semantic_success = checker.check(ast)
            
            if not semantic_success:
                checker.print_errors()
                return False
            
            if self.debug:
                print("Semantic analysis passed")
                checker.print_symbols()
            
            # Phase 3: Compilation
            if self.debug:
                print("\n--- Phase 3: Compilation ---")
            
            compiler = KorlanCompiler()
            bytecode = compiler.compile(ast)
            
            if self.debug:
                print("Bytecode generated:")
                compiler.print_bytecode()
            
            # Phase 4: Execution
            if self.debug:
                print("\n--- Phase 4: Execution ---")
            
            vm = KorlanVM(debug=self.debug)
            result = vm.execute(bytecode, compiler.constants)
            
            # Print execution statistics
            stats = vm.get_stats()
            print(f"\n=== Execution Complete ===")
            print(f"Instructions executed: {stats['instructions_executed']}")
            print(f"Max stack depth: {stats['max_stack_depth']}")
            print(f"Result: {result}")
            
            return True
            
        except LexerError as e:
            print(f"Lexer Error at line {e.line}, column {e.column}: {e.message}")
            return False
        except ParserError as e:
            print(f"Parser Error at line {e.line}, column {e.column}: {e.message}")
            return False
        except KorlanError as e:
            print(f"Korlan {e.error_type} at line {e.line}, column {e.column}: {e.message}")
            return False
        except CompilerError as e:
            print(f"Compiler Error at line {e.node.line}, column {e.node.column}: {e.message}")
            return False
        except KVMError as e:
            print(f"Runtime Error at instruction {e.ip}: {e.message}")
            if e.instruction:
                print(f"  Instruction: {e.instruction.opcode.value} {e.instruction.operand or ''}")
            return False
        except Exception as e:
            print(f"Unexpected Error: {e}")
            return False
    
    def print_ast(self, node, indent: int = 0):
        """Print AST for debugging"""
        indent_str = "  " * indent
        print(f"{indent_str}{node.type.name}: {node.value}")
        for child in node.children:
            self.print_ast(child, indent + 1)

def print_usage():
    """Print usage information"""
    print("Korlan Language Runner")
    print("Usage:")
    print("  python korlan.py <file.kor>           - Run a Korlan file")
    print("  python korlan.py --debug <file.kor>   - Run with debug output")
    print("  python korlan.py --help              - Show this help")
    print("")
    print("Examples:")
    print("  python korlan.py examples/start.kor")
    print("  python korlan.py --debug examples/start.kor")

def main():
    """Main entry point"""
    args = sys.argv[1:]
    
    if not args or "--help" in args:
        print_usage()
        return 1
    
    debug = "--debug" in args
    
    # Remove debug flag from args
    if "--debug" in args:
        args.remove("--debug")
    
    if len(args) != 1:
        print("Error: Exactly one file argument required")
        print_usage()
        return 1
    
    filepath = args[0]
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return 1
    
    # Run the file
    runner = KorlanRunner(debug=debug)
    success = runner.run_file(filepath)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())