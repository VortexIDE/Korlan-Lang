#!/usr/bin/env python3
"""
Test simple self-hosting with a minimal lexer that can actually run
"""

import sys
import os

# Add the core directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

from lexer import KorlanLexer
from parser import KorlanParser
from checker import SemanticChecker
from compiler import KorlanCompiler
from runtime.kvm import KorlanVM

def test_simple_self_hosting():
    """Test self-hosting with simple_lexer.kor"""
    
    print("=== Simple Self-Hosting Test ===")
    print("Attempting to run simple_lexer.kor using Python KVM...")
    
    # Read the simple_lexer.kor file
    lexer_file_path = os.path.join(os.path.dirname(__file__), 'core', 'self_hosted', 'simple_lexer.kor')
    
    try:
        with open(lexer_file_path, 'r', encoding='utf-8') as f:
            korlan_code = f.read()
        
        print(f"Loaded simple_lexer.kor ({len(korlan_code)} characters)")
        print("Code preview:")
        print(korlan_code[:200] + "...")
        
        # Step 1: Lex the Korlan code
        print("\n1. Lexing simple_lexer.kor...")
        lexer = KorlanLexer(korlan_code)
        tokens = lexer.tokenize()
        print(f"   Generated {len(tokens)} tokens")
        
        # Step 2: Parse the AST
        print("\n2. Parsing AST...")
        parser = KorlanParser(tokens)
        ast = parser.parse()
        print(f"   Parsed AST successfully")
        
        # Step 3: Semantic checking
        print("\n3. Semantic checking...")
        checker = SemanticChecker()
        success = checker.check(ast)
        
        if not success:
            errors = checker.get_errors()
            print(f"   Found {len(errors)} semantic errors:")
            for error in errors:
                print(f"     - {error.message}")
            print("   Cannot proceed with compilation due to semantic errors")
            return False
        else:
            print("   Semantic analysis passed")
        
        # Step 4: Compile to bytecode
        print("\n4. Compiling to bytecode...")
        compiler = KorlanCompiler()
        bytecode = compiler.compile(ast)
        print(f"   Generated {len(bytecode)} bytecode instructions")
        
        # Step 5: Execute in KVM
        print("\n5. Executing in KVM...")
        vm = KorlanVM(debug=False)
        
        try:
            result = vm.execute(bytecode, compiler.constants)
            print(f"   Execution completed successfully")
            print(f"   Result: {result}")
            print(f"   Stats: {vm.get_stats()}")
            return True
            
        except Exception as e:
            print(f"   Execution failed: {e}")
            return False
            
    except FileNotFoundError:
        print(f"Error: Could not find simple_lexer.kor at {lexer_file_path}")
        return False
    except Exception as e:
        print(f"Error during self-hosting test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Korlan Simple Self-Hosting Test")
    print("=" * 50)
    
    success = test_simple_self_hosting()
    
    if success:
        print("\n🎉 Simple self-hosting test PASSED!")
        print("A basic Korlan lexer can run on the Python KVM!")
        print("This demonstrates the foundation for Stage 1 self-hosting.")
    else:
        print("\n❌ Simple self-hosting test FAILED")
        print("Need to enhance the compiler or simplify the lexer further")
