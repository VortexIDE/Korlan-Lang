#!/usr/bin/env python3
"""
Test script for self-hosting capability
Attempts to compile and run the Korlan lexer using the Python KVM
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

def test_self_hosting():
    """Test self-hosting by running lexer.kor on itself"""
    
    print("=== Korlan Self-Hosting Test ===")
    print("Attempting to run lexer.kor using Python KVM...")
    
    # Read the lexer.kor file
    lexer_file_path = os.path.join(os.path.dirname(__file__), 'core', 'self_hosted', 'lexer.kor')
    
    try:
        with open(lexer_file_path, 'r', encoding='utf-8') as f:
            korlan_code = f.read()
        
        print(f"Loaded lexer.kor ({len(korlan_code)} characters)")
        
        # Step 1: Lex the Korlan code
        print("\n1. Lexing lexer.kor...")
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
        errors = checker.check(ast)
        
        if errors:
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
        print(f"Error: Could not find lexer.kor at {lexer_file_path}")
        return False
    except Exception as e:
        print(f"Error during self-hosting test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_korlan():
    """Test with a simple Korlan program first"""
    
    print("\n=== Simple Korlan Test ===")
    
    simple_code = '''
fun main() {
    print("Hello from Korlan!")
}
'''
    
    try:
        # Lex
        lexer = KorlanLexer(simple_code)
        tokens = lexer.tokenize()
        print(f"Lex: {len(tokens)} tokens")
        
        # Parse
        parser = KorlanParser(tokens)
        ast = parser.parse()
        print("Parse: OK")
        
        # Check
        checker = SemanticChecker()
        success = checker.check(ast)
        
        if not success:
            errors = checker.get_errors()
            print(f"Semantic errors: {len(errors)}")
            for error in errors:
                print(f"  - {error.message}")
        else:
            print("Semantic: OK")
        
        # Compile
        compiler = KorlanCompiler()
        bytecode = compiler.compile(ast)
        print(f"Compile: {len(bytecode)} instructions")
        
        # Execute
        vm = KorlanVM(debug=False)
        result = vm.execute(bytecode, compiler.constants)
        print(f"Execute: {result}")
        
        return True
        
    except Exception as e:
        print(f"Simple test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Korlan Self-Hosting Test Suite")
    print("=" * 50)
    
    # First test with simple code
    simple_success = test_simple_korlan()
    
    if simple_success:
        print("\n" + "=" * 50)
        # Then attempt self-hosting
        self_host_success = test_self_hosting()
        
        if self_host_success:
            print("\n🎉 Self-hosting test PASSED!")
            print("The Korlan lexer can run on the Python KVM!")
        else:
            print("\n❌ Self-hosting test FAILED")
            print("The lexer.kor needs to be simplified or the compiler enhanced")
    else:
        print("\n❌ Basic functionality test FAILED")
        print("Cannot proceed with self-hosting test")
