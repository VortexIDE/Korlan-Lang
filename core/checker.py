"""
Korlan Semantic Checker - The Guardian of Type Safety
Scans AST for 'mut' violations and type mismatches before VM execution.
"""

from typing import List, Dict, Set, Optional, Any
from dataclasses import dataclass
from enum import Enum

from parser import ASTNode, NodeType, ParserError

class Type(Enum):
    """Korlan types"""
    INT = "Int"
    FLOAT = "Float"
    STRING = "String"
    BOOL = "Bool"
    NULL = "Null"
    VOID = "Void"
    FUNCTION = "Function"
    UNKNOWN = "Unknown"

@dataclass
class Symbol:
    """Represents a variable or function symbol"""
    name: str
    type: Type
    is_mutable: bool
    node: ASTNode
    line: int
    column: int

class KorlanError(Exception):
    """Korlan-specific error with line and column information"""
    def __init__(self, message: str, line: int, column: int, error_type: str = "Error"):
        self.message = message
        self.line = line
        self.column = column
        self.error_type = error_type
        super().__init__(f"Korlan {error_type} at line {line}, column {column}: {message}")

class SemanticChecker:
    """Performs semantic analysis on the AST"""
    
    def __init__(self):
        self.symbols: Dict[str, Symbol] = {}  # Current scope symbols
        self.global_symbols: Dict[str, Symbol] = {}  # Global scope symbols
        self.function_stack: List[Dict[str, Symbol]] = []  # Function call stack
        self.current_function: Optional[str] = None
        self.errors: List[KorlanError] = []
        
        # Built-in function signatures
        self.builtins = {
            "print": Type.FUNCTION,
            "builtin_print": Type.FUNCTION,
            "builtin_read": Type.FUNCTION,
            "builtin_char_at": Type.FUNCTION,
            "builtin_length": Type.FUNCTION,
            "builtin_to_int_char": Type.FUNCTION,
            "builtin_to_string": Type.FUNCTION,
        }
    
    def add_error(self, message: str, node: ASTNode, error_type: str = "Error"):
        """Add a Korlan error with line/column information"""
        error = KorlanError(message, node.line, node.column, error_type)
        self.errors.append(error)
    
    def check(self, ast: ASTNode) -> bool:
        """Main semantic checking method"""
        try:
            self.check_node(ast)
            return len(self.errors) == 0
        except KorlanError:
            return False
    
    def check_node(self, node: ASTNode):
        """Check an AST node for semantic issues"""
        if node.type == NodeType.PROGRAM:
            self.check_program(node)
        elif node.type == NodeType.FUNCTION:
            self.check_function(node)
        elif node.type == NodeType.VARIABLE:
            self.check_variable(node)
        elif node.type == NodeType.EXPRESSION_STMT:
            self.check_node(node.children[0])
        elif node.type == NodeType.BINARY:
            self.check_binary(node)
        elif node.type == NodeType.CALL:
            self.check_function_call(node)
        elif node.type == NodeType.PIPELINE:
            self.check_pipeline(node)
        elif node.type == NodeType.LITERAL:
            self.check_literal(node)
        elif node.type == NodeType.IDENTIFIER:
            self.check_identifier(node)
        elif node.type == NodeType.BLOCK:
            self.check_block(node)
        elif node.type == NodeType.ASSIGN:
            self.check_assignment(node)
        # Add more node types as needed
    
    def check_program(self, node: ASTNode):
        """Check program node"""
        # First pass: collect all function declarations
        for child in node.children:
            if child.type == NodeType.FUNCTION:
                self.declare_function(child)
        
        # Second pass: check all nodes
        for child in node.children:
            self.check_node(child)
    
    def declare_function(self, node: ASTNode):
        """Declare a function symbol"""
        func_name = node.value
        if func_name in self.symbols:
            self.add_error(f"Function '{func_name}' already declared", node, "Error")
            return
        
        # Create function symbol
        symbol = Symbol(func_name, Type.FUNCTION, False, node, node.line, node.column)
        self.symbols[func_name] = symbol
        self.global_symbols[func_name] = symbol
    
    def check_function(self, node: ASTNode):
        """Check function declaration"""
        func_name = node.value
        self.current_function = func_name
        
        # Create new scope for function
        old_symbols = self.symbols
        self.symbols = {}
        self.function_stack.append(old_symbols)
        
        # Check function body
        if len(node.children) > 0:
            body = node.children[-1]
            if body.type == NodeType.BLOCK:
                self.check_node(body)
            else:
                self.check_node(body)
        
        # Restore scope
        self.symbols = self.function_stack.pop()
        self.current_function = None
    
    def check_variable(self, node: ASTNode):
        """Check variable declaration"""
        var_name = node.value
        is_mutable = var_name.startswith("mut ")
        
        if is_mutable:
            var_name = var_name[4:]  # Remove "mut " prefix
        
        # Check if variable already exists in current scope
        if var_name in self.symbols:
            self.add_error(f"Variable '{var_name}' already declared in current scope", node, "Error")
            return
        
        # Determine variable type from initializer if present
        var_type = Type.UNKNOWN
        if len(node.children) > 0:
            initializer = node.children[-1]
            var_type = self.infer_type(initializer)
            self.check_node(initializer)
        
        # Create symbol
        symbol = Symbol(var_name, var_type, is_mutable, node, node.line, node.column)
        self.symbols[var_name] = symbol
    
    def check_block(self, node: ASTNode):
        """Check block of statements"""
        for child in node.children:
            self.check_node(child)
    
    def check_binary(self, node: ASTNode):
        """Check binary expression for type consistency"""
        if len(node.children) != 2:
            self.add_error("Binary expression must have exactly 2 children", node, "Error")
            return
        
        left, right = node.children
        self.check_node(left)
        self.check_node(right)
        
        left_type = self.get_node_type(left)
        right_type = self.get_node_type(right)
        
        # Type checking based on operator
        if node.value in ["+", "-", "*", "/"]:
            if not self.is_numeric_type(left_type) or not self.is_numeric_type(right_type):
                self.add_error(f"Cannot perform arithmetic operation on {left_type.value} and {right_type.value}", node, "Type Error")
        elif node.value in ["==", "!="]:
            # Any types can be compared for equality
            pass
        elif node.value in ["<", ">", "<=", ">="]:
            if not self.is_comparable_type(left_type) or not self.is_comparable_type(right_type):
                self.add_error(f"Cannot compare {left_type.value} and {right_type.value}", node, "Type Error")
        elif node.value in ["&&", "||"]:
            if left_type != Type.BOOL or right_type != Type.BOOL:
                self.add_error(f"Logical operators require boolean operands, got {left_type.value} and {right_type.value}", node, "Type Error")
    
    def check_function_call(self, node: ASTNode):
        """Check function call"""
        func_name = node.value
        
        # Check if function exists
        if func_name not in self.symbols and func_name not in self.builtins:
            self.add_error(f"Undefined function: {func_name}", node, "Error")
            return
        
        # Check arguments
        for arg in node.children:
            self.check_node(arg)
    
    def check_pipeline(self, node: ASTNode):
        """Check pipeline expression"""
        if len(node.children) < 2:
            self.add_error("Pipeline must have at least 2 children", node, "Error")
            return
        
        # Check initial value
        self.check_node(node.children[0])
        
        # Check each transformation
        for i in range(1, len(node.children)):
            transform = node.children[i]
            if transform.type != NodeType.CALL:
                self.add_error("Pipeline transformations must be function calls", transform, "Error")
                return
            self.check_node(transform)
    
    def check_literal(self, node: ASTNode):
        """Check literal"""
        # Literals are always valid
        pass
    
    def check_identifier(self, node: ASTNode):
        """Check identifier access"""
        var_name = node.value
        
        # Check if variable exists
        if var_name not in self.symbols:
            self.add_error(f"Undefined variable: {var_name}", node, "Error")
    
    def check_assignment(self, node: ASTNode):
        """Check assignment for mut violations"""
        if len(node.children) != 2:
            self.add_error("Assignment must have exactly 2 children", node, "Error")
            return
        
        target, value = node.children
        
        if target.type != NodeType.IDENTIFIER:
            self.add_error("Assignment target must be an identifier", target, "Error")
            return
        
        var_name = target.value
        
        # Check if variable exists
        if var_name not in self.symbols:
            self.add_error(f"Undefined variable: {var_name}", target, "Error")
            return
        
        # Check if variable is mutable (THIS IS THE KEY CHECK)
        symbol = self.symbols[var_name]
        if not symbol.is_mutable:
            self.add_error(f"Cannot assign to immutable variable '{var_name}'. Use 'mut' to make it mutable.", target, "Mutability Error")
            return
        
        # Check value
        self.check_node(value)
        
        # Type checking
        var_type = symbol.type
        value_type = self.get_node_type(value)
        
        if var_type != Type.UNKNOWN and value_type != Type.UNKNOWN and var_type != value_type:
            self.add_error(f"Cannot assign {value_type.value} to variable of type {var_type.value}", node, "Type Error")
    
    def infer_type(self, node: ASTNode) -> Type:
        """Infer the type of an AST node"""
        if node.type == NodeType.LITERAL:
            if node.value == "number":
                # Check if it's float or int
                if "." in node.children[0].value:
                    return Type.FLOAT
                else:
                    return Type.INT
            elif node.value == "string":
                return Type.STRING
            elif node.value == "boolean":
                return Type.BOOL
            elif node.value == "null":
                return Type.NULL
        elif node.type == NodeType.IDENTIFIER:
            symbol = self.symbols.get(node.value)
            return symbol.type if symbol else Type.UNKNOWN
        elif node.type == NodeType.BINARY:
            if node.value in ["+", "-", "*", "/"]:
                left_type = self.infer_type(node.children[0])
                right_type = self.infer_type(node.children[1])
                # If either is float, result is float
                if left_type == Type.FLOAT or right_type == Type.FLOAT:
                    return Type.FLOAT
                else:
                    return Type.INT
            elif node.value in ["==", "!="]:
                return Type.BOOL
            elif node.value in ["<", ">", "<=", ">="]:
                return Type.BOOL
            elif node.value in ["&&", "||"]:
                return Type.BOOL
        elif node.type == NodeType.CALL:
            # For now, assume function calls return unknown type
            return Type.UNKNOWN
        
        return Type.UNKNOWN
    
    def get_node_type(self, node: ASTNode) -> Type:
        """Get the type of a node"""
        return self.infer_type(node)
    
    def is_numeric_type(self, type: Type) -> bool:
        """Check if type is numeric"""
        return type in [Type.INT, Type.FLOAT]
    
    def is_comparable_type(self, type: Type) -> bool:
        """Check if type is comparable"""
        return type in [Type.INT, Type.FLOAT, Type.STRING]
    
    def get_errors(self) -> List[KorlanError]:
        """Get all semantic errors"""
        return self.errors
    
    def print_errors(self):
        """Print all errors in a user-friendly format"""
        if not self.errors:
            print("✅ No semantic errors found")
            return
        
        print(f"❌ Found {len(self.errors)} semantic error(s):")
        for error in self.errors:
            print(f"  {error.error_type} at line {error.line}, column {error.column}: {error.message}")
    
    def print_symbols(self):
        """Debug method to print symbols"""
        print("=== Symbols ===")
        for name, symbol in self.symbols.items():
            mutable_str = "mut" if symbol.is_mutable else "immutable"
            print(f"{name}: {symbol.type.value} ({mutable_str})")

def main():
    """Test the semantic checker"""
    from lexer import KorlanLexer
    from parser import KorlanParser
    
    # Test cases
    test_cases = [
        # Valid code
        '''
fun main() {
    mut x = 5
    x = 10
    print(x)
}
''',
        # Invalid code - assigning to immutable variable
        '''
fun main() {
    x = 5
    x = 10
}
''',
        # Invalid code - type mismatch
        '''
fun main() {
    mut x = 5
    x = "hello"
}
''',
        # Invalid code - redeclaring variable
        '''
fun main() {
    mut x = 5
    mut x = 10
}
''',
    ]
    
    for i, code in enumerate(test_cases):
        print(f"\n=== Test Case {i+1} ===")
        print("Code:")
        print(code)
        
        # Parse
        lexer = KorlanLexer(code)
        tokens = lexer.tokenize()
        
        parser = KorlanParser(tokens)
        ast = parser.parse()
        
        # Check semantics
        checker = SemanticChecker()
        success = checker.check(ast)
        
        checker.print_errors()
        
        if success:
            print("✅ Semantic analysis passed")
        else:
            print("❌ Semantic analysis failed")
        
        checker.print_symbols()

if __name__ == "__main__":
    main()
