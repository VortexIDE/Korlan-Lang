"""
Korlan Semantic Analyzer - The Guardian of Type Safety
Performs semantic analysis on the AST before bytecode generation.
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

class SemanticError(Exception):
    def __init__(self, message: str, node: ASTNode):
        self.message = message
        self.node = node
        super().__init__(f"Semantic Error at line {node.line}, column {node.column}: {message}")

class SemanticAnalyzer:
    """Performs semantic analysis on the AST"""
    
    def __init__(self):
        self.symbols: Dict[str, Symbol] = {}  # Current scope symbols
        self.global_symbols: Dict[str, Symbol] = {}  # Global scope symbols
        self.function_stack: List[Dict[str, Symbol]] = []  # Function call stack
        self.current_function: Optional[str] = None
        self.errors: List[str] = []
        
        # Built-in function signatures
        self.builtins = {
            "print": Type.FUNCTION,  # print takes any args, returns null
            "builtin_print": Type.FUNCTION,
            "builtin_read": Type.FUNCTION,
            "builtin_char_at": Type.FUNCTION,
            "builtin_length": Type.FUNCTION,
            "builtin_to_int_char": Type.FUNCTION,
            "builtin_to_string": Type.FUNCTION,
        }
    
    def error(self, message: str, node: ASTNode):
        """Add semantic error"""
        error_msg = f"Semantic Error at line {node.line}, column {node.column}: {message}"
        self.errors.append(error_msg)
    
    def analyze(self, ast: ASTNode) -> bool:
        """Main analysis method"""
        try:
            self.analyze_node(ast)
            return len(self.errors) == 0
        except SemanticError:
            return False
    
    def analyze_node(self, node: ASTNode):
        """Analyze an AST node"""
        if node.type == NodeType.PROGRAM:
            self.analyze_program(node)
        elif node.type == NodeType.FUNCTION:
            self.analyze_function(node)
        elif node.type == NodeType.VARIABLE:
            self.analyze_variable(node)
        elif node.type == NodeType.EXPRESSION_STMT:
            self.analyze_node(node.children[0])
        elif node.type == NodeType.BINARY:
            self.analyze_binary(node)
        elif node.type == NodeType.CALL:
            self.analyze_function_call(node)
        elif node.type == NodeType.PIPELINE:
            self.analyze_pipeline(node)
        elif node.type == NodeType.LITERAL:
            self.analyze_literal(node)
        elif node.type == NodeType.IDENTIFIER:
            self.analyze_identifier(node)
        elif node.type == NodeType.BLOCK:
            self.analyze_block(node)
        elif node.type == NodeType.ASSIGN:
            self.analyze_assignment(node)
        # Add more node types as needed
    
    def analyze_program(self, node: ASTNode):
        """Analyze program node"""
        # First pass: collect all function declarations
        for child in node.children:
            if child.type == NodeType.FUNCTION:
                self.declare_function(child)
        
        # Second pass: analyze all nodes
        for child in node.children:
            self.analyze_node(child)
    
    def declare_function(self, node: ASTNode):
        """Declare a function symbol"""
        func_name = node.value
        if func_name in self.symbols:
            self.error(f"Function '{func_name}' already declared", node)
            return
        
        # Create function symbol
        symbol = Symbol(func_name, Type.FUNCTION, False, node)
        self.symbols[func_name] = symbol
        self.global_symbols[func_name] = symbol
    
    def analyze_function(self, node: ASTNode):
        """Analyze function declaration"""
        func_name = node.value
        self.current_function = func_name
        
        # Create new scope for function
        old_symbols = self.symbols
        self.symbols = {}
        self.function_stack.append(old_symbols)
        
        # Analyze function body
        if len(node.children) > 0:
            body = node.children[-1]
            if body.type == NodeType.BLOCK:
                self.analyze_node(body)
            else:
                self.analyze_node(body)
        
        # Restore scope
        self.symbols = self.function_stack.pop()
        self.current_function = None
    
    def analyze_variable(self, node: ASTNode):
        """Analyze variable declaration"""
        var_name = node.value
        is_mutable = var_name.startswith("mut ")
        
        if is_mutable:
            var_name = var_name[4:]  # Remove "mut " prefix
        
        # Check if variable already exists in current scope
        if var_name in self.symbols:
            self.error(f"Variable '{var_name}' already declared in current scope", node)
            return
        
        # Determine variable type from initializer if present
        var_type = Type.UNKNOWN
        if len(node.children) > 0:
            initializer = node.children[-1]
            var_type = self.infer_type(initializer)
            self.analyze_node(initializer)
        
        # Create symbol
        symbol = Symbol(var_name, var_type, is_mutable, node)
        self.symbols[var_name] = symbol
    
    def analyze_block(self, node: ASTNode):
        """Analyze block of statements"""
        for child in node.children:
            self.analyze_node(child)
    
    def analyze_binary(self, node: ASTNode):
        """Analyze binary expression"""
        if len(node.children) != 2:
            self.error("Binary expression must have exactly 2 children", node)
            return
        
        left, right = node.children
        self.analyze_node(left)
        self.analyze_node(right)
        
        left_type = self.get_node_type(left)
        right_type = self.get_node_type(right)
        
        # Type checking based on operator
        if node.value in ["+", "-", "*", "/"]:
            if not self.is_numeric_type(left_type) or not self.is_numeric_type(right_type):
                self.error(f"Cannot perform arithmetic operation on {left_type.value} and {right_type.value}", node)
        elif node.value in ["==", "!="]:
            # Any types can be compared for equality
            pass
        elif node.value in ["<", ">", "<=", ">="]:
            if not self.is_comparable_type(left_type) or not self.is_comparable_type(right_type):
                self.error(f"Cannot compare {left_type.value} and {right_type.value}", node)
        elif node.value in ["&&", "||"]:
            if left_type != Type.BOOL or right_type != Type.BOOL:
                self.error(f"Logical operators require boolean operands, got {left_type.value} and {right_type.value}", node)
    
    def analyze_function_call(self, node: ASTNode):
        """Analyze function call"""
        func_name = node.value
        
        # Check if function exists
        if func_name not in self.symbols and func_name not in self.builtins:
            self.error(f"Undefined function: {func_name}", node)
            return
        
        # Analyze arguments
        for arg in node.children:
            self.analyze_node(arg)
    
    def analyze_pipeline(self, node: ASTNode):
        """Analyze pipeline expression"""
        if len(node.children) < 2:
            self.error("Pipeline must have at least 2 children", node)
            return
        
        # Analyze initial value
        self.analyze_node(node.children[0])
        
        # Analyze each transformation
        for i in range(1, len(node.children)):
            transform = node.children[i]
            if transform.type != NodeType.CALL:
                self.error("Pipeline transformations must be function calls", transform)
                return
            self.analyze_node(transform)
    
    def analyze_literal(self, node: ASTNode):
        """Analyze literal"""
        # Literals are always valid
        pass
    
    def analyze_identifier(self, node: ASTNode):
        """Analyze identifier access"""
        var_name = node.value
        
        # Check if variable exists
        if var_name not in self.symbols:
            self.error(f"Undefined variable: {var_name}", node)
    
    def analyze_assignment(self, node: ASTNode):
        """Analyze assignment"""
        if len(node.children) != 2:
            self.error("Assignment must have exactly 2 children", node)
            return
        
        target, value = node.children
        
        if target.type != NodeType.IDENTIFIER:
            self.error("Assignment target must be an identifier", target)
            return
        
        var_name = target.value
        
        # Check if variable exists
        if var_name not in self.symbols:
            self.error(f"Undefined variable: {var_name}", target)
            return
        
        # Check if variable is mutable
        symbol = self.symbols[var_name]
        if not symbol.is_mutable:
            self.error(f"Cannot assign to immutable variable '{var_name}'. Use 'mut' to make it mutable.", target)
            return
        
        # Analyze value
        self.analyze_node(value)
        
        # Type checking
        var_type = symbol.type
        value_type = self.get_node_type(value)
        
        if var_type != Type.UNKNOWN and value_type != Type.UNKNOWN and var_type != value_type:
            self.error(f"Cannot assign {value_type.value} to variable of type {var_type.value}", node)
    
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
    
    def get_errors(self) -> List[str]:
        """Get all semantic errors"""
        return self.errors
    
    def print_symbols(self):
        """Debug method to print symbols"""
        print("=== Symbols ===")
        for name, symbol in self.symbols.items():
            mutable_str = "mut" if symbol.is_mutable else "immutable"
            print(f"{name}: {symbol.type.value} ({mutable_str})")

def main():
    """Test the semantic analyzer"""
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
        
        # Analyze
        analyzer = SemanticAnalyzer()
        success = analyzer.analyze(ast)
        
        if success:
            print("✅ Semantic analysis passed")
        else:
            print("❌ Semantic analysis failed:")
            for error in analyzer.get_errors():
                print(f"  {error}")
        
        analyzer.print_symbols()

if __name__ == "__main__":
    main()
