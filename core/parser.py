"""
Korlan Parser - The Brain of the Language
Converts tokens into an Abstract Syntax Tree (AST) with zero-ceremony syntax.
"""

from typing import List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import sys

from lexer import Token, TokenType, LexerError

class NodeType(Enum):
    # Statements
    PROGRAM = "PROGRAM"
    FUNCTION = "FUNCTION"
    VARIABLE = "VARIABLE"
    IF = "IF"
    MATCH = "MATCH"
    WHILE = "WHILE"
    FOR = "FOR"
    RETURN = "RETURN"
    BREAK = "BREAK"
    CONTINUE = "CONTINUE"
    SPAWN = "SPAWN"
    CLASS = "CLASS"
    EXPRESSION_STMT = "EXPRESSION_STMT"
    
    # Expressions
    BINARY = "BINARY"
    UNARY = "UNARY"
    CALL = "CALL"
    PIPELINE = "PIPELINE"
    LITERAL = "LITERAL"
    IDENTIFIER = "IDENTIFIER"
    ASSIGN = "ASSIGN"
    IF_EXPR = "IF_EXPR"
    MATCH_EXPR = "MATCH_EXPR"
    BLOCK = "BLOCK"
    ARRAY = "ARRAY"
    MAP = "MAP"
    PROPERTY_ACCESS = "PROPERTY_ACCESS"
    METHOD_ACCESS = "METHOD_ACCESS"
    INDEX_ACCESS = "INDEX_ACCESS"
    NULL_SAFE_ACCESS = "NULL_SAFE_ACCESS"
    ELVIS = "ELVIS"
    TYPE_CAST = "TYPE_CAST"
    LAMBDA = "LAMBDA"

@dataclass
class ASTNode:
    type: NodeType
    value: Any = None
    children: List['ASTNode'] = None
    line: int = 0
    column: int = 0
    
    def __post_init__(self):
        if self.children is None:
            self.children = []

class ParserError(Exception):
    def __init__(self, message: str, token: Token):
        self.message = message
        self.token = token
        super().__init__(f"Parser Error at line {token.line}, column {token.column}: {message}")

class KorlanParser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.position = 0
        self.current_token = self.tokens[0] if tokens else None
    
    def error(self, message: str) -> ParserError:
        token = self.current_token or Token(TokenType.EOF, "", 0, 0)
        raise ParserError(message, token)
    
    def advance(self) -> Optional[Token]:
        if self.position < len(self.tokens):
            self.position += 1
            if self.position < len(self.tokens):
                self.current_token = self.tokens[self.position]
            else:
                self.current_token = None
        return self.current_token
    
    def peek(self, offset: int = 0) -> Optional[Token]:
        peek_pos = self.position + offset
        if peek_pos < len(self.tokens):
            return self.tokens[peek_pos]
        return None
    
    def expect(self, token_type: TokenType) -> Token:
        if self.current_token and self.current_token.type == token_type:
            token = self.current_token
            self.advance()
            return token
        else:
            self.error(f"Expected {token_type.name}, got {self.current_token.type.name if self.current_token else 'EOF'}")
    
    def match(self, *token_types: TokenType) -> bool:
        if self.current_token and self.current_token.type in token_types:
            self.advance()
            return True
        return False
    
    def parse(self) -> ASTNode:
        """Main parsing method - parse the entire program"""
        statements = []
        
        while self.current_token and self.current_token.type != TokenType.EOF:
            if self.current_token.type == TokenType.NEWLINE:
                self.advance()
                continue
            
            try:
                stmt = self.parse_statement()
                if stmt:
                    statements.append(stmt)
            except ParserError as e:
                print(f"Parse error: {e.message}")
                # Try to recover by skipping to next line/statement
                while self.current_token and self.current_token.type != TokenType.NEWLINE and self.current_token.type != TokenType.EOF:
                    self.advance()
                if self.current_token:
                    self.advance()
        
        return ASTNode(NodeType.PROGRAM, children=statements)
    
    def parse_statement(self) -> Optional[ASTNode]:
        """Parse a statement"""
        if self.match(TokenType.FUN):
            return self.parse_function()
        elif self.match(TokenType.CLASS):
            return self.parse_class()
        elif self.match(TokenType.MUT):
            return self.parse_variable(mutable=True)
        elif self.current_token and self.current_token.type == TokenType.IDENTIFIER:
            # Could be variable assignment or expression statement
            if self.peek() and self.peek().type in [TokenType.ASSIGN, TokenType.COLON]:
                return self.parse_variable(mutable=False)
            else:
                expr = self.parse_expression()
                return ASTNode(NodeType.EXPRESSION_STMT, children=[expr])
        elif self.match(TokenType.IF):
            return self.parse_if_statement()
        elif self.match(TokenType.MATCH):
            return self.parse_match_statement()
        elif self.match(TokenType.WHILE):
            return self.parse_while_statement()
        elif self.match(TokenType.FOR):
            return self.parse_for_statement()
        elif self.match(TokenType.RETURN):
            return self.parse_return_statement()
        elif self.match(TokenType.BREAK):
            return ASTNode(NodeType.BREAK, line=self.current_token.line, column=self.current_token.column)
        elif self.match(TokenType.CONTINUE):
            return ASTNode(NodeType.CONTINUE, line=self.current_token.line, column=self.current_token.column)
        elif self.match(TokenType.SPAWN):
            return self.parse_spawn_statement()
        else:
            return self.parse_expression_statement()
    
    def parse_function(self) -> ASTNode:
        """Parse function declaration"""
        name_token = self.expect(TokenType.IDENTIFIER)
        
        self.expect(TokenType.LEFT_PAREN)
        params = self.parse_parameters()
        self.expect(TokenType.RIGHT_PAREN)
        
        # Parse return type (optional)
        return_type = None
        if self.match(TokenType.FUNCTION_ARROW):
            return_type = self.parse_type()
        
        # Parse function body
        if self.match(TokenType.SINGLE_EXPR_ARROW):
            # Single expression function
            body = self.parse_expression()
        else:
            # Block function
            self.expect(TokenType.LEFT_BRACE)
            body = self.parse_block()
            self.expect(TokenType.RIGHT_BRACE)
        
        func_node = ASTNode(NodeType.FUNCTION, value=name_token.value, line=name_token.line, column=name_token.column)
        func_node.children = params + [return_type, body] if return_type else params + [body]
        
        return func_node
    
    def parse_parameters(self) -> List[ASTNode]:
        """Parse function parameters"""
        params = []
        
        while self.current_token and self.current_token.type != TokenType.RIGHT_PAREN:
            param_name = self.expect(TokenType.IDENTIFIER)
            
            # Parse type annotation (optional)
            param_type = None
            if self.match(TokenType.COLON):
                param_type = self.parse_type()
            
            param_node = ASTNode(NodeType.IDENTIFIER, value=param_name.value, line=param_name.line, column=param_name.column)
            if param_type:
                param_node.children = [param_type]
            
            params.append(param_node)
            
            if not self.match(TokenType.COMMA):
                break
        
        return params
    
    def parse_type(self) -> ASTNode:
        """Parse type annotation"""
        type_token = self.expect(TokenType.IDENTIFIER)
        type_node = ASTNode(NodeType.IDENTIFIER, value=type_token.value, line=type_token.line, column=type_token.column)
        
        # Handle nullable types
        if self.match(TokenType.NULL_SAFETY):
            # For now, just mark as nullable in the value
            type_node.value = f"{type_node.value}?"
        
        return type_node
    
    def parse_variable(self, mutable: bool = False) -> ASTNode:
        """Parse variable declaration"""
        name_token = self.expect(TokenType.IDENTIFIER)
        
        # Parse type annotation (optional)
        var_type = None
        if self.match(TokenType.COLON):
            var_type = self.parse_type()
        
        # Parse initializer
        initializer = None
        if self.match(TokenType.ASSIGN):
            initializer = self.parse_expression()
        
        var_node = ASTNode(NodeType.VARIABLE, value=name_token.value, line=name_token.line, column=name_token.column)
        if mutable:
            var_node.value = f"mut {name_token.value}"
        
        if var_type:
            var_node.children.append(var_type)
        if initializer:
            var_node.children.append(initializer)
        
        return var_node
    
    def parse_block(self) -> ASTNode:
        """Parse block of statements"""
        statements = []
        
        while self.current_token and self.current_token.type != TokenType.RIGHT_BRACE:
            if self.current_token.type == TokenType.NEWLINE:
                self.advance()
                continue
            
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
        
        return ASTNode(NodeType.BLOCK, children=statements)
    
    def parse_expression(self, precedence: int = 0) -> ASTNode:
        """Parse expression with operator precedence"""
        left = self.parse_primary()
        
        while (self.current_token and 
               self.current_token.type not in [TokenType.NEWLINE, TokenType.EOF, TokenType.RIGHT_BRACE, TokenType.RIGHT_PAREN] and
               self.get_precedence(self.current_token.type) >= precedence):
            operator = self.current_token
            self.advance()
            
            # Handle arrow pipeline (lower precedence)
            if operator.type == TokenType.FUNCTION_ARROW:
                left = self.parse_pipeline(left)
            else:
                right = self.parse_expression(self.get_precedence(operator.type) + 1)
                left = ASTNode(NodeType.BINARY, value=operator.value, children=[left, right], line=operator.line, column=operator.column)
        
        return left
    
    def parse_primary(self) -> ASTNode:
        """Parse primary expression"""
        if self.match(TokenType.NUMBER):
            return ASTNode(NodeType.LITERAL, value="number", children=[ASTNode(NodeType.LITERAL, value=self.tokens[self.position-1].value)])
        elif self.match(TokenType.STRING):
            return ASTNode(NodeType.LITERAL, value="string", children=[ASTNode(NodeType.LITERAL, value=self.tokens[self.position-1].value)])
        elif self.match(TokenType.BOOLEAN):
            return ASTNode(NodeType.LITERAL, value="boolean", children=[ASTNode(NodeType.LITERAL, value=self.tokens[self.position-1].value)])
        elif self.match(TokenType.NULL):
            return ASTNode(NodeType.LITERAL, value="null")
        elif self.match(TokenType.LEFT_PAREN):
            expr = self.parse_expression()
            self.expect(TokenType.RIGHT_PAREN)
            return expr
        elif self.match(TokenType.LEFT_BRACE):
            return self.parse_block()
        elif self.current_token and self.current_token.type == TokenType.IDENTIFIER:
            name = self.current_token.value
            self.advance()
            
            # Check if it's a function call
            if self.match(TokenType.LEFT_PAREN):
                return self.parse_function_call(name)
            else:
                return ASTNode(NodeType.IDENTIFIER, value=name, line=self.tokens[self.position-1].line, column=self.tokens[self.position-1].column)
        elif self.current_token and self.current_token.type == TokenType.NEWLINE:
            # Skip newlines in expressions
            self.advance()
            return self.parse_primary()
        else:
            self.error(f"Unexpected token in expression: {self.current_token.type.name if self.current_token else 'EOF'}")
    
    def parse_function_call(self, func_name: str) -> ASTNode:
        """Parse function call"""
        args = []
        
        while (self.current_token and 
               self.current_token.type != TokenType.RIGHT_PAREN and
               self.current_token.type != TokenType.EOF):
            if self.current_token.type == TokenType.NEWLINE:
                self.advance()
                continue
            args.append(self.parse_expression())
            if not self.match(TokenType.COMMA):
                break
        
        self.expect(TokenType.RIGHT_PAREN)
        
        call_node = ASTNode(NodeType.CALL, value=func_name, children=args)
        return call_node
    
    def parse_pipeline(self, left: ASTNode) -> ASTNode:
        """Parse arrow pipeline expression"""
        transformations = []
        
        while self.current_token and self.current_token.type == TokenType.FUNCTION_ARROW:
            self.advance()
            
            if self.current_token and self.current_token.type == TokenType.IDENTIFIER:
                func_name = self.current_token.value
                self.advance()
                
                # Parse arguments if present
                args = []
                if self.match(TokenType.LEFT_PAREN):
                    while self.current_token and self.current_token.type != TokenType.RIGHT_PAREN:
                        args.append(self.parse_expression())
                        if not self.match(TokenType.COMMA):
                            break
                    self.expect(TokenType.RIGHT_PAREN)
                
                transformations.append(ASTNode(NodeType.CALL, value=func_name, children=args))
            else:
                self.error("Expected function name after arrow")
        
        return ASTNode(NodeType.PIPELINE, children=[left] + transformations)
    
    def get_precedence(self, token_type: TokenType) -> int:
        """Get operator precedence"""
        precedence_map = {
            TokenType.OR: 1,
            TokenType.AND: 2,
            TokenType.EQUALS: 3,
            TokenType.NOT_EQUALS: 3,
            TokenType.LESS_THAN: 4,
            TokenType.GREATER_THAN: 4,
            TokenType.LESS_EQUALS: 4,
            TokenType.GREATER_EQUALS: 4,
            TokenType.PLUS: 5,
            TokenType.MINUS: 5,
            TokenType.MULTIPLY: 6,
            TokenType.DIVIDE: 6,
            TokenType.MODULO: 6,
            TokenType.FUNCTION_ARROW: 0,  # Lowest precedence for pipelines
        }
        return precedence_map.get(token_type, 0)
    
    # Placeholder implementations for other statement types
    def parse_class(self) -> ASTNode:
        self.error("Class parsing not implemented yet")
    
    def parse_if_statement(self) -> ASTNode:
        self.error("If statement parsing not implemented yet")
    
    def parse_match_statement(self) -> ASTNode:
        self.error("Match statement parsing not implemented yet")
    
    def parse_while_statement(self) -> ASTNode:
        self.error("While statement parsing not implemented yet")
    
    def parse_for_statement(self) -> ASTNode:
        self.error("For statement parsing not implemented yet")
    
    def parse_return_statement(self) -> ASTNode:
        self.error("Return statement parsing not implemented yet")
    
    def parse_spawn_statement(self) -> ASTNode:
        """Parse spawn statement"""
        expr = self.parse_expression()
        return ASTNode(NodeType.SPAWN_STMT, children=[expr])
    
    def parse_expression_statement(self) -> ASTNode:
        """Parse expression statement"""
        expr = self.parse_expression()
        return ASTNode(NodeType.EXPRESSION_STMT, children=[expr])

def print_ast(node: ASTNode, indent: int = 0):
    """Debug method to print AST"""
    indent_str = "  " * indent
    print(f"{indent_str}{node.type.name}: {node.value}")
    for child in node.children:
        print_ast(child, indent + 1)

def main():
    """Test the parser with sample code"""
    from lexer import KorlanLexer
    
    sample_code = '''
fun main() {
    print("Hello, Korlan!")
}
'''
    
    print("=== Korlan Parser Test ===")
    print("Input code:")
    print(sample_code)
    
    # Lex and parse
    lexer = KorlanLexer(sample_code)
    tokens = lexer.tokenize()
    
    parser = KorlanParser(tokens)
    ast = parser.parse()
    
    print("\nAST:")
    print_ast(ast)

if __name__ == "__main__":
    main()
