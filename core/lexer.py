"""
Korlan Lexer - The Eyes of the Language
Converts Korlan source code into tokens with zero-ceremony syntax enforcement.
"""

import re
from enum import Enum
from typing import List, Tuple, Optional
from dataclasses import dataclass

class TokenType(Enum):
    # Keywords
    FUN = "FUN"
    MUT = "MUT"
    CLASS = "CLASS"
    IF = "IF"
    MATCH = "MATCH"
    SPAWN = "SPAWN"
    INIT = "INIT"
    ELSE = "ELSE"
    WHILE = "WHILE"
    FOR = "FOR"
    RETURN = "RETURN"
    BREAK = "BREAK"
    CONTINUE = "CONTINUE"
    NULL = "NULL"
    
    # Operators
    FUNCTION_ARROW = "FUNCTION_ARROW"      # ->
    SINGLE_EXPR_ARROW = "SINGLE_EXPR_ARROW"  # =>
    NULL_SAFETY = "NULL_SAFETY"           # ?
    ASSIGN = "ASSIGN"                     # =
    EQUALS = "EQUALS"                     # ==
    NOT_EQUALS = "NOT_EQUALS"             # !=
    LESS_THAN = "LESS_THAN"               # <
    GREATER_THAN = "GREATER_THAN"         # >
    LESS_EQUALS = "LESS_EQUALS"           # <=
    GREATER_EQUALS = "GREATER_EQUALS"     # >=
    PLUS = "PLUS"                         # +
    MINUS = "MINUS"                       # *
    MULTIPLY = "MULTIPLY"                 # *
    DIVIDE = "DIVIDE"                     # /
    MODULO = "MODULO"                     # %
    AND = "AND"                           # &&
    OR = "OR"                            # ||
    NOT = "NOT"                           # !
    
    # Delimiters
    LEFT_PAREN = "LEFT_PAREN"             # (
    RIGHT_PAREN = "RIGHT_PAREN"           # )
    LEFT_BRACE = "LEFT_BRACE"             # {
    RIGHT_BRACE = "RIGHT_BRACE"           # }
    LEFT_BRACKET = "LEFT_BRACKET"         # [
    RIGHT_BRACKET = "RIGHT_BRACKET"       # ]
    COMMA = "COMMA"                       # ,
    DOT = "DOT"                           # .
    COLON = "COLON"                       # :
    
    # Literals and Identifiers
    IDENTIFIER = "IDENTIFIER"
    NUMBER = "NUMBER"
    STRING = "STRING"
    BOOLEAN = "BOOLEAN"
    
    # Special
    NEWLINE = "NEWLINE"
    INDENT = "INDENT"
    DEDENT = "DEDENT"
    EOF = "EOF"
    ILLEGAL = "ILLEGAL"

@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int

class LexerError(Exception):
    def __init__(self, message: str, line: int, column: int):
        self.message = message
        self.line = line
        self.column = column
        super().__init__(f"Lexer Error at line {line}, column {column}: {message}")

class KorlanLexer:
    def __init__(self, source: str):
        self.source = source
        self.position = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []
        
        # Token specifications
        self.token_specs = [
            # Multi-character operators (must come first)
            (r'->', TokenType.FUNCTION_ARROW),
            (r'=>', TokenType.SINGLE_EXPR_ARROW),
            (r'==', TokenType.EQUALS),
            (r'!=', TokenType.NOT_EQUALS),
            (r'<=', TokenType.LESS_EQUALS),
            (r'>=', TokenType.GREATER_EQUALS),
            (r'&&', TokenType.AND),
            (r'\|\|', TokenType.OR),
            
            # Single-character tokens
            (r'\(', TokenType.LEFT_PAREN),
            (r'\)', TokenType.RIGHT_PAREN),
            (r'\{', TokenType.LEFT_BRACE),
            (r'\}', TokenType.RIGHT_BRACE),
            (r'\[', TokenType.LEFT_BRACKET),
            (r'\]', TokenType.RIGHT_BRACKET),
            (r',', TokenType.COMMA),
            (r'\.', TokenType.DOT),
            (r':', TokenType.COLON),
            (r'\?', TokenType.NULL_SAFETY),
            (r'=', TokenType.ASSIGN),
            (r'<', TokenType.LESS_THAN),
            (r'>', TokenType.GREATER_THAN),
            (r'\+', TokenType.PLUS),
            (r'-', TokenType.MINUS),
            (r'\*', TokenType.MULTIPLY),
            (r'/', TokenType.DIVIDE),
            (r'%', TokenType.MODULO),
            (r'!', TokenType.NOT),
            
            # Literals
            (r'\d+\.\d+', TokenType.NUMBER),  # Float
            (r'\d+', TokenType.NUMBER),       # Integer
            (r'".*?"', TokenType.STRING),      # String
            (r"'.*?'", TokenType.STRING),      # String (single quotes)
            
            # Identifiers and keywords
            (r'[a-zA-Z_][a-zA-Z0-9_]*', TokenType.IDENTIFIER),
        ]
        
        # Compile regex patterns
        self.compiled_specs = [(re.compile(pattern), token_type) for pattern, token_type in self.token_specs]
        
        # Keywords
        self.keywords = {
            'fun': TokenType.FUN,
            'mut': TokenType.MUT,
            'class': TokenType.CLASS,
            'if': TokenType.IF,
            'match': TokenType.MATCH,
            'spawn': TokenType.SPAWN,
            'init': TokenType.INIT,
            'else': TokenType.ELSE,
            'while': TokenType.WHILE,
            'for': TokenType.FOR,
            'return': TokenType.RETURN,
            'break': TokenType.BREAK,
            'continue': TokenType.CONTINUE,
            'null': TokenType.NULL,
            'true': TokenType.BOOLEAN,
            'false': TokenType.BOOLEAN,
        }
    
    def error(self, message: str) -> LexerError:
        raise LexerError(message, self.line, self.column)
    
    def current_char(self) -> Optional[str]:
        if self.position >= len(self.source):
            return None
        return self.source[self.position]
    
    def advance(self) -> Optional[str]:
        char = self.current_char()
        if char is not None:
            self.position += 1
            if char == '\n':
                self.line += 1
                self.column = 1
            else:
                self.column += 1
        return char
    
    def peek_char(self, offset: int = 0) -> Optional[str]:
        peek_pos = self.position + offset
        if peek_pos >= len(self.source):
            return None
        return self.source[peek_pos]
    
    def skip_whitespace(self):
        while (char := self.current_char()) is not None and char.isspace():
            if char == '\n':
                self.tokens.append(Token(TokenType.NEWLINE, char, self.line, self.column))
            self.advance()
    
    def skip_comment(self):
        if self.current_char() == '/' and self.peek_char() == '/':
            while (char := self.advance()) is not None and char != '\n':
                pass
        elif self.current_char() == '#':
            # Skip hash comments (for self-hosted lexer compatibility)
            while (char := self.advance()) is not None and char != '\n':
                pass
    
    def check_for_semicolon(self):
        """Enforce zero-ceremony: throw error if semicolon found"""
        if self.current_char() == ';':
            self.error("Ceremony Error: Semicolons are not allowed in Korlan. Remove the semicolon and embrace simplicity!")
    
    def read_string(self, quote_char: str) -> str:
        start_line = self.line
        start_column = self.column
        self.advance()  # Skip opening quote
        
        string_value = ""
        while (char := self.current_char()) is not None and char != quote_char:
            if char == '\\':  # Escape sequence
                self.advance()
                escaped_char = self.current_char()
                if escaped_char is None:
                    self.error("Unterminated string literal")
                
                # Handle escape sequences
                escape_map = {
                    'n': '\n',
                    't': '\t',
                    'r': '\r',
                    '\\': '\\',
                    '"': '"',
                    "'": "'",
                }
                string_value += escape_map.get(escaped_char, escaped_char)
                self.advance()
            else:
                string_value += char
                self.advance()
        
        if self.current_char() != quote_char:
            self.error(f"Unterminated string literal (missing closing {quote_char})")
        
        self.advance()  # Skip closing quote
        return string_value
    
    def read_number(self) -> str:
        start_pos = self.position
        while (char := self.current_char()) is not None and (char.isdigit() or char == '.'):
            self.advance()
        
        return self.source[start_pos:self.position]
    
    def read_identifier(self) -> str:
        start_pos = self.position
        while (char := self.current_char()) is not None and (char.isalnum() or char == '_'):
            self.advance()
        
        return self.source[start_pos:self.position]
    
    def tokenize(self) -> List[Token]:
        """Main tokenization method"""
        try:
            while self.position < len(self.source):
                self.skip_whitespace()
                
                if self.position >= len(self.source):
                    break
                
                current_char = self.current_char()
                
                # Check for ceremony errors
                self.check_for_semicolon()
                
                # Skip comments
                if current_char == '/' and self.peek_char() == '/':
                    self.skip_comment()
                    continue
                elif current_char == '#':
                    self.skip_comment()
                    continue
                
                # Try to match token patterns
                token_found = False
                
                # Check for strings first (they have higher priority)
                if current_char in ['"', "'"]:
                    value = self.read_string(current_char)
                    token = Token(TokenType.STRING, value, self.line, self.column - len(value))
                    self.tokens.append(token)
                    token_found = True
                else:
                    for pattern, token_type in self.compiled_specs:
                        match = pattern.match(self.source, self.position)
                        if match:
                            value = match.group(0)
                            
                            # Handle special cases
                            if token_type == TokenType.IDENTIFIER:
                                # Check if it's a keyword
                                if value in self.keywords:
                                    token_type = self.keywords[value]
                            
                            # Create token
                            token = Token(token_type, value, self.line, self.column - len(value) + 1)
                            self.tokens.append(token)
                            
                            # Update position
                            self.position = match.end()
                            
                            # Update line/column for multi-character tokens
                            for char in value:
                                if char == '\n':
                                    self.line += 1
                                    self.column = 1
                                else:
                                    self.column += 1
                            
                            token_found = True
                            break
                
                if not token_found:
                    self.error(f"Unexpected character: '{current_char}'")
            
            # Add EOF token
            self.tokens.append(Token(TokenType.EOF, "", self.line, self.column))
            
            return self.tokens
            
        except LexerError:
            # Re-raise lexer errors
            raise
        except Exception as e:
            # Wrap other exceptions
            raise LexerError(f"Unexpected error during tokenization: {str(e)}", self.line, self.column)
    
    def print_tokens(self):
        """Debug method to print all tokens"""
        for i, token in enumerate(self.tokens):
            print(f"{i:3d}: {token.type.name:15s} '{token.value}' (line {token.line}, col {token.column})")

def main():
    """Test the lexer with sample code"""
    sample_code = '''
fun main() {
    print("Hello, Korlan!")
}
'''
    
    print("=== Korlan Lexer Test ===")
    print("Input code:")
    print(sample_code)
    print("\nTokens:")
    
    lexer = KorlanLexer(sample_code)
    tokens = lexer.tokenize()
    lexer.print_tokens()

if __name__ == "__main__":
    main()
