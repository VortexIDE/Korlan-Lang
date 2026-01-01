import sys
sys.path.append('core')
from lexer import KorlanLexer
from parser import KorlanParser, TokenType

# Test const-checking - declare without mut, then try to reassign
code = '''x = 5'''

lexer = KorlanLexer(code)
tokens = lexer.tokenize()
parser = KorlanParser(tokens)

# Debug parse_statement step by step
print("Initial state:")
print("  Current:", parser.current_token.type.name if parser.current_token else "None")
print("  Peek:", parser.peek().type.name if parser.peek() else "None")

# Check the condition
condition = (parser.current_token and 
             parser.current_token.type == TokenType.IDENTIFIER and
             parser.peek() and parser.peek().type in [TokenType.ASSIGN, TokenType.COLON])

print("Condition for parse_variable:", condition)

if condition:
    print("Should call parse_variable")
    stmt = parser.parse_variable(mutable=False)
else:
    print("Would call parse_expression instead")
    stmt = parser.parse_expression()

print("Result:", stmt.type.name, stmt.value)
