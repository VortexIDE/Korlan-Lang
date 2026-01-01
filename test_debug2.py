import sys
sys.path.append('core')
from lexer import KorlanLexer
from parser import KorlanParser

# Test const-checking - declare without mut, then try to reassign
code = '''
fun main() {
    x = 5
}
'''

lexer = KorlanLexer(code)
tokens = lexer.tokenize()
parser = KorlanParser(tokens)

# Debug the parsing step by step
print("Current token after parsing function:", parser.current_token.type.name if parser.current_token else "None")
print("Peek token:", parser.peek().type.name if parser.peek() else "None")

# Manually test parse_statement
parser.position = 5  # Skip to after function declaration
parser.current_token = parser.tokens[parser.position] if parser.position < len(parser.tokens) else None

print("Before parse_statement:")
print("  Current:", parser.current_token.type.name if parser.current_token else "None")
print("  Peek:", parser.peek().type.name if parser.peek() else "None")

stmt = parser.parse_statement()
print("Parsed statement:", stmt.type.name, stmt.value)
