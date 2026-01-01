import sys
sys.path.append('core')
from lexer import KorlanLexer
from parser import KorlanParser, TokenType

# Test const-checking - declare without mut, then try to reassign
code = '''x = 5'''

lexer = KorlanLexer(code)
tokens = lexer.tokenize()
parser = KorlanParser(tokens)

print("All tokens:")
for i, token in enumerate(tokens):
    print(f'  {i}: {token.type.name} = "{token.value}"')

print("\nParser state:")
print(f'  Position: {parser.position}')
print(f'  Current token: {parser.current_token.type.name if parser.current_token else "None"}')
print(f'  Peek(0): {parser.peek(0).type.name if parser.peek(0) else "None"}')
print(f'  Peek(1): {parser.peek(1).type.name if parser.peek(1) else "None"}')

# Check the condition step by step
print("\nCondition check:")
print(f'  parser.current_token exists: {parser.current_token is not None}')
print(f'  current is IDENTIFIER: {parser.current_token.type == TokenType.IDENTIFIER if parser.current_token else "False"}')
print(f'  peek exists: {parser.peek() is not None}')
peek_token = parser.peek()
print(f'  peek is ASSIGN/COLON: {peek_token.type in [TokenType.ASSIGN, TokenType.COLON] if peek_token else "False"}')
