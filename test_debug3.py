import sys
sys.path.append('core')
from lexer import KorlanLexer
from parser import KorlanParser

# Test const-checking - declare without mut, then try to reassign
code = '''x = 5'''

lexer = KorlanLexer(code)
tokens = lexer.tokenize()
for i, token in enumerate(tokens):
    print(f'{i}: {token.type.name} = "{token.value}"')

parser = KorlanParser(tokens)
stmt = parser.parse_statement()
print("Parsed statement:", stmt.type.name, stmt.value)
if stmt.children:
    print("Children:", [child.type.name for child in stmt.children])
