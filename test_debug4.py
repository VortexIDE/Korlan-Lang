import sys
sys.path.append('core')
from lexer import KorlanLexer
from parser import KorlanParser

# Test const-checking - declare without mut, then try to reassign
code = '''x = 5'''

lexer = KorlanLexer(code)
tokens = lexer.tokenize()
parser = KorlanParser(tokens)

# Test parse_variable directly
parser.position = 0
parser.current_token = parser.tokens[0]
var_node = parser.parse_variable(mutable=False)
print("parse_variable result:", var_node.type.name, var_node.value)
print("Children:", [child.type.name for child in var_node.children] if var_node.children else "None")
