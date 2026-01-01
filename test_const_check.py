import sys
sys.path.append('core')
from lexer import KorlanLexer
from parser import KorlanParser
from checker import StaticAnalyzer

# Test const-checking - declare without mut, then try to reassign
code = '''
fun main() {
    x = 5
    x = 10  # Should error - x is immutable
}
'''

print("Testing const-checking:")
print(code)
print()

lexer = KorlanLexer(code)
tokens = lexer.tokenize()
parser = KorlanParser(tokens)
ast = parser.parse()

analyzer = StaticAnalyzer()
success = analyzer.check(ast)
analyzer.print_errors()
print("Success:", success)
