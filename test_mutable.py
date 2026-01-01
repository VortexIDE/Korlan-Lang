import sys
sys.path.append('core')
from lexer import KorlanLexer
from parser import KorlanParser
from checker import StaticAnalyzer

# Test mutable variable - should work
code = '''
fun main() {
    mut x = 5
    x = 10  # Should work - x is mutable
}
'''

print("Testing mutable variable:")
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
