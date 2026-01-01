import sys
sys.path.append('core')
from lexer import KorlanLexer
from parser import KorlanParser
from checker import StaticAnalyzer

# Test const-checking - declare without mut, then try to reassign
code = '''
fun main() {
    x = 5
    x = 10
}
'''

lexer = KorlanLexer(code)
tokens = lexer.tokenize()
for i, token in enumerate(tokens[:10]):
    print(f'{i}: {token.type.name} = "{token.value}"')

parser = KorlanParser(tokens)
ast = parser.parse()
print('\nAST:')
print(ast)

analyzer = StaticAnalyzer()
success = analyzer.check(ast)
analyzer.print_errors()
