"""
Korlan Virtual Machine (KVM) - The Heart of the Language
Executes Korlan bytecode with a stack-based architecture.
"""

from typing import List, Any, Dict, Callable, Optional
from dataclasses import dataclass
import sys

from compiler import Instruction, OpCode, CompilerError

@dataclass
class CallFrame:
    """Represents a function call frame with local variable scope"""
    return_address: int
    locals: Dict[str, Any]
    function_name: str
    stack_base: int  # Base position in the stack for this frame

class KVMError(Exception):
    def __init__(self, message: str, instruction: Optional[Instruction] = None, ip: int = 0):
        self.message = message
        self.instruction = instruction
        self.ip = ip
        super().__init__(f"KVM Error at instruction {ip}: {message}")

class KorlanVM:
    """Korlan Virtual Machine - Stack-based bytecode executor"""
    
    def __init__(self, debug: bool = False):
        self.stack: List[Any] = []
        self.call_stack: List[CallFrame] = []
        self.globals: Dict[str, Any] = {}
        self.heap: List[Any] = []
        self.ip: int = 0  # Instruction pointer
        self.debug = debug
        
        # Built-in functions
        self.builtins: Dict[str, Callable] = {
            "print": self.builtin_print,
            "builtin_print": self.builtin_print,
            "builtin_read": self.builtin_read,
            "builtin_char_at": self.builtin_char_at,
            "builtin_length": self.builtin_length,
            "builtin_to_int_char": self.builtin_to_int_char,
            "builtin_to_string": self.builtin_to_string,
        }
        
        # Statistics
        self.instructions_executed = 0
        self.max_stack_depth = 0
    
    def error(self, message: str, instruction: Optional[Instruction] = None) -> KVMError:
        raise KVMError(message, instruction, self.ip)
    
    def push(self, value: Any):
        """Push value onto stack"""
        self.stack.append(value)
        self.max_stack_depth = max(self.max_stack_depth, len(self.stack))
        if self.debug:
            print(f"STACK PUSH: {value} (depth: {len(self.stack)})")
    
    def pop(self) -> Any:
        """Pop value from stack"""
        if len(self.stack) == 0:
            self.error("Stack underflow")
        
        value = self.stack.pop()
        if self.debug:
            print(f"STACK POP: {value} (depth: {len(self.stack)})")
        return value
    
    def peek(self, offset: int = 0) -> Any:
        """Peek at stack value without popping"""
        if len(self.stack) <= offset:
            self.error("Stack underflow")
        return self.stack[-(offset + 1)]
    
    def execute(self, bytecode: List[Instruction], constants: List[Any]) -> Any:
        """Execute bytecode and return the result"""
        self.constants = constants
        self.ip = 0
        self.stack.clear()
        self.call_stack.clear()
        
        try:
            while self.ip < len(bytecode):
                instruction = bytecode[self.ip]
                
                if self.debug:
                    print(f"EXEC: {instruction.opcode.value} {instruction.operand or ''} (IP: {self.ip})")
                    print(f"STACK: {self.stack}")
                
                self.execute_instruction(instruction)
                self.instructions_executed += 1
                self.ip += 1
            
            # Return the top of stack if not empty
            return self.stack[-1] if self.stack else None
            
        except KVMError:
            raise
        except Exception as e:
            raise KVMError(f"Unexpected error during execution: {str(e)}", bytecode[self.ip] if self.ip < len(bytecode) else None, self.ip)
    
    def execute_instruction(self, instruction: Instruction):
        """Execute a single instruction"""
        opcode = instruction.opcode
        operand = instruction.operand
        
        if opcode == OpCode.PUSH_INT:
            self.push(operand)
        elif opcode == OpCode.PUSH_FLOAT:
            self.push(float(operand))
        elif opcode == OpCode.PUSH_STRING:
            if operand < 0 or operand >= len(self.constants):
                self.error(f"Invalid string constant index: {operand}", instruction)
            self.push(self.constants[operand])
        elif opcode == OpCode.PUSH_BOOL:
            self.push(bool(operand))
        elif opcode == OpCode.PUSH_NULL:
            self.push(None)
        elif opcode == OpCode.POP:
            self.pop()
        elif opcode == OpCode.DUP:
            value = self.peek()
            self.push(value)
        elif opcode == OpCode.SWAP:
            if len(self.stack) < 2:
                self.error("Stack underflow for SWAP", instruction)
            a = self.pop()
            b = self.pop()
            self.push(a)
            self.push(b)
        
        elif opcode == OpCode.LOAD_VAR:
            # Load variable from current call frame
            if not self.call_stack:
                self.error("No call frame for variable access", instruction)
            
            frame = self.call_stack[-1]
            var_name = self.get_variable_name(operand)
            if var_name not in frame.locals:
                self.error(f"Undefined variable: {var_name}", instruction)
            
            self.push(frame.locals[var_name])
        
        elif opcode == OpCode.STORE_VAR:
            # Store variable in current call frame
            if not self.call_stack:
                self.error("No call frame for variable storage", instruction)
            
            frame = self.call_stack[-1]
            var_name = self.get_variable_name(operand)
            value = self.pop()
            frame.locals[var_name] = value
        
        elif opcode == OpCode.LOAD_GLOBAL:
            # Load global variable
            var_name = self.get_variable_name(operand)
            if var_name not in self.globals:
                self.error(f"Undefined global variable: {var_name}", instruction)
            self.push(self.globals[var_name])
        
        elif opcode == OpCode.STORE_GLOBAL:
            # Store global variable
            var_name = self.get_variable_name(operand)
            value = self.pop()
            self.globals[var_name] = value
        
        elif opcode == OpCode.CALL_FUNC:
            # Call function with proper stack frame management
            func_address = operand
            if func_address < 0 or func_address >= len(self.constants):
                self.error(f"Invalid function address: {func_address}", instruction)
            
            # Create new call frame with current stack base
            frame = CallFrame(
                return_address=self.ip + 1,
                locals={},
                function_name=f"func_{func_address}",
                stack_base=len(self.stack)
            )
            
            # For now, we'll assume no parameters (simplified)
            # In a real implementation, we'd pop arguments and store them as locals
            
            self.call_stack.append(frame)
            self.ip = func_address - 1  # -1 because we increment at end of loop
        
        elif opcode == OpCode.RETURN:
            # Return from function
            if not self.call_stack:
                self.error("Return without call frame", instruction)
            
            frame = self.call_stack.pop()
            self.ip = frame.return_address
        
        elif opcode == OpCode.JUMP:
            # Unconditional jump
            self.ip = operand - 1  # -1 because we increment at end of loop
        
        elif opcode == OpCode.JUMP_IF_FALSE:
            # Conditional jump (pop condition, jump if false)
            condition = self.pop()
            if not self.is_truthy(condition):
                self.ip = operand - 1  # -1 because we increment at end of loop
        
        elif opcode == OpCode.JUMP_IF_TRUE:
            # Conditional jump (pop condition, jump if true)
            condition = self.pop()
            if self.is_truthy(condition):
                self.ip = operand - 1  # -1 because we increment at end of loop
        
        # Binary operations
        elif opcode == OpCode.ADD:
            right = self.pop()
            left = self.pop()
            self.push(self.add(left, right))
        
        elif opcode == OpCode.SUBTRACT:
            right = self.pop()
            left = self.pop()
            self.push(self.subtract(left, right))
        
        elif opcode == OpCode.MULTIPLY:
            right = self.pop()
            left = self.pop()
            self.push(self.multiply(left, right))
        
        elif opcode == OpCode.DIVIDE:
            right = self.pop()
            left = self.pop()
            self.push(self.divide(left, right))
        
        elif opcode == OpCode.MODULO:
            right = self.pop()
            left = self.pop()
            self.push(self.modulo(left, right))
        
        # Comparison operations
        elif opcode == OpCode.EQUALS:
            right = self.pop()
            left = self.pop()
            self.push(self.equals(left, right))
        
        elif opcode == OpCode.NOT_EQUALS:
            right = self.pop()
            left = self.pop()
            self.push(not self.equals(left, right))
        
        elif opcode == OpCode.LESS_THAN:
            right = self.pop()
            left = self.pop()
            self.push(self.less_than(left, right))
        
        elif opcode == OpCode.GREATER_THAN:
            right = self.pop()
            left = self.pop()
            self.push(self.greater_than(left, right))
        
        elif opcode == OpCode.LESS_EQUALS:
            right = self.pop()
            left = self.pop()
            self.push(self.less_equals(left, right))
        
        elif opcode == OpCode.GREATER_EQUALS:
            right = self.pop()
            left = self.pop()
            self.push(self.greater_equals(left, right))
        
        # Logical operations
        elif opcode == OpCode.AND:
            right = self.pop()
            left = self.pop()
            self.push(self.is_truthy(left) and self.is_truthy(right))
        
        elif opcode == OpCode.OR:
            right = self.pop()
            left = self.pop()
            self.push(self.is_truthy(left) or self.is_truthy(right))
        
        elif opcode == OpCode.NOT:
            value = self.pop()
            self.push(not self.is_truthy(value))
        
        # Built-in operations
        elif opcode == OpCode.PRINT:
            arg_count = operand or 1
            args = []
            for _ in range(arg_count):
                args.append(self.pop())
            
            # Print arguments (reverse order because of stack)
            args.reverse()
            output = " ".join(str(arg) for arg in args)
            print(output)
            self.push(None)  # Print returns null
        
        elif opcode == OpCode.PRINT_STACK:
            # Debug instruction to print stack state
            print("=== STACK STATE ===")
            print(f"IP: {self.ip}")
            print(f"Stack depth: {len(self.stack)}")
            print(f"Stack contents:")
            for i, value in enumerate(self.stack):
                print(f"  [{i}]: {value} ({type(value).__name__})")
            print(f"Call frames: {len(self.call_stack)}")
            for i, frame in enumerate(self.call_stack):
                print(f"  Frame {i}: {frame.function_name} (locals: {frame.locals})")
            print(f"Globals: {self.globals}")
            print("==================")
        
        elif opcode == OpCode.HALT:
            # Stop execution
            self.ip = len(self.constants)  # Set IP beyond bytecode to stop
        
        else:
            self.error(f"Unknown opcode: {opcode.name}", instruction)
    
    def get_variable_name(self, index: int) -> str:
        """Get variable name from index (simplified)"""
        # This is a simplified approach - real implementation would track variable names
        return f"var_{index}"
    
    def is_truthy(self, value: Any) -> bool:
        """Determine if a value is truthy"""
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return len(value) > 0
        return True
    
    # Arithmetic operations
    def add(self, left: Any, right: Any) -> Any:
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return left + right
        elif isinstance(left, str) and isinstance(right, str):
            return left + right
        else:
            raise KVMError(f"Cannot add {type(left)} and {type(right)}")
    
    def subtract(self, left: Any, right: Any) -> Any:
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return left - right
        else:
            raise KVMError(f"Cannot subtract {type(right)} from {type(left)}")
    
    def multiply(self, left: Any, right: Any) -> Any:
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return left * right
        elif isinstance(left, str) and isinstance(right, int):
            return left * right
        else:
            raise KVMError(f"Cannot multiply {type(left)} and {type(right)}")
    
    def divide(self, left: Any, right: Any) -> Any:
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if right == 0:
                raise KVMError("Division by zero")
            return left / right
        else:
            raise KVMError(f"Cannot divide {type(left)} by {type(right)}")
    
    def modulo(self, left: Any, right: Any) -> Any:
        if isinstance(left, int) and isinstance(right, int):
            return left % right
        else:
            raise KVMError(f"Cannot modulo {type(left)} by {type(right)}")
    
    # Comparison operations
    def equals(self, left: Any, right: Any) -> bool:
        return left == right
    
    def less_than(self, left: Any, right: Any) -> bool:
        if isinstance(left, (int, float, str)) and isinstance(right, (int, float, str)):
            return left < right
        else:
            raise KVMError(f"Cannot compare {type(left)} and {type(right)}")
    
    def greater_than(self, left: Any, right: Any) -> bool:
        if isinstance(left, (int, float, str)) and isinstance(right, (int, float, str)):
            return left > right
        else:
            raise KVMError(f"Cannot compare {type(left)} and {type(right)}")
    
    def less_equals(self, left: Any, right: Any) -> bool:
        return left <= right
    
    def greater_equals(self, left: Any, right: Any) -> bool:
        return left >= right
    
    # Built-in functions
    def builtin_print(self, *args):
        """Built-in print function"""
        output = " ".join(str(arg) for arg in args)
        print(output)
        return None
    
    def builtin_read(self):
        """Built-in read function"""
        return input()
    
    def builtin_char_at(self, s: str, index: int) -> str:
        """Get character at index"""
        if 0 <= index < len(s):
            return s[index]
        return ""
    
    def builtin_length(self, s: str) -> int:
        """Get string length"""
        return len(s)
    
    def builtin_to_int_char(self, c: str) -> int:
        """Convert character to integer"""
        if len(c) == 1 and c.isdigit():
            return int(c)
        return 0
    
    def builtin_to_string(self, n: int) -> str:
        """Convert integer to string"""
        return str(n)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return {
            "instructions_executed": self.instructions_executed,
            "max_stack_depth": self.max_stack_depth,
            "current_stack_depth": len(self.stack),
            "call_stack_depth": len(self.call_stack),
            "globals_count": len(self.globals),
        }

def main():
    """Test the KVM with sample bytecode"""
    from compiler import OpCode, KorlanCompiler
    from lexer import KorlanLexer
    from parser import KorlanParser
    
    sample_code = '''
fun main() {
    print("Hello, Korlan!")
}
'''
    
    print("=== Korlan VM Test ===")
    print("Input code:")
    print(sample_code)
    
    # Compile to bytecode
    lexer = KorlanLexer(sample_code)
    tokens = lexer.tokenize()
    
    parser = KorlanParser(tokens)
    ast = parser.parse()
    
    compiler = KorlanCompiler()
    bytecode = compiler.compile(ast)
    
    print("\nExecuting bytecode...")
    
    # Execute in VM
    vm = KorlanVM(debug=True)
    result = vm.execute(bytecode, compiler.constants)
    
    print(f"\nExecution completed!")
    print(f"Result: {result}")
    print(f"Stats: {vm.get_stats()}")

if __name__ == "__main__":
    main()
