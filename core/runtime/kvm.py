"""
Korlan Virtual Machine (KVM) - The Heart of the Language
Executes Korlan bytecode with a stack-based architecture.
"""

from typing import List, Any, Dict, Callable, Optional
from dataclasses import dataclass
import sys
import os

# Add the runtime directory to the path for imports
sys.path.insert(0, os.path.dirname(__file__))

from compiler import Instruction, OpCode, CompilerError
from native_methods import NativeMethods, NativeError

@dataclass
class CallFrame:
    """Represents a function call frame with proper local variable scope"""
    return_address: int
    locals: Dict[str, Any]
    function_name: str
    stack_base: int  # Base position in the stack for this frame
    parent_frame: Optional['CallFrame'] = None  # For recursion support
    operand_stack: List[Any] = None  # Frame-specific operand stack
    
    def __post_init__(self):
        if self.operand_stack is None:
            self.operand_stack = []

class KorlanVMError(Exception):
    def __init__(self, message: str, line: int = 0, column: int = 0, code_snippet: str = ""):
        self.message = message
        self.line = line
        self.column = column
        self.code_snippet = code_snippet
        super().__init__(f"Korlan Error at line {line}, column {column}: {message}")
        if code_snippet:
            super().__init__(f"Korlan Error at line {line}, column {column}: {message}\nCode: {code_snippet}")

class NativeBridge:
    """Bridge between Korlan and Python native functions"""
    
    def __init__(self):
        self.native_functions = {
            'open': self.native_open,
            'read': self.native_read,
            'sys_argv': self.native_sys_argv,
            'print': self.native_print,
        }
    
    def native_open(self, filename: str, mode: str = 'r') -> Any:
        """Native file open function"""
        try:
            return open(filename, mode)
        except Exception as e:
            raise KorlanVMError(f"Failed to open file '{filename}': {str(e)}")
    
    def native_read(self, file_obj: Any, size: int = -1) -> str:
        """Native file read function"""
        try:
            if hasattr(file_obj, 'read'):
                return file_obj.read(size) if size > 0 else file_obj.read()
            else:
                raise KorlanVMError("Object is not a file object")
        except Exception as e:
            raise KorlanVMError(f"Failed to read from file: {str(e)}")
    
    def native_sys_argv(self) -> List[str]:
        """Native sys.argv access"""
        return sys.argv
    
    def native_print(self, *args) -> None:
        """Native print function"""
        print(*args)
    
    def call_function(self, func_name: str, args: List[Any]) -> Any:
        """Call a native function"""
        if func_name not in self.native_functions:
            raise KorlanVMError(f"Native function '{func_name}' not found")
        
        try:
            return self.native_functions[func_name](*args)
        except Exception as e:
            if isinstance(e, KorlanVMError):
                raise
            raise KorlanVMError(f"Native function '{func_name}' failed: {str(e)}")
    
    def list_functions(self) -> List[str]:
        """List all available native functions"""
        return list(self.native_functions.keys())

class KorlanVM:
    """Korlan Virtual Machine - Stack-based bytecode executor with proper frame management"""
    
    def __init__(self, debug: bool = False):
        self.stack: List[Any] = []
        self.call_stack: List[CallFrame] = []
        self.globals: Dict[str, Any] = {}
        self.heap: List[Any] = []
        self.ip: int = 0  # Instruction pointer
        self.debug = debug
        self.current_frame: Optional[CallFrame] = None
        
        # Initialize native bridge
        self.native_bridge = NativeBridge()
        
        # Built-in functions (legacy support)
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
        self.max_call_depth = 0
    
    def create_frame(self, func_name: str, return_address: int) -> CallFrame:
        """Create a new call frame with proper scope isolation"""
        frame = CallFrame(
            return_address=return_address,
            locals={},
            function_name=func_name,
            stack_base=len(self.stack),
            parent_frame=self.current_frame,
            operand_stack=[]  # Each frame has its own operand stack
        )
        
        # Update call depth tracking
        self.max_call_depth = max(self.max_call_depth, len(self.call_stack) + 1)
        
        return frame
    
    def push_frame(self, frame: CallFrame):
        """Push a new call frame onto the stack"""
        self.call_stack.append(frame)
        self.current_frame = frame
    
    def pop_frame(self) -> Optional[CallFrame]:
        """Pop the current call frame"""
        if not self.call_stack:
            return None
        
        frame = self.call_stack.pop()
        self.current_frame = self.call_stack[-1] if self.call_stack else None
        
        # Clean up stack to frame's base
        if len(self.stack) > frame.stack_base:
            self.stack = self.stack[:frame.stack_base]
        
        return frame
    
    def resolve_variable(self, name: str) -> Optional[Any]:
        """Resolve variable in current frame scope chain"""
        frame = self.current_frame
        while frame:
            if name in frame.locals:
                return frame.locals[name]
            frame = frame.parent_frame
        return None
    
    def set_variable(self, name: str, value: Any) -> bool:
        """Set variable in current frame scope"""
        if self.current_frame:
            self.current_frame.locals[name] = value
            return True
        return False
    
    def error(self, message: str, instruction: Optional[Instruction] = None, line: int = 0, column: int = 0, code_snippet: str = "") -> KorlanVMError:
        if instruction:
            line = instruction.line
            column = instruction.column
        raise KorlanVMError(message, line, column, code_snippet)
    
    def push(self, value: Any):
        """Push value onto current frame's operand stack or global stack"""
        if self.current_frame and hasattr(self.current_frame, 'operand_stack'):
            self.current_frame.operand_stack.append(value)
            if self.debug:
                print(f"FRAME PUSH ({self.current_frame.function_name}): {value} (depth: {len(self.current_frame.operand_stack)})")
        else:
            self.stack.append(value)
            self.max_stack_depth = max(self.max_stack_depth, len(self.stack))
            if self.debug:
                print(f"GLOBAL PUSH: {value} (depth: {len(self.stack)})")
    
    def pop(self) -> Any:
        """Pop value from current frame's operand stack or global stack"""
        if self.current_frame and hasattr(self.current_frame, 'operand_stack') and self.current_frame.operand_stack:
            value = self.current_frame.operand_stack.pop()
            if self.debug:
                print(f"FRAME POP ({self.current_frame.function_name}): {value} (depth: {len(self.current_frame.operand_stack)})")
            return value
        elif self.stack:
            value = self.stack.pop()
            if self.debug:
                print(f"GLOBAL POP: {value} (depth: {len(self.stack)})")
            return value
        else:
            self.error("Stack underflow")
    
    def peek(self, offset: int = 0) -> Any:
        """Peek at stack value without popping"""
        if self.current_frame and hasattr(self.current_frame, 'operand_stack') and self.current_frame.operand_stack:
            frame_stack = self.current_frame.operand_stack
            if len(frame_stack) <= offset:
                self.error("Stack underflow")
            return frame_stack[-(offset + 1)]
        elif self.stack:
            if len(self.stack) <= offset:
                self.error("Stack underflow")
            return self.stack[-(offset + 1)]
        else:
            self.error("Stack underflow")
    
    def execute(self, bytecode: List[Instruction], constants: List[Any]) -> Any:
        """Execute bytecode and return the result"""
        self.constants = constants
        self.bytecode = bytecode  # Store bytecode for HALT instruction
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
            
        except KorlanVMError:
            raise
        except Exception as e:
            raise KorlanVMError(f"Unexpected error during execution: {str(e)}", bytecode[self.ip].line if self.ip < len(bytecode) else 0, bytecode[self.ip].column if self.ip < len(bytecode) else 0)
    
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
            # Load variable from current call frame scope
            var_name = self.get_variable_name(operand)
            value = self.resolve_variable(var_name)
            if value is None:
                self.error(f"Undefined variable: {var_name}", instruction)
            self.push(value)
        
        elif opcode == OpCode.STORE_VAR:
            # Store variable in current call frame scope
            var_name = self.get_variable_name(operand)
            value = self.pop()
            if not self.set_variable(var_name, value):
                self.error(f"Cannot store variable '{var_name}' - no active frame", instruction)
        
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
        
        elif opcode == OpCode.LOAD_LOCAL:
            # Load local variable from current frame
            var_name = self.get_variable_name(operand)
            if self.current_frame and var_name in self.current_frame.locals:
                self.push(self.current_frame.locals[var_name])
            else:
                self.error(f"Undefined local variable: {var_name}", instruction)
        
        elif opcode == OpCode.STORE_LOCAL:
            # Store local variable in current frame
            var_name = self.get_variable_name(operand)
            value = self.pop()
            if self.current_frame:
                self.current_frame.locals[var_name] = value
            else:
                self.error(f"Cannot store local variable '{var_name}' - no active frame", instruction)
        
        elif opcode == OpCode.GET_ATTR:
            # Get attribute from object
            attr_name = self.get_variable_name(operand)
            obj = self.pop()
            
            if obj is None:
                self.error(f"Cannot access attribute '{attr_name}' of null", instruction)
            
            # Handle dictionary-like objects
            if isinstance(obj, dict):
                if attr_name in obj:
                    self.push(obj[attr_name])
                else:
                    self.error(f"Object has no attribute '{attr_name}'", instruction)
            # Handle string methods
            elif isinstance(obj, str) and hasattr(obj, attr_name):
                attr_value = getattr(obj, attr_name)
                if callable(attr_value):
                    # For methods, we'd need to create a bound method
                    # For now, just push the method reference
                    self.push(attr_value)
                else:
                    self.push(attr_value)
            else:
                self.error(f"Cannot access attribute '{attr_name}' on {type(obj).__name__}", instruction)
        
        elif opcode == OpCode.GET_INDEX:
            # Get index from list or map
            index = self.pop()
            collection = self.pop()
            
            if collection is None:
                self.error(f"Cannot index into null collection", instruction)
            
            # Handle list-like objects
            if isinstance(collection, (list, tuple)):
                if not isinstance(index, int):
                    self.error(f"List index must be integer, got {type(index).__name__}", instruction)
                
                if index < 0 or index >= len(collection):
                    self.error(f"Index {index} out of bounds for collection of length {len(collection)}", instruction)
                
                self.push(collection[index])
            # Handle dictionary-like objects
            elif isinstance(collection, dict):
                if index in collection:
                    self.push(collection[index])
                else:
                    self.push(None)  # Return null for missing keys (like safe navigation)
            # Handle string indexing
            elif isinstance(collection, str):
                if not isinstance(index, int):
                    self.error(f"String index must be integer, got {type(index).__name__}", instruction)
                
                if index < 0 or index >= len(collection):
                    self.error(f"Index {index} out of bounds for string of length {len(collection)}", instruction)
                
                self.push(collection[index])
            else:
                self.error(f"Cannot index into {type(collection).__name__}", instruction)
        
        elif opcode == OpCode.SET_INDEX:
            # Set index in list or map
            value = self.pop()
            index = self.pop()
            collection = self.pop()
            
            if collection is None:
                self.error(f"Cannot index into null collection", instruction)
            
            # Handle list-like objects
            if isinstance(collection, list):
                if not isinstance(index, int):
                    self.error(f"List index must be integer, got {type(index).__name__}", instruction)
                
                if index < 0 or index >= len(collection):
                    self.error(f"Index {index} out of bounds for collection of length {len(collection)}", instruction)
                
                collection[index] = value
                self.push(value)  # Return the set value
            # Handle dictionary-like objects
            elif isinstance(collection, dict):
                collection[index] = value
                self.push(value)  # Return the set value
            else:
                self.error(f"Cannot set index on {type(collection).__name__}", instruction)
        
        elif opcode == OpCode.CALL_FUNC:
            # Call function with proper frame management
            func_address = operand
            if func_address < 0 or func_address >= len(self.constants):
                self.error(f"Invalid function address: {func_address}", instruction)
            
            # Create new call frame
            frame = self.create_frame(f"func_{func_address}", self.ip + 1)
            
            # For now, we'll assume no parameters (simplified)
            # In a real implementation, we'd pop arguments and store them as locals
            
            self.push_frame(frame)
            self.ip = func_address - 1  # -1 because we increment at end of loop
        
        elif opcode == OpCode.CALL_NATIVE:
            # Call native function through FFI bridge
            func_name = operand
            if func_name < 0 or func_name >= len(self.constants):
                self.error(f"Invalid native function index: {func_name}", instruction)
            
            native_func_name = self.constants[func_name]
            arg_count = self.pop() if self.stack else 0
            
            # Collect arguments from stack (in reverse order)
            args = []
            for _ in range(arg_count):
                args.append(self.pop())
            args.reverse()  # Put them back in correct order
            
            try:
                result = self.native_bridge.call_function(native_func_name, args)
                self.push(result)
            except KorlanVMError as e:
                self.error(e.message, instruction, e.line, e.column, e.code_snippet)
        
        elif opcode == OpCode.RETURN:
            # Return from function with proper frame cleanup
            if not self.call_stack:
                self.error("Return without call frame", instruction)
            
            frame = self.pop_frame()
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
            # Debug instruction to print complete VM state
            print("=== KORLAN VM STATE ===")
            print(f"IP: {self.ip}")
            print(f"Stack depth: {len(self.stack)}")
            print(f"Max stack depth: {self.max_stack_depth}")
            print(f"Call depth: {len(self.call_stack)}")
            print(f"Max call depth: {self.max_call_depth}")
            print(f"Stack contents:")
            for i, value in enumerate(self.stack):
                print(f"  [{i}]: {value} ({type(value).__name__})")
            print(f"Call frames:")
            for i, frame in enumerate(self.call_stack):
                print(f"  Frame {i}: {frame.function_name} (base: {frame.stack_base})")
                print(f"    Locals: {frame.locals}")
                if hasattr(frame, 'operand_stack'):
                    print(f"    Operand Stack: {frame.operand_stack}")
                if frame.parent_frame:
                    print(f"    Parent: {frame.parent_frame.function_name}")
            print(f"Globals: {self.globals}")
            print(f"Current frame: {self.current_frame.function_name if self.current_frame else 'None'}")
            if self.current_frame and hasattr(self.current_frame, 'operand_stack'):
                print(f"Current frame operand stack: {self.current_frame.operand_stack}")
            print("======================")
        
        elif opcode == OpCode.HALT:
            # Stop execution
            self.ip = len(self.bytecode) + 1  # Set IP beyond bytecode to stop
        
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
            raise KorlanVMError(f"Cannot add {type(left)} and {type(right)}")
    
    def subtract(self, left: Any, right: Any) -> Any:
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return left - right
        else:
            raise KorlanVMError(f"Cannot subtract {type(right)} from {type(left)}")
    
    def multiply(self, left: Any, right: Any) -> Any:
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return left * right
        elif isinstance(left, str) and isinstance(right, int):
            return left * right
        else:
            raise KorlanVMError(f"Cannot multiply {type(left)} and {type(right)}")
    
    def divide(self, left: Any, right: Any) -> Any:
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if right == 0:
                raise KorlanVMError("Division by zero")
            return left / right
        else:
            raise KorlanVMError(f"Cannot divide {type(left)} by {type(right)}")
    
    def modulo(self, left: Any, right: Any) -> Any:
        if isinstance(left, int) and isinstance(right, int):
            return left % right
        else:
            raise KorlanVMError(f"Cannot modulo {type(left)} by {type(right)}")
    
    # Comparison operations
    def equals(self, left: Any, right: Any) -> bool:
        return left == right
    
    def less_than(self, left: Any, right: Any) -> bool:
        if isinstance(left, (int, float, str)) and isinstance(right, (int, float, str)):
            return left < right
        else:
            raise KorlanVMError(f"Cannot compare {type(left)} and {type(right)}")
    
    def greater_than(self, left: Any, right: Any) -> bool:
        if isinstance(left, (int, float, str)) and isinstance(right, (int, float, str)):
            return left > right
        else:
            raise KorlanVMError(f"Cannot compare {type(left)} and {type(right)}")
    
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
            "max_call_depth": self.max_call_depth,
            "globals_count": len(self.globals),
            "current_frame": self.current_frame.function_name if self.current_frame else None,
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
