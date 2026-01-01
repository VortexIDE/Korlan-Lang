"""
Native Methods Bridge - FFI for Korlan Virtual Machine
Provides essential system functions for the Korlan compiler to operate.
"""

import os
import sys
from typing import Any, List, Dict, Callable, Optional
from dataclasses import dataclass

class NativeError(Exception):
    def __init__(self, message: str, function_name: str = ""):
        self.message = message
        self.function_name = function_name
        super().__init__(f"Native Error in {function_name}: {message}")

@dataclass
class NativeFunction:
    """Represents a native function that can be called from Korlan"""
    name: str
    func: Callable
    arg_count: int
    description: str

class NativeMethods:
    """Bridge between Korlan VM and Python system functions"""
    
    def __init__(self):
        self.functions: Dict[str, NativeFunction] = {}
        self._register_builtin_functions()
    
    def _register_builtin_functions(self):
        """Register all built-in native functions"""
        
        # I/O Functions - essential for self-hosting compiler
        self.register_function(
            "print", 
            self.native_print, 
            -1,  # Variable arguments
            "Print values to stdout"
        )
        
        self.register_function(
            "read_file",
            self.native_read_file,
            1,
            "Read entire file contents as string"
        )
        
        self.register_function(
            "write_file",
            self.native_write_file,
            2,
            "Write string to file (overwrites existing)"
        )
        
        self.register_function(
            "append_file",
            self.native_append_file,
            2,
            "Append string to file"
        )
        
        self.register_function(
            "file_exists",
            self.native_file_exists,
            1,
            "Check if file exists"
        )
        
        # String operations
        self.register_function(
            "length",
            self.native_length,
            1,
            "Get length of string or list"
        )
        
        self.register_function(
            "char_at",
            self.native_char_at,
            2,
            "Get character at index in string"
        )
        
        self.register_function(
            "substring",
            self.native_substring,
            3,
            "Get substring from start to end (exclusive)"
        )
        
        # Type conversion
        self.register_function(
            "to_string",
            self.native_to_string,
            1,
            "Convert value to string"
        )
        
        self.register_function(
            "to_int",
            self.native_to_int,
            1,
            "Convert value to integer"
        )
        
        # System operations
        self.register_function(
            "exit",
            self.native_exit,
            1,
            "Exit program with code"
        )
        
        self.register_function(
            "args",
            self.native_args,
            0,
            "Get command line arguments"
        )
        
        # Math operations
        self.register_function(
            "abs",
            self.native_abs,
            1,
            "Absolute value"
        )
        
        self.register_function(
            "min",
            self.native_min,
            2,
            "Minimum of two values"
        )
        
        self.register_function(
            "max",
            self.native_max,
            2,
            "Maximum of two values"
        )
    
    def register_function(self, name: str, func: Callable, arg_count: int, description: str):
        """Register a native function"""
        self.functions[name] = NativeFunction(name, func, arg_count, description)
    
    def call_function(self, name: str, args: List[Any]) -> Any:
        """Call a native function with arguments"""
        if name not in self.functions:
            raise NativeError(f"Undefined native function: {name}", name)
        
        native_func = self.functions[name]
        
        # Check argument count
        if native_func.arg_count >= 0 and len(args) != native_func.arg_count:
            raise NativeError(
                f"Function '{name}' expects {native_func.arg_count} arguments, got {len(args)}",
                name
            )
        
        try:
            return native_func.func(*args)
        except Exception as e:
            raise NativeError(f"Execution failed: {str(e)}", name)
    
    def get_function_info(self, name: str) -> Optional[NativeFunction]:
        """Get information about a native function"""
        return self.functions.get(name)
    
    def list_functions(self) -> List[str]:
        """List all available native functions"""
        return list(self.functions.keys())
    
    # Native function implementations
    
    def native_print(self, *args) -> None:
        """Print values to stdout"""
        output = " ".join(str(arg) for arg in args)
        print(output)
        return None
    
    def native_read_file(self, filepath: str) -> str:
        """Read entire file contents as string"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            raise NativeError(f"File not found: {filepath}", "read_file")
        except IOError as e:
            raise NativeError(f"Failed to read file '{filepath}': {str(e)}", "read_file")
    
    def native_write_file(self, filepath: str, content: str) -> None:
        """Write string to file (overwrites existing)"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        except IOError as e:
            raise NativeError(f"Failed to write file '{filepath}': {str(e)}", "write_file")
        return None
    
    def native_append_file(self, filepath: str, content: str) -> None:
        """Append string to file"""
        try:
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(content)
        except IOError as e:
            raise NativeError(f"Failed to append to file '{filepath}': {str(e)}", "append_file")
        return None
    
    def native_file_exists(self, filepath: str) -> bool:
        """Check if file exists"""
        return os.path.exists(filepath)
    
    def native_length(self, obj: Any) -> int:
        """Get length of string or list"""
        if isinstance(obj, (str, list, tuple, dict)):
            return len(obj)
        else:
            raise NativeError(f"Cannot get length of {type(obj).__name__}", "length")
    
    def native_char_at(self, s: str, index: int) -> str:
        """Get character at index in string"""
        if not isinstance(s, str):
            raise NativeError("First argument must be a string", "char_at")
        if not isinstance(index, int):
            raise NativeError("Second argument must be an integer", "char_at")
        
        if 0 <= index < len(s):
            return s[index]
        else:
            raise NativeError(f"Index {index} out of bounds for string of length {len(s)}", "char_at")
    
    def native_substring(self, s: str, start: int, end: int) -> str:
        """Get substring from start to end (exclusive)"""
        if not isinstance(s, str):
            raise NativeError("First argument must be a string", "substring")
        if not isinstance(start, int) or not isinstance(end, int):
            raise NativeError("Start and end must be integers", "substring")
        
        if start < 0 or end > len(s) or start > end:
            raise NativeError(f"Invalid range [{start}, {end}) for string of length {len(s)}", "substring")
        
        return s[start:end]
    
    def native_to_string(self, value: Any) -> str:
        """Convert value to string"""
        return str(value)
    
    def native_to_int(self, value: Any) -> int:
        """Convert value to integer"""
        if isinstance(value, int):
            return value
        elif isinstance(value, float):
            return int(value)
        elif isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                raise NativeError(f"Cannot convert string '{value}' to integer", "to_int")
        else:
            raise NativeError(f"Cannot convert {type(value).__name__} to integer", "to_int")
    
    def native_exit(self, code: int) -> None:
        """Exit program with code"""
        if not isinstance(code, int):
            raise NativeError("Exit code must be an integer", "exit")
        sys.exit(code)
    
    def native_args(self) -> List[str]:
        """Get command line arguments"""
        return sys.argv[1:]  # Skip script name
    
    def native_abs(self, value: Any) -> Any:
        """Absolute value"""
        if isinstance(value, (int, float)):
            return abs(value)
        else:
            raise NativeError(f"Cannot compute absolute value of {type(value).__name__}", "abs")
    
    def native_min(self, a: Any, b: Any) -> Any:
        """Minimum of two values"""
        try:
            return a if a < b else b
        except TypeError:
            raise NativeError(f"Cannot compare {type(a).__name__} and {type(b).__name__}", "min")
    
    def native_max(self, a: Any, b: Any) -> Any:
        """Maximum of two values"""
        try:
            return a if a > b else b
        except TypeError:
            raise NativeError(f"Cannot compare {type(a).__name__} and {type(b).__name__}", "max")

def main():
    """Test native methods"""
    print("=== Native Methods Test ===")
    
    native = NativeMethods()
    
    # Test basic functions
    print("Available functions:", native.list_functions())
    
    # Test print
    native.call_function("print", ["Hello", "from", "native", "methods!"])
    
    # Test file operations
    test_content = "Hello from Korlan Native Methods!"
    native.call_function("write_file", ["test.txt", test_content])
    read_content = native.call_function("read_file", ["test.txt"])
    print(f"File test: '{read_content}'")
    
    # Test string operations
    test_str = "Korlan"
    length = native.call_function("length", [test_str])
    char = native.call_function("char_at", [test_str, 2])
    substr = native.call_function("substring", [test_str, 1, 4])
    print(f"String ops: length={length}, char_at(2)={char}, substring(1,4)={substr}")
    
    # Test math
    result = native.call_function("min", [10, 5])
    print(f"Min(10, 5) = {result}")
    
    # Cleanup
    import os
    if os.path.exists("test.txt"):
        os.remove("test.txt")

if __name__ == "__main__":
    main()
