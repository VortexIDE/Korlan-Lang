"""
Korlan Bytecode Compiler - The Brain's Translator
Converts AST into Korlan Bytecode for the KVM to execute.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import sys

from parser import ASTNode, NodeType, ParserError

# LLVM imports for native compilation
import llvmlite.ir as ir
import llvmlite.binding as llvm

class OpCode(Enum):
    """Korlan Virtual Machine Opcodes"""
    # Stack operations
    PUSH_INT = "PUSH_INT"
    PUSH_FLOAT = "PUSH_FLOAT"
    PUSH_STRING = "PUSH_STRING"
    PUSH_BOOL = "PUSH_BOOL"
    PUSH_NULL = "PUSH_NULL"
    POP = "POP"
    DUP = "DUP"
    SWAP = "SWAP"
    
    # Variable operations
    LOAD_VAR = "LOAD_VAR"
    STORE_VAR = "STORE_VAR"
    LOAD_LOCAL = "LOAD_LOCAL"
    STORE_LOCAL = "STORE_LOCAL"
    LOAD_GLOBAL = "LOAD_GLOBAL"
    STORE_GLOBAL = "STORE_GLOBAL"
    
    # Object operations
    GET_ATTR = "GET_ATTR"
    SET_ATTR = "SET_ATTR"
    GET_INDEX = "GET_INDEX"
    SET_INDEX = "SET_INDEX"
    
    # Class and interface operations
    NEW_CLASS = "NEW_CLASS"
    NEW_INTERFACE = "NEW_INTERFACE"
    INHERIT = "INHERIT"
    IMPLEMENT = "IMPLEMENT"
    METHOD_CALL = "METHOD_CALL"
    SUPER_CALL = "SUPER_CALL"
    
    # Function operations
    CALL_FUNC = "CALL_FUNC"
    CALL_NATIVE = "CALL_NATIVE"
    RETURN = "RETURN"
    CLOSURE = "CLOSURE"
    
    # Control flow
    JUMP = "JUMP"
    JUMP_IF_FALSE = "JUMP_IF_FALSE"
    JUMP_IF_TRUE = "JUMP_IF_TRUE"
    
    # Binary operations
    ADD = "ADD"
    SUBTRACT = "SUBTRACT"
    MULTIPLY = "MULTIPLY"
    DIVIDE = "DIVIDE"
    MODULO = "MODULO"
    
    # Comparison operations
    EQUALS = "EQUALS"
    NOT_EQUALS = "NOT_EQUALS"
    LESS_THAN = "LESS_THAN"
    GREATER_THAN = "GREATER_THAN"
    LESS_EQUALS = "LESS_EQUALS"
    GREATER_EQUALS = "GREATER_EQUALS"
    
    # Logical operations
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    
    # Other operations
    PRINT = "PRINT"
    PRINT_STACK = "PRINT_STACK"
    HALT = "HALT"

@dataclass
class Instruction:
    opcode: OpCode
    operand: Any = None
    line: int = 0
    column: int = 0

class CompilerError(Exception):
    def __init__(self, message: str, node: ASTNode):
        self.message = message
        self.node = node
        super().__init__(f"Compiler Error at line {node.line}, column {node.column}: {message}")

class KorlanCompiler:
    def __init__(self):
        self.bytecode: List[Instruction] = []
        self.constants: List[Any] = []
        self.variables: Dict[str, int] = {}
        self.functions: Dict[str, int] = {}
        self.classes: Dict[str, int] = {}
        self.labels: Dict[str, int] = {}
        self.current_function = None
        self.current_class = None
        self.scope_depth = 0
        self.loop_stack = []
        
        # Built-in functions
        self.builtin_functions = {
            "print": OpCode.PRINT,
        }
    
    def error(self, message: str, node: ASTNode) -> CompilerError:
        raise CompilerError(message, node)
    
    def emit(self, opcode: OpCode, operand: Any = None, node: Optional[ASTNode] = None) -> int:
        """Emit an instruction and return its position"""
        instruction = Instruction(
            opcode, 
            operand, 
            node.line if node else 0, 
            node.column if node else 0
        )
        position = len(self.bytecode)
        self.bytecode.append(instruction)
        return position
    
    def add_constant(self, value: Any) -> int:
        """Add a constant to the constant pool"""
        self.constants.append(value)
        return len(self.constants) - 1
    
    def get_variable_index(self, name: str) -> Optional[int]:
        """Get variable index from current scope"""
        return self.variables.get(name)
    
    def set_variable_index(self, name: str) -> int:
        """Set variable index in current scope"""
        if name in self.variables:
            return self.variables[name]
        
        index = len(self.variables)
        self.variables[name] = index
        return index
    
    def compile(self, ast: ASTNode) -> List[Instruction]:
        """Main compilation method"""
        try:
            self.compile_node(ast)
            self.emit(OpCode.HALT)
            return self.bytecode
        except CompilerError:
            raise
        except Exception as e:
            raise CompilerError(f"Unexpected error during compilation: {str(e)}", ast)
    
    def compile_node(self, node: ASTNode):
        """Compile an AST node"""
        if node.type == NodeType.PROGRAM:
            for child in node.children:
                self.compile_node(child)
        elif node.type == NodeType.FUNCTION:
            self.compile_function(node)
        elif node.type == NodeType.CLASS:
            self.compile_class(node)
        elif node.type == NodeType.VARIABLE:
            self.compile_variable(node)
        elif node.type == NodeType.EXPRESSION_STMT:
            self.compile_node(node.children[0])
        elif node.type == NodeType.ASSIGN:
            self.compile_assignment(node)
        elif node.type == NodeType.BINARY:
            self.compile_binary(node)
        elif node.type == NodeType.CALL:
            self.compile_function_call(node)
        elif node.type == NodeType.PIPELINE:
            self.compile_pipeline(node)
        elif node.type == NodeType.LITERAL:
            self.compile_literal(node)
        elif node.type == NodeType.IDENTIFIER:
            self.compile_identifier(node)
        elif node.type == NodeType.BLOCK:
            for child in node.children:
                self.compile_node(child)
        else:
            self.error(f"Compilation not implemented for node type: {node.type.name}", node)
    
    def compile_function(self, node: ASTNode):
        """Compile function declaration"""
        func_name = node.value
        self.current_function = func_name
        
        # Store function position
        self.functions[func_name] = len(self.bytecode)
        
        # Compile function body
        if len(node.children) > 0:
            body = node.children[-1]  # Last child is the body
            if body.type == NodeType.BLOCK:
                for stmt in body.children:
                    self.compile_node(stmt)
            else:
                # Single expression function
                self.compile_node(body)
                self.emit(OpCode.RETURN)
        else:
            self.emit(OpCode.RETURN)
        
        self.current_function = None
    
    def compile_class(self, node: ASTNode):
        """Compile class declaration"""
        class_name = node.value
        self.current_class = class_name
        
        # Store class position
        self.classes[class_name] = len(self.bytecode)
        
        # Emit class creation instruction
        class_index = self.add_constant(class_name)
        self.emit(OpCode.NEW_CLASS, class_index, node)
        
        # Compile class body (properties and methods)
        if len(node.children) > 0:
            for child in node.children:
                if child.type == NodeType.FUNCTION:
                    # This is a method
                    self.compile_method(child, class_name)
                elif child.type == NodeType.VARIABLE:
                    # This is a property
                    self.compile_property(child, class_name)
        
        self.current_class = None
    
    def compile_method(self, node: ASTNode, class_name: str):
        """Compile class method"""
        method_name = node.value
        old_class = self.current_class
        
        # Create method signature
        full_method_name = f"{class_name}.{method_name}"
        self.functions[full_method_name] = len(self.bytecode)
        
        # Compile method body
        if len(node.children) > 0:
            body = node.children[-1]
            if body.type == NodeType.BLOCK:
                for stmt in body.children:
                    self.compile_node(stmt)
            else:
                self.compile_node(body)
                self.emit(OpCode.RETURN)
        else:
            self.emit(OpCode.RETURN)
    
    def compile_property(self, node: ASTNode, class_name: str):
        """Compile class property"""
        prop_name = node.value.replace("mut ", "")  # Remove mut prefix
        
        # Set property index in class context
        prop_index = self.set_variable_index(f"{class_name}.{prop_name}")
        
        # Compile initializer if present
        if len(node.children) > 0:
            initializer = node.children[-1]
            self.compile_node(initializer)
            self.emit(OpCode.STORE_VAR, prop_index, node)
        else:
            # Store default value (null)
            self.emit(OpCode.PUSH_NULL)
            self.emit(OpCode.STORE_VAR, prop_index, node)
    
    def compile_variable(self, node: ASTNode):
        """Compile variable declaration"""
        var_name = node.value.replace("mut ", "")  # Remove mut prefix
        
        # Set variable index
        var_index = self.set_variable_index(var_name)
        
        # Compile initializer if present
        if len(node.children) > 0:
            initializer = node.children[-1]  # Last child is the initializer
            self.compile_node(initializer)
            self.emit(OpCode.STORE_VAR, var_index, node)
        else:
            # Store default value (null)
            self.emit(OpCode.PUSH_NULL)
            self.emit(OpCode.STORE_VAR, var_index, node)
    
    def compile_assignment(self, node: ASTNode):
        """Compile assignment expression"""
        if len(node.children) != 2:
            self.error("Assignment must have exactly 2 children", node)
        
        target, value = node.children
        
        # Compile the value
        self.compile_node(value)
        
        # Store the value in the variable
        if target.type == NodeType.IDENTIFIER:
            var_name = target.value
            var_index = self.get_variable_index(var_name)
            if var_index is None:
                var_index = self.set_variable_index(var_name)
            self.emit(OpCode.STORE_VAR, var_index, node)
        else:
            self.error("Assignment target must be an identifier", target)
    
    def compile_binary(self, node: ASTNode):
        """Compile binary expression"""
        if len(node.children) != 2:
            self.error("Binary expression must have exactly 2 children", node)
        
        left, right = node.children
        self.compile_node(left)
        self.compile_node(right)
        
        # Emit appropriate operation based on operator
        op_map = {
            "+": OpCode.ADD,
            "-": OpCode.SUBTRACT,
            "*": OpCode.MULTIPLY,
            "/": OpCode.DIVIDE,
            "%": OpCode.MODULO,
            "==": OpCode.EQUALS,
            "!=": OpCode.NOT_EQUALS,
            "<": OpCode.LESS_THAN,
            ">": OpCode.GREATER_THAN,
            "<=": OpCode.LESS_EQUALS,
            ">=": OpCode.GREATER_EQUALS,
            "&&": OpCode.AND,
            "||": OpCode.OR,
        }
        
        opcode = op_map.get(node.value)
        if opcode is None:
            self.error(f"Unknown binary operator: {node.value}", node)
        
        self.emit(opcode, node=node)
    
    def compile_function_call(self, node: ASTNode):
        """Compile function call with method call support"""
        func_name = node.value
        
        # Check for method calls (object.method pattern)
        if "." in func_name and len(node.children) > 0:
            # This is likely a method call
            parts = func_name.split(".")
            if len(parts) == 2:
                object_name, method_name = parts
                
                # Compile the object first
                if object_name in self.variables:
                    var_index = self.get_variable_index(object_name)
                    self.emit(OpCode.LOAD_VAR, var_index, node)
                else:
                    self.error(f"Undefined object for method call: {object_name}", node)
                
                # Compile arguments
                for arg in node.children:
                    self.compile_node(arg)
                
                # Emit method call
                method_index = self.add_constant(method_name)
                self.emit(OpCode.METHOD_CALL, method_index, node)
                return
        
        # Check for built-in functions
        if func_name in self.builtin_functions:
            # Compile arguments
            for arg in node.children:
                self.compile_node(arg)
            
            # Emit built-in function call
            self.emit(self.builtin_functions[func_name], len(node.children), node)
        else:
            # Compile arguments
            for arg in node.children:
                self.compile_node(arg)
            
            # Call user-defined function
            func_index = self.functions.get(func_name)
            if func_index is None:
                self.error(f"Undefined function: {func_name}", node)
            
            self.emit(OpCode.CALL_FUNC, func_index, node)
    
    def compile_pipeline(self, node: ASTNode):
        """Compile arrow pipeline expression with proper first-argument handling"""
        if len(node.children) < 2:
            self.error("Pipeline must have at least 2 children", node)
        
        # Compile the initial value (left side of first arrow)
        self.compile_node(node.children[0])
        
        # Compile each transformation step
        for i in range(1, len(node.children)):
            transform = node.children[i]
            
            if transform.type != NodeType.CALL:
                self.error("Pipeline transformations must be function calls", transform)
                return
            
            func_name = transform.value
            
            # Compile additional arguments (if any)
            for arg in transform.children:
                self.compile_node(arg)
            
            # Arrow Pipeline Logic: value -> func(args) becomes func(value, args...)
            # The current stack has: [pipeline_value, arg1, arg2, ...]
            # We need to rearrange to: [arg1, arg2, ..., pipeline_value]
            # So pipeline_value becomes the FIRST argument
            
            if len(transform.children) > 0:
                # We have additional arguments, need to rearrange stack
                arg_count = len(transform.children)
                
                # Store all arguments including pipeline value
                # Current stack: [pipeline_value, arg1, arg2, ..., argN]
                
                # Store pipeline value temporarily
                self.emit(OpCode.STORE_GLOBAL, "__pipeline_temp", node=node)
                
                # Now stack has: [arg1, arg2, ..., argN]
                # Load pipeline value back first
                self.emit(OpCode.LOAD_GLOBAL, "__pipeline_temp", node=node)
                
                # Now stack has: [arg1, arg2, ..., argN, pipeline_value]
                # We need pipeline_value to be FIRST, so we need to reverse order
                
                # Store all current values temporarily
                for j in range(arg_count + 1):
                    self.emit(OpCode.STORE_GLOBAL, f"__pipeline_arg_{j}", node=node)
                
                # Load pipeline value first
                self.emit(OpCode.LOAD_GLOBAL, "__pipeline_temp", node=node)
                
                # Load additional arguments in original order
                for j in range(arg_count):
                    self.emit(OpCode.LOAD_GLOBAL, f"__pipeline_arg_{j}", node=node)
                
                # Clean up temporaries
                for j in range(arg_count + 1):
                    self.emit(OpCode.PUSH_NULL, node=node)  # Placeholder for cleanup
                    self.emit(OpCode.STORE_GLOBAL, f"__pipeline_arg_{j}", node=node)
                self.emit(OpCode.PUSH_NULL, node=node)  # Placeholder for cleanup
                self.emit(OpCode.STORE_GLOBAL, "__pipeline_temp", node=node)
                
                total_args = arg_count + 1
                
            else:
                # No additional arguments, pipeline value is the only argument
                total_args = 1
            
            # Call the function
            if func_name in self.builtin_functions:
                self.emit(self.builtin_functions[func_name], total_args, transform)
            else:
                # Call user-defined function
                func_index = self.functions.get(func_name)
                if func_index is None:
                    self.error(f"Undefined function in pipeline: {func_name}", transform)
                self.emit(OpCode.CALL_FUNC, func_index, transform)
    
    def compile_literal(self, node: ASTNode):
        """Compile literal value"""
        if len(node.children) > 0:
            value = node.children[0].value
        else:
            value = node.value
        
        if value == "number":
            # Parse number
            num_str = node.children[0].value
            if '.' in num_str:
                self.emit(OpCode.PUSH_FLOAT, float(num_str), node)
            else:
                self.emit(OpCode.PUSH_INT, int(num_str), node)
        elif value == "string":
            # Remove quotes from string
            str_value = node.children[0].value
            if str_value.startswith('"') and str_value.endswith('"'):
                str_value = str_value[1:-1]
            elif str_value.startswith("'") and str_value.endswith("'"):
                str_value = str_value[1:-1]
            
            const_index = self.add_constant(str_value)
            self.emit(OpCode.PUSH_STRING, const_index, node)
        elif value == "boolean":
            bool_value = node.children[0].value.lower() == "true"
            self.emit(OpCode.PUSH_BOOL, bool_value, node)
        elif value == "null":
            self.emit(OpCode.PUSH_NULL, node=node)
        else:
            # Direct value
            if isinstance(value, int):
                self.emit(OpCode.PUSH_INT, value, node)
            elif isinstance(value, float):
                self.emit(OpCode.PUSH_FLOAT, value, node)
            elif isinstance(value, bool):
                self.emit(OpCode.PUSH_BOOL, value, node)
            elif value is None:
                self.emit(OpCode.PUSH_NULL, node=node)
            elif isinstance(value, str):
                const_index = self.add_constant(value)
                self.emit(OpCode.PUSH_STRING, const_index, node)
            else:
                self.error(f"Unsupported literal type: {type(value)}", node)
    
    def compile_identifier(self, node: ASTNode):
        """Compile identifier (variable access)"""
        var_name = node.value
        var_index = self.get_variable_index(var_name)
        
        if var_index is None:
            self.error(f"Undefined variable: {var_name}", node)
        
        self.emit(OpCode.LOAD_VAR, var_index, node)
    
    def print_bytecode(self):
        """Debug method to print bytecode"""
        print("=== Korlan Bytecode ===")
        print(f"Constants: {self.constants}")
        print(f"Variables: {self.variables}")
        print(f"Functions: {self.functions}")
        print(f"Classes: {self.classes}")
        print("\nInstructions:")
        for i, instruction in enumerate(self.bytecode):
            operand_str = f" {instruction.operand}" if instruction.operand is not None else ""
            print(f"{i:3d}: {instruction.opcode.value:15s}{operand_str}")

class LLVMEmitter:
    """Native LLVM IR Emitter - The Heart of Performance"""
    
    def __init__(self):
        # LLVM initialization is now handled automatically
        
        # Create LLVM module and context
        self.context = ir.Context()
        self.module = ir.Module("korlan_module")
        self.module.triple = llvm.get_default_triple()
        
        # Type definitions
        self.int_type = ir.IntType(32)
        self.float_type = ir.FloatType()
        self.bool_type = ir.IntType(1)
        self.void_type = ir.VoidType()
        self.string_type = ir.IntType(8).as_pointer()
        
        # Symbol tables
        self.variables: Dict[str, ir.Value] = {}
        self.functions: Dict[str, ir.Function] = {}
        self.constants: List[ir.Value] = []
        
        # Current compilation context
        self.builder: Optional[ir.IRBuilder] = None
        self.current_function: Optional[ir.Function] = None
        
        # External C functions
        self._declare_external_functions()
    
    def _declare_external_functions(self):
        """Declare external C functions for standard library"""
        # printf
        printf_type = ir.FunctionType(self.int_type, (self.string_type,), True)
        self.printf_func = ir.Function(self.module, printf_type, "printf")
        
        # malloc
        malloc_type = ir.FunctionType(self.string_type, (self.int_type,), False)
        self.malloc_func = ir.Function(self.module, malloc_type, "malloc")
        
        # free
        free_type = ir.FunctionType(self.void_type, (self.string_type,), False)
        self.free_func = ir.Function(self.module, free_type, "free")
    
    def error(self, message: str, node: ASTNode) -> CompilerError:
        """Raise compilation error with location info"""
        raise CompilerError(message, node)
    
    def compile_to_llvm_ir(self, ast: ASTNode) -> str:
        """Compile AST to LLVM IR string"""
        try:
            self.compile_node(ast)
            
            # Add main function if not exists
            if "main" not in self.functions:
                self._create_main_function()
            
            return str(self.module)
        except CompilerError:
            raise
        except Exception as e:
            raise CompilerError(f"LLVM compilation error: {str(e)}", ast)
    
    def compile_node(self, node: ASTNode):
        """Compile an AST node to LLVM IR"""
        if node.type == NodeType.PROGRAM:
            self.compile_program(node)
        elif node.type == NodeType.FUNCTION:
            self.compile_function(node)
        elif node.type == NodeType.VARIABLE:
            self.compile_variable(node)
        elif node.type == NodeType.EXPRESSION_STMT:
            self.compile_node(node.children[0])
        elif node.type == NodeType.ASSIGN:
            self.compile_assignment(node)
        elif node.type == NodeType.BINARY:
            return self.compile_binary(node)
        elif node.type == NodeType.CALL:
            return self.compile_function_call(node)
        elif node.type == NodeType.LITERAL:
            return self.compile_literal(node)
        elif node.type == NodeType.IDENTIFIER:
            return self.compile_identifier(node)
        elif node.type == NodeType.BLOCK:
            self.compile_block(node)
        else:
            self.error(f"LLVM compilation not implemented for node type: {node.type.name}", node)
    
    def compile_program(self, node: ASTNode):
        """Compile program node"""
        for child in node.children:
            self.compile_node(child)
    
    def compile_function(self, node: ASTNode):
        """Compile function declaration"""
        func_name = node.value
        
        # Create function type (int main() for now)
        func_type = ir.FunctionType(self.int_type, ())
        func = ir.Function(self.module, func_type, func_name)
        self.functions[func_name] = func
        
        # Create entry block
        entry_block = func.append_basic_block("entry")
        self.builder = ir.IRBuilder(entry_block)
        self.current_function = func
        
        # Compile function body
        if len(node.children) > 0:
            body = node.children[-1]
            self.compile_node(body)
        
        # Add return statement if function returns int
        if func_name == "main":
            self.builder.ret(ir.Constant(self.int_type, 0))
        else:
            self.builder.ret_void()
        
        self.current_function = None
        self.builder = None
    
    def compile_variable(self, node: ASTNode):
        """Compile variable declaration"""
        var_name = node.value.replace("mut ", "")
        
        # Create alloca instruction
        if self.current_function:
            var_ptr = self.builder.alloca(self.int_type, name=var_name)
            self.variables[var_name] = var_ptr
            
            # Initialize if provided
            if len(node.children) > 0:
                initializer = self.compile_node(node.children[-1])
                self.builder.store(initializer, var_ptr)
    
    def compile_block(self, node: ASTNode):
        """Compile block of statements"""
        for child in node.children:
            self.compile_node(child)
    
    def compile_assignment(self, node: ASTNode):
        """Compile assignment expression"""
        if len(node.children) != 2:
            self.error("Assignment must have exactly 2 children", node)
        
        target, value = node.children
        
        # Compile the value
        compiled_value = self.compile_node(value)
        
        # Store the value in the variable
        if target.type == NodeType.IDENTIFIER:
            var_name = target.value
            var_ptr = self.variables.get(var_name)
            if var_ptr is None:
                self.error(f"Undefined variable: {var_name}", target)
            self.builder.store(compiled_value, var_ptr)
        else:
            self.error("Assignment target must be an identifier", target)
    
    def compile_binary(self, node: ASTNode) -> ir.Value:
        """Compile binary expression"""
        if len(node.children) != 2:
            self.error("Binary expression must have exactly 2 children", node)
        
        left = self.compile_node(node.children[0])
        right = self.compile_node(node.children[1])
        
        # Emit appropriate operation
        op_map = {
            "+": self.builder.add,
            "-": self.builder.sub,
            "*": self.builder.mul,
            "/": self.builder.sdiv,
            "==": self.builder.icmp_signed,
            "!=": lambda l, r: self.builder.icmp_signed("!=", l, r),
            "<": lambda l, r: self.builder.icmp_signed("<", l, r),
            ">": lambda l, r: self.builder.icmp_signed(">", l, r),
            "<=": lambda l, r: self.builder.icmp_signed("<=", l, r),
            ">=": lambda l, r: self.builder.icmp_signed(">=", l, r),
        }
        
        op_func = op_map.get(node.value)
        if op_func is None:
            self.error(f"Unknown binary operator: {node.value}", node)
        
        return op_func(left, right)
    
    def compile_function_call(self, node: ASTNode) -> ir.Value:
        """Compile function call"""
        func_name = node.value
        
        # Handle built-in print function
        if func_name == "print":
            if len(node.children) > 0:
                arg = self.compile_node(node.children[0])
                # Convert to string and call printf
                format_str = "%d\n"
                format_ptr = self._create_string_constant(format_str)
                return self.builder.call(self.printf_func, [format_ptr, arg])
            return ir.Constant(self.int_type, 0)
        
        # Call user-defined function
        func = self.functions.get(func_name)
        if func is None:
            self.error(f"Undefined function: {func_name}", node)
        
        args = [self.compile_node(arg) for arg in node.children]
        return self.builder.call(func, args)
    
    def compile_literal(self, node: ASTNode) -> ir.Value:
        """Compile literal value"""
        if node.value == "number":
            num_str = node.children[0].value
            if '.' in num_str:
                return ir.Constant(self.float_type, float(num_str))
            else:
                return ir.Constant(self.int_type, int(num_str))
        elif node.value == "string":
            str_value = node.children[0].value
            # Remove quotes
            if str_value.startswith('"') and str_value.endswith('"'):
                str_value = str_value[1:-1]
            return self._create_string_constant(str_value)
        elif node.value == "boolean":
            bool_value = node.children[0].value.lower() == "true"
            return ir.Constant(self.bool_type, int(bool_value))
        elif node.value == "null":
            return ir.Constant(self.int_type, 0)
        else:
            # Direct value
            if isinstance(node.value, int):
                return ir.Constant(self.int_type, node.value)
            elif isinstance(node.value, float):
                return ir.Constant(self.float_type, node.value)
            elif isinstance(node.value, bool):
                return ir.Constant(self.bool_type, int(node.value))
            elif node.value is None:
                return ir.Constant(self.int_type, 0)
            else:
                self.error(f"Unsupported literal type: {type(node.value)}", node)
    
    def compile_identifier(self, node: ASTNode) -> ir.Value:
        """Compile identifier access"""
        var_name = node.value
        var_ptr = self.variables.get(var_name)
        
        if var_ptr is None:
            self.error(f"Undefined variable: {var_name}", node)
        
        return self.builder.load(var_ptr)
    
    def _create_string_constant(self, string_value: str) -> ir.Value:
        """Create a global string constant"""
        string_bytes = string_value.encode('utf-8') + b'\x00'
        string_type = ir.ArrayType(ir.IntType(8), len(string_bytes))
        
        # Create global variable
        global_var = ir.GlobalVariable(self.module, string_type, name=f".str_{len(self.constants)}")
        global_var.global_constant = True
        global_var.linkage = 'internal'
        global_var.initializer = ir.Constant(string_type, bytearray(string_bytes))
        
        self.constants.append(global_var)
        
        # Return pointer to first character
        zero = ir.Constant(self.int_type, 0)
        return self.builder.gep(global_var, [zero, zero])
    
    def _create_main_function(self):
        """Create default main function"""
        main_type = ir.FunctionType(self.int_type, ())
        main_func = ir.Function(self.module, main_type, "main")
        self.functions["main"] = main_func
        
        entry_block = main_func.append_basic_block("entry")
        builder = ir.IRBuilder(entry_block)
        builder.ret(ir.Constant(self.int_type, 0))
    
    def print_llvm_ir(self):
        """Debug method to print LLVM IR"""
        print("=== LLVM IR ===")
        print(str(self.module))

def compile_to_binary(ir_code: str, output_file: str = "output.exe") -> str:
    """Compile LLVM IR to native binary using TargetMachine"""
    try:
        # Initialize native targets for binary compilation
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()
        
        # Parse LLVM IR
        mod = llvm.parse_assembly(ir_code)
        mod.verify()
        
        # Create target machine
        target = llvm.Target.from_default_triple()
        target_machine = target.create_target_machine(
            opt=2,  # Optimization level
            codemodel='small'
        )
        
        # Generate object file
        obj_file = output_file.replace('.exe', '.obj')
        with open(obj_file, 'wb') as f:
            f.write(target_machine.emit_object(mod))
        
        # Link to executable (simplified - would need proper linker setup)
        # For now, return the object file path
        print(f"Generated object file: {obj_file}")
        print(f"To create executable, link with: clang {obj_file} -o {output_file}")
        
        return obj_file
        
    except Exception as e:
        raise RuntimeError(f"Binary compilation failed: {str(e)}")

def main():
    """Test the compiler with LLVM native compilation"""
    from lexer import KorlanLexer, LexerError
    from parser import KorlanParser, ParserError
    from checker import StaticAnalyzer
    
    sample_code = '''
fun main() {
    mut x = 42
    print(x)
    x = x + 8
    print(x)
}
'''

    print("=== Korlan LLVM Compiler Test ===")
    print("Input code:")
    print(sample_code)
    
    # Lex, parse, and check semantics
    lexer = KorlanLexer(sample_code)
    tokens = lexer.tokenize()
    
    parser = KorlanParser(tokens)
    ast = parser.parse()
    
    # Semantic analysis
    analyzer = StaticAnalyzer()
    if not analyzer.check(ast):
        print("Semantic analysis failed:")
        analyzer.print_errors()
        return
    
    print("\n✅ Semantic analysis passed")
    
    # Test both bytecode and LLVM compilation
    print("\n=== Bytecode Compilation ===")
    bytecode_compiler = KorlanCompiler()
    bytecode = bytecode_compiler.compile(ast)
    bytecode_compiler.print_bytecode()
    
    print("\n=== LLVM IR Compilation ===")
    llvm_emitter = LLVMEmitter()
    try:
        llvm_ir = llvm_emitter.compile_to_llvm_ir(ast)
        print("Generated LLVM IR:")
        print(llvm_ir)
        
        # Test binary compilation
        print("\n=== Binary Compilation ===")
        obj_file = compile_to_binary(llvm_ir, "korlan_test.exe")
        print(f"✅ Object file generated: {obj_file}")
        
    except Exception as e:
        print(f"❌ LLVM compilation failed: {e}")

if __name__ == "__main__":
    main()
