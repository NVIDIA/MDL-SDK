#*****************************************************************************
# Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#*****************************************************************************

"""MDL Python Bindings - a high-level wrapper for a more efficient and user-friendly usage. 

Note, this is an experimental library to simply a few of the use cases of the MDL SDK.
For a complete access to the MDL SDK API, please refer to the low-level binding, `pymdlsdk.py`.
"""

# low level MDL Python Binding
if __package__ or "." in __name__:
    from . import pymdlsdk
else:
    import pymdlsdk

try:
    # used to represent Vectors, Matrices, and Colors
    import numpy
except ImportError as e:
    print("`numpy` is not installed but required by pymdl.")
    raise e

# Type annotation improves type safety and supports code completing in IDEs like VS Code
from typing import List, Set, Dict, Tuple, FrozenSet, Optional

#--------------------------------------------------------------------------------------------------
# Value Conversions
#--------------------------------------------------------------------------------------------------

def DowncastIType(itype : pymdlsdk.IType) -> pymdlsdk.IType: # or derived type
    r"""
    Cast the input type into the proper derived interface depending on the kind of type.

    Here is the mapping from the type kind to the returned interface:

    - TK_ALIAS --> IType_alias

    - TK_BOOL --> IType_bool

    - TK_INT --> IType_int

    - TK_ENUM --> IType_enumeration

    - TK_FLOAT --> IType_float

    - TK_DOUBLE --> IType_double

    - TK_STRING --> IType_string

    - TK_VECTOR --> IType_vector

    - TK_MATRIX --> IType_matrix

    - TK_COLOR --> IType_color

    - TK_ARRAY --> IType_array

    - TK_STRUCT --> IType_structure

    - TK_TEXTURE --> IType_texture

    - TK_LIGHT_PROFILE --> IType_light_profile

    - TK_BSDF_MEASUREMENT --> IType_bsdf_measurement

    - TK_BSDF --> IType_bsdf

    - TK_HAIR_BSDF --> IType_hair_bsdf

    - TK_EDF --> IType_edf

    - TK_VDF --> IType_vdf
    """
    if not itype.is_valid_interface():
        return itype
    k = itype.get_kind()
    t = None
    if   k == pymdlsdk.IType.Kind.TK_ALIAS:              t = itype.get_interface(pymdlsdk.IType_alias)
    elif k == pymdlsdk.IType.Kind.TK_BOOL:               t = itype.get_interface(pymdlsdk.IType_bool)
    elif k == pymdlsdk.IType.Kind.TK_INT:                t = itype.get_interface(pymdlsdk.IType_int)
    elif k == pymdlsdk.IType.Kind.TK_ENUM:               t = itype.get_interface(pymdlsdk.IType_enumeration)
    elif k == pymdlsdk.IType.Kind.TK_FLOAT:              t = itype.get_interface(pymdlsdk.IType_float)
    elif k == pymdlsdk.IType.Kind.TK_DOUBLE:             t = itype.get_interface(pymdlsdk.IType_double)
    elif k == pymdlsdk.IType.Kind.TK_STRING:             t = itype.get_interface(pymdlsdk.IType_string)
    elif k == pymdlsdk.IType.Kind.TK_VECTOR:             t = itype.get_interface(pymdlsdk.IType_vector)
    elif k == pymdlsdk.IType.Kind.TK_MATRIX:             t = itype.get_interface(pymdlsdk.IType_matrix)
    elif k == pymdlsdk.IType.Kind.TK_COLOR:              t = itype.get_interface(pymdlsdk.IType_color)
    elif k == pymdlsdk.IType.Kind.TK_ARRAY:              t = itype.get_interface(pymdlsdk.IType_array)
    elif k == pymdlsdk.IType.Kind.TK_STRUCT:             t = itype.get_interface(pymdlsdk.IType_structure)
    elif k == pymdlsdk.IType.Kind.TK_TEXTURE:            t = itype.get_interface(pymdlsdk.IType_texture)
    elif k == pymdlsdk.IType.Kind.TK_LIGHT_PROFILE:      t = itype.get_interface(pymdlsdk.IType_light_profile)
    elif k == pymdlsdk.IType.Kind.TK_BSDF_MEASUREMENT:   t = itype.get_interface(pymdlsdk.IType_bsdf_measurement)
    elif k == pymdlsdk.IType.Kind.TK_BSDF:               t = itype.get_interface(pymdlsdk.IType_bsdf)
    elif k == pymdlsdk.IType.Kind.TK_HAIR_BSDF:          t = itype.get_interface(pymdlsdk.IType_hair_bsdf)
    elif k == pymdlsdk.IType.Kind.TK_EDF:                t = itype.get_interface(pymdlsdk.IType_edf)
    elif k == pymdlsdk.IType.Kind.TK_VDF:                t = itype.get_interface(pymdlsdk.IType_vdf)
    itype = None
    return t

def DowncastIExpression(iexpression : pymdlsdk.IExpression) -> pymdlsdk.IExpression: # or derived type
    r"""
    Cast the input expression into the proper derived interface depending on the kind of expression.

    Here is the mapping from the expression kind to the returned interface:

    - EK_CONSTANT --> IExpression_constant
    
    - EK_CALL --> IExpression_call
    
    - EK_PARAMETER --> IExpression_parameter
    
    - EK_DIRECT_CALL --> IExpression_direct_call
    
    - EK_TEMPORARY --> IExpression_temporary
    """
    if not iexpression.is_valid_interface():
        return iexpression
    k = iexpression.get_kind()
    if   k == pymdlsdk.IExpression.Kind.EK_CONSTANT:     t = iexpression.get_interface(pymdlsdk.IExpression_constant)
    elif k == pymdlsdk.IExpression.Kind.EK_CALL:         t = iexpression.get_interface(pymdlsdk.IExpression_call)
    elif k == pymdlsdk.IExpression.Kind.EK_PARAMETER:    t = iexpression.get_interface(pymdlsdk.IExpression_parameter)
    elif k == pymdlsdk.IExpression.Kind.EK_DIRECT_CALL:  t = iexpression.get_interface(pymdlsdk.IExpression_direct_call)
    elif k == pymdlsdk.IExpression.Kind.EK_TEMPORARY:    t = iexpression.get_interface(pymdlsdk.IExpression_temporary)
    iexpression = None
    return t

def DowncastIValue(ivalue : pymdlsdk.IValue) -> pymdlsdk.IValue: # or derived type
    r"""
    Cast the input value into the proper derived interface depending on the kind of value.

    Here is the mapping from the ivalue kind to the returned interface:

    - VK_BOOL -->               IValue_bool
    
    - VK_INT -->                IValue_int

    - VK_ENUM -->               IValue_enumeration

    - VK_FLOAT -->              IValue_float

    - VK_DOUBLE -->             IValue_double

    - VK_STRING -->             IValue_string

    - VK_VECTOR -->             IValue_vector

    - VK_MATRIX -->             IValue_matrix

    - VK_COLOR -->              IValue_color

    - VK_ARRAY -->              IValue_array

    - VK_STRUCT -->             IValue_structure

    - VK_INVALID_DF -->         IValue_invalid_df

    - VK_TEXTURE -->            IValue_texture

    - VK_LIGHT_PROFILE -->      IValue_light_profile

    - VK_BSDF_MEASUREMENT -->   IValue_bsdf_measurement
    """
    if not ivalue.is_valid_interface():
        return ivalue
    k = ivalue.get_kind()
    if   k == pymdlsdk.IValue.Kind.VK_BOOL:               t = ivalue.get_interface(pymdlsdk.IValue_bool)
    elif k == pymdlsdk.IValue.Kind.VK_INT:                t = ivalue.get_interface(pymdlsdk.IValue_int)
    elif k == pymdlsdk.IValue.Kind.VK_ENUM:               t = ivalue.get_interface(pymdlsdk.IValue_enumeration)
    elif k == pymdlsdk.IValue.Kind.VK_FLOAT:              t = ivalue.get_interface(pymdlsdk.IValue_float)
    elif k == pymdlsdk.IValue.Kind.VK_DOUBLE:             t = ivalue.get_interface(pymdlsdk.IValue_double)
    elif k == pymdlsdk.IValue.Kind.VK_STRING:             t = ivalue.get_interface(pymdlsdk.IValue_string)
    elif k == pymdlsdk.IValue.Kind.VK_VECTOR:             t = ivalue.get_interface(pymdlsdk.IValue_vector)
    elif k == pymdlsdk.IValue.Kind.VK_MATRIX:             t = ivalue.get_interface(pymdlsdk.IValue_matrix)
    elif k == pymdlsdk.IValue.Kind.VK_COLOR:              t = ivalue.get_interface(pymdlsdk.IValue_color)
    elif k == pymdlsdk.IValue.Kind.VK_ARRAY:              t = ivalue.get_interface(pymdlsdk.IValue_array)
    elif k == pymdlsdk.IValue.Kind.VK_STRUCT:             t = ivalue.get_interface(pymdlsdk.IValue_structure)
    elif k == pymdlsdk.IValue.Kind.VK_INVALID_DF:         t = ivalue.get_interface(pymdlsdk.IValue_invalid_df)
    elif k == pymdlsdk.IValue.Kind.VK_TEXTURE:            t = ivalue.get_interface(pymdlsdk.IValue_texture)
    elif k == pymdlsdk.IValue.Kind.VK_LIGHT_PROFILE:      t = ivalue.get_interface(pymdlsdk.IValue_light_profile)
    elif k == pymdlsdk.IValue.Kind.VK_BSDF_MEASUREMENT:   t = ivalue.get_interface(pymdlsdk.IValue_bsdf_measurement)
    ivalue = None
    return t

def IValueToPyValues(ivalue : pymdlsdk.IValue):
    r"""
    Converts low level IValues to python friendly data types.

    Here is the mapping from the ivalue kind to the returned values:
    
    - VK_BOOL --> ivalue.get_value()

    - VK_INT --> ivalue.get_value()

    - VK_FLOAT --> ivalue.get_value()

    - VK_DOUBLE --> ivalue.get_value()

    - VK_STRING --> ivalue.get_value()

    - VK_LIGHT_PROFILE --> ivalue.get_value()

    - VK_BSDF_MEASUREMENT --> ivalue.get_value()
    
    - VK_ENUM --> (ivalue.get_name(), ivalue.get_value())

    - VK_TEXTURE --> (ivalue.get_value(), ivalue.get_gamma())

    - VK_COLOR --> numpy.array(color components)

    - VK_ARRAY --> [IValueToPyValues(array values)]

    - VK_STRUCT --> dict(structure field names, ArgumentConstant(structure fields))

    - VK_INVALID_DF --> None
    
    For vectors (kind is VK_VECTOR), each value is converted using IValueToPyValues().
    A numpy.array is returned with type depending on the vector element kind.
    Here is the mapping between the vector element kind and the array type.

    - TK_FLOAT --> numpy.float32

    - TK_DOUBLE --> numpy.float64

    - TK_INT --> numpy.int32

    - TK_BOOL --> bool

    For matrices (kind is VK_MATRIX), each value is converted using IValueToPyValues().
    A numpy.array is returned with type depending on the vector element kind.
    Here is the mapping between the matrix element kind and the array type.

    - TK_FLOAT --> numpy.float32

    - TK_DOUBLE --> numpy.float64

    - TK_INT --> numpy.int32

    - TK_BOOL --> bool
    """
    ivalue = DowncastIValue(ivalue)
    kind = ivalue.get_kind()
    if   kind == pymdlsdk.IValue.Kind.VK_BOOL or \
         kind == pymdlsdk.IValue.Kind.VK_INT or \
         kind == pymdlsdk.IValue.Kind.VK_FLOAT or \
         kind == pymdlsdk.IValue.Kind.VK_DOUBLE or \
         kind == pymdlsdk.IValue.Kind.VK_STRING:
        return ivalue.get_value()

    elif kind == pymdlsdk.IValue.Kind.VK_ENUM:
        name = ivalue.get_name()
        code = ivalue.get_value()
        return (name,  code)

    elif kind == pymdlsdk.IValue.Kind.VK_VECTOR:
        element_kind = ivalue.get_type().get_element_type().get_kind()
        if element_kind == pymdlsdk.IType.Kind.TK_FLOAT:
            dtype = numpy.float32
        elif element_kind == pymdlsdk.IType.Kind.TK_DOUBLE:
            dtype = numpy.float64
        elif element_kind == pymdlsdk.IType.Kind.TK_INT:
            dtype = numpy.int32      
        elif element_kind == pymdlsdk.IType.Kind.TK_BOOL:
            dtype = bool

        components = []
        rows = ivalue.get_size()
        for i in range(rows):
            components.append(IValueToPyValues(ivalue.get_value(i)))
        return numpy.array(components, dtype = dtype).reshape(rows, 1)

    elif kind == pymdlsdk.IValue.Kind.VK_MATRIX:
        col_type = DowncastIType(ivalue.get_type().get_element_type())
        rows = col_type.get_size()
        cols = ivalue.get_size()
        col_kind = col_type.get_kind()
        if col_kind == pymdlsdk.IType.Kind.TK_VECTOR:
            element_kind = col_type.get_component_type(0).get_kind()
            if element_kind == pymdlsdk.IType.Kind.TK_FLOAT:
                dtype = numpy.float32
            elif element_kind == pymdlsdk.IType.Kind.TK_DOUBLE:
                dtype = numpy.float64
            elif element_kind == pymdlsdk.IType.Kind.TK_INT:
                dtype = numpy.int32      
            elif element_kind == pymdlsdk.IType.Kind.TK_BOOL:
                dtype = bool

        components = []
        for c in range(cols):
            column = ivalue.get_value(c)
            for r in range(rows):
                components.append(IValueToPyValues(column.get_value(r)))
        return numpy.transpose(numpy.array(components, dtype = dtype).reshape(cols, rows))

    elif kind == pymdlsdk.IValue.Kind.VK_COLOR:
        components = []
        for i in range(ivalue.get_size()):
            components.append(ivalue.get_value(i).get_interface(pymdlsdk.IValue_float).get_value())
        return numpy.array(components)

    elif kind == pymdlsdk.IValue.Kind.VK_ARRAY:
        array = []
        for i in range(ivalue.get_size()):
            elem = ivalue.get_value(i)
            array.append(IValueToPyValues(elem))
        return array

    elif kind == pymdlsdk.IValue.Kind.VK_STRUCT:
        struct = {}
        type = DowncastIType(ivalue.get_type())
        for e in range(type.get_size()):
            e_name = type.get_field_name(e)
            e_value = ivalue.get_field(e_name)
            struct[e_name] = ArgumentConstant(e_value)
        return struct

    elif kind == pymdlsdk.IValue.Kind.VK_INVALID_DF:
        return None
        
    elif kind == pymdlsdk.IValue.Kind.VK_TEXTURE:
        return (ivalue.get_value(), ivalue.get_gamma(), ivalue.get_file_path())

    elif kind == pymdlsdk.IValue.Kind.VK_LIGHT_PROFILE or \
        kind == pymdlsdk.IValue.Kind.VK_BSDF_MEASUREMENT:
        return (ivalue.get_value(), ivalue.get_file_path())

    return None

#--------------------------------------------------------------------------------------------------
# High Level Wrappers
#--------------------------------------------------------------------------------------------------

class Type(object):
    r"""
    Wrapper around MDL type.

    Type gives access to:

    kind: Kind
        The kind of type.

    uniform: bool
        True if the type has a uniform modifier.

    varying: bool
        True if the type has a varying modifier.

    symbol: str
        If type is enum or struct, this is the qualified name of the enum or struct type.
        Otherwise this is None.

    name: str
        In case of enums or structs, this is equal to the symbol name. Scalar, vector, matrix,
        array and resource types that don't have a symbol name can use this type name.

    enumValues: (str,int)
        If type is enum, this list of pairs holds the enum values along with their numeric value.
        Otherwise this is empty.

    size: int
        In case of structs, vectors, matrices, arrays, colors, this is the compound size.

    element_type: Type
        In case of vectors, arrays this is the type of the elements.
    """
    def __init__(self, itype: pymdlsdk.IType) -> None:
        super().__init__()
        
        modifierMask = itype.get_all_type_modifiers()
        self.uniform = (modifierMask & 2) > 0
        self.varying = (modifierMask & 4) > 0

        # drops all modifiers
        itype = DowncastIType(itype.skip_all_type_aliases())
        self.kind = itype.get_kind()
        self.symbol = None
        self.size = 0
        self.element_type = None

        self.enumValues = []
        if self.kind == pymdlsdk.IType.Kind.TK_ENUM:
            self.symbol = itype.get_symbol()
            for i in range(itype.get_size()):
                name = itype.get_value_name(i)
                code = itype.get_value_code(i)
                self.enumValues.append((name, code))

        elif self.kind == pymdlsdk.IType.Kind.TK_STRUCT:
            self.symbol = itype.get_symbol()
            self.size = itype.get_size()

        elif self.kind == pymdlsdk.IType.Kind.TK_VECTOR:
            self.size = itype.get_size()
            self.element_type = Type(itype.get_element_type())
        
        elif self.kind == pymdlsdk.IType.Kind.TK_MATRIX:
            self.size = itype.get_size()

        elif self.kind == pymdlsdk.IType.Kind.TK_ARRAY:
            self.size = itype.get_size()
            self.element_type = Type(itype.get_element_type())

        elif self.kind == pymdlsdk.IType.Kind.TK_COLOR:
            self.size = itype.get_size()

        self.name = self.__printName(itype)

    r"""
    Helper function to print a type.
    Note, this will be moved into the MDL SDK soon.
    """
    def __printName(self, itype: pymdlsdk.IType) -> str:
        itype = DowncastIType(itype)
        if itype.get_kind() == pymdlsdk.IType.Kind.TK_BOOL:
            return 'bool'

        if itype.get_kind() == pymdlsdk.IType.Kind.TK_INT:
            return 'int'

        if itype.get_kind() == pymdlsdk.IType.Kind.TK_FLOAT:
            return 'float'

        if itype.get_kind() == pymdlsdk.IType.Kind.TK_DOUBLE:
            return 'double'

        if itype.get_kind() == pymdlsdk.IType.Kind.TK_STRING:
            return 'string'

        if itype.get_kind() == pymdlsdk.IType.Kind.TK_COLOR:
            return 'color'

        if itype.get_kind() == pymdlsdk.IType.Kind.TK_LIGHT_PROFILE:
            return 'light_profile'

        if itype.get_kind() == pymdlsdk.IType.Kind.TK_BSDF_MEASUREMENT:
            return 'bsdf_measurement'

        if itype.get_kind() == pymdlsdk.IType.Kind.TK_ENUM or \
           itype.get_kind() == pymdlsdk.IType.Kind.TK_STRUCT:
            return itype.get_symbol()

        if itype.get_kind() == pymdlsdk.IType.Kind.TK_VECTOR:
            with itype.get_element_type() as elementType:
                return self.__printName(elementType) + str(itype.get_size())

        if itype.get_kind() == pymdlsdk.IType.Kind.TK_MATRIX:
            with itype.get_element_type() as rowType, \
                 DowncastIType(rowType) as rowTypeVector, \
                 rowTypeVector.get_element_type() as elementType:
                    return self.__printName(elementType) + str(itype.get_size()) + 'x' + str(rowType.get_size())

        if itype.get_kind() == pymdlsdk.IType.Kind.TK_ARRAY:
            with itype.get_element_type() as elementType:
                if itype.is_immediate_sized():
                    return self.__printName(elementType) + '[' + str(itype.get_size()) + ']'
                else:
                    return self.__printName(elementType) + '[N]'

        if itype.get_kind() == pymdlsdk.IType.Kind.TK_TEXTURE:
            shape = itype.get_shape()
            if shape == pymdlsdk.IType_texture.Shape.TS_2D:
                return 'texture_2d'
            elif shape == pymdlsdk.IType_texture.Shape.TS_3D:
                return 'texture_3d'
            elif shape == pymdlsdk.IType_texture.Shape.TS_CUBE:
                return 'texture_cube'
            elif shape == pymdlsdk.IType_texture.Shape.TS_PTEX:
                return 'texture_ptex'
            elif shape == pymdlsdk.IType_texture.Shape.TS_BSDF_DATA:
                return 'texture_bsdf_data'

class Argument(object):
    r"""
    Wrapper around an MDL parameter.
    Argument combines type information, annotations, and optionally the value of an MDL parameter.
    
    Argument is used:
    
    - To describe the parameters of a FunctionCall. The value of the Argument is the value of the corresponding function call parameter.

    - To describe the parameters of a FunctionDefinition. In this case, the value of the Argument is the default value of the corresponding definition parameter if applicable (i.e. default value exists.)

    - To describe the Parameters of an Annotation along with its value.

    - To describe return types of functions and their annotations. In this case, the value is not used.

    There are two kinds of Argument: ArgumentConstant and ArgumentCall.
    ArgumentConstant holds an actual value.
    ArgumentCall refers to other function calls which allows to construct expression graphs.

    Argument gives access to:

    type: Type
        The parameter type.

    annotations: tuple(Annotation)
        The parameter annotations.
    """
    def __init__(self, type: Type, annotations: tuple() = tuple()):
        self.type = type
        self.value = None
        self.annotations = annotations

class ArgumentConstant(Argument):
    r"""
    Wrapper around an MDL parameter value.

    Type gives access to:

    value: See IValueToPyValues()
        Holds an actual parameter value.
        The MDL value is transformed using IValueToPyValues()
    """
    def __init__(self, ivalue: pymdlsdk.IValue, annotations: tuple() = tuple()):
        super(ArgumentConstant, self).__init__(Type(ivalue.get_type()), annotations)
        self.value = IValueToPyValues(ivalue)

class ArgumentCall(Argument):
    r"""
    Wrapper around an MDL FunctionCall parameter.

    ArgumentCall gives access to:

    value: str
        The DB name of the referenced function call.
    """
    def __init__(self, iexpression: pymdlsdk.IExpression_call, annotations = tuple()):
        super(ArgumentCall, self).__init__(Type(iexpression.get_type()), annotations)
        self.value = iexpression.get_call()

#--------------------------------------------------------------------------------------------------
# High Level Wrappers - Annotations
#--------------------------------------------------------------------------------------------------

class Annotation(object):
    """Wrapper around an MDL annotation."""
    def __init__(self, iannotation: pymdlsdk.IAnnotation):
        arguments = {}
        with iannotation.get_arguments() as expr_list:
            for i in range(expr_list.get_size()):
                arg_name = expr_list.get_name(i)
                with expr_list.get_expression(arg_name) as arg_expr:
                    if arg_expr.get_kind() == pymdlsdk.IExpression.Kind.EK_CONSTANT:
                        arguments[arg_name] = ArgumentConstant(arg_expr.get_interface(pymdlsdk.IExpression_constant).get_value())
                    else:
                        pass # should not happen

        self._name = iannotation.get_name() #TODO make immutable
        self._arguments = arguments

        paramTypeNames = []
        with iannotation.get_definition() as definition:
            for p in range(definition.get_parameter_count()):
                paramTypeNames.append(definition.get_mdl_parameter_type_name(p))

            self._simpleName = definition.get_mdl_simple_name()
            self._parameterTypeNames = tuple(paramTypeNames)
            self._moduleDbName = definition.get_module()

    @property
    def name(self): 
        """The full name of the annotation consisting of the module name, the simple name and the parameter list."""
        return self._name

    @property
    def moduleDbName(self)-> str:
        """The database of the module in which the annotation is defined."""
        return self._moduleDbName

    @property
    def simpleName(self)-> str:
        """The name of the annotation within the module it is defined in without its parameter list."""
        return self._simpleName

    @property
    def parameterTypeNames(self) -> List[str]:
        """The list of parameter type names of the signature of this annotation."""
        return self._parameterTypeNames

    @property
    def arguments(self) -> Dict[str, Argument]: 
        """Dictionary with the annotations arguments and their values."""
        return self._arguments


def AnnotationBlock(iannotation_block: pymdlsdk.IAnnotation_block) -> Tuple[Annotation, ...]:
    r"""
    Creates an immutable list of annotations from an MDL annotation block. 

    Parameters
    ----------
    iannotation_block : pymdlsdk.IAnnotation_block
        Low level MDL annotation block proxy generated for the Python binding.

    Returns
    -------
    tuple(Annotation)
        A list of high level wrapped annotation information read from the annotation block.
    """
    annos = []
    if iannotation_block and iannotation_block.is_valid_interface():
        for i in range(iannotation_block.get_size()) :
            with iannotation_block.get_annotation(i) as anno:
                if anno.is_valid_interface():
                    a = Annotation(anno)
                    annos.append(a)
    return tuple(annos)


#--------------------------------------------------------------------------------------------------
# High Level Wrappers - Functions
#--------------------------------------------------------------------------------------------------

class FunctionCall(object):
    r"""
    Wrapper around MDL function call.

    FunctionCall gives access to:

    functionDefinition: str
        The DB name of the corresponding function definition.

    mdlFunctionDefinition: str
        The MDL name of the corresponding function definition.
    
    parameters: Dict[str, Argument]
        Dictionary of the function call parameters as Argument.
        Key is parameter name corresponding to the Argument.
    """
    def __init__(self, func: pymdlsdk.IFunction_call, dbName: str) -> None:
        super(FunctionCall, self).__init__()

        parameters = {}
        param_anno_block = None
        with func.get_arguments() as arguments, \
            func.get_parameter_types() as param_types:

            for i in range(func.get_parameter_count()):
                param_name = func.get_parameter_name(i)

                with arguments.get_expression(i) as param_default:
                    if param_default.is_valid_interface():
                        if param_default.get_kind() == pymdlsdk.IExpression.Kind.EK_CONSTANT:
                            with param_default.get_interface(pymdlsdk.IExpression_constant) as param_default_constant, \
                                 param_default_constant.get_value() as param_default_value:
                                parameters[param_name] = ArgumentConstant(param_default_value, AnnotationBlock(param_anno_block))
                        elif param_default.get_kind() == pymdlsdk.IExpression.Kind.EK_CALL:
                            parameters[param_name] = ArgumentCall(param_default.get_interface(pymdlsdk.IExpression_call), AnnotationBlock(param_anno_block))
                    else:
                        with param_types.get_type(param_name) as param_default_type:
                            parameters[param_name] = Argument(Type(param_default_type), AnnotationBlock(param_anno_block))
                param_anno_block = None

        self.parameters: Dict[Argument]
        self.parameters = parameters #TODO make immutable

        self.functionDefinition = func.get_function_definition()
        self.mdlFunctionDefinition = func.get_mdl_function_definition()

    @staticmethod
    def _fetchFromDb(transaction: pymdlsdk.ITransaction, dbName) -> "FunctionCall":
        with transaction.access_as(pymdlsdk.IFunction_call, dbName) as f:
            if f.is_valid_interface():
                return FunctionCall(f, dbName)
        return None

class FunctionDefinition(object):
    r"""
    Wrapper around MDL function definition.
    
    FunctionDefinition gives access to:

    annotations: AnnotationBlock
        The annotations of the function definition itself, or None if there are no such annotations.

    dbName: str
        DB name of the function definitions.

    mdlModuleName:str
        The MDL name of the module containing this function definition.

    mdlName: str
        The MDL name of the function definition.

    mdlSimpleName: str
        The simple MDL name of the function definition.
        The simple name is the last component of the MDL name, i.e., without any packages and scope qualifiers, and without the parameter type names.

    moduleDbName: str
        The DB name of the module containing this function definition.

    isExported: bool
        Indicates whether the function definition is exported by its module.

    isMaterial: bool
        True in case the function is a material, i.e., the return type is a material structure.

    semantics: IFunction_definition.Semantics
        Get the semantic of known function. User-defined functions an all materials will return DS_UNKNOWN (0).
        Note, do not rely on the numeric values of the enumerators since they may change without further notice.

    parameters: Dict[str, Argument]
        Dictionary of the function definition parameters as Argument.
        Key is parameter name corresponding to the Argument.

    parameterTypeNames: tuple(str)
        The type name of all the parameters.

    returnValue: Argument
        The return type as an Argument.
    """
    def __init__(self, func: pymdlsdk.IFunction_definition, dbName: str) -> None:
        super(FunctionDefinition, self).__init__()

        parameters = {}
        with func.get_parameter_annotations() as param_annotations, \
            func.get_defaults() as param_defaults, \
            func.get_parameter_types() as param_types:
            paramTypeNames = []

            for i in range(func.get_parameter_count()):
                param_name = func.get_parameter_name(i)
                paramTypeNames.append(func.get_mdl_parameter_type_name(i))

                if param_annotations.is_valid_interface():
                    param_anno_block = param_annotations.get_annotation_block(param_name)
                else:
                    param_anno_block = None

                with param_defaults.get_expression(param_name) as param_default:
                    if param_default.is_valid_interface():
                        if param_default.get_kind() == pymdlsdk.IExpression.Kind.EK_CONSTANT:
                            with param_default.get_interface(pymdlsdk.IExpression_constant) as param_default_constant, \
                                 param_default_constant.get_value() as param_default_value:
                                parameters[param_name] = ArgumentConstant(param_default_value, AnnotationBlock(param_anno_block))
                        elif param_default.get_kind() == pymdlsdk.IExpression.Kind.EK_CALL:
                            parameters[param_name] = ArgumentCall(param_default.get_interface(pymdlsdk.IExpression_call), AnnotationBlock(param_anno_block))
                    else:
                        with param_types.get_type(param_name) as param_default_type:
                            parameters[param_name] = Argument(Type(param_default_type), AnnotationBlock(param_anno_block))
                param_anno_block = None

        self.isMaterial = func.is_material()
        self.isExported = func.is_exported()
        self.semantics = func.get_semantic()
        with func.get_return_type() as ret_type, \
            func.get_return_annotations() as ret_annos:
            self.returnValue = Argument(Type(ret_type), AnnotationBlock(ret_annos))

        self.parameters: Dict[Argument]
        self.parameters = parameters #TODO make immutable

        with func.get_annotations() as func_annos:
            self.annotations = AnnotationBlock(func_annos)
        
        self.mdlName = func.get_mdl_name()
        self.dbName = dbName

        self.mdlSimpleName = func.get_mdl_simple_name()
        self.parameterTypeNames = tuple(paramTypeNames)

        self.mdlModuleName = func.get_mdl_module_name()
        self.moduleDbName = func.get_module()

    @staticmethod
    def _fetchFromDb(transaction: pymdlsdk.ITransaction, dbName) -> "FunctionDefinition":
        with transaction.access_as(pymdlsdk.IFunction_definition, dbName) as f:
            if f.is_valid_interface():
                return FunctionDefinition(f, dbName)
        return None

#--------------------------------------------------------------------------------------------------
# High Level Wrappers - Modules
#--------------------------------------------------------------------------------------------------

class Module(object):
    r"""
    Wrapper around an MDL module.

    Module gives access to:

    dbName: str
        The DB name of the module.

    filename: str
        The name of the MDL source file from which the module was created.
    
    functionDbNames: tuple(str)
        DB names of all the function definitions and the material definitions from this module.

    functions: Dict[str, FunctionDefinition]
        Dictionary of FunctionDefinition.
        Keys is FunctionDefinition simple name.

    mdlName: str
        The MDL name of the module.

    mdlSimpleName: str 
        The simple MDL name of the module.
        The simple name is the last component of the MDL name, i.e., without any packages and scope qualifiers.

    """
    def __init__(self, transaction: pymdlsdk.ITransaction, module: pymdlsdk.IModule, dbName: str) -> None:
        super(Module, self).__init__()
        
        self.filename = module.get_filename()
        self.dbName = dbName
        self.mdlName = module.get_mdl_name()
        self.mdlSimpleName = module.get_mdl_simple_name()

        moduleTypes = []
        with module.get_types() as moduleTypeList:
            for i in range(moduleTypeList.get_size()):
                moduleTypes.append(Type(moduleTypeList.get_type(i)))
        self.types = moduleTypes #TODO make immutable

        functionDbNames = []
        for i in range(module.get_function_count()):
            functionDbNames.append(module.get_function(i))
        for i in range(module.get_material_count()):
            functionDbNames.append(module.get_material(i))
        self.functionDbNames = tuple(functionDbNames)

        functionMap = {}
        if transaction and transaction.is_valid_interface():
            for dbName in self.functionDbNames:
                func = FunctionDefinition._fetchFromDb(transaction, dbName)
                if not func.mdlSimpleName in functionMap:
                    functionMap[func.mdlSimpleName] = []
                functionMap[func.mdlSimpleName].append(func)

        self.functions: Dict[str, FunctionDefinition]
        self.functions = functionMap #TODO make immutable

        with module.get_annotations() as module_annos:
            self.annotations = AnnotationBlock(module_annos)

    @staticmethod
    def _fetchFromDb(transaction: pymdlsdk.ITransaction, dbName) -> "Module":
        with transaction.access_as(pymdlsdk.IModule, dbName) as m:
            if m.is_valid_interface():
                return Module(transaction, m, dbName)
        return None
