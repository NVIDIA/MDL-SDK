/******************************************************************************
 * Copyright (c) 2012-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *****************************************************************************/
/// \file
/// \brief Definition of compiler known types, state variables, state functions, etc.

#ifndef OPERATOR
/**
 * Defines an operator function.
 *
 * @param ret    the return type of this operator
 * @param code   the IExpression::Operator enum value for this operator
 * @param args   the operator arguments
 * @param flags  version flags
 */
#define OPERATOR(ret, code, args, flags)
#endif
#ifndef EQ_OPERATORS
/**
 * Defines an operator== and operator!= functions, always returning bool in MDL.
 *
 * @param args   the operator arguments
 * @param flags  version flags
 */
#define EQ_OPERATORS(args, flags) \
    OPERATOR(bool, OK_EQUAL, args, flags) \
    OPERATOR(bool, OK_NOT_EQUAL, args, flags)
#endif
#ifndef REL_OPERATORS
/**
 * Defines operator<, operator<=, operator>, operator>= functions, always returning bool in MDL.
 *
 * @param args   the operator arguments
 * @param flags  version flags
 */
#define REL_OPERATORS(args, flags) \
    OPERATOR(bool, OK_LESS,             args, flags) \
    OPERATOR(bool, OK_LESS_OR_EQUAL,    args, flags) \
    OPERATOR(bool, OK_GREATER,          args, flags) \
    OPERATOR(bool, OK_GREATER_OR_EQUAL, args, flags)
#endif
#ifndef BIN_LOG_OPERATORS
/**
 * Defines operator&&, operator|| functions.
 *
 * @param ret    the return type of the operators
 * @param args   the operator arguments
 * @param flags  version flags
 */
#define BIN_LOG_OPERATORS(ret, args, flags)     \
    OPERATOR(ret, OK_LOGICAL_AND, args, flags)  \
    OPERATOR(ret, OK_LOGICAL_OR,  args, flags)
#endif
#ifndef INC_DEC_OPERATORS
/**
 * Defines operator++, operator-- functions.
 *
 * @param ret    the return type of the operators
 * @param args   the operator arguments
 * @param flags  version flags
 */
#define INC_DEC_OPERATORS(ret, args, flags) \
    OPERATOR(ret, OK_PRE_INCREMENT,  args, flags) \
    OPERATOR(ret, OK_PRE_DECREMENT,  args, flags) \
    OPERATOR(ret, OK_POST_INCREMENT, args, flags) \
    OPERATOR(ret, OK_POST_DECREMENT, args, flags)
#endif
#ifndef ASSIGN_OPERATOR
/**
 * Defines operator= function
 *
 * @param ret    the return type of the operator
 * @param args   the operator arguments
 * @param flags  version flags
 */
#define ASSIGN_OPERATOR(ret, args, flags)   OPERATOR(ret, OK_ASSIGN, args, flags)
#endif
#ifndef BINARY_PLUS_MINUS_OPERATORS
/**
 * Defines binary operator+ and operator- function
 *
 * @param ret    the return type of the operator
 * @param args   the operator arguments
 * @param flags  version flags
 */
#define BINARY_PLUS_MINUS_OPERATORS(ret, args, flags) \
    OPERATOR(ret, OK_PLUS,  args, flags) \
    OPERATOR(ret, OK_MINUS, args, flags)
#endif
#ifndef UNARY_OPERATORS
/**
 * Defines unary operator+ and operator- function
 *
 * @param ret    the return type of the operator
 * @param args   the operator arguments
 * @param flags  version flags
 */
#define UNARY_OPERATORS(ret, args, flags)    \
    OPERATOR(ret, OK_POSITIVE,  args, flags) \
    OPERATOR(ret, OK_NEGATIVE, args, flags)
#endif
#ifndef ASSIGN_OPS_OPERATORS
/**
 * Defines operator+=, operator-=, operator*=, operator/= functions
 *
 * @param ret    the return type of the operator
 * @param args   the operator arguments
 * @param flags  version flags
 */
#define ASSIGN_OPS_OPERATORS(ret, args, flags)      \
    OPERATOR(ret, OK_PLUS_ASSIGN,     args, flags)  \
    OPERATOR(ret, OK_MINUS_ASSIGN,    args, flags)  \
    OPERATOR(ret, OK_MULTIPLY_ASSIGN, args, flags)  \
    OPERATOR(ret, OK_DIVIDE_ASSIGN,   args, flags)
#endif
#ifndef ASSIGN_PLUS_MINUS_OPERATORS
/**
 * Defines operator+=, operator-= functions
 *
 * @param ret    the return type of the operator
 * @param args   the operator arguments
 * @param flags  version flags
 */
#define ASSIGN_PLUS_MINUS_OPERATORS(ret, args, flags)   \
    OPERATOR(ret, OK_PLUS_ASSIGN,  args, flags)         \
    OPERATOR(ret, OK_MINUS_ASSIGN, args, flags)
#endif
#ifndef MUL_OPERATOR
/**
 * Defines operator* function
 *
 * @param ret    the return type of the operator
 * @param args   the operator arguments
 * @param flags  version flags
 */
#define MUL_OPERATOR(ret, args, flags) \
    OPERATOR(ret, OK_MULTIPLY, args, flags)
#endif
#ifndef MUL_ASSIGN_OPERATOR
/**
 * Defines operator*= function
 *
 * @param ret    the return type of the operator
 * @param args   the operator arguments
 * @param flags  version flags
 */
#define MUL_ASSIGN_OPERATOR(ret, args, flags) \
    OPERATOR(ret, OK_MULTIPLY_ASSIGN, args, flags)
#endif
#ifndef METHOD
/**
 * Defines a method of a builtin struct.
 *
 * @param classname  the name of the builtin struct this method belongs to
 * @param mod        the type modifier of the return type
 * @param ret        the return type of this method
 * @param name       the name of this method
 * @param args       the method arguments
 * @param flags      version flags
 */
#define METHOD(classname, mod, ret, name, args, flags)
#endif
#ifndef METHOD_OVERLOAD
/**
 * Defines an overload of a method of a builtin struct.
 *
 * @param classname  the name of the builtin struct this method belongs to
 * @param mod        the type modifier of the return type
 * @param ret        the return type of this method
 * @param name       the name of this method
 * @param args       the method arguments
 * @param flags      version flags
 */
#define METHOD_OVERLOAD(classname, mod, ret, name, args, flags)
#endif
#ifndef METHOD_SYNONYM
/**
 * Defines a synonym name for the previously defined method function.
 *
 * @param name   the synonym name for the previously defined method
 * @param flags  version flags
 */
#define METHOD_SYNONYM(name, flags)
#endif
#ifndef CONSTRUCTOR
/**
 * Defines a constructor of a builtin struct.
 *
 * @param kind       either EXPLICIT, EXPLICIT_WARN, or IMPLICIT
 * @param classname  the name of the builtin struct this method belongs to
 * @param args       the constructor arguments
 * @param sema       semantics
 * @param flags      version flags
 *
 * @note no overload variant
 */
#define CONSTRUCTOR(kind, classname, args, sema, flags)
#endif
#ifndef FIELD
/**
 * Defines a field of a builtin struct.
 *
 * @param classname  the name of the builtin struct this field belongs to
 * @param mod        the type modifier of this field
 * @param type       the type of the field
 * @param fieldname  the name of the field
 * @param flags      version flags
 */
#define FIELD(classname, mod, type, fieldname, flags)
#endif
#ifndef ENUM_VALUE
/**
 * Defines an enum value inside a builtin enum.
 *
 * @param classname  the name of the builtin enum this constant belongs to
 * @param name       the name of the constant
 * @param value      the integer value of the constant
 * @param flags      version flags
 */
#define ENUM_VALUE(classname, name, value, flags)
#endif
#ifndef BUILTIN_TYPE_BEGIN
/**
 * Defines the begin of a builtin type.
 *
 * @param typename      the name of the builtin type
 * @param flags         version flags
 */
#define BUILTIN_TYPE_BEGIN(typename, flags)
#endif
#ifndef BUILTIN_TYPE_END
/**
 * Defines the end of a builtin type.
 *
 * @param typename      the name of the builtin type
 */
#define BUILTIN_TYPE_END(typename)
#endif
#ifndef ARG
/**
 * Defines a function or method argument.
 *
 * @param type  the type of the argument
 * @param name  the name of the argument
 * @param arr   ARR if this is an array of type, else empty
 */
#define ARG(type, name, arr)
#endif
#ifndef UARG
/**
 * Defines a function or method uniform argument.
 *
 * @param type  the base type of the argument
 * @param name  the name of the argument
 * @param arr   ARR if this is an array of type, else empty
 */
#define UARG(type, name, arr)
#endif
#ifndef DEFARG
/**
 * Defines a function or method argument with an default expression.
 *
 * @param type  the type of the argument
 * @param name  the name of the argument
 * @param arr   ARR if this is an array of type, else empty
 * @param expr  the default expression
 */
#define DEFARG(type, name, arr, expr)
#endif
#ifndef UDEFARG
/**
 * Defines a function or method uniform argument with an default expression.
 *
 * @param type  the type of the argument
 * @param name  the name of the argument
 * @param arr   ARR if this is an array of type, else empty
 * @param expr  the default expression
 */
#define UDEFARG(type, name, arr, expr)
#endif
#ifndef EXPR_LITERAL
/**
 * Defines a literal expression.
 *
 * @param value  the literal value
 */
#define EXPR_LITERAL(value)
#endif
#ifndef EXPR_COLOR_LITERAL
/**
 * Defines a color literal expression.
 *
 * @param value  the literal value (r=b=g=value)
 */
#define EXPR_COLOR_LITERAL(value)
#endif
#ifndef EXPR_FLOAT3_LITERAL
/**
 * Defines a float3 literal expression.
 *
 * @param value  the literal value (x=y=z=value)
 */
#define EXPR_FLOAT3_LITERAL(value)
#endif
#ifndef EXPR_CONSTRUCTOR
/**
 * Defines a constructor call expression.
 *
 * @param type  the constructor type
 */
#define EXPR_CONSTRUCTOR(type)
#endif
#ifndef EXPR_STATE
/**
 * Defines a argument less state function call expression.
 *
 * @param type  the return type of the state function
 * @param name  the state function name
 */
#define EXPR_STATE(type, name)
#endif
#ifndef EXPR_TEX_ENUM
/**
 * Defines a tex enum value expression.
 *
 * @param name  the state function name
 */
#define EXPR_TEX_ENUM(name)
#endif
#ifndef EXPR_INTENSITY_MODE_ENUM
/**
 * Defines a intensity_mode enum value expression.
 *
 * @param name  the state function name
 */
#define EXPR_INTENSITY_MODE_ENUM(name)
#endif
#if 0
/** Defines no arguments */
#define ARG0()
/** Defines one argument */
#define ARG1(a)
/** Defines two arguments */
#define ARG2(a1, a2)
/** Defines three arguments */
#define ARG3(a1, a2, a3)
/** Defines four arguments */
#define ARG4(a1, a2, a3, a4)
/** Defines five arguments */
#define ARG5(a1, a2, a3, a4, a5)
/** Defines six arguments */
#define ARG6(a1, a2, a3, a4, a5, a6)
/** Defines seven arguments */
#define ARG7(a1, a2, a3, a4, a5, a6, a7)
/** Defines eight arguments */
#define ARG8(a1, a2, a3, a4, a5, a6, a7, a8)
/** Defines nine arguments */
#define ARG9(a1, a2, a3, a4, a5, a6, a7, a8, a9)
/** Defines twelve arguments */
#define ARG12(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12)
/** Defines sixteen arguments */
#define ARG16(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16)
#endif

#ifndef ARRDEF
/*
 * Defines an array of size <N>.
 */
#define ARRDEF(N)
#endif
#ifndef ARRUSE
/*
 * Defines an array of size N.
 */
#define ARRUSE(N)
#endif

#ifndef EXPLICIT_WARN
#define EXPLICIT_WARN
#endif
#ifndef IMPLICIT
#define IMPLICIT
#endif
#ifndef EXPLICIT
#define EXPLICIT
#endif

/* ------------ Variable definitions ------------ */

/* ------------ Function definitions ------------ */

/* ------------ builtin types ------------ */

BUILTIN_TYPE_BEGIN(bool, 0)
    // default/copy constructor
    CONSTRUCTOR(IMPLICIT, bool,
        ARG1(DEFARG(bool, value,, EXPR_LITERAL(false))), DS_COPY_CONSTRUCTOR, 0)
    // conversion
    CONSTRUCTOR(EXPLICIT, bool, ARG1(ARG(int, value,)), DS_CONV_CONSTRUCTOR, 0)
    CONSTRUCTOR(EXPLICIT, bool, ARG1(ARG(float, value,)), DS_CONV_CONSTRUCTOR, 0)
    CONSTRUCTOR(EXPLICIT, bool, ARG1(ARG(double, value,)), DS_CONV_CONSTRUCTOR, 0)
BUILTIN_TYPE_END(bool)

BUILTIN_TYPE_BEGIN(bool2, 0)
    // default constructor
    CONSTRUCTOR(EXPLICIT_WARN, bool2,
        ARG1(DEFARG(bool, value,, EXPR_LITERAL(false))), DS_CONV_CONSTRUCTOR, 0)
    // elementary constructor
    CONSTRUCTOR(IMPLICIT, bool2, ARG2(ARG(bool, x,), ARG(bool, y,)), DS_ELEM_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, bool2, ARG1(ARG(bool2, value,)), DS_COPY_CONSTRUCTOR, 0)
    // conversion
    CONSTRUCTOR(EXPLICIT, bool2, ARG1(ARG(int2, value,)), DS_CONV_CONSTRUCTOR, 0)
    CONSTRUCTOR(EXPLICIT, bool2, ARG1(ARG(float2, value,)), DS_CONV_CONSTRUCTOR, 0)
    CONSTRUCTOR(EXPLICIT, bool2, ARG1(ARG(double2, value,)), DS_CONV_CONSTRUCTOR, 0)

    FIELD(bool2,, bool, x, 0)
    FIELD(bool2,, bool, y, 0)
BUILTIN_TYPE_END(bool2)

BUILTIN_TYPE_BEGIN(bool3, 0)
    // default/scalar constructor
    CONSTRUCTOR(EXPLICIT_WARN, bool3,
        ARG1(DEFARG(bool, value,, EXPR_LITERAL(false))), DS_CONV_CONSTRUCTOR, 0)
    CONSTRUCTOR(IMPLICIT, bool3,
        ARG3(
            ARG(bool, x,),
            ARG(bool, y,),
            ARG(bool, z,)
        ), DS_ELEM_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, bool3, ARG1(ARG(bool3, value,)), DS_COPY_CONSTRUCTOR, 0)
    // conversion
    CONSTRUCTOR(EXPLICIT, bool3, ARG1(ARG(int3, value,)), DS_CONV_CONSTRUCTOR, 0)
    CONSTRUCTOR(EXPLICIT, bool3, ARG1(ARG(float3, value,)), DS_CONV_CONSTRUCTOR, 0)
    CONSTRUCTOR(EXPLICIT, bool3, ARG1(ARG(double3, value,)), DS_CONV_CONSTRUCTOR, 0)

    FIELD(bool3,, bool, x, 0)
    FIELD(bool3,, bool, y, 0)
    FIELD(bool3,, bool, z, 0)
BUILTIN_TYPE_END(bool3)

BUILTIN_TYPE_BEGIN(bool4, 0)
    // default/scalar constructor
    CONSTRUCTOR(EXPLICIT_WARN, bool4,
        ARG1(DEFARG(bool, value,, EXPR_LITERAL(false))), DS_CONV_CONSTRUCTOR, 0)
    // elementary constructor
    CONSTRUCTOR(IMPLICIT, bool4,
        ARG4(
            ARG(bool, x,),
            ARG(bool, y,),
            ARG(bool, z,),
            ARG(bool, w,)
        ), DS_ELEM_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, bool4, ARG1(ARG(bool4, value,)), DS_COPY_CONSTRUCTOR, 0)
    // conversion
    CONSTRUCTOR(EXPLICIT, bool4, ARG1(ARG(int4, value,)), DS_CONV_CONSTRUCTOR, 0)
    CONSTRUCTOR(EXPLICIT, bool4, ARG1(ARG(float4, value,)), DS_CONV_CONSTRUCTOR, 0)
    CONSTRUCTOR(EXPLICIT, bool4, ARG1(ARG(double4, value,)), DS_CONV_CONSTRUCTOR, 0)

    FIELD(bool4,, bool, x, 0)
    FIELD(bool4,, bool, y, 0)
    FIELD(bool4,, bool, z, 0)
    FIELD(bool4,, bool, w, 0)
BUILTIN_TYPE_END(bool4)

BUILTIN_TYPE_BEGIN(int, 0)
    // default/copy constructor
    CONSTRUCTOR(IMPLICIT, int, ARG1(DEFARG(int, value,, EXPR_LITERAL(0))), DS_COPY_CONSTRUCTOR, 0)
    // conversion
    CONSTRUCTOR(IMPLICIT, int, ARG1(ARG(bool, value,)), DS_CONV_CONSTRUCTOR, 0)
    CONSTRUCTOR(EXPLICIT, int, ARG1(ARG(float, value,)), DS_CONV_CONSTRUCTOR, 0)
    CONSTRUCTOR(EXPLICIT, int, ARG1(ARG(double, value,)), DS_CONV_CONSTRUCTOR, 0)
BUILTIN_TYPE_END(int)

BUILTIN_TYPE_BEGIN(int2, 0)
    // default/scalar constructor
    CONSTRUCTOR(EXPLICIT_WARN, int2,
        ARG1(DEFARG(int, value,, EXPR_LITERAL(0))), DS_CONV_CONSTRUCTOR, 0)
    // elementary constructor
    CONSTRUCTOR(IMPLICIT, int2, ARG2(ARG(int, x,), ARG(int, y,)), DS_ELEM_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, int2, ARG1(ARG(int2, value,)), DS_COPY_CONSTRUCTOR, 0)
    // conversion
    CONSTRUCTOR(IMPLICIT, int2, ARG1(ARG(bool2, value,)), DS_CONV_CONSTRUCTOR, 0)
    CONSTRUCTOR(EXPLICIT, int2, ARG1(ARG(float2, value,)), DS_CONV_CONSTRUCTOR, 0)
    CONSTRUCTOR(EXPLICIT, int2, ARG1(ARG(double2, value,)), DS_CONV_CONSTRUCTOR, 0)

    FIELD(int2,, int, x, 0)
    FIELD(int2,, int, y, 0)
BUILTIN_TYPE_END(int2)

BUILTIN_TYPE_BEGIN(int3, 0)
    CONSTRUCTOR(EXPLICIT_WARN, int3,
        ARG1(DEFARG(int, value,, EXPR_LITERAL(0))), DS_CONV_CONSTRUCTOR, 0)
    CONSTRUCTOR(IMPLICIT, int3,
        ARG3(
            ARG(int, x,),
            ARG(int, y,),
            ARG(int, z,)
        ), DS_ELEM_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, int3, ARG1(ARG(int3, value,)), DS_COPY_CONSTRUCTOR, 0)
    // conversion
    CONSTRUCTOR(IMPLICIT, int3, ARG1(ARG(bool3, value,)), DS_CONV_CONSTRUCTOR, 0)
    CONSTRUCTOR(EXPLICIT, int3, ARG1(ARG(float3, value,)), DS_CONV_CONSTRUCTOR, 0)
    CONSTRUCTOR(EXPLICIT, int3, ARG1(ARG(double3, value,)), DS_CONV_CONSTRUCTOR, 0)

    FIELD(int3,, int, x, 0)
    FIELD(int3,, int, y, 0)
    FIELD(int3,, int, z, 0)
BUILTIN_TYPE_END(int3)

BUILTIN_TYPE_BEGIN(int4, 0)
    // default/scalar constructor
    CONSTRUCTOR(EXPLICIT_WARN, int4,
        ARG1(DEFARG(int, value,, EXPR_LITERAL(0))), DS_CONV_CONSTRUCTOR, 0)
    // elementary constructor
    CONSTRUCTOR(IMPLICIT, int4,
        ARG4(
            ARG(int, x,),
            ARG(int, y,),
            ARG(int, z,),
            ARG(int, w,)
        ), DS_ELEM_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, int4, ARG1(ARG(int4, value,)), DS_COPY_CONSTRUCTOR, 0)
    // conversion
    CONSTRUCTOR(IMPLICIT, int4, ARG1(ARG(bool4, value,)), DS_CONV_CONSTRUCTOR, 0)
    CONSTRUCTOR(EXPLICIT, int4, ARG1(ARG(float4, value,)), DS_CONV_CONSTRUCTOR, 0)
    CONSTRUCTOR(EXPLICIT, int4, ARG1(ARG(double4, value,)), DS_CONV_CONSTRUCTOR, 0)

    FIELD(int4,, int, x, 0)
    FIELD(int4,, int, y, 0)
    FIELD(int4,, int, z, 0)
    FIELD(int4,, int, w, 0)
BUILTIN_TYPE_END(int4)

BUILTIN_TYPE_BEGIN(float, 0)
    // default/copy constructor
    CONSTRUCTOR(IMPLICIT, float,
        ARG1(DEFARG(float, value,, EXPR_LITERAL(0.0f))), DS_COPY_CONSTRUCTOR, 0)
    // conversion
    CONSTRUCTOR(IMPLICIT, float, ARG1(ARG(bool, value,)), DS_CONV_CONSTRUCTOR, 0)
    CONSTRUCTOR(IMPLICIT, float, ARG1(ARG(int, value,)), DS_CONV_CONSTRUCTOR, 0)
    CONSTRUCTOR(EXPLICIT, float, ARG1(ARG(double, value,)), DS_CONV_CONSTRUCTOR, 0)
BUILTIN_TYPE_END(float)

BUILTIN_TYPE_BEGIN(float2, 0)
    // default/scalar constructor
    CONSTRUCTOR(EXPLICIT_WARN, float2,
        ARG1(DEFARG(float, value,, EXPR_LITERAL(0.0f))), DS_CONV_CONSTRUCTOR, 0)
    // elementary constructor
    CONSTRUCTOR(IMPLICIT, float2, ARG2(ARG(float, x,), ARG(float, y,)), DS_ELEM_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, float2, ARG1(ARG(float2, value,)), DS_COPY_CONSTRUCTOR, 0) //-V525 PVS
    // conversion
    CONSTRUCTOR(IMPLICIT, float2, ARG1(ARG(bool2, value,)), DS_CONV_CONSTRUCTOR, 0)
    CONSTRUCTOR(IMPLICIT, float2, ARG1(ARG(int2, value,)), DS_CONV_CONSTRUCTOR, 0)
    CONSTRUCTOR(EXPLICIT, float2, ARG1(ARG(double2, value,)), DS_CONV_CONSTRUCTOR, 0)

    FIELD(float2,, float, x, 0)
    FIELD(float2,, float, y, 0)
BUILTIN_TYPE_END(float2)

BUILTIN_TYPE_BEGIN(float3, 0)
    // default/scalar constructor
    CONSTRUCTOR(EXPLICIT_WARN, float3,
        ARG1(DEFARG(float, value,, EXPR_LITERAL(0.0f))), DS_CONV_CONSTRUCTOR, 0)
    // elementary constructor
    CONSTRUCTOR(IMPLICIT, float3,
        ARG3(
            ARG(float, x,),
            ARG(float, y,),
            ARG(float, z,)
        ), DS_ELEM_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, float3, ARG1(ARG(float3, value,)), DS_COPY_CONSTRUCTOR, 0) //-V525 PVS
    // conversion
    CONSTRUCTOR(IMPLICIT, float3, ARG1(ARG(bool3, value,)), DS_CONV_CONSTRUCTOR, 0)
    CONSTRUCTOR(IMPLICIT, float3, ARG1(ARG(int3, value,)), DS_CONV_CONSTRUCTOR, 0)
    CONSTRUCTOR(EXPLICIT, float3, ARG1(ARG(double3, value,)), DS_CONV_CONSTRUCTOR, 0)
    // 5.5.1 A float3 vector can be constructed from color.
    CONSTRUCTOR(EXPLICIT, float3, ARG1(ARG(color, value,)), DS_CONV_CONSTRUCTOR, 0)

    FIELD(float3,, float, x, 0)
    FIELD(float3,, float, y, 0)
    FIELD(float3,, float, z, 0)
BUILTIN_TYPE_END(float3)

BUILTIN_TYPE_BEGIN(float4, 0)
    // default/scalar constructor
    CONSTRUCTOR(EXPLICIT_WARN, float4,
        ARG1(DEFARG(float, value,, EXPR_LITERAL(0.0f))), DS_CONV_CONSTRUCTOR, 0)
    // elementary constructor
    CONSTRUCTOR(IMPLICIT, float4,
        ARG4(
            ARG(float, x,),
            ARG(float, y,),
            ARG(float, z,),
            ARG(float, w,)
        ), DS_ELEM_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, float4, ARG1(ARG(float4, value,)), DS_COPY_CONSTRUCTOR, 0) //-V525 PVS
    // conversion
    CONSTRUCTOR(IMPLICIT, float4, ARG1(ARG(bool4, value,)), DS_CONV_CONSTRUCTOR, 0)
    CONSTRUCTOR(IMPLICIT, float4, ARG1(ARG(int4, value,)), DS_CONV_CONSTRUCTOR, 0)
    CONSTRUCTOR(EXPLICIT, float4, ARG1(ARG(double4, value,)), DS_CONV_CONSTRUCTOR, 0)

    FIELD(float4,, float, x, 0)
    FIELD(float4,, float, y, 0)
    FIELD(float4,, float, z, 0)
    FIELD(float4,, float, w, 0)
BUILTIN_TYPE_END(float4)

BUILTIN_TYPE_BEGIN(double, 0)
    // default/copy constructor
    CONSTRUCTOR(IMPLICIT, double,
        ARG1(DEFARG(double, value,, EXPR_LITERAL(0.0))), DS_COPY_CONSTRUCTOR, 0)
    // conversion
    CONSTRUCTOR(IMPLICIT, double, ARG1(ARG(bool, value,)), DS_CONV_CONSTRUCTOR, 0)
    CONSTRUCTOR(IMPLICIT, double, ARG1(ARG(int, value,)), DS_CONV_CONSTRUCTOR, 0)
    CONSTRUCTOR(IMPLICIT, double, ARG1(ARG(float, value,)), DS_CONV_CONSTRUCTOR, 0)
BUILTIN_TYPE_END(double)

BUILTIN_TYPE_BEGIN(double2, 0)
    // default/scalar constructor
    CONSTRUCTOR(EXPLICIT_WARN, double2,
        ARG1(DEFARG(double, value,, EXPR_LITERAL(0.0))), DS_CONV_CONSTRUCTOR, 0)
    // elementary constructor
    CONSTRUCTOR(IMPLICIT, double2, ARG2(ARG(double, x,), ARG(double, y,)), DS_ELEM_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, double2, ARG1(ARG(double2, value,)), DS_COPY_CONSTRUCTOR, 0)
    // conversion
    CONSTRUCTOR(IMPLICIT, double2, ARG1(ARG(bool2, value,)), DS_CONV_CONSTRUCTOR, 0)
    CONSTRUCTOR(IMPLICIT, double2, ARG1(ARG(int2, value,)), DS_CONV_CONSTRUCTOR, 0)
    CONSTRUCTOR(IMPLICIT, double2, ARG1(ARG(float2, value,)), DS_CONV_CONSTRUCTOR, 0)

    FIELD(double2,, double, x, 0)
    FIELD(double2,, double, y, 0)
BUILTIN_TYPE_END(double2)

BUILTIN_TYPE_BEGIN(double3, 0)
    // default/scalar constructor
    CONSTRUCTOR(EXPLICIT_WARN, double3,
        ARG1(DEFARG(double, value,, EXPR_LITERAL(0.0))), DS_CONV_CONSTRUCTOR, 0)
    // elementary constructor
    CONSTRUCTOR(IMPLICIT, double3,
        ARG3(
            ARG(double, x,),
            ARG(double, y,),
            ARG(double, z,)
        ), DS_ELEM_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, double3, ARG1(ARG(double3, value,)), DS_COPY_CONSTRUCTOR, 0)
    // conversion
    CONSTRUCTOR(IMPLICIT, double3, ARG1(ARG(bool3, value,)), DS_CONV_CONSTRUCTOR, 0)
    CONSTRUCTOR(IMPLICIT, double3, ARG1(ARG(int3, value,)), DS_CONV_CONSTRUCTOR, 0)
    CONSTRUCTOR(IMPLICIT, double3, ARG1(ARG(float3, value,)), DS_CONV_CONSTRUCTOR, 0)

    FIELD(double3,, double, x, 0)
    FIELD(double3,, double, y, 0)
    FIELD(double3,, double, z, 0)
BUILTIN_TYPE_END(double3)

BUILTIN_TYPE_BEGIN(double4, 0)
    // default/scalar constructor
    CONSTRUCTOR(EXPLICIT_WARN, double4,
        ARG1(
            DEFARG(double, value,, EXPR_LITERAL(0.0))
        ), DS_CONV_CONSTRUCTOR, 0)
    // elementary constructor
    CONSTRUCTOR(IMPLICIT, double4,
        ARG4(
            ARG(double, x,),
            ARG(double, y,),
            ARG(double, z,),
            ARG(double, w,)
        ), DS_ELEM_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, double4, ARG1(ARG(double4, value,)), DS_COPY_CONSTRUCTOR, 0)
    // conversion
    CONSTRUCTOR(IMPLICIT, double4, ARG1(ARG(bool4, value,)), DS_CONV_CONSTRUCTOR, 0)
    CONSTRUCTOR(IMPLICIT, double4, ARG1(ARG(int4, value,)), DS_CONV_CONSTRUCTOR, 0)
    CONSTRUCTOR(IMPLICIT, double4, ARG1(ARG(float4, value,)), DS_CONV_CONSTRUCTOR, 0)

    FIELD(double4,, double, x, 0)
    FIELD(double4,, double, y, 0)
    FIELD(double4,, double, z, 0)
    FIELD(double4,, double, w, 0)
BUILTIN_TYPE_END(double4)

BUILTIN_TYPE_BEGIN(string, 0)
    // default/copy constructor
    CONSTRUCTOR(IMPLICIT, string,
        ARG1(DEFARG(string, value,, EXPR_LITERAL(""))), DS_COPY_CONSTRUCTOR, 0)
BUILTIN_TYPE_END(string)

BUILTIN_TYPE_BEGIN(color, 0)
    // default/gray constructor
    CONSTRUCTOR(EXPLICIT_WARN, color,
        ARG1(
            DEFARG(float, value,, EXPR_LITERAL(0.0f))
        ), DS_CONV_CONSTRUCTOR, 0)
    // rgb constructor
    CONSTRUCTOR(IMPLICIT, color,
        ARG3(
            ARG(float, r,),
            ARG(float, g,),
            ARG(float, b,)
        ), DS_ELEM_CONSTRUCTOR, 0)
    // conversion
    CONSTRUCTOR(EXPLICIT_WARN, color, ARG1(ARG(float3, rgb,)), DS_CONV_CONSTRUCTOR, 0)
    CONSTRUCTOR(IMPLICIT, color,
        ARG2(
            ARG(float, wavelengths, ARRDEF(N)),
            ARG(float, amplitudes, ARRUSE(N))
        ), DS_COLOR_SPECTRUM_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, color, ARG1(ARG(color, value,)), DS_COPY_CONSTRUCTOR, 0)
BUILTIN_TYPE_END(color)

// matrices
BUILTIN_TYPE_BEGIN(float2x2, 0)
    // scalar constructor
    CONSTRUCTOR(EXPLICIT_WARN, float2x2,
        ARG1(
            DEFARG(float, value,, EXPR_LITERAL(0.0f))
        ), DS_MATRIX_DIAG_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, float2x2, ARG1(ARG(float2x2, value,)), DS_COPY_CONSTRUCTOR, 0)
    // elemental constructor
    CONSTRUCTOR(IMPLICIT, float2x2,
        ARG4(
            ARG(float, m00,),
            ARG(float, m01,),
            ARG(float, m10,),
            ARG(float, m11,)
        ), DS_MATRIX_ELEM_CONSTRUCTOR, 0)
    CONSTRUCTOR(IMPLICIT, float2x2,
        ARG2(
            ARG(float2, col0,),
            ARG(float2, col1,)
        ), DS_ELEM_CONSTRUCTOR, 0)
    // conversion
    CONSTRUCTOR(EXPLICIT, float2x2, ARG1(ARG(double2x2, value,)), DS_CONV_CONSTRUCTOR, 0)
BUILTIN_TYPE_END(float2x2)

BUILTIN_TYPE_BEGIN(float2x3, 0)
    // default/diagonal constructor
    CONSTRUCTOR(EXPLICIT_WARN, float2x3,
        ARG1(
            DEFARG(float, value,, EXPR_LITERAL(0.0f))
        ), DS_MATRIX_DIAG_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, float2x3, ARG1(ARG(float2x3, value,)), DS_COPY_CONSTRUCTOR, 0)
    // elemental constructor
    CONSTRUCTOR(IMPLICIT, float2x3,
        ARG6(
            ARG(float, m00,),
            ARG(float, m01,),
            ARG(float, m02,),
            ARG(float, m10,),
            ARG(float, m11,),
            ARG(float, m12,)
        ), DS_MATRIX_ELEM_CONSTRUCTOR, 0)
    CONSTRUCTOR(IMPLICIT, float2x3,
        ARG2(
            ARG(float3, col0,),
            ARG(float3, col1,)
        ), DS_ELEM_CONSTRUCTOR, 0)
    // conversion
    CONSTRUCTOR(EXPLICIT, float2x3, ARG1(ARG(double2x3, value,)), DS_CONV_CONSTRUCTOR, 0)
BUILTIN_TYPE_END(float2x3)

BUILTIN_TYPE_BEGIN(float2x4, 0)
    // default/diagonal constructor
    CONSTRUCTOR(EXPLICIT_WARN, float2x4,
        ARG1(
            DEFARG(float, value,, EXPR_LITERAL(0.0f))
        ), DS_MATRIX_DIAG_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, float2x4, ARG1(ARG(float2x4, value,)), DS_COPY_CONSTRUCTOR, 0)
    // elemental constructor
    CONSTRUCTOR(IMPLICIT, float2x4,
        ARG8(
            ARG(float, m00,),
            ARG(float, m01,),
            ARG(float, m02,),
            ARG(float, m03,),
            ARG(float, m10,),
            ARG(float, m11,),
            ARG(float, m12,),
            ARG(float, m13,)
        ), DS_MATRIX_ELEM_CONSTRUCTOR, 0)
    CONSTRUCTOR(IMPLICIT, float2x4,
        ARG2(
            ARG(float4, col0,),
            ARG(float4, col1,)
        ), DS_ELEM_CONSTRUCTOR, 0)
    // conversion
    CONSTRUCTOR(EXPLICIT, float2x4, ARG1(ARG(double2x4, value,)), DS_CONV_CONSTRUCTOR, 0)
BUILTIN_TYPE_END(float2x4)

BUILTIN_TYPE_BEGIN(float3x2, 0)
    // default/diagonal constructor
    CONSTRUCTOR(EXPLICIT_WARN, float3x2,
        ARG1(
            DEFARG(float, value,, EXPR_LITERAL(0.0f))
        ), DS_MATRIX_DIAG_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, float3x2, ARG1(ARG(float3x2, value,)), DS_COPY_CONSTRUCTOR, 0)
    // elemental constructor
    CONSTRUCTOR(IMPLICIT, float3x2,
        ARG6(
            ARG(float, m00,),
            ARG(float, m01,),
            ARG(float, m10,),
            ARG(float, m11,),
            ARG(float, m20,),
            ARG(float, m21,)
        ), DS_MATRIX_ELEM_CONSTRUCTOR, 0)
    CONSTRUCTOR(IMPLICIT, float3x2,
        ARG3(
            ARG(float2, col0,),
            ARG(float2, col1,),
            ARG(float2, col2,)
        ), DS_ELEM_CONSTRUCTOR, 0)
    // conversion
    CONSTRUCTOR(EXPLICIT, float3x2, ARG1(ARG(double3x2, value,)), DS_CONV_CONSTRUCTOR, 0)
BUILTIN_TYPE_END(float3x2)

BUILTIN_TYPE_BEGIN(float3x3, 0)
    // default/diagonal constructor
    CONSTRUCTOR(EXPLICIT_WARN, float3x3,
        ARG1(
            DEFARG(float, value,, EXPR_LITERAL(0.0f))
        ), DS_MATRIX_DIAG_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, float3x3, ARG1(ARG(float3x3, value,)), DS_COPY_CONSTRUCTOR, 0)
    // elemental constructor
    CONSTRUCTOR(IMPLICIT, float3x3,
        ARG9(
            ARG(float, m00,),
            ARG(float, m01,),
            ARG(float, m02,),
            ARG(float, m10,),
            ARG(float, m11,),
            ARG(float, m12,),
            ARG(float, m20,),
            ARG(float, m21,),
            ARG(float, m22,)
        ), DS_MATRIX_ELEM_CONSTRUCTOR, 0)
    CONSTRUCTOR(IMPLICIT, float3x3,
        ARG3(
            ARG(float3, col0,),
            ARG(float3, col1,),
            ARG(float3, col2,)
        ), DS_ELEM_CONSTRUCTOR, 0)
    // conversion
    CONSTRUCTOR(EXPLICIT, float3x3, ARG1(ARG(double3x3, value,)), DS_CONV_CONSTRUCTOR, 0)
BUILTIN_TYPE_END(float3x3)

BUILTIN_TYPE_BEGIN(float3x4, 0)
    // default/diagonal constructor
    CONSTRUCTOR(EXPLICIT_WARN, float3x4,
        ARG1(
            DEFARG(float, value,, EXPR_LITERAL(0.0f))
        ), DS_MATRIX_DIAG_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, float3x4, ARG1(ARG(float3x4, value,)), DS_COPY_CONSTRUCTOR, 0)
    // elemental constructor
    CONSTRUCTOR(IMPLICIT, float3x4,
        ARG12(
            ARG(float, m00,),
            ARG(float, m01,),
            ARG(float, m02,),
            ARG(float, m03,),
            ARG(float, m10,),
            ARG(float, m11,),
            ARG(float, m12,),
            ARG(float, m13,),
            ARG(float, m20,),
            ARG(float, m21,),
            ARG(float, m22,),
            ARG(float, m23,)
        ), DS_MATRIX_ELEM_CONSTRUCTOR, 0)
    CONSTRUCTOR(IMPLICIT, float3x4,
        ARG3(
            ARG(float4, col0,),
            ARG(float4, col1,),
            ARG(float4, col2,)
        ), DS_ELEM_CONSTRUCTOR, 0)
    // conversion
    CONSTRUCTOR(EXPLICIT, float3x4, ARG1(ARG(double3x4, value,)), DS_CONV_CONSTRUCTOR, 0)
BUILTIN_TYPE_END(float3x4)

BUILTIN_TYPE_BEGIN(float4x2, 0)
    // default/diagonal constructor
    CONSTRUCTOR(EXPLICIT_WARN, float4x2,
        ARG1(
            DEFARG(float, value,, EXPR_LITERAL(0.0f))
        ), DS_MATRIX_DIAG_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, float4x2, ARG1(ARG(float4x2, value,)), DS_COPY_CONSTRUCTOR, 0)
    // elemental constructor
    CONSTRUCTOR(IMPLICIT, float4x2,
        ARG8(
            ARG(float, m00,),
            ARG(float, m01,),
            ARG(float, m10,),
            ARG(float, m11,),
            ARG(float, m20,),
            ARG(float, m21,),
            ARG(float, m30,),
            ARG(float, m31,)
        ), DS_MATRIX_ELEM_CONSTRUCTOR, 0)
    CONSTRUCTOR(IMPLICIT, float4x2,
        ARG4(
            ARG(float2, col0,),
            ARG(float2, col1,),
            ARG(float2, col2,),
            ARG(float2, col3,)
        ), DS_ELEM_CONSTRUCTOR, 0)
    // conversion
    CONSTRUCTOR(EXPLICIT, float4x2, ARG1(ARG(double4x2, value,)), DS_CONV_CONSTRUCTOR, 0)
BUILTIN_TYPE_END(float4x2)

BUILTIN_TYPE_BEGIN(float4x3, 0)
    // default/diagonal constructor
    CONSTRUCTOR(EXPLICIT_WARN, float4x3,
        ARG1(
            DEFARG(float, value,, EXPR_LITERAL(0.0f))
        ), DS_MATRIX_DIAG_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, float4x3, ARG1(ARG(float4x3, value,)), DS_COPY_CONSTRUCTOR, 0)
    // elemental constructor
    CONSTRUCTOR(IMPLICIT, float4x3,
        ARG12(
            ARG(float, m00,),
            ARG(float, m01,),
            ARG(float, m02,),
            ARG(float, m10,),
            ARG(float, m11,),
            ARG(float, m12,),
            ARG(float, m20,),
            ARG(float, m21,),
            ARG(float, m22,),
            ARG(float, m30,),
            ARG(float, m31,),
            ARG(float, m32,)
        ), DS_MATRIX_ELEM_CONSTRUCTOR, 0)
    CONSTRUCTOR(IMPLICIT, float4x3,
        ARG4(
            ARG(float3, col0,),
            ARG(float3, col1,),
            ARG(float3, col2,),
            ARG(float3, col3,)
        ), DS_ELEM_CONSTRUCTOR, 0)
    // conversion
    CONSTRUCTOR(EXPLICIT, float4x3, ARG1(ARG(double4x3, value,)), DS_CONV_CONSTRUCTOR, 0)
BUILTIN_TYPE_END(float4x3)

BUILTIN_TYPE_BEGIN(float4x4, 0)
    // default/diagonal constructor
    CONSTRUCTOR(EXPLICIT_WARN, float4x4,
        ARG1(
            DEFARG(float, value,, EXPR_LITERAL(0.0f))
        ), DS_MATRIX_DIAG_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, float4x4, ARG1(ARG(float4x4, value,)), DS_COPY_CONSTRUCTOR, 0)
    // elemental constructor
    CONSTRUCTOR(IMPLICIT, float4x4,
        ARG16(
            ARG(float, m00,),
            ARG(float, m01,),
            ARG(float, m02,),
            ARG(float, m03,),
            ARG(float, m10,),
            ARG(float, m11,),
            ARG(float, m12,),
            ARG(float, m13,),
            ARG(float, m20,),
            ARG(float, m21,),
            ARG(float, m22,),
            ARG(float, m23,),
            ARG(float, m30,),
            ARG(float, m31,),
            ARG(float, m32,),
            ARG(float, m33,)
        ), DS_MATRIX_ELEM_CONSTRUCTOR, 0)
    CONSTRUCTOR(IMPLICIT, float4x4,
        ARG4(
            ARG(float4, col0,),
            ARG(float4, col1,),
            ARG(float4, col2,),
            ARG(float4, col3,)
        ), DS_ELEM_CONSTRUCTOR, 0)
    // conversion
    CONSTRUCTOR(EXPLICIT, float4x4, ARG1(ARG(double4x4, value,)), DS_CONV_CONSTRUCTOR, 0)
BUILTIN_TYPE_END(float4x4)

BUILTIN_TYPE_BEGIN(double2x2, 0)
    // default/diagonal constructor
    CONSTRUCTOR(EXPLICIT_WARN, double2x2,
        ARG1(
            DEFARG(double, value,, EXPR_LITERAL(0.0))
        ), DS_MATRIX_DIAG_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, double2x2, ARG1(ARG(double2x2, value,)), DS_COPY_CONSTRUCTOR, 0)
    // elemental constructor
    CONSTRUCTOR(IMPLICIT, double2x2,
        ARG4(
            ARG(double, m00,),
            ARG(double, m01,),
            ARG(double, m10,),
            ARG(double, m11,)
        ), DS_MATRIX_ELEM_CONSTRUCTOR, 0)
    CONSTRUCTOR(IMPLICIT, double2x2,
        ARG2(
            ARG(double2, col0,),
            ARG(double2, col1,)
        ), DS_ELEM_CONSTRUCTOR, 0)
    // conversion
    CONSTRUCTOR(IMPLICIT, double2x2, ARG1(ARG(float2x2, value,)), DS_CONV_CONSTRUCTOR, 0)
BUILTIN_TYPE_END(double2x2)

BUILTIN_TYPE_BEGIN(double2x3, 0)
    // default/diagonal constructor
    CONSTRUCTOR(EXPLICIT_WARN, double2x3,
        ARG1(
            DEFARG(double, value,, EXPR_LITERAL(0.0))
        ), DS_MATRIX_DIAG_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, double2x3, ARG1(ARG(double2x3, value,)), DS_COPY_CONSTRUCTOR, 0)
    // elemental constructor
    CONSTRUCTOR(IMPLICIT, double2x3,
        ARG6(
            ARG(double, m00,),
            ARG(double, m01,),
            ARG(double, m02,),
            ARG(double, m10,),
            ARG(double, m11,),
            ARG(double, m12,)
        ), DS_MATRIX_ELEM_CONSTRUCTOR, 0)
    CONSTRUCTOR(IMPLICIT, double2x3,
        ARG2(
            ARG(double3, col0,),
            ARG(double3, col1,)
        ), DS_ELEM_CONSTRUCTOR, 0)
    // conversion
    CONSTRUCTOR(IMPLICIT, double2x3, ARG1(ARG(float2x3, value,)), DS_CONV_CONSTRUCTOR, 0)
BUILTIN_TYPE_END(double2x3)

BUILTIN_TYPE_BEGIN(double2x4, 0)
    // default/diagonal constructor
    CONSTRUCTOR(EXPLICIT_WARN, double2x4,
        ARG1(
            DEFARG(double, value,, EXPR_LITERAL(0.0))
        ), DS_MATRIX_DIAG_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, double2x4, ARG1(ARG(double2x4, value,)), DS_COPY_CONSTRUCTOR, 0)
    // elemental constructor
    CONSTRUCTOR(IMPLICIT, double2x4,
        ARG8(
            ARG(double, m00,),
            ARG(double, m01,),
            ARG(double, m02,),
            ARG(double, m03,),
            ARG(double, m10,),
            ARG(double, m11,),
            ARG(double, m12,),
            ARG(double, m13,)
        ), DS_MATRIX_ELEM_CONSTRUCTOR, 0)
    CONSTRUCTOR(IMPLICIT, double2x4,
        ARG2(
            ARG(double4, col0,),
            ARG(double4, col1,)
        ), DS_ELEM_CONSTRUCTOR, 0)
    // conversion
    CONSTRUCTOR(IMPLICIT, double2x4, ARG1(ARG(float2x4, value,)), DS_CONV_CONSTRUCTOR, 0)
BUILTIN_TYPE_END(double2x4)

BUILTIN_TYPE_BEGIN(double3x2, 0)
    // default/diagonal constructor
    CONSTRUCTOR(EXPLICIT_WARN, double3x2,
        ARG1(
            DEFARG(double, value,, EXPR_LITERAL(0.0))
        ), DS_MATRIX_DIAG_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, double3x2, ARG1(ARG(double3x2, value,)), DS_COPY_CONSTRUCTOR, 0)
    // elemental constructor
    CONSTRUCTOR(IMPLICIT, double3x2,
        ARG6(
            ARG(double, m00,),
            ARG(double, m01,),
            ARG(double, m10,),
            ARG(double, m11,),
            ARG(double, m20,),
            ARG(double, m21,)
        ), DS_MATRIX_ELEM_CONSTRUCTOR, 0)
    CONSTRUCTOR(IMPLICIT, double3x2,
        ARG3(
            ARG(double2, col0,),
            ARG(double2, col1,),
            ARG(double2, col2,)
        ), DS_ELEM_CONSTRUCTOR, 0)
    // conversion
    CONSTRUCTOR(IMPLICIT, double3x2, ARG1(ARG(float3x2, value,)), DS_CONV_CONSTRUCTOR, 0)
BUILTIN_TYPE_END(double3x2)

BUILTIN_TYPE_BEGIN(double3x3, 0)
    // default/diagonal constructor
    CONSTRUCTOR(EXPLICIT_WARN, double3x3,
        ARG1(
            DEFARG(double, value,, EXPR_LITERAL(0.0))
        ), DS_MATRIX_DIAG_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, double3x3, ARG1(ARG(double3x3, value,)), DS_COPY_CONSTRUCTOR, 0)
    // elemental constructor
    CONSTRUCTOR(IMPLICIT, double3x3,
        ARG9(
            ARG(double, m00,),
            ARG(double, m01,),
            ARG(double, m02,),
            ARG(double, m10,),
            ARG(double, m11,),
            ARG(double, m12,),
            ARG(double, m20,),
            ARG(double, m21,),
            ARG(double, m22,)
        ), DS_MATRIX_ELEM_CONSTRUCTOR, 0)
    CONSTRUCTOR(IMPLICIT, double3x3,
        ARG3(
            ARG(double3, col0,),
            ARG(double3, col1,),
            ARG(double3, col2,)
        ), DS_ELEM_CONSTRUCTOR, 0)
    // conversion
    CONSTRUCTOR(IMPLICIT, double3x3, ARG1(ARG(float3x3, value,)), DS_CONV_CONSTRUCTOR, 0)
BUILTIN_TYPE_END(double3x3)

BUILTIN_TYPE_BEGIN(double3x4, 0)
    // default/diagonal constructor
    CONSTRUCTOR(EXPLICIT_WARN, double3x4,
        ARG1(
            DEFARG(double, value,, EXPR_LITERAL(0.0))
        ), DS_MATRIX_DIAG_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, double3x4, ARG1(ARG(double3x4, value,)), DS_COPY_CONSTRUCTOR, 0)
    // elemental constructor
    CONSTRUCTOR(IMPLICIT, double3x4,
        ARG12(
            ARG(double, m00,),
            ARG(double, m01,),
            ARG(double, m02,),
            ARG(double, m03,),
            ARG(double, m10,),
            ARG(double, m11,),
            ARG(double, m12,),
            ARG(double, m13,),
            ARG(double, m20,),
            ARG(double, m21,),
            ARG(double, m22,),
            ARG(double, m23,)
        ), DS_MATRIX_ELEM_CONSTRUCTOR, 0)
    CONSTRUCTOR(IMPLICIT, double3x4,
        ARG3(
            ARG(double4, col0,),
            ARG(double4, col1,),
            ARG(double4, col2,)
        ), DS_ELEM_CONSTRUCTOR, 0)
    // conversion
    CONSTRUCTOR(IMPLICIT, double3x4, ARG1(ARG(float3x4, value,)), DS_CONV_CONSTRUCTOR, 0)
BUILTIN_TYPE_END(double3x4)

BUILTIN_TYPE_BEGIN(double4x2, 0)
    // default/diagonal constructor
    CONSTRUCTOR(EXPLICIT_WARN, double4x2,
        ARG1(
            DEFARG(double, value,, EXPR_LITERAL(0.0))
        ), DS_MATRIX_DIAG_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, double4x2, ARG1(ARG(double4x2, value,)), DS_COPY_CONSTRUCTOR, 0)
    // elemental constructor
    CONSTRUCTOR(IMPLICIT, double4x2,
        ARG8(
            ARG(double, m00,),
            ARG(double, m01,),
            ARG(double, m10,),
            ARG(double, m11,),
            ARG(double, m20,),
            ARG(double, m21,),
            ARG(double, m30,),
            ARG(double, m31,)
        ), DS_MATRIX_ELEM_CONSTRUCTOR, 0)
    CONSTRUCTOR(IMPLICIT, double4x2,
        ARG4(
            ARG(double2, col0,),
            ARG(double2, col1,),
            ARG(double2, col2,),
            ARG(double2, col3,)
        ), DS_ELEM_CONSTRUCTOR, 0)
    // conversion
    CONSTRUCTOR(IMPLICIT, double4x2, ARG1(ARG(float4x2, value,)), DS_CONV_CONSTRUCTOR, 0)
BUILTIN_TYPE_END(double4x2)

BUILTIN_TYPE_BEGIN(double4x3, 0)
    // default/diagonal constructor
    CONSTRUCTOR(EXPLICIT_WARN, double4x3,
        ARG1(
            DEFARG(double, value,, EXPR_LITERAL(0.0))
        ), DS_MATRIX_DIAG_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, double4x3, ARG1(ARG(double4x3, value,)), DS_COPY_CONSTRUCTOR, 0)
    // elemental constructor
    CONSTRUCTOR(IMPLICIT, double4x3,
        ARG12(
            ARG(double, m00,),
            ARG(double, m01,),
            ARG(double, m02,),
            ARG(double, m10,),
            ARG(double, m11,),
            ARG(double, m12,),
            ARG(double, m20,),
            ARG(double, m21,),
            ARG(double, m22,),
            ARG(double, m30,),
            ARG(double, m31,),
            ARG(double, m32,)
        ), DS_MATRIX_ELEM_CONSTRUCTOR, 0)
    CONSTRUCTOR(IMPLICIT, double4x3,
        ARG4(
            ARG(double3, col0,),
            ARG(double3, col1,),
            ARG(double3, col2,),
            ARG(double3, col3,)
        ), DS_ELEM_CONSTRUCTOR, 0)
    // conversion
    CONSTRUCTOR(IMPLICIT, double4x3, ARG1(ARG(float4x3, value,)), DS_CONV_CONSTRUCTOR, 0)
BUILTIN_TYPE_END(double4x3)

BUILTIN_TYPE_BEGIN(double4x4, 0)
    // default/diagonal constructor
    CONSTRUCTOR(EXPLICIT_WARN, double4x4,
        ARG1(
            DEFARG(double, value,, EXPR_LITERAL(0.0))
        ), DS_MATRIX_DIAG_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, double4x4, ARG1(ARG(double4x4, value,)), DS_COPY_CONSTRUCTOR, 0)
    // elemental constructor
    CONSTRUCTOR(IMPLICIT, double4x4,
        ARG16(
            ARG(double, m00,),
            ARG(double, m01,),
            ARG(double, m02,),
            ARG(double, m03,),
            ARG(double, m10,),
            ARG(double, m11,),
            ARG(double, m12,),
            ARG(double, m13,),
            ARG(double, m20,),
            ARG(double, m21,),
            ARG(double, m22,),
            ARG(double, m23,),
            ARG(double, m30,),
            ARG(double, m31,),
            ARG(double, m32,),
            ARG(double, m33,)
        ), DS_MATRIX_ELEM_CONSTRUCTOR, 0)
    CONSTRUCTOR(IMPLICIT, double4x4,
        ARG4(
            ARG(double4, col0,),
            ARG(double4, col1,),
            ARG(double4, col2,),
            ARG(double4, col3,)
        ), DS_ELEM_CONSTRUCTOR, 0)
    // conversion
    CONSTRUCTOR(IMPLICIT, double4x4, ARG1(ARG(float4x4, value,)), DS_CONV_CONSTRUCTOR, 0)
BUILTIN_TYPE_END(double4x4)

BUILTIN_TYPE_BEGIN(light_profile, 0)
    // default constructor
    CONSTRUCTOR(IMPLICIT, light_profile, ARG0(), DS_INVALID_REF_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, light_profile, ARG1(ARG(light_profile, value,)), DS_COPY_CONSTRUCTOR, 0)
    // conversion
    CONSTRUCTOR(EXPLICIT, light_profile, ARG1(ARG(string, name,)), DS_CONV_CONSTRUCTOR, 0)
BUILTIN_TYPE_END(light_profile)

BUILTIN_TYPE_BEGIN(bsdf_measurement, SINCE_1_1)
    // default constructor
    CONSTRUCTOR(IMPLICIT, bsdf_measurement, ARG0(), DS_INVALID_REF_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, bsdf_measurement,
        ARG1(ARG(bsdf_measurement, value,)), DS_COPY_CONSTRUCTOR, 0)
    // conversion
    CONSTRUCTOR(EXPLICIT, bsdf_measurement, ARG1(ARG(string, name,)), DS_CONV_CONSTRUCTOR, 0)
BUILTIN_TYPE_END(bsdf_measurement)

// structs

// DF ------------------------------------------------------------------------

BUILTIN_TYPE_BEGIN(bsdf, 0)
    // default constructor
    CONSTRUCTOR(IMPLICIT, bsdf, ARG0(), DS_INVALID_REF_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, bsdf, ARG1(ARG(bsdf, value,)), DS_COPY_CONSTRUCTOR, 0)
BUILTIN_TYPE_END(bsdf)

BUILTIN_TYPE_BEGIN(hair_bsdf, 0)
    // default constructor
    CONSTRUCTOR(IMPLICIT, hair_bsdf, ARG0(), DS_INVALID_REF_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, hair_bsdf, ARG1(ARG(hair_bsdf, value,)), DS_COPY_CONSTRUCTOR, 0)
BUILTIN_TYPE_END(hair_bsdf)

BUILTIN_TYPE_BEGIN(edf, 0)
    // default constructor
    CONSTRUCTOR(IMPLICIT, edf, ARG0(), DS_INVALID_REF_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, edf, ARG1(ARG(edf, value,)), DS_COPY_CONSTRUCTOR, 0)
BUILTIN_TYPE_END(edf)

BUILTIN_TYPE_BEGIN(vdf, 0)
    // default constructor
    CONSTRUCTOR(IMPLICIT, vdf, ARG0(), DS_INVALID_REF_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, vdf, ARG1(ARG(vdf, value,)), DS_COPY_CONSTRUCTOR, 0)
BUILTIN_TYPE_END(vdf)

// Texture ----------------------------------------------------------------------

BUILTIN_TYPE_BEGIN(texture_2d, 0)
    // default constructor
    CONSTRUCTOR(IMPLICIT, texture_2d, ARG0(), DS_INVALID_REF_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, texture_2d, ARG1(UARG(texture_2d, value,)), DS_COPY_CONSTRUCTOR, 0)
    // resource constructor
    CONSTRUCTOR(EXPLICIT, texture_2d,
        ARG2(
            UARG(string, name,),
            UDEFARG(tex_gamma_mode, gamma,, EXPR_TEX_ENUM(gamma_default))),
        DS_TEXTURE_CONSTRUCTOR, 0)
BUILTIN_TYPE_END(texture_2d)

BUILTIN_TYPE_BEGIN(texture_3d, 0)
    // default constructor
    CONSTRUCTOR(IMPLICIT, texture_3d, ARG0(), DS_INVALID_REF_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, texture_3d, ARG1(UARG(texture_3d, value,)), DS_COPY_CONSTRUCTOR, 0)
    // resource constructor
    CONSTRUCTOR(EXPLICIT, texture_3d,
        ARG2(
            UARG(string, name,),
            UDEFARG(tex_gamma_mode, gamma,, EXPR_TEX_ENUM(gamma_default))),
        DS_TEXTURE_CONSTRUCTOR, 0)
BUILTIN_TYPE_END(texture_3d)

BUILTIN_TYPE_BEGIN(texture_cube, 0)
    // default constructor
    CONSTRUCTOR(IMPLICIT, texture_cube, ARG0(), DS_INVALID_REF_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, texture_cube, ARG1(UARG(texture_cube, value,)), DS_COPY_CONSTRUCTOR, 0)
    // resource constructor
    CONSTRUCTOR(EXPLICIT, texture_cube,
        ARG2(
            UARG(string, name,),
            UDEFARG(tex_gamma_mode, gamma,, EXPR_TEX_ENUM(gamma_default))),
        DS_TEXTURE_CONSTRUCTOR, 0)
BUILTIN_TYPE_END(texture_cube)

BUILTIN_TYPE_BEGIN(texture_ptex, 0)
    // default constructor
    CONSTRUCTOR(IMPLICIT, texture_ptex, ARG0(), DS_INVALID_REF_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, texture_ptex, ARG1(UARG(texture_ptex, value,)), DS_COPY_CONSTRUCTOR, 0)
    // resource constructor
    CONSTRUCTOR(EXPLICIT, texture_ptex,
        ARG2(
            UARG(string, name,),
            UDEFARG(tex_gamma_mode, gamma,, EXPR_TEX_ENUM(gamma_default))),
        DS_TEXTURE_CONSTRUCTOR, 0)
BUILTIN_TYPE_END(texture_ptex)

// Material ----------------------------------------------------------------------

BUILTIN_TYPE_BEGIN(intensity_mode, SINCE_1_1)
    ENUM_VALUE(intensity_mode, intensity_radiant_exitance, 0, SINCE_1_1)
    ENUM_VALUE(intensity_mode, intensity_power,            1, SINCE_1_1)

    // default/copy constructor
    CONSTRUCTOR(IMPLICIT, intensity_mode,
        ARG1(
            DEFARG(intensity_mode, value,, EXPR_INTENSITY_MODE_ENUM(intensity_radiant_exitance))
        ), DS_COPY_CONSTRUCTOR, SINCE_1_1)
    // conversion operator to int
    CONSTRUCTOR(IMPLICIT, int, ARG1(ARG(intensity_mode, value,)), DS_CONV_OPERATOR, SINCE_1_1)
BUILTIN_TYPE_END(intensity_mode)

BUILTIN_TYPE_BEGIN(material_emission, 0)
    FIELD(material_emission,, edf,            emission , 0)         // = edf();
    FIELD(material_emission,, color,          intensity, 0)         // = color(0.0);
    FIELD(material_emission,, intensity_mode, mode,      SINCE_1_1) // = intensity_radiant_exitance;

    // default constructor for MDL 1.0
    CONSTRUCTOR(IMPLICIT, material_emission,
        ARG2(
            DEFARG(edf,   emission,,  EXPR_CONSTRUCTOR(edf)),
            DEFARG(color, intensity,, EXPR_COLOR_LITERAL(0.0f))
        ), DS_ELEM_CONSTRUCTOR, REMOVED_1_1)
    // default constructor for MDL 1.1
    CONSTRUCTOR(IMPLICIT, material_emission,
        ARG3(
            DEFARG(edf,            emission,,  EXPR_CONSTRUCTOR(edf)),
            DEFARG(color,          intensity,, EXPR_COLOR_LITERAL(0.0f)),
            DEFARG(intensity_mode, mode,,      EXPR_INTENSITY_MODE_ENUM(intensity_radiant_exitance))
        ), DS_ELEM_CONSTRUCTOR, SINCE_1_1)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, material_emission,
        ARG1(
            ARG(material_emission, value,)
        ), DS_COPY_CONSTRUCTOR, 0)
BUILTIN_TYPE_END(material_emission)

BUILTIN_TYPE_BEGIN(material_surface, 0)
    FIELD(material_surface,, bsdf,              scattering, 0) // = bsdf();
    FIELD(material_surface,, material_emission, emission,   0) // = material_emission();

    // default constructor
    CONSTRUCTOR(IMPLICIT, material_surface,
        ARG2(
            DEFARG(bsdf,              scattering,, EXPR_CONSTRUCTOR(bsdf)),
            DEFARG(material_emission, emission,,   EXPR_CONSTRUCTOR(material_emission))
        ), DS_ELEM_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, material_surface,
        ARG1(
            ARG(material_surface, value,)
        ), DS_COPY_CONSTRUCTOR, 0)
BUILTIN_TYPE_END(material_surface)

BUILTIN_TYPE_BEGIN(material_volume, 0)
    FIELD(material_volume,, vdf,   scattering,             0) // = vdf();
    FIELD(material_volume,, color, absorption_coefficient, 0) // = 0.0;
    FIELD(material_volume,, color, scattering_coefficient, 0) // = 0.0;

    // default constructor
    CONSTRUCTOR(IMPLICIT, material_volume,
        ARG3(
            DEFARG(vdf,   scattering,,             EXPR_CONSTRUCTOR(vdf)),
            DEFARG(color, absorption_coefficient,, EXPR_COLOR_LITERAL(0.0f)),
            DEFARG(color, scattering_coefficient,, EXPR_COLOR_LITERAL(0.0f))
        ), DS_ELEM_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, material_volume,
        ARG1(
            ARG(material_volume, value,)
        ), DS_COPY_CONSTRUCTOR, 0)
BUILTIN_TYPE_END(material_volume)

BUILTIN_TYPE_BEGIN(material_geometry, 0)
    FIELD(material_geometry,        , float3, displacement,   0) // = float3(0.0);
    FIELD(material_geometry,        , float,  cutout_opacity, 0) // = 1.0f;
    FIELD(material_geometry,        , float3, normal,         0) // = state::normal;

    // default constructor
    CONSTRUCTOR(IMPLICIT, material_geometry,
        ARG3(
            DEFARG(float3, displacement,,   EXPR_FLOAT3_LITERAL(0.0f)),
            DEFARG(float,  cutout_opacity,, EXPR_LITERAL(1.0f)),
            DEFARG(float3, normal,,         EXPR_STATE(float3, normal))
        ), DS_ELEM_CONSTRUCTOR, 0)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, material_geometry,
        ARG1(
            ARG(material_geometry, value,)
        ), DS_COPY_CONSTRUCTOR, 0)
BUILTIN_TYPE_END(material_geometry)

BUILTIN_TYPE_BEGIN(material, 0)
    FIELD(material, uniform, bool,              thin_walled, 0) // = false;
    FIELD(material,        , material_surface,  surface,     0) // = material_surface();
    FIELD(material,        , material_surface,  backface,    0) // = material_surface();
    FIELD(material, uniform, color,             ior,         0) // = color(1.0);
    FIELD(material,        , material_volume,   volume,      0) // = material_volume();
    FIELD(material,        , material_geometry, geometry,    0) // = material_geometry();
    FIELD(material,        , hair_bsdf,         hair,        SINCE_1_5) // = hair_bsdf();

    // default constructor until MDL 1.5
    CONSTRUCTOR(IMPLICIT, material,
        ARG6(
            UDEFARG(bool,              thin_walled,, EXPR_LITERAL(false)),
            DEFARG(material_surface,   surface,,     EXPR_CONSTRUCTOR(material_surface)),
            DEFARG(material_surface,   backface,,    EXPR_CONSTRUCTOR(material_surface)),
            UDEFARG(color,             ior,,         EXPR_COLOR_LITERAL(1.0f)),
            DEFARG(material_volume,    volume,,      EXPR_CONSTRUCTOR(material_volume)),
            DEFARG(material_geometry,  geometry,,    EXPR_CONSTRUCTOR(material_geometry))
        ), DS_ELEM_CONSTRUCTOR, REMOVED_1_5)
    // default constructor from MDL 1.5
    CONSTRUCTOR(IMPLICIT, material,
        ARG7(
            UDEFARG(bool,              thin_walled,, EXPR_LITERAL(false)),
            DEFARG(material_surface,   surface,,     EXPR_CONSTRUCTOR(material_surface)),
            DEFARG(material_surface,   backface,,    EXPR_CONSTRUCTOR(material_surface)),
            UDEFARG(color,             ior,,         EXPR_COLOR_LITERAL(1.0f)),
            DEFARG(material_volume,    volume,,      EXPR_CONSTRUCTOR(material_volume)),
            DEFARG(material_geometry,  geometry,,    EXPR_CONSTRUCTOR(material_geometry)),
            DEFARG(hair_bsdf,          hair,,        EXPR_CONSTRUCTOR(hair_bsdf))
        ), DS_ELEM_CONSTRUCTOR, SINCE_1_5)
    // copy constructor
    CONSTRUCTOR(IMPLICIT, material,
        ARG1(
            ARG(material, value,)
        ), DS_COPY_CONSTRUCTOR, 0)
BUILTIN_TYPE_END(material)

// -------------------------- operators -------------------------------

// operator== and operator!=
EQ_OPERATORS(ARG2(ARG(bool, x,),    ARG(bool, y,)), 0)
EQ_OPERATORS(ARG2(ARG(bool2, x,),   ARG(bool2, y,)), 0)
EQ_OPERATORS(ARG2(ARG(bool, x,),    ARG(bool2, y,)), 0)
EQ_OPERATORS(ARG2(ARG(bool2, x,),   ARG(bool, y,)), 0)
EQ_OPERATORS(ARG2(ARG(bool3, x,),   ARG(bool3, y,)), 0)
EQ_OPERATORS(ARG2(ARG(bool, x,),    ARG(bool3, y,)), 0)
EQ_OPERATORS(ARG2(ARG(bool3, x,),   ARG(bool, y,)), 0)
EQ_OPERATORS(ARG2(ARG(bool4, x,),   ARG(bool4, y,)), 0)
EQ_OPERATORS(ARG2(ARG(bool, x,),    ARG(bool4, y,)), 0)
EQ_OPERATORS(ARG2(ARG(bool4, x,),   ARG(bool, y,)), 0)
EQ_OPERATORS(ARG2(ARG(int, x,),     ARG(int, y,)), 0)
EQ_OPERATORS(ARG2(ARG(int2, x,),    ARG(int2, y,)), 0)
EQ_OPERATORS(ARG2(ARG(int, x,),     ARG(int2, y,)), 0)
EQ_OPERATORS(ARG2(ARG(int2, x,),    ARG(int, y,)), 0)
EQ_OPERATORS(ARG2(ARG(int3, x,),    ARG(int3, y,)), 0)
EQ_OPERATORS(ARG2(ARG(int, x,),     ARG(int3, y,)), 0)
EQ_OPERATORS(ARG2(ARG(int3, x,),    ARG(int, y,)), 0)
EQ_OPERATORS(ARG2(ARG(int4, x,),    ARG(int4, y,)), 0)
EQ_OPERATORS(ARG2(ARG(int, x,),     ARG(int4, y,)), 0)
EQ_OPERATORS(ARG2(ARG(int4, x,),    ARG(int, y,)), 0)
EQ_OPERATORS(ARG2(ARG(float, x,),   ARG(float, y,)), 0)
EQ_OPERATORS(ARG2(ARG(float2, x,),  ARG(float2, y,)), 0)
EQ_OPERATORS(ARG2(ARG(float, x,),   ARG(float2, y,)), 0)
EQ_OPERATORS(ARG2(ARG(float2, x,),  ARG(float, y,)), 0)
EQ_OPERATORS(ARG2(ARG(float3, x,),  ARG(float3, y,)), 0)
EQ_OPERATORS(ARG2(ARG(float, x,),   ARG(float3, y,)), 0)
EQ_OPERATORS(ARG2(ARG(float3, x,),  ARG(float, y,)), 0)
EQ_OPERATORS(ARG2(ARG(float4, x,),  ARG(float4, y,)), 0)
EQ_OPERATORS(ARG2(ARG(float, x,),   ARG(float4, y,)), 0)
EQ_OPERATORS(ARG2(ARG(float4, x,),  ARG(float, y,)), 0)
EQ_OPERATORS(ARG2(ARG(double, x,),  ARG(double, y,)), 0)
EQ_OPERATORS(ARG2(ARG(double2, x,), ARG(double2, y,)), 0)
EQ_OPERATORS(ARG2(ARG(double, x,),  ARG(double2, y,)), 0)
EQ_OPERATORS(ARG2(ARG(double2, x,), ARG(double, y,)), 0)
EQ_OPERATORS(ARG2(ARG(double3, x,), ARG(double3, y,)), 0)
EQ_OPERATORS(ARG2(ARG(double, x,),  ARG(double3, y,)), 0)
EQ_OPERATORS(ARG2(ARG(double3, x,), ARG(double, y,)), 0)
EQ_OPERATORS(ARG2(ARG(double4, x,), ARG(double4, y,)), 0)
EQ_OPERATORS(ARG2(ARG(double, x,),  ARG(double4, y,)), 0)
EQ_OPERATORS(ARG2(ARG(double4, x,), ARG(double, y,)), 0)
EQ_OPERATORS(ARG2(ARG(color, x,),   ARG(color, y,)), 0)
EQ_OPERATORS(ARG2(ARG(string, x,),  ARG(string, y,)), 0)

EQ_OPERATORS(ARG2(ARG(float2x2, x,),  ARG(float2x2, y,)), 0)
EQ_OPERATORS(ARG2(ARG(float2x3, x,),  ARG(float2x3, y,)), 0)
EQ_OPERATORS(ARG2(ARG(float2x4, x,),  ARG(float2x4, y,)), 0)
EQ_OPERATORS(ARG2(ARG(float3x2, x,),  ARG(float3x2, y,)), 0)
EQ_OPERATORS(ARG2(ARG(float3x3, x,),  ARG(float3x3, y,)), 0)
EQ_OPERATORS(ARG2(ARG(float3x4, x,),  ARG(float3x4, y,)), 0)
EQ_OPERATORS(ARG2(ARG(float4x2, x,),  ARG(float4x2, y,)), 0)
EQ_OPERATORS(ARG2(ARG(float4x3, x,),  ARG(float4x3, y,)), 0)
EQ_OPERATORS(ARG2(ARG(float4x4, x,),  ARG(float4x4, y,)), 0)

EQ_OPERATORS(ARG2(ARG(double2x2, x,),  ARG(double2x2, y,)), 0)
EQ_OPERATORS(ARG2(ARG(double2x3, x,),  ARG(double2x3, y,)), 0)
EQ_OPERATORS(ARG2(ARG(double2x4, x,),  ARG(double2x4, y,)), 0)
EQ_OPERATORS(ARG2(ARG(double3x2, x,),  ARG(double3x2, y,)), 0)
EQ_OPERATORS(ARG2(ARG(double3x3, x,),  ARG(double3x3, y,)), 0)
EQ_OPERATORS(ARG2(ARG(double3x4, x,),  ARG(double3x4, y,)), 0)
EQ_OPERATORS(ARG2(ARG(double4x2, x,),  ARG(double4x2, y,)), 0)
EQ_OPERATORS(ARG2(ARG(double4x3, x,),  ARG(double4x3, y,)), 0)
EQ_OPERATORS(ARG2(ARG(double4x4, x,),  ARG(double4x4, y,)), 0)

// operator<, operator<=, opertator>, operator>=
REL_OPERATORS(ARG2(ARG(int, x,),     ARG(int, y,)), 0)
REL_OPERATORS(ARG2(ARG(float, x,),   ARG(float, y,)), 0)
REL_OPERATORS(ARG2(ARG(double, x,),  ARG(double, y,)), 0)

// operator!
OPERATOR(bool,  OK_LOGICAL_NOT, ARG1(ARG(bool, x,)), 0)
OPERATOR(bool2, OK_LOGICAL_NOT, ARG1(ARG(bool2, x,)), 0)
OPERATOR(bool3, OK_LOGICAL_NOT, ARG1(ARG(bool3, x,)), 0)
OPERATOR(bool4, OK_LOGICAL_NOT, ARG1(ARG(bool4, x,)), 0)

// operator&&, operator||
BIN_LOG_OPERATORS(bool,  ARG2(ARG(bool, x,),  ARG(bool, y,)), 0)
BIN_LOG_OPERATORS(bool2, ARG2(ARG(bool2, x,), ARG(bool2, y,)), 0)
BIN_LOG_OPERATORS(bool2, ARG2(ARG(bool, x,),  ARG(bool2, y,)), 0)
BIN_LOG_OPERATORS(bool2, ARG2(ARG(bool2, x,), ARG(bool, y,)), 0)
BIN_LOG_OPERATORS(bool3, ARG2(ARG(bool3, x,), ARG(bool3, y,)), 0)
BIN_LOG_OPERATORS(bool3, ARG2(ARG(bool, x,),  ARG(bool3, y,)), 0)
BIN_LOG_OPERATORS(bool3, ARG2(ARG(bool3, x,), ARG(bool, y,)), 0)
BIN_LOG_OPERATORS(bool4, ARG2(ARG(bool4, x,), ARG(bool4, y,)), 0)
BIN_LOG_OPERATORS(bool4, ARG2(ARG(bool, x,),  ARG(bool4, y,)), 0)
BIN_LOG_OPERATORS(bool4, ARG2(ARG(bool4, x,), ARG(bool, y,)), 0)

// operator%
OPERATOR(int,  OK_MODULO, ARG2(ARG(int, x,),  ARG(int, y,) ), 0)
OPERATOR(int2, OK_MODULO, ARG2(ARG(int2, x,), ARG(int2, y,)), 0)
OPERATOR(int2, OK_MODULO, ARG2(ARG(int, x,),  ARG(int2, y,)), 0)
OPERATOR(int2, OK_MODULO, ARG2(ARG(int2, x,), ARG(int, y,)), 0)
OPERATOR(int3, OK_MODULO, ARG2(ARG(int3, x,), ARG(int3, y,)), 0)
OPERATOR(int3, OK_MODULO, ARG2(ARG(int, x,),  ARG(int3, y,)), 0)
OPERATOR(int3, OK_MODULO, ARG2(ARG(int3, x,), ARG(int, y,)), 0)
OPERATOR(int4, OK_MODULO, ARG2(ARG(int4, x,), ARG(int4, y,)), 0)
OPERATOR(int4, OK_MODULO, ARG2(ARG(int, x,),  ARG(int4, y,)), 0)
OPERATOR(int4, OK_MODULO, ARG2(ARG(int4, x,), ARG(int, y,)), 0)

// operator%=
OPERATOR(int,  OK_MODULO_ASSIGN, ARG2(ARG(int, x,),  ARG(int, y,) ), 0)
OPERATOR(int2, OK_MODULO_ASSIGN, ARG2(ARG(int2, x,), ARG(int2, y,)), 0)
OPERATOR(int2, OK_MODULO_ASSIGN, ARG2(ARG(int2, x,), ARG(int, y,)), 0)
OPERATOR(int3, OK_MODULO_ASSIGN, ARG2(ARG(int3, x,), ARG(int3, y,)), 0)
OPERATOR(int3, OK_MODULO_ASSIGN, ARG2(ARG(int3, x,), ARG(int, y,)), 0)
OPERATOR(int4, OK_MODULO_ASSIGN, ARG2(ARG(int4, x,), ARG(int4, y,)), 0)
OPERATOR(int4, OK_MODULO_ASSIGN, ARG2(ARG(int4, x,), ARG(int, y,)), 0)

// operator<<
OPERATOR(int,  OK_SHIFT_LEFT, ARG2(ARG(int, x,),  ARG(int, y,) ), 0)
OPERATOR(int2, OK_SHIFT_LEFT, ARG2(ARG(int2, x,), ARG(int2, y,)), 0)
OPERATOR(int2, OK_SHIFT_LEFT, ARG2(ARG(int, x,),  ARG(int2, y,)), 0)
OPERATOR(int2, OK_SHIFT_LEFT, ARG2(ARG(int2, x,), ARG(int, y,)), 0)
OPERATOR(int3, OK_SHIFT_LEFT, ARG2(ARG(int3, x,), ARG(int3, y,)), 0)
OPERATOR(int3, OK_SHIFT_LEFT, ARG2(ARG(int, x,),  ARG(int3, y,)), 0)
OPERATOR(int3, OK_SHIFT_LEFT, ARG2(ARG(int3, x,), ARG(int, y,)), 0)
OPERATOR(int4, OK_SHIFT_LEFT, ARG2(ARG(int4, x,), ARG(int4, y,)), 0)
OPERATOR(int4, OK_SHIFT_LEFT, ARG2(ARG(int, x,),  ARG(int4, y,)), 0)
OPERATOR(int4, OK_SHIFT_LEFT, ARG2(ARG(int4, x,), ARG(int, y,)), 0)

// operator>>
OPERATOR(int,  OK_SHIFT_RIGHT, ARG2(ARG(int, x,),  ARG(int, y,) ), 0)
OPERATOR(int2, OK_SHIFT_RIGHT, ARG2(ARG(int2, x,), ARG(int2, y,)), 0)
OPERATOR(int2, OK_SHIFT_RIGHT, ARG2(ARG(int, x,),  ARG(int2, y,)), 0)
OPERATOR(int2, OK_SHIFT_RIGHT, ARG2(ARG(int2, x,), ARG(int, y,)), 0)
OPERATOR(int3, OK_SHIFT_RIGHT, ARG2(ARG(int3, x,), ARG(int3, y,)), 0)
OPERATOR(int3, OK_SHIFT_RIGHT, ARG2(ARG(int, x,),  ARG(int3, y,)), 0)
OPERATOR(int3, OK_SHIFT_RIGHT, ARG2(ARG(int3, x,), ARG(int, y,)), 0)
OPERATOR(int4, OK_SHIFT_RIGHT, ARG2(ARG(int4, x,), ARG(int4, y,)), 0)
OPERATOR(int4, OK_SHIFT_RIGHT, ARG2(ARG(int, x,),  ARG(int4, y,)), 0)
OPERATOR(int4, OK_SHIFT_RIGHT, ARG2(ARG(int4, x,), ARG(int, y,)), 0)

// operator>>>
OPERATOR(int,  OK_UNSIGNED_SHIFT_RIGHT, ARG2(ARG(int, x,),  ARG(int, y,) ), 0)
OPERATOR(int2, OK_UNSIGNED_SHIFT_RIGHT, ARG2(ARG(int2, x,), ARG(int2, y,)), 0)
OPERATOR(int2, OK_UNSIGNED_SHIFT_RIGHT, ARG2(ARG(int, x,),  ARG(int2, y,)), 0)
OPERATOR(int2, OK_UNSIGNED_SHIFT_RIGHT, ARG2(ARG(int2, x,), ARG(int, y,)), 0)
OPERATOR(int3, OK_UNSIGNED_SHIFT_RIGHT, ARG2(ARG(int3, x,), ARG(int3, y,)), 0)
OPERATOR(int3, OK_UNSIGNED_SHIFT_RIGHT, ARG2(ARG(int, x,),  ARG(int3, y,)), 0)
OPERATOR(int3, OK_UNSIGNED_SHIFT_RIGHT, ARG2(ARG(int3, x,), ARG(int, y,)), 0)
OPERATOR(int4, OK_UNSIGNED_SHIFT_RIGHT, ARG2(ARG(int4, x,), ARG(int4, y,)), 0)
OPERATOR(int4, OK_UNSIGNED_SHIFT_RIGHT, ARG2(ARG(int, x,),  ARG(int4, y,)), 0)
OPERATOR(int4, OK_UNSIGNED_SHIFT_RIGHT, ARG2(ARG(int4, x,), ARG(int, y,)), 0)

// operator<<=
OPERATOR(int,  OK_SHIFT_LEFT_ASSIGN, ARG2(ARG(int, x,),  ARG(int, y,) ), 0)
OPERATOR(int2, OK_SHIFT_LEFT_ASSIGN, ARG2(ARG(int2, x,), ARG(int2, y,)), 0)
OPERATOR(int2, OK_SHIFT_LEFT_ASSIGN, ARG2(ARG(int2, x,), ARG(int, y,)), 0)
OPERATOR(int3, OK_SHIFT_LEFT_ASSIGN, ARG2(ARG(int3, x,), ARG(int3, y,)), 0)
OPERATOR(int3, OK_SHIFT_LEFT_ASSIGN, ARG2(ARG(int3, x,), ARG(int, y,)), 0)
OPERATOR(int4, OK_SHIFT_LEFT_ASSIGN, ARG2(ARG(int4, x,), ARG(int4, y,)), 0)
OPERATOR(int4, OK_SHIFT_LEFT_ASSIGN, ARG2(ARG(int4, x,), ARG(int, y,)), 0)

// operator>>=
OPERATOR(int,  OK_SHIFT_RIGHT_ASSIGN, ARG2(ARG(int, x,),  ARG(int, y,) ), 0)
OPERATOR(int2, OK_SHIFT_RIGHT_ASSIGN, ARG2(ARG(int2, x,), ARG(int2, y,)), 0)
OPERATOR(int2, OK_SHIFT_RIGHT_ASSIGN, ARG2(ARG(int2, x,), ARG(int, y,)), 0)
OPERATOR(int3, OK_SHIFT_RIGHT_ASSIGN, ARG2(ARG(int3, x,), ARG(int3, y,)), 0)
OPERATOR(int3, OK_SHIFT_RIGHT_ASSIGN, ARG2(ARG(int3, x,), ARG(int, y,)), 0)
OPERATOR(int4, OK_SHIFT_RIGHT_ASSIGN, ARG2(ARG(int4, x,), ARG(int4, y,)), 0)
OPERATOR(int4, OK_SHIFT_RIGHT_ASSIGN, ARG2(ARG(int4, x,), ARG(int, y,)), 0)

// operator>>>=
OPERATOR(int,  OK_UNSIGNED_SHIFT_RIGHT_ASSIGN, ARG2(ARG(int, x,),  ARG(int, y,) ), 0)
OPERATOR(int2, OK_UNSIGNED_SHIFT_RIGHT_ASSIGN, ARG2(ARG(int2, x,), ARG(int2, y,)), 0)
OPERATOR(int2, OK_UNSIGNED_SHIFT_RIGHT_ASSIGN, ARG2(ARG(int2, x,), ARG(int, y,)), 0)
OPERATOR(int3, OK_UNSIGNED_SHIFT_RIGHT_ASSIGN, ARG2(ARG(int3, x,), ARG(int3, y,)), 0)
OPERATOR(int3, OK_UNSIGNED_SHIFT_RIGHT_ASSIGN, ARG2(ARG(int3, x,), ARG(int, y,)), 0)
OPERATOR(int4, OK_UNSIGNED_SHIFT_RIGHT_ASSIGN, ARG2(ARG(int4, x,), ARG(int4, y,)), 0)
OPERATOR(int4, OK_UNSIGNED_SHIFT_RIGHT_ASSIGN, ARG2(ARG(int4, x,), ARG(int, y,)), 0)

// operator~
OPERATOR(int,  OK_BITWISE_COMPLEMENT, ARG1(ARG(int,  x,)), 0)
OPERATOR(int2, OK_BITWISE_COMPLEMENT, ARG1(ARG(int2, x,)), 0)
OPERATOR(int3, OK_BITWISE_COMPLEMENT, ARG1(ARG(int3, x,)), 0)
OPERATOR(int4, OK_BITWISE_COMPLEMENT, ARG1(ARG(int4, x,)), 0)

// operator&
OPERATOR(int,  OK_BITWISE_AND, ARG2(ARG(int, x,),  ARG(int, y,) ), 0)
OPERATOR(int2, OK_BITWISE_AND, ARG2(ARG(int2, x,), ARG(int2, y,)), 0)
OPERATOR(int2, OK_BITWISE_AND, ARG2(ARG(int, x,),  ARG(int2, y,)), 0)
OPERATOR(int2, OK_BITWISE_AND, ARG2(ARG(int2, x,), ARG(int, y,)), 0)
OPERATOR(int3, OK_BITWISE_AND, ARG2(ARG(int3, x,), ARG(int3, y,)), 0)
OPERATOR(int3, OK_BITWISE_AND, ARG2(ARG(int, x,),  ARG(int3, y,)), 0)
OPERATOR(int3, OK_BITWISE_AND, ARG2(ARG(int3, x,), ARG(int, y,)), 0)
OPERATOR(int4, OK_BITWISE_AND, ARG2(ARG(int4, x,), ARG(int4, y,)), 0)
OPERATOR(int4, OK_BITWISE_AND, ARG2(ARG(int, x,),  ARG(int4, y,)), 0)
OPERATOR(int4, OK_BITWISE_AND, ARG2(ARG(int4, x,), ARG(int, y,)), 0)

// operator|
OPERATOR(int,  OK_BITWISE_OR, ARG2(ARG(int, x,),  ARG(int, y,) ), 0)
OPERATOR(int2, OK_BITWISE_OR, ARG2(ARG(int2, x,), ARG(int2, y,)), 0)
OPERATOR(int2, OK_BITWISE_OR, ARG2(ARG(int, x,),  ARG(int2, y,)), 0)
OPERATOR(int2, OK_BITWISE_OR, ARG2(ARG(int2, x,), ARG(int, y,)), 0)
OPERATOR(int3, OK_BITWISE_OR, ARG2(ARG(int3, x,), ARG(int3, y,)), 0)
OPERATOR(int3, OK_BITWISE_OR, ARG2(ARG(int, x,),  ARG(int3, y,)), 0)
OPERATOR(int3, OK_BITWISE_OR, ARG2(ARG(int3, x,), ARG(int, y,)), 0)
OPERATOR(int4, OK_BITWISE_OR, ARG2(ARG(int4, x,), ARG(int4, y,)), 0)
OPERATOR(int4, OK_BITWISE_OR, ARG2(ARG(int, x,),  ARG(int4, y,)), 0)
OPERATOR(int4, OK_BITWISE_OR, ARG2(ARG(int4, x,), ARG(int, y,)), 0)

// operator^
OPERATOR(int,  OK_BITWISE_XOR, ARG2(ARG(int, x,),  ARG(int, y,) ), 0)
OPERATOR(int2, OK_BITWISE_XOR, ARG2(ARG(int2, x,), ARG(int2, y,)), 0)
OPERATOR(int2, OK_BITWISE_XOR, ARG2(ARG(int, x,),  ARG(int2, y,)), 0)
OPERATOR(int2, OK_BITWISE_XOR, ARG2(ARG(int2, x,), ARG(int, y,)), 0)
OPERATOR(int3, OK_BITWISE_XOR, ARG2(ARG(int3, x,), ARG(int3, y,)), 0)
OPERATOR(int3, OK_BITWISE_XOR, ARG2(ARG(int, x,),  ARG(int3, y,)), 0)
OPERATOR(int3, OK_BITWISE_XOR, ARG2(ARG(int3, x,), ARG(int, y,)), 0)
OPERATOR(int4, OK_BITWISE_XOR, ARG2(ARG(int4, x,), ARG(int4, y,)), 0)
OPERATOR(int4, OK_BITWISE_XOR, ARG2(ARG(int, x,),  ARG(int4, y,)), 0)
OPERATOR(int4, OK_BITWISE_XOR, ARG2(ARG(int4, x,), ARG(int, y,)), 0)

// operator&=
OPERATOR(int,  OK_BITWISE_AND_ASSIGN, ARG2(ARG(int, x,),  ARG(int, y,) ), 0)
OPERATOR(int2, OK_BITWISE_AND_ASSIGN, ARG2(ARG(int2, x,), ARG(int2, y,)), 0)
OPERATOR(int2, OK_BITWISE_AND_ASSIGN, ARG2(ARG(int2, x,), ARG(int, y,)), 0)
OPERATOR(int3, OK_BITWISE_AND_ASSIGN, ARG2(ARG(int3, x,), ARG(int3, y,)), 0)
OPERATOR(int3, OK_BITWISE_AND_ASSIGN, ARG2(ARG(int3, x,), ARG(int, y,)), 0)
OPERATOR(int4, OK_BITWISE_AND_ASSIGN, ARG2(ARG(int4, x,), ARG(int4, y,)), 0)
OPERATOR(int4, OK_BITWISE_AND_ASSIGN, ARG2(ARG(int4, x,), ARG(int, y,)), 0)

// operator|=
OPERATOR(int,  OK_BITWISE_OR_ASSIGN, ARG2(ARG(int, x,),  ARG(int, y,) ), 0)
OPERATOR(int2, OK_BITWISE_OR_ASSIGN, ARG2(ARG(int2, x,), ARG(int2, y,)), 0)
OPERATOR(int2, OK_BITWISE_OR_ASSIGN, ARG2(ARG(int2, x,), ARG(int, y,)), 0)
OPERATOR(int3, OK_BITWISE_OR_ASSIGN, ARG2(ARG(int3, x,), ARG(int3, y,)), 0)
OPERATOR(int3, OK_BITWISE_OR_ASSIGN, ARG2(ARG(int3, x,), ARG(int, y,)), 0)
OPERATOR(int4, OK_BITWISE_OR_ASSIGN, ARG2(ARG(int4, x,), ARG(int4, y,)), 0)
OPERATOR(int4, OK_BITWISE_OR_ASSIGN, ARG2(ARG(int4, x,), ARG(int, y,)), 0)

// operator^=
OPERATOR(int,  OK_BITWISE_XOR_ASSIGN, ARG2(ARG(int, x,),  ARG(int, y,) ), 0)
OPERATOR(int2, OK_BITWISE_XOR_ASSIGN, ARG2(ARG(int2, x,), ARG(int2, y,)), 0)
OPERATOR(int2, OK_BITWISE_XOR_ASSIGN, ARG2(ARG(int2, x,), ARG(int, y,)), 0)
OPERATOR(int3, OK_BITWISE_XOR_ASSIGN, ARG2(ARG(int3, x,), ARG(int3, y,)), 0)
OPERATOR(int3, OK_BITWISE_XOR_ASSIGN, ARG2(ARG(int3, x,), ARG(int, y,)), 0)
OPERATOR(int4, OK_BITWISE_XOR_ASSIGN, ARG2(ARG(int4, x,), ARG(int4, y,)), 0)
OPERATOR(int4, OK_BITWISE_XOR_ASSIGN, ARG2(ARG(int4, x,), ARG(int, y,)), 0)

// prefix/postfix operator++, prefix/postfix operator--
INC_DEC_OPERATORS(int,     ARG1(ARG(int, x,)), 0)
INC_DEC_OPERATORS(int2,    ARG1(ARG(int2, x,)), 0)
INC_DEC_OPERATORS(int3,    ARG1(ARG(int3, x,)), 0)
INC_DEC_OPERATORS(int4,    ARG1(ARG(int4, x,)), 0)
INC_DEC_OPERATORS(float,   ARG1(ARG(float, x,)), 0)
INC_DEC_OPERATORS(float2,  ARG1(ARG(float2, x,)), 0)
INC_DEC_OPERATORS(float3,  ARG1(ARG(float3, x,)), 0)
INC_DEC_OPERATORS(float4,  ARG1(ARG(float4, x,)), 0)
INC_DEC_OPERATORS(double,  ARG1(ARG(double, x,)), 0)
INC_DEC_OPERATORS(double2, ARG1(ARG(double2, x,)), 0)
INC_DEC_OPERATORS(double3, ARG1(ARG(double3, x,)), 0)
INC_DEC_OPERATORS(double4, ARG1(ARG(double4, x,)), 0)

// operator=
ASSIGN_OPERATOR(bool,    ARG2(ARG(bool, x,),    ARG(bool, y,)), 0)
ASSIGN_OPERATOR(bool2,   ARG2(ARG(bool2, x,),   ARG(bool2, y,)), 0)
ASSIGN_OPERATOR(bool2,   ARG2(ARG(bool2, x,),   ARG(bool, y,)), 0)
ASSIGN_OPERATOR(bool3,   ARG2(ARG(bool3, x,),   ARG(bool3, y,)), 0)
ASSIGN_OPERATOR(bool3,   ARG2(ARG(bool3, x,),   ARG(bool, y,)), 0)
ASSIGN_OPERATOR(bool4,   ARG2(ARG(bool4, x,),   ARG(bool4, y,)), 0)
ASSIGN_OPERATOR(bool4,   ARG2(ARG(bool4, x,),   ARG(bool, y,)), 0)
ASSIGN_OPERATOR(int,     ARG2(ARG(int, x,),     ARG(int, y,)), 0)
ASSIGN_OPERATOR(int2,    ARG2(ARG(int2, x,),    ARG(int2, y,)), 0)
ASSIGN_OPERATOR(int2,    ARG2(ARG(int2, x,),    ARG(int, y,)), 0)
ASSIGN_OPERATOR(int3,    ARG2(ARG(int3, x,),    ARG(int3, y,)), 0)
ASSIGN_OPERATOR(int3,    ARG2(ARG(int3, x,),    ARG(int, y,)), 0)
ASSIGN_OPERATOR(int4,    ARG2(ARG(int4, x,),    ARG(int4, y,)), 0)
ASSIGN_OPERATOR(int4,    ARG2(ARG(int4, x,),    ARG(int, y,)), 0)
ASSIGN_OPERATOR(float,   ARG2(ARG(float, x,),   ARG(float, y,)), 0)
ASSIGN_OPERATOR(float2,  ARG2(ARG(float2, x,),  ARG(float2, y,)), 0)
ASSIGN_OPERATOR(float2,  ARG2(ARG(float2, x,),  ARG(float, y,)), 0)
ASSIGN_OPERATOR(float3,  ARG2(ARG(float3, x,),  ARG(float3, y,)), 0)
ASSIGN_OPERATOR(float3,  ARG2(ARG(float3, x,),  ARG(float, y,)), 0)
ASSIGN_OPERATOR(float4,  ARG2(ARG(float4, x,),  ARG(float4, y,)), 0)
ASSIGN_OPERATOR(float4,  ARG2(ARG(float4, x,),  ARG(float, y,)), 0)
ASSIGN_OPERATOR(double,  ARG2(ARG(double, x,),  ARG(double, y,)), 0)
ASSIGN_OPERATOR(double2, ARG2(ARG(double2, x,), ARG(double2, y,)), 0)
ASSIGN_OPERATOR(double2, ARG2(ARG(double2, x,), ARG(double, y,)), 0)
ASSIGN_OPERATOR(double3, ARG2(ARG(double3, x,), ARG(double3, y,)), 0)
ASSIGN_OPERATOR(double3, ARG2(ARG(double3, x,), ARG(double, y,)), 0)
ASSIGN_OPERATOR(double4, ARG2(ARG(double4, x,), ARG(double4, y,)), 0)
ASSIGN_OPERATOR(double4, ARG2(ARG(double4, x,), ARG(double, y,)), 0)

ASSIGN_OPERATOR(float2x2,   ARG2(ARG(float2x2, x,),  ARG(float2x2, y,)), 0)
ASSIGN_OPERATOR(float2x3,   ARG2(ARG(float2x3, x,),  ARG(float2x3, y,)), 0)
ASSIGN_OPERATOR(float2x4,   ARG2(ARG(float2x4, x,),  ARG(float2x4, y,)), 0)
ASSIGN_OPERATOR(float3x2,   ARG2(ARG(float3x2, x,),  ARG(float3x2, y,)), 0)
ASSIGN_OPERATOR(float3x3,   ARG2(ARG(float3x3, x,),  ARG(float3x3, y,)), 0)
ASSIGN_OPERATOR(float3x4,   ARG2(ARG(float3x4, x,),  ARG(float3x4, y,)), 0)
ASSIGN_OPERATOR(float4x2,   ARG2(ARG(float4x2, x,),  ARG(float4x2, y,)), 0)
ASSIGN_OPERATOR(float4x3,   ARG2(ARG(float4x3, x,),  ARG(float4x3, y,)), 0)
ASSIGN_OPERATOR(float4x4,   ARG2(ARG(float4x4, x,),  ARG(float4x4, y,)), 0)

ASSIGN_OPERATOR(double2x2,  ARG2(ARG(double2x2, x,),  ARG(double2x2, y,)), 0)
ASSIGN_OPERATOR(double2x3,  ARG2(ARG(double2x3, x,),  ARG(double2x3, y,)), 0)
ASSIGN_OPERATOR(double2x4,  ARG2(ARG(double2x4, x,),  ARG(double2x4, y,)), 0)
ASSIGN_OPERATOR(double3x2,  ARG2(ARG(double3x2, x,),  ARG(double3x2, y,)), 0)
ASSIGN_OPERATOR(double3x3,  ARG2(ARG(double3x3, x,),  ARG(double3x3, y,)), 0)
ASSIGN_OPERATOR(double3x4,  ARG2(ARG(double3x4, x,),  ARG(double3x4, y,)), 0)
ASSIGN_OPERATOR(double4x2,  ARG2(ARG(double4x2, x,),  ARG(double4x2, y,)), 0)
ASSIGN_OPERATOR(double4x3,  ARG2(ARG(double4x3, x,),  ARG(double4x3, y,)), 0)
ASSIGN_OPERATOR(double4x4,  ARG2(ARG(double4x4, x,),  ARG(double4x4, y,)), 0)

ASSIGN_OPERATOR(color,      ARG2(ARG(color, x,),      ARG(color, y,)), 0)
ASSIGN_OPERATOR(string,     ARG2(ARG(string, x,),     ARG(string, y,)), 0)

// unary operator+ and operator-
UNARY_OPERATORS(int,     ARG1(ARG(int,     x,)), 0)
UNARY_OPERATORS(int2,    ARG1(ARG(int2,    x,)), 0)
UNARY_OPERATORS(int3,    ARG1(ARG(int3,    x,)), 0)
UNARY_OPERATORS(int4,    ARG1(ARG(int4,    x,)), 0)
UNARY_OPERATORS(float,   ARG1(ARG(float,   x,)), 0)
UNARY_OPERATORS(float2,  ARG1(ARG(float2,  x,)), 0)
UNARY_OPERATORS(float3,  ARG1(ARG(float3,  x,)), 0)
UNARY_OPERATORS(float4,  ARG1(ARG(float4,  x,)), 0)
UNARY_OPERATORS(double,  ARG1(ARG(double,  x,)), 0)
UNARY_OPERATORS(double2, ARG1(ARG(double2, x,)), 0)
UNARY_OPERATORS(double3, ARG1(ARG(double3, x,)), 0)
UNARY_OPERATORS(double4, ARG1(ARG(double4, x,)), 0)
UNARY_OPERATORS(color,   ARG1(ARG(color,   x,)), 0)

// unary operator+ and operator- for matrices
UNARY_OPERATORS(float2x2,  ARG1(ARG(float2x2,  x,)), 0)
UNARY_OPERATORS(float2x3,  ARG1(ARG(float2x3,  x,)), 0)
UNARY_OPERATORS(float2x4,  ARG1(ARG(float2x4,  x,)), 0)
UNARY_OPERATORS(float3x2,  ARG1(ARG(float3x2,  x,)), 0)
UNARY_OPERATORS(float3x3,  ARG1(ARG(float3x3,  x,)), 0)
UNARY_OPERATORS(float3x4,  ARG1(ARG(float3x4,  x,)), 0)
UNARY_OPERATORS(float4x2,  ARG1(ARG(float4x2,  x,)), 0)
UNARY_OPERATORS(float4x3,  ARG1(ARG(float4x3,  x,)), 0)
UNARY_OPERATORS(float4x4,  ARG1(ARG(float4x4,  x,)), 0)
UNARY_OPERATORS(double2x2, ARG1(ARG(double2x2, x,)), 0)
UNARY_OPERATORS(double2x3, ARG1(ARG(double2x3, x,)), 0)
UNARY_OPERATORS(double2x4, ARG1(ARG(double2x4, x,)), 0)
UNARY_OPERATORS(double3x2, ARG1(ARG(double3x2, x,)), 0)
UNARY_OPERATORS(double3x3, ARG1(ARG(double3x3, x,)), 0)
UNARY_OPERATORS(double3x4, ARG1(ARG(double3x4, x,)), 0)
UNARY_OPERATORS(double4x2, ARG1(ARG(double4x2, x,)), 0)
UNARY_OPERATORS(double4x3, ARG1(ARG(double4x3, x,)), 0)
UNARY_OPERATORS(double4x4, ARG1(ARG(double4x4, x,)), 0)

// operator* for matrix * matrix
MUL_OPERATOR(float2x2,  ARG2(ARG(float2x2,  x,), ARG(float2x2,  y,)), 0)
MUL_OPERATOR(float3x2,  ARG2(ARG(float2x2,  x,), ARG(float3x2,  y,)), 0)
MUL_OPERATOR(float4x2,  ARG2(ARG(float2x2,  x,), ARG(float4x2,  y,)), 0)
MUL_OPERATOR(float2x2,  ARG2(ARG(float3x2,  x,), ARG(float2x3,  y,)), 0)
MUL_OPERATOR(float3x2,  ARG2(ARG(float3x2,  x,), ARG(float3x3,  y,)), 0)
MUL_OPERATOR(float4x2,  ARG2(ARG(float3x2,  x,), ARG(float4x3,  y,)), 0)
MUL_OPERATOR(float2x2,  ARG2(ARG(float4x2,  x,), ARG(float2x4,  y,)), 0)
MUL_OPERATOR(float3x2,  ARG2(ARG(float4x2,  x,), ARG(float3x4,  y,)), 0)
MUL_OPERATOR(float4x2,  ARG2(ARG(float4x2,  x,), ARG(float4x4,  y,)), 0)
MUL_OPERATOR(float2x3,  ARG2(ARG(float2x3,  x,), ARG(float2x2,  y,)), 0)
MUL_OPERATOR(float3x3,  ARG2(ARG(float2x3,  x,), ARG(float3x2,  y,)), 0)
MUL_OPERATOR(float4x3,  ARG2(ARG(float2x3,  x,), ARG(float4x2,  y,)), 0)
MUL_OPERATOR(float2x3,  ARG2(ARG(float3x3,  x,), ARG(float2x3,  y,)), 0)
MUL_OPERATOR(float3x3,  ARG2(ARG(float3x3,  x,), ARG(float3x3,  y,)), 0)
MUL_OPERATOR(float4x3,  ARG2(ARG(float3x3,  x,), ARG(float4x3,  y,)), 0)
MUL_OPERATOR(float2x3,  ARG2(ARG(float4x3,  x,), ARG(float2x4,  y,)), 0)
MUL_OPERATOR(float3x3,  ARG2(ARG(float4x3,  x,), ARG(float3x4,  y,)), 0)
MUL_OPERATOR(float4x3,  ARG2(ARG(float4x3,  x,), ARG(float4x4,  y,)), 0)
MUL_OPERATOR(float2x4,  ARG2(ARG(float2x4,  x,), ARG(float2x2,  y,)), 0)
MUL_OPERATOR(float3x4,  ARG2(ARG(float2x4,  x,), ARG(float3x2,  y,)), 0)
MUL_OPERATOR(float4x4,  ARG2(ARG(float2x4,  x,), ARG(float4x2,  y,)), 0)
MUL_OPERATOR(float2x4,  ARG2(ARG(float3x4,  x,), ARG(float2x3,  y,)), 0)
MUL_OPERATOR(float3x4,  ARG2(ARG(float3x4,  x,), ARG(float3x3,  y,)), 0)
MUL_OPERATOR(float4x4,  ARG2(ARG(float3x4,  x,), ARG(float4x3,  y,)), 0)
MUL_OPERATOR(float2x4,  ARG2(ARG(float4x4,  x,), ARG(float2x4,  y,)), 0)
MUL_OPERATOR(float3x4,  ARG2(ARG(float4x4,  x,), ARG(float3x4,  y,)), 0)
MUL_OPERATOR(float4x4,  ARG2(ARG(float4x4,  x,), ARG(float4x4,  y,)), 0)

MUL_OPERATOR(double2x2,  ARG2(ARG(double2x2,  x,), ARG(double2x2,  y,)), 0)
MUL_OPERATOR(double3x2,  ARG2(ARG(double2x2,  x,), ARG(double3x2,  y,)), 0)
MUL_OPERATOR(double4x2,  ARG2(ARG(double2x2,  x,), ARG(double4x2,  y,)), 0)
MUL_OPERATOR(double2x2,  ARG2(ARG(double3x2,  x,), ARG(double2x3,  y,)), 0)
MUL_OPERATOR(double3x2,  ARG2(ARG(double3x2,  x,), ARG(double3x3,  y,)), 0)
MUL_OPERATOR(double4x2,  ARG2(ARG(double3x2,  x,), ARG(double4x3,  y,)), 0)
MUL_OPERATOR(double2x2,  ARG2(ARG(double4x2,  x,), ARG(double2x4,  y,)), 0)
MUL_OPERATOR(double3x2,  ARG2(ARG(double4x2,  x,), ARG(double3x4,  y,)), 0)
MUL_OPERATOR(double4x2,  ARG2(ARG(double4x2,  x,), ARG(double4x4,  y,)), 0)
MUL_OPERATOR(double2x3,  ARG2(ARG(double2x3,  x,), ARG(double2x2,  y,)), 0)
MUL_OPERATOR(double3x3,  ARG2(ARG(double2x3,  x,), ARG(double3x2,  y,)), 0)
MUL_OPERATOR(double4x3,  ARG2(ARG(double2x3,  x,), ARG(double4x2,  y,)), 0)
MUL_OPERATOR(double2x3,  ARG2(ARG(double3x3,  x,), ARG(double2x3,  y,)), 0)
MUL_OPERATOR(double3x3,  ARG2(ARG(double3x3,  x,), ARG(double3x3,  y,)), 0)
MUL_OPERATOR(double4x3,  ARG2(ARG(double3x3,  x,), ARG(double4x3,  y,)), 0)
MUL_OPERATOR(double2x3,  ARG2(ARG(double4x3,  x,), ARG(double2x4,  y,)), 0)
MUL_OPERATOR(double3x3,  ARG2(ARG(double4x3,  x,), ARG(double3x4,  y,)), 0)
MUL_OPERATOR(double4x3,  ARG2(ARG(double4x3,  x,), ARG(double4x4,  y,)), 0)
MUL_OPERATOR(double2x4,  ARG2(ARG(double2x4,  x,), ARG(double2x2,  y,)), 0)
MUL_OPERATOR(double3x4,  ARG2(ARG(double2x4,  x,), ARG(double3x2,  y,)), 0)
MUL_OPERATOR(double4x4,  ARG2(ARG(double2x4,  x,), ARG(double4x2,  y,)), 0)
MUL_OPERATOR(double2x4,  ARG2(ARG(double3x4,  x,), ARG(double2x3,  y,)), 0)
MUL_OPERATOR(double3x4,  ARG2(ARG(double3x4,  x,), ARG(double3x3,  y,)), 0)
MUL_OPERATOR(double4x4,  ARG2(ARG(double3x4,  x,), ARG(double4x3,  y,)), 0)
MUL_OPERATOR(double2x4,  ARG2(ARG(double4x4,  x,), ARG(double2x4,  y,)), 0)
MUL_OPERATOR(double3x4,  ARG2(ARG(double4x4,  x,), ARG(double3x4,  y,)), 0)
MUL_OPERATOR(double4x4,  ARG2(ARG(double4x4,  x,), ARG(double4x4,  y,)), 0)

// operator* for matrix * vector
MUL_OPERATOR(float2,  ARG2(ARG(float2x2,  x,), ARG(float2,  y,)), 0)
MUL_OPERATOR(float2,  ARG2(ARG(float3x2,  x,), ARG(float3,  y,)), 0)
MUL_OPERATOR(float2,  ARG2(ARG(float4x2,  x,), ARG(float4,  y,)), 0)
MUL_OPERATOR(float3,  ARG2(ARG(float2x3,  x,), ARG(float2,  y,)), 0)
MUL_OPERATOR(float3,  ARG2(ARG(float3x3,  x,), ARG(float3,  y,)), 0)
MUL_OPERATOR(float3,  ARG2(ARG(float4x3,  x,), ARG(float4,  y,)), 0)
MUL_OPERATOR(float4,  ARG2(ARG(float2x4,  x,), ARG(float2,  y,)), 0)
MUL_OPERATOR(float4,  ARG2(ARG(float3x4,  x,), ARG(float3,  y,)), 0)
MUL_OPERATOR(float4,  ARG2(ARG(float4x4,  x,), ARG(float4,  y,)), 0)
MUL_OPERATOR(double2, ARG2(ARG(double2x2, x,), ARG(double2, y,)), 0)
MUL_OPERATOR(double2, ARG2(ARG(double3x2, x,), ARG(double3, y,)), 0)
MUL_OPERATOR(double2, ARG2(ARG(double4x2, x,), ARG(double4, y,)), 0)
MUL_OPERATOR(double3, ARG2(ARG(double2x3, x,), ARG(double2, y,)), 0)
MUL_OPERATOR(double3, ARG2(ARG(double3x3, x,), ARG(double3, y,)), 0)
MUL_OPERATOR(double3, ARG2(ARG(double4x3, x,), ARG(double4, y,)), 0)
MUL_OPERATOR(double4, ARG2(ARG(double2x4, x,), ARG(double2, y,)), 0)
MUL_OPERATOR(double4, ARG2(ARG(double3x4, x,), ARG(double3, y,)), 0)
MUL_OPERATOR(double4, ARG2(ARG(double4x4, x,), ARG(double4, y,)), 0)

// operator* for vector * matrix
MUL_OPERATOR(float2,  ARG2(ARG(float2,  x,), ARG(float2x2,  y,)), 0)
MUL_OPERATOR(float3,  ARG2(ARG(float2,  x,), ARG(float3x2,  y,)), 0)
MUL_OPERATOR(float4,  ARG2(ARG(float2,  x,), ARG(float4x2,  y,)), 0)
MUL_OPERATOR(float2,  ARG2(ARG(float3,  x,), ARG(float2x3,  y,)), 0)
MUL_OPERATOR(float3,  ARG2(ARG(float3,  x,), ARG(float3x3,  y,)), 0)
MUL_OPERATOR(float4,  ARG2(ARG(float3,  x,), ARG(float4x3,  y,)), 0)
MUL_OPERATOR(float2,  ARG2(ARG(float4,  x,), ARG(float2x4,  y,)), 0)
MUL_OPERATOR(float3,  ARG2(ARG(float4,  x,), ARG(float3x4,  y,)), 0)
MUL_OPERATOR(float4,  ARG2(ARG(float4,  x,), ARG(float4x4,  y,)), 0)
MUL_OPERATOR(double2, ARG2(ARG(double2, x,), ARG(double2x2, y,)), 0)
MUL_OPERATOR(double3, ARG2(ARG(double2, x,), ARG(double3x2, y,)), 0)
MUL_OPERATOR(double4, ARG2(ARG(double2, x,), ARG(double4x2, y,)), 0)
MUL_OPERATOR(double2, ARG2(ARG(double3, x,), ARG(double2x3, y,)), 0)
MUL_OPERATOR(double3, ARG2(ARG(double3, x,), ARG(double3x3, y,)), 0)
MUL_OPERATOR(double4, ARG2(ARG(double3, x,), ARG(double4x3, y,)), 0)
MUL_OPERATOR(double2, ARG2(ARG(double4, x,), ARG(double2x4, y,)), 0)
MUL_OPERATOR(double3, ARG2(ARG(double4, x,), ARG(double3x4, y,)), 0)
MUL_OPERATOR(double4, ARG2(ARG(double4, x,), ARG(double4x4, y,)), 0)

// operator* for matrix * scalar and scalar * matrix
MUL_OPERATOR(float2x2,  ARG2(ARG(float2x2, x,), ARG(float, y,)), 0)
MUL_OPERATOR(float2x2,  ARG2(ARG(float, x,), ARG(float2x2, y,)), 0)
MUL_OPERATOR(float2x3,  ARG2(ARG(float2x3, x,), ARG(float, y,)), 0)
MUL_OPERATOR(float2x3,  ARG2(ARG(float, x,), ARG(float2x3, y,)), 0)
MUL_OPERATOR(float2x4,  ARG2(ARG(float2x4, x,), ARG(float, y,)), 0)
MUL_OPERATOR(float2x4,  ARG2(ARG(float, x,), ARG(float2x4, y,)), 0)
MUL_OPERATOR(float3x2,  ARG2(ARG(float3x2, x,), ARG(float, y,)), 0)
MUL_OPERATOR(float3x2,  ARG2(ARG(float, x,), ARG(float3x2, y,)), 0)
MUL_OPERATOR(float3x3,  ARG2(ARG(float3x3, x,), ARG(float, y,)), 0)
MUL_OPERATOR(float3x3,  ARG2(ARG(float, x,), ARG(float3x3, y,)), 0)
MUL_OPERATOR(float3x4,  ARG2(ARG(float3x4, x,), ARG(float, y,)), 0)
MUL_OPERATOR(float3x4,  ARG2(ARG(float, x,), ARG(float3x4, y,)), 0)
MUL_OPERATOR(float4x2,  ARG2(ARG(float4x2, x,), ARG(float, y,)), 0)
MUL_OPERATOR(float4x2,  ARG2(ARG(float, x,), ARG(float4x2, y,)), 0)
MUL_OPERATOR(float4x3,  ARG2(ARG(float4x3, x,), ARG(float, y,)), 0)
MUL_OPERATOR(float4x3,  ARG2(ARG(float, x,), ARG(float4x3, y,)), 0)
MUL_OPERATOR(float4x4,  ARG2(ARG(float4x4, x,), ARG(float, y,)), 0)
MUL_OPERATOR(float4x4,  ARG2(ARG(float, x,), ARG(float4x4, y,)), 0)
MUL_OPERATOR(double2x2, ARG2(ARG(double2x2, x,), ARG(double, y,)), 0)
MUL_OPERATOR(double2x2, ARG2(ARG(double, x,), ARG(double2x2, y,)), 0)
MUL_OPERATOR(double2x3, ARG2(ARG(double2x3, x,), ARG(double, y,)), 0)
MUL_OPERATOR(double2x3, ARG2(ARG(double, x,), ARG(double2x3, y,)), 0)
MUL_OPERATOR(double2x4, ARG2(ARG(double2x4, x,), ARG(double, y,)), 0)
MUL_OPERATOR(double2x4, ARG2(ARG(double, x,), ARG(double2x4, y,)), 0)
MUL_OPERATOR(double3x2, ARG2(ARG(double3x2, x,), ARG(double, y,)), 0)
MUL_OPERATOR(double3x2, ARG2(ARG(double, x,), ARG(double3x2, y,)), 0)
MUL_OPERATOR(double3x3, ARG2(ARG(double3x3, x,), ARG(double, y,)), 0)
MUL_OPERATOR(double3x3, ARG2(ARG(double, x,), ARG(double3x3, y,)), 0)
MUL_OPERATOR(double3x4, ARG2(ARG(double3x4, x,), ARG(double, y,)), 0)
MUL_OPERATOR(double3x4, ARG2(ARG(double, x,), ARG(double3x4, y,)), 0)
MUL_OPERATOR(double4x2, ARG2(ARG(double4x2, x,), ARG(double, y,)), 0)
MUL_OPERATOR(double4x2, ARG2(ARG(double, x,), ARG(double4x2, y,)), 0)
MUL_OPERATOR(double4x3, ARG2(ARG(double4x3, x,), ARG(double, y,)), 0)
MUL_OPERATOR(double4x3, ARG2(ARG(double, x,), ARG(double4x3, y,)), 0)
MUL_OPERATOR(double4x4, ARG2(ARG(double4x4, x,), ARG(double, y,)), 0)
MUL_OPERATOR(double4x4, ARG2(ARG(double, x,), ARG(double4x4, y,)), 0)

// operator*= for vector * matrix
MUL_ASSIGN_OPERATOR(float2,  ARG2(ARG(float2,  x,), ARG(float2x2,  y,)), 0)
MUL_ASSIGN_OPERATOR(float3,  ARG2(ARG(float3,  x,), ARG(float3x3,  y,)), 0)
MUL_ASSIGN_OPERATOR(float4,  ARG2(ARG(float4,  x,), ARG(float4x4,  y,)), 0)
MUL_ASSIGN_OPERATOR(double2, ARG2(ARG(double2, x,), ARG(double2x2, y,)), 0)
MUL_ASSIGN_OPERATOR(double3, ARG2(ARG(double3, x,), ARG(double3x3, y,)), 0)
MUL_ASSIGN_OPERATOR(double4, ARG2(ARG(double4, x,), ARG(double4x4, y,)), 0)

// operator*= for square matrix case: [n x n] *= [n x n]
MUL_ASSIGN_OPERATOR(float2x2,  ARG2(ARG(float2x2,  x,), ARG(float2x2,  y,)), 0)
MUL_ASSIGN_OPERATOR(double2x2, ARG2(ARG(double2x2, x,), ARG(double2x2, y,)), 0)
MUL_ASSIGN_OPERATOR(float3x3,  ARG2(ARG(float3x3,  x,), ARG(float3x3,  y,)), 0)
MUL_ASSIGN_OPERATOR(double3x3, ARG2(ARG(double3x3, x,), ARG(double3x3, y,)), 0)
MUL_ASSIGN_OPERATOR(float4x4,  ARG2(ARG(float4x4,  x,), ARG(float4x4,  y,)), 0)
MUL_ASSIGN_OPERATOR(double4x4, ARG2(ARG(double4x4, x,), ARG(double4x4, y,)), 0)

// operator*= for matrix * scalar case: [n x m] *= t
MUL_ASSIGN_OPERATOR(float2x2,  ARG2(ARG(float2x2,  x,), ARG(float,  y,)), 0)
MUL_ASSIGN_OPERATOR(float2x3,  ARG2(ARG(float2x3,  x,), ARG(float,  y,)), 0)
MUL_ASSIGN_OPERATOR(float2x4,  ARG2(ARG(float2x4,  x,), ARG(float,  y,)), 0)
MUL_ASSIGN_OPERATOR(float3x2,  ARG2(ARG(float3x2,  x,), ARG(float,  y,)), 0)
MUL_ASSIGN_OPERATOR(float3x3,  ARG2(ARG(float3x3,  x,), ARG(float,  y,)), 0)
MUL_ASSIGN_OPERATOR(float3x4,  ARG2(ARG(float3x4,  x,), ARG(float,  y,)), 0)
MUL_ASSIGN_OPERATOR(float4x2,  ARG2(ARG(float4x2,  x,), ARG(float,  y,)), 0)
MUL_ASSIGN_OPERATOR(float4x3,  ARG2(ARG(float4x3,  x,), ARG(float,  y,)), 0)
MUL_ASSIGN_OPERATOR(float4x4,  ARG2(ARG(float4x4,  x,), ARG(float,  y,)), 0)
MUL_ASSIGN_OPERATOR(double2x2, ARG2(ARG(double2x2, x,), ARG(double, y,)), 0)
MUL_ASSIGN_OPERATOR(double2x3, ARG2(ARG(double2x3, x,), ARG(double, y,)), 0)
MUL_ASSIGN_OPERATOR(double2x4, ARG2(ARG(double2x4, x,), ARG(double, y,)), 0)
MUL_ASSIGN_OPERATOR(double3x2, ARG2(ARG(double3x2, x,), ARG(double, y,)), 0)
MUL_ASSIGN_OPERATOR(double3x3, ARG2(ARG(double3x3, x,), ARG(double, y,)), 0)
MUL_ASSIGN_OPERATOR(double3x4, ARG2(ARG(double3x4, x,), ARG(double, y,)), 0)
MUL_ASSIGN_OPERATOR(double4x2, ARG2(ARG(double4x2, x,), ARG(double, y,)), 0)
MUL_ASSIGN_OPERATOR(double4x3, ARG2(ARG(double4x3, x,), ARG(double, y,)), 0)
MUL_ASSIGN_OPERATOR(double4x4, ARG2(ARG(double4x4, x,), ARG(double, y,)), 0)

// operator* for mixed scalar, vector
MUL_OPERATOR(int,     ARG2(ARG(int,     x,), ARG(int,     y,)), 0)
MUL_OPERATOR(int2,    ARG2(ARG(int2,    x,), ARG(int2,    y,)), 0)
MUL_OPERATOR(int2,    ARG2(ARG(int,     x,), ARG(int2,    y,)), 0)
MUL_OPERATOR(int2,    ARG2(ARG(int2,    x,), ARG(int,     y,)), 0)
MUL_OPERATOR(int3,    ARG2(ARG(int3,    x,), ARG(int3,    y,)), 0)
MUL_OPERATOR(int3,    ARG2(ARG(int,     x,), ARG(int3,    y,)), 0)
MUL_OPERATOR(int3,    ARG2(ARG(int3,    x,), ARG(int,     y,)), 0)
MUL_OPERATOR(int4,    ARG2(ARG(int4,    x,), ARG(int4,    y,)), 0)
MUL_OPERATOR(int4,    ARG2(ARG(int,     x,), ARG(int4,    y,)), 0)
MUL_OPERATOR(int4,    ARG2(ARG(int4,    x,), ARG(int,     y,)), 0)
MUL_OPERATOR(float,   ARG2(ARG(float,   x,), ARG(float,   y,)), 0)
MUL_OPERATOR(float2,  ARG2(ARG(float2,  x,), ARG(float2,  y,)), 0)
MUL_OPERATOR(float2,  ARG2(ARG(float,   x,), ARG(float2,  y,)), 0)
MUL_OPERATOR(float2,  ARG2(ARG(float2,  x,), ARG(float,   y,)), 0)
MUL_OPERATOR(float3,  ARG2(ARG(float3,  x,), ARG(float3,  y,)), 0)
MUL_OPERATOR(float3,  ARG2(ARG(float,   x,), ARG(float3,  y,)), 0)
MUL_OPERATOR(float3,  ARG2(ARG(float3,  x,), ARG(float,   y,)), 0)
MUL_OPERATOR(float4,  ARG2(ARG(float4,  x,), ARG(float4,  y,)), 0)
MUL_OPERATOR(float4,  ARG2(ARG(float,   x,), ARG(float4,  y,)), 0)
MUL_OPERATOR(float4,  ARG2(ARG(float4,  x,), ARG(float,   y,)), 0)
MUL_OPERATOR(double,  ARG2(ARG(double,  x,), ARG(double,  y,)), 0)
MUL_OPERATOR(double2, ARG2(ARG(double2, x,), ARG(double2, y,)), 0)
MUL_OPERATOR(double2, ARG2(ARG(double,  x,), ARG(double2, y,)), 0)
MUL_OPERATOR(double2, ARG2(ARG(double2, x,), ARG(double,  y,)), 0)
MUL_OPERATOR(double3, ARG2(ARG(double3, x,), ARG(double3, y,)), 0)
MUL_OPERATOR(double3, ARG2(ARG(double,  x,), ARG(double3, y,)), 0)
MUL_OPERATOR(double3, ARG2(ARG(double3, x,), ARG(double,  y,)), 0)
MUL_OPERATOR(double4, ARG2(ARG(double4, x,), ARG(double4, y,)), 0)
MUL_OPERATOR(double4, ARG2(ARG(double,  x,), ARG(double4, y,)), 0)
MUL_OPERATOR(double4, ARG2(ARG(double4, x,), ARG(double,  y,)), 0)

// operator* for mixed float, color
MUL_OPERATOR(color, ARG2(ARG(color, x,), ARG(color, y,)), 0)
MUL_OPERATOR(color, ARG2(ARG(float, x,), ARG(color, y,)), 0)
MUL_OPERATOR(color, ARG2(ARG(color, x,), ARG(float, y,)), 0)

// operator/ for mixed scalar, vector
OPERATOR(int,     OK_DIVIDE, ARG2(ARG(int,     x,), ARG(int,     y,)), 0)
OPERATOR(int2,    OK_DIVIDE, ARG2(ARG(int2,    x,), ARG(int2,    y,)), 0)
OPERATOR(int2,    OK_DIVIDE, ARG2(ARG(int,     x,), ARG(int2,    y,)), 0)
OPERATOR(int2,    OK_DIVIDE, ARG2(ARG(int2,    x,), ARG(int,     y,)), 0)
OPERATOR(int3,    OK_DIVIDE, ARG2(ARG(int3,    x,), ARG(int3,    y,)), 0)
OPERATOR(int3,    OK_DIVIDE, ARG2(ARG(int,     x,), ARG(int3,    y,)), 0)
OPERATOR(int3,    OK_DIVIDE, ARG2(ARG(int3,    x,), ARG(int,     y,)), 0)
OPERATOR(int4,    OK_DIVIDE, ARG2(ARG(int4,    x,), ARG(int4,    y,)), 0)
OPERATOR(int4,    OK_DIVIDE, ARG2(ARG(int,     x,), ARG(int4,    y,)), 0)
OPERATOR(int4,    OK_DIVIDE, ARG2(ARG(int4,    x,), ARG(int,     y,)), 0)
OPERATOR(float,   OK_DIVIDE, ARG2(ARG(float,   x,), ARG(float,   y,)), 0)
OPERATOR(float2,  OK_DIVIDE, ARG2(ARG(float2,  x,), ARG(float2,  y,)), 0)
OPERATOR(float2,  OK_DIVIDE, ARG2(ARG(float,   x,), ARG(float2,  y,)), 0)
OPERATOR(float2,  OK_DIVIDE, ARG2(ARG(float2,  x,), ARG(float,   y,)), 0)
OPERATOR(float3,  OK_DIVIDE, ARG2(ARG(float3,  x,), ARG(float3,  y,)), 0)
OPERATOR(float3,  OK_DIVIDE, ARG2(ARG(float,   x,), ARG(float3,  y,)), 0)
OPERATOR(float3,  OK_DIVIDE, ARG2(ARG(float3,  x,), ARG(float,   y,)), 0)
OPERATOR(float4,  OK_DIVIDE, ARG2(ARG(float4,  x,), ARG(float4,  y,)), 0)
OPERATOR(float4,  OK_DIVIDE, ARG2(ARG(float,   x,), ARG(float4,  y,)), 0)
OPERATOR(float4,  OK_DIVIDE, ARG2(ARG(float4,  x,), ARG(float,   y,)), 0)
OPERATOR(double,  OK_DIVIDE, ARG2(ARG(double,  x,), ARG(double,  y,)), 0)
OPERATOR(double2, OK_DIVIDE, ARG2(ARG(double2, x,), ARG(double2, y,)), 0)
OPERATOR(double2, OK_DIVIDE, ARG2(ARG(double,  x,), ARG(double2, y,)), 0)
OPERATOR(double2, OK_DIVIDE, ARG2(ARG(double2, x,), ARG(double,  y,)), 0)
OPERATOR(double3, OK_DIVIDE, ARG2(ARG(double3, x,), ARG(double3, y,)), 0)
OPERATOR(double3, OK_DIVIDE, ARG2(ARG(double,  x,), ARG(double3, y,)), 0)
OPERATOR(double3, OK_DIVIDE, ARG2(ARG(double3, x,), ARG(double,  y,)), 0)
OPERATOR(double4, OK_DIVIDE, ARG2(ARG(double4, x,), ARG(double4, y,)), 0)
OPERATOR(double4, OK_DIVIDE, ARG2(ARG(double,  x,), ARG(double4, y,)), 0)
OPERATOR(double4, OK_DIVIDE, ARG2(ARG(double4, x,), ARG(double,  y,)), 0)

// operator/ for color
OPERATOR(color,   OK_DIVIDE, ARG2(ARG(color,   x,), ARG(color,   y,)), 0)
OPERATOR(color,   OK_DIVIDE, ARG2(ARG(color,   x,), ARG(float,   y,)), 0)

// operator/ for matrix / scalar
OPERATOR(float2x2,  OK_DIVIDE, ARG2(ARG(float2x2,  x,), ARG(float,  y,)), 0)
OPERATOR(float2x3,  OK_DIVIDE, ARG2(ARG(float2x3,  x,), ARG(float,  y,)), 0)
OPERATOR(float2x4,  OK_DIVIDE, ARG2(ARG(float2x4,  x,), ARG(float,  y,)), 0)
OPERATOR(float3x2,  OK_DIVIDE, ARG2(ARG(float3x2,  x,), ARG(float,  y,)), 0)
OPERATOR(float3x3,  OK_DIVIDE, ARG2(ARG(float3x3,  x,), ARG(float,  y,)), 0)
OPERATOR(float3x4,  OK_DIVIDE, ARG2(ARG(float3x4,  x,), ARG(float,  y,)), 0)
OPERATOR(float4x2,  OK_DIVIDE, ARG2(ARG(float4x2,  x,), ARG(float,  y,)), 0)
OPERATOR(float4x3,  OK_DIVIDE, ARG2(ARG(float4x3,  x,), ARG(float,  y,)), 0)
OPERATOR(float4x4,  OK_DIVIDE, ARG2(ARG(float4x4,  x,), ARG(float,  y,)), 0)
OPERATOR(double2x2, OK_DIVIDE, ARG2(ARG(double2x2, x,), ARG(double, y,)), 0)
OPERATOR(double2x3, OK_DIVIDE, ARG2(ARG(double2x3, x,), ARG(double, y,)), 0)
OPERATOR(double2x4, OK_DIVIDE, ARG2(ARG(double2x4, x,), ARG(double, y,)), 0)
OPERATOR(double3x2, OK_DIVIDE, ARG2(ARG(double3x2, x,), ARG(double, y,)), 0)
OPERATOR(double3x3, OK_DIVIDE, ARG2(ARG(double3x3, x,), ARG(double, y,)), 0)
OPERATOR(double3x4, OK_DIVIDE, ARG2(ARG(double3x4, x,), ARG(double, y,)), 0)
OPERATOR(double4x2, OK_DIVIDE, ARG2(ARG(double4x2, x,), ARG(double, y,)), 0)
OPERATOR(double4x3, OK_DIVIDE, ARG2(ARG(double4x3, x,), ARG(double, y,)), 0)
OPERATOR(double4x4, OK_DIVIDE, ARG2(ARG(double4x4, x,), ARG(double, y,)), 0)

// binary operator+ and operator-, scalars, vectors, and color
BINARY_PLUS_MINUS_OPERATORS(int,     ARG2(ARG(int,     x,), ARG(int,     y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(int2,    ARG2(ARG(int2,    x,), ARG(int2,    y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(int2,    ARG2(ARG(int,     x,), ARG(int2,    y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(int2,    ARG2(ARG(int2,    x,), ARG(int,     y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(int3,    ARG2(ARG(int3,    x,), ARG(int3,    y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(int3,    ARG2(ARG(int,     x,), ARG(int3,    y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(int3,    ARG2(ARG(int3,    x,), ARG(int,     y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(int4,    ARG2(ARG(int4,    x,), ARG(int4,    y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(int4,    ARG2(ARG(int,     x,), ARG(int4,    y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(int4,    ARG2(ARG(int4,    x,), ARG(int,     y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(float,   ARG2(ARG(float,   x,), ARG(float,   y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(float2,  ARG2(ARG(float2,  x,), ARG(float2,  y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(float2,  ARG2(ARG(float,   x,), ARG(float2,  y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(float2,  ARG2(ARG(float2,  x,), ARG(float,   y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(float3,  ARG2(ARG(float3,  x,), ARG(float3,  y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(float3,  ARG2(ARG(float,   x,), ARG(float3,  y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(float3,  ARG2(ARG(float3,  x,), ARG(float,   y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(float4,  ARG2(ARG(float4,  x,), ARG(float4,  y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(float4,  ARG2(ARG(float,   x,), ARG(float4,  y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(float4,  ARG2(ARG(float4,  x,), ARG(float,   y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(double,  ARG2(ARG(double,  x,), ARG(double,  y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(double2, ARG2(ARG(double2, x,), ARG(double2, y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(double2, ARG2(ARG(double,  x,), ARG(double2, y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(double2, ARG2(ARG(double2, x,), ARG(double,  y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(double3, ARG2(ARG(double3, x,), ARG(double3, y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(double3, ARG2(ARG(double,  x,), ARG(double3, y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(double3, ARG2(ARG(double3, x,), ARG(double,  y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(double4, ARG2(ARG(double4, x,), ARG(double4, y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(double4, ARG2(ARG(double,  x,), ARG(double4, y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(double4, ARG2(ARG(double4, x,), ARG(double,  y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(color,   ARG2(ARG(color,   x,), ARG(color,   y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(color,   ARG2(ARG(float,   x,), ARG(color,   y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(color,   ARG2(ARG(color,   x,), ARG(float,   y,)), 0)

// binary operator+ and operator- for matrices
BINARY_PLUS_MINUS_OPERATORS(float2x2,  ARG2(ARG(float2x2,  x,), ARG(float2x2,  y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(float2x3,  ARG2(ARG(float2x3,  x,), ARG(float2x3,  y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(float2x4,  ARG2(ARG(float2x4,  x,), ARG(float2x4,  y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(float3x2,  ARG2(ARG(float3x2,  x,), ARG(float3x2,  y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(float3x3,  ARG2(ARG(float3x3,  x,), ARG(float3x3,  y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(float3x4,  ARG2(ARG(float3x4,  x,), ARG(float3x4,  y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(float4x2,  ARG2(ARG(float4x2,  x,), ARG(float4x2,  y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(float4x3,  ARG2(ARG(float4x3,  x,), ARG(float4x3,  y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(float4x4,  ARG2(ARG(float4x4,  x,), ARG(float4x4,  y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(double2x2, ARG2(ARG(double2x2, x,), ARG(double2x2, y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(double2x3, ARG2(ARG(double2x3, x,), ARG(double2x3, y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(double2x4, ARG2(ARG(double2x4, x,), ARG(double2x4, y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(double3x2, ARG2(ARG(double3x2, x,), ARG(double3x2, y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(double3x3, ARG2(ARG(double3x3, x,), ARG(double3x3, y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(double3x4, ARG2(ARG(double3x4, x,), ARG(double3x4, y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(double4x2, ARG2(ARG(double4x2, x,), ARG(double4x2, y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(double4x3, ARG2(ARG(double4x3, x,), ARG(double4x3, y,)), 0)
BINARY_PLUS_MINUS_OPERATORS(double4x4, ARG2(ARG(double4x4, x,), ARG(double4x4, y,)), 0)

// operator+=, operator-=, operator*=, operator/= for (vector|scalar) op scalar
ASSIGN_OPS_OPERATORS(int,     ARG2(ARG(int,     x,), ARG(int,     y,)), 0)
ASSIGN_OPS_OPERATORS(int2,    ARG2(ARG(int2,    x,), ARG(int2,    y,)), 0)
ASSIGN_OPS_OPERATORS(int2,    ARG2(ARG(int2,    x,), ARG(int,     y,)), 0)
ASSIGN_OPS_OPERATORS(int3,    ARG2(ARG(int3,    x,), ARG(int3,    y,)), 0)
ASSIGN_OPS_OPERATORS(int3,    ARG2(ARG(int3,    x,), ARG(int,     y,)), 0)
ASSIGN_OPS_OPERATORS(int4,    ARG2(ARG(int4,    x,), ARG(int4,    y,)), 0)
ASSIGN_OPS_OPERATORS(int4,    ARG2(ARG(int4,    x,), ARG(int,     y,)), 0)
ASSIGN_OPS_OPERATORS(float,   ARG2(ARG(float,   x,), ARG(float,   y,)), 0)
ASSIGN_OPS_OPERATORS(float2,  ARG2(ARG(float2,  x,), ARG(float2,  y,)), 0)
ASSIGN_OPS_OPERATORS(float2,  ARG2(ARG(float2,  x,), ARG(float,   y,)), 0)
ASSIGN_OPS_OPERATORS(float3,  ARG2(ARG(float3,  x,), ARG(float3,  y,)), 0)
ASSIGN_OPS_OPERATORS(float3,  ARG2(ARG(float3,  x,), ARG(float,   y,)), 0)
ASSIGN_OPS_OPERATORS(float4,  ARG2(ARG(float4,  x,), ARG(float4,  y,)), 0)
ASSIGN_OPS_OPERATORS(float4,  ARG2(ARG(float4,  x,), ARG(float,   y,)), 0)
ASSIGN_OPS_OPERATORS(double,  ARG2(ARG(double,  x,), ARG(double,  y,)), 0)
ASSIGN_OPS_OPERATORS(double2, ARG2(ARG(double2, x,), ARG(double2, y,)), 0)
ASSIGN_OPS_OPERATORS(double2, ARG2(ARG(double2, x,), ARG(double,  y,)), 0)
ASSIGN_OPS_OPERATORS(double3, ARG2(ARG(double3, x,), ARG(double3, y,)), 0)
ASSIGN_OPS_OPERATORS(double3, ARG2(ARG(double3, x,), ARG(double,  y,)), 0)
ASSIGN_OPS_OPERATORS(double4, ARG2(ARG(double4, x,), ARG(double4, y,)), 0)
ASSIGN_OPS_OPERATORS(double4, ARG2(ARG(double4, x,), ARG(double,  y,)), 0)

// operator+=, operator-=, operator*=, operator/= for color
ASSIGN_OPS_OPERATORS(color,   ARG2(ARG(color,   x,), ARG(color,   y,)), 0)
ASSIGN_OPS_OPERATORS(color,   ARG2(ARG(color,   x,), ARG(float,   y,)), 0)

// operator+=, operator-= for matrix op matrix
ASSIGN_PLUS_MINUS_OPERATORS(float2x2,  ARG2(ARG(float2x2,  x,), ARG(float2x2,  y,)), 0)
ASSIGN_PLUS_MINUS_OPERATORS(float2x3,  ARG2(ARG(float2x3,  x,), ARG(float2x3,  y,)), 0)
ASSIGN_PLUS_MINUS_OPERATORS(float2x4,  ARG2(ARG(float2x4,  x,), ARG(float2x4,  y,)), 0)
ASSIGN_PLUS_MINUS_OPERATORS(float3x2,  ARG2(ARG(float3x2,  x,), ARG(float3x2,  y,)), 0)
ASSIGN_PLUS_MINUS_OPERATORS(float3x3,  ARG2(ARG(float3x3,  x,), ARG(float3x3,  y,)), 0)
ASSIGN_PLUS_MINUS_OPERATORS(float3x4,  ARG2(ARG(float3x4,  x,), ARG(float3x4,  y,)), 0)
ASSIGN_PLUS_MINUS_OPERATORS(float4x2,  ARG2(ARG(float4x2,  x,), ARG(float4x2,  y,)), 0)
ASSIGN_PLUS_MINUS_OPERATORS(float4x3,  ARG2(ARG(float4x3,  x,), ARG(float4x3,  y,)), 0)
ASSIGN_PLUS_MINUS_OPERATORS(float4x4,  ARG2(ARG(float4x4,  x,), ARG(float4x4,  y,)), 0)
ASSIGN_PLUS_MINUS_OPERATORS(double2x2, ARG2(ARG(double2x2, x,), ARG(double2x2, y,)), 0)
ASSIGN_PLUS_MINUS_OPERATORS(double2x3, ARG2(ARG(double2x3, x,), ARG(double2x3, y,)), 0)
ASSIGN_PLUS_MINUS_OPERATORS(double2x4, ARG2(ARG(double2x4, x,), ARG(double2x4, y,)), 0)
ASSIGN_PLUS_MINUS_OPERATORS(double3x2, ARG2(ARG(double3x2, x,), ARG(double3x2, y,)), 0)
ASSIGN_PLUS_MINUS_OPERATORS(double3x3, ARG2(ARG(double3x3, x,), ARG(double3x3, y,)), 0)
ASSIGN_PLUS_MINUS_OPERATORS(double3x4, ARG2(ARG(double3x4, x,), ARG(double3x4, y,)), 0)
ASSIGN_PLUS_MINUS_OPERATORS(double4x2, ARG2(ARG(double4x2, x,), ARG(double4x2, y,)), 0)
ASSIGN_PLUS_MINUS_OPERATORS(double4x3, ARG2(ARG(double4x3, x,), ARG(double4x3, y,)), 0)
ASSIGN_PLUS_MINUS_OPERATORS(double4x4, ARG2(ARG(double4x4, x,), ARG(double4x4, y,)), 0)

// operator /= for matrix / scalar
OPERATOR(float2x2,  OK_DIVIDE_ASSIGN, ARG2(ARG(float2x2,  x,), ARG(float,  y,)), 0)
OPERATOR(float2x3,  OK_DIVIDE_ASSIGN, ARG2(ARG(float2x3,  x,), ARG(float,  y,)), 0)
OPERATOR(float2x4,  OK_DIVIDE_ASSIGN, ARG2(ARG(float2x4,  x,), ARG(float,  y,)), 0)
OPERATOR(float3x2,  OK_DIVIDE_ASSIGN, ARG2(ARG(float3x2,  x,), ARG(float,  y,)), 0)
OPERATOR(float3x3,  OK_DIVIDE_ASSIGN, ARG2(ARG(float3x3,  x,), ARG(float,  y,)), 0)
OPERATOR(float3x4,  OK_DIVIDE_ASSIGN, ARG2(ARG(float3x4,  x,), ARG(float,  y,)), 0)
OPERATOR(float4x2,  OK_DIVIDE_ASSIGN, ARG2(ARG(float4x2,  x,), ARG(float,  y,)), 0)
OPERATOR(float4x3,  OK_DIVIDE_ASSIGN, ARG2(ARG(float4x3,  x,), ARG(float,  y,)), 0)
OPERATOR(float4x4,  OK_DIVIDE_ASSIGN, ARG2(ARG(float4x4,  x,), ARG(float,  y,)), 0)
OPERATOR(double2x2, OK_DIVIDE_ASSIGN, ARG2(ARG(double2x2, x,), ARG(double, y,)), 0)
OPERATOR(double2x3, OK_DIVIDE_ASSIGN, ARG2(ARG(double2x3, x,), ARG(double, y,)), 0)
OPERATOR(double2x4, OK_DIVIDE_ASSIGN, ARG2(ARG(double2x4, x,), ARG(double, y,)), 0)
OPERATOR(double3x2, OK_DIVIDE_ASSIGN, ARG2(ARG(double3x2, x,), ARG(double, y,)), 0)
OPERATOR(double3x3, OK_DIVIDE_ASSIGN, ARG2(ARG(double3x3, x,), ARG(double, y,)), 0)
OPERATOR(double3x4, OK_DIVIDE_ASSIGN, ARG2(ARG(double3x4, x,), ARG(double, y,)), 0)
OPERATOR(double4x2, OK_DIVIDE_ASSIGN, ARG2(ARG(double4x2, x,), ARG(double, y,)), 0)
OPERATOR(double4x3, OK_DIVIDE_ASSIGN, ARG2(ARG(double4x3, x,), ARG(double, y,)), 0)
OPERATOR(double4x4, OK_DIVIDE_ASSIGN, ARG2(ARG(double4x4, x,), ARG(double, y,)), 0)

// ----------------------------------------------------------------------

#if 0
#undef EXPLICIT
#undef IMPLICIT
#undef ARRUSE
#undef ARRDEF
#undef ARG16
#undef ARG12
#undef ARG9
#undef ARG8
#undef ARG7
#undef ARG6
#undef ARG5
#undef ARG4
#undef ARG3
#undef ARG2
#undef ARG1
#undef ARG0
#undef UDEFARG
#undef DEFARG
#undef UARG
#undef ARG
#endif
#undef BUILTIN_TYPE_END
#undef BUILTIN_TYPE_BEGIN
#undef ENUM_VALUE
#undef FIELD
#undef CONSTRUCTOR
#undef METHOD_SYNONYM
#undef METHOD_OVERLOAD
#undef METHOD
#undef MUL_ASSIGN_OPERATOR
#undef MUL_OPERATOR
#undef ASSIGN_PLUS_MINUS_OPERATORS
#undef ASSIGN_OPS_OPERATORS
#undef UNARY_OPERATORS
#undef BINARY_PLUS_MINUS_OPERATORS
#undef ASSIGN_OPERATOR
#undef INC_DEC_OPERATORS
#undef BIN_LOG_OPERATORS
#undef REL_OPERATORS
#undef EQ_OPERATORS
#undef OPERATOR
