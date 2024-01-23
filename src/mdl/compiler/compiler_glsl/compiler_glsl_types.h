/******************************************************************************
 * Copyright (c) 2015-2024, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_COMPILER_GLSL_TYPES_H
#define MDL_COMPILER_GLSL_TYPES_H 1

#include <mi/mdl/mdl_iowned.h>

#include "mdl/compiler/compilercore/compilercore_memory_arena.h"
#include "mdl/compiler/compilercore/compilercore_array_ref.h"

#include "compiler_glsl_assert.h"
#include "compiler_glsl_cc_conf.h"
#include "compiler_glsl_symbols.h"

namespace mi {
namespace mdl {
namespace glsl {

class Symbol;

/// The GLSL type.
class Type {
public:
    enum Kind {
        TK_ALIAS,       ///< An alias for another type, aka typedef.

        TK_VOID,        ///< The void type.
        TK_BOOL,        ///< The boolean type.
        TK_INT,         ///< The signed integer type.
        TK_UINT,        ///< The unsigned integer type. Available since 1.30.
        TK_HALF,        ///< The half-precision floating-point scalar. Available with ext.
        TK_FLOAT,       ///< The single-precision floating-point scalar.
        TK_DOUBLE,      ///< The single-precision floating-point scalar. Available since 4.00.

        TK_VECTOR,      ///< A vector type.
        TK_MATRIX,      ///< A matrix type.
        TK_ARRAY,       ///< An array type.
        TK_STRUCT,      ///< A struct type.
        TK_FUNCTION,    ///< A function type.
        TK_POINTER,     ///< A pointer type.

        TK_ATOMIC_UINT, ///< The atomic_uint type. Available since ES 3.30, non-es 4.20, ext.
        TK_IMAGE,       ///< An opaque image type.
        TK_SAMPLER,     ///< A opaque texture handle type.

        TK_INT8_T,      ///< The signed 8bit type. Available with ext.
        TK_UINT8_T,     ///< The unsigned 8bit type. Available with ext.
        TK_INT16_T,     ///< The signed 16bit type. Available with ext.
        TK_UINT16_T,    ///< The unsigned 16bit type. Available with ext.
        TK_INT64_T,     ///< The signed 64bit type. Available with ext.
        TK_UINT64_T,    ///< The unsigned 64bit type. Available with ext.

        TK_ERROR,       ///< The error type.
    };

    /// The type classes from the GLSL spec.
    enum Type_class {
        TC_other,       ///< Any other type not falling into this classes.
        TC_genType,     ///< float, vec2, vec3, or vec4
        TC_genIType,    ///< int, ivec2, ivec3, or ivec4
        TC_genUType,    ///< uint, uvec2, uvec3, or uvec4
        TC_genBType,    ///< bool, bvec2, bvec3, or bvec4
        TC_genDType,    ///< double, dvec2, dvec3, dvec4
        TC_mat,         ///< float matrix
        TC_dmat,        ///< double matrix
    };

    /// The possible kinds of type modifiers.
    enum Modifier {
        MK_NONE    = 0,         ///< auto-typed
        MK_CONST   = (1 << 1),  ///< a constant type
        MK_UNIFORM = (1 << 2),  ///< a uniform type
        MK_VARYING = (1 << 4),  ///< a varying type
        MK_LOWP    = (1 << 5),  ///< a low precision type
        MK_MEDIUMP = (1 << 6),  ///< a medium precision type
        MK_HIGHP   = (1 << 7),  ///< a high precision type
    };

    /// A bitset of type modifier.
    typedef unsigned Modifiers;

public:
    /// Get the type name.
    Symbol *get_sym() { return m_sym; }

    /// Get the type kind.
    virtual Kind get_kind() = 0;

    /// Access to the base type.
    virtual Type *skip_type_alias();

    /// Get the type modifiers of a type
    virtual Modifiers get_type_modifiers();

    /// Get the type class of this type.
    virtual Type_class get_type_class() = 0;

protected:
    /// Constructor.
    ///
    /// \param sym  the name symbol of this type
    explicit Type(Symbol *sym);

private:
    /// The name symbol of the type.
    Symbol *m_sym;
};

/// GLSL alias types. We use this only to add modifiers.
class Type_alias : public Type
{
    typedef Type Base;
    friend class mi::mdl::Arena_builder;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_ALIAS;

    /// Get the type kind.
    Kind get_kind() GLSL_FINAL;

    /// Access to the base type.
    Type *skip_type_alias() GLSL_FINAL;

    /// Get the modifier set of this type.
    Modifiers get_type_modifiers() GLSL_FINAL;

    /// Get the type class of this type.
    Type_class get_type_class() GLSL_FINAL;

    /// Get the aliased type.
    Type *get_aliased_type() { return m_aliased_type; }

private:
    /// Constructor.
    ///
    /// \param aliased    the aliased type
    /// \param modifiers  the type modifiers
    /// \param sym        the name symbol of this alias type
    explicit Type_alias(
        Type       *aliased,
        Modifiers  modifiers,
        Symbol     *sym);

private:
    /// The aliased type.
    Type * const m_aliased_type;

    /// The type modifiers of this alias type.
    Modifiers const m_modifiers;
};

/// The error type represents a type error in the syntax tree.
class Type_error : public Type
{
    typedef Type Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_ERROR;

    /// Get the type kind.
    Kind get_kind() GLSL_FINAL;

    /// Get the type class of this type.
    Type_class get_type_class() GLSL_FINAL;

public:
    /// Constructor.
    explicit Type_error();
};

/// An scalar type.
class Type_scalar : public Type
{
    typedef Type Base;
protected:
    /// Constructor.
    ///
    /// \param sym  the name symbol of this scalar type
    explicit Type_scalar(Symbol *sym);
};

/// The GLSL void type.
class Type_void : public Type_scalar
{
    typedef Type_scalar Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_VOID;

    /// Get the type kind.
    Kind get_kind() GLSL_FINAL;

    /// Get the type class of this type.
    Type_class get_type_class() GLSL_FINAL;

public:
    /// Constructor.
    explicit Type_void();
};

/// The GLSL boolean type.
class Type_bool : public Type_scalar
{
    typedef Type_scalar Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_BOOL;

    /// Get the type kind.
    Kind get_kind() GLSL_FINAL;

    /// Get the type class of this type.
    Type_class get_type_class() GLSL_FINAL;

public:
    /// Constructor.
    explicit Type_bool();
};

/// The two-complement type.
class Type_two_complement : public Type_scalar
{
    typedef Type_scalar Base;

protected:
    /// Get the bit-size of this type.
    size_t get_bit_size() const { return m_bitsize; }

protected:
    /// Constructor.
    ///
    /// \param sym      the symbol of this type
    /// \param bitsize  the bit-size of this type
    explicit Type_two_complement(
        Symbol *sym,
        size_t bitsize);

private:
    /// The bitsize of this type.
    size_t m_bitsize;
};

/// The signed two-complement type.
class Type_signed_int : public Type_two_complement
{
    typedef Type_two_complement Base;

public:
    /// Get the type class of this type.
    Type_class get_type_class() GLSL_FINAL;

protected:
    /// Constructor.
    ///
    /// \param sym      the symbol of this type
    /// \param bitsize  the bit-size of this type
    explicit Type_signed_int(
        Symbol *sym,
        size_t bitsize);
};

/// The unsigned two-complement type.
class Type_unsigned_int : public Type_two_complement
{
    typedef Type_two_complement Base;

public:
    /// Get the type class of this type.
    Type_class get_type_class() GLSL_FINAL;

protected:
    /// Constructor.
    ///
    /// \param sym      the symbol of this type
    /// \param bitsize  the bit-size of this type
    explicit Type_unsigned_int(
        Symbol *sym,
        size_t bitsize);
};

/// The GLSL signed integer type.
class Type_int : public Type_signed_int
{
    typedef Type_signed_int Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_INT;

    /// Get the type kind.
    Kind get_kind() GLSL_FINAL;

public:
    /// Constructor.
    explicit Type_int();
};

/// The GLSL unsigned integer type. Available since version 1.30.
class Type_uint : public Type_unsigned_int
{
    typedef Type_unsigned_int Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_UINT;

    /// Get the type kind.
    Kind get_kind() GLSL_FINAL;

public:
    /// Constructor.
    explicit Type_uint();
};

/// The GLSL signed 8bit integer type. Available with GL_ARB_gpu_shader5 extension.
class Type_int8_t : public Type_signed_int
{
    typedef Type_signed_int Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_INT8_T;

    /// Get the type kind.
    Kind get_kind() GLSL_FINAL;

public:
    /// Constructor.
    explicit Type_int8_t();
};

/// The GLSL unsigned 8bit integer type. Available with GL_ARB_gpu_shader5 extension.
class Type_uint8_t : public Type_unsigned_int
{
    typedef Type_unsigned_int Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_UINT8_T;

    /// Get the type kind.
    Kind get_kind() GLSL_FINAL;

public:
    /// Constructor.
    explicit Type_uint8_t();
};

/// The GLSL signed 16bit integer type. Available with GL_ARB_gpu_shader5 extension.
class Type_int16_t : public Type_signed_int
{
    typedef Type_signed_int Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_INT16_T;

    /// Get the type kind.
    Kind get_kind() GLSL_FINAL;

public:
    /// Constructor.
    explicit Type_int16_t();
};

/// The GLSL unsigned 16bit integer type. Available with GL_ARB_gpu_shader5 extension.
class Type_uint16_t : public Type_unsigned_int
{
    typedef Type_unsigned_int Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_UINT16_T;

    /// Get the type kind.
    Kind get_kind() GLSL_FINAL;

public:
    /// Constructor.
    explicit Type_uint16_t();
};

/// The GLSL signed 32bit integer type. Available with GL_ARB_gpu_shader5 extension
/// and equivalent to int.
typedef Type_int  Type_int32_t;

/// The GLSL unsigned 32bit integer type. Available with GL_ARB_gpu_shader5 extension
/// and equivalent to int.
typedef Type_uint Type_uint32_t;

/// The GLSL signed 64bit integer type. Available with GL_ARB_gpu_shader5 extension.
class Type_int64_t : public Type_signed_int
{
    typedef Type_signed_int Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_INT64_T;

    /// Get the type kind.
    Kind get_kind() GLSL_FINAL;

public:
    /// Constructor.
    explicit Type_int64_t();
};

/// The GLSL unsigned 64bit integer type. Available with GL_ARB_gpu_shader5 extension.
class Type_uint64_t : public Type_unsigned_int
{
    typedef Type_unsigned_int Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_UINT64_T;

    /// Get the type kind.
    Kind get_kind() GLSL_FINAL;

public:
    /// Constructor.
    explicit Type_uint64_t();
};

/// The GLSL half type.
class Type_half : public Type_scalar
{
    typedef Type_scalar Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_HALF;

    /// Get the type kind.
    Kind get_kind() GLSL_FINAL;

    /// Get the type class of this type.
    Type_class get_type_class() GLSL_FINAL;

public:
    /// Constructor.
    explicit Type_half();
};

/// The GLSL float type.
class Type_float : public Type_scalar
{
    typedef Type_scalar Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_FLOAT;

    /// Get the type kind.
    Kind get_kind() GLSL_FINAL;

    /// Get the type class of this type.
    Type_class get_type_class() GLSL_FINAL;

public:
    /// Constructor.
    explicit Type_float();
};

/// The GLSL double type.  Available since version 4.00.
class Type_double : public Type_scalar
{
    typedef Type_scalar Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_DOUBLE;

    /// Get the type kind.
    Kind get_kind() GLSL_FINAL;

    /// Get the type class of this type.
    Type_class get_type_class() GLSL_FINAL;

public:
    /// Constructor.
    explicit Type_double();
};

/// A compound type.
class Type_compound : public Type
{
    typedef Type Base;
public:
    /// Get the compound type at index.
    virtual Type *get_compound_type(size_t index) = 0;

    /// Get the number of compound elements.
    virtual size_t get_compound_size() = 0;

protected:
    /// Constructor.
    ///
    /// \param sym  the name symbol of this compound type
    explicit Type_compound(Symbol *sym);
};

/// GLSL vector types.
class Type_vector : public Type_compound
{
    typedef Type_compound Base;
public:
    /// The kind of this subclass.
    static const Kind s_kind = TK_VECTOR;

    /// Get the type kind.
    Kind get_kind() GLSL_FINAL;

    /// Get the type class of this type.
    Type_class get_type_class() GLSL_FINAL;

    /// Get the compound type at index.
    Type *get_compound_type(size_t index) GLSL_FINAL;

    /// Get the number of compound elements.
    size_t get_compound_size() GLSL_FINAL;

    /// Get the type of the vector elements.
    Type_scalar *get_element_type() { return m_base; }

    /// Get the number of vector elements.
    size_t get_size() { return m_size; }

public:
    /// Constructor.
    ///
    /// \param base  the base type
    /// \param size  the size of the vector
    explicit Type_vector(
        Type_scalar *base,
        size_t      size);

private:
    /// Get the symbol for given type.
    Symbol *get_vector_sym(
        Type_scalar *base,
        size_t      size);

private:
    /// The base type.
    Type_scalar * const m_base;

    /// The vector size.
    size_t const m_size;
};

/// GLSL matrix types.
class Type_matrix : public Type_compound
{
    typedef Type_compound Base;
public:
    /// The kind of this subclass.
    static const Kind s_kind = TK_MATRIX;

    /// Get the type kind.
    Kind get_kind() GLSL_FINAL;

    /// Get the type class of this type.
    Type_class get_type_class() GLSL_FINAL;

    /// Get the compound type at index.
    Type *get_compound_type(size_t index) GLSL_FINAL;

    /// Get the number of compound elements.
    size_t get_compound_size() GLSL_FINAL;

    /// Get the type of the matrix elements.
    Type_vector *get_element_type() { return m_column_type; }

    /// Get the number of matrix columns.
    size_t get_columns() { return m_columns; }

public:
    /// Constructor.
    ///
    /// \param column_type  the type of a matrix column
    /// \param n_columns    number of columns
    explicit Type_matrix(
        Type_vector *column_type,
        size_t      n_columns);

private:
    /// Get the symbol for given type.
    Symbol *get_matrix_sym(
        Type_vector *column_type,
        size_t      columns);

private:
    /// The type of a column.
    Type_vector * const m_column_type;

    /// The number of columns.
    size_t const m_columns;
};

/// GLSL array types.
class Type_array : public Type_compound
{
    typedef Type_compound Base;
    friend class mi::mdl::Arena_builder;
public:
    /// The kind of this subclass.
    static const Kind s_kind = TK_ARRAY;

    /// Get the type kind.
    Kind get_kind() GLSL_FINAL;

    /// Get the type class of this type.
    Type_class get_type_class() GLSL_FINAL;

    /// Get the compound type at index.
    Type *get_compound_type(size_t index) GLSL_FINAL;

    /// Get the number of compound elements.
    size_t get_compound_size() GLSL_FINAL;

    /// Get the type of the array elements.
    Type *get_element_type() { return m_element_type; }

    /// Get the size of the array.
    size_t get_size() { return m_size; }

    /// Returns true if this is an unsized array.
    bool is_unsized() const { return m_size == 0; }

private:
    /// Constructor.
    ///
    /// \param element_type  the array element type
    /// \param size          the array size
    /// \param sym           the array name symbol
    explicit Type_array(
        Type   *element_type,
        size_t size,
        Symbol *sym);

private:
    /// The element type of this array type.
    Type * const m_element_type;

    /// The size of this array.
    size_t const m_size;
};

/// GLSL struct types.
class Type_struct : public Type_compound
{
    typedef Type_compound Base;
    friend class mi::mdl::Arena_builder;
public:
    /// A struct field.
    class Field {
    public:
        /// Get the type of this field.
        Type *get_type() { return m_type; }

        /// Get the name symbol of this field.
        Symbol *get_symbol() { return m_symbol; }

    public:
        /// Constructor.
        Field(Type *type, Symbol *sym)
        : m_type(type)
        , m_symbol(sym)
        {
        }

        /// Default Constructor.
        Field()
        : m_type(NULL)
        , m_symbol(NULL)
        {
        }

    private:
        /// The type of this field.
        Type *m_type;

        /// The name of this field.
        Symbol *m_symbol;
    };

public:
    /// The kind of this subclass.
    static const Kind s_kind = TK_STRUCT;

    /// Get the type kind.
    Kind get_kind() GLSL_FINAL;

    /// Get the type class of this type.
    Type_class get_type_class() GLSL_FINAL;

    /// Get the compound type at index.
    Type *get_compound_type(size_t index) GLSL_FINAL;

    /// Get the number of compound elements.
    size_t get_compound_size() GLSL_FINAL;

    /// Get the number of fields.
    size_t get_field_count() { return m_n_fields; }

    /// Get a field.
    /// \param index    The index of the field.
    Field *get_field(size_t index);

private:
    /// Constructor.
    ///
    /// \param arena   the arena to allocate on
    /// \param fields  struct fields
    /// \param sym     the name symbol of this struct
    explicit Type_struct(
        Memory_arena           *arena,
        Array_ref<Field> const &fields,
        Symbol                 *sym);

private:
    /// The fields of this struct.
    Field * const m_fields;

    /// The number of fields.
    size_t const m_n_fields;
};

/// GLSL function type.
class Type_function : public Type
{
    typedef Type Base;
    friend class mi::mdl::Arena_builder;
public:
    /// A function parameter.
    class Parameter {
    public:
        /// Possible parameter modifiers.
        enum Modifier {
            PM_IN    = 1,              ///< copy-in
            PM_OUT   = 2,              ///< copy out
            PM_INOUT = PM_IN + PM_OUT  ///< copy in and out
        };

        /// Get the type of the parameter.
        Type *get_type() const { return m_type; }

        /// Get the parameter modifier of this parameter.
        Modifier get_modifier() { return m_modifier; }

    public:
        /// Constructor.
        ///
        /// \param type  the type of this parameter
        /// \param mod   the modifier of this parameter
        Parameter(
            Type     *type,
            Modifier mod)
        : m_type(type)
        , m_modifier(mod)
        {
        }

    private:
        /// The type of the parameter.
        Type *m_type;

        /// The parameter modifier.
        Modifier m_modifier;
    };

public:
    /// The kind of this subclass.
    static const Kind s_kind = TK_FUNCTION;

    /// Get the type kind.
    Kind get_kind() GLSL_FINAL;

    /// Get the type class of this type.
    Type_class get_type_class() GLSL_FINAL;

    /// Get the return type of the function.
    Type *get_return_type() { return m_ret_type; }

    /// Get the number of parameters of the function.
    size_t get_parameter_count() { return m_n_params; }

    /// Get a parameter of the function type.
    ///
    /// \param index    The index of the parameter in the parameter list.
    Parameter *get_parameter(size_t index);

private:
    /// Constructor.
    ///
    /// \param arena     the arena to allocate on
    /// \param ret_type  the return type
    /// \param params    the function parameters
    explicit Type_function(
        Memory_arena               *arena,
        Type                       *ret_type,
        Array_ref<Parameter> const &params);

private:
    /// The return type.
    Type * const m_ret_type;

    /// The parameters of this function type.
    Parameter * const m_params;

    /// The number of parameters.
    size_t const m_n_params;
};

/// GLSL pointer types (available with some extensions).
class Type_pointer : public Type
{
    typedef Type Base;
    friend class mi::mdl::Arena_builder;
public:
    /// The kind of this subclass.
    static const Kind s_kind = TK_POINTER;

    /// Get the type kind.
    Kind get_kind() GLSL_FINAL;

    /// Get the type class of this type.
    Type_class get_type_class() GLSL_FINAL;

    /// Get the points-to type of the pointer elements.
    Type *get_points_to_type() { return m_points_to_type; }

private:
    /// Constructor.
    ///
    /// \param sym             the name symbol of this parameter or NULL if nameless
    /// \param points_to_type  the points-to type
    explicit Type_pointer(
        Symbol *sym,
        Type   *points_to_type);

private:
    /// The points-to type of this pointer type.
    Type * const m_points_to_type;
};

/// The base class for all opaque types.
class Type_opaque : public Type
{
    typedef Type Base;
public:
    /// Get the type class of this type.
    Type_class get_type_class() GLSL_FINAL;

    /// Get the base type of this opaque type.
    Type *get_base_type() { return m_base_tp; }

protected:
    /// Constructor.
    ///
    /// \param sym  the symbol of this type
    explicit Type_opaque(
        Symbol      *sym,
        Type_scalar *base);

protected:
    /// The base type of this opaque type.
    Type_scalar *m_base_tp;
};

/// The GLSL atomic unsigned integer type. Available since ES 3.30, non-ES 4.30, or with
/// GL_ARB_shader_atomic_counters extension.
class Type_atomic_uint : public Type_opaque
{
    typedef Type_opaque Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_ATOMIC_UINT;

    /// Get the type kind.
    Kind get_kind() GLSL_FINAL;

public:
    /// Constructor.
    explicit Type_atomic_uint();
};

/// GLSL texture shapes.
enum Texture_shape {
    SHAPE_1D,                   ///< 1D texture
    SHAPE_2D,                   ///< 2D texture
    SHAPE_3D,                   ///< 3D texture
    SHAPE_CUBE,                 ///< cube mapped texture

    SHAPE_1D_SHADOW,            ///< 1D depth texture with comparison
    SHAPE_2D_SHADOW,            ///< 2D depth texture with comparison
    SHAPE_CUBE_SHADOW,          ///< cube map depth texture with comparison

    SHAPE_1D_ARRAY,             ///< 1D array texture
    SHAPE_2D_ARRAY,             ///< 2D array texture

    SHAPE_1D_ARRAY_SHADOW,      ///< 1D array depth texture with comparison
    SHAPE_2D_ARRAY_SHADOW,      ///< 2D array depth texture with comparison

    SHAPE_2D_RECT,              ///< rectangle texture
    SHAPE_2D_RECT_SHADOW,       ///< rectangle texture with comparison

    SHAPE_BUFFER,               ///< buffer texture

    SHAPE_2DMS,                 ///< 2D multi-sample texture
    SHAPE_2DMS_ARRAY,           ///< 2D multi-sample array texture

    SHAPE_CUBE_ARRAY,           ///< cube map array texture
    SHAPE_CUBE_ARRAY_SHADOW,    ///< cube map array depth texture with comparison

    SHAPE_EXTERNAL_OES,         ///< external texture; needs GL_OES_EGL_image_external
};

/// A GLSL texture sampler type.
class Type_sampler : public Type_opaque
{
    typedef Type_opaque Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_SAMPLER;

    /// Get the type kind.
    Kind get_kind() GLSL_FINAL;

    /// Get the texture shape.
    Texture_shape get_shape() { return m_shape; }

public:
    /// Constructor.
    ///
    /// \param base_tp  the base type of this sampler
    /// \param shape    the texture shape of this sampler
    explicit Type_sampler(
        Type_scalar   *base_tp,
        Texture_shape shape);

private:
    /// The texture shape of this sampler.
    Texture_shape m_shape;
};

/// A GLSL image type.
class Type_image : public Type_opaque
{
    typedef Type_opaque Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_IMAGE;

    /// Get the type kind.
    Kind get_kind() GLSL_FINAL;

    /// Get the texture shape.
    Texture_shape get_shape() { return m_shape; }

public:
    /// Constructor.
    ///
    /// \param base_tp  the base type of this sampler
    /// \param shape    the texture shape of this sampler
    explicit Type_image(
        Type_scalar   *base_tp,
        Texture_shape shape);

private:
    /// The texture shape of this sampler.
    Texture_shape m_shape;
};

/// Cast to subtype or return NULL if types do not match.
template<typename T>
T *as(Type *type) {
    type = type->skip_type_alias();
    return (type->get_kind() == T::s_kind) ? static_cast<T *>(type) : NULL;
}

/// Cast to Type_alias or return NULL if types do not match.
template<>
inline Type_alias *as<Type_alias>(Type *type) {
    return (type->get_kind() == Type_alias::s_kind) ? static_cast<Type_alias *>(type) : NULL;
}

/// Cast to Type_scalar or return NULL if types do not match.
template<>
inline Type_scalar *as<Type_scalar>(Type *type) {
    type = type->skip_type_alias();
    switch (type->get_kind()) {
    case Type::TK_BOOL:
    case Type::TK_INT:
    case Type::TK_UINT:
    case Type::TK_FLOAT:
    case Type::TK_DOUBLE:
        return static_cast<Type_scalar *>(type);
    default:
        return NULL;
    }
}

/// Cast to Type_two_complement or return NULL if types do not match.
template<>
inline Type_two_complement *as<Type_two_complement>(Type *type) {
    type = type->skip_type_alias();
    switch (type->get_kind()) {
    case Type::TK_INT:
    case Type::TK_UINT:
        return static_cast<Type_two_complement *>(type);
    default:
        return NULL;
    }
}

/// Cast to Type_compound or return NULL if types do not match.
template<>
inline Type_compound *as<Type_compound>(Type *type) {
    type = type->skip_type_alias();
    switch (type->get_kind()) {
    case Type::TK_VECTOR:
    case Type::TK_MATRIX:
    case Type::TK_ARRAY:
    case Type::TK_STRUCT:
        return static_cast<Type_compound *>(type);
    default:
        return 0;
    }
}

/// Cast to Type_opaque or return NULL if types do not match.
template<>
inline Type_opaque *as<Type_opaque>(Type *type) {
    type = type->skip_type_alias();
    switch (type->get_kind()) {
    case Type::TK_ATOMIC_UINT:
    case Type::TK_IMAGE:
    case Type::TK_SAMPLER:
        return static_cast<Type_opaque *>(type);
    default:
        return NULL;
    }
}
/// Check if a type is of a certain type.
template<typename T>
bool is(Type *type) {
    return type->get_kind() == T::s_kind;
}

/// Check if a type is of type Type_scalar.
template<>
inline bool is<Type_scalar>(Type *type) {
    switch (type->get_kind()) {
    case Type::TK_BOOL:
    case Type::TK_INT:
    case Type::TK_UINT:
    case Type::TK_INT8_T:
    case Type::TK_UINT8_T:
    case Type::TK_INT16_T:
    case Type::TK_UINT16_T:
    case Type::TK_INT64_T:
    case Type::TK_UINT64_T:
    case Type::TK_HALF:
    case Type::TK_FLOAT:
    case Type::TK_DOUBLE:
        return true;
    default:
        return false;
    }
}

/// Check if a type is of type Type_two_complement.
template<>
inline bool is<Type_two_complement>(Type *type) {
    switch (type->get_kind()) {
    case Type::TK_INT8_T:
    case Type::TK_UINT8_T:
    case Type::TK_INT16_T:
    case Type::TK_UINT16_T:
    case Type::TK_INT:
    case Type::TK_UINT:
    case Type::TK_INT64_T:
    case Type::TK_UINT64_T:
        return true;
    default:
        return false;
    }
}

/// Check if a type is of type Type_compound.
template<>
inline bool is<Type_compound>(Type *type) {
    switch (type->get_kind()) {
    case Type::TK_VECTOR:
    case Type::TK_MATRIX:
    case Type::TK_ARRAY:
    case Type::TK_STRUCT:
        return true;
    default:
        return false;
    }
}

/// Check if a type is of type Type_opaque.
template<>
inline bool is<Type_opaque>(Type *type) {
    switch (type->get_kind()) {
    case Type::TK_ATOMIC_UINT:
    case Type::TK_IMAGE:
    case Type::TK_SAMPLER:
        return true;
    default:
        return false;
    }
}

/// A static_cast with check in debug mode
template <typename T>
inline T *cast(Type *arg) {
    GLSL_ASSERT(arg == NULL || is<T>(arg));
    return static_cast<T *>(arg);
}

/// The interface for creating types.
/// An IType_factory interface can be obtained by calling
/// the method get_type_factory() on the interfaces IModule and IValue_factory.
class Type_factory : public Interface_owned
{
    /// A type cache key.
    struct Type_cache_key {
        enum Kind {
            KEY_FUNC_TYPE, ///< a function type itself
            KEY_FUNC_KEY,  ///< a function type search key
            KEY_ALIAS,     ///< an alias search key
            KEY_ARRAY,     ///< an array search key
            KEY_POINTER,   ///< a pointer search key
        };
        Kind kind;

        struct Function_type {
            Type_function::Parameter const *params;
            size_t                         n_params;
        };

        Type *type;

        union {
            // empty for KEY_FUNC_TYPE

            // for KEY_FUNC_KEY
            Function_type func;

            // for KEY_ALIAS
            struct {
                Type::Modifiers mod;
            } alias;

            // for KEY_ARRAY
            struct {
                size_t          size;
            } array;
        } u; //-V730_NOINIT

        /// Create a key for a function type.
        /*implicit*/ Type_cache_key(Type_function *func)
        : kind(KEY_FUNC_TYPE), type(func)
        {
        }

        /// Create a key for a function type.
        Type_cache_key(
            Type                           *ret,
            Type_function::Parameter const *params,
            size_t                         n)
        : kind(KEY_FUNC_KEY), type(ret)
        {
            u.func.params   = params;
            u.func.n_params = n;
        }

        /// Create a key for an alias type.
        Type_cache_key(Type *t, Type::Modifiers m)
        : kind(KEY_ALIAS), type(t)
        {
            u.alias.mod = m;
        }

        /// Create a key for an array type.
        Type_cache_key(size_t size, Type *t)
        : kind(KEY_ARRAY), type(t)
        {
            u.array.size = size;
        }

        /// Create a key for a pointer type.
        explicit Type_cache_key(Type *t)
        : kind(KEY_POINTER), type(t)
        {
        }

        /// Functor to hash a type cache keys.
        struct Hash {
            size_t operator()(Type_cache_key const &key) const
            {
                switch (key.kind) {
                case KEY_FUNC_TYPE:
                    {
                        Type_function *ft = static_cast<Type_function *>(key.type);
                        Type  *ret_type = ft->get_return_type();
                        size_t n_params = size_t(ft->get_parameter_count());

                        size_t t = ret_type - (Type *)0;
                        t = ((t) >> 3) ^ (t >> 16) ^           //-V2007
                            size_t(KEY_FUNC_TYPE) ^ n_params;

                        for (size_t i = 0; i < n_params; ++i) {
                            Type_function::Parameter *param  = ft->get_parameter(i);
                            Type                     *p_type = param->get_type();

                            t *= 3;
                            t ^= (char *)p_type - (char *)0;
                        }
                        return t;
                    }
                case KEY_FUNC_KEY:
                    {
                        size_t t = key.type - (Type *)0;
                        t = ((t) >> 3) ^ (t >> 16) ^                      //-V2007
                            size_t(KEY_FUNC_TYPE) ^ key.u.func.n_params;

                        Type_function::Parameter const *p = key.u.func.params;
                        for (size_t i = 0; i < key.u.func.n_params; ++i) {
                            t *= 3;
                            t ^= (char *)p[i].get_type() - (char *)0;
                        }
                        return t;
                    }
                case KEY_ALIAS:
                    {
                        size_t t = key.type - (Type *)0;
                        return ((t) >> 3) ^ (t >> 16) ^
                            size_t(key.kind) ^
                            key.u.alias.mod;
                    }
                case KEY_ARRAY:
                    {
                        size_t t = key.type - (Type *)0;
                        return ((t) >> 3) ^ (t >> 16) ^
                            size_t(key.kind) ^
                            key.u.array.size;
                    }
                case KEY_POINTER:
                    {
                        size_t t = key.type - (Type *)0;
                        return ((t) >> 3) ^ (t >> 16) ^
                            size_t(key.kind);
                    }
                default:
                    return 0;
                }
            }
        };

        /// Functor to compare two type cache keys.
        struct Equal {
            bool operator() (Type_cache_key const &a, Type_cache_key const &b) const
            {
                if (a.kind != b.kind) {
                    Type_function       *ft = NULL;
                    Type                *rt = NULL;
                    Function_type const *sk = NULL;

                    // compare a function type and a function search key
                    if (a.kind == KEY_FUNC_TYPE && b.kind == KEY_FUNC_KEY) {
                        ft = static_cast<Type_function *>(a.type);
                        sk = &b.u.func;
                        rt = b.type;
                    } else if (a.kind == KEY_FUNC_KEY && b.kind == KEY_FUNC_TYPE) {
                        ft = static_cast<Type_function *>(b.type);
                        sk = &a.u.func;
                        rt = a.type;
                    }

                    if (ft != NULL) {
                        if (rt != ft->get_return_type())
                            return false;
                        if (ft->get_parameter_count() != sk->n_params)
                            return false;

                        for (size_t i = 0; i < sk->n_params; ++i) {
                            Type_function::Parameter *param = ft->get_parameter(i);

                            Type   *p_type = param->get_type();

                            if (p_type != sk->params[i].get_type()) {
                                return false;
                            }
                        }
                        return true;
                    }
                    return false;
                }
                switch (a.kind) {
                case KEY_FUNC_TYPE:
                    return a.type == b.type;
                case KEY_FUNC_KEY:
                    // should be NEVER inside the type hash
                    GLSL_ASSERT(!"function search key in type cache detected");
                    return false;
                case KEY_ALIAS:
                    return a.type == b.type && a.u.alias.mod == b.u.alias.mod;
                case KEY_ARRAY:
                    return a.type == b.type && a.u.array.size == b.u.array.size;
                case KEY_POINTER:
                    return a.type == b.type;
                default:
                    return false;
                }
            }
        };
    };

public:
    /// Get a new type alias instance.
    ///
    /// \param type       The aliased type.
    /// \param modifiers  The type modifiers.
    Type *get_alias(
        Type            *type,
        Type::Modifiers modifiers);

    /// Get the (singleton) error type instance.
    Type_error *get_error();

    /// Get the (singleton) void type instance.
    Type_void *get_void();

    /// Get the (singleton) bool type instance.
    Type_bool *get_bool();

    /// Get the (singleton) int type instance.
    Type_int *get_int();

    /// Get the (singleton) uint type instance.
    Type_uint *get_uint();

    /// Get the (singleton) int8_t type instance.
    Type_int8_t *get_int8_t();

    /// Get the (singleton) uint8_t type instance.
    Type_uint8_t *get_uint8_t();

    /// Get the (singleton) int16_t type instance.
    Type_int16_t *get_int16_t();

    /// Get the (singleton) uint16_t type instance.
    Type_uint16_t *get_uint16_t();

    /// Get the (singleton) int32_t type instance, same as int.
    Type_int32_t *get_int32_t();

    /// Get the (singleton) uint32_t type instance, same as uint.
    Type_uint32_t *get_uint32_t();

    /// Get the (singleton) int64_t type instance.
    Type_int64_t *get_int64_t();

    /// Get the (singleton) uint64_t type instance.
    Type_uint64_t *get_uint64_t();

    /// Get the (singleton) half type instance.
    Type_half *get_half();

    /// Get the (singleton) float type instance.
    Type_float *get_float();

    /// Get the (singleton) double type instance.
    Type_double *get_double();

    /// Get a vector type 1instance.
    ///
    /// \param element_type The type of the vector elements.
    /// \param size         The size of the vector.
    Type_vector *get_vector(
        Type_scalar *element_type,
        size_t      size);

    /// Get a matrix type instance.
    ///
    /// \param element_type The type of the matrix elements.
    /// \param columns      The number of columns.
    Type_matrix *get_matrix(
        Type_vector *element_type,
        size_t      columns);

    /// Get an array type instance.
    ///
    /// \param element_type The element type of the array.
    /// \param size         The size of the array.
    ///
    /// \return Type_error if element_type was of Type_error, an Type_array instance else.
    Type *get_array(
        Type   *element_type,
        size_t size);

    /// Get an unsized array type instance.
    ///
    /// \param element_type The element type of the array.
    ///
    /// \return Type_error if element_type was of Type_error, an Type_array instance else.
    Type *get_unsized_array(
        Type *element_type);

    /// Get a function type instance.
    ///
    /// \param return_type   The return type of the function.
    /// \param parameters    The parameters of the function.
    Type_function *get_function(
        Type                                      *return_type,
        Array_ref<Type_function::Parameter> const &parameters);

    /// Get a pointer type instance.
    ///
    /// \param points_to_type   The points-to type.
    Type *get_pointer(
        Type *points_to_type);

    /// Get a struct type instance.
    ///
    /// \param fields  The fields of the struct.
    /// \param sym     The name symbol of the struct.
    Type_struct *get_struct(
        Array_ref<Type_struct::Field> const &fields,
        Symbol                              *sym);

    /// Get the (singleton) opaque atomic_uint type instance.
    Type_atomic_uint *get_atomic_uint();

    /// Get an opaque sampler type
    ///
    /// \param base_tp  the base type of this sampler
    /// \param shape    the texture shape of this sampler
    Type_sampler *get_sampler(
        Type_scalar   *base_tp,
        Texture_shape shape);

    /// Get an opaque image type
    ///
    /// \param base_tp  the base type of this sampler
    /// \param shape    the texture shape of this sampler
    Type_image *get_image(
        Type_scalar   *base_tp,
        Texture_shape shape);

    /// If a given type has an unsigned variant, return it.
    ///
    /// \param type  the type that should be converted to unsigned
    ///
    /// \return the corresponding unsigned type or NULL if such type does not exists
    Type *to_unsigned_type(Type *type);

    /// Return the symbol table of this type factory.
    Symbol_table &get_symbol_table() { return m_symtab; }

    /// Get the size of a GLSL type in bytes.
    size_t get_type_size(Type *type);

    /// Get the alignment of a GLSL type in bytes.
    size_t get_type_alignment(Type *type);

public:
    /// Constructs a new type factory.
    ///
    /// \param arena         the memory arena used to allocate new types
    /// \param sym_tab       the symbol table for symbols inside types
    explicit Type_factory(
        Memory_arena  &arena,
        Symbol_table  &sym_tab);

private:
    /// The builder for types.
    Arena_builder m_builder;

    /// The symbol table used to create new symbols for types.
    Symbol_table &m_symtab;

    /// Hashtable of cached types.
    typedef Arena_hash_map<
        Type_cache_key,
        Type *,
        Type_cache_key::Hash,
        Type_cache_key::Equal>::Type Type_cache;

    /// Cache of composed immutable types that could be reused (alias, array, function types).
    Type_cache m_type_cache;
};

}  // glsl
}  // mdl
}  // mi

#endif // MDL_COMPILER_GLSL_TYPES_H
