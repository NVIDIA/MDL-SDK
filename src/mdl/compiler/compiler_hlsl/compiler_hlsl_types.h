/******************************************************************************
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_COMPILER_HLSL_TYPES_H
#define MDL_COMPILER_HLSL_TYPES_H 1

#include <mi/mdl/mdl_iowned.h>

#include "mdl/compiler/compilercore/compilercore_memory_arena.h"
#include "mdl/compiler/compilercore/compilercore_array_ref.h"

#include "compiler_hlsl_assert.h"
#include "compiler_hlsl_cc_conf.h"
#include "compiler_hlsl_symbols.h"

namespace mi {
namespace mdl {
namespace hlsl {

class Symbol;

/// The HLSL type.
class Type {
public:
    enum Kind {
        TK_ALIAS,       ///< An alias for another type, aka typedef.

        TK_VOID,        ///< The void type.
        TK_BOOL,        ///< The boolean type.
        TK_INT,         ///< The 32bit signed integer type.
        TK_UINT,        ///< The 32bit unsigned integer type.
        TK_HALF,        ///< The half-precision floating-point scalar.
        TK_FLOAT,       ///< The single-precision floating-point scalar.
        TK_DOUBLE,      ///< The single-precision floating-point scalar.

        TK_MIN12INT,    ///< minimum 12 bit int
        TK_MIN16INT,    ///< minimum 16 bit int
        TK_MIN16UINT,   ///< minimum 16 bit unsigned int
        TK_MIN10FLOAT,  ///< minimum 10bit float
        TK_MIN16FLOAT,  ///< minimum 16bit float

        TK_VECTOR,      ///< A vector type.
        TK_MATRIX,      ///< A matrix type.
        TK_ARRAY,       ///< An array type.
        TK_STRUCT,      ///< A struct type.
        TK_FUNCTION,    ///< A function type.

        TK_TEXTURE,     ///< A texture type.

        TK_ERROR,       ///< The error type.
    };


    /// The possible kinds of type modifiers.
    enum Modifier {
        MK_NONE      = 0,         ///< none
        MK_CONST     = 1 << 0,    ///< constant
        MK_COL_MAJOR = 1 << 1,    ///< column major (matrix types only)
        MK_ROW_MAJOR = 1 << 2,    ///< row major (matrix types only)
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

protected:
    /// Constructor.
    ///
    /// \param sym  the name symbol of this type
    explicit Type(Symbol *sym);

private:
    /// The name symbol of the type.
    Symbol *m_sym;
};

/// HLSL alias types. We use this only to add modifiers.
class Type_alias : public Type
{
    typedef Type Base;
    friend class mi::mdl::Arena_builder;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_ALIAS;

    /// Get the type kind.
    Kind get_kind() HLSL_FINAL;

    /// Access to the base type.
    Type *skip_type_alias() HLSL_FINAL;

    /// Get the modifier set of this type.
    Modifiers get_type_modifiers() HLSL_FINAL;

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
    Kind get_kind() HLSL_FINAL;

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
    /// \param sym  the name symbol of this atomic type
    explicit Type_scalar(Symbol *sym);
};

/// The HLSL void type.
class Type_void : public Type_scalar
{
    typedef Type_scalar Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_VOID;

    /// Get the type kind.
    Kind get_kind() HLSL_FINAL;

public:
    /// Constructor.
    explicit Type_void();
};

/// The HLSL boolean type.
class Type_bool : public Type_scalar
{
    typedef Type_scalar Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_BOOL;

    /// Get the type kind.
    Kind get_kind() HLSL_FINAL;

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

protected:
    /// Constructor.
    ///
    /// \param sym      the symbol of this type
    /// \param bitsize  the bit-size of this type
    explicit Type_unsigned_int(
        Symbol *sym,
        size_t bitsize);
};

/// The HLSL signed 32bit integer type.
class Type_int : public Type_signed_int
{
    typedef Type_signed_int Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_INT;

    /// Get the type kind.
    Kind get_kind() HLSL_FINAL;

public:
    /// Constructor.
    explicit Type_int();
};

/// The HLSL unsigned 32bit integer type.
class Type_uint : public Type_unsigned_int
{
    typedef Type_unsigned_int Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_UINT;

    /// Get the type kind.
    Kind get_kind() HLSL_FINAL;

public:
    /// Constructor.
    explicit Type_uint();
};

/// The HLSL signed minimum 12bit integer type.
class Type_min12int : public Type_signed_int
{
    typedef Type_signed_int Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_MIN12INT;

    /// Get the type kind.
    Kind get_kind() HLSL_FINAL;

public:
    /// Constructor.
    explicit Type_min12int();
};

/// The HLSL signed minimum 16bit integer type.
class Type_min16int : public Type_signed_int
{
    typedef Type_signed_int Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_MIN16INT;

    /// Get the type kind.
    Kind get_kind() HLSL_FINAL;

public:
    /// Constructor.
    explicit Type_min16int();
};

/// The HLSL unsigned minimum 16bit integer type.
class Type_min16uint : public Type_signed_int
{
    typedef Type_signed_int Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_MIN16UINT;

    /// Get the type kind.
    Kind get_kind() HLSL_FINAL;

public:
    /// Constructor.
    explicit Type_min16uint();
};


/// The HLSL unsigned 32bit dword type, equivalent to to uint.
typedef Type_uint  Type_dword;


/// The HLSL half type.
class Type_half : public Type_scalar
{
    typedef Type_scalar Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_HALF;

    /// Get the type kind.
    Kind get_kind() HLSL_FINAL;

public:
    /// Constructor.
    explicit Type_half();
};

/// The HLSL float type.
class Type_float : public Type_scalar
{
    typedef Type_scalar Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_FLOAT;

    /// Get the type kind.
    Kind get_kind() HLSL_FINAL;

public:
    /// Constructor.
    explicit Type_float();
};

/// The HLSL double type.  Available since version 4.00.
class Type_double : public Type_scalar
{
    typedef Type_scalar Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_DOUBLE;

    /// Get the type kind.
    Kind get_kind() HLSL_FINAL;

public:
    /// Constructor.
    explicit Type_double();
};

/// The HLSL minimum 10 bit float type.
class Type_min10float : public Type_scalar
{
    typedef Type_scalar Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_MIN10FLOAT;

    /// Get the type kind.
    Kind get_kind() HLSL_FINAL;

public:
    /// Constructor.
    explicit Type_min10float();
};

/// The HLSL minimum 16 bit float type.
class Type_min16float : public Type_scalar
{
    typedef Type_scalar Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_MIN16FLOAT;

    /// Get the type kind.
    Kind get_kind() HLSL_FINAL;

public:
    /// Constructor.
    explicit Type_min16float();
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

/// HLSL vector types.
class Type_vector : public Type_compound
{
    typedef Type_compound Base;
public:
    /// The kind of this subclass.
    static const Kind s_kind = TK_VECTOR;

    /// Get the type kind.
    Kind get_kind() HLSL_FINAL;

    /// Get the compound type at index.
    Type *get_compound_type(size_t index) HLSL_FINAL;

    /// Get the number of compound elements.
    size_t get_compound_size() HLSL_FINAL;

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

/// HLSL matrix types.
class Type_matrix : public Type_compound
{
    typedef Type_compound Base;
public:
    /// The kind of this subclass.
    static const Kind s_kind = TK_MATRIX;

    /// Get the type kind.
    Kind get_kind() HLSL_FINAL;

    /// Get the compound type at index.
    Type *get_compound_type(size_t index) HLSL_FINAL;

    /// Get the number of compound elements.
    size_t get_compound_size() HLSL_FINAL;

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

/// HLSL array types.
class Type_array : public Type_compound
{
    typedef Type_compound Base;
    friend class mi::mdl::Arena_builder;
public:
    /// The kind of this subclass.
    static const Kind s_kind = TK_ARRAY;

    /// Get the type kind.
    Kind get_kind() HLSL_FINAL;

    /// Get the compound type at index.
    Type *get_compound_type(size_t index) HLSL_FINAL;

    /// Get the number of compound elements.
    size_t get_compound_size() HLSL_FINAL;

    /// Get the modifier set of this type.
    Modifiers get_type_modifiers() HLSL_FINAL;

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

/// HLSL struct types.
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
    Kind get_kind() HLSL_FINAL;

    /// Get the compound type at index.
    Type *get_compound_type(size_t index) HLSL_FINAL;

    /// Get the number of compound elements.
    size_t get_compound_size() HLSL_FINAL;

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

/// HLSL function type.
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
    Kind get_kind() HLSL_FINAL;

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

/// HLSL texture shapes.
enum Texture_shape {
    SHAPE_UNKNOWN,              ///< Unknown shape
    SHAPE_1D,                   ///< 1D texture
    SHAPE_2D,                   ///< 2D texture
    SHAPE_3D,                   ///< 3D texture
    SHAPE_CUBE,                 ///< cube mapped texture

    SHAPE_1D_ARRAY,             ///< 1D array texture
    SHAPE_2D_ARRAY,             ///< 2D array texture
};

/// A HLSL texture sampler type.
class Type_texture : public Type
{
    typedef Type Base;
    friend class mi::mdl::Arena_builder;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_TEXTURE;

    /// Get the type kind.
    Kind get_kind() HLSL_FINAL;

    /// Get the texture shape.
    Texture_shape get_shape() { return m_shape; }

public:
    /// Constructor.
    ///
    /// \param shape    the texture shape of this sampler
    explicit Type_texture(
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
    case Type::TK_HALF:
    case Type::TK_FLOAT:
    case Type::TK_DOUBLE:
    case Type::TK_MIN12INT:
    case Type::TK_MIN16INT:
    case Type::TK_MIN16UINT:
    case Type::TK_MIN10FLOAT:
    case Type::TK_MIN16FLOAT:
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
    case Type::TK_MIN12INT:
    case Type::TK_MIN16INT:
    case Type::TK_MIN16UINT:
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
    case Type::TK_HALF:
    case Type::TK_FLOAT:
    case Type::TK_DOUBLE:
    case Type::TK_MIN12INT:
    case Type::TK_MIN16INT:
    case Type::TK_MIN16UINT:
    case Type::TK_MIN10FLOAT:
    case Type::TK_MIN16FLOAT:
        return true;
    default:
        return false;
    }
}

/// Check if a type is of type Type_two_complement.
template<>
inline bool is<Type_two_complement>(Type *type) {
    switch (type->get_kind()) {
    case Type::TK_INT:
    case Type::TK_UINT:
    case Type::TK_MIN12INT:
    case Type::TK_MIN16INT:
    case Type::TK_MIN16UINT:
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

/// A static_cast with check in debug mode
template <typename T>
inline T *cast(Type *arg) {
    HLSL_ASSERT(arg == NULL || is<T>(arg));
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
                        t = ((t) >> 3) ^ (t >> 16) ^
                            size_t(KEY_FUNC_TYPE) ^ n_params;

                        for (size_t i = 0; i < n_params; ++i) {
                            Type_function::Parameter *param  = ft->get_parameter(i);
                            Type                     *p_type = param->get_type();

                            t *= 3;
                            t ^= ((char *)p_type - (char *)0);
                        }
                        return t;
                    }
                case KEY_FUNC_KEY:
                    {
                        size_t t = key.type - (Type *)0;
                        t = ((t) >> 3) ^ (t >> 16) ^
                            size_t(KEY_FUNC_TYPE) ^ key.u.func.n_params;

                        Type_function::Parameter const *p = key.u.func.params;
                        for (size_t i = 0; i < key.u.func.n_params; ++i) {
                            t *= 3;
                            t ^= ((char *)p[i].get_type() - (char *)0);
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
                    Type                *rt;
                    Function_type const *sk;

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

                            if (p_type != sk->params[i].get_type())
                                return false;
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
                    HLSL_ASSERT(!"function search key in type cache detected");
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

    /// Get the (singleton) half type instance.
    Type_half *get_half();

    /// Get the (singleton) float type instance.
    Type_float *get_float();

    /// Get the (singleton) double type instance.
    Type_double *get_double();

    /// Get the (singleton) min12int type instance.
    Type_min12int *get_min12int();

    /// Get the (singleton) min16int type instance.
    Type_min16int *get_min16int();

    /// Get the (singleton) min16uint type instance.
    Type_min16uint *get_min16uint();

    /// Get the (singleton) min10float type instance.
    Type_min10float *get_min10float();

    /// Get the (singleton) min16float type instance.
    Type_min16float *get_min16float();

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

    /// Get a struct type instance.
    ///
    /// \param fields  The fields of the struct.
    /// \param sym     The name symbol of the struct.
    Type_struct *get_struct(
        Array_ref<Type_struct::Field> const &fields,
        Symbol                              *sym);

    /// Get an texture type
    ///
    /// \param shape    the texture shape of this texture
    Type_texture *get_texture(
        Texture_shape shape);

    /// Return the symbol table of this type factory.
    Symbol_table &get_symbol_table() { return m_symtab; }

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

}  // hlsl
}  // mdl
}  // mi

#endif // MDL_COMPILER_HLSL_TYPES_H
