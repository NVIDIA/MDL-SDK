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

#ifndef MDL_COMPILER_HLSL_VALUES_H
#define MDL_COMPILER_HLSL_VALUES_H 1

#include <cstddef>

#include <mi/mdl/mdl_iowned.h>
#include "compiler_hlsl_cc_conf.h"
#include "compiler_hlsl_assert.h"
#include "compiler_hlsl_types.h"

namespace mi {
namespace mdl {

class Memory_arena;

namespace hlsl {

class Symbol;
class Value_factory;

/// A HLSL value.
class Value : public Interface_owned
{
public:
    /// The possible kinds of values.
    enum Kind {
        VK_BAD,                 ///< The singleton bad value.
        VK_VOID,                ///< The singleton void value.
        VK_BOOL,                ///< A boolean value.
        VK_INT,                 ///< An integer value.
        VK_UINT,                ///< An unsigned integer value.
        VK_HALF,                ///< A half value.
        VK_FLOAT,               ///< A float value.
        VK_DOUBLE,              ///< A double value.
        VK_MIN12INT,            ///< An min12int value.
        VK_MIN16INT,            ///< An min16int value.
        VK_MIN16UINT,           ///< An min16uint value.
        VK_VECTOR,              ///< A vector value.
        VK_MATRIX,              ///< A matrix value.
        VK_ARRAY,               ///< An array value.
        VK_STRUCT,              ///< A struct value.
    };

    /// Get the kind of value.
    virtual Kind get_kind() = 0;

    /// Get the type of this value.
    virtual Type *get_type();

    /// Negate this value.
    ///
    /// \param factory  a factory to create new values.
    ///
    /// \return Value_bad if not supported
    virtual Value *minus(Value_factory &factory);

    /// Bitwise negate this value.
    ///
    /// \param factory  a factory to create new values.
    ///
    /// \return Value_bad if not supported
    virtual Value *bitwise_not(Value_factory &factory);

    /// Logically negate this value.
    ///
    /// \param factory  a factory to create new values.
    ///
    /// \return Value_bad if not supported
    virtual Value *logical_not(Value_factory &factory);

    /// Extract from a compound.
    ///
    /// \param factory  a factory to create new values.
    /// \param index    the compound index
    ///
    /// \return Value_bad if not supported
    virtual Value *extract(Value_factory &factory, size_t index);

    /// Multiply.
    ///
    /// \param factory  a factory to create new values.
    /// \param rhs      the right hand operand
    ///
    /// \return Value_bad if not supported
    virtual Value *multiply(Value_factory &factory, Value *rhs);

    /// Divide.
    ///
    /// \param factory  a factory to create new values.
    /// \param rhs      the right hand operand
    ///
    /// \return Value_bad if not supported
    virtual Value *divide(Value_factory &factory, Value *rhs);

    /// Modulo.
    ///
    /// \param factory  a factory to create new values.
    /// \param rhs      the right hand operand
    ///
    /// \return Value_bad if not supported
    virtual Value *modulo(Value_factory &factory, Value *rhs);

    /// Add.
    ///
    /// \param factory  a factory to create new values.
    /// \param rhs      the right hand operand
    ///
    /// \return Value_bad if not supported
    virtual Value *add(Value_factory &factory, Value *rhs);

    /// Subtract.
    ///
    /// \param factory  a factory to create new values.
    /// \param rhs      the right hand operand
    ///
    /// \return Value_bad if not supported
    virtual Value *sub(Value_factory &factory, Value *rhs);

    /// Shift left.
    ///
    /// \param factory  a factory to create new values.
    /// \param rhs      the right hand operand
    ///
    /// \return Value_bad if not supported
    virtual Value *shl(Value_factory &factory, Value *rhs);

    /// Shift right.
    ///
    /// \param factory  a factory to create new values.
    /// \param rhs      the right hand operand
    ///
    /// \return Value_bad if not supported
    virtual Value *shr(Value_factory &factory, Value *rhs);

    /// Xor.
    ///
    /// \param factory  a factory to create new values.
    /// \param rhs      the right hand operand
    ///
    /// \return Value_bad if not supported
    virtual Value *bitwise_xor(Value_factory &factory, Value *rhs);

    /// Bitwise Or.
    ///
    /// \param factory  a factory to create new values.
    /// \param rhs      the right hand operand
    ///
    /// \return Value_bad if not supported
    virtual Value *bitwise_or(Value_factory &factory, Value *rhs);

    /// BitWise And.
    ///
    /// \param factory  a factory to create new values.
    /// \param rhs      the right hand operand
    ///
    /// \return Value_bad if not supported
    virtual Value *bitwise_and(Value_factory &factory, Value *rhs);

    /// Logical Or.
    ///
    /// \param factory  a factory to create new values.
    /// \param rhs      the right hand operand
    ///
    /// \return Value_bad if not supported
    virtual Value *logical_or(Value_factory &factory, Value *rhs);

    /// Logical And.
    ///
    /// \param factory  a factory to create new values.
    /// \param rhs      the right hand operand
    ///
    /// \return Value_bad if not supported
    virtual Value *logical_and(Value_factory &factory, Value *rhs);

    /// Logical Xor.
    ///
    /// \param factory  a factory to create new values.
    /// \param rhs      the right hand operand
    ///
    /// \return Value_bad if not supported
    virtual Value *logical_xor(Value_factory &factory, Value *rhs);

    /// Possible compare results (relations)
    enum Compare_results {
        CR_EQ  = 0x01,                  ///< equal
        CR_LT  = 0x02,                  ///< less than
        CR_LE  = CR_LT | CR_EQ,         ///< less or equal
        CR_GT  = 0x04,                  ///< greater than
        CR_GE  = CR_GT | CR_EQ,         ///< greater or equal
        CR_NE  = CR_GT | CR_LT,         ///< not equal
        CR_O   = CR_LT | CR_EQ | CR_GT, ///< ordered
        CR_UO  = 0x08,                  ///< unordered
        CR_UEQ = CR_UO | CR_EQ,         ///< unordered or equal
        CR_ULT = CR_UO | CR_LT,         ///< unordered or less than
        CR_ULE = CR_UO | CR_LE,         ///< unordered or less or equal
        CR_UGT = CR_UO | CR_GT,         ///< unordered or greater than
        CR_UGE = CR_UO | CR_GE,         ///< unordered or greater or equal
        CR_UNE = CR_UO | CR_NE,         ///< unordered or not equal
        CR_BAD = 0x10                   ///< unknown
    };

    /// Get the inverse compare result, i.e. R^-1.
    ///
    /// \param r  the compare result
    static Compare_results inverse(Compare_results r);

    /// Compare.
    ///
    /// \param rhs      the right hand operand
    ///
    /// \return compare results
    virtual Compare_results compare(Value *rhs);

    /// Convert a value.
    ///
    /// \param factory  a factory to create new values.
    /// \param dst_tp   the destination type
    ///
    /// \return Value_bad if conversion is not supported
    virtual Value *convert(Value_factory &factory, Type *dst_tp);

    /// Returns true if the value is the ZERO (i.e. additive neutral).
    virtual bool is_zero();

    /// Returns true if the value is the ONE (i.e. multiplicative neutral).
    ///
    /// \note: returns NEVER true for matrix types, because there are two neutrals.
    virtual bool is_one();

    /// Returns true if all components of this value are ONE.
    virtual bool is_all_one();

    /// Returns true if this value is finite (i.e. neither Inf nor NaN).
    virtual bool is_finite();

protected:
    /// Constructor.
    explicit Value(Type *type);

protected:
    /// The type of this value
    Type *m_type;
};

/// A value of type error.
class Value_bad : public Value
{
    typedef Value Base;
    friend class mi::mdl::Arena_builder;
public:
    /// The kind of this subclass.
    static const Kind s_kind = VK_BAD;

    /// Get the kind of value.
    Kind get_kind() HLSL_FINAL;

    /// Get the type of this value.
    Type_error *get_type() HLSL_FINAL;

    /// Returns true if this value is finite (i.e. neither Inf nor NaN).
    bool is_finite() HLSL_FINAL;

private:
    /// Constructor.
    explicit Value_bad(Type_error *type);
};

/// A value of type void.
class Value_void : public Value
{
    typedef Value Base;
    friend class mi::mdl::Arena_builder;
public:
    /// The kind of this subclass.
    static const Kind s_kind = VK_VOID;

    /// Get the kind of value.
    Kind get_kind() HLSL_FINAL;

    /// Get the type of this value.
    Type_void *get_type() HLSL_FINAL;

    /// Returns true if this value is finite (i.e. neither Inf nor NaN).
    bool is_finite() HLSL_FINAL;

private:
    /// Constructor.
    explicit Value_void(Type_void *type);
};

/// An scalar value.
class Value_scalar : public Value
{
    typedef Value Base;
public:
    /// Get the type of this value.
    Type_scalar *get_type() HLSL_OVERRIDE;

    /// Convert a value.
    Value *convert(Value_factory &factory, Type *dst_tp) HLSL_OVERRIDE;

protected:
    /// Constructor.
    explicit Value_scalar(Type_scalar *type);
};

/// A value of type boolean.
class Value_bool : public Value_scalar
{
    typedef Value_scalar Base;
    friend class mi::mdl::Arena_builder;
public:
    /// The kind of this subclass.
    static const Kind s_kind = VK_BOOL;

    /// Get the kind of value.
    Kind get_kind() HLSL_FINAL;

    /// Get the type of this value.
    Type_bool *get_type() HLSL_FINAL;

    /// Logically negate this value.
    Value *logical_not(Value_factory &factory) HLSL_FINAL;

    /// Logical Or.
    Value *logical_or(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Logical And.
    Value *logical_and(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Logical Xor.
    Value *logical_xor(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Compare.
    Compare_results compare(Value *rhs) HLSL_FINAL;

    /// Convert an scalar value.
    Value *convert(Value_factory &factory, Type *dst_tp) HLSL_FINAL;
        
    /// Returns true if the value is the ZERO (i.e. additive neutral).
    bool is_zero() HLSL_FINAL;

    /// Returns true if the value is the ONE (i.e. multiplicative neutral).
    bool is_one() HLSL_FINAL;

    /// Returns true if all components of this value are ONE.
    bool is_all_one() HLSL_FINAL;

    /// Returns true if this value is finite (i.e. neither Inf nor NaN).
    bool is_finite() HLSL_FINAL;

    /// Get the value.
    bool get_value() { return m_value; }

private:
    /// Constructor.
    explicit Value_bool(Type_bool *type, bool value);

private:
    /// The value.
    bool m_value;
};

// helper trait
template<size_t S>
struct Bit_width {
};

template<>
struct Bit_width<12> {
    enum {
        S_KIND = Value::VK_MIN12INT,
        U_KIND = Value::VK_BAD,
    };

    typedef int16_t  S_TYPE;
    typedef uint16_t U_TYPE;

    typedef Type_min12int  TYPE_INT;
    typedef Type_error     TYPE_UINT;
};

template<>
struct Bit_width<16> {
    enum {
        S_KIND = Value::VK_MIN16INT,
        U_KIND = Value::VK_MIN16UINT,
    };

    typedef int16_t  S_TYPE;
    typedef uint16_t U_TYPE;

    typedef Type_min16int  TYPE_INT;
    typedef Type_min16uint TYPE_UINT;
};

template<>
struct Bit_width<32> {
    enum {
        S_KIND = Value::VK_INT,
        U_KIND = Value::VK_UINT,
    };

    typedef int32_t  S_TYPE;
    typedef uint32_t U_TYPE;

    typedef Type_int  TYPE_INT;
    typedef Type_uint TYPE_UINT;
};

/// An two-complement based value, base class for Value_int and Value_uint and N bit variants.
template<size_t S>
class Value_two_complement : public Value_scalar
{
    typedef Value_scalar Base;
    typedef Value_two_complement<S> THIS_TYPE;

protected:
    typedef typename Bit_width<S>::S_TYPE S_TYPE;
    typedef typename Bit_width<S>::U_TYPE U_TYPE;

    typedef typename Bit_width<S>::TYPE_INT  TYPE_INT;
    typedef typename Bit_width<S>::TYPE_UINT TYPE_UINT;

    static const Value::Kind U_KIND = Value::Kind(Bit_width<S>::U_KIND);
    static const Value::Kind S_KIND = Value::Kind(Bit_width<S>::S_KIND);

public:
    /// Get the type of this value.
    Type_two_complement *get_type() HLSL_OVERRIDE;

    /// Negate this value.
    Value *minus(Value_factory &factory) HLSL_FINAL;

    /// Bitwise negate this value.
    Value *bitwise_not(Value_factory &factory) HLSL_FINAL;

    /// Add.
    Value *add(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Subtract.
    Value *sub(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Shift left.
    Value *shl(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Xor.
    Value *bitwise_xor(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Or.
    Value *bitwise_or(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// And.
    Value *bitwise_and(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Convert a value.
    Value *convert(Value_factory &factory, Type *dst_tp) HLSL_FINAL;

    /// Returns true if the value is the ZERO (i.e. additive neutral).
    bool is_zero() HLSL_FINAL;

    /// Returns true if the value is the ONE (i.e. multiplicative neutral).
    bool is_one() HLSL_FINAL;

    /// Returns true if all components of this value are ONE.
    bool is_all_one() HLSL_FINAL;

    /// Returns true if this value is finite (i.e. neither Inf nor NaN).
    bool is_finite() HLSL_FINAL;

    /// Get the value as signed integer.
    S_TYPE get_value_signed() { return S_TYPE(m_value); }

    /// Get the value as unsigned integer.
    U_TYPE get_value_unsigned() { return U_TYPE(m_value); }

protected:
    /// Constructor.
    explicit Value_two_complement(Type_two_complement *type, U_TYPE value);

protected:
    /// The value;
    U_TYPE m_value;
};

/// A value of type integer.
template<size_t S>
class Value_int : public Value_two_complement<S>
{
    typedef Value_two_complement<S> Base;
    friend class mi::mdl::Arena_builder;

    typedef typename Base::TYPE_INT  TYPE_INT;
    typedef typename Base::S_TYPE    S_TYPE;
    typedef typename Base::U_TYPE    U_TYPE;

public:
    /// The kind of this subclass.
    static const Value::Kind s_kind = Base::S_KIND;

    /// Get the kind of value.
    Value::Kind get_kind() HLSL_FINAL;

    /// Get the type of this value.
    TYPE_INT *get_type() HLSL_FINAL;

    /// Multiply.
    Value *multiply(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Integer Divide.
    Value *divide(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Integer Modulo.
    Value *modulo(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Arithmetic shift right.
    Value *shr(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Compare.
    Value::Compare_results compare(Value *rhs) HLSL_FINAL;

    /// Get the value.
    S_TYPE get_value() { return Base::get_value_signed(); }

private:
    /// Constructor.
    ///
    /// \param type   the type of this value
    /// \param value  the value in host representation
    Value_int(TYPE_INT *type, S_TYPE value);
};

/// A value of type unsigned integer.
template<size_t S>
class Value_uint : public Value_two_complement<S>
{
    typedef Value_two_complement<S> Base;
    friend class mi::mdl::Arena_builder;

    typedef typename Base::TYPE_UINT TYPE_UINT;
    typedef typename Base::S_TYPE    S_TYPE;
    typedef typename Base::U_TYPE    U_TYPE;
public:
    /// The kind of this subclass.
    static const Value::Kind s_kind = Base::U_KIND;

    /// Get the kind of value.
    Value::Kind get_kind() HLSL_FINAL;

    /// Get the type of this value.
    TYPE_UINT *get_type() HLSL_FINAL;

    /// Multiply.
    Value *multiply(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Integer Divide.
    Value *divide(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Integer Modulo.
    Value *modulo(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Arithmetic shift right.
    Value *shr(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Compare.
    Value::Compare_results compare(Value *rhs) HLSL_FINAL;

    /// Get the value.
    U_TYPE get_value() { return Base::get_value_unsigned(); }

private:
    /// Constructor.
    ///
    /// \param type   the type of this value
    /// \param value  the value in host representation
    Value_uint(TYPE_UINT *type, U_TYPE value);
};

typedef Value_int<12>  Value_int_12;
typedef Value_int<16>  Value_int_16;
typedef Value_int<32>  Value_int_32;
typedef Value_uint<16> Value_uint_16;
typedef Value_uint<32> Value_uint_32;

/// An floating point value.
class Value_fp : public Value_scalar
{
    typedef Value_scalar Base;
public:
    /// Check if something is a Plus zero, aka +0.0.
    virtual bool is_plus_zero() = 0;

    /// Check if something is a Minus zero, , aka -0.0.
    virtual bool is_minus_zero() = 0;

protected:
    /// Constructor.
    explicit Value_fp(Type_scalar *type)
    : Base(type)
    {
    }
};


/// A value of type float.
class Value_half : public Value_fp
{
    typedef Value_fp Base;
    friend class mi::mdl::Arena_builder;
public:
    /// The kind of this subclass.
    static const Kind s_kind = VK_HALF;

    /// Get the kind of value.
    Kind get_kind() HLSL_FINAL;

    /// Get the type of this value.
    Type_half *get_type() HLSL_FINAL;

    /// Negate this value.
    Value *minus(Value_factory &factory) HLSL_FINAL;

    /// Multiply.
    Value *multiply(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Divide.
    Value *divide(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Add.
    Value *add(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Subtract.
    Value *sub(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Compare.
    Compare_results compare(Value *rhs) HLSL_FINAL;

    /// Convert a float value.
    Value *convert(Value_factory &factory, Type *dst_tp) HLSL_FINAL;

    /// Returns true if the value is the ZERO (i.e. additive neutral).
    bool is_zero() HLSL_FINAL;

    /// Returns true if the value is the ONE (i.e. multiplicative neutral).
    bool is_one() HLSL_FINAL;

    /// Returns true if all components of this value are ONE.
    bool is_all_one() HLSL_FINAL;

    /// Returns true if this value is finite (i.e. neither Inf nor NaN).
    bool is_finite() HLSL_FINAL;

    /// Get the value.
    float get_value() { return m_value; }

    /// Check if something is a Plus zero, aka +0.0.
    bool is_plus_zero() HLSL_FINAL;

    /// Check if something is a Minus zero, aka -0.0.
    bool is_minus_zero() HLSL_FINAL;

private:
    /// Constructor.
    explicit Value_half(Type_half *type, float value);

private:
    /// The value.
    float m_value;
};

/// A value of type float.
class Value_float : public Value_fp
{
    typedef Value_fp Base;
    friend class mi::mdl::Arena_builder;
public:
    /// The kind of this subclass.
    static const Kind s_kind = VK_FLOAT;

    /// Get the kind of value.
    Kind get_kind() HLSL_FINAL;

    /// Get the type of this value.
    Type_float *get_type() HLSL_FINAL;

    /// Negate this value.
    Value_float *minus(Value_factory &factory) HLSL_FINAL;

    /// Multiply.
    Value *multiply(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Divide.
    Value *divide(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Add.
    Value *add(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Subtract.
    Value *sub(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Compare.
    Compare_results compare(Value *rhs) HLSL_FINAL;

    /// Convert a float value.
    Value *convert(Value_factory &factory, Type *dst_tp) HLSL_FINAL;

    /// Returns true if the value is the ZERO (i.e. additive neutral).
    bool is_zero() HLSL_FINAL;

    /// Returns true if the value is the ONE (i.e. multiplicative neutral).
    bool is_one() HLSL_FINAL;

    /// Returns true if all components of this value are ONE.
    bool is_all_one() HLSL_FINAL;

    /// Returns true if this value is finite (i.e. neither Inf nor NaN).
    bool is_finite() HLSL_FINAL;

    /// Get the value.
    float get_value() { return m_value; }

    // Check if something is a Plus zero, aka +0.0.
    bool is_plus_zero() HLSL_FINAL;

    /// Check if something is a Minus zero, aka -0.0.
    bool is_minus_zero() HLSL_FINAL;

private:
    /// Constructor.
    explicit Value_float(Type_float *type, float value);

private:
    /// The value.
    float m_value;
};

/// A value of type double.
class Value_double : public Value_fp
{
    typedef Value_fp Base;
    friend class mi::mdl::Arena_builder;
public:
    /// The kind of this subclass.
    static const Kind s_kind = VK_DOUBLE;

    /// Get the kind of value.
    Kind get_kind() HLSL_FINAL;

    /// Get the type of this value.
    Type_double *get_type() HLSL_FINAL;

    /// Negate this value.
    Value_double *minus(Value_factory &factory) HLSL_FINAL;

    /// Multiply.
    Value *multiply(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Divide.
    Value *divide(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Add.
    Value *add(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Subtract.
    Value *sub(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Compare.
    Value::Compare_results compare(Value *rhs) HLSL_FINAL;

    /// Convert a double value.
    Value *convert(Value_factory &factory, Type *dst_tp) HLSL_FINAL;

    /// Returns true if the value is the ZERO (i.e. additive neutral).
    bool is_zero() HLSL_FINAL;

    /// Returns true if the value is the ONE (i.e. multiplicative neutral).
    bool is_one() HLSL_FINAL;

    /// Returns true if all components of this value are ONE.
    bool is_all_one() HLSL_FINAL;

    /// Returns true if this value is finite (i.e. neither Inf nor NaN).
    bool is_finite() HLSL_FINAL;

    /// Get the value.
    double get_value() { return m_value; }

    // Check if something is a Plus zero, aka +0.0.
    bool is_plus_zero() HLSL_FINAL;

    /// Check if something is a Minus zero, aka -0.0.
    bool is_minus_zero() HLSL_FINAL;

private:
    /// Constructor.
    explicit Value_double(Type_double *type, double value);

private:
    /// The value.
    double m_value;
};

/// A compound value.
class Value_compound : public Value
{
    typedef Value Base;
public:
    /// Get the type of this value.
    Type_compound *get_type() HLSL_OVERRIDE;

    /// Get the number of components in this compound value.
    size_t get_component_count() { return m_values.size(); }

    /// Extract from a compound.
    Value *extract(Value_factory &factory, size_t index) HLSL_OVERRIDE;

    /// Returns true if this value (i.e. all of its sub-values) is finite.
    bool is_finite() HLSL_OVERRIDE;

    /// Get the value at index.
    ///
    /// \param index  the index
    Value *get_value(size_t index);

protected:
    /// Constructor.
    explicit Value_compound(
        Memory_arena             *arena,
        Type_compound            *type,
        Array_ref<Value *> const &values);

protected:
    /// The compound values.
    Arena_VLA<Value *> m_values;
};

/// A value of type vector.
class Value_vector : public Value_compound
{
    typedef Value_compound Base;
    friend class mi::mdl::Arena_builder;
public:
    /// The kind of this subclass.
    static const Kind s_kind = VK_VECTOR;

    /// Get the kind of value.
    Kind get_kind() HLSL_FINAL;

    /// Get the type of this value.
    Type_vector *get_type() HLSL_FINAL;

    /// Negate this value.
    Value *minus(Value_factory &factory) HLSL_FINAL;

    /// Bitwise negate this value.
    Value *bitwise_not(Value_factory &factory) HLSL_FINAL;

    /// Logically negate this value.
    Value *logical_not(Value_factory &factory) HLSL_FINAL;

    /// Multiply.
    Value *multiply(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Divide.
    Value *divide(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Modulo.
    Value *modulo(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Add.
    Value *add(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Subtract.
    Value *sub(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Shift left.
    Value *shl(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Shift right.
    Value *shr(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Xor.
    Value *bitwise_xor(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Or.
    Value *bitwise_or(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// And.
    Value *bitwise_and(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Logical Or.
    Value *logical_or(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Logical And.
    Value *logical_and(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Compare.
    Value::Compare_results compare(Value *rhs) HLSL_FINAL;

    /// Convert a vector value.
    Value *convert(Value_factory &factory, Type *dst_tp) HLSL_FINAL;

    /// Returns true if the value is the ZERO (i.e. additive neutral).
    bool is_zero() HLSL_FINAL;

    /// Returns true if the value is the ONE (i.e. multiplicative neutral).
    bool is_one() HLSL_FINAL;

    /// Returns true if all components of this value are ALL ONE.
    bool is_all_one() HLSL_FINAL;

private:
    /// Constructor.
    explicit Value_vector(
        Memory_arena           *arena,
        Type_vector            *type,
        Array_ref<Value_scalar *> const &values);
};

/// A value of type matrix.
class Value_matrix : public Value_compound
{
    typedef Value_compound Base;
    friend class mi::mdl::Arena_builder;
public:
    /// The kind of this subclass.
    static const Kind s_kind = VK_MATRIX;

    /// Get the kind of value.
    Kind get_kind() HLSL_FINAL;

    /// Get the type of this value.
    Type_matrix *get_type() HLSL_FINAL;

    /// Negate this value.
    Value *minus(Value_factory &factory) HLSL_FINAL;

    /// Matrix multiplication.
    Value *multiply(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Divide.
    Value *divide(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Add.
    Value *add(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Subtract.
    Value *sub(Value_factory &factory, Value *rhs) HLSL_FINAL;

    /// Compare.
    Compare_results compare(Value *rhs) HLSL_FINAL;

    /// Convert a matrix value.
    Value *convert(Value_factory &factory, Type *dst_tp) HLSL_FINAL;

    /// Returns true if the value is the ZERO (i.e. additive neutral).
    bool is_zero() HLSL_FINAL;

    /// Returns true if the value is the identity matrix (i.e. multiplicative neutral).
    bool is_one() HLSL_FINAL;
private:
    /// Constructor.
    explicit Value_matrix(
        Memory_arena                    *arena,
        Type_matrix                     *type,
        Array_ref<Value_vector *> const &values);
};

/// A value of type array.
class Value_array : public Value_compound
{
    typedef Value_compound Base;
    friend class mi::mdl::Arena_builder;
public:

    /// The kind of this subclass.
    static const Kind s_kind = VK_ARRAY;

    /// Get the kind of value.
    Kind get_kind() HLSL_FINAL;

    /// Get the type of this value.
    Type_array *get_type() HLSL_FINAL;

private:
    /// Constructor.
    explicit Value_array(
        Memory_arena             *arena,
        Type_array               *type,
        Array_ref<Value *> const &values);
};

/// A value of type struct.
class Value_struct : public Value_compound
{
    typedef Value_compound Base;
    friend class mi::mdl::Arena_builder;
public:
    /// The kind of this subclass.
    static const Kind s_kind = VK_STRUCT;

    /// Get the kind of value.
    Kind get_kind() HLSL_FINAL;

    /// Get the type of this value.
    Type_struct *get_type() HLSL_FINAL;

    /// Get a field.
    /// \param name     The name of the field.
    /// \returns        The value of the field.
    Value *get_field(Symbol *name);

    /// Get a field.
    /// \param name     The name of the field.
    /// \returns        The value of the field.
    Value *get_field(char const *name);

private:
    /// Constructor.
    explicit Value_struct(
        Memory_arena             *arena,
        Type_struct              *type,
        Array_ref<Value *> const &values);
};


/// Cast to subtype or return NULL if types do not match.
template<typename T>
T *as(Value *value) {
    return (value->get_kind() == T::s_kind) ? static_cast<T *>(value) : 0;
}

/// Cast to Value_scalar or return null if types do not match.
template<>
inline Value_scalar *as<Value_scalar>(Value *value) {
    switch(value->get_kind()) {
    case Value::VK_BOOL:
    case Value::VK_INT:
    case Value::VK_UINT:
    case Value::VK_HALF:
    case Value::VK_FLOAT:
    case Value::VK_DOUBLE:
    case Value::VK_MIN12INT:
    case Value::VK_MIN16INT:
    case Value::VK_MIN16UINT:
        return static_cast<Value_scalar *>(value);
    default:
        return NULL;
    }
}

/// Cast to Value_fp or return null if types do not match.
template<>
inline Value_fp *as<Value_fp>(Value *value) {
    switch (value->get_kind()) {
    case Value::VK_HALF:
    case Value::VK_FLOAT:
    case Value::VK_DOUBLE:
        return static_cast<Value_fp *>(value);
    default:
        return NULL;
    }
}

/// Cast to Value_two_complement<8> or return NULL if types do not match.
template<>
inline Value_two_complement<12> *as<Value_two_complement<12> >(Value *value) {
    switch (value->get_kind()) {
    case Value::VK_MIN12INT:
        return static_cast<Value_two_complement<12> *>(value);
    default:
        return NULL;
    }
}

/// Cast to Value_two_complement<16> or return NULL if types do not match.
template<>
inline Value_two_complement<16> *as<Value_two_complement<16> >(Value *value) {
    switch (value->get_kind()) {
    case Value::VK_MIN16INT:
    case Value::VK_MIN16UINT:
        return static_cast<Value_two_complement<16> *>(value);
    default:
        return NULL;
    }
}

/// Cast to Value_two_complement<32> or return NULL if types do not match.
template<>
inline Value_two_complement<32> *as<Value_two_complement<32> >(Value *value) {
    switch (value->get_kind()) {
    case Value::VK_INT:
    case Value::VK_UINT:
        return static_cast<Value_two_complement<32> *>(value);
    default:
        return NULL;
    }
}

/// Cast to Value_compound or return NULL if types do not match.
template<>
inline Value_compound *as<Value_compound>(Value *value) {
    switch(value->get_kind()) {
    case Value::VK_VECTOR:
    case Value::VK_MATRIX:
    case Value::VK_ARRAY:
    case Value::VK_STRUCT:
        return static_cast<Value_compound *>(value);
    default:
        return 0;
    }
}

/// Check if a value is of a certain type.
template<typename T>
bool is(Value *value) {
    return as<T>(value) != NULL;
}

/// A static_cast with check in debug mode
template <typename T>
inline T *cast(Value *arg) {
    HLSL_ASSERT(arg == NULL || is<T>(arg));
    return static_cast<T *>(arg);
}

/// The factory for creating values.
class Value_factory : public Interface_owned
{
    typedef Interface_owned Base;

    struct Value_hash {
        size_t operator()(Value *value) const;
    };

    struct Value_equal {
        bool operator()(Value *a, Value *b) const;
    };

    typedef hash_set<Value *, Value_hash, Value_equal>::Type Value_table;

public:
    /// Get the (singleton) bad value.
    Value_bad *get_bad();

    /// Get the (singleton) void value.
    Value_void *get_void();

    /// Get a value of type boolean.
    ///
    /// \param value The value of the boolean.
    Value_bool *get_bool(bool value);

    /// Get a value of type 32bit integer.
    ///
    /// \param value The value of the integer.
    Value_int_32 *get_int32(int32_t value);

    /// Get a value of type 32bit unsigned int.
    ///
    /// \param value The value of the unsigned integer.
    Value_uint_32 *get_uint32(uint32_t value);

    /// Get a value of type minimum 12bit integer.
    ///
    /// \param value The value of the integer.
    Value_int_12 *get_int12(int16_t value);

    /// Get a value of type minimum 16bit integer.
    ///
    /// \param value The value of the integer.
    Value_int_16 *get_int16(int16_t value);

    /// Get a value of type minimum 16bit unsigned int.
    ///
    /// \param value The value of the unsigned integer.
    Value_uint_16 *get_uint16(uint16_t value);

    /// Get a value of type half.
    ///
    /// \param value The value of the float.
    Value_half *get_half(float value);

    /// Get a value of type float.
    ///
    /// \param value The value of the float.
    Value_float *get_float(float value);

    /// Get a value of type double.
    ///
    /// \param value The value of the double.
    Value_double *get_double(double value);

    /// Get a value of type vector.
    ///
    /// \param type    The type of the vector.
    /// \param values  The values for the elements of the vector.
    Value_vector *get_vector(
        Type_vector                     *type,
        Array_ref<Value_scalar *> const &values);

    /// Get a value of type matrix.
    ///
    /// \param type    The type of the matrix.
    /// \param values  The values for the elements of the matrix columns.
    Value_matrix *get_matrix(
        Type_matrix                     *type,
        Array_ref<Value_vector *> const &values);

    /// Get a value of type array.
    ///
    /// \param type    The type of the array.
    /// \param values  The values for the elements of the array.
    Value_array *get_array(
        Type_array               *type,
        Array_ref<Value *> const &values);

    /// Get a value of type struct.
    ///
    /// \param type    The type of the struct.
    /// \param values  The values for the fields of the struct.
    Value_struct *get_struct(
        Type_struct              *type,
        Array_ref<Value *> const &values);

    /// Get a two-complement value.
    ///
    /// \param kind   The value kind of this two complement value
    /// \param value  The value of the integer.
    Value *get_two_complement(
        Value::Kind kind,
        uint32_t    value);

    /// Get an additive neutral zero if supported for the given type.
    ///
    /// \param type   The type of the constant to create
    /// \return A zero constant or Value_bad if type does not support addition.
    Value *get_zero(Type *type);

    /// Get a zero initializer if supported for the given type.
    ///
    /// \param type   The type of the constant to create
    /// \return A zero constant or Value_bad if type does not support zero initializers.
    Value *get_zero_initializer(Type *type);

    /// Return the type factory of this value factory.
    ///
    /// Note: the returned type factory can create built-in types only.
    Type_factory &get_type_factory();

    /// Dump all values owned by this Value table.
    void dump() const;

public:
    /// Constructor.
    ///
    /// \param arena  a memory arena to allocate from
    /// \param f      the type factory to be used
    Value_factory(Memory_arena &arena, Type_factory &tf);

private:
    /// The builder for values.
    Arena_builder m_builder;

    /// A type factory, use to get the scalar types.
    Type_factory & m_tf;

    /// The value table.
    Value_table m_vt;

    /// The bad value.
    Value_bad *const m_bad_value;

    /// The void value.
    Value_void *const m_void_value;

    /// The true value.
    Value_bool *const m_true_value;

    /// The false value.
    Value_bool *const m_false_value;
};

}  // hlsl
}  // mdl
}  // mi

#endif
