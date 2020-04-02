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

#include "pch.h"

#include <mi/math/function.h>
#include <mi/base/handle.h>
#include "mdl/compiler/compilercore/compilercore_streams.h"
#include "mdl/compiler/compilercore/compilercore_tools.h"

#include "compiler_hlsl_assert.h"
#include "compiler_hlsl_half.h"
#include "compiler_hlsl_values.h"
#include "compiler_hlsl_symbols.h"

namespace mi {
namespace mdl {
namespace hlsl {

typedef Array_ref<Value *>        Values_ref;
typedef Array_ref<Value_scalar *> Atomics_ref;
typedef Array_ref<Value_vector *> Vectors_ref;

/// Array reference casts.
template<typename T, typename F>
inline T const &cast(Array_ref<F> const &array)
{
    return array;
}

template<>
inline Values_ref const &cast(Vectors_ref const &array)
{
    // Safe case
    return reinterpret_cast<Values_ref const &>(array);
}

template<>
inline Values_ref const &cast(Atomics_ref const &array)
{
    // Safe case
    return reinterpret_cast<Values_ref const &>(array);
}

// Constructor.
Value::Value(Type *type)
: m_type(type)
{
}

// Get the type of this value.
Type *Value::get_type()
{
    return m_type;
}

// Negate this value.
Value *Value::minus(Value_factory &factory)
{
    return factory.get_bad();
}

// Bitwise negate this value.
Value *Value::bitwise_not(Value_factory &factory)
{
    return factory.get_bad();
}

// Logically negate this value.
Value *Value::logical_not(Value_factory &factory)
{
    return factory.get_bad();
}

// Extract from a compound.
Value *Value::extract(Value_factory &factory, size_t)
{
    return factory.get_bad();
}

// Multiply.
Value *Value::multiply(Value_factory &factory, Value *)
{
    return factory.get_bad();
}

// Divide.
Value *Value::divide(Value_factory &factory, Value *)
{
    return factory.get_bad();
}

// Modulo.
Value *Value::modulo(Value_factory &factory, Value *)
{
    return factory.get_bad();
}

// Add.
Value *Value::add(Value_factory &factory, Value *)
{
    return factory.get_bad();
}

// Subtract.
Value *Value::sub(Value_factory &factory, Value *)
{
    return factory.get_bad();
}

// Shift left.
Value *Value::shl(Value_factory &factory, Value *)
{
    return factory.get_bad();
}

// Shift right.
Value *Value::shr(Value_factory &factory, Value *)
{
    return factory.get_bad();
}

// Bitwise Xor.
Value *Value::bitwise_xor(Value_factory &factory, Value *)
{
    return factory.get_bad();
}

// Bitwise Or.
Value *Value::bitwise_or(Value_factory &factory, Value *)
{
    return factory.get_bad();
}

// Bitwise And.
Value *Value::bitwise_and(Value_factory &factory, Value *)
{
    return factory.get_bad();
}

// Logical Or.
Value *Value::logical_or(Value_factory &factory, Value *)
{
    return factory.get_bad();
}

// Logical And.
Value *Value::logical_and(Value_factory &factory, Value *)
{
    return factory.get_bad();
}

// Logical Xor.
Value *Value::logical_xor(Value_factory &factory, Value *)
{
    return factory.get_bad();
}

// Compare.
Value::Compare_results Value::compare(Value *)
{
    return Value::CR_BAD;
}

// Convert a value.
Value *Value::convert(Value_factory &factory, Type *dst_type)
{
    dst_type = dst_type->skip_type_alias();
    if (dst_type == m_type) {
        // no conversion
        return this;
    }
    return factory.get_bad();
}

// Returns true if the value is the ZERO (i.e. additive neutral).
bool Value::is_zero()
{
    return false;
}

// Returns true if the value is the ONE (i.e. multiplicative neutral).
bool Value::is_one()
{
    return false;
}

// Returns true if all components of this value are ONE.
bool Value::is_all_one()
{
    return false;
}

// Returns true if this value is finite (i.e. neither Inf nor NaN).
bool Value::is_finite()
{
    return false;
}

// Get the inverse compare result, i.e. R^-1.
Value::Compare_results Value::inverse(Compare_results r)
{
    unsigned x = r & CR_NE;
    if (x == CR_LT || x == CR_GT) {
        // contains either < or >, turn around
        return Compare_results(r ^ CR_NE);
    }
    return r;
}

// ---------------------------------- Value_bad ----------------------------------

// Constructor.
Value_bad::Value_bad(Type_error *type)
: Base(type)
{
}

// Get the kind of value.
Value::Kind Value_bad::get_kind()
{
    return s_kind;
}

// Get the type of this value.
Type_error *Value_bad::get_type()
{
    return cast<Type_error>(m_type);
}

// Returns true if this value is finite (i.e. neither Inf nor NaN).
bool Value_bad::is_finite()
{
    return false;
}

// ---------------------------------- Value_void ----------------------------------

// Constructor.
Value_void::Value_void(Type_void *type)
: Base(type)
{
}

// Get the kind of value.
Value::Kind Value_void::get_kind()
{
    return s_kind;
}

// Get the type of this value.
Type_void *Value_void::get_type()
{
    return cast<Type_void>(m_type);
}

// Returns true if this value is finite (i.e. neither Inf nor NaN).
bool Value_void::is_finite()
{
    return false;
}

// ---------------------------------- Value_scalar ----------------------------------

// Constructor.
Value_scalar::Value_scalar(Type_scalar *type)
: Base(type)
{
}

// Get the type of this value.
Type_scalar *Value_scalar::get_type()
{
    return cast<Type_scalar>(m_type);
}

// Convert an scalar value.
Value *Value_scalar::convert(Value_factory &factory, Type *dst_tp)
{
    dst_tp = dst_tp->skip_type_alias();
    if (dst_tp->get_kind() == Type::TK_VECTOR) {
        // convert from scalar to vector
        Type_vector *v_type = cast<Type_vector>(dst_tp);
        Value *v = convert(factory, v_type->get_element_type());
        if (is<Value_bad>(v))
            return v;
        Value_scalar *a_v = cast<Value_scalar>(v);
        Value_scalar *args[4] = { a_v, a_v, a_v, a_v };
        return factory.get_vector(v_type, Atomics_ref(args, v_type->get_size()));
    }
    return Base::convert(factory, dst_tp);
}

// ---------------------------------- Value_bool ----------------------------------

// Constructor.
Value_bool::Value_bool(Type_bool *type, bool value)
: Base(type)
, m_value(value)
{
}

// Get the kind of value.
Value::Kind Value_bool::get_kind()
{
    return s_kind;
}

// Get the type of this value.
Type_bool *Value_bool::get_type()
{
    return cast<Type_bool>(m_type);
}

// Logically negate this value.
Value *Value_bool::logical_not(Value_factory &factory)
{
    return factory.get_bool(!m_value);
}

// Logical Or.
Value *Value_bool::logical_or(Value_factory &factory, Value *rhs)
{
    Kind r_kind = rhs->get_kind();
    if (r_kind == s_kind)
        return factory.get_bool(m_value || cast<Value_bool>(rhs)->get_value());
    else if (r_kind == VK_VECTOR)
        return rhs->logical_or(factory, this);
    return Base::logical_or(factory, rhs);
}

// Logical And.
Value *Value_bool::logical_and(Value_factory &factory, Value *rhs)
{
    Kind r_kind = rhs->get_kind();
    if (r_kind == s_kind)
        return factory.get_bool(m_value && cast<Value_bool>(rhs)->get_value());
    else if (r_kind == VK_VECTOR)
        return rhs->logical_and(factory, this);
    return Base::logical_and(factory, rhs);
}

// Logical Xor.
Value *Value_bool::logical_xor(Value_factory &factory, Value *rhs)
{
    Kind r_kind = rhs->get_kind();
    if (r_kind == s_kind)
        return factory.get_bool(m_value != cast<Value_bool>(rhs)->get_value());
    else if (r_kind == VK_VECTOR)
        return rhs->logical_xor(factory, this);
    return Base::logical_or(factory, rhs);
}

// Compare.
Value::Compare_results Value_bool::compare(Value *rhs)
{
    Kind r_kind = rhs->get_kind();
    if (r_kind == s_kind) {
        bool other = cast<Value_bool>(rhs)->get_value();
        if (m_value == other)
            return Value::CR_EQ;
        return Value::CR_NE;
    } else if (r_kind == VK_VECTOR) {
        return inverse(rhs->compare(this));
    }
    return Base::compare(rhs);
}

// Convert an scalar value.
Value *Value_bool::convert(Value_factory &factory, Type *dst_tp)
{
    dst_tp = dst_tp->skip_type_alias();
    switch (dst_tp->get_kind()) {
    case Type::TK_INT:        return factory.get_int32(int32_t(m_value));
    case Type::TK_UINT:       return factory.get_uint32(uint32_t(m_value));
    case Type::TK_HALF:       return factory.get_half(float(m_value));
    case Type::TK_FLOAT:      return factory.get_float(float(m_value));
    case Type::TK_DOUBLE:     return factory.get_double(double(m_value));
    case Type::TK_MIN12INT:   return factory.get_int12(int16_t(m_value));
    case Type::TK_MIN16INT:   return factory.get_int16(int16_t(m_value));
    case Type::TK_MIN16UINT:  return factory.get_uint16(uint32_t(m_value));
    case Type::TK_MIN10FLOAT: return factory.get_half(float(m_value));
    case Type::TK_MIN16FLOAT: return factory.get_half(float(m_value));
    default:
        // no other conversions supported
        break;
    }
    return Base::convert(factory, dst_tp);
}

// Returns true if the value is the ZERO (i.e. additive neutral).
bool Value_bool::is_zero()
{
    // false is the additive neutral
    return m_value == false;
}

// Returns true if the value is the ONE (i.e. multiplicative neutral).
bool Value_bool::is_one()
{
    // true is the multiplicative neutral
    return m_value == true;
}

// Returns true if all components of this value are ONE.
bool Value_bool::is_all_one()
{
    // bool has only ONE component
    return Value_bool::is_one();
}

// Returns true if this value is finite (i.e. neither Inf nor NaN).
bool Value_bool::is_finite()
{
    return true;
}

// ---------------------------------- Value_two_complement ----------------------------------

static uint32_t mask_bits(uint32_t value, unsigned bits)
{
    // kill upper bits
    value = value & (~0u >> (32 - bits));

    if ((value & (1u << (bits - 1))) != 0) {
        // expand sign bit
        value |= ~0u << bits;
    }
    return value;
}

// Constructor.
template<size_t S>
Value_two_complement<S>::Value_two_complement(Type_two_complement *type, U_TYPE value)
: Base(type)
, m_value(((S & (S - 1)) != 0) ? U_TYPE(mask_bits(value, S)) : value)
{
}

// Get the type of this value.
template<size_t S>
Type_two_complement *Value_two_complement<S>::get_type()
{
    return cast<Type_two_complement>(m_type);
}

// Negate this value.
template<size_t S>
Value *Value_two_complement<S>::minus(Value_factory &factory)
{
    return factory.get_two_complement(get_kind(), U_TYPE(-S_TYPE(m_value)));
}

// Bitwise negate this value.
template<size_t S>
Value *Value_two_complement<S>::bitwise_not(Value_factory &factory)
{
    return factory.get_two_complement(get_kind(), U_TYPE(~m_value));
}

// Add.
template<size_t S>
Value *Value_two_complement<S>::add(Value_factory &factory, Value *rhs)
{
    Kind r_kind = rhs->get_kind();
    if (r_kind == S_KIND || r_kind == U_KIND) {
        U_TYPE value = m_value + cast<THIS_TYPE>(rhs)->get_value_unsigned();
        Kind res_kind = r_kind == this->get_kind() ? r_kind : U_KIND;
        return factory.get_two_complement(res_kind, value);
    } else if (r_kind == VK_VECTOR) {
        return rhs->add(factory, this);
    }
    return Base::add(factory, rhs);
}

// Subtract.
template<size_t S>
Value *Value_two_complement<S>::sub(Value_factory &factory, Value *rhs)
{
    Kind r_kind = rhs->get_kind();
    if (r_kind == S_KIND || r_kind == U_KIND) {
        U_TYPE value = m_value - cast<THIS_TYPE>(rhs)->get_value_unsigned();
        Kind res_kind = r_kind == this->get_kind() ? r_kind : U_KIND;
        return factory.get_two_complement(res_kind, value);
    } else if (r_kind == VK_VECTOR) {
        // scalar - vector = (-vector) + scalar
        return rhs->minus(factory)->add(factory, this);
    }
    return Base::sub(factory, rhs);
}

/// Shift left.
template<size_t S>
Value *Value_two_complement<S>::shl(Value_factory &factory, Value *rhs)
{
    Kind r_kind = rhs->get_kind();
    if (r_kind == S_KIND || r_kind == U_KIND) {
        // in HLSL the result type of a shift is always the left type
        U_TYPE value = m_value << cast<THIS_TYPE>(rhs)->get_value_unsigned();
        return factory.get_two_complement(get_kind(), value);
    } else if (r_kind == VK_VECTOR) {
        // scalar by vector element wise shift left
        Value_scalar *values[4];
        Value_vector *o = cast<Value_vector>(rhs);
        Type_vector  *v_type = o->get_type();
        size_t       n = v_type->get_size();

        HLSL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            Value *tmp = this->shl(factory, o->get_value(i));
            if (is<Value_bad>(tmp))
                return tmp;
            values[i] = cast<Value_scalar>(tmp);
        }

        // in HLSL the result type of a shift is always the left type
        Type_vector *res_type = factory.get_type_factory().get_vector(
            get_type(), v_type->get_size());
        return factory.get_vector(res_type, Atomics_ref(values, n));
    }
    return Base::shl(factory, rhs);
}

// Xor.
template<size_t S>
Value *Value_two_complement<S>::bitwise_xor(Value_factory &factory, Value *rhs)
{
    Kind r_kind = rhs->get_kind();
    if (r_kind == S_KIND || r_kind == U_KIND) {
        U_TYPE value = m_value ^ cast<THIS_TYPE>(rhs)->get_value_unsigned();
        Kind res_kind = r_kind == this->get_kind() ? r_kind : U_KIND;
        return factory.get_two_complement(res_kind, value);
    } else if (r_kind == VK_VECTOR) {
        return rhs->bitwise_xor(factory, this);
    }
    return Base::bitwise_xor(factory, rhs);
}

// Or.
template<size_t S>
Value *Value_two_complement<S>::bitwise_or(Value_factory &factory, Value *rhs)
{
    Kind r_kind = rhs->get_kind();
    if (r_kind == S_KIND || r_kind == U_KIND) {
        U_TYPE value = m_value | cast<THIS_TYPE>(rhs)->get_value_unsigned();
        Kind res_kind = r_kind == this->get_kind() ? r_kind : VK_UINT;
        return factory.get_two_complement(res_kind, value);
    } else if (r_kind == VK_VECTOR) {
        return rhs->bitwise_or(factory, this);
    }
    return Base::bitwise_or(factory, rhs);
}

// And.
template<size_t S>
Value *Value_two_complement<S>::bitwise_and(Value_factory &factory, Value *rhs)
{
    Kind r_kind = rhs->get_kind();
    if (r_kind == S_KIND || r_kind == U_KIND) {
        U_TYPE value = m_value & cast<Value_two_complement>(rhs)->get_value_unsigned();
        Kind res_kind = r_kind == this->get_kind() ? r_kind : U_KIND;
        return factory.get_two_complement(res_kind, value);
    } else if (r_kind == VK_VECTOR) {
        return rhs->bitwise_and(factory, this);
    }
    return Base::bitwise_and(factory, rhs);
}

// Convert a value.
template<size_t S>
Value *Value_two_complement<S>::convert(Value_factory &factory, Type *dst_tp)
{
    dst_tp = dst_tp->skip_type_alias();
    switch (dst_tp->get_kind()) {
    case Type::TK_BOOL:       return factory.get_bool(m_value != 0);
    case Type::TK_INT:        return factory.get_int32(int32_t(m_value));
    case Type::TK_UINT:       return factory.get_uint32(uint32_t(m_value));
    case Type::TK_MIN12INT:   return factory.get_int12(int16_t(m_value));
    case Type::TK_MIN16INT:   return factory.get_int16(int16_t(m_value));
    case Type::TK_MIN16UINT:  return factory.get_uint16(uint16_t(m_value));
    case Type::TK_HALF:       return factory.get_half(float(m_value));
    case Type::TK_FLOAT:      return factory.get_float(float(m_value));
    case Type::TK_DOUBLE:     return factory.get_double(double(m_value));
    case Type::TK_MIN10FLOAT: return factory.get_half(float(m_value));
    case Type::TK_MIN16FLOAT: return factory.get_half(float(m_value));
    default:
        // no other conversions supported
        break;
    }
    return Base::convert(factory, dst_tp);
}

// Returns true if the value is the ZERO (i.e. additive neutral).
template<size_t S>
bool Value_two_complement<S>::is_zero()
{
    return m_value == U_TYPE(0u);
}

// Returns true if the value is the ONE (i.e. multiplicative neutral).
template<size_t S>
bool Value_two_complement<S>::is_one()
{
    return m_value == U_TYPE(1u);
}

// Returns true if all components of this value are ONE.
template<size_t S>
bool Value_two_complement<S>::is_all_one()
{
    // treat bits as components
    if ((S & (S - 1)) != 0)
        return m_value == U_TYPE(mask_bits(~0u, S));

    return m_value == ~U_TYPE(0u);
}

// Returns true if this value is finite (i.e. neither Inf nor NaN).
template<size_t S>
bool Value_two_complement<S>::is_finite()
{
    return true;
}

// ---------------------------------- Value_int ----------------------------------

// Constructor.
template<size_t S>
Value_int<S>::Value_int(TYPE_INT *type, S_TYPE value)
: Base(type, U_TYPE(value))
{
}

// Get the kind of value.
template<size_t S>
Value::Kind Value_int<S>::get_kind()
{
    return s_kind;
}

// Get the type of this value.
template<size_t S>
typename Value_int<S>::TYPE_INT *Value_int<S>::get_type()
{
    return cast<TYPE_INT>(this->m_type);
}

// Multiply.
template<size_t S>
Value *Value_int<S>::multiply(Value_factory &factory, Value *rhs)
{
    Value::Kind r_kind = rhs->get_kind();
    if (r_kind == s_kind) {
        return factory.get_two_complement(
            s_kind,
            S_TYPE(S_TYPE(this->m_value) * cast<Value_int<S> >(rhs)->get_value()));
    } else if (r_kind == Base::U_KIND) {
        // unsigned if one is unsigned
        return factory.get_two_complement(
            s_kind,
            U_TYPE(U_TYPE(this->m_value) * cast<Value_uint<S> >(rhs)->get_value()));
    } else if (r_kind == Value::VK_VECTOR) {
        return rhs->multiply(factory, this);
    }
    return Base::multiply(factory, rhs);
}

// Multiply 12 bit.
template<>
Value *Value_int<12>::multiply(Value_factory &factory, Value *rhs)
{
    Value::Kind r_kind = rhs->get_kind();
    if (r_kind == s_kind) {
        return factory.get_two_complement(
            s_kind,
            S_TYPE(S_TYPE(this->m_value) * cast<Value_int<12> >(rhs)->get_value()));
    } else if (r_kind == Value::VK_VECTOR) {
        return rhs->multiply(factory, this);
    }
    return Base::multiply(factory, rhs);
}

// Integer Divide.
template<size_t S>
Value *Value_int<S>::divide(Value_factory &factory, Value *rhs)
{
    Value::Kind r_kind = rhs->get_kind();
    if (r_kind == s_kind) {
        S_TYPE divisor = cast<Value_int<S> >(rhs)->get_value();
        if (divisor != 0)
            return factory.get_two_complement(
                s_kind,
                S_TYPE(S_TYPE(this->m_value) / divisor));
        return factory.get_bad();
    } else if (r_kind == Base::U_KIND) {
        // unsigned if one is unsigned
        U_TYPE divisor = cast<Value_uint<S> >(rhs)->get_value();
        if (divisor != 0)
            return factory.get_two_complement(
                s_kind,
                U_TYPE(U_TYPE(this->m_value) / divisor));
        return factory.get_bad();
    } else if (r_kind == Value::VK_VECTOR) {
        // scalar by vector element wise division
        Value_scalar *values[4];
        Value_vector *o = cast<Value_vector>(rhs);
        Type_vector  *v_type = o->get_type();
        size_t       n = v_type->get_size();

        HLSL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            Value *tmp = this->divide(factory, o->get_value(i));
            if (is<Value_bad>(tmp))
                return tmp;
            values[i] = cast<Value_scalar>(tmp);
        }
        // result is unsigned if right type is unsigned
        return factory.get_vector(v_type, Atomics_ref(values, n));
    }
    return Base::divide(factory, rhs);
}

// Integer Divide 12 bit.
template<>
Value *Value_int<12>::divide(Value_factory &factory, Value *rhs)
{
    Value::Kind r_kind = rhs->get_kind();
    if (r_kind == s_kind) {
        S_TYPE divisor = cast<Value_int<12> >(rhs)->get_value();
        if (divisor != 0)
            return factory.get_two_complement(
                s_kind,
                S_TYPE(S_TYPE(this->m_value) / divisor));
        return factory.get_bad();
    } else if (r_kind == Value::VK_VECTOR) {
        // scalar by vector element wise division
        Value_scalar *values[4];
        Value_vector *o = cast<Value_vector>(rhs);
        Type_vector  *v_type = o->get_type();
        size_t       n = v_type->get_size();

        HLSL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            Value *tmp = this->divide(factory, o->get_value(i));
            if (is<Value_bad>(tmp))
                return tmp;
            values[i] = cast<Value_scalar>(tmp);
        }
        // result is unsigned if right type is unsigned
        return factory.get_vector(v_type, Atomics_ref(values, n));
    }
    return Base::divide(factory, rhs);
}

// Integer Modulo.
template<size_t S>
Value *Value_int<S>::modulo(Value_factory &factory, Value *rhs)
{
    Value::Kind r_kind = rhs->get_kind();
    if (r_kind == s_kind) {
        S_TYPE divisor = cast<Value_int<S> >(rhs)->get_value();
        if (divisor != 0)
            return factory.get_two_complement(
                s_kind,
                S_TYPE(S_TYPE(this->m_value) % divisor));
        return factory.get_bad();
    } else if (r_kind == Base::U_KIND) {
        // unsigned if one is unsigned
        U_TYPE divisor = cast<Value_uint<S> >(rhs)->get_value();
        if (divisor != 0)
            return factory.get_two_complement(
                s_kind,
                U_TYPE(U_TYPE(this->m_value) % divisor));
        return factory.get_bad();
    } else if (r_kind == Value::VK_VECTOR) {
        // scalar by vector element wise modulo
        Value_scalar *values[4];
        Value_vector *o = cast<Value_vector>(rhs);
        Type_vector  *v_type = o->get_type();
        size_t       n = v_type->get_size();

        HLSL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            Value *tmp = this->modulo(factory, o->get_value(i));
            if (is<Value_bad>(tmp))
                return tmp;
            values[i] = cast<Value_scalar>(tmp);
        }
        // result is unsigned if right type is unsigned
        return factory.get_vector(v_type, Atomics_ref(values, n));
    }
    return Base::modulo(factory, rhs);
}

// Integer Modulo, 12bit.
template<>
Value *Value_int<12>::modulo(Value_factory &factory, Value *rhs)
{
    Value::Kind r_kind = rhs->get_kind();
    if (r_kind == s_kind) {
        S_TYPE divisor = cast<Value_int<12> >(rhs)->get_value();
        if (divisor != 0)
            return factory.get_two_complement(
                s_kind,
                S_TYPE(S_TYPE(this->m_value) % divisor));
        return factory.get_bad();
    } else if (r_kind == Value::VK_VECTOR) {
        // scalar by vector element wise modulo
        Value_scalar *values[4];
        Value_vector *o = cast<Value_vector>(rhs);
        Type_vector  *v_type = o->get_type();
        size_t       n = v_type->get_size();

        HLSL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            Value *tmp = this->modulo(factory, o->get_value(i));
            if (is<Value_bad>(tmp))
                return tmp;
            values[i] = cast<Value_scalar>(tmp);
        }
        // result is unsigned if right type is unsigned
        return factory.get_vector(v_type, Atomics_ref(values, n));
    }
    return Base::modulo(factory, rhs);
}

// Arithmetic shift right.
template<size_t S>
Value *Value_int<S>::shr(Value_factory &factory, Value *rhs)
{
    Value::Kind r_kind = rhs->get_kind();
    if (r_kind == s_kind) {
        return factory.get_two_complement(
            s_kind,
            S_TYPE(S_TYPE(this->m_value) >> cast<Value_int<S> >(rhs)->get_value()));
    } else if (r_kind == Base::U_KIND) {
        // in HLSL the result type of a shift is the left type
        return factory.get_two_complement(
            s_kind,
            S_TYPE(S_TYPE(this->m_value) >> cast<Value_uint<S> >(rhs)->get_value()));
    } else if (r_kind == Value::VK_VECTOR) {
        // scalar by vector element wise arithmetic shift right
        Value_scalar *values[4];
        Value_vector *o = cast<Value_vector>(rhs);
        Type_vector  *v_type = o->get_type();
        size_t       n = v_type->get_size();

        HLSL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            Value *tmp = this->shr(factory, o->get_value(i));
            if (is<Value_bad>(tmp))
                return tmp;
            values[i] = cast<Value_scalar>(tmp);
        }
        // in HLSL the result type of a shift is the left type
        Type_vector *res_type = factory.get_type_factory().get_vector(get_type(), n);
        return factory.get_vector(res_type, Atomics_ref(values, n));
    }
    return Base::shr(factory, rhs);
}

// Arithmetic shift right, 12bit.
template<>
Value *Value_int<12>::shr(Value_factory &factory, Value *rhs)
{
    Value::Kind r_kind = rhs->get_kind();
    if (r_kind == s_kind) {
        return factory.get_two_complement(
            s_kind,
            S_TYPE(S_TYPE(this->m_value) >> cast<Value_int<12> >(rhs)->get_value()));
    } else if (r_kind == Value::VK_VECTOR) {
        // scalar by vector element wise arithmetic shift right
        Value_scalar *values[4];
        Value_vector *o = cast<Value_vector>(rhs);
        Type_vector  *v_type = o->get_type();
        size_t       n = v_type->get_size();

        HLSL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            Value *tmp = this->shr(factory, o->get_value(i));
            if (is<Value_bad>(tmp))
                return tmp;
            values[i] = cast<Value_scalar>(tmp);
        }
        // in HLSL the result type of a shift is the left type
        Type_vector *res_type = factory.get_type_factory().get_vector(get_type(), n);
        return factory.get_vector(res_type, Atomics_ref(values, n));
    }
    return Base::shr(factory, rhs);
}

/// Compare.
template<size_t S>
Value::Compare_results Value_int<S>::compare(Value *rhs)
{
    Value::Kind r_kind = rhs->get_kind();
    if (r_kind == s_kind) {
        S_TYPE other = cast<Value_int<S> >(rhs)->get_value();
        if (S_TYPE(this->m_value) == other)
            return Value::CR_EQ;
        return S_TYPE(this->m_value) < other ? Value::CR_LT : Value::CR_GT;
    } else if (r_kind == Base::U_KIND) {
        // do unsigned compare if one is unsigned
        U_TYPE other = cast<Base>(rhs)->get_value_unsigned();
        if (U_TYPE(this->m_value) == other)
            return Value::CR_EQ;
        return U_TYPE(this->m_value) < other ? Value::CR_LT : Value::CR_GT;
    } else if (r_kind == Value::VK_VECTOR) {
        return this->inverse(rhs->compare(this));
    }
    return Base::compare(rhs);
}

// ---------------------------------- Value_uint ----------------------------------

// Constructor.
template<size_t S>
Value_uint<S>::Value_uint(TYPE_UINT *type, U_TYPE value)
: Base(type, value)
{
}

// Get the kind of value.
template<size_t S>
Value::Kind Value_uint<S>::get_kind()
{
    return s_kind;
}

// Get the type of this value.
template<size_t S>
typename Value_uint<S>::TYPE_UINT *Value_uint<S>::get_type()
{
    return cast<TYPE_UINT>(this->m_type);
}

// Multiply.
template<size_t S>
Value *Value_uint<S>::multiply(Value_factory &factory, Value *rhs)
{
    Value::Kind r_kind = rhs->get_kind();
    if (r_kind == Base::U_KIND || r_kind == Base::S_KIND) {
        // result is unsigned if one is unsigned
        return factory.get_two_complement(
            s_kind,
            U_TYPE(get_value() * cast<Base>(rhs)->get_value_unsigned()));
    } else if (r_kind == Value::VK_VECTOR) {
        return rhs->multiply(factory, this);
    }
    return Base::multiply(factory, rhs);
}

// Integer Divide.
template<size_t S>
Value *Value_uint<S>::divide(Value_factory &factory, Value *rhs)
{
    Value::Kind r_kind = rhs->get_kind();
    if (r_kind == Base::U_KIND || r_kind == Base::S_KIND) {
        // result is unsigned if one is unsigned
        S_TYPE divisor = cast<Base>(rhs)->get_value_unsigned();
        if (divisor != 0)
            return factory.get_two_complement(
                s_kind,
                U_TYPE(get_value() / divisor));
        return factory.get_bad();
    } else if (r_kind == Value::VK_VECTOR) {
        // scalar by vector element wise division
        Value_scalar *values[4];
        Value_vector *o = cast<Value_vector>(rhs);
        Type_vector  *v_type = o->get_type();
        size_t       n = v_type->get_size();

        HLSL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            Value *tmp = this->divide(factory, o->get_value(i));
            if (is<Value_bad>(tmp))
                return tmp;
            values[i] = cast<Value_scalar>(tmp);
        }
        // result is unsigned if one is unsigned
        Type_vector *res_type = factory.get_type_factory().get_vector(get_type(), n);
        return factory.get_vector(res_type, Atomics_ref(values, n));
    }
    return Base::divide(factory, rhs);
}

// Integer Modulo.
template<size_t S>
Value *Value_uint<S>::modulo(Value_factory &factory, Value *rhs)
{
    Value::Kind r_kind = rhs->get_kind();
    if (r_kind == Base::U_KIND || r_kind == Base::S_KIND) {
       // result is unsigned if one is unsigned
        U_TYPE divisor = cast<Base>(rhs)->get_value_unsigned();
        if (divisor != 0)
            return factory.get_two_complement(
                s_kind,
                U_TYPE(get_value() % divisor));
        return factory.get_bad();
    } else if (r_kind == Value::VK_VECTOR) {
        // scalar by vector element wise modulo
        Value_scalar *values[4];
        Value_vector *o = cast<Value_vector>(rhs);
        Type_vector  *v_type = o->get_type();
        size_t       n = v_type->get_size();

        HLSL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            Value *tmp = this->modulo(factory, o->get_value(i));
            if (is<Value_bad>(tmp))
                return tmp;
            values[i] = cast<Value_scalar>(tmp);
        }
        // result is unsigned if one is unsigned
        Type_vector *res_type = factory.get_type_factory().get_vector(get_type(), n);
        return factory.get_vector(res_type, Atomics_ref(values, n));
    }
    return Base::modulo(factory, rhs);
}

// Logical shift right.
template<size_t S>
Value *Value_uint<S>::shr(Value_factory &factory, Value *rhs)
{
    Value::Kind r_kind = rhs->get_kind();
    if (r_kind == Base::U_KIND || r_kind == Base::S_KIND) {
        // in HLSL, the result type is the left type
        return factory.get_two_complement(
            s_kind,
            U_TYPE(get_value() >> cast<Base>(rhs)->get_value_unsigned()));
    } else if (r_kind == Value::VK_VECTOR) {
        // scalar by vector element wise arithmetic shift right
        Value_scalar *values[4];
        Value_vector *o = cast<Value_vector>(rhs);
        Type_vector  *v_type = o->get_type();
        size_t       n = v_type->get_size();

        HLSL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            Value *tmp = this->shr(factory, o->get_value(i));
            if (is<Value_bad>(tmp))
                return tmp;
            values[i] = cast<Value_scalar>(tmp);
        }
        // in HLSL the result type of a shift is the left type
        Type_vector *res_type = factory.get_type_factory().get_vector(get_type(), n);
        return factory.get_vector(res_type, Atomics_ref(values, n));
    }
    return Base::shr(factory, rhs);
}

// Compare.
template<size_t S>
Value::Compare_results Value_uint<S>::compare(Value *rhs)
{
    Value::Kind r_kind = rhs->get_kind();
    if (r_kind == Base::U_KIND || r_kind == Base::S_KIND) {
        // compare as unsigned
        U_TYPE other = cast<Base>(rhs)->get_value_unsigned();
        if (get_value() == other)
            return Value::CR_EQ;
        return get_value() < other ? Value::CR_LT : Value::CR_GT;
    } else if (r_kind == Value::VK_VECTOR) {
        return this->inverse(this->inverse(rhs->compare(this)));
    }
    return Base::compare(rhs);
}

// ---------------------------------- Value_half ----------------------------------

/// Adjust a float value to the representable range of a half.
///
/// \param f  the float value
///
/// \return the adjusted value
static float convert_to_half(float f)
{
    union bin_cast {
        float    f;
        uint32_t u;
    } bc;

    bc.f = f;

    uint16_t h = bit_single_to_half(bc.u);
    bc.u = bit_half_to_single(h);

    return bc.f;
}

// Constructor.
Value_half::Value_half(Type_half *type, float value)
: Base(type)
, m_value(convert_to_half(value))
{
}

// Get the kind of value.
Value::Kind Value_half::get_kind()
{
    return s_kind;
}

// Get the type of this value.
Type_half *Value_half::get_type()
{
    return cast<Type_half>(m_type);
}

// Negate this value.
Value *Value_half::minus(Value_factory &factory)
{
    return factory.get_half(-m_value);
}

// Multiply.
Value *Value_half::multiply(Value_factory &factory, Value *rhs)
{
    Kind r_kind = rhs->get_kind();
    if (r_kind == s_kind)
        return factory.get_half(m_value * cast<Value_half>(rhs)->get_value());
    else if (r_kind == VK_VECTOR) {
        // float * vector
        return rhs->multiply(factory, this);
    } else if (r_kind == VK_MATRIX) {
        // float * matrix
        return rhs->multiply(factory, this);
    }
    return Base::multiply(factory, rhs);
}

// Divide.
Value *Value_half::divide(Value_factory &factory, Value *rhs)
{
    Kind r_kind = rhs->get_kind();
    if (r_kind == s_kind)
        return factory.get_half(m_value / cast<Value_half>(rhs)->get_value());
    else if (r_kind == VK_VECTOR) {
        // scalar by vector element wise division
        Value_scalar *values[4];
        Value_vector *o = cast<Value_vector>(rhs);
        Type_vector  *v_type = o->get_type();
        size_t       n = v_type->get_size();

        HLSL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            Value *tmp = this->divide(factory, o->get_value(i));
            if (is<Value_bad>(tmp))
                return tmp;
            values[i] = cast<Value_scalar>(tmp);
        }
        return factory.get_vector(v_type, Atomics_ref(values, n));
    }
    return Base::divide(factory, rhs);
}

// Add.
Value *Value_half::add(Value_factory &factory, Value *rhs)
{
    Kind r_kind = rhs->get_kind();
    if (r_kind == s_kind)
        return factory.get_half(m_value + cast<Value_half>(rhs)->get_value());
    else if (r_kind == VK_VECTOR) {
        // float + vector
        return rhs->add(factory, this);
    } else if (r_kind == VK_MATRIX) {
        // float + matrix
        return rhs->add(factory, this);
    }
    return Base::add(factory, rhs);
}

// Subtract.
Value *Value_half::sub(Value_factory &factory, Value *rhs)
{
    Kind r_kind = rhs->get_kind();
    if (r_kind == s_kind)
        return factory.get_half(m_value - cast<Value_half>(rhs)->get_value());
    else if (r_kind == VK_VECTOR) {
        // float - vector = -vector + float
        return rhs->minus(factory)->add(factory, this);
    } else if (r_kind == VK_MATRIX) {
        // float - matrix = -matrix + float
        return rhs->minus(factory)->add(factory, this);
    }
    return Base::sub(factory, rhs);
}

// Compare.
Value::Compare_results Value_half::compare(Value *rhs)
{
    Value::Kind r_kind = rhs->get_kind();
    if (r_kind == s_kind) {
        float other = cast<Value_half>(rhs)->get_value();
        if (m_value == other)
            return Value::CR_EQ;
        if (m_value < other)
            return Value::CR_LT;
        if (m_value > other)
            return Value::CR_GT;
        return Value::CR_UO;
    } else if (r_kind == VK_VECTOR) {
        return inverse(rhs->compare(this));
    }
    return Base::compare(rhs);
}

// Convert a half value.
Value *Value_half::convert(Value_factory &factory, Type *dst_tp)
{
    dst_tp = dst_tp->skip_type_alias();
    switch (dst_tp->get_kind()) {
    case Type::TK_MATRIX:
        {
            Type_matrix *x_type = cast<Type_matrix>(dst_tp);
            Type_vector *v_type = x_type->get_element_type();
            Type_scalar *a_type = v_type->get_element_type();

            if (a_type == this->m_type) {
                // convert a float into a diagonal float matrix
                Value_scalar *zero = factory.get_float(0.0f);

                Value_vector *column_vals[4];
                size_t n_cols = x_type->get_columns();
                size_t n_rows = v_type->get_size();

                for (size_t col = 0; col < n_cols; ++col) {
                    Value_scalar *row_vals[4];
                    for (size_t row = 0; row < n_rows; ++row) {
                        row_vals[row] = col == row ? this : zero;
                    }
                    column_vals[col] = factory.get_vector(v_type, Atomics_ref(row_vals, n_rows));
                }
                return factory.get_matrix(x_type, Vectors_ref(column_vals, n_cols));
            }
        }
        break;
    default:
        break;
    }
    return Base::convert(factory, dst_tp);
}

// Returns true if the value is the ZERO (i.e. additive neutral).
bool Value_half::is_zero()
{
    // we have +0.0 and -0.0, but both are additive neutral
    return m_value == 0.0f;
}

// Returns true if the value is the ONE (i.e. multiplicative neutral).
bool Value_half::is_one()
{
    return m_value == 1.0f;
}

// Returns true if all components of this value are ONE.
bool Value_half::is_all_one()
{
    // half has no components
    return false;
}

// Returns true if this value is finite (i.e. neither Inf nor NaN).
bool Value_half::is_finite()
{
    return mi::math::isfinite(m_value) != 0;
}

// Check if something is a Plus zero, aka +0.0.
bool Value_half::is_plus_zero()
{
    return m_value == 0.0f && !mi::math::sign_bit(m_value);
}

// Check if something is a Minus zero, aka -0.0.
bool Value_half::is_minus_zero()
{
    return m_value == 0.0f && mi::math::sign_bit(m_value);
}

// ---------------------------------- Value_float ----------------------------------

// Constructor.
Value_float::Value_float(Type_float *type, float value)
: Base(type)
, m_value(value)
{
}

// Get the kind of value.
Value::Kind Value_float::get_kind()
{
    return s_kind;
}

// Get the type of this value.
Type_float *Value_float::get_type()
{
    return cast<Type_float>(m_type);
}

// Negate this value.
Value_float *Value_float::minus(Value_factory &factory)
{
    return factory.get_float(-m_value);
}

// Multiply.
Value *Value_float::multiply(Value_factory &factory, Value *rhs)
{
    Kind r_kind = rhs->get_kind();
    if (r_kind == s_kind)
        return factory.get_float(m_value * cast<Value_float>(rhs)->get_value());
    else if (r_kind == VK_VECTOR) {
        // float * vector
        return rhs->multiply(factory, this);
    } else if (r_kind == VK_MATRIX) {
        // float * matrix
        return rhs->multiply(factory, this);
    }
    return Base::multiply(factory, rhs);
}

// Divide.
Value *Value_float::divide(Value_factory &factory, Value *rhs)
{
    Kind r_kind = rhs->get_kind();
    if (r_kind == s_kind)
        return factory.get_float(m_value / cast<Value_float>(rhs)->get_value());
    else if (r_kind == VK_VECTOR) {
        // scalar by vector element wise division
        Value_scalar *values[4];
        Value_vector *o = cast<Value_vector>(rhs);
        Type_vector  *v_type = o->get_type();
        size_t       n = v_type->get_size();

        HLSL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            Value *tmp = this->divide(factory, o->get_value(i));
            if (is<Value_bad>(tmp))
                return tmp;
            values[i] = cast<Value_scalar>(tmp);
        }
        return factory.get_vector(v_type, Atomics_ref(values, n));
    }
    return Base::divide(factory, rhs);
}

// Add.
Value *Value_float::add(Value_factory &factory, Value *rhs)
{
    Kind r_kind = rhs->get_kind();
    if (r_kind == s_kind)
        return factory.get_float(m_value + cast<Value_float>(rhs)->get_value());
    else if (r_kind == VK_VECTOR) {
        // float + vector
        return rhs->add(factory, this);
    } else if (r_kind == VK_MATRIX) {
        // float + matrix
        return rhs->add(factory, this);
    }
    return Base::add(factory, rhs);
}

// Subtract.
Value *Value_float::sub(Value_factory &factory, Value *rhs)
{
    Kind r_kind = rhs->get_kind();
    if (r_kind == s_kind)
        return factory.get_float(m_value - cast<Value_float>(rhs)->get_value());
    else if (r_kind == VK_VECTOR) {
        // float - vector = -vector + float
        return rhs->minus(factory)->add(factory, this);
    } else if (r_kind == VK_MATRIX) {
        // float - matrix = -matrix + float
        return rhs->minus(factory)->add(factory, this);
    }
    return Base::sub(factory, rhs);
}

// Compare.
Value::Compare_results Value_float::compare(Value *rhs)
{
    Value::Kind r_kind = rhs->get_kind();
    if (r_kind == s_kind) {
        float other = cast<Value_float>(rhs)->get_value();
        if (m_value == other)
            return Value::CR_EQ;
        if (m_value < other)
            return Value::CR_LT;
        if (m_value > other)
            return Value::CR_GT;
        return Value::CR_UO;
    } else if (r_kind == VK_VECTOR) {
        return inverse(rhs->compare(this));
    }
    return Base::compare(rhs);
}

// Convert a float value.
Value *Value_float::convert(Value_factory &factory, Type *dst_tp)
{
    dst_tp = dst_tp->skip_type_alias();
    switch (dst_tp->get_kind()) {
    case Type::TK_MATRIX:
        {
            Type_matrix *x_type = cast<Type_matrix>(dst_tp);
            Type_vector *v_type = x_type->get_element_type();
            Type_scalar *a_type = v_type->get_element_type();

            if (a_type == this->m_type) {
                // convert a float into a diagonal float matrix
                Value_scalar *zero = factory.get_float(0.0f);

                Value_vector *column_vals[4];
                size_t n_cols = x_type->get_columns();
                size_t n_rows = v_type->get_size();

                for (size_t col = 0; col < n_cols; ++col) {
                    Value_scalar *row_vals[4];
                    for (size_t row = 0; row < n_rows; ++row) {
                        row_vals[row] = col == row ? this : zero;
                    }
                    column_vals[col] = factory.get_vector(v_type, Atomics_ref(row_vals, n_rows));
                }
                return factory.get_matrix(x_type, Vectors_ref(column_vals, n_cols));
            }
        }
        break;
    default:
        break;
    }
    return Base::convert(factory, dst_tp);
}

// Returns true if the value is the ZERO (i.e. additive neutral).
bool Value_float::is_zero()
{
    // we have +0.0 and -0.0, but both are additive neutral
    return m_value == 0.0f;
}

// Returns true if the value is the ONE (i.e. multiplicative neutral).
bool Value_float::is_one()
{
    return m_value == 1.0f;
}

// Returns true if all components of this value are ONE.
bool Value_float::is_all_one()
{
    // float has no components
    return false;
}

// Returns true if this value is finite (i.e. neither Inf nor NaN).
bool Value_float::is_finite()
{
    return mi::math::isfinite(m_value) != 0;
}

// Check if something is a Plus zero, aka +0.0.
bool Value_float::is_plus_zero()
{
    return m_value == 0.0f && !mi::math::sign_bit(m_value);
}

// Check if something is a Minus zero, aka -0.0.
bool Value_float::is_minus_zero()
{
    return m_value == 0.0f && mi::math::sign_bit(m_value);
}

// ---------------------------------- Value_double ----------------------------------

// Constructor.
Value_double::Value_double(Type_double *type, double value)
: Base(type)
, m_value(value)
{
}

// Get the kind of value.
Value::Kind Value_double::get_kind()
{
    return s_kind;
}

// Get the type of this value.
Type_double *Value_double::get_type()
{
    return cast<Type_double>(m_type);
}

/// Negate this value.
Value_double *Value_double::minus(Value_factory &factory)
{
    return factory.get_double(-m_value);
}

// Multiply.
Value *Value_double::multiply(Value_factory &factory, Value *rhs)
{
    Kind r_kind = rhs->get_kind();
    if (r_kind == s_kind)
        return factory.get_double(m_value * cast<Value_double>(rhs)->get_value());
    else if (r_kind == VK_VECTOR)
        return rhs->multiply(factory, this);
    else if (r_kind == VK_MATRIX)
        return rhs->multiply(factory, this);
    return Base::multiply(factory, rhs);
}

// Divide.
Value *Value_double::divide(Value_factory &factory, Value *rhs)
{
    Kind r_kind = rhs->get_kind();
    if (r_kind == s_kind)
        return factory.get_double(m_value / cast<Value_double>(rhs)->get_value());
    else if (r_kind == VK_VECTOR) {
        // scalar by vector element wise division
        Value_scalar *values[4];
        Value_vector *o = cast<Value_vector>(rhs);
        Type_vector  *v_type = o->get_type();
        size_t       n = v_type->get_size();

        HLSL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            Value *tmp = this->divide(factory, o->get_value(i));
            if (is<Value_bad>(tmp))
                return tmp;
            values[i] = cast<Value_scalar>(tmp);
        }
        return factory.get_vector(v_type, Atomics_ref(values, n));
    }
    return Base::divide(factory, rhs);
}

// Add.
Value *Value_double::add(Value_factory &factory, Value *rhs)
{
    Kind r_kind = rhs->get_kind();
    if (r_kind == s_kind)
        return factory.get_double(m_value + cast<Value_double>(rhs)->get_value());
    else if (r_kind == VK_VECTOR)
        return rhs->add(factory, this);
    else if (r_kind == VK_MATRIX)
        return rhs->add(factory, this);
    return Base::add(factory, rhs);
}

// Subtract.
Value *Value_double::sub(Value_factory &factory, Value *rhs)
{
    Kind r_kind = rhs->get_kind();
    if (r_kind == s_kind)
        return factory.get_double(m_value - cast<Value_double>(rhs)->get_value());
    else if (r_kind == VK_VECTOR) {
        // double - vector = (-vector) + double
        return rhs->minus(factory)->add(factory, this);
    } else if (r_kind == VK_MATRIX) {
        // double - matrix = (-matrix) + double
        return rhs->minus(factory)->add(factory, this);
    }
    return Base::sub(factory, rhs);
}

// Compare.
Value::Compare_results Value_double::compare(Value *rhs)
{
    Value::Kind r_kind = rhs->get_kind();
    if (r_kind == s_kind) {
        double other = cast<Value_double>(rhs)->get_value();
        if (m_value == other)
            return Value::CR_EQ;
        if (m_value < other)
            return Value::CR_LT;
        if (m_value > other)
            return Value::CR_GT;
        return Value::CR_UO;
    } else if (r_kind == VK_VECTOR) {
        return inverse(rhs->compare(this));
    }
    return Base::compare(rhs);
}

// Convert a double value.
Value *Value_double::convert(Value_factory &factory, Type *dst_tp)
{
    dst_tp = dst_tp->skip_type_alias();
    if (dst_tp->get_kind() == Type::TK_MATRIX) {
        // convert a double into a diagonal double matrix
        Type_matrix *x_type = cast<Type_matrix>(dst_tp);
        Type_vector *v_type = x_type->get_element_type();
        Type_scalar *a_type = v_type->get_element_type();

        if (a_type == this->m_type) {
            Value_scalar *zero = factory.get_double(0.0);

            Value_vector *column_vals[4];
            size_t n_cols = x_type->get_columns();
            size_t n_rows = v_type->get_size();

            for (size_t col = 0; col < n_cols; ++col) {
                Value_scalar *row_vals[4];
                for (size_t row = 0; row < n_rows; ++row) {
                    row_vals[row] = col == row ? this : zero;
                }
                column_vals[col] = factory.get_vector(v_type, Atomics_ref(row_vals, n_rows));
            }
            return factory.get_matrix(x_type, Vectors_ref(column_vals, n_cols));
        }
    }
    return Base::convert(factory, dst_tp);
}

// Returns true if the value is the ZERO (i.e. additive neutral).
bool Value_double::is_zero()
{
    // we have +0.0 and -0.0, but both are additive neutral
    return m_value == 0.0;
}

// Returns true if the value is the ONE (i.e. multiplicative neutral).
bool Value_double::is_one()
{
    return m_value == 1.0;
}

// Returns true if all components of this value are ONE.
bool Value_double::is_all_one()
{
    // double has no components
    return false;
}

// Returns true if this value is finite (i.e. neither Inf nor NaN).
bool Value_double::is_finite()
{
    return mi::math::isfinite(m_value) != 0;
}

// Check if something is a Plus zero, aka +0.0.
bool Value_double::is_plus_zero()
{
    return m_value == 0.0f && !mi::math::sign_bit(m_value);
}

// Check if something is a Minus zero, aka -0.0.
bool Value_double::is_minus_zero()
{
    return m_value == 0.0f && mi::math::sign_bit(m_value);
}

// ---------------------------------- Value_compound ----------------------------------

/// Constructor.
Value_compound::Value_compound(
    Memory_arena             *arena,
    Type_compound            *type,
    Array_ref<Value *> const &values)
: Base(type)
, m_values(*arena, values.size())
{
    for (size_t i = 0, n = values.size(); i < n; ++i)
        m_values[i] = values[i];
}

// Get the type of this value.
Type_compound *Value_compound::get_type()
{
    return cast<Type_compound>(m_type);
}

// Extract from a compound.
Value *Value_compound::extract(Value_factory &factory, size_t index)
{
    if (index < m_values.size())
        return m_values[index];
    return factory.get_bad();
}

// Returns true if this value is finite.
bool Value_compound::is_finite()
{
    for (size_t i = 0, n = m_values.size(); i < n; ++i) {
        if (!m_values[i]->is_finite())
            return false;
    }
    return true;
}

// Get the value at index.
Value *Value_compound::get_value(size_t index)
{
    HLSL_ASSERT(index < m_values.size() && "Index out of bounds");
    return m_values[index];
}

// ---------------------------------- Value_vector ----------------------------------

// Constructor.
Value_vector::Value_vector(
    Memory_arena                    *arena,
    Type_vector                     *type,
    Array_ref<Value_scalar *> const &values)
: Base(arena, type, cast<Values_ref>(values))
{
    HLSL_ASSERT(values.size() == type->get_size());
}

// Get the kind of value.
Value::Kind Value_vector::get_kind()
{
    return s_kind;
}

// Get the type of this value.
Type_vector *Value_vector::get_type()
{
    return cast<Type_vector>(m_type);
}

// Negate this value.
Value *Value_vector::minus(Value_factory &factory)
{
    Value_scalar *values[4];
    size_t       n = m_values.size();

    HLSL_ASSERT(n <= 4);
    for (size_t i = 0; i < n; ++i) {
        Value *tmp = m_values[i]->minus(factory);
        if (is<Value_bad>(tmp))
            return tmp;
        values[i] = cast<Value_scalar>(tmp);
    }
    return factory.get_vector(Value_vector::get_type(), Atomics_ref(values, n));
}

// Bitwise negate this value.
Value *Value_vector::bitwise_not(Value_factory &factory)
{
    Value_scalar *values[4];
    size_t       n = m_values.size();

    HLSL_ASSERT(n <= 4);
    for (size_t i = 0; i < n; ++i) {
        Value *tmp = m_values[i]->bitwise_not(factory);
        if (is<Value_bad>(tmp))
            return tmp;
        values[i] = cast<Value_scalar>(tmp);
    }
    return factory.get_vector(Value_vector::get_type(), Atomics_ref(values, n));
}

// Logically negate this value.
Value *Value_vector::logical_not(Value_factory &factory)
{
    Value_scalar *values[4];
    size_t       n = m_values.size();

    HLSL_ASSERT(n <= 4);
    for (size_t i = 0; i < n; ++i) {
        Value *tmp = m_values[i]->logical_not(factory);
        if (is<Value_bad>(tmp))
            return tmp;
        values[i] = cast<Value_scalar>(tmp);
    }
    return factory.get_vector(Value_vector::get_type(), Atomics_ref(values, n));
}

// Multiply.
Value *Value_vector::multiply(Value_factory &factory, Value *rhs)
{
    Type_vector *lhs_type = Value_vector::get_type();
    Type        *rhs_type = rhs->get_type();

    if (rhs_type == lhs_type) {
        // vector by vector element wise multiplication
        Value_scalar *values[4];
        size_t       n = m_values.size();
        Value_vector *o = cast<Value_vector>(rhs);

        HLSL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            Value *tmp = m_values[i]->multiply(factory, o->get_value(i));
            if (is<Value_bad>(tmp))
                return tmp;
            values[i] = cast<Value_scalar>(tmp);
        }
        return factory.get_vector(lhs_type, Atomics_ref(values, n));
    } else if (rhs_type == lhs_type->get_element_type()) {
        // vector by scalar element wise multiplication
        Value_scalar *values[4];
        size_t       n = m_values.size();

        HLSL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            Value *tmp = m_values[i]->multiply(factory, rhs);
            if (is<Value_bad>(tmp))
                return tmp;
            values[i] = cast<Value_scalar>(tmp);
        }
        return factory.get_vector(lhs_type, Atomics_ref(values, n));
    } else if (Type_matrix *x_type = as<Type_matrix>(rhs_type)) {
        // vector by matrix multiplication
        Type_vector *v_type = x_type->get_element_type();
        size_t      n_rows  = v_type->get_size();
        if (lhs_type->get_size() == n_rows) {
            size_t       n_cols  = x_type->get_columns();
            Value_matrix *o      = cast<Value_matrix>(rhs);
            Type_factory &t_fact = factory.get_type_factory();
            Type_scalar  *a_type = lhs_type->get_element_type();
            Type_vector  *r_type = t_fact.get_vector(a_type, n_cols);

            Value_scalar *values[4];
            for (size_t col = 0; col < n_cols; ++col) {
                Value        *e   = m_values[0];
                Value_vector *r   = cast<Value_vector>(o->get_value(col));
                Value        *tmp = e->multiply(factory, r->get_value(0));
                for (size_t row = 1; row < n_rows; ++row) {
                    e   = m_values[row];
                    tmp = tmp->add(factory, e->multiply(factory, r->get_value(row)));
                }
                if (is<Value_bad>(tmp))
                    return tmp;
                values[col] = cast<Value_scalar>(tmp);
            }
            return factory.get_vector(r_type, Atomics_ref(values, n_cols));
        }
    }
    return Base::multiply(factory, rhs);
}

// Divide.
Value *Value_vector::divide(Value_factory &factory, Value *rhs)
{
    Type_vector *lhs_type = Value_vector::get_type();
    Type        *rhs_type = rhs->get_type();

    if (rhs_type == lhs_type) {
        // vector by vector element wise division
        Value_scalar *values[4];
        size_t       n = m_values.size();
        Value_vector *o = cast<Value_vector>(rhs);

        HLSL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            Value *tmp = m_values[i]->divide(factory, o->get_value(i));
            if (is<Value_bad>(tmp))
                return tmp;
            values[i] = cast<Value_scalar>(tmp);
        }
        return factory.get_vector(lhs_type, Atomics_ref(values, n));
    } else if (rhs_type == lhs_type->get_element_type()) {
        // vector by scalar element wise division
        Value_scalar *values[4];
        size_t n = m_values.size();

        HLSL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            Value *tmp = m_values[i]->divide(factory, rhs);
            if (is<Value_bad>(tmp))
                return tmp;
            values[i] = cast<Value_scalar>(tmp);
        }
        return factory.get_vector(lhs_type, Atomics_ref(values, n));
    }
    return Base::divide(factory, rhs);
}

// Modulo.
Value *Value_vector::modulo(Value_factory &factory, Value *rhs)
{
    Type_vector *lhs_type = Value_vector::get_type();
    Type        *rhs_type = rhs->get_type();

    if (rhs_type == lhs_type) {
        // vector by vector element wise modulo
        Value_scalar *values[4];
        size_t       n = m_values.size();
        Value_vector *o = cast<Value_vector>(rhs);

        HLSL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            Value *tmp = m_values[i]->modulo(factory, o->get_value(i));
            if (is<Value_bad>(tmp))
                return tmp;
            values[i] = cast<Value_scalar>(tmp);
        }
        return factory.get_vector(lhs_type, Atomics_ref(values, n));
    } else if (rhs_type == lhs_type->get_element_type()) {
        // vector by scalar element wise modulo
        Value_scalar *values[4];
        size_t       n = m_values.size();

        HLSL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            Value *tmp = m_values[i]->modulo(factory, rhs);
            if (is<Value_bad>(tmp))
                return tmp;
            values[i] = cast<Value_scalar>(tmp);
        }
        return factory.get_vector(lhs_type, Atomics_ref(values, n));
    }
    return Base::modulo(factory, rhs);
}

// Add.
Value *Value_vector::add(Value_factory &factory, Value *rhs)
{
    Type_vector *lhs_type = Value_vector::get_type();
    Type        *rhs_type = rhs->get_type();

    if (rhs_type == lhs_type) {
        // vector by vector element wise addition
        Value_scalar *values[4];
        size_t       n = m_values.size();
        Value_vector *o = cast<Value_vector>(rhs);
            
        HLSL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            Value *tmp = m_values[i]->add(factory, o->get_value(i));
            if (is<Value_bad>(tmp))
                return tmp;
            values[i] = cast<Value_scalar>(tmp);
        }
        return factory.get_vector(lhs_type, Atomics_ref(values, n));
    } else if (rhs_type == lhs_type->get_element_type()) {
        // vector by scalar element wise addition
        Value_scalar *values[4];
        size_t       n = m_values.size();

        HLSL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            Value *tmp = m_values[i]->add(factory, rhs);
            if (is<Value_bad>(tmp))
                return tmp;
            values[i] = cast<Value_scalar>(tmp);
        }
        return factory.get_vector(lhs_type, Atomics_ref(values, n));
    }
    return Base::add(factory, rhs);
}

// Subtract.
Value *Value_vector::sub(Value_factory &factory, Value *rhs)
{
    Type_vector *lhs_type = Value_vector::get_type();
    Type        *rhs_type = rhs->get_type();

    if (rhs_type == lhs_type) {
        // vector by vector element wise subtraction
        Value_scalar *values[4];
        size_t       n = m_values.size();
        Value_vector *o = cast<Value_vector>(rhs);

        HLSL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            Value *tmp = m_values[i]->sub(factory, o->get_value(i));
            if (is<Value_bad>(tmp))
                return tmp;
            values[i] = cast<Value_scalar>(tmp);
        }
        return factory.get_vector(lhs_type, Atomics_ref(values, n));
    } else if (rhs_type == lhs_type->get_element_type()) {
        // vector by scalar element wise subtraction
        Value_scalar *values[4];
        size_t       n = m_values.size();

        HLSL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            Value *tmp = m_values[i]->sub(factory, rhs);
            if (is<Value_bad>(tmp))
                return tmp;
            values[i] = cast<Value_scalar>(tmp);
        }
        return factory.get_vector(lhs_type, Atomics_ref(values, n));
    }
    return Base::sub(factory, rhs);
}

// Shift left.
Value *Value_vector::shl(Value_factory &factory, Value *rhs)
{
    Type_vector *lhs_type = Value_vector::get_type();
    Type        *rhs_type = rhs->get_type();

    if (rhs_type == lhs_type) {
        // vector by vector element wise shift left
        Value_scalar *values[4];
        size_t       n = m_values.size();
        Value_vector *o = cast<Value_vector>(rhs);

        HLSL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            Value *tmp = m_values[i]->shl(factory, o->get_value(i));
            if (is<Value_bad>(tmp))
                return tmp;
            values[i] = cast<Value_scalar>(tmp);
        }
        return factory.get_vector(lhs_type, Atomics_ref(values, n));
    } else if (rhs_type == lhs_type->get_element_type()) {
        // vector by scalar element wise shift left
        Value_scalar *values[4];
        size_t       n = m_values.size();

        HLSL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            Value *tmp = m_values[i]->shl(factory, rhs);
            if (is<Value_bad>(tmp))
                return tmp;
            values[i] = cast<Value_scalar>(tmp);
        }
        return factory.get_vector(lhs_type, Atomics_ref(values, n));
    }
    return Base::shl(factory, rhs);
}

// Shift right.
Value *Value_vector::shr(Value_factory &factory, Value *rhs)
{
    Type_vector *lhs_type = Value_vector::get_type();
    Type        *rhs_type = rhs->get_type();

    if (rhs_type == lhs_type) {
        // vector by vector element wise  shift right
        Value_scalar *values[4];
        size_t       n = m_values.size();
        Value_vector *o = cast<Value_vector>(rhs);

        HLSL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            Value *tmp = m_values[i]->shr(factory, o->get_value(i));
            if (is<Value_bad>(tmp))
                return tmp;
            values[i] = cast<Value_scalar>(tmp);
        }
        return factory.get_vector(lhs_type, Atomics_ref(values, n));
    } else if (rhs_type == lhs_type->get_element_type()) {
        // vector by scalar element wise shift right
        Value_scalar *values[4];
        size_t       n = m_values.size();

        HLSL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            Value *tmp = m_values[i]->shr(factory, rhs);
            if (is<Value_bad>(tmp))
                return tmp;
            values[i] = cast<Value_scalar>(tmp);
        }
        return factory.get_vector(lhs_type, Atomics_ref(values, n));
    }
    return Base::shr(factory, rhs);
}

// Xor.
Value *Value_vector::bitwise_xor(Value_factory &factory, Value *rhs)
{
    Type_vector *lhs_type = Value_vector::get_type();
    Type        *rhs_type = rhs->get_type();

    if (rhs_type == lhs_type) {
        // vector by vector element wise exclusive or
        Value_scalar *values[4];
        size_t       n = m_values.size();
        Value_vector *o = cast<Value_vector>(rhs);

        HLSL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            Value *tmp = m_values[i]->bitwise_xor(factory, o->get_value(i));
            if (is<Value_bad>(tmp))
                return tmp;
            values[i] = cast<Value_scalar>(tmp);
        }
        return factory.get_vector(lhs_type, Atomics_ref(values, n));
    } else if (rhs_type == lhs_type->get_element_type()) {
        // vector by scalar element wise exclusive or
        Value_scalar *values[4];
        size_t       n = m_values.size();

        HLSL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            Value *tmp = m_values[i]->bitwise_xor(factory, rhs);
            if (is<Value_bad>(tmp))
                return tmp;
            values[i] = cast<Value_scalar>(tmp);
        }
        return factory.get_vector(lhs_type, Atomics_ref(values, n));
    }
    return Base::bitwise_xor(factory, rhs);
}

// Or.
Value *Value_vector::bitwise_or(Value_factory &factory, Value *rhs)
{
    Type_vector *lhs_type = Value_vector::get_type();
    Type        *rhs_type = rhs->get_type();

    if (rhs_type == lhs_type) {
        // vector by vector element wise or
        Value_scalar *values[4];
        size_t       n = m_values.size();
        Value_vector *o = cast<Value_vector>(rhs);

        HLSL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            Value *tmp = m_values[i]->bitwise_or(factory, o->get_value(i));
            if (is<Value_bad>(tmp))
                return tmp;
            values[i] = cast<Value_scalar>(tmp);
        }
        return factory.get_vector(lhs_type, Atomics_ref(values, n));
    } else if (rhs_type == lhs_type->get_element_type()) {
        // vector by scalar element wise or
        Value_scalar *values[4];
        size_t       n = m_values.size();

        HLSL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            Value *tmp = m_values[i]->bitwise_or(factory, rhs);
            if (is<Value_bad>(tmp))
                return tmp;
            values[i] = cast<Value_scalar>(tmp);
        }
        return factory.get_vector(lhs_type, Atomics_ref(values, n));
    }
    return Base::bitwise_or(factory, rhs);
}

// And.
Value *Value_vector::bitwise_and(Value_factory &factory, Value *rhs)
{
    Type_vector *lhs_type = Value_vector::get_type();
    Type        *rhs_type = rhs->get_type();

    if (rhs_type == lhs_type) {
        // vector by vector element wise and
        Value_scalar *values[4];
        size_t       n = m_values.size();
        Value_vector *o = cast<Value_vector>(rhs);

        HLSL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            Value *tmp = m_values[i]->bitwise_and(factory, o->get_value(i));
            if (is<Value_bad>(tmp))
                return tmp;
            values[i] = cast<Value_scalar>(tmp);
        }
        return factory.get_vector(lhs_type, Atomics_ref(values, n));
    } else if (rhs_type == lhs_type->get_element_type()) {
        // vector by scalar element wise and
        Value_scalar *values[4];
        size_t       n = m_values.size();

        HLSL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            Value *tmp = m_values[i]->bitwise_and(factory, rhs);
            if (is<Value_bad>(tmp))
                return tmp;
            values[i] = cast<Value_scalar>(tmp);
        }
        return factory.get_vector(lhs_type, Atomics_ref(values, n));
    }
    return Base::bitwise_and(factory, rhs);
}

// Logical Or.
Value *Value_vector::logical_or(Value_factory &factory, Value *rhs)
{
    Type_vector *lhs_type = Value_vector::get_type();
    Type        *rhs_type = rhs->get_type();

    if (rhs_type == lhs_type) {
        // vector element wise logical or
        Value_scalar *values[4];
        size_t       n = m_values.size();
        Value_vector *o = cast<Value_vector>(rhs);

        HLSL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            Value *tmp = m_values[i]->logical_or(factory, o->get_value(i));
            if (is<Value_bad>(tmp))
                return tmp;
            values[i] = cast<Value_scalar>(tmp);
        }
        return factory.get_vector(lhs_type, Atomics_ref(values, n));
    } else if (rhs_type == lhs_type->get_element_type()) {
        // vector by scalar element wise logical or
        Value_scalar *values[4];
        size_t n = m_values.size();

        HLSL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            Value *tmp = m_values[i]->logical_or(factory, rhs);
            if (is<Value_bad>(tmp))
                return tmp;
            values[i] = cast<Value_scalar>(tmp);
        }
        return factory.get_vector(lhs_type, Atomics_ref(values, n));
    }
    return Base::logical_or(factory, rhs);
}

// Logical And.
Value *Value_vector::logical_and(Value_factory &factory, Value *rhs)
{
    Type_vector *lhs_type = Value_vector::get_type();
    Type        *rhs_type = rhs->get_type();

    if (rhs_type == lhs_type) {
        // vector element wise logical and
        Value_scalar *values[4];
        size_t       n = m_values.size();
        Value_vector *o = cast<Value_vector>(rhs);

        HLSL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            Value *tmp = m_values[i]->logical_and(factory, o->get_value(i));
            if (is<Value_bad>(tmp))
                return tmp;
            values[i] = cast<Value_scalar>(tmp);
        }
        return factory.get_vector(lhs_type, Atomics_ref(values, n));
    } else if (rhs_type == lhs_type->get_element_type()) {
        // vector by scalar element wise logical and
        Value_scalar *values[4];
        size_t       n = m_values.size();

        HLSL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            Value *tmp = m_values[i]->logical_and(factory, rhs);
            if (is<Value_bad>(tmp))
                return tmp;
            values[i] = cast<Value_scalar>(tmp);
        }
        return factory.get_vector(lhs_type, Atomics_ref(values, n));
    }
    return Base::logical_and(factory, rhs);
}

// Compare.
Value::Compare_results Value_vector::compare(Value *rhs)
{
    Type_vector *lhs_type = Value_vector::get_type();
    Type        *rhs_type = rhs->get_type();

    if (rhs_type == lhs_type) {
        // vector by vector compare
        size_t       n = m_values.size();
        Value_vector *o = cast<Value_vector>(rhs);

        Value::Compare_results res = Value::CR_EQ;
        for (size_t i = 0; i < n; ++i) {
            unsigned tmp = m_values[i]->compare(o->get_value(i));
            if (tmp & Value::CR_UO)
                return Value::CR_UO;
            if ((tmp & Value::CR_EQ) == 0)
                res = Value::CR_NE;
        }
        return res; 
    } else if (rhs_type == lhs_type->get_element_type()) {
        // vector by scalar element wise compare
        size_t n = m_values.size();

        Value::Compare_results res = Value::CR_EQ;
        for (size_t i = 0; i < n; ++i) {
            unsigned tmp = m_values[i]->compare(rhs);
            if (tmp & Value::CR_UO)
                return Value::CR_UO;
            if ((tmp & Value::CR_EQ) == 0)
                res = Value::CR_NE;
        }
        return res; 
    }
    return Base::compare(rhs);
}

// Convert a vector value.
Value *Value_vector::convert(Value_factory &factory, Type *dst_tp)
{
    if (Type_vector *v_type = as<Type_vector>(dst_tp)) {
        size_t n = m_values.size();

        if (v_type->get_size() == n) {
            Type_scalar *a_type = v_type->get_element_type();
            Value_scalar *values[4];

            for (size_t i = 0; i < n; ++i) {
                Value *tmp = m_values[i]->convert(factory, a_type);
                if (is<Value_bad>(tmp))
                    return tmp;
                values[i] = cast<Value_scalar>(tmp);
            }
            return factory.get_vector(v_type, Atomics_ref(values, n));
        }
    }
    return Base::convert(factory, dst_tp);
}

// Returns true if the value is the ZERO (i.e. additive neutral).
bool Value_vector::is_zero()
{
    bool   neutral = true;
    size_t n = m_values.size();

    for (size_t i = 0; i < n; ++i) {
        neutral &= m_values[i]->is_zero();
    }
    return neutral;
}

// Returns true if the value is the ONE (i.e. multiplicative neutral).
bool Value_vector::is_one()
{
    bool   all_one = true;
    size_t n = m_values.size();

    for (size_t i = 0; i < n; ++i) {
        all_one &= m_values[i]->is_one();
    }
    return all_one;
}

// Returns true if all components of this value are ONE.
bool Value_vector::is_all_one()
{
    bool   all_all_one = true;
    size_t n = m_values.size();

    for (size_t i = 0; i < n; ++i) {
        all_all_one &= m_values[i]->is_all_one();
    }
    return all_all_one;
}

// ---------------------------------- Value_matrix ----------------------------------

// Constructor.
Value_matrix::Value_matrix(
    Memory_arena                    *arena,
    Type_matrix                     *type,
    Array_ref<Value_vector *> const &values)
: Base(arena, type, cast<Values_ref>(values))
{
    HLSL_ASSERT(values.size() == type->get_columns());
}

// Get the kind of value.
Value::Kind Value_matrix::get_kind()
{
    return s_kind;
}

// Get the type of this value.
Type_matrix *Value_matrix::get_type()
{
    return cast<Type_matrix>(m_type);
}

// Negate this value.
Value *Value_matrix::minus(Value_factory &factory)
{
    Value_vector *values[4];
    size_t       n = m_values.size();

    HLSL_ASSERT(n <= 4);
    for (size_t i = 0; i < n; ++i) {
        Value *tmp = m_values[i]->minus(factory);
        if (is<Value_bad>(tmp))
            return tmp;
        values[i] = cast<Value_vector>(tmp);
    }
    return factory.get_matrix(Value_matrix::get_type(), Vectors_ref(values, n));
}

// Matrix multiplication.
Value *Value_matrix::multiply(Value_factory &factory, Value *rhs)
{
    Type_matrix *lhs_type = Value_matrix::get_type();
    Type        *rhs_type = rhs->get_type();

    if (Type_matrix *o_type = as<Type_matrix>(rhs_type)) {
        // matrix by matrix MxN * KxM ==> KxN
        size_t const M = lhs_type->get_columns();
        if (M != o_type->get_element_type()->get_size())
            return factory.get_bad();

        Type_vector *v_type = lhs_type->get_element_type();

        size_t const N = v_type->get_size();
        size_t const K = o_type->get_columns();

        Value_matrix *o = cast<Value_matrix>(rhs);
        Type_factory &t_fact = factory.get_type_factory();

        // the vector type of the result is a a vector of length N
        v_type = t_fact.get_vector(v_type->get_element_type(), N);

        Value_vector *columns[4];
        for (size_t col = 0; col < K; ++col) {
            Value_scalar *column[4];
            for (size_t row = 0; row < N; ++row) {
                Value        *l  = cast<Value_vector>(m_values[0])->get_value(row);
                Value_vector *rv = cast<Value_vector>(o->get_value(col));
                Value        *tmp = l->multiply(factory, rv->get_value(0));

                for (size_t m = 1; m < M; ++m) {
                    l = cast<Value_vector>(m_values[m])->get_value(row);
                    tmp = tmp->add(factory, l->multiply(factory, rv->get_value(m)));
                }
                if (is<Value_bad>(tmp))
                    return tmp;
                column[row] = cast<Value_scalar>(tmp);
            }
            columns[col] = factory.get_vector(v_type, Atomics_ref(column, N));
        }
        Type_matrix *r_type = t_fact.get_matrix(v_type, K);
        return factory.get_matrix(r_type, Vectors_ref(columns, K));
    }
    if (rhs_type == lhs_type->get_element_type()->get_element_type()) {
        // matrix by scalar element wise multiplication
        Value_vector *values[4];
        size_t       n = m_values.size();

        HLSL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            Value *tmp = m_values[i]->multiply(factory, rhs);
            if (is<Value_bad>(tmp))
                return tmp;
            values[i] = cast<Value_vector>(tmp);
        }
        return factory.get_matrix(lhs_type, Vectors_ref(values, n));
    }
    if (Type_vector *v_type = as<Type_vector>(rhs_type)) {
        // matrix by vector
        size_t n_cols = lhs_type->get_columns();

        if (v_type->get_size() == n_cols) {
            Type_vector  *v_type = lhs_type->get_element_type();
            size_t       n_rows  = v_type->get_size();
            Value_vector *o      = cast<Value_vector>(rhs);

            Value_scalar *values[4];
            for (size_t row = 0; row < n_rows; ++row) {
                Value *e   = cast<Value_vector>(m_values[0])->get_value(row);
                Value *r   = o->get_value(0);
                Value *tmp = e->multiply(factory, r);
                for (size_t col = 1; col < n_cols; ++col) {
                    e = cast<Value_vector>(m_values[col])->get_value(row);
                    r = o->get_value(col);
                    e = e->multiply(factory, r);
                    tmp = tmp->add(factory, e);
                }
                if (is<Value_bad>(tmp))
                    return tmp;
                values[row] = cast<Value_scalar>(tmp);
            }
            return factory.get_vector(lhs_type->get_element_type(), Atomics_ref(values, n_rows));
        }
        return factory.get_bad();
    }
    return Base::multiply(factory, rhs);
}

// Divide.
Value *Value_matrix::divide(Value_factory &factory, Value *rhs)
{
    Type_matrix *lhs_type = Value_matrix::get_type();
    Type        *rhs_type = rhs->get_type();

    if (rhs_type == lhs_type->get_element_type()->get_element_type()) {
        // matrix by scalar element wise division
        Value_vector *values[4];
        size_t       n = m_values.size();

        HLSL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            Value *tmp = m_values[i]->divide(factory, rhs);
            if (is<Value_bad>(tmp))
                return tmp;
            values[i] = cast<Value_vector>(tmp);
        }
        return factory.get_matrix(lhs_type, Vectors_ref(values, n));
    }
    return Base::divide(factory, rhs);
}

// Add.
Value *Value_matrix::add(Value_factory &factory, Value *rhs)
{
    Type_matrix *lhs_type = Value_matrix::get_type();
    Type        *rhs_type = rhs->get_type();

    if (rhs_type == lhs_type) {
        // matrix element wise addition
        Value_vector *values[4];
        size_t       n = m_values.size();
        Value_matrix *o = cast<Value_matrix>(rhs);
            
        HLSL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            Value *tmp = m_values[i]->add(factory, o->get_value(i));
            if (is<Value_bad>(tmp))
                return tmp;
            values[i] = cast<Value_vector>(tmp);
        }
        return factory.get_matrix(lhs_type, Vectors_ref(values, n));
    }
    return factory.get_bad();
}

// Subtract.
Value *Value_matrix::sub(Value_factory &factory, Value *rhs)
{
    Type_matrix *lhs_type = Value_matrix::get_type();
    Type        *rhs_type = rhs->get_type();

    if (rhs_type == lhs_type) {
        // matrix element wise subtraction
        Value_vector *values[4];
        size_t       n = m_values.size();
        Value_matrix *o = cast<Value_matrix>(rhs);

        HLSL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            Value *tmp = m_values[i]->sub(factory, o->get_value(i));
            if (is<Value_bad>(tmp))
                return tmp;
            values[i] = cast<Value_vector>(tmp);
        }
        return factory.get_matrix(lhs_type, Vectors_ref(values, n));
    }
    return Base::sub(factory, rhs);
}

// Compare.
Value::Compare_results Value_matrix::compare(Value *rhs)
{
    if (rhs->get_kind() == s_kind) {
        // matrix by matrix element wise compare
        size_t       n = m_values.size();
        Value_matrix *o = cast<Value_matrix>(rhs);

        Value::Compare_results res = Value::CR_EQ;
        for (size_t i = 0; i < n; ++i) {
            unsigned tmp = m_values[i]->compare(o->get_value(i));
            if (tmp & Value::CR_UO)
                return Value::CR_UO;
            if ((tmp & Value::CR_EQ) == 0)
                res = Value::CR_NE;
        }
        return res;
    }
    return Base::compare(rhs);
}

// Convert a matrix value.
Value *Value_matrix::convert(Value_factory &factory, Type *dst_tp)
{
    if (Type_matrix *x_type = as<Type_matrix>(dst_tp)) {
        size_t n_cols = m_values.size();

        if (x_type->get_columns() == n_cols) {
            Type_vector  *v_type = x_type->get_element_type();
            Value_vector *values[4];

            for (size_t i = 0; i < n_cols; ++i) {
                Value *tmp = m_values[i]->convert(factory, v_type);
                if (is<Value_bad>(tmp))
                    return tmp;
                values[i] = cast<Value_vector>(tmp);
            }
            return factory.get_matrix(x_type, Vectors_ref(values, n_cols));
        }
    }
    return Base::convert(factory, dst_tp);
}

// Returns true if the value is the ZERO (i.e. additive neutral).
bool Value_matrix::is_zero()
{
    bool   neutral = true;
    size_t n = m_values.size();

    for (size_t i = 0; i < n; ++i) {
        neutral &= m_values[i]->is_zero();
    }
    return neutral;
}

// Returns true if the value is the identity matrix (i.e. multiplicative neutral).
bool Value_matrix::is_one()
{
    size_t n = m_values.size();
    Type_vector *e_type = Value_matrix::get_type()->get_element_type();

    if (e_type->get_size() == n) {
        // square matrix

        for (size_t i = 0; i < n; ++i) {
            Value_vector *vec = cast<Value_vector>(m_values[i]);

            for (size_t j = 0; j < n; ++j) {
                Value *v = vec->get_value(j);
                if (j == i) {
                    if (!v->is_one())
                        return false;
                } else {
                    if (!v->is_zero())
                        return false;
                }
            }
        }
        return true;
    }
    return false;
}

// ---------------------------------- Value array ----------------------------------

// Constructor.
Value_array::Value_array(
    Memory_arena             *arena,
    Type_array               *type,
    Array_ref<Value *> const &values)
: Base(arena, type, values)
{
    HLSL_ASSERT(values.size() == type->get_size());
}

// Get the kind of value.
Value::Kind Value_array::get_kind()
{
    return s_kind;
}

// Get the type of this value.
Type_array *Value_array::get_type()
{
    return cast<Type_array>(m_type);
}

// ---------------------------------- Value_struct ----------------------------------

// Constructor.
Value_struct::Value_struct(
    Memory_arena             *arena,
    Type_struct              *type,
    Array_ref<Value *> const &values)
: Base(arena, type, values)
{
    HLSL_ASSERT(values.size() == type->get_field_count());
}

// Get the kind of value.
Value::Kind Value_struct::get_kind()
{
    return s_kind;
}

// Get the type of this value.
Type_struct *Value_struct::get_type()
{
    return cast<Type_struct>(m_type);
}

// Get a field.
Value *Value_struct::get_field(Symbol *name)
{
    Type_struct *s_type = Value_struct::get_type();
    for (size_t i = 0, n = s_type->get_field_count(); i < n; ++i) {
        Type_struct::Field *field = s_type->get_field(i);

        if (field->get_symbol() == name) {
            // found
            return get_value(i);
        }
    }
    HLSL_ASSERT(!"field name not found");
    return NULL;
}

// Get a field.
Value *Value_struct::get_field(char const *name)
{
    Type_struct *s_type = Value_struct::get_type();
    for (size_t i = 0, n = s_type->get_field_count(); i < n; ++i) {
        Type_struct::Field *field = s_type->get_field(i);

        if (strcmp(field->get_symbol()->get_name(), name) == 0) {
            // found
            return get_value(i);
        }
    }
    HLSL_ASSERT(!"field name not found");
    return NULL;
}

// ---------------------------------- value factory ----------------------------------

// Constructor.
Value_factory::Value_factory(Memory_arena &arena, Type_factory &tf)
: Base()
, m_builder(arena)
, m_tf(tf)
, m_vt(0, Value_hash(), Value_equal(), arena.get_allocator())
// Note: the following values are allocated on an arena, no free is needed
, m_bad_value(m_builder.create<Value_bad>(tf.get_error()))
, m_void_value(m_builder.create<Value_void>(tf.get_void()))
, m_true_value(m_builder.create<Value_bool>(tf.get_bool(), true))
, m_false_value(m_builder.create<Value_bool>(tf.get_bool(), false))
{
}

// Get the (singleton) bad value.
Value_bad *Value_factory::get_bad()
{
    return m_bad_value;
}

// Get the (singleton) void value.
Value_void *Value_factory::get_void()
{
    return m_void_value;
}

// Get a value of type boolean.
Value_bool *Value_factory::get_bool(bool value)
{
    return value ? m_true_value : m_false_value;
}

// Get a value of type 12bit integer.
Value_int_12 *Value_factory::get_int12(int16_t value)
{
    Value_int_12 *v = m_builder.create<Value_int_12>(m_tf.get_min12int(), value);

    std::pair<Value_table::iterator, bool> res = m_vt.insert(v);
    if (!res.second) {
        m_builder.get_arena()->drop(v);
        return cast<Value_int_12>(*res.first);
    }
    return v;
}

// Get a value of type 16bit integer.
Value_int_16 *Value_factory::get_int16(int16_t value)
{
    Value_int_16 *v = m_builder.create<Value_int_16>(m_tf.get_min16int(), value);

    std::pair<Value_table::iterator, bool> res = m_vt.insert(v);
    if (!res.second) {
        m_builder.get_arena()->drop(v);
        return cast<Value_int_16>(*res.first);
    }
    return v;
}

// Get a value of type 16bit unsigned.
Value_uint_16 *Value_factory::get_uint16(uint16_t value)
{
    Value_uint_16 *v = m_builder.create<Value_uint_16>(m_tf.get_min16uint(), value);

    std::pair<Value_table::iterator, bool> res = m_vt.insert(v);
    if (!res.second) {
        m_builder.get_arena()->drop(v);
        return cast<Value_uint_16>(*res.first);
    }
    return v;
}

// Get a value of type 32bit integer.
Value_int_32 *Value_factory::get_int32(int32_t value)
{
    Value_int_32 *v = m_builder.create<Value_int_32>(m_tf.get_int(), value);

    std::pair<Value_table::iterator, bool> res = m_vt.insert(v);
    if (!res.second) {
        m_builder.get_arena()->drop(v);
        return cast<Value_int_32>(*res.first);
    }
    return v;
}

// Get a value of type 32bit unsigned.
Value_uint_32 *Value_factory::get_uint32(uint32_t value)
{
    Value_uint_32 *v = m_builder.create<Value_uint_32>(m_tf.get_uint(), value);

    std::pair<Value_table::iterator, bool> res = m_vt.insert(v);
    if (!res.second) {
        m_builder.get_arena()->drop(v);
        return cast<Value_uint_32>(*res.first);
    }
    return v;
}

// Get a value of type half.
Value_half *Value_factory::get_half(float value)
{
    Value_half *v = m_builder.create<Value_half>(m_tf.get_half(), value);
    std::pair<Value_table::iterator, bool> res = m_vt.insert(v);
    if (!res.second) {
        m_builder.get_arena()->drop(v);
        return cast<Value_half>(*res.first);
    }
    return v;
}

// Get a value of type float.
Value_float *Value_factory::get_float(float value)
{
    Value_float *v = m_builder.create<Value_float>(m_tf.get_float(), value);
    std::pair<Value_table::iterator, bool> res = m_vt.insert(v);
    if (!res.second) {
        m_builder.get_arena()->drop(v);
        return cast<Value_float>(*res.first);
    }
    return v;
}

// Get a value of type double.
Value_double *Value_factory::get_double(double value)
{
    Value_double *v = m_builder.create<Value_double>(m_tf.get_double(), value);
    std::pair<Value_table::iterator, bool> res = m_vt.insert(v);
    if (!res.second) {
        m_builder.get_arena()->drop(v);
        return cast<Value_double>(*res.first);
    }
    return v;
}

// Get a value of type vector.
Value_vector *Value_factory::get_vector(
    Type_vector                     *type,
    Array_ref<Value_scalar *> const &values)
{
    Value_vector *v = m_builder.create<Value_vector>(m_builder.get_arena(), type, values);
    std::pair<Value_table::iterator, bool> res = m_vt.insert(v);
    if (!res.second) {
        m_builder.get_arena()->drop(v);
        return cast<Value_vector>(*res.first);
    }
    return v;
}

// Get a value of type matrix.
Value_matrix *Value_factory::get_matrix(
    Type_matrix                     *type,
    Array_ref<Value_vector *> const &values)
{
    Value_matrix *v = m_builder.create<Value_matrix>(m_builder.get_arena(), type, values);
    std::pair<Value_table::iterator, bool> res = m_vt.insert(v);
    if (!res.second) {
        m_builder.get_arena()->drop(v);
        return cast<Value_matrix>(*res.first);
    }
    return v;
}

// Get a  value of type array.
Value_array *Value_factory::get_array(
    Type_array               *type,
    Array_ref<Value *> const &values)
{
    Value_array *v = m_builder.create<Value_array>(m_builder.get_arena(), type, values);
    std::pair<Value_table::iterator, bool> res = m_vt.insert(v);
    if (!res.second) {
        m_builder.get_arena()->drop(v);
        return cast<Value_array>(*res.first);
    }
    return v;
}

// Get a value of type struct.
Value_struct *Value_factory::get_struct(
    Type_struct              *type,
    Array_ref<Value *> const &values)
{
    Value_struct *v = m_builder.create<Value_struct>(m_builder.get_arena(), type, values);
    std::pair<Value_table::iterator, bool> res = m_vt.insert(v);
    if (!res.second) {
        m_builder.get_arena()->drop(v);
        return cast<Value_struct>(*res.first);
    }
    return v;
}

// Get a two-complement 32bit value.
Value *Value_factory::get_two_complement(
    Value::Kind kind,
    uint32_t    value)
{
    switch (kind) {
    case Value::VK_INT:
        return get_int32(int(value));
    case Value::VK_UINT:
        return get_uint32(value);
    case Value::VK_MIN12INT:
        return get_int16(int16_t(mask_bits(value, 12)));
    case Value::VK_MIN16INT:
        return get_int16(int16_t(value));
    case Value::VK_MIN16UINT:
        return get_uint16(uint16_t(value));
    default:
        HLSL_ASSERT(!"Unsupported two complement kind");
        return get_bad();
    }
}

// Create a additive neutral zero if supported for the given type.
Value *Value_factory::get_zero(Type *type)
{
    switch (type->get_kind()) {
    case Type::TK_BOOL:
        return get_bool(false);
    case Type::TK_INT:
        return get_int32(int32_t(0));
    case Type::TK_UINT:
        return get_uint32(uint32_t(0));
    case Type::TK_HALF:
        return get_half(0.0f);
    case Type::TK_FLOAT:
        return get_float(0.0f);
    case Type::TK_DOUBLE:
        return get_double(0.0);
    case Type::TK_MIN12INT:
        return get_int12(int32_t(0));
    case Type::TK_MIN16INT:
        return get_int16(int32_t(0));
    case Type::TK_MIN16UINT:
        return get_uint16(uint32_t(0));
    case Type::TK_MIN10FLOAT:
        return get_half(0.0f);
    case Type::TK_MIN16FLOAT:
        return get_half(0.0f);
    case Type::TK_VECTOR:
        {
            Type_vector *v_type = cast<Type_vector>(type);
            Type_scalar *a_type = v_type->get_element_type();

            Value_scalar *zero = cast<Value_scalar>(get_zero(a_type));

            HLSL_ASSERT(!is<Value_bad>(zero) && v_type->get_size() <= 4);
            Value_scalar *values[4] = { zero, zero, zero, zero };

            return get_vector(v_type, Atomics_ref(values, v_type->get_size()));
        }
    case Type::TK_MATRIX:
        {
            Type_matrix *m_type = cast<Type_matrix>(type);
            Type_vector *v_type = m_type->get_element_type();

            Value_vector *zero = cast<Value_vector>(get_zero(v_type));

            HLSL_ASSERT(!is<Value_bad>(zero) && m_type->get_columns() <= 4);
            Value_vector *values[4] = { zero, zero, zero, zero };

            return get_matrix(m_type, Vectors_ref(values, m_type->get_columns()));
        }
    case Type::TK_ARRAY:
        {
            Type_array *a_type = cast<Type_array>(type);
            Type       *e_type = a_type->get_element_type();

            Small_VLA<Value *, 8> zero_values(
                m_builder.get_arena()->get_allocator(), a_type->get_size());

            Value *zero = get_zero(e_type);
            for (size_t i = 0, n = zero_values.size(); i < n; ++i) {
                zero_values[i] = zero;
            }

            return get_array(a_type, zero_values);
        }
    default:
        return get_bad();
    }
}

// Get a zero initializer if supported for the given type.
Value *Value_factory::get_zero_initializer(Type *type)
{
    switch (type->get_kind()) {
    case Type::TK_BOOL:
        return get_bool(false);
    case Type::TK_INT:
        return get_int32(int32_t(0));
    case Type::TK_UINT:
        return get_uint32(uint32_t(0));
    case Type::TK_HALF:
        return get_half(0.0f);
    case Type::TK_FLOAT:
        return get_float(0.0f);
    case Type::TK_DOUBLE:
        return get_double(0.0);
    case Type::TK_MIN12INT:
        return get_int12(int32_t(0));
    case Type::TK_MIN16INT:
        return get_int16(int32_t(0));
    case Type::TK_MIN16UINT:
        return get_uint16(uint32_t(0));
    case Type::TK_MIN10FLOAT:
        return get_half(0.0f);
    case Type::TK_MIN16FLOAT:
        return get_half(0.0f);
    case Type::TK_VECTOR:
        {
            Type_vector *v_type = cast<Type_vector>(type);
            Type_scalar *a_type = v_type->get_element_type();

            Value_scalar *zero = cast<Value_scalar>(get_zero(a_type));

            HLSL_ASSERT(!is<Value_bad>(zero) && v_type->get_size() <= 4);
            Value_scalar *values[4] = { zero, zero, zero, zero };

            return get_vector(v_type, Atomics_ref(values, v_type->get_size()));
        }
    case Type::TK_MATRIX:
        {
            Type_matrix *m_type = cast<Type_matrix>(type);
            Type_vector *v_type = m_type->get_element_type();

            Value_vector *zero = cast<Value_vector>(get_zero(v_type));

            HLSL_ASSERT(!is<Value_bad>(zero) && m_type->get_columns() <= 4);
            Value_vector *values[4] = { zero, zero, zero, zero };

            return get_matrix(m_type, Vectors_ref(values, m_type->get_columns()));
        }
    case Type::TK_STRUCT:
        {
            Type_struct *s_type = cast<Type_struct>(type);
            size_t num_fields = s_type->get_compound_size();
            Small_VLA<Value *, 8> values(
                m_builder.get_arena()->get_allocator(), num_fields);
            for (size_t i = 0; i < num_fields; ++i) {
                values[i] = get_zero_initializer(s_type->get_compound_type(i));
            }
            return get_struct(s_type, Values_ref(values));
        }
    case Type::TK_ARRAY:
        {
            Type_array *a_type = cast<Type_array>(type);
            Type       *e_type = a_type->get_element_type();
            size_t a_size = a_type->get_size();
            Small_VLA<Value *, 8> values(
                m_builder.get_arena()->get_allocator(), a_size);
            for (size_t i = 0; i < a_size; ++i) {
                values[i] = get_zero_initializer(e_type);
            }
            return get_array(a_type, Values_ref(values));
        }
    default:
        return get_bad();
    }
}

// Return the type factory of this value factory.
Type_factory &Value_factory::get_type_factory()
{
    return m_tf;
}

// Dump all values owned by this Value table.
void Value_factory::dump() const
{
#if 0
    /// A local helper class that creates a printer for the debug output stream.
    class Alloc {
    public:
        Alloc(IAllocator *a) : m_builder(a) {}

        Debug_Output_stream *dbg() {
            return m_builder.create<Debug_Output_stream>(m_builder.get_allocator());
        }

        Printer *prt(IOutput_stream *s) {
            return m_builder.create<Printer>(m_builder.get_allocator(), s);
        }

    private:
        Allocator_builder m_builder;
    };
    
    Alloc alloc(m_builder.get_arena()->get_allocator());

    mi::base::Handle<Debug_Output_stream> dbg(alloc.dbg());
    mi::base::Handle<HLSL_Printer>        printer(alloc.prt(dbg.get()));

    long i = 0;
    for (Value_table::const_iterator it(m_vt.begin()), end(m_vt.end());
         it != end;
         ++it)
    {
        Value *val = *it;

        printer->print(i++);
        printer->print(" ");
        printer->print(val);
        printer->print("\n");
    }
#endif
}

size_t Value_factory::Value_hash::operator() (Value *value) const
{
    size_t h = (size_t(value->get_type()) >> 4) * 5;
    Value::Kind kind = value->get_kind();
    switch (kind) {
    case Value::VK_BAD:
    case Value::VK_VOID:
        return h;
    case Value::VK_BOOL:
        return h + size_t(cast<Value_bool>(value)->get_value());
    case Value::VK_INT:
    case Value::VK_UINT:
        return h + size_t(cast<Value_two_complement<32> >(value)->get_value_unsigned());
    case Value::VK_MIN12INT:
        return h + size_t(cast<Value_two_complement<12> >(value)->get_value_unsigned());
    case Value::VK_MIN16INT:
    case Value::VK_MIN16UINT:
        return h + size_t(cast<Value_two_complement<16> >(value)->get_value_unsigned());
    case Value::VK_HALF:
        {
            union { size_t z; float f; } u;
            u.z = 0;
            u.f = cast<Value_half>(value)->get_value();
            return h + u.z;
        }
    case Value::VK_FLOAT:
        {
            union { size_t z; float f; } u;
            u.z = 0;
            u.f = cast<Value_float>(value)->get_value();
            return h + u.z;
        }
    case Value::VK_DOUBLE:
        {
            union { size_t z[2]; double d; } u;
            u.z[0] = 0;
            u.z[1] = 0;
            u.d = cast<Value_double>(value)->get_value();
            return h + (u.z[0] ^ u.z[1]);
        }
    case Value::VK_VECTOR:
    case Value::VK_MATRIX:
    case Value::VK_ARRAY:
    case Value::VK_STRUCT:
        {
            Value_compound *c = cast<Value_compound>(value);
            size_t h = 0;
            for (size_t i = 0, size = c->get_component_count(); i < size; ++i) {
                h = h * 5 + operator()(c->get_value(i));
            }
            return h;
        }
    }
    return 0;
}

bool Value_factory::Value_equal::operator()(Value *a, Value *b) const
{
    Value::Kind kind = a->get_kind();
    if (kind != b->get_kind())
        return false;

    if (a->get_type() != b->get_type())
        return false;

    switch (kind) {
    case Value::VK_BAD:
    case Value::VK_VOID:
        // depends only on type
        return true;
    case Value::VK_BOOL:
        return cast<Value_bool>(a)->get_value() == cast<Value_bool>(b)->get_value();
    case Value::VK_INT:
        return cast<Value_int_32>(a)->get_value() == cast<Value_int_32>(b)->get_value();
    case Value::VK_UINT:
        return cast<Value_uint_32>(a)->get_value() == cast<Value_uint_32>(b)->get_value();
    case Value::VK_MIN12INT:
        return cast<Value_int_12>(a)->get_value() == cast<Value_int_12>(b)->get_value();
    case Value::VK_MIN16INT:
        return cast<Value_int_16>(a)->get_value() == cast<Value_int_16>(b)->get_value();
    case Value::VK_MIN16UINT:
        return cast<Value_uint_16>(a)->get_value() == cast<Value_uint_16>(b)->get_value();
    case Value::VK_HALF:
        return bit_equal_float(
            cast<Value_half>(a)->get_value(), cast<Value_half>(b)->get_value());
    case Value::VK_FLOAT:
        return bit_equal_float(
            cast<Value_float>(a)->get_value(), cast<Value_float>(b)->get_value());
    case Value::VK_DOUBLE:
        return bit_equal_float(
            cast<Value_double>(a)->get_value(), cast<Value_double>(b)->get_value());
    case Value::VK_VECTOR:
    case Value::VK_MATRIX:
    case Value::VK_ARRAY:
    case Value::VK_STRUCT:
        {
            Value_compound *ca = cast<Value_compound>(a);
            Value_compound *cb = cast<Value_compound>(b);
            size_t size = ca->get_component_count();
            if (size != cb->get_component_count())
                return false;
            for (size_t i = 0; i < size; ++i) {
                if (!operator()(ca->get_value(i), cb->get_value(i)))
                    return false;
            }
            return true;
        }
    }
    return false;
}

}  // hlsl
}  // mdl
}  // mi
