/******************************************************************************
 * Copyright (c) 2011-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <mi/mdl/mdl_values.h>
#include <mi/mdl/mdl_iowned.h>
#include <mi/base/iallocator.h>
#include <mi/math/function.h>

#include "compilercore_cc_conf.h"
#include "compilercore_memory_arena.h"
#include "compilercore_factories.h"
#include "compilercore_tools.h"
#include "compilercore_assert.h"
#include "compilercore_streams.h"
#include "compilercore_printers.h"
#include "compilercore_serializer.h"

namespace mi {
namespace mdl {

template<>
inline Value_factory *impl_cast(IValue_factory *t) {
    return static_cast<Value_factory *>(t);
}

/// A mixin base class for all base IValue methods.
///
/// \tparam Interface  the base class for the desired IValue
/// \tparam Type       the type of all IValues of kind Interface
template <typename Interface, typename Type>
class Value_base : public Interface
{
    typedef Interface Base;
public:
    /// Get the kind of expression.
    typename Interface::Kind get_kind() const MDL_FINAL { return Interface::s_kind; }

    /// Get the type of this value.
    Type const *get_type() const MDL_FINAL { return m_type; }

    /// Negate this value.
    IValue const *minus(IValue_factory *factory) const MDL_OVERRIDE {
        return factory->create_bad();
    }

    /// Bitwise negate this value.
    IValue const *bitwise_not(IValue_factory *factory) const MDL_OVERRIDE {
        return factory->create_bad();
    }

    /// Logically negate this value.
    IValue const *logical_not(IValue_factory *factory) const MDL_OVERRIDE {
        return factory->create_bad();
    }

    /// Extract from a compound.
    IValue const *extract(IValue_factory *factory, int) const MDL_OVERRIDE {
        return factory->create_bad();
    }

    /// Multiply.
    IValue const *multiply(IValue_factory *factory, IValue const *) const MDL_OVERRIDE {
        return factory->create_bad();
    }

    /// Divide.
    IValue const *divide(IValue_factory *factory, IValue const *) const MDL_OVERRIDE {
        return factory->create_bad();
    }

    /// Modulo.
    IValue const *modulo(IValue_factory *factory, IValue const *) const MDL_OVERRIDE {
        return factory->create_bad();
    }

    /// Add.
    IValue const *add(IValue_factory *factory, IValue const *) const MDL_OVERRIDE {
        return factory->create_bad();
    }

    /// Subtract.
    IValue const *sub(IValue_factory *factory, IValue const *) const MDL_OVERRIDE {
        return factory->create_bad();
    }

    /// Shift left.
    IValue const *shl(IValue_factory *factory, IValue const *) const MDL_OVERRIDE {
        return factory->create_bad();
    }

    /// Arithmetic shift right.
    IValue const *asr(IValue_factory *factory, IValue const *) const MDL_OVERRIDE {
        return factory->create_bad();
    }

    /// Logical shift right.
    IValue const *lsr(IValue_factory *factory, IValue const *) const MDL_OVERRIDE {
        return factory->create_bad();
    }

    /// Bitwise Xor.
    IValue const *bitwise_xor(IValue_factory *factory, IValue const *) const MDL_OVERRIDE {
        return factory->create_bad();
    }

    /// Bitwise Or.
    IValue const *bitwise_or(IValue_factory *factory, IValue const *) const MDL_OVERRIDE {
        return factory->create_bad();
    }

    /// Bitwise And.
    IValue const *bitwise_and(IValue_factory *factory, IValue const *) const MDL_OVERRIDE {
        return factory->create_bad();
    }

    /// Logical Or.
    IValue const *logical_or(IValue_factory *factory, IValue const *) const MDL_OVERRIDE {
        return factory->create_bad();
    }

    /// Logical And.
    IValue const *logical_and(IValue_factory *factory, IValue const *) const MDL_OVERRIDE {
        return factory->create_bad();
    }

    /// Compare.
    IValue::Compare_results compare(IValue const *) const MDL_OVERRIDE {
        return IValue::CR_BAD;
    }

    /// Convert a value.
    IValue const *convert(
        IValue_factory *factory,
        IType const *dst_type) const MDL_OVERRIDE
    {
        dst_type = dst_type->skip_type_alias();
        if (dst_type == m_type) {
            // no conversion
            return this;
        }
        return factory->create_bad();
    }

    /// Returns true if the value is the ZERO (i.e. additive neutral).
    bool is_zero() const MDL_OVERRIDE {
        return false;
    }

    /// Returns true if the value is the ONE (i.e. multiplicative neutral).
    bool is_one() const MDL_OVERRIDE {
        return false;
    }

    /// Returns true if all components of this value are ONE.
    bool is_all_one() const MDL_OVERRIDE {
        return false;
    }

protected:
    /// Constructor.
    explicit Value_base(Type const *type)
    : Base()
    , m_type(type)
    {
    }

    /// The type of this value.
    Type const *const m_type;
};

/// Implementation of the invalid reference value.
class Value_invalid_ref : public Value_base<IValue_invalid_ref, IType_reference>
{
    typedef Value_base<IValue_invalid_ref, IType_reference> Base;
public:
    /// Returns true if this value is finite (i.e. neither Inf nor NaN).
    bool is_finite() const MDL_FINAL {
        return true;
    }

    /// Constructor.
    explicit Value_invalid_ref(IType_reference const *type)
    : Base(type)
    {
    }
};

/// A mixin base class for all atomic value classes.
///
/// \tparam Interface  the base class for the desired IValue
/// \tparam Type       the type of all IValues of kind Interface
/// \tparam ValueType  a scalar type that can hold values of this IValue
template<typename Interface, typename Type, typename ValueType>
class Value_atomic : public Value_base<Interface, Type>
{
    typedef Value_base<Interface, Type> Base;
public:

    /// Get the value.
    ValueType get_value() const MDL_OVERRIDE { return m_value; }

    /// Convert an atomic value.
    IValue const *convert(IValue_factory *factory, IType const *dst_tp) const MDL_OVERRIDE
    {
        dst_tp = dst_tp->skip_type_alias();
        switch (dst_tp->get_kind()) {
        case IType::TK_BOOL:   return factory->create_bool(m_value != ValueType(0));
        case IType::TK_INT:    return factory->create_int(int(m_value));
        case IType::TK_FLOAT:  return factory->create_float(float(m_value));
        case IType::TK_DOUBLE: return factory->create_double(double(m_value));
        case IType::TK_VECTOR:
            {
                IType_vector const *v_type = cast<IType_vector>(dst_tp);
                IValue const *v = convert(factory, v_type->get_element_type());
                if (is<IValue_bad>(v))
                    break;
                IValue const *args[4] = { v, v, v, v };
                return factory->create_vector(v_type, args, v_type->get_size());
            }
        default:
            // no other conversions supported
            break;
        }
        return Base::convert(factory, dst_tp);
    }

protected:
    /// Constructor.
    explicit Value_atomic(Type const *type, ValueType value)
    : Base(type)
    , m_value(value)
    {
    }

    /// The atomic value.
    ValueType const m_value;
};

/// Implementation of the bad value.
class Value_bad : public Value_base<IValue_bad, IType_error>
{
    typedef Value_base<IValue_bad, IType_error> Base;
public:
    /// Returns true if this value is finite (i.e. neither Inf nor NaN).
    bool is_finite() const MDL_FINAL {
        return false;
    }

    /// Constructor.
    explicit Value_bad(IType_error const *type)
    : Base(type)
    {
    }
};

/// Implementation of the IValue_bool class
class Value_bool : public Value_atomic<IValue_bool, IType_bool, bool>
{
    typedef Value_atomic<IValue_bool, IType_bool, bool> Base;
public:
    /// Logically negate this value.
    IValue const *logical_not(IValue_factory *factory) const MDL_FINAL {
        return factory->create_bool(!m_value);
    }

    /// Logical Or.
    IValue const *logical_or(IValue_factory *factory, IValue const *rhs) const MDL_FINAL {
        Kind r_kind = rhs->get_kind();
        if (r_kind == s_kind)
            return factory->create_bool(m_value || cast<IValue_bool>(rhs)->get_value());
        else if (r_kind == VK_VECTOR)
            return rhs->logical_or(factory, this);
        return factory->create_bad();
    }

    /// Logical And.
    IValue const *logical_and(IValue_factory *factory, IValue const *rhs) const MDL_FINAL {
        Kind r_kind = rhs->get_kind();
        if (r_kind == s_kind)
            return factory->create_bool(m_value && cast<IValue_bool>(rhs)->get_value());
        else if (r_kind == VK_VECTOR)
            return rhs->logical_and(factory, this);
        return factory->create_bad();
    }

    /// Compare.
    IValue::Compare_results compare(IValue const *rhs) const MDL_FINAL {
        Kind r_kind = rhs->get_kind();
        if (r_kind == s_kind) {
            bool other = cast<IValue_bool>(rhs)->get_value();
            if (m_value == other)
                return IValue::CR_EQ;
            return IValue::CR_NE;
        } else if (r_kind == VK_VECTOR) {
            return IValue::inverse(rhs->compare(this));
        }
        return IValue::CR_BAD;
    }

    /// Returns true if the value is the ZERO (i.e. additive neutral).
    bool is_zero() const MDL_FINAL {
        // false is the additive neutral
        return m_value == false;
    }

    /// Returns true if the value is the ONE (i.e. multiplicative neutral).
    bool is_one() const MDL_FINAL {
        // true is the multiplicative neutral
        return m_value == true;
    }

    /// Returns true if all components of this value are ONE.
    bool is_all_one() const MDL_FINAL {
        // bool has only ONE component
        return Value_bool::is_one();
    }

    /// Returns true if this value is finite (i.e. neither Inf nor NaN).
    bool is_finite() const MDL_FINAL {
        return true;
    }

    /// Constructor.
    explicit Value_bool(IType_bool const *type, bool value)
    : Base(type, value)
    {
    }
};

/// Implementation of the IValue_int class
class Value_int : public Value_atomic<IValue_int, IType_int, int>
{
    typedef Value_atomic<IValue_int, IType_int, int> Base;
public:
    /// Negate this value.
    IValue_int const *minus(IValue_factory *factory) const MDL_FINAL {
        return factory->create_int(-m_value);
    }

    /// Bitwise negate this value.
    IValue const *bitwise_not(IValue_factory *factory) const MDL_FINAL {
        return factory->create_int(~m_value);
    }

    /// Multiply.
    IValue const *multiply(IValue_factory *factory, IValue const *rhs) const MDL_FINAL {
        Kind r_kind = rhs->get_kind();
        if (r_kind == s_kind)
            return factory->create_int(m_value * cast<IValue_int>(rhs)->get_value());
        else if (r_kind == VK_VECTOR)
            return rhs->multiply(factory, this);
        return factory->create_bad();
    }

    /// Integer Divide.
    IValue const *divide(IValue_factory *factory, IValue const *rhs) const MDL_FINAL {
        Kind r_kind = rhs->get_kind();
        if (r_kind == s_kind) {
            int divisor = cast<IValue_int>(rhs)->get_value();
            if (divisor != 0)
                return factory->create_int(m_value / divisor);
            return factory->create_bad();
        } else if (r_kind == VK_VECTOR) {
            // scalar by vector element wise division
            IValue const        *values[4];
            IValue_vector const *o = cast<IValue_vector>(rhs);
            IType_vector const  *v_type = o->get_type();
            size_t              n = v_type->get_size();

            MDL_ASSERT(n <= 4);
            for (size_t i = 0; i < n; ++i) {
                IValue const *tmp = this->divide(factory, o->get_value(i));
                if (is<IValue_bad>(tmp))
                    return tmp;
                values[i] = tmp;
            }
            return factory->create_vector(v_type, values, n);
        }
        return factory->create_bad();
    }

    /// Integer Modulo.
    IValue const *modulo(IValue_factory *factory, IValue const *rhs) const MDL_FINAL {
        Kind r_kind = rhs->get_kind();
        if (r_kind == s_kind) {
            int divisor = cast<IValue_int>(rhs)->get_value();
            if (divisor != 0)
                return factory->create_int(m_value % divisor);
            return factory->create_bad();
        } else if (r_kind == VK_VECTOR) {
            // scalar by vector element wise modulo
            IValue const        *values[4];
            IValue_vector const *o = cast<IValue_vector>(rhs);
            IType_vector const  *v_type = o->get_type();
            size_t              n = v_type->get_size();

            MDL_ASSERT(n <= 4);
            for (size_t i = 0; i < n; ++i) {
                IValue const *tmp = this->modulo(factory, o->get_value(i));
                if (is<IValue_bad>(tmp))
                    return tmp;
                values[i] = tmp;
            }
            return factory->create_vector(v_type, values, n);
        }
        return factory->create_bad();
    }

    /// Add.
    IValue const *add(IValue_factory *factory, IValue const *rhs) const MDL_FINAL {
        Kind r_kind = rhs->get_kind();
        if (r_kind == s_kind)
            return factory->create_int(m_value + cast<IValue_int>(rhs)->get_value());
        else if (r_kind == VK_VECTOR)
            return rhs->add(factory, this);
        return factory->create_bad();
    }

    /// Subtract.
    IValue const *sub(IValue_factory *factory, IValue const *rhs) const MDL_FINAL {
        Kind r_kind = rhs->get_kind();
        if (r_kind == s_kind)
            return factory->create_int(m_value - cast<IValue_int>(rhs)->get_value());
        else if (r_kind == VK_VECTOR) {
            // scalar - vector = (-vector) + scalar
            return rhs->minus(factory)->add(factory, this);
        }
        return factory->create_bad();
    }

    /// Shift left.
    IValue const *shl(IValue_factory *factory, IValue const *rhs) const MDL_FINAL {
        Kind r_kind = rhs->get_kind();
        if (r_kind == s_kind)
            return factory->create_int(m_value << cast<IValue_int>(rhs)->get_value());
        else if (r_kind == VK_VECTOR) {
            // scalar by vector element wise shift left
            IValue const        *values[4];
            IValue_vector const *o = cast<IValue_vector>(rhs);
            IType_vector const  *v_type = o->get_type();
            size_t              n = v_type->get_size();

            MDL_ASSERT(n <= 4);
            for (size_t i = 0; i < n; ++i) {
                IValue const *tmp = this->shl(factory, o->get_value(i));
                if (is<IValue_bad>(tmp))
                    return tmp;
                values[i] = tmp;
            }
            return factory->create_vector(v_type, values, n);
        }
        return factory->create_bad();
    }

    /// Arithmetic shift right.
    IValue const *asr(IValue_factory *factory, IValue const *rhs) const MDL_FINAL {
        Kind r_kind = rhs->get_kind();
        if (r_kind == s_kind)
            return factory->create_int(m_value >> cast<IValue_int>(rhs)->get_value());
        else if (r_kind == VK_VECTOR) {
            // scalar by vector element wise arithmetic shift right
            IValue const        *values[4];
            IValue_vector const *o = cast<IValue_vector>(rhs);
            IType_vector const  *v_type = o->get_type();
            size_t              n = v_type->get_size();

            MDL_ASSERT(n <= 4);
            for (size_t i = 0; i < n; ++i) {
                IValue const *tmp = this->asr(factory, o->get_value(i));
                if (is<IValue_bad>(tmp))
                    return tmp;
                values[i] = tmp;
            }
            return factory->create_vector(v_type, values, n);
        }
        return factory->create_bad();
    }

    /// Logical shift right.
    IValue const *lsr(IValue_factory *factory, IValue const *rhs) const MDL_FINAL {
        Kind r_kind = rhs->get_kind();
        if (r_kind == s_kind)
            return factory->create_int(unsigned(m_value) >> cast<IValue_int>(rhs)->get_value());
        else if (r_kind == VK_VECTOR) {
            // scalar by vector element wise logical shift right
            IValue const        *values[4];
            IValue_vector const *o = cast<IValue_vector>(rhs);
            IType_vector const  *v_type = o->get_type();
            size_t              n = v_type->get_size();

            MDL_ASSERT(n <= 4);
            for (size_t i = 0; i < n; ++i) {
                IValue const *tmp = this->lsr(factory, o->get_value(i));
                if (is<IValue_bad>(tmp))
                    return tmp;
                values[i] = tmp;
            }
            return factory->create_vector(v_type, values, n);
        }
        return factory->create_bad();
    }

    /// Xor.
    IValue const *bitwise_xor(IValue_factory *factory, IValue const *rhs) const MDL_FINAL {
        Kind r_kind = rhs->get_kind();
        if (r_kind == s_kind)
            return factory->create_int(m_value ^ cast<IValue_int>(rhs)->get_value());
        else if (r_kind == VK_VECTOR)
            return rhs->bitwise_xor(factory, this);
        return factory->create_bad();
    }

    /// Or.
    IValue const *bitwise_or(IValue_factory *factory, IValue const *rhs) const MDL_FINAL {
        Kind r_kind = rhs->get_kind();
        if (r_kind == s_kind)
            return factory->create_int(m_value | cast<IValue_int>(rhs)->get_value());
        else if (r_kind == VK_VECTOR)
            return rhs->bitwise_or(factory, this);
        return factory->create_bad();
    }

    /// And.
    IValue const *bitwise_and(IValue_factory *factory, IValue const *rhs) const MDL_FINAL {
        Kind r_kind = rhs->get_kind();
        if (r_kind == s_kind)
            return factory->create_int(m_value & cast<IValue_int>(rhs)->get_value());
        else if (r_kind == VK_VECTOR)
            return rhs->bitwise_and(factory, this);
        return factory->create_bad();
    }

    /// Compare.
    IValue::Compare_results compare(IValue const *rhs) const MDL_FINAL {
        Kind r_kind = rhs->get_kind();
        if (r_kind == s_kind) {
            int other = cast<IValue_int>(rhs)->get_value();
            if (m_value == other)
                return IValue::CR_EQ;
            return m_value < other ? IValue::CR_LT : IValue::CR_GT;
        } else if (r_kind == VK_VECTOR) {
            return IValue::inverse(rhs->compare(this));
        }
        return IValue::CR_BAD;
    }

    /// Returns true if the value is the ZERO (i.e. additive neutral).
    bool is_zero() const MDL_FINAL {
        return m_value == 0;
    }

    /// Returns true if the value is the ONE (i.e. multiplicative neutral).
    bool is_one() const MDL_FINAL {
        return m_value == 1;
    }

    /// Returns true if all components of this value are ONE.
    bool is_all_one() const MDL_FINAL {
        // treat bits as components
        return m_value == ~0;
    }

    /// Returns true if this value is finite (i.e. neither Inf nor NaN).
    bool is_finite() const MDL_FINAL {
        return true;
    }

    /// Constructor.
    explicit Value_int(IType_int const *type, int value)
    : Base(type, value)
    {
    }
};

/// Implementation of the IValue_float class
class Value_float : public Value_atomic<IValue_float, IType_float, float>
{
    typedef Value_atomic<IValue_float, IType_float, float> Base;
public:
    /// Negate this value.
    IValue_float const *minus(IValue_factory *factory) const MDL_FINAL {
        return factory->create_float(-m_value);
    }

    /// Multiply.
    IValue const *multiply(IValue_factory *factory, IValue const *rhs) const MDL_FINAL {
        Kind r_kind = rhs->get_kind();
        if (r_kind == s_kind)
            return factory->create_float(m_value * cast<IValue_float>(rhs)->get_value());
        else if (r_kind == VK_VECTOR) {
            // float * vector
            return rhs->multiply(factory, this);
        } else if (r_kind == VK_RGB_COLOR) {
            // float * color
            return rhs->multiply(factory, this);
        } else if (r_kind == VK_MATRIX) {
            // float * matrix
            return rhs->multiply(factory, this);
        }
        return factory->create_bad();
    }

    /// Divide.
    IValue const *divide(IValue_factory *factory, IValue const *rhs) const MDL_FINAL {
        Kind r_kind = rhs->get_kind();
        if (r_kind == s_kind)
            return factory->create_float(m_value / cast<IValue_float>(rhs)->get_value());
        else if (r_kind == VK_VECTOR) {
            // scalar by vector element wise division
            IValue const        *values[4];
            IValue_vector const *o = cast<IValue_vector>(rhs);
            IType_vector const  *v_type = o->get_type();
            size_t              n = v_type->get_size();

            MDL_ASSERT(n <= 4);
            for (size_t i = 0; i < n; ++i) {
                IValue const *tmp = this->divide(factory, o->get_value(i));
                if (is<IValue_bad>(tmp))
                    return tmp;
                values[i] = tmp;
            }
            return factory->create_vector(v_type, values, n);
        }
        return factory->create_bad();
    }

    /// Add.
    IValue const *add(IValue_factory *factory, IValue const *rhs) const MDL_FINAL {
        Kind r_kind = rhs->get_kind();
        if (r_kind == s_kind)
            return factory->create_float(m_value + cast<IValue_float>(rhs)->get_value());
        else if (r_kind == VK_VECTOR) {
            // float + vector
            return rhs->add(factory, this);
        } else if (r_kind == VK_RGB_COLOR) {
            // float + color
            return rhs->add(factory, this);
        } else if (r_kind == VK_MATRIX) {
            // float + matrix
            return rhs->add(factory, this);
        }
        return factory->create_bad();
    }

    /// Subtract.
    IValue const *sub(IValue_factory *factory, IValue const *rhs) const MDL_FINAL {
        Kind r_kind = rhs->get_kind();
        if (r_kind == s_kind)
            return factory->create_float(m_value - cast<IValue_float>(rhs)->get_value());
        else if (r_kind == VK_VECTOR) {
            // float - vector = -vector + float
            return rhs->minus(factory)->add(factory, this);
        } else if (r_kind == VK_RGB_COLOR) {
            // float - color = -color + float
            return rhs->minus(factory)->add(factory, this);
        } else if (r_kind == VK_MATRIX) {
            // float - matrix = -matrix + float
            return rhs->minus(factory)->add(factory, this);
        }
        return factory->create_bad();
    }

    /// Compare.
    IValue::Compare_results compare(IValue const *rhs) const MDL_FINAL {
        IValue::Kind r_kind = rhs->get_kind();
        if (r_kind == s_kind) {
            float other = cast<IValue_float>(rhs)->get_value();
            if (m_value == other)
                return IValue::CR_EQ;
            if (m_value < other)
                return IValue::CR_LT;
            if (m_value > other)
                return IValue::CR_GT;
            return IValue::CR_UO;
        } else if (r_kind == VK_VECTOR) {
            return IValue::inverse(rhs->compare(this));
        }
        return IValue::CR_BAD;
    }

    /// Convert a float value.
    IValue const *convert(IValue_factory *factory, IType const *dst_tp) const MDL_FINAL {
        dst_tp = dst_tp->skip_type_alias();
        switch (dst_tp->get_kind()) {
        case IType::TK_COLOR:
            return factory->create_rgb_color(this, this, this);
        case IType::TK_MATRIX:
            {
                IType_matrix const *x_type = cast<IType_matrix>(dst_tp);
                IType_vector const *v_type = x_type->get_element_type();
                IType_atomic const *a_type = v_type->get_element_type();

                if (a_type == this->m_type) {
                    // convert a float into a diagonal float matrix
                    IValue const *zero = factory->create_float(0.0f);

                    IValue const *column_vals[4];
                    size_t n_cols = x_type->get_columns();
                    size_t n_rows = v_type->get_size();

                    for (size_t col = 0; col < n_cols; ++col) {
                        IValue const *row_vals[4];
                        for (size_t row = 0; row < n_rows; ++row) {
                            row_vals[row] = col == row ? this : zero;
                        }
                        column_vals[col] = factory->create_vector(v_type, row_vals, n_rows);
                    }
                    return factory->create_matrix(x_type, column_vals, n_cols);
                }
            }
            break;
        default:
            break;
        }
        return Base::convert(factory, dst_tp);
    }

    /// Returns true if the value is the ZERO (i.e. additive neutral).
    bool is_zero() const MDL_FINAL {
        // we have +0.0 and -0.0, but both are additive neutral
        return m_value == 0.0f;
    }

    /// Returns true if the value is the ONE (i.e. multiplicative neutral).
    bool is_one() const MDL_FINAL {
        return m_value == 1.0f;
    }

    /// Returns true if all components of this value are ONE.
    bool is_all_one() const MDL_FINAL {
        // float has no components
        return false;
    }

    /// Returns true if this value is finite (i.e. neither Inf nor NaN).
    bool is_finite() const MDL_FINAL {
        return mi::math::isfinite(m_value) != 0;
    }

    /// Return the FP class of this value.
    FP_class get_fp_class() const MDL_FINAL {
        if (m_value == 0.0f)
            return mi::math::sign_bit(m_value) ? FPC_MINUS_ZERO : FPC_PLUS_ZERO;
        if (mi::math::isinfinite(m_value))
            return mi::math::sign_bit(m_value) ? FPC_MINUS_INF : FPC_PLUS_INF;
        if (mi::math::isnan(m_value))
            return FPC_NAN;
        return FPC_NORMAL;
    }

    /// Constructor.
    explicit Value_float(IType_float const *type, float value)
    : Base(type, value)
    {
    }
};

/// Implementation of the IValue_double class.
class Value_double : public Value_atomic<IValue_double, IType_double, double>
{
    typedef Value_atomic<IValue_double, IType_double, double> Base;
public:
    /// Negate this value.
    IValue_double const *minus(IValue_factory *factory) const MDL_FINAL{
        return factory->create_double(-m_value);
    }

    /// Multiply.
    IValue const *multiply(IValue_factory *factory, IValue const *rhs) const MDL_FINAL {
        Kind r_kind = rhs->get_kind();
        if (r_kind == s_kind)
            return factory->create_double(m_value * cast<IValue_double>(rhs)->get_value());
        else if (r_kind == VK_VECTOR)
            return rhs->multiply(factory, this);
        else if (r_kind == VK_MATRIX)
            return rhs->multiply(factory, this);
        return factory->create_bad();
    }

    /// Divide.
    IValue const *divide(IValue_factory *factory, IValue const *rhs) const MDL_FINAL {
        Kind r_kind = rhs->get_kind();
        if (r_kind == s_kind)
            return factory->create_double(m_value / cast<IValue_double>(rhs)->get_value());
        else if (r_kind == VK_VECTOR) {
            // scalar by vector element wise division
            IValue const        *values[4];
            IValue_vector const *o = cast<IValue_vector>(rhs);
            IType_vector const  *v_type = o->get_type();
            size_t              n = v_type->get_size();

            MDL_ASSERT(n <= 4);
            for (size_t i = 0; i < n; ++i) {
                IValue const *tmp = this->divide(factory, o->get_value(i));
                if (is<IValue_bad>(tmp))
                    return tmp;
                values[i] = tmp;
            }
            return factory->create_vector(v_type, values, n);
        }
        return factory->create_bad();
    }

    /// Add.
    IValue const *add(IValue_factory *factory, IValue const *rhs) const MDL_FINAL {
        Kind r_kind = rhs->get_kind();
        if (r_kind == s_kind)
            return factory->create_double(m_value + cast<IValue_double>(rhs)->get_value());
        else if (r_kind == VK_VECTOR)
            return rhs->add(factory, this);
        else if (r_kind == VK_MATRIX)
            return rhs->add(factory, this);
        return factory->create_bad();
    }

    /// Subtract.
    IValue const *sub(IValue_factory *factory, IValue const *rhs) const MDL_FINAL {
        Kind r_kind = rhs->get_kind();
        if (r_kind == s_kind)
            return factory->create_double(m_value - cast<IValue_double>(rhs)->get_value());
        else if (r_kind == VK_VECTOR) {
            // double - vector = (-vector) + double
            return rhs->minus(factory)->add(factory, this);
        } else if (r_kind == VK_MATRIX) {
            // double - matrix = (-matrix) + double
            return rhs->minus(factory)->add(factory, this);
        }
        return factory->create_bad();
    }

    /// Compare.
    IValue::Compare_results compare(IValue const *rhs) const MDL_FINAL {
        IValue::Kind r_kind = rhs->get_kind();
        if (r_kind == s_kind) {
            double other = cast<IValue_double>(rhs)->get_value();
            if (m_value == other)
                return IValue::CR_EQ;
            if (m_value < other)
                return IValue::CR_LT;
            if (m_value > other)
                return IValue::CR_GT;
            return IValue::CR_UO;
        } else if (r_kind == VK_VECTOR) {
            return IValue::inverse(rhs->compare(this));
        }
        return IValue::CR_BAD;
    }

    /// Convert a double value.
    IValue const *convert(IValue_factory *factory, IType const *dst_tp) const MDL_FINAL {
        dst_tp = dst_tp->skip_type_alias();
        if (dst_tp->get_kind() == IType::TK_MATRIX) {
            // convert a double into a diagonal double matrix
            IType_matrix const *x_type = cast<IType_matrix>(dst_tp);
            IType_vector const *v_type = x_type->get_element_type();
            IType_atomic const *a_type = v_type->get_element_type();

            if (a_type == this->m_type) {
                IValue const *zero = factory->create_double(0.0);

                IValue const *column_vals[4];
                size_t n_cols = x_type->get_columns();
                size_t n_rows = v_type->get_size();

                for (size_t col = 0; col < n_cols; ++col) {
                    IValue const *row_vals[4];
                    for (size_t row = 0; row < n_rows; ++row) {
                        row_vals[row] = col == row ? this : zero;
                    }
                    column_vals[col] = factory->create_vector(v_type, row_vals, n_rows);
                }
                return factory->create_matrix(x_type, column_vals, n_cols);
            }
        }
        return Base::convert(factory, dst_tp);
    }

    /// Returns true if the value is the ZERO (i.e. additive neutral).
    bool is_zero() const MDL_FINAL {
        // we have +0.0 and -0.0, but both are additive neutral
        return m_value == 0.0;
    }

    /// Returns true if the value is the ONE (i.e. multiplicative neutral).
    bool is_one() const MDL_FINAL {
        return m_value == 1.0;
    }

    /// Returns true if all components of this value are ONE.
    bool is_all_one() const MDL_FINAL {
        // double has no components
        return false;
    }

    /// Returns true if this value is finite (i.e. neither Inf nor NaN).
    bool is_finite() const MDL_FINAL {
        return mi::math::isfinite(m_value) != 0;
    }

    /// Return the FP class of this value.
    FP_class get_fp_class() const MDL_FINAL {
        if (m_value == 0.0)
            return mi::math::sign_bit(m_value) ? FPC_MINUS_ZERO : FPC_PLUS_ZERO;
        if (mi::math::isinfinite(m_value))
            return mi::math::sign_bit(m_value) ? FPC_MINUS_INF : FPC_PLUS_INF;
        if (mi::math::isnan(m_value))
            return FPC_NAN;
        return FPC_NORMAL;
    }

    /// Constructor.
    explicit Value_double(IType_double const *type, double value)
    : Base(type, value)
    {
    }
};

/// Implementation of an enum value.
class Value_enum : public Value_atomic<IValue_enum, IType_enum, int>
{
    typedef Value_atomic<IValue_enum, IType_enum, int> Base;
public:
    /// Constructor.
    explicit Value_enum(IType_enum const *type, int index)
    : Base(type, index)
    {
    }

    /// Get the index of this enum value.
    size_t get_index() const MDL_FINAL {
        return m_value;
    };

    /// Get the (integer) value of this enum value.
    int get_value() const MDL_FINAL {
        IType_enum const *e_type = get_type();

        ISymbol const *sym;
        int           code = 0;
        bool          res = true;

        res = e_type->get_value(m_value, sym, code);
        MDL_ASSERT(res && "Could not lookup enum value index");

        return code;
    }

    /// Convert an enum value.
    IValue const *convert(IValue_factory *factory, IType const *dst_tp) const MDL_FINAL
    {
        dst_tp = dst_tp->skip_type_alias();
        int v = get_value();
        switch (dst_tp->get_kind()) {
        case IType::TK_BOOL:   return factory->create_bool(v != 0);
        case IType::TK_INT:    return factory->create_int(int(v));
        case IType::TK_FLOAT:  return factory->create_float(float(v));
        case IType::TK_DOUBLE: return factory->create_double(double(v));
        case IType::TK_VECTOR:
            {
                IType_vector const *v_type = cast<IType_vector>(dst_tp);
                IValue const *v = convert(factory, v_type->get_element_type());
                if (is<IValue_bad>(v))
                    break;
                IValue const *args[4] = { v, v, v, v };
                return factory->create_vector(v_type, args, v_type->get_size());
            }
        case IType::TK_ENUM:
            {
                // enum to enum conversion (cast<> operator)
                IType_enum const *e_type = cast<IType_enum>(dst_tp);

                for (int i = 0, n = e_type->get_value_count(); i < n; ++i) {
                    ISymbol const *v_sym;
                    int           v_code = 0;

                    e_type->get_value(i, v_sym, v_code);

                    if (v_code == v)
                        return factory->create_enum(e_type, i);
                }
                // not found
                return factory->create_bad();
            }
        default:
            // no other conversions supported
            break;
        }
        return Base::convert(factory, dst_tp);
    }

    /// Returns true if this value is finite (i.e. neither Inf nor NaN).
    bool is_finite() const MDL_FINAL {
        return true;
    }
};

/// Implementation of the IValue_string class.
class Value_string : public Value_base<IValue_string, IType_string>
{
    typedef Value_base<IValue_string, IType_string> Base;
public:

    /// Get the value.
    const char *get_value() const MDL_FINAL { return m_value; }

    /// Constructor.
    ///
    /// \param arena   the memory arena that will hold the string itself
    /// \param type    the string type
    /// \param value   the string literal value
    explicit Value_string(Memory_arena *arena, IType_string const *type, char const *value)
    : Base(type)
    , m_value(Arena_strdup(*arena, value))
    {
    }

    /// Compare two strings.
    IValue::Compare_results compare(IValue const *rhs) const MDL_FINAL {
        IValue::Kind r_kind = rhs->get_kind();
        if (r_kind == s_kind) {
            char const *other = cast<IValue_string>(rhs)->get_value();
            if (strcmp(m_value, other) == 0)
                return IValue::CR_EQ;
            return IValue::CR_NE;
        }
        return IValue::CR_BAD;
    }

    /// Convert a string value.
    IValue const *convert(IValue_factory *factory, IType const *dst_tp) const MDL_FINAL {
        dst_tp = dst_tp->skip_type_alias();
        switch (dst_tp->get_kind()) {
        case IType::TK_TEXTURE:
            {
                IType_texture const *res_tp = cast<IType_texture>(dst_tp);
                // convert to texture with default gamma
                return factory->create_texture(
                    res_tp,
                    m_value,
                    IValue_texture::gamma_default,
                    /*tag_value=*/0,
                    /*tag_version=*/0);
            }
        case IType::TK_LIGHT_PROFILE:
            {
                IType_light_profile const *res_tp = cast<IType_light_profile>(dst_tp);
                // convert to light profile
                return factory->create_light_profile(
                    res_tp,
                    m_value,
                    /*tag_value=*/0,
                    /*tag_version=*/0);
            }
        case IType::TK_BSDF_MEASUREMENT:
            {
                IType_bsdf_measurement const *res_tp = cast<IType_bsdf_measurement>(dst_tp);
                // convert to bsdf measurement
                return factory->create_bsdf_measurement(
                    res_tp,
                    m_value,
                    /*tag_value=*/0,
                    /*tag_version=*/0);
            }
        default:
            return Base::convert(factory, dst_tp);
        }
    }

    /// Returns true if this value is finite (i.e. neither Inf nor NaN).
    bool is_finite() const MDL_FINAL {
        return true;
    }

private:
    /// The string literal value, stored on a memory arena.
    char const *const m_value;
};

/// Implementation of the IValue_texture class.
class Value_texture : public Value_base<IValue_texture, IType_texture>
{
    typedef Value_base<IValue_texture, IType_texture> Base;
public:

    /// Returns true if this value is finite (i.e. neither Inf nor NaN).
    bool is_finite() const MDL_FINAL {
        return true;
    }

    /// Get the string value.
    char const *get_string_value() const MDL_FINAL { return m_string_value; }

    /// Get the gamma mode.
    gamma_mode get_gamma_mode() const MDL_FINAL { return m_gamma_mode; }

    /// Get the tag value.
    int get_tag_value() const MDL_FINAL { return m_tag_value; }

    /// Get the tag version.
    unsigned get_tag_version() const MDL_FINAL { return m_tag_version; }

    /// Get the BSDF data kind for BSDF data textures.
    Bsdf_data_kind get_bsdf_data_kind() const MDL_FINAL { return m_bsdf_data_kind; }

    /// Constructor.
    explicit Value_texture(
        Memory_arena        *arena,
        IType_texture const *type,
        char const          *value,
        gamma_mode          gamma,
        int                 tag_value,
        unsigned            tag_version)
    : Base(type)
    , m_string_value(Arena_strdup(*arena, value))
    , m_gamma_mode(gamma)
    , m_tag_value(tag_value)
    , m_tag_version(tag_version)
    , m_bsdf_data_kind(BDK_NONE)
    {
    }

    /// Constructor for bsdf data texture.
    explicit Value_texture(
        Memory_arena        *arena,
        IType_texture const *type,
        Bsdf_data_kind      kind,
        int                 tag_value,
        unsigned            tag_version)
    : Base(type)
    , m_string_value("")
    , m_gamma_mode(gamma_linear)
    , m_tag_value(tag_value)
    , m_tag_version(tag_version)
    , m_bsdf_data_kind(kind)
    {
    }

private:
    /// The string value.
    char const *const m_string_value;
    /// The gamma mode.
    gamma_mode m_gamma_mode;
    /// The tag value.
    int const m_tag_value;
    /// The tag version.
    unsigned m_tag_version;
    /// The BSDF data kind.
    Bsdf_data_kind m_bsdf_data_kind;
};

/// Implementation of the "string" valued IValue_light_profile class.
class Value_light_profile : public Value_base<IValue_light_profile, IType_light_profile>
{
    typedef Value_base<IValue_light_profile, IType_light_profile> Base;
public:

    /// Returns true if this value is finite (i.e. neither Inf nor NaN).
    bool is_finite() const MDL_FINAL {
        return true;
    }

    /// Get the string value.
    char const *get_string_value() const MDL_FINAL { return m_string_value; }

    /// Get the tag value.
    int get_tag_value() const MDL_FINAL { return m_tag_value; }

    /// Get the tag version.
    unsigned get_tag_version() const MDL_FINAL { return m_tag_version; }

    /// Constructor.
    explicit Value_light_profile(
        Memory_arena              *arena,
        IType_light_profile const *type,
        char const                *value,
        int                       tag_value,
        unsigned                  tag_version)
    : Base(type)
    , m_string_value(Arena_strdup(*arena, value))
    , m_tag_value(tag_value)
    , m_tag_version(tag_version)
    {
    }

private:
    /// The string value.
    char const *const m_string_value;
    /// The tag value.
    int const m_tag_value;
    /// The tag version.
    unsigned const m_tag_version;
};

/// Implementation of the "string" valued IValue_bsdf_measurement class.
class Value_bsdf_measurement : public Value_base<IValue_bsdf_measurement, IType_bsdf_measurement>
{
    typedef Value_base<IValue_bsdf_measurement, IType_bsdf_measurement> Base;
public:
    /// Returns true if this value is finite (i.e. neither Inf nor NaN).
    bool is_finite() const MDL_FINAL {
        return true;
    }

    /// Get the string value.
    char const *get_string_value() const MDL_FINAL { return m_string_value; }

    /// Get the tag value.
    int get_tag_value() const MDL_FINAL { return m_tag_value; }

    /// Get the tag version.
    unsigned get_tag_version() const MDL_FINAL { return m_tag_version; }

    /// Constructor.
    explicit Value_bsdf_measurement(
        Memory_arena                 *arena,
        IType_bsdf_measurement const *type,
        char const                   *value,
        int                          tag_value,
        unsigned                     tag_version)
    : Base(type)
    , m_string_value(Arena_strdup(*arena, value))
    , m_tag_value(tag_value)
    , m_tag_version(tag_version)
    {
    }

private:
    /// The string value.
    char const *const m_string_value;
    /// The tag value.
    int const m_tag_value;
    /// The tag version.
    unsigned const m_tag_version;
};

/// A mixin base class for all compound value methods.
///
/// \tparam Interface   the base class for the desired IValue
/// \tparam Type        the type of all IValues of kind Interface
/// \tparam Value_type  the type of the child values
template<typename Interface, typename Type, typename Value_type = IValue>
class Value_compound : public Value_base<Interface, Type>
{
    typedef Value_base<Interface, Type> Base;
public:
    /// Get the number of components in this compound value.
    int get_component_count() const MDL_FINAL { return m_values.size(); }

    /// Get the value at index.
    ///
    /// \param index  the index
    Value_type const *get_value(int index) const MDL_FINAL {
        MDL_ASSERT(0 <= index && size_t(index) < m_values.size() && "Index out of bounds");
        return m_values[index];
    }

    /// Get the value by its MDL (field) name.
    ///
    /// \param name  the (field) name
    ///
    /// \return NULL if name does not exists, else return the corresponding value
    IValue const *get_value(char const *name) const MDL_OVERRIDE {
        return NULL;
    }

    /// Return an the array of values.
    IValue const * const * get_values() const MDL_FINAL {
        // contra-variance does not work on arrays
        return (IValue const * const *)m_values.data();
    }

    /// Extract from a compound.
    IValue const *extract(IValue_factory *factory, int index) const MDL_FINAL {
        if (0 <= index && size_t(index) < m_values.size())
            return m_values[index];
        return factory->create_bad();
    }

    /// Returns true if this value is finite (i.e. neither Inf nor NaN).
    bool is_finite() const MDL_FINAL {
        for (size_t i = 0, n = m_values.size(); i < n; ++i) {
            if (!m_values[i]->is_finite())
                return false;
        }
        return true;
    }

protected:
    /// Constructor.
    explicit Value_compound(
        Memory_arena             *arena,
        Type const               *type,
        Value_type const * const values[],
        size_t                   size)
    : Base(type)
    , m_values(*arena, size)
    {
        for (size_t i = 0; i < size; ++i)
            m_values[i] = values[i];
    }

    explicit Value_compound(
        Memory_arena *arena,
        Type const   *type,
        Value_type const *value_0,
        Value_type const *value_1,
        Value_type const *value_2)
    : Base(type)
    , m_values(*arena, 3)
    {
        m_values[0] = value_0;
        m_values[1] = value_1;
        m_values[2] = value_2;
    }
    /// The compound values.
    Arena_VLA<Value_type const *> m_values;
};

/// Implementation of the IValue_vector interface.
class Value_vector : public Value_compound<IValue_vector, IType_vector>
{
    typedef Value_compound<IValue_vector, IType_vector> Base;
public:
    /// Negate this value.
    IValue const *minus(IValue_factory *factory) const MDL_FINAL {
        IValue const *values[4];
        size_t       n = m_values.size();

        MDL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            IValue const *tmp = m_values[i]->minus(factory);
            if (is<IValue_bad>(tmp))
                return tmp;
            values[i] = tmp;
        }
        return factory->create_vector(m_type, values, n);
    }

    /// Bitwise negate this value.
    IValue const *bitwise_not(IValue_factory *factory) const MDL_FINAL {
        IValue const *values[4];
        size_t       n = m_values.size();

        MDL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            IValue const *tmp = m_values[i]->bitwise_not(factory);
            if (is<IValue_bad>(tmp))
                return tmp;
            values[i] = tmp;
        }
        return factory->create_vector(m_type, values, n);
    }

    /// Logically negate this value.
    IValue const *logical_not(IValue_factory *factory) const MDL_FINAL {
        IValue const *values[4];
        size_t       n = m_values.size();

        MDL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            IValue const *tmp = m_values[i]->logical_not(factory);
            if (is<IValue_bad>(tmp))
                return tmp;
            values[i] = tmp;
        }
        return factory->create_vector(m_type, values, n);
    }

    /// Multiply.
    IValue const *multiply(IValue_factory *factory, IValue const *rhs) const MDL_FINAL {
        IType const *rhs_type = rhs->get_type();
        if (rhs_type == m_type) {
            // vector by vector element wise multiplication
            IValue const        *values[4];
            size_t               n = m_values.size();
            IValue_vector const *o = cast<IValue_vector>(rhs);

            MDL_ASSERT(n <= 4);
            for (size_t i = 0; i < n; ++i) {
                IValue const *tmp = m_values[i]->multiply(factory, o->get_value(i));
                if (is<IValue_bad>(tmp))
                    return tmp;
                values[i] = tmp;
            }
            return factory->create_vector(m_type, values, n);
        } else if (rhs_type == m_type->get_element_type()) {
            // vector by scalar element wise multiplication
            IValue const        *values[4];
            size_t               n = m_values.size();

            MDL_ASSERT(n <= 4);
            for (size_t i = 0; i < n; ++i) {
                IValue const *tmp = m_values[i]->multiply(factory, rhs);
                if (is<IValue_bad>(tmp))
                    return tmp;
                values[i] = tmp;
            }
            return factory->create_vector(m_type, values, n);
        } else if (IType_matrix const *x_type = as<IType_matrix>(rhs_type)) {
            // vector by matrix multiplication
            IType_vector const *v_type = x_type->get_element_type();
            size_t              n_rows  = v_type->get_size();
            if (m_type->get_size() == n_rows) {
                size_t              n_cols  = x_type->get_columns();
                IValue_matrix const *o      = cast<IValue_matrix>(rhs);
                IType_factory       *t_fact = factory->get_type_factory();
                IType_atomic const  *a_type = m_type->get_element_type();
                IType_vector const  *r_type = t_fact->create_vector(a_type, n_cols);

                IValue const *values[4];
                for (size_t col = 0; col < n_cols; ++col) {
                    IValue const        *e   = m_values[0];
                    IValue_vector const *r   = cast<IValue_vector>(o->get_value(col));
                    IValue const        *tmp = e->multiply(factory, r->get_value(0));
                    for (size_t row = 1; row < n_rows; ++row) {
                        e   = m_values[row];
                        tmp = tmp->add(factory, e->multiply(factory, r->get_value(row)));
                    }
                    if (is<IValue_bad>(tmp))
                        return tmp;
                    values[col] = tmp;
                }
                return factory->create_vector(r_type, values, n_cols);
            }
            return factory->create_bad();
        }
        return factory->create_bad();
    }

    /// Divide.
    IValue const *divide(IValue_factory *factory, IValue const *rhs) const MDL_FINAL {
        IType const *rhs_type = rhs->get_type();
        if (rhs_type == m_type) {
            // vector by vector element wise division
            IValue const        *values[4];
            size_t               n = m_values.size();
            IValue_vector const *o = cast<IValue_vector>(rhs);

            MDL_ASSERT(n <= 4);
            for (size_t i = 0; i < n; ++i) {
                IValue const *tmp = m_values[i]->divide(factory, o->get_value(i));
                if (is<IValue_bad>(tmp))
                    return tmp;
                values[i] = tmp;
            }
            return factory->create_vector(m_type, values, n);
        } else if (rhs_type == m_type->get_element_type()) {
            // vector by scalar element wise division
            IValue const        *values[4];
            size_t               n = m_values.size();

            MDL_ASSERT(n <= 4);
            for (size_t i = 0; i < n; ++i) {
                IValue const *tmp = m_values[i]->divide(factory, rhs);
                if (is<IValue_bad>(tmp))
                    return tmp;
                values[i] = tmp;
            }
            return factory->create_vector(m_type, values, n);
        }
        return factory->create_bad();
    }

    /// Modulo.
    IValue const *modulo(IValue_factory *factory, IValue const *rhs) const MDL_FINAL {
        IType const *rhs_type = rhs->get_type();
        if (rhs_type == m_type) {
            // vector by vector element wise modulo
            IValue const        *values[4];
            size_t               n = m_values.size();
            IValue_vector const *o = cast<IValue_vector>(rhs);

            MDL_ASSERT(n <= 4);
            for (size_t i = 0; i < n; ++i) {
                IValue const *tmp = m_values[i]->modulo(factory, o->get_value(i));
                if (is<IValue_bad>(tmp))
                    return tmp;
                values[i] = tmp;
            }
            return factory->create_vector(m_type, values, n);
        } else if (rhs_type == m_type->get_element_type()) {
            // vector by scalar element wise modulo
            IValue const        *values[4];
            size_t               n = m_values.size();

            MDL_ASSERT(n <= 4);
            for (size_t i = 0; i < n; ++i) {
                IValue const *tmp = m_values[i]->modulo(factory, rhs);
                if (is<IValue_bad>(tmp))
                    return tmp;
                values[i] = tmp;
            }
            return factory->create_vector(m_type, values, n);
        }
        return factory->create_bad();
    }

    /// Add.
    IValue const *add(IValue_factory *factory, IValue const *rhs) const MDL_FINAL {
        IType const *rhs_type = rhs->get_type();
        if (rhs_type == m_type) {
            // vector by vector element wise addition
            IValue const        *values[4];
            size_t               n = m_values.size();
            IValue_vector const *o = cast<IValue_vector>(rhs);

            MDL_ASSERT(n <= 4);
            for (size_t i = 0; i < n; ++i) {
                IValue const *tmp = m_values[i]->add(factory, o->get_value(i));
                if (is<IValue_bad>(tmp))
                    return tmp;
                values[i] = tmp;
            }
            return factory->create_vector(m_type, values, n);
        } else if (rhs_type == m_type->get_element_type()) {
            // vector by scalar element wise addition
            IValue const        *values[4];
            size_t               n = m_values.size();

            MDL_ASSERT(n <= 4);
            for (size_t i = 0; i < n; ++i) {
                IValue const *tmp = m_values[i]->add(factory, rhs);
                if (is<IValue_bad>(tmp))
                    return tmp;
                values[i] = tmp;
            }
            return factory->create_vector(m_type, values, n);
        }
        return factory->create_bad();
    }

    /// Subtract.
    IValue const *sub(IValue_factory *factory, IValue const *rhs) const MDL_FINAL {
        IType const *rhs_type = rhs->get_type();
        if (rhs_type == m_type) {
            // vector by vector element wise subtraction
            IValue const        *values[4];
            size_t               n = m_values.size();
            IValue_vector const *o = cast<IValue_vector>(rhs);

            MDL_ASSERT(n <= 4);
            for (size_t i = 0; i < n; ++i) {
                IValue const *tmp = m_values[i]->sub(factory, o->get_value(i));
                if (is<IValue_bad>(tmp))
                    return tmp;
                values[i] = tmp;
            }
            return factory->create_vector(m_type, values, n);
        } else if (rhs_type == m_type->get_element_type()) {
            // vector by scalar element wise subtraction
            IValue const        *values[4];
            size_t               n = m_values.size();

            MDL_ASSERT(n <= 4);
            for (size_t i = 0; i < n; ++i) {
                IValue const *tmp = m_values[i]->sub(factory, rhs);
                if (is<IValue_bad>(tmp))
                    return tmp;
                values[i] = tmp;
            }
            return factory->create_vector(m_type, values, n);
        }
        return factory->create_bad();
    }

    /// Shift left.
    IValue const *shl(IValue_factory *factory, IValue const *rhs) const MDL_FINAL {
        IType const *rhs_type = rhs->get_type();
        if (rhs_type == m_type) {
            // vector by vector element wise shift left
            IValue const        *values[4];
            size_t               n = m_values.size();
            IValue_vector const *o = cast<IValue_vector>(rhs);

            MDL_ASSERT(n <= 4);
            for (size_t i = 0; i < n; ++i) {
                IValue const *tmp = m_values[i]->shl(factory, o->get_value(i));
                if (is<IValue_bad>(tmp))
                    return tmp;
                values[i] = tmp;
            }
            return factory->create_vector(m_type, values, n);
        } else if (rhs_type == m_type->get_element_type()) {
            // vector by scalar element wise shift left
            IValue const        *values[4];
            size_t               n = m_values.size();

            MDL_ASSERT(n <= 4);
            for (size_t i = 0; i < n; ++i) {
                IValue const *tmp = m_values[i]->shl(factory, rhs);
                if (is<IValue_bad>(tmp))
                    return tmp;
                values[i] = tmp;
            }
            return factory->create_vector(m_type, values, n);
        }
        return factory->create_bad();
    }

    /// Arithmetic shift right.
    IValue const *asr(IValue_factory *factory, IValue const *rhs) const MDL_FINAL {
        IType const *rhs_type = rhs->get_type();
        if (rhs_type == m_type) {
            // vector by vector element wise arithmetic shift right
            IValue const        *values[4];
            size_t               n = m_values.size();
            IValue_vector const *o = cast<IValue_vector>(rhs);

            MDL_ASSERT(n <= 4);
            for (size_t i = 0; i < n; ++i) {
                IValue const *tmp = m_values[i]->asr(factory, o->get_value(i));
                if (is<IValue_bad>(tmp))
                    return tmp;
                values[i] = tmp;
            }
            return factory->create_vector(m_type, values, n);
        } else if (rhs_type == m_type->get_element_type()) {
            // vector by scalar element wise arithmetic shift right
            IValue const        *values[4];
            size_t               n = m_values.size();

            MDL_ASSERT(n <= 4);
            for (size_t i = 0; i < n; ++i) {
                IValue const *tmp = m_values[i]->asr(factory, rhs);
                if (is<IValue_bad>(tmp))
                    return tmp;
                values[i] = tmp;
            }
            return factory->create_vector(m_type, values, n);
        }
        return factory->create_bad();
    }

    /// Logical shift right.
    IValue const *lsr(IValue_factory *factory, IValue const *rhs) const MDL_FINAL {
        IType const *rhs_type = rhs->get_type();
        if (rhs_type == m_type) {
            // vector by vector element wise logical shift right
            IValue const        *values[4];
            size_t               n = m_values.size();
            IValue_vector const *o = cast<IValue_vector>(rhs);

            MDL_ASSERT(n <= 4);
            for (size_t i = 0; i < n; ++i) {
                IValue const *tmp = m_values[i]->lsr(factory, o->get_value(i));
                if (is<IValue_bad>(tmp))
                    return tmp;
                values[i] = tmp;
            }
            return factory->create_vector(m_type, values, n);
        } else if (rhs_type == m_type->get_element_type()) {
            // vector by scalar element wise arithmetic shift right
            IValue const        *values[4];
            size_t               n = m_values.size();

            MDL_ASSERT(n <= 4);
            for (size_t i = 0; i < n; ++i) {
                IValue const *tmp = m_values[i]->lsr(factory, rhs);
                if (is<IValue_bad>(tmp))
                    return tmp;
                values[i] = tmp;
            }
            return factory->create_vector(m_type, values, n);
        }
        return factory->create_bad();
    }

    /// Xor.
    IValue const *bitwise_xor(IValue_factory *factory, IValue const *rhs) const MDL_FINAL {
        IType const *rhs_type = rhs->get_type();
        if (rhs_type == m_type) {
            // vector by vector element wise exclusive or
            IValue const        *values[4];
            size_t               n = m_values.size();
            IValue_vector const *o = cast<IValue_vector>(rhs);

            MDL_ASSERT(n <= 4);
            for (size_t i = 0; i < n; ++i) {
                IValue const *tmp = m_values[i]->bitwise_xor(factory, o->get_value(i));
                if (is<IValue_bad>(tmp))
                    return tmp;
                values[i] = tmp;
            }
            return factory->create_vector(m_type, values, n);
        } else if (rhs_type == m_type->get_element_type()) {
            // vector by scalar element wise exclusive or
            IValue const        *values[4];
            size_t               n = m_values.size();

            MDL_ASSERT(n <= 4);
            for (size_t i = 0; i < n; ++i) {
                IValue const *tmp = m_values[i]->bitwise_xor(factory, rhs);
                if (is<IValue_bad>(tmp))
                    return tmp;
                values[i] = tmp;
            }
            return factory->create_vector(m_type, values, n);
        }
        return factory->create_bad();
    }

    /// Or.
    IValue const *bitwise_or(IValue_factory *factory, IValue const *rhs) const MDL_FINAL {
        IType const *rhs_type = rhs->get_type();
        if (rhs_type == m_type) {
            // vector by vector element wise or
            IValue const        *values[4];
            size_t               n = m_values.size();
            IValue_vector const *o = cast<IValue_vector>(rhs);

            MDL_ASSERT(n <= 4);
            for (size_t i = 0; i < n; ++i) {
                IValue const *tmp = m_values[i]->bitwise_or(factory, o->get_value(i));
                if (is<IValue_bad>(tmp))
                    return tmp;
                values[i] = tmp;
            }
            return factory->create_vector(m_type, values, n);
        } else if (rhs_type == m_type->get_element_type()) {
            // vector by scalar element wise or
            IValue const        *values[4];
            size_t               n = m_values.size();

            MDL_ASSERT(n <= 4);
            for (size_t i = 0; i < n; ++i) {
                IValue const *tmp = m_values[i]->bitwise_or(factory, rhs);
                if (is<IValue_bad>(tmp))
                    return tmp;
                values[i] = tmp;
            }
            return factory->create_vector(m_type, values, n);
        }
        return factory->create_bad();
    }

    /// And.
    IValue const *bitwise_and(IValue_factory *factory, IValue const *rhs) const MDL_FINAL {
        IType const *rhs_type = rhs->get_type();
        if (rhs_type == m_type) {
            // vector by vector element wise and
            IValue const        *values[4];
            size_t               n = m_values.size();
            IValue_vector const *o = cast<IValue_vector>(rhs);

            MDL_ASSERT(n <= 4);
            for (size_t i = 0; i < n; ++i) {
                IValue const *tmp = m_values[i]->bitwise_and(factory, o->get_value(i));
                if (is<IValue_bad>(tmp))
                    return tmp;
                values[i] = tmp;
            }
            return factory->create_vector(m_type, values, n);
        } else if (rhs_type == m_type->get_element_type()) {
            // vector by scalar element wise and
            IValue const        *values[4];
            size_t               n = m_values.size();

            MDL_ASSERT(n <= 4);
            for (size_t i = 0; i < n; ++i) {
                IValue const *tmp = m_values[i]->bitwise_and(factory, rhs);
                if (is<IValue_bad>(tmp))
                    return tmp;
                values[i] = tmp;
            }
            return factory->create_vector(m_type, values, n);
        }
        return factory->create_bad();
    }

    /// Logical Or.
    IValue const *logical_or(IValue_factory *factory, IValue const *rhs) const MDL_FINAL {
        IType const *rhs_type = rhs->get_type();
        if (rhs_type == m_type) {
            // vector element wise logical or
            IValue const        *values[4];
            size_t              n = m_values.size();
            IValue_vector const *o = cast<IValue_vector>(rhs);

            MDL_ASSERT(n <= 4);
            for (size_t i = 0; i < n; ++i) {
                IValue const *tmp = m_values[i]->logical_or(factory, o->get_value(i));
                if (is<IValue_bad>(tmp))
                    return tmp;
                values[i] = tmp;
            }
            return factory->create_vector(m_type, values, n);
        } else if (rhs_type == m_type->get_element_type()) {
            // vector by scalar element wise logical or
            IValue const        *values[4];
            size_t               n = m_values.size();

            MDL_ASSERT(n <= 4);
            for (size_t i = 0; i < n; ++i) {
                IValue const *tmp = m_values[i]->logical_or(factory, rhs);
                if (is<IValue_bad>(tmp))
                    return tmp;
                values[i] = tmp;
            }
            return factory->create_vector(m_type, values, n);
        }
        return factory->create_bad();
    }

    /// Logical And.
    IValue const *logical_and(IValue_factory *factory, IValue const *rhs) const MDL_FINAL {
        IType const *rhs_type = rhs->get_type();
        if (rhs_type == m_type) {
            // vector element wise logical and
            IValue const        *values[4];
            size_t              n = m_values.size();
            IValue_vector const *o = cast<IValue_vector>(rhs);

            MDL_ASSERT(n <= 4);
            for (size_t i = 0; i < n; ++i) {
                IValue const *tmp = m_values[i]->logical_and(factory, o->get_value(i));
                if (is<IValue_bad>(tmp))
                    return tmp;
                values[i] = tmp;
            }
            return factory->create_vector(m_type, values, n);
        } else if (rhs_type == m_type->get_element_type()) {
            // vector by scalar element wise logical and
            IValue const        *values[4];
            size_t               n = m_values.size();

            MDL_ASSERT(n <= 4);
            for (size_t i = 0; i < n; ++i) {
                IValue const *tmp = m_values[i]->logical_and(factory, rhs);
                if (is<IValue_bad>(tmp))
                    return tmp;
                values[i] = tmp;
            }
            return factory->create_vector(m_type, values, n);
        }
        return factory->create_bad();
    }

    /// Compare.
    IValue::Compare_results compare(IValue const *rhs) const MDL_FINAL {
        IType const *rhs_type = rhs->get_type();
        if (rhs_type == m_type) {
            // vector by vector compare
            size_t               n = m_values.size();
            IValue_vector const *o = cast<IValue_vector>(rhs);

            IValue::Compare_results res = IValue::CR_EQ;
            for (size_t i = 0; i < n; ++i) {
                unsigned tmp = m_values[i]->compare(o->get_value(i));
                if (tmp & IValue::CR_UO)
                    return IValue::CR_UO;
                if ((tmp & IValue::CR_EQ) == 0)
                    res = IValue::CR_NE;
            }
            return res; 
        } else if (rhs_type == m_type->get_element_type()) {
            // vector by scalar element wise compare
            size_t               n = m_values.size();

            IValue::Compare_results res = IValue::CR_EQ;
            for (size_t i = 0; i < n; ++i) {
                unsigned tmp = m_values[i]->compare(rhs);
                if (tmp & IValue::CR_UO)
                    return IValue::CR_UO;
                if ((tmp & IValue::CR_EQ) == 0)
                    res = IValue::CR_NE;
            }
            return res; 
        }
        return IValue::CR_BAD;
    }

    /// Convert a vector value.
    IValue const *convert(IValue_factory *factory, IType const *dst_tp) const MDL_FINAL {
        dst_tp = dst_tp->skip_type_alias();
        switch (dst_tp->get_kind()) {
        case IType::TK_VECTOR:
            {
                size_t             n = m_type->get_size();
                IType_vector const *v_type = cast<IType_vector>(dst_tp);

                if (v_type->get_size() == n) {
                    IType_atomic const *a_type = v_type->get_element_type();
                    IValue const       *values[4];

                    for (size_t i = 0; i < n; ++i) {
                        IValue const *tmp = m_values[i]->convert(factory, a_type);
                        if (is<IValue_bad>(tmp))
                            return tmp;
                        values[i] = tmp;
                    }
                    return factory->create_vector(v_type, values, n);
                }
                break;
            }
        case IType::TK_COLOR:
            if (m_type->get_size() == 3) {
                IType_atomic const *e_type = m_type->get_element_type();
                if (e_type->get_kind() == IType::TK_FLOAT) {
                    // float3 can be converted to color
                    return factory->create_rgb_color(
                        cast<IValue_float>(m_values[0]),
                        cast<IValue_float>(m_values[1]),
                        cast<IValue_float>(m_values[2]));
                }
            }
            break;
        default:
            break;
        }
        return Base::convert(factory, dst_tp);
    }

    /// Returns true if the value is the ZERO (i.e. additive neutral).
    bool is_zero() const MDL_FINAL {
        bool   neutral = true;
        size_t n = m_values.size();

        for (size_t i = 0; i < n; ++i) {
            neutral &= m_values[i]->is_zero();
        }
        return neutral;
    }

    /// Returns true if the value is the ONE (i.e. multiplicative neutral).
    bool is_one() const MDL_FINAL {
        bool   all_one = true;
        size_t n = m_values.size();

        for (size_t i = 0; i < n; ++i) {
            all_one &= m_values[i]->is_one();
        }
        return all_one;
    }

    /// Returns true if all components of this value are ONE.
    bool is_all_one() const MDL_FINAL {
        bool   all_all_one = true;
        size_t n = m_values.size();

        for (size_t i = 0; i < n; ++i) {
            all_all_one &= m_values[i]->is_all_one();
        }
        return all_all_one;
    }

    /// Get the value by its MDL (field) name.
    ///
    /// \param name  the (field) name
    ///
    /// \return NULL if name does not exists, else return the corresponding value
    IValue const *get_value(char const *name) const MDL_FINAL {
        int index = -1;

        if (name != NULL && name[0] != '\0' && name[1] == '\0') {
            switch (name[0]) {
            case 'x': index = 0; break;
            case 'y': index = 1; break;
            case 'z': index = 2; break;
            case 'w': index = 3; break;
            }
        }
        if (index < 0 || size_t(index) >= m_values.size())
            return NULL;
        return m_values[index];
    }

    /// Constructor.
    explicit Value_vector(
        Memory_arena         *arena,
        IType_vector const   *type,
        IValue const * const values[],
        size_t               size)
    : Base(arena, type, values, size)
    {
        MDL_ASSERT(size == type->get_size());
    }
};

/// Implementation of the IValue_matrix interface.
class Value_matrix : public Value_compound<IValue_matrix, IType_matrix>
{
    typedef Value_compound<IValue_matrix, IType_matrix> Base;
public:
    /// Negate this value.
    IValue const *minus(IValue_factory *factory) const MDL_FINAL {
        IValue const *values[4];
        size_t       n = m_values.size();

        MDL_ASSERT(n <= 4);
        for (size_t i = 0; i < n; ++i) {
            IValue const *tmp = m_values[i]->minus(factory);
            if (is<IValue_bad>(tmp))
                return tmp;
            values[i] = tmp;
        }
        return factory->create_matrix(m_type, values, n);
    }

    /// Matrix multiplication.
    IValue const *multiply(IValue_factory *factory, IValue const *rhs) const MDL_FINAL {
        IType const *rhs_type = rhs->get_type();

        if (IType_matrix const *o_type = as<IType_matrix>(rhs_type)) {
            // matrix by matrix MxN * KxM ==> KxN
            size_t const M = m_type->get_columns();
            if (M != o_type->get_element_type()->get_size())
                return factory->create_bad();

            IType_vector const *v_type = m_type->get_element_type();

            size_t const N = v_type->get_size();
            size_t const K = o_type->get_columns();

            IValue_matrix const *o = cast<IValue_matrix>(rhs);
            IType_factory       *t_fact = factory->get_type_factory();

            // the vector type of the result is a a vector of length N
            v_type = t_fact->create_vector(v_type->get_element_type(), N);

            IValue const *columns[4];
            for (size_t col = 0; col < K; ++col) {
                IValue const *column[4];
                for (size_t row = 0; row < N; ++row) {
                    IValue const        *l  = cast<IValue_vector>(m_values[0])->get_value(row);
                    IValue_vector const *rv = cast<IValue_vector>(o->get_value(col));
                    IValue const        *tmp = l->multiply(factory, rv->get_value(0));

                    for (size_t m = 1; m < M; ++m) {
                        l = cast<IValue_vector>(m_values[m])->get_value(row);
                        tmp = tmp->add(factory, l->multiply(factory, rv->get_value(m)));
                    }
                    if (is<IValue_bad>(tmp))
                        return tmp;
                    column[row] = tmp;
                }
                columns[col] = factory->create_vector(v_type, column, N);
            }
            IType_matrix const *r_type = t_fact->create_matrix(v_type, K);
            return factory->create_matrix(r_type, columns, K);
        }
        if (rhs_type == m_type->get_element_type()->get_element_type()) {
            // matrix by scalar element wise multiplication
            IValue const        *values[4];
            size_t               n = m_values.size();

            MDL_ASSERT(n <= 4);
            for (size_t i = 0; i < n; ++i) {
                IValue const *tmp = m_values[i]->multiply(factory, rhs);
                if (is<IValue_bad>(tmp))
                    return tmp;
                values[i] = tmp;
            }
            return factory->create_matrix(m_type, values, n);
        }
        if (IType_vector const *v_type = as<IType_vector>(rhs_type)) {
            // matrix by vector
            size_t n_cols = m_type->get_columns();

            if (v_type->get_size() == n_cols) {
                IType_vector const  *v_type = m_type->get_element_type();
                size_t              n_rows  = v_type->get_size();
                IValue_vector const *o      = cast<IValue_vector>(rhs);

                IValue const *values[4];
                for (size_t row = 0; row < n_rows; ++row) {
                    IValue const *e   = cast<IValue_vector>(m_values[0])->get_value(row);
                    IValue const *r   = o->get_value(0);
                    IValue const *tmp = e->multiply(factory, r);
                    for (size_t col = 1; col < n_cols; ++col) {
                        e = cast<IValue_vector>(m_values[col])->get_value(row);
                        r = o->get_value(col);
                        e = e->multiply(factory, r);
                        tmp = tmp->add(factory, e);
                    }
                    if (is<IValue_bad>(tmp))
                        return tmp;
                    values[row] = tmp;
                }
                return factory->create_vector(m_type->get_element_type(), values, n_rows);
            }
            return factory->create_bad();
        }
        return factory->create_bad();
    }

    /// Divide.
    IValue const *divide(IValue_factory *factory, IValue const *rhs) const MDL_FINAL {
        IType const *rhs_type = rhs->get_type();
        if (rhs_type == m_type->get_element_type()->get_element_type()) {
            // matrix by scalar element wise division
            IValue const        *values[4];
            size_t               n = m_values.size();

            MDL_ASSERT(n <= 4);
            for (size_t i = 0; i < n; ++i) {
                IValue const *tmp = m_values[i]->divide(factory, rhs);
                if (is<IValue_bad>(tmp))
                    return tmp;
                values[i] = tmp;
            }
            return factory->create_matrix(m_type, values, n);
        }
        return factory->create_bad();
    }

    /// Add.
    IValue const *add(IValue_factory *factory, IValue const *rhs) const MDL_FINAL {
        IType const *rhs_type = rhs->get_type();
        if (rhs_type == m_type) {
            // matrix element wise addition
            IValue const        *values[4];
            size_t               n = m_values.size();
            IValue_matrix const *o = cast<IValue_matrix>(rhs);
            
            MDL_ASSERT(n <= 4);
            for (size_t i = 0; i < n; ++i) {
                IValue const *tmp = m_values[i]->add(factory, o->get_value(i));
                if (is<IValue_bad>(tmp))
                    return tmp;
                values[i] = tmp;
            }
            return factory->create_matrix(m_type, values, n);
        }
        return factory->create_bad();
    }

    /// Subtract.
    IValue const *sub(IValue_factory *factory, IValue const *rhs) const MDL_FINAL {
        IType const *rhs_type = rhs->get_type();
        if (rhs_type == m_type) {
            // matrix element wise subtraction
            IValue const        *values[4];
            size_t               n = m_values.size();
            IValue_matrix const *o = cast<IValue_matrix>(rhs);

            MDL_ASSERT(n <= 4);
            for (size_t i = 0; i < n; ++i) {
                IValue const *tmp = m_values[i]->sub(factory, o->get_value(i));
                if (is<IValue_bad>(tmp))
                    return tmp;
                values[i] = tmp;
            }
            return factory->create_matrix(m_type, values, n);
        }
        return factory->create_bad();
    }

    /// Compare.
    IValue::Compare_results compare(IValue const *rhs) const MDL_FINAL {
        if (rhs->get_kind() == s_kind) {
            // matrix by matrix element wise compare
            size_t               n = m_values.size();
            IValue_matrix const *o = cast<IValue_matrix>(rhs);

            IValue::Compare_results res = IValue::CR_EQ;
            for (size_t i = 0; i < n; ++i) {
                unsigned tmp = m_values[i]->compare(o->get_value(i));
                if (tmp & IValue::CR_UO)
                    return IValue::CR_UO;
                if ((tmp & IValue::CR_EQ) == 0)
                    res = IValue::CR_NE;
            }
            return res;
        }
        return IValue::CR_BAD;
    }

    /// Convert a matrix value.
    IValue const *convert(IValue_factory *factory, IType const *dst_tp) const MDL_FINAL {
        dst_tp = dst_tp->skip_type_alias();
        switch (dst_tp->get_kind()) {
        case IType::TK_MATRIX:
            {
                size_t             n_cols = m_type->get_columns();
                IType_matrix const *x_type = cast<IType_matrix>(dst_tp);

                if (x_type->get_columns() == n_cols) {
                    IType_vector const *v_type = x_type->get_element_type();
                    IValue const       *values[4];

                    for (size_t i = 0; i < n_cols; ++i) {
                        IValue const *tmp = m_values[i]->convert(factory, v_type);
                        if (is<IValue_bad>(tmp))
                            return tmp;
                        values[i] = tmp;
                    }
                    return factory->create_matrix(x_type, values, n_cols);
                }
                break;
            }
        default:
            break;
        }
        return Base::convert(factory, dst_tp);
    }

    /// Returns true if the value is the ZERO (i.e. additive neutral).
    bool is_zero() const MDL_FINAL {
        bool   neutral = true;
        size_t n = m_values.size();

        for (size_t i = 0; i < n; ++i) {
            neutral &= m_values[i]->is_zero();
        }
        return neutral;
    }

    /// Returns true if the value is the identity matrix (i.e. multiplicative neutral).
    bool is_one() const MDL_FINAL {
        size_t n = m_values.size();
        IType_vector const *e_type = m_type->get_element_type();

        if (e_type->get_size() == n) {
            // square matrix

            for (size_t i = 0; i < n; ++i) {
                IValue_vector const *vec = cast<IValue_vector>(m_values[i]);

                for (size_t j = 0; j < n; ++j) {
                    IValue const *v = vec->get_value(j);
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

    /// Returns true if all components of this value are ONE.
    bool is_all_one() const MDL_FINAL {
        bool   all_one = true;
        size_t n = m_values.size();

        for (size_t i = 0; i < n; ++i) {
            all_one &= m_values[i]->is_all_one();
        }
        return all_one;
    }

    /// Constructor.
    explicit Value_matrix(
        Memory_arena         *arena,
        IType_matrix const   *type,
        IValue const * const values[],
        size_t               size)
    : Base(arena, type, values, size)
    {
        MDL_ASSERT(size == type->get_columns());
    }
};

/// Implementation of the IValue_array interface.
class Value_array : public Value_compound<IValue_array, IType_array>
{
    typedef Value_compound<IValue_array, IType_array> Base;
public:
    /// Constructor.
    explicit Value_array(
        Memory_arena         *arena,
        IType_array const    *type,
        IValue const * const values[],
        size_t               size)
    : Base(arena, type, values, size)
    {
        MDL_ASSERT(size == type->get_size());
    }
};

/// Implementation of the IValue_rgb_color interface.
class Value_rgb_color : public Value_compound<IValue_rgb_color, IType_color, IValue_float>
{
    typedef Value_compound<IValue_rgb_color, IType_color, IValue_float> Base;
public:
    /// Negate a color value.
    IValue const *minus(IValue_factory *factory) const MDL_FINAL {
        IValue_float const *values[3];

        for (size_t i = 0; i < 3; ++i) {
            IValue const *tmp = m_values[i]->minus(factory);
            values[i] = cast<IValue_float>(tmp);
        }
        return factory->create_rgb_color(values[0], values[1], values[2]);
    }

    /// Multiply.
    IValue const *multiply(IValue_factory *factory, IValue const *rhs) const MDL_FINAL {
        Kind rhs_kind = rhs->get_kind();
        if (rhs_kind == s_kind) {
            // color by color element wise multiplication
            IValue_float const     *values[3];
            IValue_rgb_color const *o = cast<IValue_rgb_color>(rhs);

            for (size_t i = 0; i < 3; ++i) {
                IValue const *tmp = m_values[i]->multiply(factory, o->get_value(i));
                values[i] = cast<IValue_float>(tmp);
            }
            return factory->create_rgb_color(values[0], values[1], values[2]);
        } else if (rhs_kind == VK_FLOAT) {
            // color by float element wise multiplication
            IValue_float const *values[3];

            for (size_t i = 0; i < 3; ++i) {
                IValue const *tmp = m_values[i]->multiply(factory, rhs);
                values[i] = cast<IValue_float>(tmp);
            }
            return factory->create_rgb_color(values[0], values[1], values[2]);
        }
        return factory->create_bad();
    }

    /// Divide.
    IValue const *divide(IValue_factory *factory, IValue const *rhs) const MDL_FINAL {
        Kind rhs_kind = rhs->get_kind();
        if (rhs_kind == s_kind) {
            // color by color element wise division
            IValue_float const *values[3];
            IValue_rgb_color const *o = cast<IValue_rgb_color>(rhs);

            for (size_t i = 0; i < 3; ++i) {
                IValue const *tmp = m_values[i]->divide(factory, o->get_value(i));
                values[i] = cast<IValue_float>(tmp);
            }
            return factory->create_rgb_color(values[0], values[1], values[2]);
        } else if (rhs_kind == VK_FLOAT) {
            // color by float element wise division
            IValue_float const *values[3];

            for (size_t i = 0; i < 3; ++i) {
                IValue const *tmp = m_values[i]->divide(factory, rhs);
                values[i] = cast<IValue_float>(tmp);
            }
            return factory->create_rgb_color(values[0], values[1], values[2]);
        }
        return factory->create_bad();
    }

    /// Add.
    IValue const *add(IValue_factory *factory, IValue const *rhs) const MDL_FINAL {
        IType const *rhs_type = rhs->get_type();
        if (rhs->get_kind() == s_kind) {
            // color by color element wise addition
            IValue_float const     *values[3];
            IValue_rgb_color const *o = cast<IValue_rgb_color>(rhs);

            for (size_t i = 0; i < 3; ++i) {
                IValue const *tmp = m_values[i]->add(factory, o->get_value(i));
                values[i] = cast<IValue_float>(tmp);
            }
            return factory->create_rgb_color(values[0], values[1], values[2]);
        } else if (is<IType_float>(rhs_type)) {
            // color by float element wise addition
            IValue_float const *values[3];

            for (size_t i = 0; i < 3; ++i) {
                IValue const *tmp = m_values[i]->add(factory, rhs);
                values[i] = cast<IValue_float>(tmp);
            }
            return factory->create_rgb_color(values[0], values[1], values[2]);
        }
        return factory->create_bad();
    }

    /// Subtract.
    IValue const *sub(IValue_factory *factory, IValue const *rhs) const MDL_FINAL {
        IType const *rhs_type = rhs->get_type();
        if (rhs->get_kind() == s_kind) {
            // color by color element wise subtraction
            IValue_float const     *values[3];
            IValue_rgb_color const *o = cast<IValue_rgb_color>(rhs);

            for (size_t i = 0; i < 3; ++i) {
                IValue const *tmp = m_values[i]->sub(factory, o->get_value(i));
                values[i] = cast<IValue_float>(tmp);
            }
            return factory->create_rgb_color(values[0], values[1], values[2]);
        } else if (is<IType_float>(rhs_type)) {
            // color by float element wise subtraction
            IValue_float const *values[3];

            for (size_t i = 0; i < 3; ++i) {
                IValue const *tmp = m_values[i]->sub(factory, rhs);
                values[i] = cast<IValue_float>(tmp);
            }
            return factory->create_rgb_color(values[0], values[1], values[2]);
        }
        return factory->create_bad();
    }

    /// Compare.
    IValue::Compare_results compare(IValue const *rhs) const MDL_FINAL {
        if (rhs->get_kind() == s_kind) {
            // color by color compare
            IValue_rgb_color const *o = cast<IValue_rgb_color>(rhs);

            IValue::Compare_results res = IValue::CR_EQ;
            for (size_t i = 0; i < 3; ++i) {
                unsigned tmp = m_values[i]->compare(o->get_value(i));
                if (tmp & IValue::CR_UO)
                    return IValue::CR_UO;
                if ((tmp & IValue::CR_EQ) == 0)
                    res = IValue::CR_NE;
            }
            return res;
        }
        return IValue::CR_BAD;
    }

    /// Convert a color value.
    IValue const *convert(IValue_factory *factory, IType const *dst_tp) const MDL_FINAL {
        dst_tp = dst_tp->skip_type_alias();
        switch (dst_tp->get_kind()) {
        case IType::TK_VECTOR:
            {
                IType_vector const *v_type = cast<IType_vector>(dst_tp);
                if (v_type->get_size() == 3) {
                    // color can be converted into float3
                    IType_atomic const *a_type = v_type->get_element_type();
                    IValue const       *values[3];

                    for (size_t i = 0; i < 3; ++i) {
                        IValue const *tmp = m_values[i]->convert(factory, a_type);
                        if (is<IValue_bad>(tmp))
                            return tmp;
                        values[i] = tmp;
                    }
                    return factory->create_vector(v_type, values, 3);
                }
                break;
            }
        default:
            break;
        }
        return Base::convert(factory, dst_tp);
    }

    /// Returns true if the value is the ZERO (i.e. additive neutral).
    bool is_zero() const MDL_FINAL {
        bool   neutral = true;
        size_t n = m_values.size();

        for (size_t i = 0; i < n; ++i) {
            neutral &= m_values[i]->is_zero();
        }
        return neutral;
    }

    /// Returns true if the value is the ONE (i.e. multiplicative neutral).
    bool is_one() const MDL_FINAL {
        // color supports only component-wise multiplication
        return Value_rgb_color::is_all_one();
    }

    /// Returns true if all components of this value are ONE.
    bool is_all_one() const MDL_FINAL {
        bool   all_one = true;
        size_t n = m_values.size();

        for (size_t i = 0; i < n; ++i) {
            all_one &= m_values[i]->is_one();
        }
        return all_one;
    }

    /// Constructor.
    explicit Value_rgb_color(
        Memory_arena       *arena,
        IType_color const  *type,
        IValue_float const *value_r,
        IValue_float const *value_g,
        IValue_float const *value_b)
    // FIXME: currently RGB only values
    : Base(arena, type, value_r, value_g, value_b)
    {
    }
};

/// Implementation of the IValue_struct interface.
class Value_struct : public Value_compound<IValue_struct, IType_struct>
{
    typedef Value_compound<IValue_struct, IType_struct> Base;
public:

    /// Get a field.
    /// \param name     The name of the field.
    /// \returns        The value of the field.
    const IValue *get_field(ISymbol const *name) const MDL_FINAL {
        for (size_t i = 0, n = m_type->get_field_count(); i < n; ++i) {
            const IType   *f_type;
            const ISymbol *f_sym;

            m_type->get_field(i, f_type, f_sym);

            if (f_sym == name) {
                // found
                return Base::get_value(i);
            }
        }
        MDL_ASSERT(!"field name not found");
        return NULL;
    }

    /// Get a field.
    /// \param name     The name of the field.
    /// \returns        The value of the field.
    IValue const *get_field(char const *name) const MDL_FINAL {
        for (size_t i = 0, n = m_type->get_field_count(); i < n; ++i) {
            IType const   *f_type;
            ISymbol const *f_sym;

            m_type->get_field(i, f_type, f_sym);

            if (strcmp(f_sym->get_name(), name) == 0) {
                // found
                return Base::get_value(i);
            }
        }
        MDL_ASSERT(!"field name not found");
        return NULL;
    }

    /// Get the value by its MDL (field) name.
    ///
    /// \param name  the (field) name
    ///
    /// \return NULL if name does not exists, else return the corresponding value
    IValue const *get_value(char const *name) const MDL_FINAL {
        for (size_t i = 0, n = m_type->get_field_count(); i < n; ++i) {
            IType const   *f_type;
            ISymbol const *f_sym;

            m_type->get_field(i, f_type, f_sym);

            if (strcmp(f_sym->get_name(), name) == 0) {
                // found
                return Base::get_value(i);
            }
        }
        return NULL;
    }

    /// Convert a value.
    IValue const *convert(
        IValue_factory *factory,
        IType const    *dst_type) const MDL_OVERRIDE
    {
        if (IType_struct const *s_type = as<IType_struct>(dst_type)) {
            // conversion to another struct type (cast operator)
            if (s_type == m_type)
                return this;

            int n_fields = m_type->get_field_count();
            if (s_type->get_field_count() != n_fields)
                return factory->create_bad();

            Value_factory *vf = impl_cast<Value_factory>(factory);
            Small_VLA<IValue const *, 8> values(vf->get_allocator(), n_fields);
            for (int i = 0; i < n_fields; ++i) {
                IType const   *f_type = NULL;
                ISymbol const *f_sym  = NULL;

                s_type->get_field(i, f_type, f_sym);
                IValue const *v = m_values[i]->convert(factory, f_type);

                if (is<IValue_bad>(v))
                    return v;
                values[i] = v;
            }
            return factory->create_struct(s_type, values.data(), values.size());
        }
        return Base::convert(factory, dst_type);
    }

    /// Constructor.
    explicit Value_struct(
        Memory_arena         *arena,
        IType_struct const   *type,
        IValue const * const values[],
        size_t               size)
    : Base(arena, type, values, size)
    {
        MDL_ASSERT(size == type->get_field_count());
    }
};

// ---------------------------------- value factory ----------------------------------

// Create the bad value.
IValue_bad const *Value_factory::create_bad()
{
    return m_bad_value;
}

// Create a new value of type boolean.
IValue_bool const *Value_factory::create_bool(bool value)
{
    return value ? m_true_value : m_false_value;
}

// Create a new value of type integer.
IValue_int const *Value_factory::create_int(int value)
{
    IValue_int *v = m_builder.create<Value_int>(m_tf.create_int(), value);

    std::pair<Value_table::iterator, bool> res = m_vt.insert(v);
    if (! res.second) {
        m_builder.get_arena()->drop(v);
        return cast<IValue_int>(*res.first);
    }
    return v;
}

// Create a new value of type enum.
IValue_enum const *Value_factory::create_enum(IType_enum const *type, size_t index)
{
    MDL_ASSERT(index < size_t(type->get_value_count()) && "enum index out of range");
    IValue_enum *v = m_builder.create<Value_enum>(type, (int)index);
    std::pair<Value_table::iterator, bool> res = m_vt.insert(v);
    if (! res.second) {
        m_builder.get_arena()->drop(v);
        return cast<IValue_enum>(*res.first);
    }
    return v;
}

// Create a new value of type float.
IValue_float const *Value_factory::create_float(float value)
{
    IValue_float *v = m_builder.create<Value_float>(m_tf.create_float(), value);
    std::pair<Value_table::iterator, bool> res = m_vt.insert(v);
    if (! res.second) {
        m_builder.get_arena()->drop(v);
        return cast<IValue_float>(*res.first);
    }
    return v;
}

// Create a new value of type double.
IValue_double const *Value_factory::create_double(double value)
{
    IValue_double *v = m_builder.create<Value_double>(m_tf.create_double(), value);
    std::pair<Value_table::iterator, bool> res = m_vt.insert(v);
    if (! res.second) {
        m_builder.get_arena()->drop(v);
        return cast<IValue_double>(*res.first);
    }
    return v;
}

// Create a new value of type string.
IValue_string const *Value_factory::create_string(char const *value)
{
    MDL_ASSERT(value != NULL && "<NULL> strings are not allowed");
    IValue_string *v = m_builder.create<Value_string>(
        m_builder.get_arena(), m_tf.create_string(), value);
    std::pair<Value_table::iterator, bool> res = m_vt.insert(v);
    if (! res.second) {
        m_builder.get_arena()->drop(v);
        return cast<IValue_string>(*res.first);
    }
    return v;
}

// Create a new value of type vector.
IValue_vector const *Value_factory::create_vector(
    IType_vector const   *type,
    IValue const * const values[],
    size_t               size)
{
    IValue_vector *v = m_builder.create<Value_vector>(m_builder.get_arena(), type, values, size);
    std::pair<Value_table::iterator, bool> res = m_vt.insert(v);
    if (! res.second) {
        m_builder.get_arena()->drop(v);
        return cast<IValue_vector>(*res.first);
    }
    return v;
}

// Create a new value of type matrix.
IValue_matrix const *Value_factory::create_matrix(
    IType_matrix const   *type,
    IValue const * const values[],
    size_t               size)
{
    IValue_matrix *v = m_builder.create<Value_matrix>(m_builder.get_arena(), type, values, size);
    std::pair<Value_table::iterator, bool> res = m_vt.insert(v);
    if (! res.second) {
        m_builder.get_arena()->drop(v);
        return cast<IValue_matrix>(*res.first);
    }
    return v;
}

// Create a new value of type array.
IValue_array const *Value_factory::create_array(
    IType_array const    *type,
    IValue const * const values[],
    size_t               size)
{
    MDL_ASSERT(type->is_immediate_sized() && "Array values must have a immediate sized array type");
    IValue_array *v = m_builder.create<Value_array>(m_builder.get_arena(), type, values, size);
    std::pair<Value_table::iterator, bool> res = m_vt.insert(v);
    if (! res.second) {
        m_builder.get_arena()->drop(v);
        return cast<IValue_array>(*res.first);
    }
    return v;
}

// Create a new value of type color.
IValue_rgb_color const *Value_factory::create_rgb_color(
    IValue_float const *value_r,
    IValue_float const *value_g,
    IValue_float const *value_b)
{
    IValue_rgb_color *v = m_builder.create<Value_rgb_color>(
        m_builder.get_arena(), m_tf.create_color(), value_r, value_g, value_b);
    std::pair<Value_table::iterator, bool> res = m_vt.insert(v);
    if (! res.second) {
        m_builder.get_arena()->drop(v);
        return cast<IValue_rgb_color>(*res.first);
    }
    return v;
}

// Create a new value of type color from a spectrum.
IValue const *Value_factory::create_spectrum_color(
    IValue_array const *wavelengths,
    IValue_array const *amplitudes)
{
    IValue_float const *zero = create_float(0.0f);

    IValue_rgb_color *v = m_builder.create<Value_rgb_color>(
        m_builder.get_arena(), m_tf.create_color(), zero, zero, zero);
    std::pair<Value_table::iterator, bool> res = m_vt.insert(v);
    if (!res.second) {
        m_builder.get_arena()->drop(v);
        return cast<IValue_rgb_color>(*res.first);
    }
    return v;
}

// Create a new value of type struct.
IValue_struct const *Value_factory::create_struct(
    IType_struct const   *type,
    IValue const * const values[],
    size_t               size)
{
    IValue_struct *v = m_builder.create<Value_struct>(m_builder.get_arena(), type, values, size);
    std::pair<Value_table::iterator, bool> res = m_vt.insert(v);
    if (! res.second) {
        m_builder.get_arena()->drop(v);
        return cast<IValue_struct>(*res.first);
    }
    return v;
}

// Create a new texture value.
IValue_texture const *Value_factory::create_texture(
    IType_texture const            *type,
    char const                     *value,
    IValue_texture::gamma_mode     gamma,
    int                            tag_value,
    unsigned                       tag_version)
{
    MDL_ASSERT(value != NULL && "<NULL> textures are not allowed");
    IValue_texture *v = m_builder.create<Value_texture>(
        m_builder.get_arena(), type, value, gamma, tag_value, tag_version);
    std::pair<Value_table::iterator, bool> res = m_vt.insert(v);
    if (!res.second) {
        m_builder.get_arena()->drop(v);
        return cast<IValue_texture>(*res.first);
    }
    return v;
}

// Create a new bsdf_data texture value.
IValue_texture const *Value_factory::create_bsdf_data_texture(
    IValue_texture::Bsdf_data_kind bsdf_data_kind,
    int                            tag_value,
    unsigned                       tag_version)
{
    MDL_ASSERT(bsdf_data_kind != IValue_texture::BDK_NONE && "NONE kind is not allowed");
    IType_texture const *type = m_tf.create_texture(IType_texture::TS_BSDF_DATA);
    IValue_texture *v = m_builder.create<Value_texture>(
        m_builder.get_arena(), type, bsdf_data_kind, tag_value, tag_version);
    std::pair<Value_table::iterator, bool> res = m_vt.insert(v);
    if (!res.second) {
        m_builder.get_arena()->drop(v);
        return cast<IValue_texture>(*res.first);
    }
    return v;
}

// Create a new string light profile value.
IValue_light_profile const *Value_factory::create_light_profile(
    IType_light_profile const *type,
    char const                *value,
    int                       tag_value,
    unsigned                  tag_version)
{
    MDL_ASSERT(value != NULL && "<NULL> light_profiles are not allowed");
    IValue_light_profile *v = m_builder.create<Value_light_profile>(
            m_builder.get_arena(), type, value, tag_value, tag_version);
    std::pair<Value_table::iterator, bool> res = m_vt.insert(v);
    if (!res.second) {
        m_builder.get_arena()->drop(v);
        return cast<IValue_light_profile>(*res.first);
    }
    return v;
}

// Create a new string bsdf measurement value.
IValue_bsdf_measurement const *Value_factory::create_bsdf_measurement(
    IType_bsdf_measurement const *type,
    char const                   *value,
    int                          tag_value,
    unsigned                     tag_version)
{
    MDL_ASSERT(value != NULL && "<NULL> bsdf_measurements are not allowed");
    IValue_bsdf_measurement *v = m_builder.create<Value_bsdf_measurement>(
        m_builder.get_arena(), type, value, tag_value, tag_version);
    std::pair<Value_table::iterator, bool> res = m_vt.insert(v);
    if (!res.second) {
        m_builder.get_arena()->drop(v);
        return cast<IValue_bsdf_measurement>(*res.first);
    }
    return v;
}

// Create a new invalid reference.
IValue_invalid_ref const *Value_factory::create_invalid_ref(
    IType_reference const *type)
{
    IValue_invalid_ref *v = m_builder.create<Value_invalid_ref>(type);
    std::pair<Value_table::iterator, bool> res = m_vt.insert(v);
    if (! res.second) {
        m_builder.get_arena()->drop(v);
        return cast<IValue_invalid_ref>(*res.first);
    }
    return v;
}

// Create a new compound value.
IValue_compound const *Value_factory::create_compound(
    IType_compound const *type,
    IValue const * const values[],
    size_t               size)
{
    switch (type->get_kind()) {
    case IType::TK_VECTOR:
        {
            IType_vector const *v_type = cast<IType_vector>(type);
            return Value_factory::create_vector(v_type, values, size);
        }
    case IType::TK_MATRIX:
        {
            IType_matrix const *m_type = cast<IType_matrix>(type);
            return Value_factory::create_matrix(m_type, values, size);
        }
    case IType::TK_STRUCT:
        {
            IType_struct const *s_type = cast<IType_struct>(type);
            return Value_factory::create_struct(s_type, values, size);
        }
    case IType::TK_ARRAY:
        {
            IType_array const *a_type = cast<IType_array>(type);
            return Value_factory::create_array(a_type, values, size);
        }
    case IType::TK_COLOR:
        {
            if (size == 3) {
                IValue_float const *r = as<IValue_float>(values[0]);
                if (r == NULL) return NULL;
                IValue_float const *g = as<IValue_float>(values[1]);
                if (g == NULL) return NULL;
                IValue_float const *b = as<IValue_float>(values[2]);
                if (b == NULL) return NULL;
                return Value_factory::create_rgb_color(r, g, b);
            }
            return NULL;
        }
    default:
        MDL_ASSERT(!"unsupported compound type");
        return NULL;
    }
}

// Create a additive neutral zero if supported for the given type.
IValue const *Value_factory::create_zero(IType const *type)
{
    switch (type->get_kind()) {
    case IType::TK_BOOL:
        return create_bool(false);
    case IType::TK_INT:
        return create_int(0);
    case IType::TK_FLOAT:
        return create_float(0.0f);
    case IType::TK_DOUBLE:
        return create_double(0.0);
    case IType::TK_VECTOR:
        {
            IType_vector const *v_type = cast<IType_vector>(type);
            IType_atomic const *a_type = v_type->get_element_type();

            IValue const *zero = create_zero(a_type);

            MDL_ASSERT(!is<IValue_bad>(zero) && v_type->get_size() <= 4);
            IValue const *values[4] = { zero, zero, zero, zero };

            return create_vector(v_type, values, v_type->get_size());
        }
    case IType::TK_MATRIX:
        {
            IType_matrix const *m_type = cast<IType_matrix>(type);
            IType_vector const *v_type = m_type->get_element_type();

            IValue const *zero = create_zero(v_type);

            MDL_ASSERT(!is<IValue_bad>(zero) && m_type->get_columns() <= 4);
            IValue const *values[4] = { zero, zero, zero, zero };

            return create_matrix(m_type, values, m_type->get_columns());
        }
    case IType::TK_COLOR:
        {
            IValue_float const *zero = create_float(0.0f);
            return create_rgb_color(zero, zero, zero);
        }
    default:
        return create_bad();
    }
}

// Return the type factory of this value factory.
Type_factory *Value_factory::get_type_factory()
{
    return &m_tf;
}

// Import a value from another value factory.
IValue const *Value_factory::import(IValue const *value)
{
    switch (value->get_kind()) {
    case IValue::VK_BAD:
        return create_bad();
    case IValue::VK_BOOL:
        {
            IValue_bool const *v = cast<IValue_bool>(value);
            return create_bool(v->get_value());
        }
    case IValue::VK_INT:
        {
            IValue_int const *v = cast<IValue_int>(value);
            return create_int(v->get_value());
        }
    case IValue::VK_ENUM:
        {
            IValue_enum const *v  = cast<IValue_enum>(value);
            IType_enum const  *tp = cast<IType_enum>(m_tf.import(v->get_type()));
            return create_enum(tp, v->get_index());
        }
    case IValue::VK_FLOAT:
        {
            IValue_float const *v = cast<IValue_float>(value);
            return create_float(v->get_value());
        }
    case IValue::VK_DOUBLE:
        {
            IValue_double const *v = cast<IValue_double>(value);
            return create_double(v->get_value());
        }
    case IValue::VK_STRING:
        {
            IValue_string const *v = cast<IValue_string>(value);
            return create_string(v->get_value());
        }
    case IValue::VK_VECTOR:
    case IValue::VK_MATRIX:
    case IValue::VK_ARRAY:
    case IValue::VK_RGB_COLOR:
    case IValue::VK_STRUCT:
        {
            IValue_compound const *v = cast<IValue_compound>(value);
            IType_compound const *tp = cast<IType_compound>(m_tf.import(v->get_type()));

            size_t count = v->get_component_count();
            VLA<IValue const *> values(m_builder.get_arena()->get_allocator(), count);
            for (size_t i = 0; i < count; ++i) {
                values[i] = import(v->get_value(i));
            }
            return create_compound(tp, values.data(), count);
        }
    case IValue::VK_INVALID_REF:
        {
            IType_reference const *tp =
                static_cast<IType_reference const *>(m_tf.import(value->get_type()));
            return create_invalid_ref(tp);
        }
    case IValue::VK_TEXTURE:
        {
            IValue_texture const *v  = cast<IValue_texture>(value);
            IType_texture const  *tp =
                static_cast<IType_texture const *>(m_tf.import(v->get_type()));
            if (tp->get_shape() == IType_texture::TS_BSDF_DATA) {
                return create_bsdf_data_texture(
                    v->get_bsdf_data_kind(), v->get_tag_value(), v->get_tag_version());

            }
            return create_texture(
                tp, v->get_string_value(), v->get_gamma_mode(),
                v->get_tag_value(), v->get_tag_version());
        }
    case IValue::VK_LIGHT_PROFILE:
        {
            IValue_light_profile const *v  = cast<IValue_light_profile>(value);
            IType_light_profile const  *tp =
                static_cast<IType_light_profile const *>(m_tf.import(v->get_type()));
            return create_light_profile(
                tp, v->get_string_value(), v->get_tag_value(), v->get_tag_version());
        }
    case IValue::VK_BSDF_MEASUREMENT:
        {
            IValue_bsdf_measurement const *v  = cast<IValue_bsdf_measurement>(value);
            IType_bsdf_measurement const  *tp =
                static_cast<IType_bsdf_measurement const *>(m_tf.import(v->get_type()));
            return create_bsdf_measurement(
                tp, v->get_string_value(), v->get_tag_value(), v->get_tag_version());
        }
    }
    MDL_ASSERT(!"unsupported value kind");
    return NULL;
}

/// Dump all values owned by this Value table.
void Value_factory::dump() const
{
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
    mi::base::Handle<Printer>             printer(alloc.prt(dbg.get()));

    long i = 0;
    for (Value_table::const_iterator it(m_vt.begin()), end(m_vt.end());
         it != end;
         ++it)
    {
        IValue const *val = *it;

        printer->print(i++);
        printer->print(" ");
        printer->print(val);
        printer->print("\n");
    }
}

// Get the begin value of this Value factory.
Value_factory::const_value_iterator Value_factory::values_begin() const
{
    return m_vt.begin();
}

// Get the end value of this Value factory.
Value_factory::const_value_iterator Value_factory::values_end() const
{
    return m_vt.end();
}

// Serialize the type table.
void Value_factory::serialize(Factory_serializer &serializer) const
{
    // register predefined values
    serializer.register_value(m_bad_value);
    serializer.register_value(m_true_value);
    serializer.register_value(m_false_value);

    serializer.write_section_tag(Serializer::ST_VALUE_TABLE);
    DOUT(("value factory {\n"));
    INC_SCOPE();

    // reference the used type factory
    Tag_t type_factory_tag = serializer.get_type_factory_tag(&m_tf);
    serializer.write_encoded_tag(type_factory_tag);

    DOUT(("used type factory %u\n", unsigned(type_factory_tag)));

    for (Value_table::const_iterator it(m_vt.begin()), end(m_vt.end()); it != end; ++it) {
        IValue const *v = *it;

        serializer.enqueue_value(v);
    }

    serializer.write_enqueued_values();
    DEC_SCOPE();
    DOUT(("value factory }\n"));
}

// Deserialize the type table.
void Value_factory::deserialize(Factory_deserializer &deserializer)
{
    // register predefined values
    deserializer.register_value(Tag_t(1), m_bad_value);
    deserializer.register_value(Tag_t(2), m_true_value);
    deserializer.register_value(Tag_t(3), m_false_value);

    Tag_t t;

    t = deserializer.read_section_tag();
    MDL_ASSERT(t == Serializer::ST_VALUE_TABLE);
    DOUT(("value factory {\n"));
    INC_SCOPE();

    // reference the used type factory
    Tag_t type_factory_tag = deserializer.read_encoded_tag();

    DOUT(("used type factory %u\n", unsigned(type_factory_tag)));

    IType_factory const *tf;
    tf = deserializer.get_type_factory(type_factory_tag);
    MDL_ASSERT(tf == &m_tf);

    // read the number of values
    size_t n_values = deserializer.read_encoded_tag();

    DOUT(("#values %u\n", unsigned(n_values)));
    INC_SCOPE();

    for (size_t i = 0; i < n_values; ++i) {
        deserializer.read_value(*this);
    }
    DEC_SCOPE();

    DEC_SCOPE();
    DOUT(("value factory }\n"));
}

// Check if this value factory is the owner of the given value.
bool Value_factory::is_owner(IValue const *value) const
{
    // handle predefined first
    if (value == m_bad_value)
        return true;
    if (value == m_true_value)
        return true;
    if (value == m_false_value)
        return true;

    return m_builder.get_arena()->contains(value);
}

// Constructor.
Value_factory::Value_factory(Memory_arena &arena, Type_factory &tf)
: Base()
, m_builder(arena)
, m_tf(tf)
, m_vt(0, IValue_hash(), IValue_equal(), arena.get_allocator())
// Note: the following values are allocated on an arena, no free is needed
, m_bad_value(m_builder.create<Value_bad>(tf.create_error()))
, m_true_value(m_builder.create<Value_bool>(tf.create_bool(), true))
, m_false_value(m_builder.create<Value_bool>(tf.create_bool(), false))
{
}

static size_t hash_string(char const *s)
{
    size_t h = 0;
    for (; *s; ++s)
        h = 5 * h + *s;
    return h;
}

size_t Value_factory::IValue_hash::operator() (IValue const *value) const
{
    size_t h = (size_t(value->get_type()) >> 4) * 5;
    IValue::Kind kind = value->get_kind();
    switch (kind) {
    case IValue::VK_BAD:
    case IValue::VK_INVALID_REF:
        return h;
    case IValue::VK_BOOL:
        return h + size_t(cast<IValue_bool>(value)->get_value());
    case IValue::VK_INT:
        return h + size_t(cast<IValue_int>(value)->get_value());
    case IValue::VK_ENUM:
        return h + size_t(cast<IValue_enum>(value)->get_value());
    case IValue::VK_FLOAT:
        {
            union { size_t z; float f; } u;
            u.z = 0;
            u.f = cast<IValue_float>(value)->get_value();
            return h + u.z;
        }
    case IValue::VK_DOUBLE:
        {
            union { size_t z[2]; double d; } u;
            u.z[0] = 0;
            u.z[1] = 0;
            u.d = cast<IValue_double>(value)->get_value();
            return h + (u.z[0] ^ u.z[1]);
        }
    case IValue::VK_STRING:
        {
            char const *s = cast<IValue_string>(value)->get_value();
            return h + hash_string(s) + size_t(kind) * 3;
        }
    case IValue::VK_TEXTURE:
        {
            IValue_texture const *resource = cast<IValue_texture>(value);
            char const *s = resource->get_string_value();
            return
                h + hash_string(s) + size_t(kind) * 3 + resource->get_gamma_mode() * 17391 +
                size_t(resource->get_tag_value()) * 9 + size_t(resource->get_tag_version()) * 5 +
                resource->get_bsdf_data_kind() * 7;
        }
    case IValue::VK_LIGHT_PROFILE:
    case IValue::VK_BSDF_MEASUREMENT:
        {
            IValue_resource const *resource = cast<IValue_resource>(value);
            char const *s = resource->get_string_value();
            return
                h + hash_string(s) + size_t(kind) * 3 +
                size_t(resource->get_tag_value()) * 9 + size_t(resource->get_tag_version()) * 5;
        }
    case IValue::VK_VECTOR:
    case IValue::VK_MATRIX:
    case IValue::VK_ARRAY:
    case IValue::VK_RGB_COLOR:
    case IValue::VK_STRUCT:
        {
            IValue_compound const *c = cast<IValue_compound>(value);
            size_t h = 0;
            for (size_t i = 0, size = c->get_component_count(); i < size; ++i) {
                h = h * 5 + operator()(c->get_value(i));
            }
            return h;
        }
    }
    return 0;
}

bool Value_factory::IValue_equal::operator()(IValue const *a, IValue const *b) const
{
    IValue::Kind kind = a->get_kind();
    if (kind != b->get_kind())
        return false;

    if (a->get_type() != b->get_type())
        return false;

    switch (kind) {
    case IValue::VK_BAD:
    case IValue::VK_INVALID_REF:
        // depends only on type
        return true;
    case IValue::VK_BOOL:
        return cast<IValue_bool>(a)->get_value() == cast<IValue_bool>(b)->get_value();
    case IValue::VK_INT:
        return cast<IValue_int>(a)->get_value() == cast<IValue_int>(b)->get_value();
    case IValue::VK_ENUM:
        return cast<IValue_enum>(a)->get_value() == cast<IValue_enum>(b)->get_value();
    case IValue::VK_FLOAT:
        return bit_equal_float(
            cast<IValue_float>(a)->get_value(), cast<IValue_float>(b)->get_value());
    case IValue::VK_DOUBLE:
        return bit_equal_float(
            cast<IValue_double>(a)->get_value(), cast<IValue_double>(b)->get_value());
    case IValue::VK_STRING:
        return strcmp(
            cast<IValue_string>(a)->get_value(),
            cast<IValue_string>(b)->get_value()) == 0;
    case IValue::VK_TEXTURE:
        {
            IValue_texture const *ra = cast<IValue_texture>(a);
            IValue_texture const *rb = cast<IValue_texture>(b);
            if (ra->get_bsdf_data_kind() != rb->get_bsdf_data_kind())
                return false;
            if (ra->get_gamma_mode() != rb->get_gamma_mode())
                return false;
            if (strcmp(ra->get_string_value(), rb->get_string_value()) != 0)
                return false;
            if (ra->get_tag_value() != rb->get_tag_value())
                return false;
            if (ra->get_tag_version() != rb->get_tag_version())
                return false;
            return true;
        }
    case IValue::VK_LIGHT_PROFILE:
    case IValue::VK_BSDF_MEASUREMENT:
        {
            IValue_resource const *ra = cast<IValue_resource>(a);
            IValue_resource const *rb = cast<IValue_resource>(b);
            if (ra->get_tag_value() != rb->get_tag_value())
                return false;
            if (ra->get_tag_version() != rb->get_tag_version())
                return false;
            if (strcmp(ra->get_string_value(), rb->get_string_value()) != 0)
                return false;
            return true;
        }
    case IValue::VK_VECTOR:
    case IValue::VK_MATRIX:
    case IValue::VK_ARRAY:
    case IValue::VK_RGB_COLOR:
    case IValue::VK_STRUCT:
        {
            IValue_compound const *ca = cast<IValue_compound>(a);
            IValue_compound const *cb = cast<IValue_compound>(b);
            size_t size = ca->get_component_count();
            if (size != cb->get_component_count())
                return false;
            for (size_t i = 0; i < size; ++i) {
                if (! operator()(ca->get_value(i), cb->get_value(i)))
                    return false;
            }
            return true;
        }
    }
    return false;
}

}  // mdl
}  // mi
