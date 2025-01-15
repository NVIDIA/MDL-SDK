/******************************************************************************
 * Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
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
#include <mdl/compiler/compilercore/compilercore_streams.h>
#include <mdl/compiler/compilercore/compilercore_tools.h>
#include <mdl/compiler/compilercore/compilercore_cstring_hash.h>

#include "mdltlc_symbols.h"
#include "mdltlc_values.h"
#include "mdltlc_compilation_unit.h"

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

// ---------------------------------- Value_atomic ----------------------------------

// Constructor.
Value_atomic::Value_atomic(Type_atomic *type)
  : Base(type)
{
}

// Get the type of this value.
Type_atomic *Value_atomic::get_type()
{
    return cast<Type_atomic>(m_type);
}

// ---------------------------------- Value_bool ----------------------------------

// Constructor.
Value_bool::Value_bool(Type_bool *type, bool value)
  : Base(type)
  , m_value(value)
{
}

// Get the kind of value.
Value::Kind Value_bool::get_kind() const
{
    return s_kind;
}

// Get the type of this value.
Type_bool *Value_bool::get_type()
{
    return cast<Type_bool>(m_type);
}

// ---------------------------------- Value_int ----------------------------------

// Constructor.
Value_int::Value_int(Type_int *type, int value)
  : Base(type)
  , m_value(value)
{
}

// Get the kind of value.
Value::Kind Value_int::get_kind() const
{
    return s_kind;
}

// Get the type of this value.
Type_int *Value_int::get_type()
{
    return cast<Type_int>(m_type);
}

// ---------------------------------- Value_float ----------------------------------

// Constructor.
Value_float::Value_float(Type_float *type, float value, char const *s_value)
  : Base(type)
  , m_value(value)
  , m_s_value(s_value)
{
}

// Get the kind of value.
Value::Kind Value_float::get_kind() const
{
    return s_kind;
}

// Get the type of this value.
Type_float *Value_float::get_type()
{
    return cast<Type_float>(m_type);
}

// ---------------------------------- Value_string ----------------------------------

// Constructor.
Value_string::Value_string(Type_string *type, char const *value)
  : Base(type)
  , m_value(value)
{
}

// Get the kind of value.
Value::Kind Value_string::get_kind() const
{
    return s_kind;
}

// Get the type of this value.
Type_string *Value_string::get_type()
{
    return cast<Type_string>(m_type);
}

// ---------------------------------- value factory ----------------------------------

// Constructor.
Value_factory::Value_factory(mi::mdl::Memory_arena *arena, Compilation_unit *compilation_unit, Type_factory &tf)
  : Base()
  , m_compilation_unit(compilation_unit)
  , m_arena(*arena)
  , m_builder(m_arena)
  , m_tf(tf)
  , m_vt(0, Value_hash(), Value_equal(), m_arena.get_allocator())
{
}

// Get a value of type bool.
Value_bool *Value_factory::get_bool(bool value)
{
    Value_bool *v = m_builder.create<Value_bool>(m_tf.get_bool(), value);

    std::pair<Value_table::iterator, bool> res = m_vt.insert(v);
    if (!res.second) {
        m_builder.get_arena()->drop(v);
        return cast<Value_bool>(*res.first);
    }
    return v;
}

// Get a value of type integer.
Value_int *Value_factory::get_int(int value)
{
    Value_int *v = m_builder.create<Value_int>(m_tf.get_int(), value);

    std::pair<Value_table::iterator, bool> res = m_vt.insert(v);
    if (!res.second) {
        m_builder.get_arena()->drop(v);
        return cast<Value_int>(*res.first);
    }
    return v;
}

// Get a value of type float.
Value_float *Value_factory::get_float(float value, char const *s_value)
{
    Value_float *v = m_builder.create<Value_float>(m_tf.get_float(), value, mi::mdl::Arena_strdup(m_arena, s_value));
    std::pair<Value_table::iterator, bool> res = m_vt.insert(v);
    if (!res.second) {
        m_builder.get_arena()->drop(v);
        return cast<Value_float>(*res.first);
    }
    return v;
}

// Get a value of type string.
Value_string *Value_factory::get_string(char const *value)
{
    Symbol *sym = m_tf.get_symbol_table().get_symbol(value);
    Value_string *v = m_builder.create<Value_string>(m_tf.get_string(), sym->get_name());
    std::pair<Value_table::iterator, bool> res = m_vt.insert(v);
    if (!res.second) {
        m_builder.get_arena()->drop(v);
        return cast<Value_string>(*res.first);
    }
    return v;
}

// Return the type factory of this value factory.
Type_factory &Value_factory::get_type_factory()
{
    return m_tf;
}

size_t Value_factory::Value_hash::operator() (Value *value) const
{
    size_t h = (size_t(value->get_type()) >> 4) * 5;
    Value::Kind kind = value->get_kind();
    switch (kind) {
    case Value::VK_BOOL:
        return h + size_t(cast<Value_bool>(value)->get_value());
    case Value::VK_INT:
        return h + size_t(cast<Value_int>(value)->get_value());
    case Value::VK_FLOAT:
        {
            union { size_t z; float f; } u;
            u.z = 0;
            u.f = cast<Value_float>(value)->get_value();
            return h + u.z;
        }
    case Value::VK_STRING:
        {
            mi::mdl::cstring_hash hasher;
            return hasher(cast<Value_string>(value)->get_value());
        }
    }
    return 0;
}

bool Value_factory::Value_equal::operator()(Value *a, Value *b) const
{
    Value::Kind kind = a->get_kind();
    if (kind != b->get_kind())
        return false;

    // This equality test is safe because primitive types are uniq'ed.
    if (a->get_type() != b->get_type())
        return false;

    switch (kind) {
    case Value::VK_BOOL:
        return cast<Value_bool>(a)->get_value() == cast<Value_bool>(b)->get_value();
    case Value::VK_INT:
        return cast<Value_int>(a)->get_value() == cast<Value_int>(b)->get_value();
    case Value::VK_FLOAT:
        return mi::mdl::bit_equal_float(
           cast<Value_float>(a)->get_value(), cast<Value_float>(b)->get_value());
    case Value::VK_STRING:
        return strcmp(cast<Value_string>(a)->get_value(), cast<Value_string>(b)->get_value()) == 0;
    }
    return false;
}

