/******************************************************************************
 * Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDLTLC_VALUES_H
#define MDLTLC_VALUES_H 1

#include <mi/mdl/mdl_iowned.h>

#include "mdltlc_types.h"

class Compilation_unit;
class Value_factory;

/// An mdltl value.
class Value : public mi::mdl::Interface_owned
{
public:
    /// The possible kinds of values.
    enum Kind {
        VK_BOOL,                ///< A bool value.
        VK_INT,                 ///< An integer value.
        VK_FLOAT,               ///< A floating point value.
        VK_STRING,              ///< A string value.
    };

    /// Get the kind of value.
    virtual Kind get_kind() const = 0;

    /// Get the type of this value.
    virtual Type *get_type();

protected:
    /// Constructor.
    explicit Value(Type *type);

protected:
    /// The type of this value
    Type *m_type;
};

/// An atomic value.
class Value_atomic : public Value
{
    typedef Value Base;
public:
    /// Get the type of this value.
    Type_atomic *get_type();

protected:
    /// Constructor.
    explicit Value_atomic(Type_atomic *type);
};

/// A value of type bool.
class Value_bool : public Value_atomic
{
    typedef Value_atomic Base;
    friend class mi::mdl::Arena_builder;
public:
    /// The kind of this subclass.
    static const Kind s_kind = VK_BOOL;

    /// Get the kind of value.
    Kind get_kind() const;

    /// Get the type of this value.
    Type_bool *get_type();

    /// Get the value.
    bool get_value() { return m_value; }

    /// Get the value.
    bool get_value() const { return m_value; }

private:
    /// Constructor.
    explicit Value_bool(Type_bool *type, bool value);

private:
    /// The value.
    bool m_value;
};

/// A value of type int.
class Value_int : public Value_atomic
{
    typedef Value_atomic Base;
    friend class mi::mdl::Arena_builder;
public:
    /// The kind of this subclass.
    static const Kind s_kind = VK_INT;

    /// Get the kind of value.
    Kind get_kind() const;

    /// Get the type of this value.
    Type_int *get_type();

    /// Get the value.
    int get_value() { return m_value; }

    /// Get the value.
    int get_value() const { return m_value; }

private:
    /// Constructor.
    explicit Value_int(Type_int *type, int value);

private:
    /// The value.
    int m_value;
};

/// A value of type float.
class Value_float : public Value_atomic
{
    typedef Value_atomic Base;
    friend class mi::mdl::Arena_builder;
public:
    /// The kind of this subclass.
    static const Kind s_kind = VK_FLOAT;

    /// Get the kind of value.
    Kind get_kind() const;

    /// Get the type of this value.
    Type_float *get_type();

    /// Get the value.
    float get_value() { return m_value; }

    /// Get the value.
    float get_value() const { return m_value; }

    /// Get the original string representation of the value.
    char const *get_s_value() const { return m_s_value; }

private:
    /// Constructor.
    explicit Value_float(Type_float *type, float value, char const *s_value);

private:
    /// The value.
    float m_value;

    /// Original string representation of the value.
    char const *m_s_value;
};

/// A value of type string.
class Value_string : public Value_atomic
{
    typedef Value_atomic Base;
    friend class mi::mdl::Arena_builder;
public:
    /// The kind of this subclass.
    static const Kind s_kind = VK_STRING;

    /// Get the kind of value.
    Kind get_kind() const;

    /// Get the type of this value.
    Type_string *get_type();

    /// Get the value.
    char const *get_value() const { return m_value; }

    /// Get the length of the string.
    size_t strlen() const { return ::strlen(m_value); }

    /// Check for prefix.
    bool is_prefix(Value_string *prefix) {
        size_t l = prefix->strlen();
        return ::strncmp(m_value, prefix->m_value, l) == 0;
    }

    /// Check for suffix.
    bool is_suffix(Value_string *suffix) {
        size_t s_len = suffix->strlen();
        size_t len   = strlen();
        return len >= s_len && ::strcmp(m_value + len - s_len, suffix->m_value) == 0;
    }

private:
    /// Constructor.
    explicit Value_string(Type_string *type, char const *value);

private:
    /// The value.
    char const *m_value;
};

/// Cast to subtype or return NULL if types do not match.
template<typename T>
T *as(Value *value) {
    return (value->get_kind() == T::s_kind) ? static_cast<T *>(value) : NULL;
}

/// Cast to subtype or return NULL if types do not match.
template<typename T>
T const *as(Value const *value) {
    return (value->get_kind() == T::s_kind) ? static_cast<T const *>(value) : NULL;
}

/// Cast to Value_atomic or return NULL if types do not match.
template<>
inline Value_atomic *as<Value_atomic>(Value *value) {
    switch (value->get_kind()) {
    case Value::VK_BOOL:
    case Value::VK_INT:
    case Value::VK_FLOAT:
    case Value::VK_STRING:
        return static_cast<Value_atomic *>(value);
    default:
        return NULL;
    }
}

/// Check if a value is of a certain type.
template<typename T>
bool is(Value *value) {
    return as<T>(value) != NULL;
}

/// Check if a value is of a certain type.
template<typename T>
bool is(Value const *value) {
    return as<T>(value) != NULL;
}

/// A static_cast with check in debug mode
template <typename T>
inline T *cast(Value *arg) {
    MDL_ASSERT(arg == NULL || is<T>(arg));
    return static_cast<T *>(arg);
}

/// A static_cast with check in debug mode
template <typename T>
inline T const *cast(Value const *arg) {
    MDL_ASSERT(arg == NULL || is<T>(arg));
    return static_cast<T const *>(arg);
}

/// The factory for creating values.
class Value_factory : public mi::mdl::Interface_owned
{
    friend class Compilation_unit;

    typedef Interface_owned Base;

    struct Value_hash {
        size_t operator()(Value *value) const;
    };

    struct Value_equal {
        bool operator()(Value *a, Value *b) const;
    };

    typedef mi::mdl::hash_set<Value *, Value_hash, Value_equal>::Type Value_table;

public:
    /// Get a value of type bool.
    ///
    /// \param value The value of the bool.
    Value_bool *get_bool(bool value);

    /// Get a value of type int.
    ///
    /// \param value The value of the integer.
    Value_int *get_int(int value);

    /// Get a value of type float.
    ///
    /// \param value The value of the float.
    Value_float *get_float(float value, char const *s_value);

    /// Get a value of type string.
    ///
    /// \param value The value of the string.
    Value_string *get_string(char const *value);

    /// Return the type factory of this value factory.
    ///
    /// Note: the returned type factory can create built-in types only.
    Type_factory &get_type_factory();

    /// Get the allocator.
    mi::mdl::IAllocator *get_allocator() const {
        return m_builder.get_arena()->get_allocator();
    }

private:
    /// Constructor.
    ///
    /// \param arena  a memory arena to allocate from
    /// \param f      the type factory to be used
    explicit Value_factory(mi::mdl::Memory_arena *arena,
                           Compilation_unit *compilation_unit,
                           Type_factory &tf);

private:

    /// Compilation unit this value factory is used for.
    Compilation_unit *m_compilation_unit;

    /// The arena for values.
    mi::mdl::Memory_arena &m_arena;

    /// The builder for values.
    mi::mdl::Arena_builder m_builder;

    /// A type factory, use to get the atomic types.
    Type_factory &m_tf;

    /// The value table.
    Value_table m_vt;
};

#endif
