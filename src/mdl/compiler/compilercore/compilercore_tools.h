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

#ifndef MDL_COMPILERCORE_TOOLS_H
#define MDL_COMPILERCORE_TOOLS_H 1

#include <mi/mdl/mdl_types.h>

#include "compilercore_mdl.h"
#include "compilercore_def_table.h"
#include "compilercore_modules.h"
#include "compilercore_factories.h"
#include "compilercore_messages.h"
#include "compilercore_thread_context.h"
#include "compilercore_assert.h"

namespace mi {
namespace mdl {

class IType;
class IExpression;
class IStatement;
class IDeclaration;
class IArgument;
class IValue;
class IType_array_size;

/// Returns the dimension of an array.
template<typename T, size_t n>
inline size_t dimension_of(T (&c)[n]) { return n; }

/// RAII-like store/restore facility for lvalues.
template<typename T>
class Store {
public:
    /// Constructor.
    Store(T &store, T const &value)
    : m_store(store)
    , m_old_value(store)
    {
        store = value;
    }

    /// Destructor.
    ~Store()
    {
        m_store = m_old_value;
    }

private:
    T &m_store;
    T m_old_value;
};

/// RAII line enable/disable facility for T (*)(T option) like member functions.
template<typename C, typename T>
class Option_store {
public:
    /// Constructor.
    Option_store(
        C &obj,
        T (C::*option)(T),
        T const &value)
    : m_obj(obj)
    , m_option(option)
    , m_old_value((obj.*option)(value))
    {
    }

    /// Destructor.
    ~Option_store()
    {
        (m_obj.*m_option)(m_old_value);
    }

private:
    C &m_obj;
    T (C::*m_option)(T);
    T m_old_value;
};

/// RAII line .clear() facility.
template<typename T>
class Clear_scope {
public:
    /// Constructor.
    Clear_scope(T &obj)
    : m_obj(obj)
    {
    }

    /// Destructor.
    ~Clear_scope()
    {
        m_obj.clear();
    }

private:
    T &m_obj;
};

// An impl_cast allows casting from an Interface pointer to its (only) Implementation class.
template <typename T, typename I>
T const *impl_cast(I const *);

template <typename T, typename I>
T *impl_cast(I *);

template <typename T, typename I>
T const &impl_cast(I const &);

template <typename T, typename I>
T &impl_cast(I &);

template<>
inline Messages_impl const *impl_cast(Messages const *t) {
    return static_cast<Messages_impl const *>(t);
}

template<>
inline Definition const *impl_cast(IDefinition const *t) {
    return static_cast<const Definition *>(t);
}

template<>
inline Module const *impl_cast(IModule const *t) {
    return static_cast<Module const *>(t);
}

template<>
inline Module *impl_cast(IModule *t) {
    return static_cast<Module *>(t);
}

template<>
inline Thread_context *impl_cast(IThread_context *t) {
    return static_cast<Thread_context *>(t);
}

template<>
inline MDL *impl_cast(IMDL *t) {
    return static_cast<MDL *>(t);
}

template<>
inline Type_factory *impl_cast(IType_factory *t) {
    return static_cast<Type_factory *>(t);
}

/// A static_cast with check in debug mode
template <typename T, typename F>
inline T *cast(F *arg) {
    MDL_ASSERT(arg == NULL || is<T>(arg));
    return static_cast<T *>(arg);
}

template <typename T, typename F>
inline T const *cast(F const *arg) {
    MDL_ASSERT(arg == NULL || is<T>(arg));
    return static_cast<T const *>(arg);
}

/// Check if the given type is the material type.
///
/// \param type  the type to check
static inline bool is_material_type(IType const *type)
{
    if (IType_struct const *s_type = as<IType_struct>(type))
        return s_type->get_predefined_id() == IType_struct::SID_MATERIAL;
    return false;
}

/// Check if the given type is the material type or a material subtype.
///
/// \param type  the type to check
static inline bool is_material_type_or_sub_type(IType const *type)
{
    if (IType_struct const *s_type = as<IType_struct>(type))
        return s_type->get_predefined_id() != IType_struct::SID_USER;
    return false;
}

/// Checks if the given MDL type is a derivative type.
///
/// \param type  the type to check
static inline bool is_deriv_type(IType const *type)
{
    if (IType_struct const *struct_type = as<IType_struct>(type)) {
        return struct_type->get_symbol()->get_name()[0] == '#';
    }
    return false;
}

/// Get the base value type of a derivative type.
///
/// \param type  the derivative type
static inline IType const *get_deriv_base_type(IType const *deriv_type)
{
    return cast<IType_struct>(deriv_type)->get_compound_type(0);
}

/// Check if a given type is the texture_2d type.
///
/// \param type  the type to check
static inline bool is_tex_2d(IType const *type)
{
    if (IType_texture const *tex_tp = as<IType_texture>(type))
        return tex_tp->get_shape() == IType_texture::TS_2D;
    return false;
}

/// Create an int2(0,0) constant.
///
/// \param vf  the value factory that will own the constant
static inline IValue const *create_int2_zero(
    IValue_factory &vf)
{
    IType_factory      &tf      = *vf.get_type_factory();
    IType_vector const *int2_tp = tf.create_vector(tf.create_int(), 2);

    return vf.create_zero(int2_tp);
}

/// Check, if two float values are BITWISE identical.
static inline bool bit_equal_float(float a, float b)
{
    union { size_t z; float f; } u1, u2; //-V117

    u1.z = 0;
    u1.f = a;

    u2.z = 0;
    u2.f = b;

    return u1.z == u2.z;
}

/// Check, if two float values are BITWISE identical.
static inline bool bit_equal_float(double a, double b)
{
    union { size_t z[2]; double d; } u1, u2; //-V117

    u1.z[0] = 0;
    u1.z[1] = 0;
    u1.d = a;

    u2.z[0] = 0;
    u2.z[1] = 0;
    u2.d = b;

    return u1.z[0] == u2.z[0] && u1.z[1] == u2.z[1];
}
}  // mdl
}  // mi

#endif
