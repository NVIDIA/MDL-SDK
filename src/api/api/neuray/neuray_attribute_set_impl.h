/***************************************************************************************************
 * Copyright (c) 2010-2020, NVIDIA CORPORATION. All rights reserved.
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
 **************************************************************************************************/

/** \file
 ** \brief Header for the IAttribute_set implementation.
 **/

#ifndef API_API_NEURAY_NEURAY_ATTRIBUTE_SET_IMPL_H
#define API_API_NEURAY_NEURAY_ATTRIBUTE_SET_IMPL_H

#include <mi/base/config.h>
#include <mi/neuraylib/iattribute_set.h>

#include "neuray_attribute_set_impl_helper.h"

#include <map>
#include <base/lib/log/i_log_assert.h>

// disable C4505: <class::method>: unreferenced local function has been removed
#ifdef MI_COMPILER_MSC
#pragma warning( disable : 4505 )
#endif

namespace MI {

namespace ATTR { class Attribute_set; }

namespace NEURAY {

class Transaction_impl;

template <typename T>
class Attribute_set_impl : public T
{
public:

    /// Constructor.
    Attribute_set_impl();

    /// Destructor.
    ~Attribute_set_impl();

    // public API methods

    mi::IData* create_attribute( const char* name, const char* type_name);

    using T::create_attribute;

    bool destroy_attribute( const char* name);

    const mi::IData* access_attribute( const char* name) const;

    using T::access_attribute;

    mi::IData* edit_attribute( const char* name);

    using T::edit_attribute;

    bool is_attribute( const char* name) const;

    const char* get_attribute_type_name( const char* name) const;

    mi::Sint32 set_attribute_propagation( const char* name, mi::neuraylib::Propagation_type value);

    mi::neuraylib::Propagation_type get_attribute_propagation( const char* name) const;

    const char* enumerate_attributes( mi::Sint32 index) const;

    // internal methods

    /// Returns the attribute set being worked on.
    ///
    /// Might return \c NULL if a const non-default attribute set was selected.
    ATTR::Attribute_set* get_attribute_set();

    /// Returns the attribute set being worked on.
    const ATTR::Attribute_set* get_attribute_set() const;

protected:

    /// Sets the attribute set being worked on.
    ///
    /// This method (including its variant below) may be called only once before any other method
    /// has been called. It allows to switch the attribute set being worked on from the default
    /// attribute set of every scene element to a different attribute set. The \p owner parameter
    /// is used for reference counting to ensure that \p attribute_set remains valid.
    ///
    /// This method is protected here, and made accessible by a public (internal) method on
    /// Attribute_container_impl to avoid misuse.
    ///
    /// Used by various MDL classes.
    void set_attribute_set(
        ATTR::Attribute_set* attribute_set, const mi::base::IInterface* owner);

    /// Sets the attribute set being worked on.
    ///
    /// This method (including its variant above) may be called only once before any other method
    /// has been called. It allows to switch the attribute set being worked on from the default
    /// attribute set of every scene element to a different attribute set. The \p owner parameter
    /// is used for reference counting to ensure that \p attribute_set remains valid.
    ///
    /// If you call this method passing a const attribute set, you may later calling only other
    /// const methods. Calling non-const methods later will fail.
    ///
    /// This method is protected here, and made accessible by a public (internal) method on
    /// Attribute_container_impl to avoid misuse.
    ///
    /// Used by various MDL classes.
    void set_attribute_set(
        const ATTR::Attribute_set* attribute_set, const mi::base::IInterface* owner) const;

private:

    /// Stores the attribute set being worked on.
    ///
    /// Might be \c NULL if not yet initialized or a const non-default attribute set was selected.
    mutable ATTR::Attribute_set* m_attribute_set;

    /// Stores the attribute set being worked on.
    ///
    /// Might be \c NULL if not yet initialized.
    mutable const ATTR::Attribute_set* m_const_attribute_set;

    /// Used for reference counting to ensure that pointers to non-default attribute sets remain
    /// valid.
    mutable const mi::base::IInterface* m_owner;

    /// Caches the return values of get_attribute_type_name(). The signature requires us to
    /// return a const char*, but the strings are not constant (e.g., in case of arrays).
    ///
    /// The map does not guarantee that all strings remain valid. Since we guarantee only that the
    /// last one remains valid we need only to cache the last string. But since the implementation
    /// cached all strings and these strings might still be valid, leave it now as is.
    mutable std::map<std::string, std::string> m_cached_type_names;
};

template <typename T>
Attribute_set_impl<T>::Attribute_set_impl()
  : m_attribute_set( nullptr),
    m_const_attribute_set( nullptr),
    m_owner( nullptr)
{
}

template <typename T>
Attribute_set_impl<T>::~Attribute_set_impl()
{
    if( m_owner)
        m_owner->release();
}

// Note the specialization for IAttribute_container in its implementation class.
template <typename T>
mi::IData* Attribute_set_impl<T>::create_attribute(
    const char* name, const char* type_name)
{
    ATTR::Attribute_set* attribute_set = get_attribute_set();
    return Attribute_set_impl_helper::create_attribute(
        attribute_set, this, name, type_name, /*skip_type_check*/ false);
}

template <typename T>
bool Attribute_set_impl<T>::destroy_attribute( const char* name)
{
    ATTR::Attribute_set* attribute_set = get_attribute_set();
    return Attribute_set_impl_helper::destroy_attribute( attribute_set, this, name);
}

template <typename T>
const mi::IData* Attribute_set_impl<T>::access_attribute( const char* name) const
{
    const ATTR::Attribute_set* attribute_set = get_attribute_set();
    return Attribute_set_impl_helper::access_attribute( attribute_set, this, name);
}

template <typename T>
mi::IData* Attribute_set_impl<T>::edit_attribute( const char* name)
{
    ATTR::Attribute_set* attribute_set = get_attribute_set();
    return Attribute_set_impl_helper::edit_attribute( attribute_set, this, name);
}

template <typename T>
bool Attribute_set_impl<T>::is_attribute( const char* name) const
{
    const ATTR::Attribute_set* attribute_set = get_attribute_set();
    return Attribute_set_impl_helper::is_attribute( attribute_set, this, name);
}

template <typename T>
const char* Attribute_set_impl<T>::get_attribute_type_name( const char* name) const
{
    const ATTR::Attribute_set* attribute_set = get_attribute_set();
    std::string result =
        Attribute_set_impl_helper::get_attribute_type_name( attribute_set, this, name);
    if( result.empty())
        return nullptr;
    m_cached_type_names[name] = result;
    return m_cached_type_names[name].c_str();
}

template <typename T>
mi::Sint32 Attribute_set_impl<T>::set_attribute_propagation(
    const char* name, mi::neuraylib::Propagation_type value)
{
    ATTR::Attribute_set* attribute_set = get_attribute_set();
    return Attribute_set_impl_helper::set_attribute_propagation( attribute_set, this, name, value);
}

template <typename T>
mi::neuraylib::Propagation_type Attribute_set_impl<T>::get_attribute_propagation(
    const char* name) const
{
    const ATTR::Attribute_set* attribute_set = get_attribute_set();
    return Attribute_set_impl_helper::get_attribute_propagation( attribute_set, this, name);
}

template <typename T>
const char* Attribute_set_impl<T>::enumerate_attributes( mi::Sint32 index) const
{
    const ATTR::Attribute_set* attribute_set = get_attribute_set();
    return Attribute_set_impl_helper::enumerate_attributes( attribute_set, this, index);
}

template <typename T>
ATTR::Attribute_set* Attribute_set_impl<T>::get_attribute_set()
{
    // The magic constant 0x1 indicates that the underlying attribute set is constant (in contrast
    // to NULL which means that the default attribute set of scene elements should be used).
    if( m_attribute_set == (ATTR::Attribute_set*) 0x1) //-V566 PVS
        return nullptr;
    if( m_attribute_set == 0)
        m_attribute_set = this->get_db_element()->get_attributes();
    return m_attribute_set;
}

template <typename T>
const ATTR::Attribute_set* Attribute_set_impl<T>::get_attribute_set() const
{
    if( m_const_attribute_set == 0)
        m_const_attribute_set = this->get_db_element()->get_attributes();
    return m_const_attribute_set;
}

template <typename T>
void Attribute_set_impl<T>::set_attribute_set(
    ATTR::Attribute_set* attribute_set, const mi::base::IInterface* owner)
{
    ASSERT( M_NEURAY_API, !m_attribute_set && !m_const_attribute_set && !m_owner);
    ASSERT( M_NEURAY_API, attribute_set && owner);
    m_attribute_set = attribute_set;
    m_const_attribute_set = attribute_set;
    m_owner = owner;
    m_owner->retain();
}

template <typename T>
void Attribute_set_impl<T>::set_attribute_set(
    const ATTR::Attribute_set* attribute_set, const mi::base::IInterface* owner) const
{
    ASSERT( M_NEURAY_API, !m_attribute_set && !m_const_attribute_set && !m_owner);
    ASSERT( M_NEURAY_API, attribute_set && owner);
    // Use the magic constant 0x1 to indicate that the underlying attribute set is constant.
    // Using NULL instead would cause get_attribute_set() to use the default attribute set of
    // scene elements instead and m_attribute_set and m_const_attribute_set would point to
    // different attribute sets.
    m_attribute_set = (ATTR::Attribute_set*) 0x1; //-V566 PVS
    m_const_attribute_set = attribute_set;
    m_owner = owner;
    m_owner->retain();
}

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_NEURAY_ATTRIBUTE_SET_IMPL_H
