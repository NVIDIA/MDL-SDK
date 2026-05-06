/***************************************************************************************************
 * Copyright (c) 2010-2026, NVIDIA CORPORATION. All rights reserved.
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

#include <unordered_map>

#include <base/data/idata/i_idata_string_cache.h>
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
    using Parent_type = Attribute_set_impl<T>;

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

private:

    /// Caches the return values of get_attribute_type_name().
    mutable IDATA::String_cache m_string_cache;
};

// Note the specialization for IAttribute_container in its implementation class.
template <typename T>
mi::IData* Attribute_set_impl<T>::create_attribute(
    const char* name, const char* type_name)
{
    ATTR::Attribute_set* attribute_set = this->get_db_element()->get_attributes();
    return Attribute_set_impl_helper::create_attribute(
        attribute_set, this, name, type_name, /*skip_type_check*/ false);
}

template <typename T>
bool Attribute_set_impl<T>::destroy_attribute( const char* name)
{
    ATTR::Attribute_set* attribute_set = this->get_db_element()->get_attributes();
    return Attribute_set_impl_helper::destroy_attribute( attribute_set, this, name);
}

template <typename T>
const mi::IData* Attribute_set_impl<T>::access_attribute( const char* name) const
{
    const ATTR::Attribute_set* attribute_set = this->get_db_element()->get_attributes();
    return Attribute_set_impl_helper::access_attribute( attribute_set, this, name);
}

template <typename T>
mi::IData* Attribute_set_impl<T>::edit_attribute( const char* name)
{
    ATTR::Attribute_set* attribute_set = this->get_db_element()->get_attributes();
    return Attribute_set_impl_helper::edit_attribute( attribute_set, this, name);
}

template <typename T>
bool Attribute_set_impl<T>::is_attribute( const char* name) const
{
    const ATTR::Attribute_set* attribute_set = this->get_db_element()->get_attributes();
    return Attribute_set_impl_helper::is_attribute( attribute_set, this, name);
}

template <typename T>
const char* Attribute_set_impl<T>::get_attribute_type_name( const char* name) const
{
    const ATTR::Attribute_set* attribute_set = this->get_db_element()->get_attributes();
    std::string result =
        Attribute_set_impl_helper::get_attribute_type_name( attribute_set, this, name);
    if( result.empty())
        return nullptr;
    return m_string_cache.add( result);
}

template <typename T>
mi::Sint32 Attribute_set_impl<T>::set_attribute_propagation(
    const char* name, mi::neuraylib::Propagation_type value)
{
    ATTR::Attribute_set* attribute_set = this->get_db_element()->get_attributes();
    return Attribute_set_impl_helper::set_attribute_propagation( attribute_set, this, name, value);
}

template <typename T>
mi::neuraylib::Propagation_type Attribute_set_impl<T>::get_attribute_propagation(
    const char* name) const
{
    const ATTR::Attribute_set* attribute_set = this->get_db_element()->get_attributes();
    return Attribute_set_impl_helper::get_attribute_propagation( attribute_set, this, name);
}

template <typename T>
const char* Attribute_set_impl<T>::enumerate_attributes( mi::Sint32 index) const
{
    const ATTR::Attribute_set* attribute_set = this->get_db_element()->get_attributes();
    return Attribute_set_impl_helper::enumerate_attributes( attribute_set, this, index);
}

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_NEURAY_ATTRIBUTE_SET_IMPL_H
