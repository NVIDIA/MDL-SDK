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
 ** \brief Header for the INumber implementation.
 **/

#ifndef API_API_NEURAY_NEURAY_NUMBER_IMPL_H
#define API_API_NEURAY_NEURAY_NUMBER_IMPL_H

#include <mi/neuraylib/inumber.h>

#include <mi/base/interface_implement.h>
#include <mi/base/interface_merger.h>
#include <mi/base/handle.h>

#include "i_neuray_proxy.h"

#include <boost/core/noncopyable.hpp>

// see documentation of mi::base::Interface_merger
#include <mi/base/config.h>
#ifdef MI_COMPILER_MSC
#pragma warning( disable : 4505 )
#endif

namespace mi { namespace neuraylib { class ITransaction; } }

namespace MI {

namespace NEURAY {

/// Default implementation of interfaces derived from INumber
///
/// The default implementation Number_impl of interfaces derived from INumber owns the memory used
/// to store the actual value. See the proxy implementation Number_impl_proxy for a variant that
/// does not own the memory.
///
/// Note that only a fixed set of types is permitted for the template parameters I and T.
/// Hence we use explicit template instantiation in the corresponding .cpp file.
template<typename I, typename T>
class Number_impl
  : public mi::base::Interface_implement<I>,
    public boost::noncopyable
{
public:

    static mi::base::IInterface* create_api_class(
        mi::neuraylib::ITransaction* transaction,
        mi::Uint32 argc,
        const mi::base::IInterface* argv[]);

    /// Constructor
    ///
    /// The value is initialized to T( 0).
    Number_impl();

    // public API methods

    const char* get_type_name() const;

    void get_value( bool& value) const;

    void set_value( bool value);

    void get_value( mi::Uint8& value) const;

    void set_value( mi::Uint8 value);

    void get_value( mi::Uint16& value) const;

    void set_value( mi::Uint16 value);

    void get_value( mi::Uint32& value) const;

    void set_value( mi::Uint32 value);

    void get_value( mi::Uint64& value) const;

    void set_value( mi::Uint64 value);

    void get_value( mi::Sint8& value) const;

    void set_value( mi::Sint8 value);

    void get_value( mi::Sint16& value) const;

    void set_value( mi::Sint16 value);

    void get_value( mi::Sint32& value) const;

    void set_value( mi::Sint32 value);

    void get_value( mi::Sint64& value) const;

    void set_value( mi::Sint64 value);

    void get_value( mi::Float32& value) const;

    void set_value( mi::Float32 value);

    void get_value( mi::Float64& value) const;

    void set_value( mi::Float64 value);

    // internal methods

private:
    /// Storage
    T m_storage;
};

/// Proxy implementation of interfaces derived from INumber
///
/// The proxy implementation Number_impl_proxy of interfaces derived from INumber does not own the
/// memory used to store the actual value. See the default implementation Number_impl for a variant
/// that does own the memory.
///
/// Users are not supposed to construct instances of this class directly. They might get
/// an instance of this class though, e.g., when accessing attributes.
///
/// Note that only a fixed set of types is permitted for the template parameters I and T.
/// Hence we use explicit template instantiation in the corresponding .cpp file.
template<typename I, typename T>
class Number_impl_proxy
  : public mi::base::Interface_merger<mi::base::Interface_implement<I>, IProxy>,
    public boost::noncopyable
{
public:

    static mi::base::IInterface* create_api_class(
        mi::neuraylib::ITransaction* transaction,
        mi::Uint32 argc,
        const mi::base::IInterface* argv[]);

    /// Constructor
    Number_impl_proxy();

    // public API methods

    const char* get_type_name() const;

    void get_value( bool& value) const;

    void set_value( bool value);

    void get_value( mi::Uint8& value) const;

    void set_value( mi::Uint8 value);

    void get_value( mi::Uint16& value) const;

    void set_value( mi::Uint16 value);

    void get_value( mi::Uint32& value) const;

    void set_value( mi::Uint32 value);

    void get_value( mi::Uint64& value) const;

    void set_value( mi::Uint64 value);

    void get_value( mi::Sint8& value) const;

    void set_value( mi::Sint8 value);

    void get_value( mi::Sint16& value) const;

    void set_value( mi::Sint16 value);

    void get_value( mi::Sint32& value) const;

    void set_value( mi::Sint32 value);

    void get_value( mi::Sint64& value) const;

    void set_value( mi::Sint64 value);

    void get_value( mi::Float32& value) const;

    void set_value( mi::Float32 value);

    void get_value( mi::Float64& value) const;

    void set_value( mi::Float64 value);

    // internal methods (of IProxy)

    void set_pointer_and_owner( void* pointer, const mi::base::IInterface* owner);

    void release_referenced_memory();

private:
    /// Pointer to the storage
    T* m_pointer;

    /// Owner of the storage
    ///
    /// The class uses reference counting on the owner to ensure that the pointer to the storage
    /// is valid.
    mi::base::Handle<const mi::base::IInterface> m_owner;

};

/// Helper class to map from the actual type to the type name.
///
/// Used by Number_impl, Number_impl_proxy, and Compound_impl.
///
/// For the first two use cases the template parameter could also be the corresponding interface I.
/// However, in Compound_impl the interface I of the element type T is not available, only the
/// type T itself. However, if T is used, then we cannot use the Type_traits for mi::Size and
/// mi::Difference, because they are either mi::Uint32/mi::Sint32 or mi::Uint64/mi::Sint64,
/// but have different type names.
///
/// Note that only a fixed set of types is permitted for the template parameter T.
/// Hence we use explicit template instantiation in the corresponding .cpp file.
template<typename T>
class Type_traits
{
public:
    static const char* get_type_name();
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_NEURAY_NUMBER_IMPL_H
