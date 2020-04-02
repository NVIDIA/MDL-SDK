/***************************************************************************************************
 * Copyright (c) 2013-2020, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Header for the IEnum implementation.
 **/

#ifndef API_API_NEURAY_NEURAY_ENUM_IMPL_H
#define API_API_NEURAY_NEURAY_ENUM_IMPL_H

#include <mi/base/interface_implement.h>
#include <mi/base/interface_merger.h>
#include <mi/base/handle.h>
#include <mi/neuraylib/ienum.h>

#include "i_neuray_proxy.h"

#include <string>
#include <boost/core/noncopyable.hpp>

// see documentation of mi::base::Interface_merger
#include <mi/base/config.h>
#ifdef MI_COMPILER_MSC
#pragma warning( disable : 4505 )
#endif

namespace mi { class IString; namespace neuraylib { class ITransaction; } }

namespace MI {

namespace NEURAY {

/// Default implementation of IEnum
///
/// The default implementation Enum_impl of IEnum owns the memory used to store the actual
/// value. See the proxy implementation Enum_impl_proxy for a variant that does not own the
/// memory.
class Enum_impl
  : public mi::base::Interface_implement<mi::IEnum>,
    public boost::noncopyable
{
public:
    /// The factory expects exactly one argument of type IEnum_decl and one argument of type
    /// IString. The argument is passed to the constructor in this order.
    static mi::base::IInterface* create_api_class(
        mi::neuraylib::ITransaction* transaction,
        mi::Uint32 argc,
        const mi::base::IInterface* argv[]);

    /// Constructor
    Enum_impl( const mi::IEnum_decl* enum_decl, const char* type_name);

    /// Destructor
    ~Enum_impl();

    // public API methods (IData)

    const char* get_type_name() const;

    // public API methods (IEnum)

    void get_value( mi::Sint32& value) const;

    const char* get_value_by_name() const;

    mi::Sint32 set_value( mi::Sint32 value);

    mi::Sint32 set_value_by_name( const char* name);

    const mi::IEnum_decl* get_enum_decl() const;

    // internal methods

private:
    /// The type name of the enum itself.
    std::string m_type_name;

    /// Storage
    ///
    /// Note that the storage holds the index of the enumerator in the declaration, not the value
    /// of the enumerator.
    mi::Uint32 m_storage;

    /// The corresponding enum declaration.
    mi::base::Handle<const mi::IEnum_decl> m_enum_decl;
};


/// Proxy implementation of IEnum
///
/// The proxy implementation Enum_impl_proxy of IEnum does not own the memory used to
/// store the actual value. See the default implementation Enum_impl for a variant that does
/// own the memory.
///
/// Users are not supposed to construct instances of this class directly. They might get
/// an instance of this class though, e.g., when accessing attributes.
class Enum_impl_proxy
  : public mi::base::Interface_merger<mi::base::Interface_implement<mi::IEnum>, IProxy>,
    public boost::noncopyable
{
public:
    /// The factory expects exactly one argument of type IEnum_decl and two arguments of type
    /// IString. The argument is passed to the constructor in this order.
    static mi::base::IInterface* create_api_class(
        mi::neuraylib::ITransaction* transaction,
        mi::Uint32 argc,
        const mi::base::IInterface* argv[]);

    /// Constructor
    Enum_impl_proxy( const mi::IEnum_decl* enum_decl, const char* type_name);

    /// Destructor
    ~Enum_impl_proxy();

    // public API methods (IData)

    const char* get_type_name() const;

    // public API methods (IEnum)

    void get_value( mi::Sint32& value) const;

    const char* get_value_by_name() const;

    mi::Sint32 set_value( mi::Sint32 value);

    mi::Sint32 set_value_by_name( const char* name);

    const mi::IEnum_decl* get_enum_decl() const;

    // internal methods (of IProxy)

    void set_pointer_and_owner( void* pointer, const mi::base::IInterface* owner);

    void release_referenced_memory();

private:
    /// The type name of the enum itself.
    std::string m_type_name;

    /// Pointer to the storage.
    ///
    /// Note that the pointer points to the index of the enumerator in the declaration, not to
    /// the value of the enumerator.
    mi::Uint32* m_pointer;

    /// The corresponding enum declaration.
    mi::base::Handle<const mi::IEnum_decl> m_enum_decl;

    /// Owner of the storage
    ///
    /// The class uses reference counting on the owner to ensure that the pointer to the storage
    /// is valid.
    mi::base::Handle<const mi::base::IInterface> m_owner;
};


} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_NEURAY_ENUM_IMPL_H

