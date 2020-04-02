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
 ** \brief Header for the IRef implementation.
 **/

#ifndef API_API_NEURAY_NEURAY_REF_IMPL_H
#define API_API_NEURAY_NEURAY_REF_IMPL_H

#include <mi/base/interface_implement.h>
#include <mi/base/interface_merger.h>
#include <mi/base/handle.h>
#include <mi/neuraylib/iref.h>

#include "i_neuray_proxy.h"

#include <boost/core/noncopyable.hpp>
#include <base/data/db/i_db_tag.h>

// see documentation of mi::base::Interface_merger
#include <mi/base/config.h>
#ifdef MI_COMPILER_MSC
#pragma warning( disable : 4505 )
#endif

namespace mi { namespace neuraylib { class ITransaction; } }

namespace MI {

namespace NEURAY {

class Transaction_impl;

/// Default implementation of IRef
///
/// The default implementation Ref_impl of IRef owns the memory used to store the actual
/// value. See the proxy implementation Ref_impl_proxy for a variant that does not own the
/// memory.
class Ref_impl
  : public mi::base::Interface_implement<mi::IRef>,
    public boost::noncopyable
{
public:

    static mi::base::IInterface* create_api_class(
        mi::neuraylib::ITransaction* transaction,
        mi::Uint32 argc,
        const mi::base::IInterface* argv[]);

    Ref_impl( mi::neuraylib::ITransaction* transaction);

    // public API methods

    const char* get_type_name() const;

    mi::Sint32 set_reference( const IInterface* db_element);

    mi::Sint32 set_reference( const char* name);

    const IInterface* get_reference() const;

    IInterface* get_reference();

    const char* get_reference_name() const;

    // internal methods

    /// Returns the embedded transaction.
    mi::neuraylib::ITransaction* get_transaction() const;

private:
    /// Storage
    DB::Tag m_storage;

    /// The transaction used to convert names to tags, and to access the referenced element.
    mi::base::Handle<Transaction_impl> m_transaction;
};

/// Proxy implementation of IRef
///
/// The proxy implementation Ref_impl_proxy of IRef does not own the memory used to store the
/// actual value. See the default implementation Ref_impl for a variant that does own the
/// memory.
///
/// Users are not supposed to construct instances of this class directly. They might get
/// an instance of this class though, e.g., when accessing attributes.
class Ref_impl_proxy
  : public mi::base::Interface_merger<mi::base::Interface_implement<mi::IRef>, IProxy>,
    public boost::noncopyable
{
public:

    static mi::base::IInterface* create_api_class(
        mi::neuraylib::ITransaction* transaction,
        mi::Uint32 argc,
        const mi::base::IInterface* argv[]);

    Ref_impl_proxy( mi::neuraylib::ITransaction* transaction);

    // public API methods

    const char* get_type_name() const;

    mi::Sint32 set_reference( const IInterface* db_element);

    mi::Sint32 set_reference( const char* name);

    const IInterface* get_reference() const;

    IInterface* get_reference();

    const char* get_reference_name() const;

    // internal methods (IProxy)

    void set_pointer_and_owner( void* pointer, const mi::base::IInterface* owner);

    void release_referenced_memory();

    // internal methods

    /// Returns the embedded transaction.
    mi::neuraylib::ITransaction* get_transaction() const;

private:
    /// Pointer to the storage
    DB::Tag* m_pointer;

    /// Owner of the storage
    ///
    /// The class uses reference counting on the owner to ensure that the pointer to the storage
    /// is valid.
    mi::base::Handle<const mi::base::IInterface> m_owner;

    /// The transaction used to convert names to tags, and to access to referenced element.
    mi::base::Handle<Transaction_impl> m_transaction;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_NEURAY_REF_IMPL_H
