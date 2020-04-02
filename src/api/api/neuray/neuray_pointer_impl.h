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
 ** \brief Header for the Pointer_impl and Const_pointer_impl implementation.
 **/

#ifndef API_API_NEURAY_NEURAY_POINTER_IMPL_H
#define API_API_NEURAY_NEURAY_POINTER_IMPL_H

#include <mi/base/interface_implement.h>
#include <mi/base/handle.h>
#include <mi/neuraylib/ipointer.h>

#include <string>
#include <boost/core/noncopyable.hpp>

namespace mi { namespace neuraylib { class ITransaction; } }

namespace MI {

namespace NEURAY {

class Pointer_impl
  : public mi::base::Interface_implement<mi::IPointer>,
    public boost::noncopyable
{
public:
    /// The factory expects exactly one argument of type IString. The arguments is passed to the
    /// constructor.
    static mi::base::IInterface* create_api_class(
        mi::neuraylib::ITransaction* transaction,
        mi::Uint32 argc,
        const mi::base::IInterface* argv[]);

    /// Constructor
    Pointer_impl(
       mi::neuraylib::ITransaction* transaction,
       const char* value_type_name);

    // public API methods (IData)

    const char* get_type_name() const;

    // public API methods (IPointer)

    mi::Sint32 set_pointer( mi::base::IInterface* pointer);

    mi::base::IInterface* get_pointer() const;

    // internal methods

    /// Returns the embedded transaction.
    mi::neuraylib::ITransaction* get_transaction() const;

private:

    /// Indicates whether the constructor successfully constructed the instance.
    ///
    /// Note that a structure type name can become invalid because it was unregistered between check
    /// and the actual construction. Hence, create_api_class() checks whether the constructor was
    /// successful.
    bool successfully_constructed() { return m_successfully_constructed; }

    /// Indicates whether the value has the correct value type.
    ///
    /// For untyped pointers, always returns \c true. For typed pointers,
    /// #mi::IData::get_type_name() is compared against m_value_type_name.
    bool has_correct_value_type( const mi::base::IInterface* value) const;

    /// The actual pointer.
    mi::base::Handle<mi::base::IInterface> m_pointer;

    /// The type name of the pointer itself.
    std::string m_type_name;

    /// The type name of the pointed value, or "Interface" for untyped pointers.
    std::string m_value_type_name;

    /// The transaction that might be needed to construct the wrapped pointer (might be \c NULL).
    mi::base::Handle<mi::neuraylib::ITransaction> m_transaction;

    /// Indicates whether the constructor successfully constructed the instance.
    /// \see #successfully_constructed()
    bool m_successfully_constructed;
};

class Const_pointer_impl
  : public mi::base::Interface_implement<mi::IConst_pointer>,
    public boost::noncopyable
{
public:
    /// The factory expects exactly one argument of type IString. The arguments is passed to the
    /// constructor.
    static mi::base::IInterface* create_api_class(
        mi::neuraylib::ITransaction* transaction,
        mi::Uint32 argc,
        const mi::base::IInterface* argv[]);

    /// Constructor
    Const_pointer_impl(
       mi::neuraylib::ITransaction* transaction,
       const char* value_type_name);

    // public API methods (IData)

    const char* get_type_name() const;

    // public API methods (IConst_pointer)

    mi::Sint32 set_pointer( const mi::base::IInterface* pointer);

    const mi::base::IInterface* get_pointer() const;

    // internal methods

    /// Returns the embedded transaction.
    mi::neuraylib::ITransaction* get_transaction() const;

private:

    /// Indicates whether the constructor successfully constructed the instance.
    ///
    /// Note that a structure type name can become invalid because it was unregistered between check
    /// and the actual construction. Hence, create_api_class() checks whether the constructor was
    /// successful.
    bool successfully_constructed() { return m_successfully_constructed; }

    /// Indicates whether the value has the correct value type.
    ///
    /// For untyped pointers, always returns \c true. For typed pointers,
    /// #mi::IData::get_type_name() is compared against m_value_type_name.
    bool has_correct_value_type( const mi::base::IInterface* value) const;

    /// The actual pointer.
    mi::base::Handle<const mi::base::IInterface> m_pointer;

    /// The type name of the pointer itself.
    std::string m_type_name;

    /// The type name of the pointed value, or "Interface" for untyped pointers.
    std::string m_value_type_name;

    /// The transaction that might be needed to construct the wrapped pointer (might be \c NULL).
    mi::base::Handle<mi::neuraylib::ITransaction> m_transaction;

    /// Indicates whether the constructor successfully constructed the instance.
    /// \see #successfully_constructed()
    bool m_successfully_constructed;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_NEURAY_POINTER_IMPL_H
