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
 ** \brief Header for the IProxy declaration.
 **/

#ifndef API_API_NEURAY_I_NEURAY_PROXY_H
#define API_API_NEURAY_I_NEURAY_PROXY_H

#include <mi/base/iinterface.h>
#include <mi/base/interface_declare.h>

namespace MI {

namespace NEURAY {

/// The interface for proxies.
///
/// Most interfaces derived from IData have two implementations: a default implementation (which
/// owns the corresponding memory), and a proxy implementation (which does not own the corresponding
/// memory). The interfaces derived from ICompound have just one implementation which handles
/// both use cases simultaneously.
///
/// The proxy implementations are derived from this interface which offers a single method to set
/// the pointer to the memory to be used and the owner of that memory. The proxy implementations
/// are typically used to provide access to attributes (the owner is of type IAttribute_context
/// then). Another use case is to provide interface pointers for compound elements (the owner is
/// of type ICompound then).
class IProxy : public
    mi::base::Interface_declare<0x82240810,0xa358,0x4175,0x82,0xa7,0xc3,0x45,0xff,0x3a,0x3d,0x42,
                                mi::base::IInterface>
{
public:
    /// Sets the pointer to the memory and its owner.
    ///
    /// \param pointer       The pointer to the memory to be used by the proxy.
    /// \param owner         The owner of the memory. Reference counting on \p owner is used to
    ///                      ensure \p pointer is valid.
    virtual void set_pointer_and_owner( void* pointer, const mi::base::IInterface* owner) = 0;

    /// Releases referenced memory.
    ///
    /// Releases memory that is referenced and owned by the attribute's data, but not part of
    /// the data itself. Note that memory is typically *not* released in the destructor because
    /// IProxy does not own the memory. However, if dynamic arrays are shrunk we need to release
    /// the memory references by elements that are destroyed.
    ///
    /// For IString, the method releases the memory of the string. For other instances of
    /// IData_simple the method does nothing. For IData_collection it recursively invokes itself
    /// on the collection's elements. For IDynamic_array, in addition, it releases the memory
    /// referenced in the Dynamic_array struct.
    ///
    /// Optimization: for ICompound, no recursion happens because there is no need to do so (neither
    /// strings nor collections can be elements of compounds).
    virtual void release_referenced_memory() = 0;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_I_NEURAY_PROXY_H
