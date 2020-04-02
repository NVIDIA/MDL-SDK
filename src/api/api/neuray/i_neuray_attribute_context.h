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
 ** \brief Header for the IAttribute_context declaration.
 **/

#ifndef API_API_NEURAY_I_NEURAY_ATTRIBUTE_CONTEXT_H
#define API_API_NEURAY_I_NEURAY_ATTRIBUTE_CONTEXT_H

#include <mi/base/iinterface.h>
#include <mi/base/interface_declare.h>

namespace MI {

namespace ATTR { class Type; }

namespace NEURAY {

class IDb_element;

/// This small helper class retains an ATTR::Attribute and the corresponding DB element.
///
/// It provides the necessary context for accesses to array elements or structure members.
///
/// Retaining the ATTR::Attribute is important if a user removes the attribute while he/she still
/// holds a reference for accessing/editing the attribute. Not sure whether it is necessary to
/// retain the DB element as well.
class IAttribute_context : public
    mi::base::Interface_declare<0x923a0667,0xeea8,0x4e64,0x9c,0x9d,0x9e,0x89,0xb4,0xa7,0xf8,0x89,
                                mi::base::IInterface>
{
public:
    /// Returns the DB element that is retained by the attribute context.
    virtual const IDb_element* get_db_element() const = 0;

    /// Returns the ATTR type for the (part of the) attribute identified by \p attribute_name.
    ///
    /// Note that for array elements the returned type is not correct. ATTR::Type::lookup() returns
    /// a type tree where the top-level element has the array size of the array itself (and
    /// not 1 as one would expect for a non-nested array). This is due to the fact that
    /// ATTR::Type::lookup() returns a pointer to a subtree of the type tree of the attribute
    /// itself.
    ///
    /// \return    The type, or \c NULL in case of failure (e.g., \p attribute_name does
    ///           identify a part of the attribute).
    virtual const ATTR::Type* get_type( const char* attribute_name) const = 0;

    /// Compute the memory address for the (part of the) attribute identified by \p attribute_name.
    ///
    /// \return   The memory address, or \c NULL in case of failure (e.g., \p attribute_name does
    ///           identify a part of the attribute).
    virtual void* get_address( const char* attribute_name) const = 0;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_I_NEURAY_ATTRIBUTE_CONTEXT_H
