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
/// \file
/// \brief Structure type.

#ifndef MI_NEURAYLIB_ISTRUCTURE_H
#define MI_NEURAYLIB_ISTRUCTURE_H

#include <mi/neuraylib/idata.h>

namespace mi {

class IStructure_decl;

/** \addtogroup mi_neuray_collections
@{
*/

/// This interface represents structures, i.e., a key-value based data structure.
///
/// Structures are based on a structure declaration which defines the structure members (their
/// types, name, and the order). The type name of a structure is the name that was used to
/// register its structure declaration. This type name can be used to create instances of a
/// particular structure declaration (note that \c "Structure" itself is not a valid type name as it
/// does not contain any information about a concrete structure type).
///
/// This interface does not offer any specialized methods, except #get_structure_decl(). All the
/// structure functionality is available via methods inherited from #mi::IData_collection where the
/// name of a structure member equals the key. The key indices correspond with the indices in the
/// structure declaration.
///
/// \note
///   The value returned by #mi::IData::get_type_name() might start with \c '{' which indicates that
///   it has been automatically generated. In this case the type name should be treated as an opaque
///   string since its format might change unexpectedly. It is perfectly fine to pass it to other
///   methods, e.g., #mi::neuraylib::IFactory::create(), but you should not attempt to interpret
///   the value in any way. Use #get_structure_decl() to obtain information about the type itself.
///
/// \see #mi::IStructure_decl
class IStructure :
    public base::Interface_declare<0xd23152f6,0x5640,0x4ea0,0x8c,0x59,0x27,0x3e,0xdf,0xab,0xd1,0x8e,
                                   IData_collection>
{
public:
    /// Returns the structure declaration for this structure.
    virtual const IStructure_decl* get_structure_decl() const = 0;
};

/*@}*/ // end group mi_neuray_collections

} // namespace mi

#endif // MI_NEURAYLIB_ISTRUCTURE_H
