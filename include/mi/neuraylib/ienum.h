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
/// \file
/// \brief Numeric types.

#ifndef MI_NEURAYLIB_IENUM_H
#define MI_NEURAYLIB_IENUM_H

#include <mi/neuraylib/idata.h>

namespace mi {

class IEnum_decl;

/** \addtogroup mi_neuray_simple_types
@{
*/

/// This interface represents enums.
///
/// Enums are based on an enum declaration which defines the enumerators of the enum (their names
/// and values). The type name of an enum is the name that was used to register its enum
/// declaration. This type name can be used to create instances of a particular enum declaration
/// (note that \c "Enum" itself is not a valid type name as it does not contain any information
/// about a concrete enum type).
///
/// \note
///   The value returned by #mi::IData::get_type_name() might start with \c '{' which indicates that
///   it has been automatically generated. In this case the type name should be treated as an opaque
///   string since its format might change unexpectedly. It is perfectly fine to pass it to other
///   methods, e.g., #mi::neuraylib::IFactory::create(), but you should not attempt to interpret
///   the value in any way. Use #get_enum_decl() to obtain information about the type itself.
///
/// \see #mi::IEnum_decl
class IEnum :
    public base::Interface_declare<0x4e10d0e4,0x456b,0x45a5,0xa6,0xa7,0xdf,0x0a,0xa1,0x9a,0x0c,0xd2,
                                   IData_simple>
{
public:
    /// Returns the value of the enum as value of the corresponding enumerator.
    ///
    /// \see get_value_by_name()
    virtual void get_value( Sint32& value) const = 0;

    /// Returns the value of the enum as value of the corresponding enumerator.
    ///
    /// \see get_value_by_name()
    Sint32 get_value() const { Sint32 value; get_value( value); return value; }

    /// Returns the value of the enum as name of the enumerator.
    ///
    /// \see get_value()
    virtual const char* get_value_by_name() const = 0;

    /// Sets the enum via the value of an enumerator.
    ///
    /// \see set_value_by_name()
    ///
    /// \param value   The new enumerator, specified by its value. If there are multiple
    ///                enumerators with the same value the one with the smallest index in the
    ///                corresponding enum declaration is chosen.
    /// \return
    ///                -  0: Success.
    ///                - -1: This enum type has no enumerator with value \p value.
    virtual Sint32 set_value( Sint32 value) = 0;

    /// Sets the enum via the name of an enumerator.
    ///
    /// \see set_value()
    ///
    /// \param name    The new enumerator, specified by its name.
    /// \return
    ///                -  0: Success.
    ///                - -1: This enum type has no enumerator with name \p name.
    virtual Sint32 set_value_by_name( const char* name) = 0;

    /// Returns the enum declaration for this enum.
    virtual const IEnum_decl* get_enum_decl() const = 0;
};

/*@}*/ // end group mi_neuray_simple_types

} // namespace mi

#endif // MI_NEURAYLIB_IENUM_H
