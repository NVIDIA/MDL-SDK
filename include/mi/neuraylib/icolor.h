/***************************************************************************************************
 * Copyright (c) 2006-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Color type.

#ifndef MI_NEURAYLIB_ICOLOR_H
#define MI_NEURAYLIB_ICOLOR_H

#include <mi/math/color.h>
#include <mi/neuraylib/icompound.h>
#include <mi/neuraylib/typedefs.h>

namespace mi {

/** \addtogroup mi_neuray_compounds
@{
*/

/// This interface represents RGBA colors.
///
/// It can be used to represent colors by an interface derived from #mi::base::IInterface.
///
/// \see #mi::Color_struct
class IColor :
    public base::Interface_declare<0x10a52754,0xa1c7,0x454c,0x8a,0x0b,0x56,0xd4,0xd4,0xdc,0x62,0x18,
                                   ICompound>
{
public:
    /// Returns the color represented by this interface.
    virtual Color_struct get_value() const = 0;

    /// Returns the color represented by this interface.
    virtual void get_value( Color_struct& value) const = 0;

    /// Sets the color represented by this interface.
    virtual void set_value( const Color_struct& value) = 0;

    using ICompound::get_value;

    using ICompound::set_value;
};

/// This interface represents RGB colors.
///
/// It can be used to represent colors by an interface derived from #mi::base::IInterface.
///
/// \see #mi::Color_struct
class IColor3 :
    public base::Interface_declare<0x1189e839,0x6d86,0x4bac,0xbc,0x72,0xb0,0xc0,0x2d,0xa9,0x3c,0x6c,
                                   ICompound>
{
public:
    /// Returns the color represented by this interface.
    ///
    /// The alpha component of the return value is set to 1.0.
    virtual Color_struct get_value() const = 0;

    /// Returns the color represented by this interface.
    ///
    /// The alpha component of \p value is set to 1.0.
    virtual void get_value( Color_struct& value) const = 0;

    /// Sets the color represented by this interface.
    ///
    /// The alpha component of \p value is ignored.
    virtual void set_value( const Color_struct& value) = 0;

    using ICompound::get_value;

    using ICompound::set_value;
};

/*@}*/ // end group mi_neuray_compounds

} // namespace mi

#endif // MI_NEURAYLIB_ICOLOR_H
