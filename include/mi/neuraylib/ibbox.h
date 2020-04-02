/***************************************************************************************************
 * Copyright (c) 2009-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Bounding box type.

#ifndef MI_NEURAYLIB_IBBOX_H
#define MI_NEURAYLIB_IBBOX_H

#include <mi/math/bbox.h>
#include <mi/neuraylib/icompound.h>
#include <mi/neuraylib/typedefs.h>

namespace mi {

/** \addtogroup mi_neuray_compounds
@{
*/

/// This interface represents bounding boxes.
///
/// It can be used to represent bounding boxes by an interface derived from #mi::base::IInterface.
///
/// \see #mi::Bbox3_struct
class IBbox3 :
    public base::Interface_declare<0x107953d0,0x70a0,0x48f5,0xb1,0x17,0x68,0x8e,0x7b,0xf8,0x85,0xa1,
                                   ICompound>
{
public:
    /// Returns the bounding box represented by this interface.
    virtual Bbox3_struct get_value() const = 0;

    /// Returns the bounding box represented by this interface.
    virtual void get_value( Bbox3_struct& value) const = 0;

    /// Sets the bounding box represented by this interface.
    virtual void set_value( const Bbox3_struct& value) = 0;

    /// Returns the bounding box represented by this interface.
    ///
    /// This inline method exists for the user's convenience since #mi::math::Bbox
    /// is not derived from #mi::math::Bbox_struct.
    inline void get_value( Bbox3& value) const {
        Bbox3_struct value_struct;
        get_value( value_struct);
        value = value_struct;
    }

    /// Sets the bounding box represented by this interface.
    ///
    /// This inline method exists for the user's convenience since #mi::math::Bbox
    /// is not derived from #mi::math::Bbox_struct.
    inline void set_value( const Bbox3& value) {
        Bbox3_struct value_struct = value;
        set_value( value_struct);
    }

    using ICompound::get_value;

    using ICompound::set_value;
};

/*@}*/ // end group mi_neuray_compounds

} // namespace mi

#endif // MI_NEURAYLIB_IBBOX_H
