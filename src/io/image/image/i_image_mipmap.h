/***************************************************************************************************
 * Copyright (c) 2011-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef IO_IMAGE_IMAGE_IMAGE_I_MIPMAP_H
#define IO_IMAGE_IMAGE_IMAGE_I_MIPMAP_H

#include <mi/base/interface_declare.h>

namespace mi { namespace neuraylib { class ICanvas; } }

namespace MI {

namespace IMAGE {

/// A mipmap is an array of related images with different resolutions.
///
/// The first miplevel is the base image. Higher miplevels are variants of this base image at
/// smaller resolutions. The resolution of a miplevel is the resolution of the previous level
/// divided by 2 and rounded down if necessary. Therefore, the number of miplevels is given by
/// 1 + floor( log2( max( w, h ))), where w and h are the width and height of the base image.
/// A pixel of a certain miplevel is the average of the four corresponding pixels of the previous
/// miplevel.
///
/// The management of higher miplevels is up to the implementation. Typically, they are computed
/// lazily when needed.
class IMipmap : public
    mi::base::Interface_declare<0x3ad15f9b,0xd3c1,0x49e0,0xb5,0xf0,0x83,0x37,0xa5,0x4a,0xdc,0xa5>
{
public:
    /// Returns the number of miplevels in this mipmap.
    ///
    /// Never returns 0.
    virtual mi::Uint32 get_nlevels() const = 0;

    /// Returns a given miplevel.
    ///
    /// Note that this variant returns a const canvas. It should always be used unless you really
    /// need to modify a miplevel.
    ///
    /// \param level   The miplevel to return. Valid values are from 0 to #get_nlevels()-1.
    /// \return        The requested miplevel, or \c NULL if \p level is greater than or equal to
    ///                #get_nlevels().
    virtual const mi::neuraylib::ICanvas* get_level( mi::Uint32 level) const = 0;

    /// Returns a given miplevel.
    ///
    /// Note that this variant returns a mutable canvas. It should only be used if absolutely
    /// necessary. All higher miplevel are automatically destroyed (and will be recomputed if
    /// needed).
    ///
    /// \param level   The miplevel to return. Valid values are from 0 to #get_nlevels()-1.
    /// \return        The requested miplevel, or \c NULL if \p level is greater than or equal to
    ///                #get_nlevels().
    virtual mi::neuraylib::ICanvas* get_level( mi::Uint32 level) = 0;

    /// Indicates whether this mipmap represents a cubemap.
    virtual bool get_is_cubemap() const = 0;

    /// Returns the memory used by this element in bytes, including all substructures.
    ///
    /// Used to implement DB::Element_base::get_size() for DBIMAGE::Image.
    virtual mi::Size get_size() const = 0;
};

} // namespace IMAGE

} // namespace MI

#endif // MI_IO_IMAGE_IMAGE_I_IMAGE_MIPMAP_H
