/***************************************************************************************************
 * Copyright (c) 2006-2023, NVIDIA CORPORATION. All rights reserved.
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
/// \brief      Scene element Texture

#ifndef MI_NEURAYLIB_ITEXTURE_H
#define MI_NEURAYLIB_ITEXTURE_H

#include <mi/neuraylib/iscene_element.h>
#include <mi/neuraylib/typedefs.h>

namespace mi {

namespace neuraylib {

class IImage;

/** \addtogroup mi_neuray_misc
@{
*/

/// Texture compression method.
enum Texture_compression
{
    TEXTURE_NO_COMPRESSION     = 0, ///< no compression
    TEXTURE_MEDIUM_COMPRESSION = 1, ///< medium compression ratio
    TEXTURE_HIGH_COMPRESSION   = 2, ///< high compression ratio
    TEXTURE_COMPRESSION_FORCE_32_BIT = 0xffffffffU
};

mi_static_assert( sizeof( Texture_compression) == sizeof( Uint32));

/// Supported filter types.
///
/// The filter type (or filter kernel) specifies how multiple samples are to be combined into
/// a single pixel value.
///
/// \if IRAY_API \see attribute \c filter on #mi::neuraylib::IOptions \endif
enum Filter_type
{
    FILTER_BOX          = 0,    ///< box filter
    FILTER_TRIANGLE     = 1,    ///< triangle filter
    FILTER_GAUSS        = 2,    ///< Gaussian filter
    FILTER_CMITCHELL    = 3,    ///< clipped Mitchell filter
    FILTER_CLANCZOS     = 4,    ///< clipped Lanczos filter
    FILTER_FAST         = 5,    ///< a fast filter, could be GPU anti-aliasing, or any
    FILTER_FORCE_32_BIT = 0xffffffffU
};

mi_static_assert( sizeof( Filter_type) == sizeof( Uint32));

/// Textures add image processing options to images.
///
/// A texture appears in the scene as an argument of an MDL function call (see
/// #mi::neuraylib::IFunction_call) or default argument of an MDL function definition (see
/// #mi::neuraylib::IFunction_definition). The type of such an argument is
/// #mi::neuraylib::IType_texture or an alias of it.
///
/// \if IRAY_API Textures may also serve as references to volume data. Note that gamma and
/// compression options are not available for volume data. \endif
///
/// \see #mi::neuraylib::IImage
/// \if IRAY_API \see #mi::neuraylib::IVolume_data \endif
class ITexture :
    public base::Interface_declare<0x012c847c,0xaf47,0x4338,0xb7,0xc4,0x78,0x67,0xa3,0x55,0x47,0x18,
                                   neuraylib::IScene_element>
{
public:
    /// \name Methods related to the referenced image
    //@{

    /// Sets the referenced image.
    ///
    /// \return
    ///           -  0: Success.
    ///           - -1: Invalid parameters (\c NULL pointer).
    ///           - -2: There is no element with that name.
    ///           - -3: The element can not be referenced because it is in a more private scope
    ///                 than the texture.
    ///           - -4: The element is not an image.
    virtual Sint32 set_image( const char* name) = 0;

    /// Returns the referenced image.
    ///
    /// \return   The referenced image, or \c NULL if no image is referenced.
    virtual const char* get_image() const = 0;

    //@}
    /// \if IRAY_API
    /// \name Methods related to the referenced volume instead of an image.
    /// \endif
    //@{

    /// \if IRAY_API
    /// Sets the referenced volume.
    ///
    /// \return
    ///           -  0: Success.
    ///           - -1: Invalid parameters (\c NULL pointer).
    ///           - -2: There is no element with that name.
    ///           - -3: The element can not be referenced because it is in a more private scope
    ///                 than the texture.
    ///           - -4: The element is not a volume.
    /// \endif
    virtual Sint32 set_volume( const char* name) = 0;

    /// \if IRAY_API
    /// Returns the referenced volume data.
    ///
    /// \return   The referenced volume, or \c NULL if no volume is referenced.
    /// \endif
    virtual const char* get_volume() const = 0;

    //@}
    /// \name Methods related to the gamma value
    //@{

    /// Sets the gamma value of this texture.
    ///
    /// The gamma value of the texture is an override for the gamma value of the underlying
    /// image. The special value 0.0 means that the override is not set.
    ///
    /// This override value is also used when MDL source code is to be generated:
    /// - a value of 1.0 is converted to \c "::tex::gamma_linear",
    /// - a value of 2.2 is converted to \c "::tex::gamma_srgb", and
    /// - all other values are converted to \c "::tex::gamma_default".
    /// Therefore, if you want to avoid \c "::tex::gamma_default" in generated MDL source code, it
    /// makes sense to replicate the gamma value of the underlying image here (instead of not using
    /// the override).
    ///
    /// \see #get_gamma(), #get_effective_gamma()
    virtual void set_gamma( Float32 gamma) = 0;

    /// Returns the gamma value of this texture.
    ///
    /// \see #set_gamma(), #get_effective_gamma()
    virtual Float32 get_gamma() const = 0;

    /// Returns the effective gamma value.
    ///
    /// Returns the gamma value of this texture, unless no override is set. In this case the gamma
    /// value of the underlying image at the given frame and uvtile index is returned. If no such
    /// image exists, 0.0 is returned.
    ///
    /// \param frame_id    The frame ID of the mipmap the gamma value is requested for when no
    ///                    override is set.
    /// \param uvtile_id   The uv-tile id of the mipmap the gamma value is requested for when no
    ///                    override is set.
    ///
    /// \see #set_gamma(), #get_gamma()
    virtual Float32 get_effective_gamma( Size frame_id, Size uvtile_id) const = 0;

#ifdef MI_NEURAYLIB_DEPRECATED_12_1
    inline  Float32 get_effective_gamma( Uint32 uvtile_id = 0) const
    { return get_effective_gamma( 0, uvtile_id); }
#endif // MI_NEURAYLIB_DEPRECATED_12_1

    //@}
    /// \name Miscellaneous methods
    //@{

    /// Returns the selector (or \c NULL).
    virtual const char* get_selector() const = 0;

    /// Sets the texture compression method.
    ///
    /// \note This setting does not affect the referenced image itself, it only affects image data
    ///       that has been processed by the render modes. For example, in order to save GPU memory
    ///       processed image data can be compressed before being uploaded to the GPU.
    ///
    /// \see #mi::neuraylib::Texture_compression
    virtual void set_compression( Texture_compression compression) = 0;

    /// Returns the texture compression method.
    ///
    /// \note This setting does not affect the referenced image itself, it only affects image data
    ///       that has been processed by the render modes. For example, in order to save GPU memory
    ///       processed image data can be compressed before being uploaded to the GPU.
    ///
    /// \see #mi::neuraylib::Texture_compression
    virtual Texture_compression get_compression() const = 0;

    //@}
};

/**@}*/ // end group mi_neuray_misc

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_ITEXTURE_H
