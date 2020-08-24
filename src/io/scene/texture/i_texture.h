/***************************************************************************************************
 * Copyright (c) 2007-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Handles the DB element Texture.

#ifndef IO_SCENE_TEXTURE_I_TEXTURE_H
#define IO_SCENE_TEXTURE_I_TEXTURE_H

#include <mi/base/uuid.h>
#include <base/data/db/i_db_tag.h>
#include <base/data/db/i_db_journal_type.h>
#include <io/scene/scene/i_scene_scene_element.h>

namespace mi { namespace neuraylib { class IReader; } }

namespace MI {

namespace DB { class Transaction; }
namespace DBIMAGE { class Image_set; }
namespace SERIAL { class Serializer; class Deserializer; }

namespace TEXTURE {

/// Texture compression method.
enum Texture_compression {
    TEXTURE_NO_COMPRESSION     = 0, ///< no compression
    TEXTURE_MEDIUM_COMPRESSION = 1, ///< medium compression ratio
    TEXTURE_HIGH_COMPRESSION   = 2  ///< high compression ratio
};

/// The class ID for the #Texture class.
static const SERIAL::Class_id ID_TEXTURE = 0x5f546578;// '_Tex'

/// The texture class.
///
/// A texture references an image and adds remapping information to the image.
class Texture : public SCENE::Scene_element<Texture, ID_TEXTURE>
{
public:

    /// Default constructor.
    ///
    /// The reference is set to a NULL tag.
    Texture();

    Texture& operator=( const Texture&) = delete;

    /// Sets the reference image.
    void set_image( DB::Tag image);

    /// Returns the referenced image.
    DB::Tag get_image() const;

    /// Sets the gamma value of this texture.
    ///
    /// The gamma value of the texture is an override for the gamma value of the underlying
    /// image. The special value 0.0 means that the override is not set.
    void set_gamma( mi::Float32 gamma);

    /// Returns the gamma value of this texture.
    ///
    /// The gamma value of the texture is an override for the gamma value of the underlying
    /// image. The special value 0.0 means that the override is not set.
    mi::Float32 get_gamma() const;

    /// Returns the effective gamma value.
    ///
    /// Returns the gamma value of this texture, unless no override is set. In this case the
    /// gamma value of the underlying image is returned (or 0.0 if no image is set).
    mi::Float32 get_effective_gamma( DB::Transaction* transaction, mi::Uint32 uvtile_id = 0) const;

    /// Sets the texture compression method.
    ///
    /// \note This setting does not affect the referenced image itself, it only affects image data
    ///       that has been processed by the render modes. For example, in order to save GPU memory
    ///       processed image data can be compressed before being uploaded to the GPU.
    ///
    /// \see ::Texture_compression
    void set_compression( Texture_compression compression);

    /// Returns the texture compression method.
    ///
    /// \note This setting does not affect the referenced image itself, it only affects image data
    ///       that has been processed by the render modes. For example, in order to save GPU memory
    ///       processed image data can be compressed before being uploaded to the GPU.
    ///
    /// \see ::Texture_compression
    Texture_compression get_compression() const;

    // methods of SERIAL::Serializable

    const SERIAL::Serializable* serialize( SERIAL::Serializer* serializer) const;

    SERIAL::Serializable* deserialize( SERIAL::Deserializer* deserializer);

    void dump() const;

    // methods of DB::Element_base

    size_t get_size() const;

    DB::Journal_type get_journal_flags() const;

    Uint bundle( DB::Tag* results, Uint size) const;

    // methods of SCENE::Scene_element_base

    void get_scene_element_references( DB::Tag_set* result) const;

private:

    DB::Tag m_image;                   ///< The referenced image.
    mi::Float32 m_gamma;               ///< The gamma value.
    Texture_compression m_compression; ///< The compression method.
};

/// Loads a default texture and stores it in the DB.
///
/// Used by the MDL integration to process textures that appear in default arguments
/// (similar to the lightprofile and BSDF measurement loaders). A fixed mapping from the resolved
/// filename to DB element name is used to detect already loaded textures/images. In such a case,
/// the tag of the existing DB element is returned.
///
/// \param transaction           The DB transaction to be used.
/// \param image_set_desc        Description of the resolved image set to be loaded.
/// \param impl_hash             Hash of the data in the implementation DB element. Use {0,0,0,0} if
///                              hash is not known.
/// \param shared_proxy          Indicates whether a possibly already existing proxy DB element for
///                              that resource should simply be reused (the decision is based on
///                              the DB element name derived from \c image_set, not on
///                              \c impl_hash). Otherwise, an independent proxy DB element is
///                              created, even if the resource has already been loaded.
/// \param gamma                 The gamma value of the texture.
/// \return                      The tag of that texture (invalid in case of failures).
DB::Tag load_mdl_texture(
    DB::Transaction* transaction,
    DBIMAGE::Image_set* image_set,
    const mi::base::Uuid& impl_hash,
    bool shared_proxy,
    mi::Float32 gamma);

} // namespace TEXTURE

} // namespace MI

#endif // IO_SCENE_TEXTURE_I_TEXTURE_H
