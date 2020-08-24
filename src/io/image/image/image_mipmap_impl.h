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

#ifndef IO_IMAGE_IMAGE_IMAGE_MIPMAP_IMPL_H
#define IO_IMAGE_IMAGE_IMAGE_MIPMAP_IMPL_H

#include "i_image_mipmap.h"

#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>
#include <mi/base/lock.h>

#include "i_image_utilities.h"

#include <string>
#include <vector>
#include <boost/core/noncopyable.hpp>

namespace mi { namespace neuraylib { class IReader; } }

namespace MI {

namespace IMAGE {

/// A simple implementation of the IMipmap interface.
///
/// The canvas is either file-based (constructed from a file name), or archive-based (constructed
/// from a reader and archive file/member name), or memory-based (constructed from parameters like
/// pixel type, width, height, etc., or from a given canvas or reader). File-based or archive-based
/// mipmaps load the tile data of the base level lazily when needed. Memory-based mipmaps create all
/// tiles for the base level right in the constructor.
///
/// Construction for higher-level mipmaps is done lazily, but when a certain level is requested
/// all tiles of it are computed (and hence all tiles from the previous level are needed).
///
/// File-based or archive-based mipmaps could flush unused tiles if memory gets tight (not yet
/// implemented).
class Mipmap_impl
  : public mi::base::Interface_implement<IMipmap>,
    public boost::noncopyable
{
public:
    /// Default constructor.
    ///
    /// Creates a dummy mipmap (1x1 pink pixel)
    Mipmap_impl();

    /// Constructor.
    ///
    /// Creates a memory-based mipmap with given pixel type, width, height, and layers of the
    /// base level.
    ///
    /// \param pixel_type         The desired pixel type.
    /// \param width              The desired width.
    /// \param height             The desired height.
    /// \param tile_width         The desired tile width. The special value 0 currently implies the
    ///                           width of the base level (but this might change without further
    ///                           notice).
    /// \param tile_height        The desired tile height. The special value 0 currently implies the
    ///                           height of the base level (but this might change without further
    ///                           notice).
    /// \param layers             The desired number of layers (depth).
    /// \param is_cubemap         Flag that indicates whether this mipmap represents a cubemap.
    /// \param gamma              The desired gamma value. The special value 0.0 represents the
    ///                           default gamma which is 1.0 for HDR pixel types and 2.2 for LDR
    ///                           pixel types.
    Mipmap_impl(
        Pixel_type pixel_type,
        mi::Uint32 width,
        mi::Uint32 height,
        mi::Uint32 tile_width,
        mi::Uint32 tile_height,
        mi::Uint32 layers,
        bool is_cubemap,
        mi::Float32 gamma);

    /// Constructor.
    ///
    /// Creates a file-based mipmap that represents the given file on disk.
    ///
    /// \param filename           The file that shall be represented by this mipmap.
    /// \param tile_width         The desired tile width. The special value 0 currently implies the
    ///                           width of the base level (but this might change without further
    ///                           notice).
    /// \param tile_height        The desired tile height. The special value 0 currently implies the
    ///                           height of the base level (but this might change without further
    ///                           notice).
    /// \param only_first_level   Indicates whether only the first (or all) miplevels should be
    ///                           read from the file.
    /// \param errors[out]        An optional pointer to an #mi::Sint32 to which an error code will
    ///                           be written. The error codes have the following meaning:
    ///                           -  0: Success.
    ///                           - -3: Failure to open the file.
    ///                           - -4: No image plugin found to handle the file.
    ///                           - -5: The image plugin failed to import the file.
    Mipmap_impl(
        const std::string& filename,
        mi::Uint32 tile_width,
        mi::Uint32 tile_height,
        bool only_first_level,
        mi::Sint32* errors = 0);

    /// Constructor.
    ///
    /// Creates an archive-based mipmap obtained from a reader.
    ///
    /// \param reader             The reader to be used to obtain the mipmap. Needs to support
    ///                           absolute access.
    /// \param archive_filename   The resolved filename of the archive itself.
    /// \param member_filename    The relative filename of the mipmap in the archive.
    /// \param tile_width         The desired tile width. The special value 0 currently implies the
    ///                           width of the base level (but this might change without further
    ///                           notice).
    /// \param tile_height        The desired tile height. The special value 0 currently implies the
    ///                           height of the base level (but this might change without further
    ///                           notice).
    /// \param only_first_level   Indicates whether only the first (or all) miplevels should be
    ///                           read from the reader.
    /// \param errors[out]        An optional pointer to an #mi::Sint32 to which an error code will
    ///                           be written. The error codes have the following meaning:
    ///                           -  0: Success.
    ///                           - -3: Invalid reader, or the reader does not support absolute
    ///                                 access.
    ///                           - -4: No image plugin found to handle the data.
    ///                           - -5: The image plugin failed to import the data.
    Mipmap_impl(
        Container_based,
        mi::neuraylib::IReader* reader,
        const std::string& archive_filename,
        const std::string& member_filename,
        mi::Uint32 tile_width,
        mi::Uint32 tile_height,
        bool only_first_level,
        mi::Sint32* errors = 0);

    /// Constructor.
    ///
    /// Creates a memory-based mipmap obtained from a reader.
    ///
    /// \param reader             The reader to be used to obtain the mipmap. Needs to support
    ///                           absolute access.
    /// \param image_format       The image format of the buffer.
    /// \param mdl_file_path      The resolved MDL file path (to be used for log messages only),
    ///                           or \c NULL in other contexts.
    /// \param tile_width         The desired tile width. The special value 0 currently implies the
    ///                           width of the base level (but this might change without further
    ///                           notice).
    /// \param tile_height        The desired tile height. The special value 0 currently implies the
    ///                           height of the base level (but this might change without further
    ///                           notice).
    /// \param only_first_level   Indicates whether only the first (or all) miplevels should be
    ///                           read from the reader.
    /// \param errors[out]        An optional pointer to an #mi::Sint32 to which an error code will
    ///                           be written. The error codes have the following meaning:
    ///                           -  0: Success.
    ///                           - -3: Invalid reader, or the reader does not support absolute
    ///                                 access.
    ///                           - -4: No image plugin found to handle the data.
    ///                           - -5: The image plugin failed to import the data.
    Mipmap_impl(
        Memory_based,
        mi::neuraylib::IReader* reader,
        const char* image_format,
        const char* mdl_file_path,
        mi::Uint32 tile_width,
        mi::Uint32 tile_height,
        bool only_first_level,
        mi::Sint32* errors = 0);

    /// Constructor.
    ///
    /// Creates a memory-based mipmap with given canvases, starting from the base level.
    ///
    /// Note that the canvases are not copied, but shared. See Image_module::copy_canvas() if
    /// sharing is not desired.
    ///
    /// \param canvases     The array of canvases to create the mipmap from, starting with the base
    ///                     level.
    /// \param is_cubemap   Flag that indicates whether this mipmap represents a cubemap.
    Mipmap_impl(
        std::vector<mi::base::Handle<mi::neuraylib::ICanvas> >& canvases, bool is_cubemap);

    // methods of mi::neuraylib::IMipmap

    mi::Uint32 get_nlevels() const;

    const mi::neuraylib::ICanvas* get_level( mi::Uint32 level) const;

    mi::neuraylib::ICanvas* get_level( mi::Uint32 level);

    bool get_is_cubemap() const { return m_is_cubemap; }

    mi::Size get_size() const;

private:

    /// The number of miplevels of this mipmap.
    ///
    /// The number of miplevels is determined from the width and height of the base level. The last
    /// miplevel has width and height 1.
    mi::Uint32 m_nr_of_levels;

    /// The number of miplevels in the file (or provided during construction).
    ///
    /// Up to this level, miplevels will be loaded from file (or were provided during construction).
    /// Higher miplevels will be computed.
    mi::Uint32 m_nr_of_provided_levels;

    /// The lock that protects m_levels and m_last_created_level;
    mutable mi::base::Lock m_lock;

    /// The last miplevel that was already created.
    ///
    /// \note Any access needs to be protected by m_lock.
    mutable mi::Uint32 m_last_created_level;

    /// The array of miplevels.
    ///
    /// m_levels[0] is the base level, m_levels[m_nr_of_levels-1] is the last miplevel. Note that
    /// the actual construction of the miplevels is done lazily when necessary.
    ///
    /// \note Any access needs to be protected by m_lock.
    mutable std::vector<mi::base::Handle<mi::neuraylib::ICanvas> > m_levels;

    /// Flag for cubemaps.
    bool m_is_cubemap;
};

} // namespace IMAGE

} // namespace MI

#endif // MI_IO_IMAGE_IMAGE_IMAGE_MIPMAP_IMPL_H
