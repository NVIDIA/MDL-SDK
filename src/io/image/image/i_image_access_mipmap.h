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

#ifndef IO_IMAGE_IMAGE_IMAGE_I_ACCESS_MIPMAP_H
#define IO_IMAGE_IMAGE_IMAGE_I_ACCESS_MIPMAP_H

#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>
#include <mi/base/lock.h>
#include <mi/math/color.h>

#include "i_image_access_canvas.h"
#include "i_image_utilities.h"

#include <vector>

namespace mi { namespace neuraylib { class ICanvas; } }

namespace MI {

namespace IMAGE {

class IMipmap;

/// Wraps a mipmap and provides simplified access methods.
///
/// This class caches every miplevel ever seen. This is very important to avoid the high number of
/// retain() and release() calls in the lookup() method, which are very bad for the performance in
/// multi-threaded settings.
///
/// \note There is also an Edit_mipmap class for mutable mipmaps.
///
/// \note This class caches the canvases for the various miplevels. It assumes that no writes
///       happen at the same time (which would decouple the cached canvases for higher levels from
///       the mipmap). Use Edit_mipmap if this is needed.
///
/// \note Each read request first has to select the required miplevel. If you know that all
///       requests will use the same miplevel (e.g. the first one), it is faster to get the first
///       miplevel via IMipmap::get_level(), and then use Access_canvas directly on that miplevel.
class Access_mipmap
{
public:
    /// Constructor.
    Access_mipmap( const IMipmap* mipmap = 0);

    /// Copy constructor. Explicit because locks are not copyable.
    Access_mipmap( const Access_mipmap& rhs);

    /// Assignment operator. Explicit because locks are not assignable.
    Access_mipmap& operator=( const Access_mipmap& rhs);

    /// Sets this access to a new mipmap.
    ///
    /// \c NULL can be passed to release the previous canvas.
    void set( const IMipmap* mipmap);

    /// Returns the mipmap wrapped by this access.
    const IMipmap* get() const;

    /// Indicates whether this access points to a valid mipmap.
    bool is_valid() const { return m_mipmap; }

    /// Returns the x-resolution of a given miplevel.
    mi::Uint32 get_resolution_x( mi::Uint32 miplevel) const;

    /// Returns the y-resolution of a given miplevel.
    mi::Uint32 get_resolution_y( mi::Uint32 miplevel) const;

    /// Returns the number of layers of any miplevel.
    mi::Uint32 get_layers_size() const { return m_nr_of_layers; }

    /// Reads a rectangular area of pixels from the mipmap into a caller-specified buffer.
    ///
    /// If needed, pixel data is converted according to the given pixel type. The desired row
    /// order can be specified, too.
    ///
    /// \param buffer             The buffer to write the pixel data to.
    /// \param buffer_topdown     Indicates whether the buffer has the rows in top-down order.
    /// \param buffer_pixel_type  The pixel type of the buffer.
    /// \param miplevel_x         x coordinate of lower left corner of the rectangle in the miplevel
    /// \param miplevel_y         y coordinate of lower left corner of the rectangle in the miplevel
    /// \param width              Width of the rectangle to read.
    /// \param height             Height of the rectangle to read.
    /// \param buffer_padding     The padding between subsequent rows in the buffer in bytes.
    /// \param miplevel_layer     The layer of the miplevel to use.
    /// \param miplevel           The miplevel to use.
    /// \return                   \c true in case of success, \c false in case of errors, e.g.,
    ///                           no valid mipmap, no pixel type conversion, or invalid parameters.
    bool read_rect(
        mi::Uint8* buffer,
        bool buffer_topdown,
        Pixel_type buffer_pixel_type,
        mi::Uint32 miplevel_x,
        mi::Uint32 miplevel_y,
        mi::Uint32 width,
        mi::Uint32 height,
        mi::Uint32 buffer_padding = 0,
        mi::Uint32 miplevel_layer = 0,
        mi::Uint32 miplevel = 0) const;

    /// Reads a single pixel from the mipmap.
    ///
    /// To avoid lock contention and cache misses, it is recommended to use this method only from
    /// one thread at a time.
    ///
    /// \param color      The pixel will be returned here.
    /// \param x          The x coordinate of the pixel in the miplevel.
    /// \param y          The y coordinate of the pixel in the miplevel.
    /// \param z          The z coordinate of the pixel in the miplevel.
    /// \param miplevel   The miplevel to use.
    /// \return           \c true in case of success,
    ///                   \c false in case of errors, e.g., invalid parameters
    bool lookup(
        mi::math::Color& color, mi::Uint32 x, mi::Uint32 y, mi::Uint32 z = 0,
        mi::Uint32 miplevel = 0) const;

private:
    /// The wrapped mipmap.
    mi::base::Handle<const IMipmap> m_mipmap;
    /// The canvases for each miplevel (cached, lazily initialized), the size is m_miplevels.
    mutable std::vector<Access_canvas> m_access_canvases;
    /// Lock for m_access_canvases.
    mutable mi::base::Lock m_access_canvases_lock;

    /// The number of miplevels (cached).
    mi::Uint32 m_miplevels;
    /// The x resolution of the base miplevel (cached).
    mi::Uint32 m_resolution_x_0;
    /// The y resolution of the base miplevel (cached).
    mi::Uint32 m_resolution_y_0;
    /// The number of layers of each miplevel (cached).
    mi::Uint32 m_nr_of_layers;
};

/// Wraps a mipmap and provides simplified access methods.
///
/// This class caches every miplevel ever seen. This is very important to avoid the high number of
/// retain() and release() calls in the lookup() and store() methods, which are very bad for the
/// performance in multi-threaded settings.
///
/// \note There is also an Access_mipmap class for const mipmaps. Never use this class if
///       Access_mipmap is sufficient. The single-pixel methods in this class can be very slow
///       because only the canvas for the first miplevel can be cached.
///
/// \note Each read/write request first has to select the required miplevel. If you know that all
///       requests will use the same miplevel (e.g. the first one), it is faster to get the first
///       miplevel via IMipmap::get_level(), and then use Edit_canvas directly on that miplevel.
class Edit_mipmap
{
public:
    /// Constructor.
    Edit_mipmap( IMipmap* mipmap = 0);

    /// Copy constructor. Explicit because locks are not copyable.
    Edit_mipmap( const Edit_mipmap& rhs);

    /// Assignment operator. Explicit because locks are not assignable.
    Edit_mipmap& operator=( const Edit_mipmap& rhs);

    /// Set this edit to a new mipmap.
    ///
    /// \c NULL can be passed to release the previous canvas.
    void set( IMipmap* mipmap);

    /// Returns the mipmap wrapped by this edit.
    IMipmap* get() const;

    /// Indicates whether this edit points to a valid mipmap.
    bool is_valid() const { return m_mipmap; }

    /// Returns the x-resolution of a given miplevel.
    mi::Uint32 get_resolution_x( mi::Uint32 miplevel) const;

    /// Returns the y-resolution of a given miplevel.
    mi::Uint32 get_resolution_y( mi::Uint32 miplevel) const;

    /// Returns the number of layers of any miplevel.
    mi::Uint32 get_layers_size() const { return m_nr_of_layers; }

    /// Reads a rectangular area of pixels from the mipmap into a caller-specified buffer.
    ///
    /// If needed, pixel data is converted according to the given pixel type. The desired row
    /// order can be specified, too.
    ///
    /// \param buffer             The buffer to write the pixel data to.
    /// \param buffer_topdown     Indicates whether the buffer has the rows in top-down order.
    /// \param buffer_pixel_type  The pixel type of the buffer.
    /// \param miplevel_x         x coordinate of lower left corner of the rectangle in the miplevel
    /// \param miplevel_y         y coordinate of lower left corner of the rectangle in the miplevel
    /// \param width              Width of the rectangle to read.
    /// \param height             Height of the rectangle to read.
    /// \param buffer_padding     The padding between subsequent rows in the buffer in bytes.
    /// \param miplevel_layer     The layer of the miplevel to use.
    /// \param miplevel           The miplevel to use.
    /// \return                   \c true in case of success, \c false in case of errors, e.g.,
    ///                           no valid mipmap, no pixel type conversion, or invalid parameters.
    bool read_rect(
        mi::Uint8* buffer,
        bool buffer_topdown,
        Pixel_type buffer_pixel_type,
        mi::Uint32 miplevel_x,
        mi::Uint32 miplevel_y,
        mi::Uint32 width,
        mi::Uint32 height,
        mi::Uint32 buffer_padding = 0,
        mi::Uint32 miplevel_layer = 0,
        mi::Uint32 miplevel = 0) const;

    /// Writes a rectangular area of pixels from a caller-specified buffer into the mipmap.
    ///
    /// If needed, pixel data is converted from the given pixel type. The row order of the buffer
    /// can be specified, too.
    ///
    /// \param buffer             The buffer to read the pixel data from.
    /// \param buffer_topdown     Indicates whether the buffer has the rows in top-down order.
    /// \param buffer_pixel_type  The pixel type of the buffer.
    /// \param miplevel_x         x coordinate of lower left corner of the rectangle in the miplevel
    /// \param miplevel_y         y coordinate of lower left corner of the rectangle in the miplevel
    /// \param width              Width of the rectangle to write.
    /// \param height             Height of the rectangle to write.
    /// \param buffer_padding     The padding between subsequent rows in the buffer in bytes.
    /// \param miplevel_layer     The layer of the miplevel to use.
    /// \param miplevel           The miplevel to use.
    /// \return                   \c true in case of success, \c false in case of errors, e.g.,
    ///                           no valid mipmap, no pixel type conversion, or invalid parameters.
    bool write_rect(
        const mi::Uint8* buffer,
        bool buffer_topdown,
        Pixel_type buffer_pixel_type,
        mi::Uint32 miplevel_x,
        mi::Uint32 miplevel_y,
        mi::Uint32 width,
        mi::Uint32 height,
        mi::Uint32 buffer_padding = 0,
        mi::Uint32 miplevel_layer = 0,
        mi::Uint32 miplevel = 0);

    /// Reads a single pixel from the mipmap.
    ///
    /// To avoid lock contention and cache misses, it is recommended to use this method only from
    /// one thread at a time.
    ///
    /// \param color      The pixel will be returned here.
    /// \param x          The x coordinate of the pixel in the miplevel.
    /// \param y          The y coordinate of the pixel in the miplevel.
    /// \param z          The z coordinate of the pixel in the miplevel.
    /// \param miplevel   The miplevel to use.
    /// \return           \c true in case of success,
    ///                   \c false in case of errors, e.g., invalid parameters
    bool lookup(
        mi::math::Color& color, mi::Uint32 x, mi::Uint32 y, mi::Uint32 z = 0,
        mi::Uint32 miplevel = 0) const;

    /// Writes a single pixel to the mipmap.
    ///
    /// To avoid lock contention and cache misses, it is recommended to use this method only from
    /// one thread at a time.
    ///
    /// \param color      The pixel to store.
    /// \param x          The x coordinate of the pixel in the miplevel.
    /// \param y          The y coordinate of the pixel in the miplevel.
    /// \param z          The z coordinate of the pixel in the miplevel.
    /// \param miplevel   The miplevel to use.
    /// \return           \c true in case of success,
    ///                   \c false in case of errors, e.g., invalid parameters
    bool store(
        const mi::math::Color& color, mi::Uint32 x, mi::Uint32 y, mi::Uint32 z = 0,
        mi::Uint32 miplevel = 0);

private:
    /// The wrapped mipmap.
    mi::base::Handle<IMipmap> m_mipmap;
    /// The first miplevel (cached).
    Edit_canvas m_edit_canvas_0;

    /// The number of miplevels (cached).
    mi::Uint32 m_miplevels;
    /// The x resolution of the base miplevel (cached).
    mi::Uint32 m_resolution_x_0;
    /// The y resolution of the base miplevel (cached).
    mi::Uint32 m_resolution_y_0;
    /// The number of layers of each miplevel (cached).
    mi::Uint32 m_nr_of_layers;
};

} // namespace IMAGE

} // namespace MI

#endif // MI_IO_IMAGE_IMAGE_I_IMAGE_ACCESS_MIPMAP_H
