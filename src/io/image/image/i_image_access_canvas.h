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

#ifndef IO_IMAGE_IMAGE_IMAGE_I_ACCESS_CANVAS_H
#define IO_IMAGE_IMAGE_IMAGE_I_ACCESS_CANVAS_H

#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>
#include <mi/base/lock.h>
#include <mi/math/color.h>
#include <mi/neuraylib/icanvas.h>
#include <mi/neuraylib/itile.h>

#include "i_image_utilities.h"

#include <vector>

namespace mi { namespace neuraylib { class ITile; } }

namespace MI {

namespace IMAGE {

/// Wraps a canvas and provides simplified access methods.
///
/// This class caches every tile ever seen. This is very important to avoid the high number of
/// retain() and release() calls in the lookup() method, which are very bad for the performance in
/// multi-threaded settings.
///
/// \note There is also an Edit_mipmap class for mutable canvases.
class Access_canvas
{
public:
    /// Constructor.
    ///
    /// \param canvas     The wrapped canvas.
    /// \param lockless   The lockless variant of Access_canvas prefetches all tiles of the canvas
    ///                   to avoid locking in the lookup() method. Note that operator= and the
    ///                   set() method are not thread-safe in the lockless variant.
    Access_canvas( const mi::neuraylib::ICanvas* canvas = 0, bool lockless = false);

    /// Copy constructor. Explicit because locks are not copyable.
    Access_canvas( const Access_canvas& rhs);

    /// Assignment operator. Explicit because locks are not assignable.
    ///
    /// \note This method is not thread-safe if the instance is lockless.
    Access_canvas& operator=( const Access_canvas& rhs);

    /// Sets this access to a new canvas.
    ///
    /// \c NULL can be passed to release the previous canvas.
    ///
    /// \note This method is not thread-safe if the instance is lockless.
    void set( const mi::neuraylib::ICanvas* canvas);

    /// Returns the canvas wrapped by this access.
    const mi::neuraylib::ICanvas* get() const;

    /// Indicates whether this access points to a valid canvas.
    bool is_valid() const { return m_canvas; }

    /// Reads a rectangular area of pixels from the canvas into a caller-specified buffer.
    ///
    /// If needed, pixel data is converted according to the given pixel type. The desired row
    /// order can be specified, too.
    ///
    /// \param buffer              The buffer to write the pixel data to.
    /// \param buffer_topdown      Indicates whether the buffer has the rows in top-down order.
    /// \param buffer_pixel_type   The pixel type of the buffer.
    /// \param canvas_x            x coordinate of lower left corner of the rectangle in the canvas.
    /// \param canvas_y            y coordinate of lower left corner of the rectangle in the canvas.
    /// \param width               Width of the rectangle to read.
    /// \param height              Height of the rectangle to read.
    /// \param buffer_padding      The padding between subsequent rows in the buffer in bytes.
    /// \param canvas_layer        The layer of the canvas to use.
    /// \return                    \c true in case of success, \c false in case of errors, e.g.,
    ///                            no valid canvas, no pixel type conversion, or invalid parameters.
    bool read_rect(
        mi::Uint8* buffer,
        bool buffer_topdown,
        Pixel_type buffer_pixel_type,
        mi::Uint32 canvas_x,
        mi::Uint32 canvas_y,
        mi::Uint32 width,
        mi::Uint32 height,
        mi::Uint32 buffer_padding = 0,
        mi::Uint32 canvas_layer = 0) const;

    /// Reads a single pixel from the canvas.
    ///
    /// To avoid lock contention and cache misses, it is recommended to use this method only from
    /// one thread at a time. (This does not apply to the lockless variant.)
    ///
    /// \param color   The pixel will be returned here.
    /// \param x       The x coordinate of the pixel in the canvas.
    /// \param y       The y coordinate of the pixel in the canvas.
    /// \param z       The z coordinate of the pixel in the canvas.
    /// \return        \c true in case of success,
    ///                \c false in case of errors, e.g., invalid parameters
    bool lookup( mi::math::Color& color, mi::Uint32 x, mi::Uint32 y, mi::Uint32 z = 0) const;

private:
    /// The wrapped canvas.
    mi::base::Handle<const mi::neuraylib::ICanvas> m_canvas;
    /// The tiles (cached, lazily initialized), the size is m_nr_of_tiles.
    mutable std::vector<mi::base::Handle<const mi::neuraylib::ITile> > m_tiles;
    /// Lock for m_tiles.
    mutable mi::base::Lock m_tiles_lock;
    /// Indicates whether m_tiles_lock is to be used at all.
    bool m_lockless;

    /// The number of layers (cached).
    mi::Uint32 m_nr_of_layers;
    /// The width of the canvas (cached).
    mi::Uint32 m_canvas_width;
    /// The height of the canvas (cached).
    mi::Uint32 m_canvas_height;
    /// The pixel type of the canvas (cached).
    Pixel_type m_canvas_pixel_type;
    /// The tile width of the canvas (cached).
    mi::Uint32 m_tile_width;
    /// The tile height of the canvas (cached).
    mi::Uint32 m_tile_height;
    /// The number of tiles in x direction (cached).
    mi::Uint32 m_nr_of_tiles_x;
    /// The number of tiles in y direction (cached).
    mi::Uint32 m_nr_of_tiles_y;
    /// The total number of tiles (cached).
    mi::Uint32 m_nr_of_tiles;
};

/// Wraps a canvas and provides simplified access methods.
///
/// This class caches every tile ever seen. This is very important to avoid the high number of
/// retain() and release() calls in the lookup() and store() methods, which are very bad for the
/// performance in multi-threaded settings.
///
/// \note There is also an Access_canvas class for const canvases.
class Edit_canvas
{
public:
    /// Constructor for mutable canvases.
    ///
    /// \param canvas     The wrapped canvas.
    /// \param lockless   The lockless variant of Edit_canvas prefetches all tiles of the canvas
    ///                   to avoid locking in the lookup() and store() methods. Note that operator=
    ///                   and the set() method are not thread-safe in the lockless variant.
    Edit_canvas( mi::neuraylib::ICanvas* canvas = 0, bool lockless = false);

    /// Copy constructor. Explicit because locks are not copyable.
    Edit_canvas( const Edit_canvas& rhs);

    /// Assignment operator. Explicit because locks are not assignable.
    ///
    /// \note This method is not thread-safe if the instance is lockless.
    Edit_canvas& operator=( const Edit_canvas& rhs);

    /// Set this edit to a new canvas.
    ///
    /// \c NULL can be passed to release the previous canvas.
    ///
    /// \note This method is not thread-safe if the instance is lockless.
    void set( mi::neuraylib::ICanvas* canvas);

    /// Returns the canvas wrapped by this edit.
    mi::neuraylib::ICanvas* get() const;

    /// Indicates whether this access points to a valid canvas.
    bool is_valid() const { return m_canvas; }

    /// Reads a rectangular area of pixels from the canvas into a caller-specified buffer.
    ///
    /// If needed, pixel data is converted according to the given pixel type. The desired row
    /// order can be specified, too.
    ///
    /// \param buffer              The buffer to write the pixel data to.
    /// \param buffer_topdown      Indicates whether the buffer has the rows in top-down order.
    /// \param buffer_pixel_type   The pixel type of the buffer.
    /// \param canvas_x            x coordinate of lower left corner of the rectangle in the canvas.
    /// \param canvas_y            y coordinate of lower left corner of the rectangle in the canvas.
    /// \param width               Width of the rectangle to read.
    /// \param height              Height of the rectangle to read.
    /// \param buffer_padding      The padding between subsequent rows in the buffer in bytes.
    /// \param canvas_layer        The layer of the canvas to use.
    /// \return                    \c true in case of success, \c false in case of errors, e.g.,
    ///                            no valid canvas, no pixel type conversion, or invalid parameters.
    bool read_rect(
        mi::Uint8* buffer,
        bool buffer_topdown,
        Pixel_type buffer_pixel_type,
        mi::Uint32 canvas_x,
        mi::Uint32 canvas_y,
        mi::Uint32 width,
        mi::Uint32 height,
        mi::Uint32 buffer_padding = 0,
        mi::Uint32 canvas_layer = 0) const;

    /// Writes a rectangular area of pixels from a caller-specified buffer into the canvas.
    ///
    /// If needed, pixel data is converted from the given pixel type. The row order of the buffer
    /// can be specified, too.
    ///
    /// \param buffer              The buffer to read the pixel data from.
    /// \param buffer_topdown      Indicates whether the buffer has the rows in top-down order.
    /// \param buffer_pixel_type   The pixel type of the buffer.
    /// \param canvas_x            x coordinate of lower left corner of the rectangle in the canvas.
    /// \param canvas_y            y coordinate of lower left corner of the rectangle in the canvas.
    /// \param width               Width of the rectangle to write.
    /// \param height              Height of the rectangle to write.
    /// \param buffer_padding      The padding between subsequent rows in the buffer in bytes.
    /// \param canvas_layer        The layer of the canvas to use.
    /// \return                    \c true in case of success, \c false in case of errors, e.g.,
    ///                            no valid canvas, no pixel type conversion, or invalid parameters.
    bool write_rect(
        const mi::Uint8* buffer,
        bool buffer_topdown,
        Pixel_type buffer_pixel_type,
        mi::Uint32 canvas_x,
        mi::Uint32 canvas_y,
        mi::Uint32 width,
        mi::Uint32 height,
        mi::Uint32 buffer_padding = 0,
        mi::Uint32 canvas_layer = 0);

    /// Reads a single pixel from the canvas.
    ///
    /// To avoid lock contention and cache misses, it is recommended to use this method only from
    /// one thread at a time. (This does not apply to the lockless variant.)
    ///
    /// \param color   The pixel will be returned here.
    /// \param x       The x coordinate of the pixel in the canvas.
    /// \param y       The y coordinate of the pixel in the canvas.
    /// \param z       The z coordinate of the pixel in the canvas.
    /// \return        \c true in case of success,
    ///                \c false in case of errors, e.g., invalid parameters
    bool lookup( mi::math::Color& color, mi::Uint32 x, mi::Uint32 y, mi::Uint32 z = 0) const;

    /// Writes a single pixel to the canvas.
    ///
    /// To avoid lock contention and cache misses, it is recommended to use this method only from
    /// one thread at a time. (This does not apply to the lockless variant.)
    ///
    /// \param color   The pixel to store.
    /// \param x       The x coordinate of the pixel in the canvas.
    /// \param y       The y coordinate of the pixel in the canvas.
    /// \param z       The z coordinate of the pixel in the canvas.
    /// \return        \c true in case of success,
    ///                \c false in case of errors, e.g., invalid parameters
    bool store( const mi::math::Color& color, mi::Uint32 x, mi::Uint32 y, mi::Uint32 z = 0);

private:
    /// The wrapped canvas.
    mi::base::Handle<mi::neuraylib::ICanvas> m_canvas;
    /// The tiles (cached, lazily initialized), the size is m_nr_of_tiles.
    mutable std::vector<mi::base::Handle<mi::neuraylib::ITile> > m_tiles;
    /// Lock for m_tiles.
    mutable mi::base::Lock m_tiles_lock;
    /// Indicates whether m_tiles_lock is to be used at all.
    bool m_lockless;

    /// The number of layers (cached).
    mi::Uint32 m_nr_of_layers;
    /// The width of the canvas (cached).
    mi::Uint32 m_canvas_width;
    /// The height of the canvas (cached).
    mi::Uint32 m_canvas_height;
    /// The pixel type of the canvas (cached).
    Pixel_type m_canvas_pixel_type;
    /// The tile width of the canvas (cached).
    mi::Uint32 m_tile_width;
    /// The tile height of the canvas (cached).
    mi::Uint32 m_tile_height;
    /// The number of tiles in x direction (cached).
    mi::Uint32 m_nr_of_tiles_x;
    /// The number of tiles in y direction (cached).
    mi::Uint32 m_nr_of_tiles_y;
    /// The total number of tiles (cached).
    mi::Uint32 m_nr_of_tiles;
};

} // namespace IMAGE

} // namespace MI

#endif // MI_IO_IMAGE_IMAGE_I_IMAGE_ACCESS_CANVAS_H
