/***************************************************************************************************
 * Copyright (c) 2006-2024, NVIDIA CORPORATION. All rights reserved.
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
/// \brief      Scene element Image

#ifndef MI_NEURAYLIB_IIMAGE_H
#define MI_NEURAYLIB_IIMAGE_H

#include <mi/neuraylib/iscene_element.h>

namespace mi {

class IArray;

namespace neuraylib {

class ICanvas;
class IReader;

/** \defgroup mi_neuray_misc Miscellaneous
    \ingroup mi_neuray_scene_element

    Miscellaneous scene graph elements, for example, textures, light profiles, BSDF measurements,
    or decals.
*/

/** \addtogroup mi_neuray_misc
@{
*/

/// This interface represents a pixel image file. It supports different pixel types, 2D and 3D image
/// data, and mipmap levels. Its main usage is in textures, see the #mi::neuraylib::ITexture class.
///
/// The image coordinate system has its origin in the lower left corner in the case of 2D image
/// data.
///
/// \par Editing and copying an image
///
/// Note that editing an existing image has unusual semantics that differ from all other DB
/// elements. Usually, when editing a database element, an identical copy of the database element is
/// created (the existing one cannot be used because it might be needed for other transactions,
/// other scopes, or in case the transaction is aborted). For images, this implies a copy of all the
/// pixel data which is very expensive.
///
/// There are only two mutable methods on this interface, #reset_file() and #set_from_canvas();
/// all other methods are const. Both methods eventually replace the entire pixel data anyway.
/// Therefore, when an image is edited, the pixel data is not copied, but replaced by a dummy image
/// of size 1x1. This approach saves the unneeded, but expensive copy of the original pixel data.
/// When afterwards one of two methods above is called, the image uses the correct pixel data
/// again.
///
/// Note that this also affects the results from methods like #resolution_x(), etc. (if you want
/// to know the resolution of an existing image without changing it, you should access the image,
/// not edit it). Furthermore, you might end up with the dummy image if you do not call
/// #reset_file() or #set_from_canvas() (or if these methods fail).
///
/// Note that using the transaction's copy function has the same semantics when used on an image.
/// Thus after copying it is necessary to use either #reset_file() or #set_from_canvas() on the
/// copy.
///
class IImage :
    public base::Interface_declare<0xca59b977,0x30ee,0x4172,0x91,0x53,0xb7,0x70,0x2c,0x6b,0x3a,0x76,
                                   neuraylib::IScene_element>
{
public:
    /// \name Modification of the scene element
    //@{

    /// Sets the image to a file identified by \p filename.
    ///
    /// Note that support for a given image format requires an image plugin capable of handling
    /// that format.
    ///
    /// The filename can include one of the following three uv-tileset markers in the filename:
    /// \c &lt;UDIM&gt;, \c &lt;UVTILE0&gt;, or \c &lt;UVTILE1&gt;. The image refers then to a
    /// whole uv-tileset, a set of images used together as a single large two-dimensional image.
    /// The different markers indicate the different filename conventions that encode where each
    /// image file is placed in the uv texture space.
    ///
    /// <table>
    /// <tr>
    ///   <th>Marker</th>
    ///   <th>Pattern</th>
    ///   <th>(0,0) index</th>
    ///   <th>Convention to format a (u, v)-index</th>
    /// </tr>
    /// <tr>
    ///   <td>&lt;UDIM&gt;</td>
    ///   <td>DDDD</td>
    ///   <td>1001</td>
    ///   <td>UDIM, expands to the four digit number 1000+(u+1+v*10)</td>
    /// </tr>
    /// <tr>
    ///   <td>&lt;UVTILE0&gt;</td>
    ///   <td>"_u"I"_v"I</td>
    ///   <td>_u0_v0</td>
    ///   <td>0-based uv-tileset, expands to "_u"u"_v"v"</td>
    /// </tr>
    /// <tr>
    ///   <td>&lt;UVTILE1&gt;</td>
    ///   <td>"_u"I"_v"I</td>
    ///   <td>_u1_v1</td>
    ///   <td>1-based uv-tileset, expands to "_u"(u+1)"_v"(v+1)"</td>
    /// </tr>
    /// </table>
    ///
    /// The filename can also include a sequence marker for the frame number of animated textures:
    /// \c &lt;#...&gt; (with a non-zero count of \c '#' characters). The marker matches any
    /// non-negative integral number of at most as many digits as there are \c '#' characters in the
    /// sequence marker. The number may have leading zeros, which are ignored for its numerical
    /// interpretation. Multiple occurrences of the same number is undefined behavior (can happen
    /// in case of leading zeros).
    ///
    /// Images without sequences marker are treated as a single frame with frame number 0. Images
    /// without uv-tileset marker are treated as a single uv-tile (per frame) with u- and
    /// v-coordinates of 0.
    ///
    /// \param filename       The filename of the image to load.
    /// \param selector       The selector, or \c NULL. See section 2.3.1 in [\ref MDLLS] for
    ///                       details.
    /// \return
    ///                       -   0: Success.
    ///                       -  -1: Invalid parameters (\c NULL pointer).
    ///                       -  -3: No image plugin found to handle the file.
    ///                       -  -4: Failure to resolve the given filename, e.g., the file does not
    ///                              exist.
    ///                       -  -5: Failure to open the resolved file.
    ///                       -  -7: The image plugin failed to import the file.
    ///                       - -10: Failure to apply the given selector.
    ///
    /// \see #mi::neuraylib::IMdl_factory::create_texture() for a way to create a texture based
    ///      on an MDL file path instead of a filename.
    virtual Sint32 reset_file( const char* filename, const char* selector = 0) = 0;

    /// Sets the image to the data provided by a reader.
    ///
    /// \param reader         The reader that provides the data for the image. The reader needs to
    ///                       support absolute access.
    /// \param image_format   The image format of the data, e.g., \c "jpg". Note that support for a
    ///                       given image format requires an image plugin capable of handling that
    ///                       format.
    /// \param selector       The selector, or \c NULL. See section 2.3.1 in [\ref MDLLS] for
    ///                       details.
    /// \return
    ///                       -   0: Success.
    ///                       -  -1: Invalid parameters (\c NULL pointer).
    ///                       -  -3: No image plugin found to handle the file.
    ///                       -  -6: The reader does not support absolute access.
    ///                       -  -7: The image plugin failed to import the data.
    ///                       - -10: Failure to apply the given selector.
    virtual Sint32 reset_reader(
        IReader* reader, const char* image_format, const char* selector = 0) = 0;

    /// Sets the image to the frame/uv-tile data provided by an array of readers.
    ///
    /// \param reader         A static or dynamic array of structures of type \c Uvtile_reader. Such
    ///                       a structure has the following members:
    ///                       - #mi::Sint32 \b u \n
    ///                         The u-coordinate of this uv-tile.
    ///                       - #mi::Sint32 \b v \n
    ///                         The v-coordinate of this uv-tile.
    ///                       - #mi::Size \b frame \n
    ///                         The frame number of this uv-tile.
    ///                       - #mi::neuraylib::IReader* \b reader \n
    ///                         The reader that provides the data for this uv-tile. The reader needs
    ///                         to support absolute access.
    /// \param image_format   The image format of the data, e.g., \c "jpg". Note that support for a
    ///                       given image format requires an image plugin capable of handling that
    ///                       format.
    /// \param selector       The selector, or \c NULL. See section 2.3.1 in [\ref MDLLS] for
    ///                       details.
    /// \return
    ///                       -   0: Success.
    ///                       -  -1: Invalid parameters (\c NULL pointer).
    ///                       -  -3: No image plugin found to handle the file.
    ///                       -  -6: The reader does not support absolute access.
    ///                       -  -7: The image plugin failed to import the data.
    ///                       - -10: Failure to apply the given selector.
    ///                       - -11: Invalid reader array (empty, wrong element type, or invalid
    ///                              array slots or struct fields).
    ///                       - -12: Repeated u/v coordinates (per frame).
    virtual Sint32 reset_reader(
        IArray* reader, const char* image_format, const char* selector = 0) = 0;

    /// Sets the image to the passed canvas (without sharing).
    ///
    /// \param canvas     The pixel data to be used by this image. Note that the pixel data is
    ///                   copied, not shared. If sharing is intended, use
    ///            #mi::neuraylib::IImage::set_from_canvas(mi::neuraylib::ICanvas*,const char*,bool)
    ///                   instead.
    /// \param selector   The selector, or \c NULL. See section 2.3.1 in [\ref MDLLS] for
    ///                   details. Note that the selector is \em not applied to the canvas, the
    ///                   information is just stored for retrieval via #get_selector().
    /// \return           \c true if the pixel data of this image has been set correctly, and
    ///                   \c false otherwise.
    virtual bool set_from_canvas(
        const ICanvas* canvas, const char* selector = 0) = 0;

    /// Sets the image to the passed canvas (possibly sharing the pixel data).
    ///
    /// \param canvas     The pixel data to be used by this image.
    /// \param selector   The selector, or \c NULL. See section 2.3.1 in [\ref MDLLS] for
    ///                   details. Note that the selector is \em not applied to the canvas, the
    ///                   information is just stored for retrieval via #get_selector().
    /// \param shared     If \c false (the default), the pixel data is copied from \c canvas and the
    ///                   method does the same as
    ///          #mi::neuraylib::IImage::set_from_canvas(const mi::neuraylib::ICanvas*,const char*).
    ///                   If set to \c true, the image uses the canvas directly (doing reference
    ///                   counting on the canvas pointer). You must not modify the canvas content
    ///                   after this call.
    /// \return           \c true if the pixel data of this image has been set correctly, and
    ///                   \c false otherwise.
    virtual bool set_from_canvas(
        ICanvas* canvas, const char* selector = 0, bool shared = false) = 0;

#ifdef MI_NEURAYLIB_DEPRECATED_14_0
    inline bool set_from_canvas( ICanvas* canvas, bool shared)
    { return set_from_canvas( canvas, 0, shared); }
#endif

    /// Sets the frames/uv-tiles of this image to the passed canvases (without sharing).
    ///
    /// \param uvtiles    A static or dynamic array of structures of type \c Uvtile. Such a
    ///                   structure has the following members:
    ///                   - #mi::Sint32 \b u \n
    ///                     The u-coordinate of this uv-tile.
    ///                   - #mi::Sint32 \b v \n
    ///                     The v-coordinate of this uv-tile.
    ///                   - #mi::Size \b frame \n
    ///                     The frame number of this uv-tile.
    ///                   - #mi::neuraylib::ICanvas* \b canvas \n
    ///                     The pixel data to be used for this image. Note that the pixel data is
    ///                     copied, not shared. If sharing is intended, use
    ///                     #mi::neuraylib::IImage::set_from_canvas(mi::IArray*,const char*,bool)
    ///                     instead.
    /// \param selector   The selector, or \c NULL. See section 2.3.1 in [\ref MDLLS] for
    ///                   details. Note that the selector is \em not applied to the canvas, the
    ///                   information is just stored for retrieval via #get_selector().
    /// \return           \c true if the pixel data of this image has been set correctly, and
    ///                   \c false otherwise.
    virtual bool set_from_canvas(
        const IArray* uvtiles, const char* selector = 0) = 0;

    /// Sets the frames/uv-tiles of this image based to the passed canvases (possibly sharing the
    /// pixel data).
    ///
    /// \param uvtiles    A static or dynamic array of structures of type \c Uvtile. Such a
    ///                   structure has the following members:
    ///                   - #mi::Sint32 \b u \n
    ///                     The u-coordinate of this uv-tile.
    ///                   - #mi::Sint32 \b v \n
    ///                     The v-coordinate of this uv-tile.
    ///                   - #mi::Size \b frame \n
    ///                     The frame number of this uv-tile.
    ///                   - #mi::neuraylib::ICanvas* \b canvas \n
    ///                     The pixel data to be used for this image.
    /// \param selector   The selector, or \c NULL. See section 2.3.1 in [\ref MDLLS] for
    ///                   details. Note that the selector is \em not applied to the canvas, the
    ///                   information is just stored for retrieval via #get_selector().
    /// \param shared     If \c false (the default), the pixel data is copied from \c canvas and
    ///                   the method does the same as
    ///          #mi::neuraylib::IImage::set_from_canvas(const mi::neuraylib::ICanvas*,const char*).
    ///                   If set to \c true, the image uses the canvases directly (doing reference
    ///                   counting on the canvas pointers). You must not modify the canvas contents
    ///                   after this call.
    /// \return           \c true if the pixel data of this image has been set correctly, and
    ///                   \c false otherwise.
    virtual bool set_from_canvas(
        IArray* uvtiles, const char* selector = 0, bool shared = false) = 0;

#ifdef MI_NEURAYLIB_DEPRECATED_14_0
    inline bool set_from_canvas( IArray* uvtiles, bool shared)
    { return set_from_canvas( uvtiles, 0, shared); }
#endif

    //@}
    /// \name Methods related to frames of animated textures
    //@{

    /// Indicates whether this image represents an animated texture.
    ///
    /// The return value \c false implies that there is a single frame with frame number 0.
    virtual bool is_animated() const = 0;

    /// Returns the number of frames of this image. Never zero.
    virtual Size get_length() const = 0;

    /// Returns the frame number for a give frame ID.
    ///
    /// This function is strictly monotonically increasing. Frame numbers are not necessarily
    /// consecutive, there can be missing frame numbers.
    ///
    /// \param frame_id    The frame ID of the frame.
    /// \return            The frame number, or -1 if \p frame_id is out of bounds.
    virtual Size get_frame_number( Size frame_id) const = 0;

    /// Returns the frame ID for a given frame number.
    ///
    /// \param frame_number   The frame number of the frame.
    /// \return               The frame ID, or -1 if \p frame_number is not a valid frame number.
    virtual Size get_frame_id( Size frame_number) const = 0;

    //@}
    /// \name Methods related to uvtiles
    //@{

    /// Indicates whether this image represents a uvtile sequence.
    ///
    /// The return value \c false implies that there is a single uv-tile (per frame) with u- and v-
    /// coordinates of 0.
    virtual bool is_uvtile() const = 0;

    /// Returns the number of uv-tiles for a given frame (or 0 if \p frame_id is out of bounds).
    virtual Size get_frame_length( Size frame_id) const = 0;

#ifdef MI_NEURAYLIB_DEPRECATED_12_1
    inline Size get_uvtile_length() const { return get_frame_length( 0); }
#endif

    /// Returns the u- and v- coordinates corresponding to a uv-tile ID.
    ///
    /// \param frame_id    The frame ID of the frame.
    /// \param uvtile_id   The uv-tile ID of the uv-tile.
    /// \param u           The u-coordinate of the uv-tile.
    /// \param v           The v-coordinate of the uv-tile.
    /// \return            0 on success, -1 if \p uvtile_id is out of range.
    virtual Sint32 get_uvtile_uv( Size frame_id, Size uvtile_id, Sint32& u, Sint32& v) const = 0;

#ifdef MI_NEURAYLIB_DEPRECATED_12_1
    inline Sint32 get_uvtile_uv( Uint32 uvtile_id, Sint32& u, Sint32& v) const
    { return get_uvtile_uv( 0, uvtile_id, u, v); }
#endif

    /// Returns the uv-tile ID corresponding to u- and v-coordinates.
    ///
    /// \param frame_id    The frame ID of the frame.
    /// \param u           The u-coordinate of the uv-tile.
    /// \param v           The v-coordinate of the uv-tile..
    /// \return            The uv-tile ID, or -1 of there is no uv-tile with the given coordinates.
    virtual Size get_uvtile_id( Size frame_id, Sint32 u, Sint32 v) const = 0;

#ifdef MI_NEURAYLIB_DEPRECATED_12_1
    inline Uint32 get_uvtile_id( Sint32 u, Sint32 v) const
    { return static_cast<Uint32>( get_uvtile_id( 0, u, v)); }
#endif

    /// Returns the ranges of u- and v-coordinates.
    ///
    /// \param frame_id    The frame ID of the frame.
    /// \param[out] min_u  Smallest u-coordinate for that frame.
    /// \param[out] min_v  Smallest v-coordinate for that frame.
    /// \param[out] max_u  Largest u-coordinate for that frame.
    /// \param[out] max_v  Largest v-coordinate for that frame.
    virtual void get_uvtile_uv_ranges(
        Size frame_id, Sint32& min_u, Sint32& min_v, Sint32& max_u, Sint32& max_v) const = 0;

#ifdef MI_NEURAYLIB_DEPRECATED_12_1
    inline void get_uvtile_uv_ranges(
        Sint32& min_u, Sint32& min_v, Sint32& max_u, Sint32& max_v) const
    { return get_uvtile_uv_ranges( 0, min_u, min_v, max_u, max_v); }
#endif

    //@}
    /// \name Methods to query filenames, selector, and canvases
    //@{

    /// Returns the resolved file name of a mipmap of the image.
    ///
    /// The method returns \c NULL if there is no file associated with the mipmap, e.g., after
    /// default construction, calls to #set_from_canvas(), or failures to resolve the file name
    /// passed to #reset_file().
    ///
    /// \param frame_id    The frame ID of the mipmap.
    /// \param uvtile_id   The uv-tile ID of the mipmap.
    ///
    /// \see #get_original_filename()
    virtual const char* get_filename( Size frame_id, Size uvtile_id) const = 0;

#ifdef MI_NEURAYLIB_DEPRECATED_12_1
    inline const char* get_filename( Uint32 uvtile_id = 0) const
    { return get_filename( 0, uvtile_id); }
#endif

    /// Returns the unresolved file as passed to #reset_file().
    ///
    /// The method returns \c NULL after default construction or calls to #set_from_canvas().
    ///
    /// \see #get_filename()
    virtual const char* get_original_filename() const = 0;

    /// Returns the selector (or \c NULL).
    virtual const char* get_selector() const = 0;

    /// Returns a canvas with the pixel data of the image.
    ///
    /// Note that it is not possible to manipulate the pixel data.
    ///
    /// \param frame_id    The frame ID of the canvas.
    /// \param uvtile_id   The uv-tile ID of the canvas.
    /// \param level       The desired mipmap level. Level 0 is the highest resolution.
    /// \return            A canvas pointing to the pixel data of the image, or \c NULL in case of
    ///                    failure, e.g., because of an invalid uv-tile ID.
    virtual const ICanvas* get_canvas( Size frame_id, Size uvtile_id, Uint32 level) const = 0;

#ifdef MI_NEURAYLIB_DEPRECATED_12_1
    inline const ICanvas* get_canvas( Uint32 level = 0, Uint32 uvtile_id = 0) const
    { return get_canvas( 0, uvtile_id, level); }
#endif

    //@}
    /// \name Properties of the canvases (convenience methods)
    //@{

    /// Returns the pixel type of a mipmap.
    ///
    /// \param frame_id    The frame ID of the mipmap.
    /// \param uvtile_id   The uv-tile ID of the mipmap to get the pixel type for.
    /// \return            The pixel type, or \c NULL in case of an invalid frame ID or uv-tile ID.
    ///
    /// See \ref mi_neuray_types for a list of supported pixel types.
    virtual const char* get_type( Size frame_id, Size uvtile_id) const = 0 ;

#ifdef MI_NEURAYLIB_DEPRECATED_12_1
    inline const char* get_type( Uint32 uvtile_id = 0) const { return get_type( 0, uvtile_id); }
#endif

    /// Returns the number of levels in the mipmap pyramid.
    ///
    /// \param frame_id    The frame ID of the mipmap.
    /// \param uvtile_id   The uv-tile ID of the mipmap to get the number of levels for.
    /// \return            The number of levels, or -1 in case of an invalid frame ID or uv-tile ID.
    virtual Uint32 get_levels( Size frame_id, Size uvtile_id) const = 0;

#ifdef MI_NEURAYLIB_DEPRECATED_12_1
    inline Uint32 get_levels( Uint32 uvtile_id = 0) const { return get_levels( 0, uvtile_id); }
#endif

    /// Returns the horizontal resolution of a canvas.
    ///
    /// \param frame_id    The frame ID of the canvas.
    /// \param uvtile_id   The uv-tile ID of the canvas to get the resolution for.
    /// \param level       The desired mipmap level. Level 0 is the highest resolution.
    /// \return            The horizontal resolution, or -1 in case of an invalid frame ID,
    ///                    uv-tile ID, or level.
    virtual Uint32 resolution_x( Size frame_id, Size uvtile_id, Uint32 level) const = 0;

#ifdef MI_NEURAYLIB_DEPRECATED_12_1
    inline Uint32 resolution_x( Uint32 level = 0, Uint32 uvtile_id = 0) const
    { return resolution_x( 0, uvtile_id, level); }
#endif

    /// Returns the vertical resolution of a canvas.
    ///
    /// \param frame_id    The frame ID of the canvas.
    /// \param uvtile_id   The uv-tile ID of the canvas to get the resolution for.
    /// \param level       The desired mipmap level. Level 0 is the highest resolution.
    /// \return            The vertical resolution, or -1 in case of an invalid frame ID,
    ///                    uv-tile ID, or level.
    virtual Uint32 resolution_y( Size frame_id, Size uvtile_id, Uint32 level) const = 0;

#ifdef MI_NEURAYLIB_DEPRECATED_12_1
    inline Uint32 resolution_y( Uint32 level = 0, Uint32 uvtile_id = 0) const
    { return resolution_y( 0, uvtile_id, level); }
#endif

    /// Returns the number of layers of a canvas.
    ///
    /// \param frame_id    The frame ID of the canvas.
    /// \param uvtile_id   The uv-tile ID of the canvas to get the resolution for.
    /// \param level       The desired mipmap level. Level 0 is the highest resolution.
    /// \return            The number of layers, or -1 in case of an invalid frame ID,
    ///                    uv-tile ID, or level.
    virtual Uint32 resolution_z( Size frame_id, Size uvtile_id, Uint32 level) const = 0;

#ifdef MI_NEURAYLIB_DEPRECATED_12_1
    inline Uint32 resolution_z( Uint32 level = 0, Uint32 uvtile_id = 0) const
    { return resolution_z( 0, uvtile_id, level); }
#endif

    //@}
};

/**@}*/ // end group mi_neuray_misc

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IIMAGE_H
