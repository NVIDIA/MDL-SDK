/***************************************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
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

#ifndef IO_SCENE_DBIMAGE_I_DBIMAGE_H
#define IO_SCENE_DBIMAGE_I_DBIMAGE_H

#include <vector>

#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>
#include <base/system/main/access_module.h>
#include <io/scene/scene/i_scene_scene_element.h>

namespace mi { namespace neuraylib { class IReader; class ICanvas; } }

namespace MI {

namespace IMAGE { class IMipmap; }
namespace SERIAL { class Serializer; class Deserializer; }

namespace DBIMAGE {

/// An image set is used as high-level representation to construct instances of the Image class.
///
/// It can represent a single "classic" texture, an ordered set of uvtiles, an animated texture,
/// and a combination of both, i.e., animated uvtiles.
///
/// \note Frame numbers are not necessarily consecutive, there can be missing frames. Do not
///       confuse the zero-based frame ID with the frame number.
///
/// \note Uv-tiles do not necessarily fill an entire rectangle of u/v indices, there can be missing
///       uvtiles.
///
/// \note Do not confuse "uvtile" with the tiles of a canvas.
class Image_set : public mi::base::Interface_implement<mi::base::IInterface>
{
public:
    /// Returns \c true if the image set is contained in an MDL archive or an MDLE.
    virtual bool is_mdl_container() const  = 0;

    /// Returns the original filename.
    ///
    /// Returns an empty string if the set is not file-based.
    virtual const char* get_original_filename() const = 0;

    /// Returns the container filename of the image set.
    ///
    /// Returns an empty string if the set is not container-based.
    virtual const char* get_container_filename() const = 0;

    /// Returns the absolute MDL file path of the image set.
    ///
    /// Returns an empty string if this image set is not an MDL resource.
    virtual const char* get_mdl_file_path() const = 0;

    /// Returns the selector, or \p NULL if there is none.
    virtual const char* get_selector() const = 0;

    /// Returns the image format, or the empty string if not available.
    virtual const char* get_image_format() const = 0;

    /// Indicates whether the image set results from a sequence marker in the MDL file path/original
    /// filename, or (if there is no MDL file path/original filename) whether it has more than one
    /// frame.
    ///
    /// The return value \c false implies that there is a single frame with frame number 0.
    virtual bool is_animated() const = 0;

    /// Indicates whether the image set results from a uv-tile marker in the MDL file path/original
    /// filename, or (if there is no MDL file path/original filename) whether it has a frame with
    /// more than one uv-tile.
    ///
    /// The return value \c false implies that there is a single uv-tile (per frame) with u- and v-
    /// coordinates of 0.
    virtual bool is_uvtile() const = 0;

    /// Returns the number of frames in the image set. Never zero.
    virtual mi::Size get_length() const = 0;

    /// Returns the frame number for an element of the image set.
    ///
    /// This function is strictly monotonically increasing.
    ///
    /// \param    Returns -1 if \p f is out of bounds.
    virtual mi::Size get_frame_number( mi::Size f) const = 0;

    /// Returns the number of tiles in the frame \p f. Never zero.
    virtual mi::Size get_frame_length( mi::Size f) const = 0;

    /// Returns the (u,v) coordinates for an element of the image set.
    ///
    /// \param      f  The frame ID.
    /// \param      i  The uv-tile ID.
    /// \param[out] u  The u coordinate corresponding to \p f and \p i.
    /// \param[out] v  The v coordinate corresponding to \p f and \p i.
    virtual void get_uvtile_uv( mi::Size f, mi::Size i, mi::Sint32 &u, mi::Sint32 &v) const = 0;

    /// Returns the resolved file name for frame \p f, uvtile \p i.
    ///
    /// Returns an empty string if this uv-tile is not file-based.
    virtual const char* get_resolved_filename( mi::Size f, mi::Size i) const = 0;

    /// Returns the container member name for frame \p f, uvtile \p i.
    ///
    /// Returns an empty string if this uv-tile is not container-based.
    virtual const char* get_container_membername( mi::Size f, mi::Size i) const = 0;

    /// Returns a reader for frame \p f, uvtile \p i.
    ///
    /// Returns \c NULL if not supported.
    virtual mi::neuraylib::IReader* open_reader( mi::Size f, mi::Size i) const = 0;

    /// Returns a canvas for frame \p f, uvtile \p i.
    ///
    /// Returns \c NULL if not supported.
    virtual mi::neuraylib::ICanvas* get_canvas( mi::Size f, mi::Size i) const = 0;

    /// Creates a mipmap for frame \p f, uvtile \p i.
    ///
    /// \param      f             The frame ID.
    /// \param      i             The uv-tile ID.
    /// \param[out] errors        The error codes have the following meaning:
    ///                           -   0: Success.
    ///                           -  -1: Invalid reader.
    ///                           -  -3: No image plugin found to handle the data.
    ///                           -  -5: Failure to open the file.
    ///                           -  -6: The reader does not support absolute access.
    ///                           -  -7: The image plugin failed to import the data.
    ///                           - -10: Failure to apply the given selector.
    /// \return                   The requested mipmap, or a dummy mipmap with a 1x1 pink pixel in
    ///                           case of errors.
    IMAGE::IMipmap* create_mipmap( mi::Size f, mi::Size i, mi::Sint32& errors) const;
};

/// Represents the pixel data of an uv-tile plus the corresponding coordinates.
///
/// Part of the low-level representation of images. Used by Image and Image_impl. Passed as argument
/// to non-trivial constructor of Image_impl.
struct Uvtile
{
    /// The mipmap referenced by this uv-tile.
    ///
    /// The handle is always invalid when this struct is part the Image class. The handle is always
    /// valid when this struct is part of the Image_impl class.
    mi::base::Handle<IMAGE::IMipmap> m_mipmap;

    /// The u coordinate of the uv-tile.
    mi::Sint32 m_u = 0;

    /// The v coordinate of the uv-tile.
    mi::Sint32 m_v = 0;
};

/// Allows to look up the uv-tile ID for a given position (u,v).
///
/// Part of the low-level representation of images. Used by Image and Image_impl. Passed as argument
/// to non-trivial constructor of Image_impl.
struct Uv_to_id
{
    mi::Uint32 m_count_u;               ///< Number of uv-tiles in dimension u
    mi::Uint32 m_count_v;               ///< Number of uv-tiles in dimension v
    mi::Sint32 m_min_u;                 ///< Smallest uv-tile position in dimension u.
    mi::Sint32 m_min_v;                 ///< Smallest uv-tile position in dimension v.
    std::vector<mi::Uint32> m_ids;      ///< Array of m_count_u * m_count_v IDs.

    /// Default constructor.
    ///
    /// Creates an empty array with a single uv-tile.
    Uv_to_id() : m_count_u( 1), m_count_v( 1), m_min_u( 0), m_min_v( 0), m_ids{ 0} { }

    /// Constructor.
    ///
    /// Creates max_u-min_u+1 times max_v-min_v+1 uv-tiles. If this is exactly one uv-tile, its ID
    /// is set to 0. Otherwise, all indices are initially set to ~0u and need to be set properly
    /// via #set() below.
    Uv_to_id( mi::Sint32 min_u, mi::Sint32 max_u, mi::Sint32 min_v, mi::Sint32 max_v);

    /// Sets the ID for the uv-tile at position (u,v)
    ///
    /// Intended to be used with the non-trivial constructor above. Does not allow to change an
    /// already set ID.
    ///
    /// Returns \c true in case of success, \c false if \p u or \p v are out of bounds, or if the
    /// ID has already been set.
    bool set( mi::Sint32 u, mi::Sint32 v, mi::Uint32 id);

    /// Returns the ID for the uv-tile at position (u,v)
    mi::Uint32 get( mi::Sint32 u, mi::Sint32 v) const;
};

/// Represents the filenames related to an uv-tile.
///
/// Part of the low-level representation of images. Only used by Image. Not used by Image_impl.
/// Therefore not part of Uvtile.
struct Uvfilenames
{
    /// The file that contains the data of this uv-tile.
    ///
    /// Non-empty exactly for file-based uv-tiles.
    ///
    /// This is the filename as it has been resolved in set_mipmap() or deserialize().
    std::string m_resolved_filename;

    /// The container member that contains the data of this DB element.
    ///
    /// Non-empty exactly for container-based uv-tiles.
    std::string m_container_membername;

    /// The resolved container file name (including the container name) for this image.
    ///
    /// Non-empty exactly for container-based uv-tiles.
    std::string m_resolved_container_membername;
};

/// All information about a frame, with exception of Frame_filenames.
struct Frame
{
    mi::Size m_frame_number = 0;    ///< Frame number of this frame.
    std::vector<Uvtile> m_uvtiles;  ///< Uv-tiles of this frame.
    Uv_to_id m_uv_to_id;            ///< Maps UV coordinates to IDs.
};

/// Vector of frames.
using Frames = std::vector<Frame>;

/// Maps frame numbers to frame IDs (see Frames above).
using Frame_to_id = std::vector<mi::Size>;

/// Uv filenames for each frame.
///
/// Since Uvfilenames is not part of Uvtile, Frame_filenames is not part of Frame.
using Frame_filenames = std::vector<Uvfilenames>;

/// Uv filenames for all frame.
///
/// Since Uvfilenames is not part of Uvtile, Frames_filenames is not part of Frames.
using Frames_filenames = std::vector<Frame_filenames>;

class Image_impl;

/// The class ID for the #Image class.
static constexpr SERIAL::Class_id ID_IMAGE = 0x5f496d67; // '_Img'

/// \note Frame numbers are not necessarily consecutive, there can be missing frames. Do not
///       confuse the zero-based frame ID with the frame number.
///
/// \note Uv-tiles do not necessarily fill an entire rectangle of u/v indices, there can be missing
///       uvtiles.
///
class Image : public SCENE::Scene_element<Image, ID_IMAGE>
{
public:
    /// Default constructor.
    ///
    /// Sets a memory-based dummy mipmap with a 1x1 canvas with a pink pixel.
    Image();

    /// Copy constructor.
    ///
    /// \note The copy constructor actually does not copy the image, but creates a dummy mipmap with
    ///       a 1x1 canvas with a pink pixel (same as after default construction). See
    ///       #mi::neuraylib::IImage for the rationale.
    Image( const Image& other);

    Image& operator=( const Image&) = delete;

    /// Destructor.
    ///
    /// Explicit trivial destructor because the implicitly generated one requires the full
    /// definition of IMAGE::IMipmap.
    ~Image();

    /// Imports a mipmap from a file.
    ///
    /// Uvtiles/animated textures in containers are not supported.
    ///
    /// \param original_filename     The filename of the mipmap. The resource search paths are
    ///                              used to locate the file.
    /// \param selector              The selector (or \c NULL).
    /// \param impl_hash             Hash of the data in the implementation class. Use {0,0,0,0} if
    ///                              hash is not known.
    /// \return
    ///                              -   0: Success.
    ///                              -  -3: No image plugin found to handle the file.
    ///                              -  -4: Failure to resolve the given filename, e.g., the file
    ///                                     does not exist.
    ///                              -  -5: Failure to open the file.
    ///                              -  -7: The image plugin failed to import the file.
    ///                              - -10: Failure to apply the given selector.
    Sint32 reset_file(
        DB::Transaction* transaction,
        const std::string& original_filename,
        const char* selector,
        const mi::base::Uuid& impl_hash);

    /// Imports a mipmap from a reader.
    ///
    /// \param reader                The reader for the mipmap.
    /// \param image_format          The image format.
    /// \param selector              The selector (or \c NULL).
    /// \param impl_hash             Hash of the data in the implementation class. Use {0,0,0,0} if
    ///                              hash is not known.
    /// \return
    ///                              -   0: Success.
    ///                              -  -1: Invalid reader.
    ///                              -  -3: No image plugin found to handle the data.
    ///                              -  -6: The reader does not support absolute access.
    ///                              -  -7: The image plugin failed to import the data.
    ///                              - -10: Failure to apply the given selector.
    Sint32 reset_reader(
        DB::Transaction* transaction,
        mi::neuraylib::IReader* reader,
        const char* image_format,
        const char* selector,
        const mi::base::Uuid& impl_hash);

    /// Imports mipmaps according to an image set.
    ///
    /// \param image_set             The image set to use.
    /// \param impl_hash             Hash of the data in the implementation class. Use {0,0,0,0} if
    ///                              hash is not known.
    /// \return
    ///                              -   0: Success.
    ///                              -  -1: Invalid image set.
    ///                              -  -3: No image plugin found to handle the data.
    ///                              -  -5: Failure to open the file.
    ///                              -  -6: The reader does not support absolute access.
    ///                              -  -7: The image plugin failed to import the data.
    ///                              - -10: Failure to apply the given selector.
    ///                              - -12: Repeated u/v coordinates (per frame).
    ///                              - -99: Inconsistent image set (neither file-, nor container-,
    ///                                     nor reader-, nor canvas-based).
    Sint32 reset_image_set(
        DB::Transaction* transaction,
        const Image_set* image_set,
        const mi::base::Uuid& impl_hash);

    /// Sets a memory-based mipmap.
    ///
    /// Actually, the mipmap might not be memory-based, but it will be treated as if it was a
    /// memory-based mipmap, in particular for serialization purposes.
    ///
    /// A \c NULL pointer can be passed to restore the state after default construction.
    ///
    /// \param selector              The selector (or \c NULL). Not applied to \p mipmap, only for
    ///                              information.
    /// \param impl_hash             Hash of the data in the implementation class. Use {0,0,0,0} if
    ///                              hash is not known.
    void set_mipmap(
        DB::Transaction* transaction,
        IMAGE::IMipmap* mipmap,
        const char* selector,
        const mi::base::Uuid& impl_hash);

    /// Returns the mipmap referenced by the given uv-tile.
    ///
    /// \param frame_id   The frame ID of the mimap.
    /// \param uvtile_id  The uv-tile ID of the mimap.
    /// \return           Returns the requested mipmap, or \c NULL if \p frame_id or \p uvtile_id
    ///                   is out of bounds. In contrast to other resources, the method does \em not
    ///                   return \c NULL if #is_valid() returns \c false, but a dummy mipmap.
    const IMAGE::IMipmap* get_mipmap(
        DB::Transaction* transaction, mi::Size frame_id, mi::Size uvtile_id) const;

    /// Returns the resolved filename of the given uv-tile.
    ///
    /// \param frame_id   The frame ID of the mimap.
    /// \param uvtile_id  The uv-tile ID of the mipmap.
    /// \return           The resolved filename of the referenced uv-tile, or the empty string if
    ///                   the uv-tile is not file-based (including failure to resolve the filename).
    const std::string& get_filename( mi::Size frame_id, mi::Size uvtile_id) const;

    /// Returns the original filename of the referenced mipmap.
    ///
    /// \return   The original filename as passed to #reset_file(), or the empty  string if the
    ///           mipmap is not file-based.
    const std::string& get_original_filename() const;

    /// Returns the absolute MDL file path of the referenced mipmap.
    ///
    /// \return   The absolute MDL file path, or the empty string if not available.
    const std::string& get_mdl_file_path() const;

    /// Returns the selector of the referenced mipmap.
    ///
    /// \return   The selector, or the empty string if not available.
    const std::string& get_selector() const;

    /// Indicates whether the referenced mipmap represents a cubemap.
    bool get_is_cubemap() const { return m_cached_is_cubemap; }

    /// Indicates whether this image references a valid mipmap.
    ///
    /// After default construction and after set_mipmap() is called with a \c NULL pointer this
    /// image references a dummy mipmap with a 1x1 canvas with a pink pixel.
    bool is_valid() const { return m_cached_is_valid; }

    /// Indicates whether the image set results from a sequence marker in the MDL file path/original
    /// filename, or (if there is no MDL file path/original filename) whether it has more than one
    /// frame.
    ///
    /// The return value \c false implies that there is a single frame with frame number 0.
    bool is_animated() const { return m_cached_is_animated; }

    /// Indicates whether the image set results from a uv-tile marker in the MDL file path/original
    /// filename, or (if there is no MDL file path/original filename) whether it has a frame with
    /// more than one uv-tile.
    ///
    /// The return value \c false implies that there is a single uv-tile (per frame) with u- and v-
    /// coordinates of 0.
    bool is_uvtile() const { return m_cached_is_uvtile; }

    /// Returns the number of frames in the image set. Never zero.
    mi::Size get_length() const { return m_cached_frames.size(); }

    /// Returns the frame number for a given frame ID, or -1 if \p frame_id is out of bounds.
    ///
    /// This function is strictly monotonically increasing.
    mi::Size get_frame_number( mi::Size frame_id) const;

    /// Returns the frame ID for a given frame number, or -1 if \p frame_number is not a valid
    /// frame.
    mi::Size get_frame_id( mi::Size frame_number) const;

    /// Returns the number of uv-tiles of this frame (non-zero), or 0 if \p frame_id is out of
    /// bounds.
    mi::Size get_frame_length( mi::Size frame_id) const;

    /// Returns the ranges of u and v coordinates, or all 0 if \p frame_id is out of bounds.
    void get_uvtile_uv_ranges(
        mi::Size frame_id,
        mi::Sint32& min_u,
        mi::Sint32& min_v,
        mi::Sint32& max_u,
        mi::Sint32& max_v) const;

    /// Returns the u and v coordinates for a given uv-tile.
    ///
    /// \param frame_id    The frame ID of the uv-tile.
    /// \param uvtile_id   The uv-tile ID of the uv-tile.
    /// \param[out] u      The u coordinate corresponding to \p uvtile_id.
    /// \param[out] v      The v coordinate corresponding to \p uvtile_id.
    /// \return            0 on success, -1 if \p frame_id or \p uvtile_id is out of bounds.
    mi::Sint32 get_uvtile_uv(
        mi::Size frame_id, Size uvtile_id, Sint32& u, Sint32& v) const;

    /// Returns the uv-tile ID for the uv-tile at given coordinates.
    ///
    /// \param frame_id    The frame ID of the uv-tile.
    /// \param u           The u coordinate of the uv-tile.
    /// \param v           The v coordinate of the uv-tile.
    /// \return            The corresponding uv-tile ID or -1 of there is no uv-tile with the given
    ///                    coordinates.
    mi::Size get_uvtile_id( mi::Size frame_id, Sint32 u, Sint32 v) const;

    // methods of SERIAL::Serializable

    const SERIAL::Serializable* serialize( SERIAL::Serializer* serializer) const;

    SERIAL::Serializable* deserialize( SERIAL::Deserializer* deserializer);

    void dump() const;

    // methods of DB::Element_base

    size_t get_size() const;

    DB::Journal_type get_journal_flags() const;

    Uint bundle( DB::Tag* results, Uint size) const { return 0; }

    // methods of SCENE::Scene_element_base

    void get_scene_element_references( DB::Tag_set* result) const;

    // internal methods

    /// Indicates whether this mipmap is file-based.
    bool is_file_based() const;

    /// Indicates whether this mipmap is container-based.
    bool is_container_based() const { return !m_resolved_container_filename.empty(); }

    /// Indicates whether this mipmap is memory-based.
    bool is_memory_based() const { return !is_file_based() && !is_container_based(); }

    /// Returns the container file name for container-based mipmaps, and the empty string otherwise.
    const std::string& get_container_filename() const { return m_resolved_container_filename; }

    /// Returns the container member name for container-based mipmaps, and the empty string
    /// otherwise.
    const std::string& get_container_membername( mi::Size frame_id, mi::Size uvtile_id) const;

    /// Returns the resolved file name for container-based mipmaps (including the container name),
    /// and the empty string otherwise.
    const std::string& get_resolved_container_membername(
        mi::Size frame_id, mi::Size uvtile_id) const;

    /// Returns the tag of the implementation class.
    ///
    /// Might be an invalid tag after default construction.
    DB::Tag get_impl_tag() const { return m_impl_tag; }

    /// Indicates whether a hash for the implementation class is available.
    bool is_impl_hash_valid() const { return m_impl_hash != mi::base::Uuid{0,0,0,0}; }

    /// Returns the hash of the implementation class (or zero-initialized hash if invalid).
    const mi::base::Uuid& get_impl_hash() const { return m_impl_hash; }

    // internal methods

    /// Returns the frames.
    const Frames& get_frames_vector() const { return m_cached_frames; }

    /// Returns frame number to ID mapping.
    const Frame_to_id& get_frame_to_id() const { return m_cached_frame_to_id; }

    /// Access to low level tile mapping, used by material converter
    const mi::Uint32* get_uvtile_mapping(
        mi::Size frame_id,
        mi::Uint32& num_u,
        mi::Uint32& num_v,
        mi::Sint32& offset_u,
        mi::Sint32& offset_v) const;

private:
    /// Searches for files matching the given path or udim/uv-tile pattern
    ///
    /// \param path       Path to resolve.
    /// \param selector   The selector.
    /// \return           Image set containing the resolved filenames for \p path, or \c NULL in
    ///                   case of error.
    static Image_set* resolve_filename( const std::string& path, const char* selector);

    /// Set an image from frames of uv-tiles.
    ///
    /// Implements the common functionality for all \c reset_*() and \c set_*() methods above.
    void reset_shared(
        DB::Transaction* transaction,
        bool is_animated,
        bool is_uvtile,
        const Frames& frames,
        const Frame_to_id& frame_to_id,
        const mi::base::Uuid& impl_hash);

    /// Set up all cached values based on the values in \p impl.
    void setup_cached_values( const Image_impl* impl);

    /// The file (or MDL file path) that contains the data of this DB element.
    ///
    /// Non-empty for file-based images.
    ///
    /// This is the filename as it has been passed into set_mipmap().
    std::string m_original_filename;

    /// The absolute mdl file path of this image
    ///
    /// Non-empty for mdl resource images
    std::string m_mdl_file_path;

    /// The container that contains the data of this DB element
    ///
    /// Non-empty exactly for container-based light images.
    std::string m_resolved_container_filename;

    /// Per-frame, per uv-tile filenames.
    ///
    /// Same size as m_cached_frames.
    Frames_filenames m_frames_filenames;

    /// The selector.
    std::string m_selector;

    /// The implementation class that holds the bulk data and non-filename related properties.
    DB::Tag m_impl_tag;

    /// Hash of the data in the implementation class.
    mi::base::Uuid m_impl_hash;

    // Non-filename related properties from the implementation class (cached here for efficiency).
    //@{

    bool m_cached_is_valid;
    bool m_cached_is_animated;
    bool m_cached_is_uvtile;
    bool m_cached_is_cubemap;

    /// Cached uv-tiles.
    ///
    /// Same size as m_frame_filenames. Positive size exactly for valid instances. m_mipmap members
    /// are never valid (not cached).
    Frames m_cached_frames;

    /// The cached frame number to ID mapping for the vector above.
    Frame_to_id m_cached_frame_to_id;

    //@}
};

/// The class ID for the #Image class.
static constexpr SERIAL::Class_id ID_IMAGE_IMPL = 0x5f496d69; // '_Imi'

/// \note Frame numbers are not necessarily consecutive, there can be missing frames. Do not
///       confuse the zero-based frame ID with the frame number.
///
/// \note Uv-tiles do not necessarily fill an entire rectangle of u/v indices, there can be missing
///       uvtiles.
///
class Image_impl : public SCENE::Scene_element<Image_impl, ID_IMAGE_IMPL>
{
public:
    /// Default constructor.
    ///
    /// Should only be used for derserialization.
    Image_impl();

    /// Constructor.
    ///
    /// \p frames must not be empty. The \c m_mipmap handles in \p frames must be valid.
    Image_impl(
        bool is_animated, bool is_uvtile, const Frames& frames, const Frame_to_id& frame_to_id);

    /// Copy constructor.
    ///
    /// \note The copy constructor actually does not copy the image, but creates a
    ///       default-constructed instance. See #mi::neuraylib::IImage for the rationale.
    Image_impl( const Image_impl& other);

    Image_impl& operator=( const Image_impl&) = delete;

    /// Destructor.
    ///
    /// Explicit trivial destructor because the implicitly generated one requires the full
    /// definition of IMAGE::IMipmap.
    ~Image_impl();

    /// Returns the mipmap referenced by the given uv-tile.
    ///
    /// \param frame_id    The frame ID of the mipmap.
    /// \param uvtile_id   The uv-tile ID of the mipmap.
    ///
    /// Returns the requested mipmap, or \c NULL if \p frame_id or \p uvtile_id is out of bounds.
    const IMAGE::IMipmap* get_mipmap( mi::Size frame_id, mi::Size uvtile_id) const;

    /// Indicates whether the referenced mipmap represents a cubemap.
    bool get_is_cubemap() const { return m_is_cubemap; }

    /// Indicates whether this image references a valid mipmap.
    ///
    /// The image does not references a valid mipmap after default or copy construction (or if
    /// deserialized from such a state). In all other situations it references a valid mipmap.
    bool is_valid() const { return m_is_valid; }

    /// Indicates whether the image set results from a sequence marker in the MDL file path/original
    /// filename, or (if there is no MDL file path/original filename) whether it has more than one
    /// frame.
    ///
    /// The return value \c false implies that there is a single frame with frame number 0.
    bool is_animated() const { return m_is_animated; }

    /// Indicates whether the image set results from a uv-tile marker in the MDL file path/original
    /// filename, or (if there is no MDL file path/original filename) whether it has a frame with
    /// more than one uv-tile.
    ///
    /// The return value \c false implies that there is a single uv-tile (per frame) with u- and v-
    /// coordinates of 0.
    bool is_uvtile() const { return m_is_uvtile; }

    /// Returns the number of frames in the image set. Never zero.
    mi::Size get_length() const { return m_frames.size(); }

    /// Returns the frame number for a given frame ID, or -1 if \p frame_id is out of bounds.
    ///
    /// This function is strictly monotonically increasing.
    mi::Size get_frame_number( mi::Size frame_id) const;

    /// Returns the frame ID for a given frame number, or -1 if \p frame_number is not a valid
    /// frame.
    mi::Size get_frame_id( mi::Size frame_number) const;

    /// Returns the number of uv-tiles of this frame (non-zero), or 0 if \p frame_id is out of
    /// bounds.
    mi::Size get_frame_length( mi::Size frame_id) const;

    /// Returns the ranges of u and v coordinates, or all 0 if \p frame_id is out of bounds.
    void get_uvtile_uv_ranges(
        mi::Size frame_id,
        mi::Sint32& min_u,
        mi::Sint32& min_v,
        mi::Sint32& max_u,
        mi::Sint32& max_v) const;

    /// Returns the u and v coordinates for a given uv-tile.
    ///
    /// \param frame_id    The frame ID of the uv-tile.
    /// \param uvtile_id   The uv-tile ID of the uv-tile.
    /// \param[out] u      The u coordinate corresponding to \p uvtile_id.
    /// \param[out] v      The v coordinate corresponding to \p uvtile_id.
    /// \return            0 on success, -1 if \p frame_id or \p uvtile_id is out of bounds.
    mi::Sint32 get_uvtile_uv(
        mi::Size frame_id, Size uvtile_id, Sint32& u, Sint32& v) const;

    /// Returns the uv-tile ID for the uv-tile at given coordinates.
    ///
    /// \param frame_id    The frame ID of the uv-tile.
    /// \param u           The u coordinate of the uv-tile.
    /// \param v           The v coordinate of the uv-tile.
    /// \return            The corresponding uv-tile ID or -1 of there is no uv-tile with the given
    ///                    coordinates.
    mi::Size get_uvtile_id( mi::Size frame_id, Sint32 u, Sint32 v) const;

    // methods of SERIAL::Serializable

    const SERIAL::Serializable* serialize( SERIAL::Serializer* serializer) const;

    SERIAL::Serializable* deserialize( SERIAL::Deserializer* deserializer);

    void dump() const;

    // methods of DB::Element_base

    size_t get_size() const;

    DB::Journal_type get_journal_flags() const;

    Uint bundle( DB::Tag* results, Uint size) const { return 0; }

    // methods of SCENE::Scene_element_base

    void get_scene_element_references( DB::Tag_set* result) const { }

    // internal methods

    /// Returns the frames.
    const Frames& get_frames_vector() const { return m_frames; }

    /// Returns frame number to ID mapping.
    const Frame_to_id& get_frame_to_id() const { return m_frame_to_id; }

    /// Access to low level tile mapping, used by material converter.
    const mi::Uint32* get_uvtile_mapping(
        mi::Size frame_id,
        mi::Uint32& num_u,
        mi::Uint32& num_v,
        mi::Sint32& offset_u,
        mi::Sint32& offset_v) const;

private:
    // All members below are essentially const, but cannot be declared as such due to deserialize().

    /// Indicates whether the image is valid (not default- or copy-constructed, nor deserialized
    /// from such a state).
    bool m_is_valid;

    /// Indicates whether the image set results from a sequence marker in the MDL file path/original
    /// filename, or (if there is no MDL file path/original filename) whether it has more than one
    /// frame.
    bool m_is_animated;

    /// Indicates whether the image set results from a uv-tile marker in the MDL file path/original
    /// filename, or (if there is no MDL file path/original filename) whether it has a frame with
    /// more than one uv-tile.
    bool m_is_uvtile;

    /// Indicates whether the image represents a cubemap.
    bool m_is_cubemap;

    /// The frames.
    Frames m_frames;

    /// The frame number to ID mapping for the vector above.
    Frame_to_id m_frame_to_id;
};

} // namespace DBIMAGE

} // namespace MI

#endif // IO_SCENE_DBIMAGE_I_DBIMAGE_H
