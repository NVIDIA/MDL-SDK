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
/// It can represent a single uv-tile as well as an ordered set of such uv-tiles resulting from an
/// udim marker in an MDL file path.
///
/// \note To avoid confusion with tiles of a canvas we use the term uv-tile consistently, even if
///       the image set does not result from udim/uvtile markers.
class Image_set : public mi::base::Interface_implement<mi::base::IInterface>
{
public:
    /// Returns the number of items in the image set. Never zero.
    virtual mi::Size get_length() const = 0;

    /// Returns \c true if the image set results from a udim/uvtile marker in the MDL file path.
    /// Otherwise, returns \c false and #get_length() == 1.
    virtual bool is_uvtile() const = 0;

    /// Returns \c true if the image set is contained in an MDL archive or an MDLE.
    virtual bool is_mdl_container() const  = 0;

    /// Returns the (u,v) coordinates for an element of the image set.
    ///
    /// \param      i  The uv-tile ID.
    /// \param[out] u  The u coordinate corresponding to \p i.
    /// \param[out] v  The v coordinate corresponding to \p i.
    /// \return        \c true if \p i is valid, \c false if \p i is out of bounds.
    virtual void get_uv_mapping( mi::Size i, mi::Sint32 &u, mi::Sint32 &v) const = 0;

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

    /// Returns the i'th resolved file name.
    ///
    /// Returns an empty string if this uv-tile is not file-based.
    virtual const char* get_resolved_filename( mi::Size i) const = 0;

    /// Returns the i'th container member name.
    ///
    /// Returns an empty string if this uv-tile is not container-based.
    virtual const char* get_container_membername( mi::Size i) const = 0;

    /// Returns a reader for the i'th uv-tile.
    ///
    /// Returns \c NULL if not supported.
    virtual mi::neuraylib::IReader* open_reader( mi::Size i) const = 0;

    /// Returns a reader for the i'th uv-tile.
    ///
    /// Returns \c NULL if not supported.
    virtual mi::neuraylib::ICanvas* get_canvas( mi::Size i) const = 0;

    /// Returns the image format, or the empy string if not available.
    virtual const char* get_image_format() const = 0;

    /// Creates a mipmap for the i'th uv-tile.
    ///
    /// Never returns \c NULL.
    IMAGE::IMipmap* create_mipmap( mi::Size i) const;
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

/// Represents the filenames related to an uv-tile.
///
/// Part of the low-level representation of images. Only used by Image. Not used by Image_impl.
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

/// Allows to look up the uv-tile ID for a given position (u,v).
///
/// Part of the low-level representation of images. Used by Image and Image_impl. Passed as argument
/// to non-trivial constructor of Image_impl.
struct Uv_to_id
{
    mi::Sint32 m_count_u;               ///< Number of uv-tiles in dimension u
    mi::Sint32 m_count_v;               ///< Number of uv-tiles in dimension v
    mi::Sint32 m_min_u;                 ///< Smallest uv-tile position in dimension u.
    mi::Sint32 m_min_v;                 ///< Smallest uv-tile position in dimension v.
    std::vector<mi::Uint32> m_ids;      ///< Array of m_count_u * m_count_v IDs.

    /// Default constructor.
    ///
    /// Creates an empty array with no uv-tiles.
    Uv_to_id() : m_count_u( 0), m_count_v( 0), m_min_u( 0), m_min_v( 0) { }

    /// Constructor.
    ///
    /// Creates max_u-min_u+1 times max_v-min_v+1 uv-tiles. If this is exactly one uv-tile, its ID
    /// is set to 0. Otherwise, all indices are initially set to ~0u and need to be set properly
    /// via #set() below.
    Uv_to_id( mi::Sint32 min_u, mi::Sint32 max_u, mi::Sint32 min_v, mi::Sint32 max_v);

    /// Sets the ID for the uv-tile at position (u,v)
    ///
    /// Indented to be used with the non-trivial constructor above. Does not allow to change an
    /// already set ID.
    ///
    /// Returns \c true in case of success, \c false if \p u or \p v are out of bounds, or if the
    /// ID has already been set.
    bool set( mi::Sint32 u, mi::Sint32 v, mi::Uint32 id);

    /// Returns the ID for uv-tile at position (u,v)
    mi::Uint32 get( mi::Sint32 u, mi::Sint32 v) const;
};

class Image_impl;

/// The class ID for the #Image class.
static const SERIAL::Class_id ID_IMAGE = 0x5f496d67; // '_Img'

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
    /// \param original_filename     The filename of the mipmap. The resource search paths are
    ///                              used to locate the file.
    /// \param impl_hash             Hash of the data in the implementation class. Use {0,0,0,0} if
    ///                              hash is not known.
    /// \return
    ///                              -  0: Success.
    ///                              - -2: Failure to resolve the given filename, e.g., the file
    ///                                    does not exist.
    ///                              - -3: Failure to open the file.
    ///                              - -4: No image plugin found to handle the file.
    ///                              - -5: The image plugin failed to import the file.
    Sint32 reset_file(
        DB::Transaction* transaction,
        const std::string& original_filename,
        const mi::base::Uuid& impl_hash);

    /// Imports a mipmap from a reader.
    ///
    /// \param reader                The reader for the mipmap.
    /// \param image_format          The image format.
    /// \param impl_hash             Hash of the data in the implementation class. Use {0,0,0,0} if
    ///                              hash is not known.
    /// \return
    ///                              -  0: Success.
    ///                              - -3: Invalid reader, or the reader does not support absolute
    ///                                    access.
    ///                              - -4: No image plugin found to handle the data.
    ///                              - -5: The image plugin failed to import the data.
    Sint32 reset_reader(
        DB::Transaction* transaction,
        mi::neuraylib::IReader* reader,
        const char* image_format,
        const mi::base::Uuid& impl_hash);

    /// Imports mipmaps according to an image set.
    ///
    /// The image set can either describe a single image file name and reader,
    /// or a set of texture-atlas image names, associated uvs and readers.
    ///
    /// \param image_set             The image set to use.
    /// \param impl_hash             Hash of the data in the implementation class. Use {0,0,0,0} if
    ///                              hash is not known.
    /// \return
    ///                              -  0: Success.
    ///                              - -1: The image set is \c NULL or empty.
    ///                              - -2: The image plugin failed to import the data.
    Sint32 reset_image_set(
        DB::Transaction* transaction,
        const Image_set* image_set,
        const mi::base::Uuid& impl_hash);

    /// Sets a memory-based mipmap.
    ///
    /// Actually, the mipmap might not be memory-based, but it will be treated as if it was a
    /// memory-based mipmap in particular, for serialization purposes).
    ///
    /// A \c NULL pointer can be passed to restore the state after default construction.
    ///
    /// \param impl_hash             Hash of the data in the implementation class. Use {0,0,0,0} if
    ///                              hash is not known.
    void set_mipmap(
        DB::Transaction* transaction,
        IMAGE::IMipmap* mipmap,
        const mi::base::Uuid& impl_hash);

    /// Returns the mipmap referenced by the given uv-tile.
    ///
    /// \param uvtile_id  The uv-tile ID of the mimap.
    /// \return           Returns the requested mipmap, or \c NULL if \p uvtile_id is out of bounds.
    ///                   In contrast to other resources, the method does \em not return \c NULL if
    ///                   #is_valid() returns \c false, but a dummy mipmap.
    const IMAGE::IMipmap* get_mipmap( DB::Transaction* transaction, mi::Uint32 uvtile_id = 0) const;

    /// Returns the resolved filename of the given uv-tile.
    ///
    /// \param uvtile_id   The uv-tile ID of the canvas.
    /// \return   The resolved filename of the referenced uv-tile, or the empty string if the
    ///           uv-tile is not file-based (including failure to resolve the filename).
    const std::string& get_filename( mi::Uint32 uvtile_id = 0) const;

    /// Returns the original filename of the referenced mipmap.
    ///
    /// \return   The original filename as passed to #reset_file(), or the empty  string if the
    ///           mipmap is not file-based.
    const std::string& get_original_filename() const;

    /// Returns the absolute MDL file path of the referenced mipmap.
    ///
    /// \return   The absolute MDL file path, or the empty string if not available.
    const std::string& get_mdl_file_path() const;

    /// Indicates whether the referenced mipmap represents a cubemap.
    bool get_is_cubemap() const { return m_cached_is_cubemap; }

    /// Indicates whether this image references a valid mipmap.
    ///
    /// After default construction and after set_mipmap() is called with a \c NULL pointer this
    /// image references a dummy mipmap with a 1x1 canvas with a pink pixel.
    bool is_valid() const { return m_cached_is_valid; }

    /// Returns \c true if the image set results from a uvtil/udim marker.
    bool is_uvtile() const { return m_cached_is_uvtile; }

    /// Returns the ranges of u and v coordinates (or all values zero if #is_uvtile() returns
    /// \c false).
    void get_uvtile_uv_ranges(
        mi::Sint32& min_u, mi::Sint32& min_v, mi::Sint32& max_u, mi::Sint32& max_v) const;

    /// Returns the number of uv-tiles of this image.
    mi::Size get_uvtile_length() const;

    /// Returns the u and v coordinates for a given uv-tile.
    ///
    /// \param uvtile_id   The uv-tile ID.
    /// \param[out] u      The u coordinate corresponding to \p uvtile_id.
    /// \param[out] v      The v coordinate corresponding to \p uvtile_id.
    /// \return            0 on success, -1 if \p uvtile_id is out of range.
    mi::Sint32 get_uvtile_uv( Uint32 uvtile_id, Sint32& u, Sint32& v) const;

    // Returns the uv-tile ID for the uv-tile at given coordinates.
    ///
    /// \param u           The u coordinate of the uv-tile.
    /// \param v           The v coordinate of the uv-tile.
    /// \return            The corresponding uv-tile ID or -1 of there is no uv-tile with the given
    ///                    coordinates.
    mi::Uint32 get_uvtile_id( Sint32 u, Sint32 v) const;

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
    bool is_file_based() const
    { return !m_uvfilenames.empty() && !m_uvfilenames[0].m_resolved_filename.empty(); }

    /// Indicates whether this mipmap is container-based.
    bool is_container_based() const { return !m_resolved_container_filename.empty(); }

    /// Indicates whether this mipmap is memory-based.
    bool is_memory_based() const { return !is_file_based() && !is_container_based(); }

    /// Returns the container file name for container-based mipmaps, and \c NULL otherwise.
    const std::string& get_container_filename() const { return m_resolved_container_filename; }

    /// Returns the container member name for container-based mipmaps, and \c NULL otherwise.
    const std::string& get_container_membername( mi::Uint32 uvtile_id = 0) const ;

    /// Returns the resolved file name for container-based mipmaps (including the container name),
    /// and \c NULL otherwise.
    const std::string& get_resolved_container_membername( mi::Uint32 uvtile_id = 0) const;

    /// Retuns the tag of the implementation class.
    ///
    /// Might be an invalid tag after default construction.
    DB::Tag get_impl_tag() const { return m_impl_tag; }

    /// Indicates whether a hash for the implementation class is available.
    bool is_impl_hash_valid() const { return m_impl_hash != mi::base::Uuid{0,0,0,0}; }

    /// Returns the hash of the implementation class (or zero-initialized hash if invalid).
    const mi::base::Uuid& get_impl_hash() const { return m_impl_hash; }

    /// Access to low level tile mapping, used by material converter
    const unsigned int* get_tile_mapping(
        mi::Uint32& num_u,
        mi::Uint32& num_v,
        mi::Sint32& offset_u,
        mi::Sint32& offset_v) const;

private:
    /// Searches for files matching the given path or udim/uv-tile pattern
    ///
    /// \param path path to resolve
    /// \return Image_set containing the resolved filenames for or \c NULL in case of error
    static Image_set* resolve_filename( const std::string& path);

    /// Set an image from uv-tiles.
    ///
    /// Implements the common functionality for all \c reset_*() and \c set_*() methods above.
    void reset_shared(
        DB::Transaction* transaction,
        bool is_uvtile,
        const std::vector<Uvtile>& uvtiles,
        const Uv_to_id& uv_to_id,
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

    /// Per-uv-tile filenames.
    ///
    /// Same size as m_cached_uvtiles. Positive size exactly for valid instances.
    std::vector<Uvfilenames> m_uvfilenames;

    /// The implementation class that holds the bulk data and non-filename related properties.
    DB::Tag m_impl_tag;

    /// Hash of the data in the implementation class.
    mi::base::Uuid m_impl_hash;

    // Non-filename related properties from the implementation class (cached here for efficiency).
    //@{

    bool                m_cached_is_valid;
    bool                m_cached_is_uvtile;
    bool                m_cached_is_cubemap;

    /// Cached uv-tiles.
    ///
    /// Same size as m_uvfilenames. Positive size exactly for valid instances. m_mipmap members are
    /// never valid (not cached).
    std::vector<Uvtile> m_cached_uvtiles;

    Uv_to_id         m_cached_uv_to_id;

    //@}
};

/// The class ID for the #Image class.
static const SERIAL::Class_id ID_IMAGE_IMPL = 0x5f496d69; // '_Imi'

class Image_impl : public SCENE::Scene_element<Image_impl, ID_IMAGE_IMPL>
{
public:
    /// Default constructor.
    ///
    /// Should only be used for derserialization.
    Image_impl();

    /// Constructor.
    ///
    /// \p uvtiles must not be empty. The \c m_mipmap handles in \p uvtiles must be valid.
    Image_impl( bool is_uvtile, const std::vector<Uvtile>& uvtiles, const Uv_to_id& uv_to_id);

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
    /// \param uvtile_id   The uv-tile ID of the mipmap.
    ///
    /// Returns \c NULL if a uv-tile for the provided \p uvtile_id does not exist.
    const IMAGE::IMipmap* get_mipmap( mi::Uint32 uvtile_id = 0) const;

    /// \return           Returns the requested mipmap, or \c NULL if \p uvtile_id is out of bounds.
    bool get_is_cubemap() const { return m_is_cubemap; }

    /// Indicates whether this image references a valid mipmap.
    ///
    /// The image does not references a valid mipmap after default or copy construction (or if
    /// deserialized from such a state). In all other situations it references a valid mipmap.
    bool is_valid() const { return m_is_valid; }

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

    /// Returns \c true if the image set results from a uvtil/udim marker.
    bool is_uvtile() const { return m_is_uvtile; }

    /// Returns the ranges of u and v coordinates (or all values zero if #is_uvtile() returns
    /// \c false).
    void get_uvtile_uv_ranges(
        mi::Sint32& min_u, mi::Sint32& min_v, mi::Sint32& max_u, mi::Sint32& max_v) const;

    /// Returns the number of uv-tiles of this image.
    mi::Size get_uvtile_length() const;

    /// Returns the u and v coordinates for a given uv-tile.
    ///
    /// \param uvtile_id   The uv-tile ID.
    /// \param[out] u      The u coordinate corresponding to \p uvtile_id.
    /// \param[out] v      The v coordinate corresponding to \p uvtile_id.
    /// \return            0 on success, -1 if \p uvtile_id is out of range.
    mi::Sint32 get_uvtile_uv( Uint32 uvtile_id, Sint32& u, Sint32& v) const;

    // Returns the uv-tile ID for the uv-tile at given coordinates.
    ///
    /// \param u           The u coordinate of the uv-tile.
    /// \param v           The v coordinate of the uv-tile.
    /// \return            The corresponding uv-tile ID or -1 of there is no uv-tile with the given
    ///                    coordinates.
    mi::Uint32 get_uvtile_id( Sint32 u, Sint32 v) const;

    /// Returns the uv-tiles.
    const std::vector<Uvtile>& get_uvtiles() const { return m_uvtiles; }

    /// Returns the (u,v) to ID mapping.
    const Uv_to_id& get_uv_to_id() const { return m_uv_to_id; }

    /// Access to low level tile mapping, used by material converter
    const unsigned int* get_tile_mapping(
        mi::Uint32& num_u,
        mi::Uint32& num_v,
        mi::Sint32& offset_u,
        mi::Sint32& offset_v) const;

private:
    // All members below are essentially const, but cannot be declared as such due to deserialize().

    /// Indicates whether the image is valid (not default- or copy-constructed, nor deserialized
    /// from such a state).
    bool m_is_valid;

    /// Indicates whether the image represents a texture atlas mapping.
    bool m_is_uvtile;

    /// Indicates whether the image represents a cubemap.
    bool m_is_cubemap;

    /// The uv-tiles.
    std::vector<Uvtile> m_uvtiles;

    /// The (u,v) to ID mapping for the vector above.
    Uv_to_id m_uv_to_id;
};

} // namespace DBIMAGE

} // namespace MI

#endif // IO_SCENE_DBIMAGE_I_DBIMAGE_H
