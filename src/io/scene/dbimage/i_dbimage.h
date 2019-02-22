/***************************************************************************************************
 * Copyright (c) 2011-2019, NVIDIA CORPORATION. All rights reserved.
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

enum Uvtile_mode
{
    MODE_OFF,
    MODE_UDIM,
    MODE_U0_V0,
    MODE_U1_V1
};

/// An interface containing an ordered set of texture atlas names, uvs and associated readers
class Image_set : public
    mi::base::Interface_implement<mi::base::IInterface>
{
public:

    /// Get the number of resolved images.
    virtual mi::Size get_length() const = 0;

    /// If the ordered set represents a texture atlas mapping, returns it, otherwise NULL.
    ///
    /// \param[in]  i  the index
    /// \param[out] u  the u coordinate
    /// \param[out] v  the v coordinate
    ///
    /// \returns true if a mapping is available, false otherwise
    virtual bool get_uv_mapping(mi::Size i, mi::Sint32 &u, mi::Sint32 &v) const = 0;

    /// Get the archive filename of this image set.
    /// Returns an empty string if the set is not archive-based.
    virtual char const * get_archive_filename() const;

    /// Get the original filename of this image set.
    /// Returns an empty string if the set does not have one.
    virtual char const * get_original_filename() const;

    /// Get the absolute mdl file path of this image set.
    /// Returns an empty string if this image is not an mdl resource.
    virtual char const* get_mdl_file_path() const;

    /// Get the i'th MDL url of the image set.
    ///
    /// \param i  the index
    ///
    /// \returns the i'th MDL url of the set or NULL if the index is out of range.
    virtual char const *get_mdl_url(mi::Size i) const;

    /// Get the i'th file name of the ordered set.
    ///
    /// \param i  the index
    ///
    /// \returns the i'th file name of the set or NULL if the index is out of range
    ///          returns an empty string if the image set is mdl archive or memory-based
    virtual char const *get_resolved_filename(mi::Size i) const;

    /// Get the i'th mdl archive member name of the ordered set.
    ///
    /// \param i  the index
    ///
    /// \returns the i'th file name of the set or NULL if the index is out of range.

    virtual char const *get_archive_membername(mi::Size i) const;

    /// Opens a reader for the i'th entry.
    ///
    /// \param i  the index
    ///
    /// \returns a reader for the i'th entry of the set or NULL if the index is out of range.
    virtual mi::neuraylib::IReader *open_reader(mi::Size i) const;

    /// Returns a canvas for the i'th entry.
    ///
    /// \param i  the index
    ///
    /// \returns a canvas for the i'th entry of the set or NULL if the index is out of range.
    virtual mi::neuraylib::ICanvas* get_canvas(mi::Size i) const;

    /// Returns true if the image set represents a texture atlas mapping
    virtual bool is_uvtile() const;

    /// Returns true if the image set is contained in an mdl archive
    virtual bool is_mdl_archive() const;

    /// Returns the image format of the reader based image set
    virtual const char* get_image_format() const;

    /// Creates a mipmap from the description 
    ///
    /// \param i  the index
    /// \returns a mipmap for the i'th entry of the set or NULL if the index is out of range.
    MI::IMAGE::IMipmap* create_mipmap(mi::Size i) const;
};

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

    /// Destructor.
    ///
    /// Explicit trivial destructor because the implicitly generated one requires the full
    /// definition of IMAGE::IMipmap.
    ~Image();

    /// Imports a mipmap from a file.
    ///
    /// \param original_filename     The filename of the mipmap. The resource search paths are
    ///                              used to locate the file.
    /// \return
    ///                              -  0: Success.
    ///                              - -2: Failure to resolve the given filename, e.g., the file
    ///                                    does not exist.
    ///                              - -3: Failure to open the file.
    ///                              - -4: No image plugin found to handle the file.
    ///                              - -5: The image plugin failed to import the file.
    Sint32 reset_file( const std::string& original_filename);

    /// Imports a mipmap from a reader.
    ///
    /// \param reader                The reader for the mipmap.
    /// \param image_format          The image format.
    /// \return
    ///                              -  0: Success.
    ///                              - -3: Invalid reader, or the reader does not support absolute
    ///                                    access.
    ///                              - -4: No image plugin found to handle the data.
    ///                              - -5: The image plugin failed to import the data.
    Sint32 reset_reader( mi::neuraylib::IReader* reader, const char* image_format);

    /// Imports mipmaps according to an image description. 
    /// The image description can either describe  a single image file name and reader 
    /// or a set of texture-atlas image names, associated uvs and readers
    ///
    /// \param image_set_dec         The image description to use. 
    ///                              filename resolution rules.
    /// \return
    ///                              -  0: Success.
    ///                              - -1: The image set is NULL or empty
    ///                              - -2: The image plugin failed to import the data.
    Sint32 reset(
        const Image_set* image_set);

    /// Sets a memory-based mipmap.
    ///
    /// Actually, the mipmap might not be memory-based, but it will be treated as if it was a
    /// memory-based mipmap in particular, for serialization purposes).
    ///
    /// A \c NULL pointer can be passed to restore the state after default construction (a dummy
    /// mipmap with a 1x1 canvas with a pink pixel).
    void set_mipmap( IMAGE::IMipmap* mipmap);

    /// Returns the mipmap referenced by this DB element (mutable).
    ///
    /// Never returns \c NULL.
    ///
    /// \note The copy constructor actually does not copy the image, but creates a reference to a
    ///       dummy mipmap with a 1x1 canvas with a pink pixel (same as after default construction).
    ///       See #mi::neuraylib::IImage for the rationale.
    IMAGE::IMipmap* get_mipmap();

    /// Returns the mipmap referenced by the uv-tile at the given index of the DB element (const).
    ///
    /// \param uvtile_id   The uv-tile id of the canvas.
    /// Returns \c NULL if a tile for the provided uvtile_id does not exist.
    const IMAGE::IMipmap* get_mipmap( mi::Uint32 uvtile_id = 0) const;

    /// Returns the resolved filename of the uv-tile at the given index of the referenced image.
    ///
    /// \param uvtile_id   The uv-tile id of the canvas.
    /// \return   The resolved filename of the referenced uv-tile, or the empty string if the 
    ///           uv-tile is not file-based (including failure to resolve the filename).
    const std::string& get_filename( mi::Uint32 uvtile_id = 0) const;

    /// Returns the original filename of the referenced mipmap.
    ///
    /// \return   The original filename as passed to set_mipmap(const std::string&), or the empty
    ///           string if the mipmap is not file-based.
    const std::string& get_original_filename() const;

    /// Returns the absolute MDL file path of the referenced mipmap.
    ///
    /// \return   The absolute MDL file path, or the empty string if not available.
    const std::string& get_mdl_file_path( ) const;

    /// Indicates whether the referenced mipmap represents a cubemap.
    bool get_is_cubemap() const;

    /// Indicates whether this image references a valid mipmap.
    ///
    /// After default construction and after set_mipmap() is called with a \c NULL pointer or an
    /// invalid filename this image references a dummy mipmap with a 1x1 canvas with a pink pixel.
    bool is_valid() const;

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

    // internal methods

    /// Indicates whether this mipmap is file-based.
    bool is_file_based() const { return !m_uvtiles[0].m_resolved_filename.empty(); }

    /// Indicates whether this mipmap is archive-based.
    bool is_archive_based() const { return !m_resolved_archive_filename.empty(); }

    /// Indicates whether this mipmap is memory-based.
    bool is_memory_based() const { return !is_file_based() && !is_archive_based(); }

    /// Returns the archive file name for archive-based mipmaps, and \c NULL otherwise.
    const std::string& get_archive_filename() const { return m_resolved_archive_filename; }

    /// Returns the archive member name for archive-based mipmaps, and \c NULL otherwise.
    const std::string& get_archive_membername( mi::Uint32 uvtile_id = 0) const ;

    /// Returns the number of uvtiles of this image
    mi::Size get_uvtile_length() const;

    /// Returns the u and v tile indices of the uv-tile at the given index.
    ///
    /// \param uvtile_id   The uv-tile id of the canvas.
    /// \param u           The u-component of the uv-tile
    /// \param v           The v-component of the uv-tile
    /// \return            0 on success, -1 if uvtile_id is out of range.
    mi::Sint32 get_uvtile_uv( Uint32 uvtile_id, Sint32& u, Sint32& v) const;

    // Returns the uvtile-id corresponding to the tile at u,v.
    ///
    /// \param u           The u-component of the uv-tile
    /// \param v           The v-component of the uv-tile
    /// \return The uvtile-id or -1 of there is no tile with the given coordinates.
    mi::Uint32 get_uvtile_id( Sint32 u, Sint32 v) const;

    /// Returns true if the image set represents a texture atlas mapping
    bool is_uvtile() const;

    /// Access to low level tile mapping, used by material converter
    const unsigned int *get_tile_mapping(
        mi::Uint32 &num_u,
        mi::Uint32 &num_v,
        mi::Sint32 &offset_u, mi::Sint32 &offset_v) const
    {
        if (m_uv_to_index.m_uv.empty()) {
            num_u = num_v = offset_u = offset_v = 0;
            return NULL;
        }
        num_u = m_uv_to_index.m_nu;
        num_v = m_uv_to_index.m_nv;
        offset_u = m_uv_to_index.m_offset_u;
        offset_v = m_uv_to_index.m_offset_v;
        return &m_uv_to_index.m_uv[0];
    }
        
private:
    struct Uvtile
    {
        /// The file that contains the data of this uv tile
        ///
        /// Non-empty exactly for file-based images.
        ///
        /// This is the filename as it has been resolved in set_mipmap() or deserialize().
        std::string m_resolved_filename;

        /// The absolute MDL file path of a specific tile
        std::string m_mdl_file_path;

        /// The archive member that contains the data of this DB element.
        ///
        /// Non-empty exactly for archive-based images.
        std::string m_resolved_archive_membername;

        /// The mipmap referenced by this tile
        mi::base::Handle<IMAGE::IMipmap> m_mipmap;

        mi::Sint32 m_u;
        mi::Sint32 m_v;

        Uvtile();
    };

    /// UV to index lookup table
    struct Uv_to_index
    {
        /// UV index table
        std::vector<mi::Uint32> m_uv;

        /// table dimension in u
        mi::Uint32 m_nu;

        /// table dimension in v
        mi::Uint32 m_nv;

        /// offsets for negative indices
        mi::Sint32 m_offset_u, m_offset_v;

        Uv_to_index() : m_nu(0), m_nv(0), m_offset_u(0), m_offset_v(0)
        {}

        /// Resets the table to one entry with index 1
        void reset() 
        {
            m_nu = 1;
            m_nv = 1;
            m_offset_u = 0;
            m_offset_v = 0;
            m_uv.resize(1, 1);
        }

        /// Resets table to u_max-u_min+1 x v_max-v_min+1 entries with index -1 and
        /// calculates the corresponding offsets
        void reset(mi::Sint32 u_min, mi::Sint32 u_max, mi::Sint32 v_min, mi::Sint32 v_max);

        /// Gets the index at u,v
        /// 
        /// \param u        tile position in u
        /// \param v        tile position in v
        /// 
        /// \return         tile index
        mi::Uint32 get(mi::Sint32 u, mi::Sint32 v) const;

        /// Sets the index at u,v
        /// 
        /// \param u        tile position in u
        /// \param v        tile position in v
        /// \param index    tile index
        /// \return         true if the index at position (u,v) has not already been set
        ///                 false otherwise
        bool set(mi::Sint32 u, mi::Sint32 v, mi::Uint32 index);
    };

    /// Comments on DB::Element_base and DB::Element say that the copy constructor is needed.
    /// But the assignment operator is not implemented, although usually, they are implemented both
    /// or none. Let's make the assignment operator private for now.
    Image& operator=( const Image&);

    /// Sets a dummy mipmap with a 1x1 canvas with a pink pixel.
    ///
    /// Does not affect the stored filenames.
    void set_default_pink_dummy_mipmap();

    /// Searches for files matching the given path or udim/uv-tile pattern 
    /// 
    /// \param path path to resolve
    /// \return Image_set containing the resolved filenames for or NULL in case of error
    Image_set* resolve_filename(const std::string& path) const;


    /// Parses the uv indizes from the given uvtile/udim string
    /// 
    ///\pram mode       uvtile/udim mode
    ///\param str       string containing the indices, e.g. 1001 in udim mode
    ///\param u         resulting u index
    ///\param v         resulting v index
    static void parse_u_v(
        const Uvtile_mode mode,
        const char *str,
        mi::Sint32& u,
        mi::Sint32& v);

    /// The uvtile array referenced by this DB element.
    std::vector<Uvtile> m_uvtiles;

    /// UV to uv-tile index mapping
    Uv_to_index m_uv_to_index;

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

    /// The archive that contains the data of this DB element
    ///
    /// Non-empty exactly for archive-based light images.
    std::string m_resolved_archive_filename;

    bool m_is_uvtile;
};

} // namespace DBIMAGE

} // namespace MI

#endif // IO_SCENE_DBIMAGE_I_DBIMAGE_H
