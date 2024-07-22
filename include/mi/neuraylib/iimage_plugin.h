/***************************************************************************************************
 * Copyright (c) 2008-2024, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Image plugin API

#ifndef MI_NEURAYLIB_IIMAGE_PLUGIN_H
#define MI_NEURAYLIB_IIMAGE_PLUGIN_H

#include <mi/base/interface_declare.h>
#include <mi/base/plugin.h>
#include <mi/base/types.h>
#include <mi/neuraylib/iimpexp_base.h>

namespace mi {

class IMap;

namespace neuraylib {

class IImage_file;
class IPlugin_api;
class IReader;
class ITile;
class IWriter;

/**
    \if MDL_SDK_API
    \defgroup mi_neuray_plugins Extensions and Plugins
    \ingroup mi_neuray

    Various ways to extend the \NeurayApiName, for example, image plugins.
    \endif
*/

/** \addtogroup mi_neuray_plugins
@{
*/

/// Type of image plugins
#define MI_NEURAY_IMAGE_PLUGIN_TYPE "image v38"

/// Abstract interface for image plugins.
///
/// The image plugin API allows to add support for third-party image formats. Such an image format
/// will then be supported in import, export, and streaming operations. The image plugin API
/// comprises the interfaces #mi::neuraylib::IImage_plugin and #mi::neuraylib::IImage_file. It also
/// uses the interfaces #mi::neuraylib::IReader, #mi::neuraylib::IWriter, and #mi::neuraylib::ITile.
///
/// Image plugins need to return #MI_NEURAY_IMAGE_PLUGIN_TYPE in #mi::base::Plugin::get_type().
///
/// A plugin to support a certain image format is selected as follows. For import operations the
/// reader is presented to each plugin for testing. Each plugin indicates whether it recognizes the
/// header format and whether it can handle the image format. For export operations a matching
/// plugin is selected according to the file name extension (see
/// #mi::neuraylib::IImage_plugin::get_file_extension()).
class IImage_plugin : public base::Plugin
{
public:
    /// Returns the name of the plugin.
    ///
    /// For image plugins, typically the name of the image format is used, for example, \c "jpeg".
    ///
    /// \note This method from #mi::base::Plugin is repeated here only for documentation purposes.
    virtual const char* get_name() const = 0;

    /// Initializes the plugin.
    ///
    /// \param plugin_api   Provides access to API components available for plugins.
    /// \return             \c true in case of success, and \c false otherwise.
    virtual bool init( IPlugin_api* plugin_api) = 0;

    /// De-initializes the plugin.
    ///
    /// \param plugin_api   Provides access to API components available for plugins.
    /// \return             \c true in case of success, and \c false otherwise.
    virtual bool exit( IPlugin_api* plugin_api) = 0;

    /// Returns the \p index -th supported file extension.
    ///
    /// \param index   The index of the file extensions to be returned.
    /// \return        The \p index -th supported file extension, \c NULL if \p index is out of
    ///                bounds.
    virtual const char* get_file_extension( Uint32 index) const = 0;

    /// Returns the \p index -th supported pixel type for exporting.
    ///
    /// The pixel types should be ordered, from index 0 for the most common to the least useful
    /// type. See \ref mi_neuray_types for a list of supported pixel types.
    ///
    /// \param index   The index of the pixel type to be returned.
    /// \return        The \p index -th supported pixel type, \c NULL if \p index is out of
    ///                bounds.
    virtual const char* get_supported_type( Uint32 index) const = 0;

    /// Returns the priority of the image plugin.
    ///
    /// The priority expresses the confidence of the plugin that its #test() method can identify the
    /// image format and that the image format is fully supported.
    virtual Impexp_priority get_priority() const = 0;

    /// Indicates whether the image plugin implements selector support.
    ///
    /// \return         \c true if #open_for_reading() handles the \c selector parameter,
    ///                 \c false otherwise. In the latter case a generic, possibly less efficient,
    ///                 support for RGBA channel selectors only is applied to the tiles returned by
    ///                 #mi::neuraylib::IImage_file::read() by the caller.
    virtual bool supports_selectors() const = 0;

    /// Indicates whether the image plugin can handle a given image.
    ///
    /// \param reader   A reader representing the stream to read from.
    /// \return         \c true if the plugin can handle the image, \c false otherwise.
    virtual bool test( IReader* reader) const = 0;

    /// Creates an object that reads an image from a stream.
    ///
    /// This method is called to start an image import operation.
    ///
    /// \param reader         A reader representing the stream to read from.
    /// \param selector       The selector (or \c NULL).
    /// \return               The object that reads the image from the reader.
    virtual IImage_file* open_for_reading( IReader* reader, const char* selector) const = 0;

    /// Creates an object that writes an image to a stream.
    ///
    /// This method is called to start an image export operation.
    ///
    /// \param writer             A writer representing the stream to write to.
    /// \param pixel_type         The pixel type of the image tiles. This is one of the pixel types
    ///                           returned by #get_supported_type().
    /// \param resolution_x       The resolution of the image in x direction.
    /// \param resolution_y       The resolution of the image in y direction.
    /// \param nr_of_layers       The number of layers in the image.
    /// \param miplevels          The number of mipmap levels in the image.
    /// \param is_cubemap         \c true if the image is supposed to be cubemap, \c false
    ///                           otherwise.
    /// \param gamma              The gamma value of the image.
    /// \param export_options     See \ref mi_image_export_options for options that should be
    ///                           supported by an exporter for the corresponding format. The option
    ///                           \c "force_default_gamma" is handled by \neurayProductName and
    ///                           can be ignored in the plugin.
    /// \return                   The object that writes the image to a stream.
    virtual IImage_file* open_for_writing(
        IWriter* writer,
        const char* pixel_type,
        Uint32 resolution_x,
        Uint32 resolution_y,
        Uint32 nr_of_layers,
        Uint32 miplevels,
        bool is_cubemap,
        Float32 gamma,
        const IMap* export_options) const = 0;
};

/// Abstract interface for image files.
///
/// Instance of this interface are created by #mi::neuraylib::IImage_plugin::open_for_writing() or
/// #mi::neuraylib::IImage_plugin::open_for_reading().
class IImage_file
  : public base::Interface_declare<0x26db4186,0xace2,0x42e8,0xa0,0x3d,0xe0,0xfa,0xfc,0xed,0x05,0xf3>
{
public:
    /// Returns the pixel type of the image (or the requested channel).
    ///
    /// See \ref mi_neuray_types for a list of supported pixel types.
    virtual const char* get_type() const = 0;

    /// Returns the resolution of the image in x direction.
    ///
    /// \param level   The mipmap level (always 0 if the image is not a mipmap).
    /// \return        The resolution of the image in x direction.
    virtual Uint32 get_resolution_x( Uint32 level) const = 0;

    /// Returns the resolution of the image in y direction.
    ///
    /// \param level   The mipmap level (always 0 if the image is not a mipmap).
    /// \return        The resolution of the image in y direction.
    virtual Uint32 get_resolution_y( Uint32 level) const = 0;

    /// Returns the number of layers of the image.
    ///
    /// \param level   The mipmap level (always 0 if the image is not a mipmap).
    /// \return        The number of layers of the image.
    virtual Uint32 get_layers_size( Uint32 level) const = 0;

    /// Returns number of miplevels.
    virtual Uint32 get_miplevels() const = 0;

    /// Indicates whether the image represents a cubemap.
    ///
    /// \return \c if the image represents a cubemap, \c false otherwise.
    virtual bool get_is_cubemap() const = 0;

    /// Returns the gamma value of the image.
    virtual Float32 get_gamma() const = 0;

    /// Read pixels from the image file into a tile.
    ///
    /// This method will never be called if this instance was obtained from
    /// #mi::neuraylib::IImage_plugin::open_for_writing().
    ///
    /// \param z     The z layer (for 3d textures or cubemaps).
    /// \param level The mipmap level (always 0 if the image is not a mipmap).
    /// \return      The tile with the read data, or \c NULL in case of failures.
    virtual ITile* read( Uint32 z, Uint32 level) const = 0;

    /// Write pixels from a tile into the image file.
    ///
    /// This method will never be called if this instance was obtained from
    /// #mi::neuraylib::IImage_plugin::open_for_reading().
    ///
    /// \param tile  The tile to read the data from.
    /// \param z     The z layer (for 3d textures or cubemaps).
    /// \param level The mipmap level (always 0 if the image is not a mipmap).
    /// \return      \c true if the tile was successfully written, \c false otherwise.
    virtual bool write( const ITile* tile, Uint32 z, Uint32 level) = 0;
};

/**@}*/ // end group mi_neuray_plugins

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IIMAGE_PLUGIN_H
