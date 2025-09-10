/***************************************************************************************************
 * Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
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

#ifndef SHADERS_PLUGIN_OPENIMAGEIO_OPENIMAGEIO_UTILITIES_H
#define SHADERS_PLUGIN_OPENIMAGEIO_OPENIMAGEIO_UTILITIES_H

#include <mi/base.h>

#include <OpenImageIO/imageio.h>

#include <io/image/image/i_image_utilities.h>

namespace mi {
namespace neuraylib {
class IDebug_configuration;
class IImage_api;
class IReader;
class ITile;
class IWriter;
} // namespace neuraylib
} // namespace mi

namespace MI {

namespace MI_OIIO {

/// Value of the debug option \c ignore_gamma_metadata_on_import.
///
/// \c true:  metadata about the gamma value is ignored
/// \c false: metadata about the gamma value is taked into account
extern bool g_ignore_gamma_metadata_on_import;

/// Value of the debug option \c linear_gamma_for_single_uint16_channel.
///
/// \c true:  one channel of OIIO::TypeUInt16 has gamma 1.0
/// \c false: one channel of OIIO::TypeUInt16 has gamma 2.2
///
/// Note that gamma metadata has higher priority.
extern bool g_linear_gamma_for_single_uint16_channel;

/// The logger used by #log() below. Do not use it directly, because it might be invalid if the
/// plugin API is not available. Use #log() instead.
extern mi::base::Handle<mi::base::ILogger> g_logger;

/// Logs a message.
///
/// The message is discarded if the plugin API and the logger is not available.
void log( mi::base::Message_severity severity, const char* message);

/// Returns the OIIO base type for a \em single channel of our pixel type.
OIIO::TypeDesc::BASETYPE get_base_type( IMAGE::Pixel_type pixel_type);

/// Returns the OIIO image spec for our pixel type and resolution.
OIIO::ImageSpec get_image_spec(
    IMAGE::Pixel_type,
    mi::Uint32 resolution_x,
    mi::Uint32 resolution_y,
    mi::Uint32 resolution_z);

/// Returns our pixel type and the gamma value that should be used for the give image spec.
///
/// The result is solely based on the channel characteristics and ignores metadata about color
/// spaces.
std::pair<IMAGE::Pixel_type, mi::Float32> get_pixel_type_and_gamma_internal(
    const OIIO::ImageSpec& spec);

/// Returns our pixel type and the gamma value that should be used for the give image spec.
std::pair<IMAGE::Pixel_type, mi::Float32> get_pixel_type_and_gamma( const OIIO::ImageSpec& spec);

/// Returns the gamma value derived from the oiio::Colorspace metadata.
///
/// Supports "linear", "sRGB", "rec709", and "GammaX.Y" (case does not matter). Returns 0.0 in case
/// of errors.
mi::Float32 get_gamma_from_metadata( const OIIO::ImageSpec& spec);

/// Premultiplies color channels by alpha channel (as expected by OIIO for export by default).
const mi::neuraylib::ITile* associate_alpha(
    mi::neuraylib::IImage_api* image_api, const mi::neuraylib::ITile* tile, mi::Float32 gamma);

/// Un-premultiplies color channels by alpha channel (as delivered by OIIO upon import by default).
mi::neuraylib::ITile* unassociate_alpha(
    mi::neuraylib::IImage_api* image_api, mi::neuraylib::ITile* tile, mi::Float32 gamma);

/// Wraps a reader into an input proxy.
OIIO::Filesystem::IOProxy* create_input_proxy(
    mi::neuraylib::IReader* reader, bool use_buffer, std::vector<char>* buffer);

/// Wraps a writer into an output proxy. Unused (slower than local buffer).
OIIO::Filesystem::IOProxy* create_output_proxy( mi::neuraylib::IWriter* writer);

/// Expands gray-alpha (YA) data row by row into RGBA data.
///
/// All parameters below can be computed from the corresponding ITile interface.
///
/// \param bpc            Bytes per channel. Implementation supports only bpc == 1, 2, or 4.
/// \param resolution_x   Resolution in x-direction.
/// \param resolution_y   Resolution in y-direction.
/// \param data           Data buffer.
void expand_ya_to_rgba(
    int bpc, mi::Uint32 resolution_x, mi::Uint32 resolution_y, mi::Uint8* data);

/// Computes properties that depend on the selector (and the image spec).
///
/// \note The following limitations apply for layer:
///       - Channels of a layer need to be contiguous.
///       - Channels of a layer need be sorted in RGB(A) order, followed by other channels.
///       The limitations should be met for OpenEXR images because:
///       - OpenEXR seems to sort channels by layers (no interleaving).
///       - OpenImageIO sorts channels within a layer (RGB(A), followed by other channels).
///
/// \note The selector needs to match a channel name, or the name of a layer (selector plus dot as
///       a prefix) without further nested layers (no more dots). For multipart images, the selector
///       contains additionally the part name plus a dot as prefix. The restriction of no further
///       nested layers could be lifted if there is only a \em single nested layer (the selector
///       would still be unambiguous then) by checking that the longest common prefix does not
///       contain a dot (and adapting the YA expansion check).
///
/// \param input                The image input. The method seeks to the returned subimage.
/// \param selector             The selector (can be \c NULL).
/// \param[out] subimage        The index of the subimage.
/// \param[out] resolution_x    The resolution of the subimage in x-direction.
/// \param[out] resolution_y    The resolution of the subimage in y-direction.
/// \param[out] resolution_z    The resolution of the subimage in z-direction.
/// \param[out] pixel_type      The pixel type of the subimage.
/// \param[out] gamma           The gamma value of the subimage.
/// \param[out] channel_names   The channel names (for layers: after stripping the selector prefix).
/// \param[out] channel_start   The first channel index to import.
/// \param[out] channel_end     The last channel index+1 to import.
/// \return                     Success or failure.
bool compute_properties(
    OIIO::ImageInput* input,
    const char* selector,
    mi::Uint32& subimage,
    mi::Uint32& resolution_x,
    mi::Uint32& resolution_y,
    mi::Uint32& resolution_z,
    IMAGE::Pixel_type& pixel_type,
    mi::Float32& gamma,
    std::vector<std::string>& channel_names,
    mi::Sint32& channel_start,
    mi::Sint32& channel_end);

/// Returns an OIIO colorspace indentification string for the given gamma value.
///
/// Returns "linear" for 1.0, "sRGB" for 2.2, "GammaX.Y" for other positive values X.Y, and the
/// empty string on errors (zero or negative values).
std::string get_oiio_colorspace( float gamma);

/// Converts the value of the debug option \p key into a boolean value.
///
/// Does not change the value of \p value if there is no option for \p key set, or if the conversion
/// via operator>> fails.
///
/// Returns \c true in case of success (including \p key not existing), \c false otherwise.
bool option_to_flag(
    mi::neuraylib::IDebug_configuration* debug_configuration, const char* key, bool& value);

} // namespace MI_OIIO

} // namespace MI

#endif // SHADERS_PLUGIN_OPENIMAGEIO_OPENIMAGEIO_UTILITIES_H
