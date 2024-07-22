/***************************************************************************************************
 * Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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

#include "pch.h"

#include "openimageio_utilities.h"

#include <cassert>

#include <OpenImageIO/filesystem.h>

#include <mi/math/function.h>
#include <mi/neuraylib/iimage_api.h>
#include <mi/neuraylib/ireader.h>
#include <mi/neuraylib/itile.h>
#include <mi/neuraylib/iwriter.h>

namespace MI {

namespace MI_OIIO {

mi::base::Handle<mi::base::ILogger> g_logger;

void log( mi::base::Message_severity severity, const char* message)
{
    if( !g_logger) {
        // Missing the info message in init() is not problematic, but flag warnings, errors, and
        // fatals if there is no logger present (should happen only in unit tests).
        assert(    severity != mi::base::MESSAGE_SEVERITY_FATAL
                && severity != mi::base::MESSAGE_SEVERITY_ERROR
                && severity != mi::base::MESSAGE_SEVERITY_WARNING);
        return;
    }

    g_logger->message( severity, "OIIO:IMAGE", message);
}

namespace {

bool is_rgb( const OIIO::ImageSpec& spec)
{
    assert( spec.nchannels >= 3);
    assert( spec.channelnames.size() >= 3);
    return spec.channelnames[0] == "R"
        && spec.channelnames[1] == "G"
        && spec.channelnames[2] == "B";
}

bool is_rgba( const OIIO::ImageSpec& spec)
{
    if( spec.nchannels < 4)
        return false;

    assert( spec.channelnames.size() >= 4);
    // Do not check whether spec.channelnames[3] == "A" or "Alpha". There are files where the alpha
    // channel has a generic name, e.g., "channel3".
    return spec.channelnames[0] == "R"
        && spec.channelnames[1] == "G"
        && spec.channelnames[2] == "B";
}

bool has_alpha( const OIIO::ImageSpec& spec)
{
    return spec.alpha_channel != -1;
}

} // namespace

IMAGE::Pixel_type get_pixel_type( const OIIO::ImageSpec& spec)
{
    if( spec.deep)
        return IMAGE::PT_UNDEF;
    if( !spec.channelformats.empty())
        return IMAGE::PT_UNDEF;

    if( spec.nchannels == 1) {

        // ignore channel name
        if( spec.format == OIIO::TypeUInt8)
            return IMAGE::PT_SINT8;
        else if( spec.format == OIIO::TypeUInt16) // Or PT_SINT32, but PT_FLOAT32 is closer to
            return IMAGE::PT_FLOAT32;             // what FreeImage was doing.
        else if( spec.format == OIIO::TypeInt32)
            return IMAGE::PT_SINT32;
        else if( spec.format == OIIO::TypeHalf)
            return IMAGE::PT_FLOAT32;
        else if( spec.format == OIIO::TypeFloat)
            return IMAGE::PT_FLOAT32;
        else
            return IMAGE::PT_UNDEF;

    } else if( (spec.nchannels == 2) && !has_alpha( spec)) {

        if( spec.format == OIIO::TypeHalf)
            return IMAGE::PT_FLOAT32_2;
        if( spec.format == OIIO::TypeFloat)
            return IMAGE::PT_FLOAT32_2;
        else
            return IMAGE::PT_UNDEF;

    } else if(     (spec.nchannels == 3)
               || ((spec.nchannels >= 5) && !has_alpha( spec))) {

        if( !is_rgb( spec) && !has_alpha( spec))
            return IMAGE::PT_FLOAT32_3;

        if( spec.format == OIIO::TypeUInt8)
            return IMAGE::PT_RGB;
        else if( spec.format == OIIO::TypeUInt16)
            return IMAGE::PT_RGB_16;
        else if( spec.format == OIIO::TypeInt16)
            return IMAGE::PT_RGB_16;
        else if( spec.format == OIIO::TypeHalf)
            return IMAGE::PT_RGB_FP;
        else if( spec.format == OIIO::TypeFloat)
            return IMAGE::PT_RGB_FP;
        else
            return IMAGE::PT_UNDEF;

    } else if(    ((spec.nchannels == 2) && has_alpha( spec))
               ||  (spec.nchannels == 4)
               || ((spec.nchannels >= 5) && has_alpha( spec))) {

        if( !is_rgba( spec) && !has_alpha( spec))
            return IMAGE::PT_FLOAT32_4;

        if( spec.format == OIIO::TypeUInt8)
            return IMAGE::PT_RGBA;
        else if( spec.format == OIIO::TypeUInt16)
            return IMAGE::PT_RGBA_16;
        else if( spec.format == OIIO::TypeInt16)
            return IMAGE::PT_RGBA_16;
        else if( spec.format == OIIO::TypeHalf)
            return IMAGE::PT_COLOR;
        else if( spec.format == OIIO::TypeFloat)
            return IMAGE::PT_COLOR;
        else
            return IMAGE::PT_UNDEF;

    } else {

        return IMAGE::PT_UNDEF;
    }

}

OIIO::TypeDesc::BASETYPE get_base_type( IMAGE::Pixel_type pixel_type)
{
    switch( pixel_type) {
        case IMAGE::PT_UNDEF:     assert( false); return OIIO::TypeDesc::UNKNOWN;
        case IMAGE::PT_SINT8:     return OIIO::TypeDesc::UINT8;
        case IMAGE::PT_SINT32:    return OIIO::TypeDesc::INT32;
        case IMAGE::PT_FLOAT32:   return OIIO::TypeDesc::FLOAT;
        case IMAGE::PT_FLOAT32_2: return OIIO::TypeDesc::FLOAT;
        case IMAGE::PT_FLOAT32_3: return OIIO::TypeDesc::FLOAT;
        case IMAGE::PT_FLOAT32_4: return OIIO::TypeDesc::FLOAT;
        case IMAGE::PT_RGB:       return OIIO::TypeDesc::UINT8;
        case IMAGE::PT_RGBA:      return OIIO::TypeDesc::UINT8;
        case IMAGE::PT_RGBE:      return OIIO::TypeDesc::FLOAT;
        case IMAGE::PT_RGBEA:     return OIIO::TypeDesc::FLOAT;
        case IMAGE::PT_RGB_16:    return OIIO::TypeDesc::UINT16;
        case IMAGE::PT_RGBA_16:   return OIIO::TypeDesc::UINT16;
        case IMAGE::PT_RGB_FP:    return OIIO::TypeDesc::FLOAT;
        case IMAGE::PT_COLOR:     return OIIO::TypeDesc::FLOAT;
    }

    assert( false);
    return OIIO::TypeDesc::UNKNOWN;
}

OIIO::ImageSpec get_image_spec(
    IMAGE::Pixel_type pixel_type,
    mi::Uint32 resolution_x,
    mi::Uint32 resolution_y,
    mi::Uint32 resolution_z)
{
    int cpp = IMAGE::get_components_per_pixel( pixel_type);
    OIIO::ROI roi( 0, resolution_x, 0, resolution_y, 0, resolution_z, 0, cpp);
    OIIO::TypeDesc type_desc( get_base_type( pixel_type));
    return OIIO::ImageSpec( roi, type_desc);
}

namespace {

constexpr double get_bound( mi::Uint8 dummy)   { return   255.0; }
constexpr double get_bound( mi::Uint16 dummy)  { return 65535.0; }
constexpr double get_bound( mi::Float32 dummy) { return     1.0; }

// Convert to associated alpha, arbitrary gamma.
template <class T>
void associate_alpha( mi::neuraylib::ITile* tile, mi::Float32 gamma)
{
    const mi::Uint32 tile_width   = tile->get_resolution_x();
    const mi::Uint32 tile_height  = tile->get_resolution_y();
    const mi::Size   nr_of_pixels = tile_width * static_cast<mi::Size>( tile_height);

    constexpr auto inv_bound = static_cast<float>( (1.0 / get_bound( T())));

    T* const __restrict data = static_cast<T*>( tile->get_data());
    for( mi::Size i = 0; i < 4*nr_of_pixels; i += 4) {
        float a = mi::math::fast_pow( data[i+3] * inv_bound, gamma);
        data[i+0] = static_cast<T>( data[i+0] * a);
        data[i+1] = static_cast<T>( data[i+1] * a);
        data[i+2] = static_cast<T>( data[i+2] * a);
    }
}

// Convert to associated alpha, optimized for gamma = 1.0.
template <class T>
void associate_alpha( mi::neuraylib::ITile* tile)
{
    const mi::Uint32 tile_width   = tile->get_resolution_x();
    const mi::Uint32 tile_height  = tile->get_resolution_y();
    const mi::Size   nr_of_pixels = tile_width * static_cast<mi::Size>( tile_height);

    constexpr auto inv_bound = static_cast<float>( (1.0 / get_bound( T())));

    T* const __restrict data = static_cast<T*>( tile->get_data());
    for( mi::Size i = 0; i < 4*nr_of_pixels; i += 4) {
        float a = data[i+3] * inv_bound;
        data[i+0] = static_cast<T>( data[i+0] * a);
        data[i+1] = static_cast<T>( data[i+1] * a);
        data[i+2] = static_cast<T>( data[i+2] * a);
    }
}

// Convert to unassociated alpha, arbitrary gamma, mi::Uint8/16.
template <class T>
void unassociate_alpha( mi::neuraylib::ITile* tile, mi::Float32 gamma)
{
    const mi::Uint32 tile_width   = tile->get_resolution_x();
    const mi::Uint32 tile_height  = tile->get_resolution_y();
    const mi::Size   nr_of_pixels = tile_width * static_cast<mi::Size>( tile_height);

    constexpr auto bound = static_cast<float>( get_bound( T()));

    T* const __restrict data = static_cast<T*>( tile->get_data());
    for( mi::Size i = 0; i < 4*nr_of_pixels; i += 4) {
        float a = mi::math::fast_pow( bound / data[i+3], gamma);
        data[i+0] = static_cast<T>( std::min( bound, data[i+0] * a));
        data[i+1] = static_cast<T>( std::min( bound, data[i+1] * a));
        data[i+2] = static_cast<T>( std::min( bound, data[i+2] * a));
    }
}

// Convert to unassociated alpha, arbitrary gamma, mi::Float32.
template <>
void unassociate_alpha<mi::Float32>( mi::neuraylib::ITile* tile, mi::Float32 gamma)
{
    const mi::Uint32 tile_width   = tile->get_resolution_x();
    const mi::Uint32 tile_height  = tile->get_resolution_y();
    const mi::Size   nr_of_pixels = tile_width * static_cast<mi::Size>( tile_height);

    auto* const __restrict data = static_cast<float*>( tile->get_data());
    for( mi::Size i = 0; i < 4*nr_of_pixels; i += 4) {
        float a = mi::math::fast_pow( 1.0f / data[i+3], gamma);
        data[i+0] = static_cast<float>( data[i+0] * a);
        data[i+1] = static_cast<float>( data[i+1] * a);
        data[i+2] = static_cast<float>( data[i+2] * a);
    }
}

// Convert to unassociated alpha, optimized for gamma = 1.0, mi::Uint8/16.
template <class T>
void unassociate_alpha( mi::neuraylib::ITile* tile)
{
    const mi::Uint32 tile_width   = tile->get_resolution_x();
    const mi::Uint32 tile_height  = tile->get_resolution_y();
    const mi::Size   nr_of_pixels = tile_width * static_cast<mi::Size>( tile_height);

    constexpr auto bound = static_cast<float>( get_bound( T()));

    T* const __restrict data = static_cast<T*>( tile->get_data());
    for( mi::Size i = 0; i < 4*nr_of_pixels; i += 4) {
        float a = bound / data[i+3];
        data[i+0] = static_cast<T>( std::min( bound, data[i+0] * a));
        data[i+1] = static_cast<T>( std::min( bound, data[i+1] * a));
        data[i+2] = static_cast<T>( std::min( bound, data[i+2] * a));
    }
}

// Convert to unassociated alpha, optimized for gamma = 1.0, mi::Float32.
template <>
void unassociate_alpha<mi::Float32>( mi::neuraylib::ITile* tile)
{
    const mi::Uint32 tile_width   = tile->get_resolution_x();
    const mi::Uint32 tile_height  = tile->get_resolution_y();
    const mi::Size   nr_of_pixels = tile_width * static_cast<mi::Size>( tile_height);

    auto* const __restrict data = static_cast<float*>( tile->get_data());
    for( mi::Size i = 0; i < 4*nr_of_pixels; i += 4) {
        float a = 1.0f / data[i+3];
        data[i+0] = static_cast<float>( data[i+0] * a);
        data[i+1] = static_cast<float>( data[i+1] * a);
        data[i+2] = static_cast<float>( data[i+2] * a);
    }
}

} // namespace

const mi::neuraylib::ITile* associate_alpha(
    mi::neuraylib::IImage_api* image_api, const mi::neuraylib::ITile* tile, mi::Float32 gamma)
{
    const char* pixel_type = tile->get_type();
    IMAGE::Pixel_type pixel_type_enum = IMAGE::convert_pixel_type_string_to_enum( pixel_type);

    switch( pixel_type_enum) {
        case IMAGE::PT_UNDEF:     assert( false); tile->retain(); return tile;

        case IMAGE::PT_SINT8:
        case IMAGE::PT_SINT32:
        case IMAGE::PT_FLOAT32:
        case IMAGE::PT_FLOAT32_2:
        case IMAGE::PT_FLOAT32_3:
        case IMAGE::PT_FLOAT32_4:
        case IMAGE::PT_RGB:
        case IMAGE::PT_RGBE:
        case IMAGE::PT_RGB_16:
        case IMAGE::PT_RGB_FP:    tile->retain(); return tile;

        case IMAGE::PT_RGBA:
        case IMAGE::PT_RGBEA:
        case IMAGE::PT_RGBA_16:
        case IMAGE::PT_COLOR:     break;
    }

    // Convert pixel type "Rgbea" to "Color".
    mi::base::Handle<mi::neuraylib::ITile> tile2;
    if( pixel_type_enum == IMAGE::PT_RGBEA)
        tile2 = image_api->convert( tile, "Color");
    else
        tile2 = image_api->clone_tile( tile);
    tile = nullptr; // prevent accidental misuse

    // Associate alpha.
    if( pixel_type_enum == IMAGE::PT_RGBA)
        gamma == 1.0f
            ? associate_alpha<mi::Uint8>( tile2.get())
            : associate_alpha<mi::Uint8>( tile2.get(), gamma);
    else if( pixel_type_enum == IMAGE::PT_RGBA_16)
        gamma == 1.0f
            ? associate_alpha<mi::Uint16>( tile2.get())
            : associate_alpha<mi::Uint16>( tile2.get(), gamma);
    else if( pixel_type_enum == IMAGE::PT_COLOR || pixel_type_enum == IMAGE::PT_RGBEA)
        gamma == 1.0f
            ? associate_alpha<mi::Float32>( tile2.get())
            : associate_alpha<mi::Float32>( tile2.get(), gamma);
    else
        assert( false);

    // Convert back to pixel type "Rgbea" if necessary.
    if( pixel_type_enum == IMAGE::PT_RGBEA)
        tile2 = image_api->convert( tile, pixel_type);

    return tile2.extract();
}

mi::neuraylib::ITile* unassociate_alpha(
    mi::neuraylib::IImage_api* image_api, mi::neuraylib::ITile* tile, mi::Float32 gamma)
{
    const char* pixel_type = tile->get_type();
    IMAGE::Pixel_type pixel_type_enum = IMAGE::convert_pixel_type_string_to_enum( pixel_type);

    switch( pixel_type_enum) {
        case IMAGE::PT_UNDEF:     assert( false); tile->retain(); return tile;

        case IMAGE::PT_SINT8:
        case IMAGE::PT_SINT32:
        case IMAGE::PT_FLOAT32:
        case IMAGE::PT_FLOAT32_2:
        case IMAGE::PT_FLOAT32_3:
        case IMAGE::PT_FLOAT32_4:
        case IMAGE::PT_RGB:
        case IMAGE::PT_RGBE:
        case IMAGE::PT_RGB_16:
        case IMAGE::PT_RGB_FP:    tile->retain(); return tile;

        case IMAGE::PT_RGBA:
        case IMAGE::PT_RGBEA:
        case IMAGE::PT_RGBA_16:
        case IMAGE::PT_COLOR:     break;
     }

    // Convert pixel type "Rgbea" to "Color".
    mi::base::Handle<mi::neuraylib::ITile> tile2;
    if( pixel_type_enum == IMAGE::PT_RGBEA)
        tile2 = image_api->convert( tile, "Color");
    else
        tile2 = image_api->clone_tile( tile);
    tile = nullptr; // prevent accidental misuse

    // Associate alpha.
    if( pixel_type_enum == IMAGE::PT_RGBA)
        gamma == 1.0f
            ? unassociate_alpha<mi::Uint8>( tile2.get())
            : unassociate_alpha<mi::Uint8>( tile2.get(), gamma);
    else if( pixel_type_enum == IMAGE::PT_RGBA_16)
        gamma == 1.0f
            ? unassociate_alpha<mi::Uint16>( tile2.get())
            : unassociate_alpha<mi::Uint16>( tile2.get(), gamma);
    else if( pixel_type_enum == IMAGE::PT_COLOR || pixel_type_enum == IMAGE::PT_RGBEA)
        gamma == 1.0f
            ? unassociate_alpha<mi::Float32>( tile2.get())
            : unassociate_alpha<mi::Float32>( tile2.get(), gamma);
    else
        assert( false);

    // Convert back to pixel type "Rgbea" if necessary.
    if( pixel_type_enum == IMAGE::PT_RGBEA)
        tile2 = image_api->convert( tile, pixel_type);

    return tile2.extract();
}

// Wraps IReader as IOProxy.
class Input_proxy : public OIIO::Filesystem::IOProxy
{
public:
    Input_proxy( mi::neuraylib::IReader* reader);

    const char* proxytype () const override { return "nv_reader"; }
    void close() override { m_reader.reset(); }
    int64_t tell() override { return m_reader->tell_absolute(); }
    bool seek( int64_t offset) override { return m_reader->seek_absolute( offset); }
    size_t read( void* buf, size_t size) override;
    size_t write( const void* buf, size_t size) override;
    size_t pread( void* buf, size_t size, int64_t offset) override;
    size_t pwrite( const void* buf, size_t size, int64_t offset) override;
    size_t size() const override { return m_reader->get_file_size(); }
    void flush() const override { }

private:
    /// The wrapped reader.
    mi::base::Handle<mi::neuraylib::IReader> m_reader;
    /// Lock that protects parallel calls to pread().
    mi::base::Lock m_lock;
};

Input_proxy::Input_proxy( mi::neuraylib::IReader* reader)
  : OIIO::Filesystem::IOProxy( "", OIIO::Filesystem::IOProxy::Read),
    m_reader( reader, mi::base::DUP_INTERFACE)
{
    assert( m_reader);
    assert( m_reader->supports_absolute_access());
}

size_t Input_proxy::read( void* buf, size_t size)
{
    mi::Sint64 result = m_reader->read( static_cast<char*>( buf), size);
    return result == -1 ? 0 : result;
}

size_t Input_proxy::write( const void* buf, size_t size)
{
    assert( false);
    return 0;
}

size_t Input_proxy::pread( void* buf, size_t size, int64_t offset)
{
    mi::base::Lock::Block block( &m_lock);
    mi::Sint64 pos = m_reader->tell_absolute();
    bool success = pos != offset ? m_reader->seek_absolute( offset) : true;
    assert( success);
    (void) success;
    size_t result = read( buf, size);
    m_reader->seek_absolute( pos);
    return result;
}

size_t Input_proxy::pwrite( const void* buf, size_t size, int64_t offset)
{
    assert( false);
    return 0;
}

OIIO::Filesystem::IOProxy* create_input_proxy(
    mi::neuraylib::IReader* reader, bool use_buffer, std::vector<char>* buffer)
{
    if( use_buffer) {
        assert( buffer);
        constexpr size_t chunk = 65536;
        size_t size = 0;
        do {
            buffer->resize( size + chunk);
            size_t result = reader->read( &(*buffer)[size], chunk);
            size += result;
        } while( !reader->eof());
        buffer->resize( size);
        return new OIIO::Filesystem::IOMemReader( &(*buffer)[0], buffer->size());
    }

    return new Input_proxy( reader);
}

// Wraps IWriter as IOProxy. Unused (slower than local buffer).
class Output_proxy : public OIIO::Filesystem::IOProxy
{
public:
    Output_proxy( mi::neuraylib::IWriter* writer);

    const char* proxytype () const override { return "nv_writer"; }
    void close() override { m_writer.reset(); }
    int64_t tell() override { return m_writer->tell_absolute(); }
    bool seek( int64_t offset) override { return m_writer->seek_absolute( offset); }
    size_t read( void* buf, size_t size) override;
    size_t write( const void* buf, size_t size) override;
    size_t pread( void* buf, size_t size, int64_t offset) override;
    size_t pwrite( const void* buf, size_t size, int64_t offset) override;
    size_t size() const override { return m_writer->get_file_size(); }
    void flush() const override { m_writer->flush(); }

private:
    /// The wrapped writer.
    mi::base::Handle<mi::neuraylib::IWriter> m_writer;
    /// Lock that protects parallel calls to pwrite().
    mi::base::Lock m_lock;
};

Output_proxy::Output_proxy( mi::neuraylib::IWriter* writer)
  : OIIO::Filesystem::IOProxy( "", OIIO::Filesystem::IOProxy::Write),
    m_writer( writer, mi::base::DUP_INTERFACE)
{
    assert( m_writer);
    assert( m_writer->supports_absolute_access());
}

size_t Output_proxy::read( void* buf, size_t size)
{
    assert( false);
    return 0;
}

size_t Output_proxy::write( const void* buf, size_t size)
{
    mi::Sint64 result = m_writer->write( static_cast<const char*>( buf), size);
    return result == -1 ? 0 : result;
}

size_t Output_proxy::pread( void* buf, size_t size, int64_t offset)
{
    assert( false);
    return 0;
}

size_t Output_proxy::pwrite( const void* buf, size_t size, int64_t offset)
{
    mi::base::Lock::Block block( &m_lock);
    mi::Sint64 pos = m_writer->tell_absolute();
    bool success = pos != offset ? m_writer->seek_absolute( offset) : true;
    assert( success);
    (void) success;
    size_t result = write( buf, size);
    m_writer->seek_absolute( pos);
    return result;
}

OIIO::Filesystem::IOProxy* create_output_proxy( mi::neuraylib::IWriter* writer)
{
    return new Output_proxy( writer);
}

namespace {

template<class T>
void expand_ya_to_rgba(
    mi::Uint32 resolution_x, mi::Uint32 resolution_y, mi::Uint8* data)
{
    const T* src = reinterpret_cast<T*>( data) + 4 * resolution_x * resolution_y;
    T* dst       = reinterpret_cast<T*>( data) + 4 * resolution_x * resolution_y;

    for( mi::Uint32 y = resolution_y; y > 0; --y) { // offset by 1

        src -= 2 * resolution_x;

        for( mi::Uint32 x = resolution_x; x > 0; --x) { // offset by 1
            --src;
            --dst;
            *dst = *src;  // A to A
            --src;
            --dst;
            *dst = *src;  // Y to B
            --dst;
            *dst = *src;  // Y to G
            --dst;
            *dst = *src;  // Y to R
        }
    }
}

} // namespace

void expand_ya_to_rgba(
    int bpc, mi::Uint32 resolution_x, mi::Uint32 resolution_y, mi::Uint8* data)
{
    switch( bpc) {
        case 1: expand_ya_to_rgba<mi::Uint8 >( resolution_x, resolution_y, data); return;
        case 2: expand_ya_to_rgba<mi::Uint16>( resolution_x, resolution_y, data); return;
        case 4: expand_ya_to_rgba<mi::Uint32>( resolution_x, resolution_y, data); return;
    }

    assert( false);
}

namespace {

/// Returns the channel index for which the concatention of \p part_prefix and the channel name
/// equals \p selector.
int get_channel_index(
    const OIIO::ImageSpec& spec, const std::string& part_prefix, const char* selector)
{
    for( int i = 0; i < spec.nchannels;  ++i)
        if( part_prefix + spec.channelnames[i] == selector)
            return i;
    return -1;
}

/// Returns the channel index for which the concatention of \p part_prefix and the channel name
/// equals \p selector1 or \p selector2.
///
/// Search in parallel instead of one after the other mimics what OpenImageIO itself is doing for
/// OpenEXR images.
int get_channel_index(
    const OIIO::ImageSpec& spec,
    const std::string& part_prefix,
    const char* selector1,
    const char* selector2)
{
    for( int i = 0; i < spec.nchannels;  ++i) {
        std::string s = part_prefix + spec.channelnames[i];
        if( s == selector1 || s == selector2)
            return i;
        }
    return -1;
}

/// Returns the range of channel indices for which the concatention of \p part_prefix and the
/// channel name is a prefix of selector_prefix, or (-1,-1) in case of failure.
std::pair<int,int> get_selector_channel_range(
    const OIIO::ImageSpec& spec, const std::string& part_prefix, const std::string& selector_prefix)
{
    int begin = -1;
    int end   = -1;

    size_t n = selector_prefix.size();

    for( int i = 0; i < spec.nchannels;  ++i) {
        if( (part_prefix + spec.channelnames[i]).substr( 0, n) == selector_prefix) {
            // Reject layers with non-contiguous channels.
            if( end != -1)
                return std::pair<int,int>( -1, -1);
            // Reject layers that contain nested layers.
            if( spec.channelnames[i].find( '.', n) != std::string::npos)
                return std::pair<int,int>( -1, -1);
            if( begin == -1)
                begin = i;
        } else {
            if( (begin != -1) && (end == -1))
                end = i;
        }
    }

    if( (begin != -1) && (end == -1))
        end = spec.channelnames.size();

    assert( !((begin == -1) ^ (end == -1)));
    return std::pair<int,int>( begin, end);
}

/// Computes an image spec with only the given channel range.
///
/// \note The compute image spec does \em not match the input image, do \em not pass it to any OIIO
///       functions. It is used as input for get_pixel_type() and to transport the modified channel
///       names.
///
/// \param input           The input spec with all channels.
/// \param channel_begin   The first channel index to import.
/// \param channel_end     The last channel index+1 to import.
/// \param prefix          The prefix to strip from all channel names.
/// \return                The computed image spec.
OIIO::ImageSpec extract_channels(
    const OIIO::ImageSpec& input,
    int channel_begin,
    int channel_end,
    const std::string& prefix)
{
    OIIO::ImageSpec output = input;

    output.nchannels = channel_end - channel_begin;

    if( !input.channelformats.empty())
        output.channelformats = std::vector<OIIO::TypeDesc>(
            input.channelformats.begin() + channel_begin,
            input.channelformats.begin() + channel_end);

    output.channelnames = std::vector<std::string>(
        input.channelnames.begin() + channel_begin,
        input.channelnames.begin() + channel_end);
    size_t n = prefix.size();
    for( auto& s: output.channelnames) {
        assert( s.substr( 0, n) == prefix);
        s = s.substr( n);
    }

    output.alpha_channel = get_channel_index( output, std::string(), "A", "Alpha");
    output.z_channel     = get_channel_index( output, std::string(), "Z");

    return output;
}

} // namespace

bool compute_properties(
    OIIO::ImageInput* input,
    const char* selector,
    mi::Uint32& subimage,
    mi::Uint32& resolution_x,
    mi::Uint32& resolution_y,
    mi::Uint32& resolution_z,
    IMAGE::Pixel_type& pixel_type,
    std::vector<std::string>& channel_names,
    mi::Sint32& channel_start,
    mi::Sint32& channel_end)
{
    // No selector, use first subimage.
    if( !selector) {
        subimage      = 0;
        const OIIO::ImageSpec& spec = input->spec();
        resolution_x  = spec.width;
        resolution_y  = spec.height;
        resolution_z  = spec.depth;
        pixel_type    = get_pixel_type( spec);
        channel_names = spec.channelnames;
        channel_start = 0;
        channel_end   = IMAGE::get_components_per_pixel( pixel_type);
        return pixel_type != IMAGE::PT_UNDEF;
    }

    std::string selector_prefix = std::string( selector) + ".";

    // Loop over subimages.
    subimage = 0;
    while( true) {

        const OIIO::ImageSpec& spec = input->spec();

        // Check whether the part name (if available) is a prefix of the selector.
        std::string part_prefix = spec.get_string_attribute( "oiio:subimagename");
        if( !part_prefix.empty())
            part_prefix += ".";
        if( selector_prefix.substr( 0, part_prefix.size()) == part_prefix) {

            // Check whether selector matches exactly one particular channel name
            int channel = get_channel_index( spec, part_prefix, selector);
            if( channel != -1) {
                resolution_x  = spec.width;
                resolution_y  = spec.height;
                resolution_z  = spec.depth;
                std::string strip_prefix = selector + part_prefix.size();
                OIIO::ImageSpec modified_spec
                    = extract_channels( spec, channel, channel+1, strip_prefix);
                pixel_type = get_pixel_type( modified_spec);
                channel_names = modified_spec.channelnames;
                channel_start = channel;
                channel_end   = channel+1;
                return pixel_type != IMAGE::PT_UNDEF;
            }

            // Check whether selector matches a layer
            std::pair<int,int> range
                = get_selector_channel_range( spec, part_prefix, selector_prefix);
            if( range.first != -1) {
                resolution_x  = spec.width;
                resolution_y  = spec.height;
                resolution_z  = spec.depth;
                std::string strip_prefix = selector_prefix.substr( part_prefix.size());
                OIIO::ImageSpec modified_spec
                    = extract_channels( spec, range.first, range.second, strip_prefix);
                pixel_type = get_pixel_type( modified_spec);
                channel_names = modified_spec.channelnames;
                channel_start = range.first;
                channel_end   = range.second;
                return pixel_type != IMAGE::PT_UNDEF;
            }

        }

        if( !input->seek_subimage( ++subimage, /*miplevel*/ 0))
            break;
    }

    // No channel/layer matches the selector
    subimage      = 0;
    resolution_x  = 1;
    resolution_y  = 1;
    resolution_z  = 1;
    pixel_type    = IMAGE::PT_UNDEF;
    channel_names.clear();
    channel_start = -1;
    channel_end   = -1;
    return false;
}

} // namespace MI_OIIO

} // namespace MI
