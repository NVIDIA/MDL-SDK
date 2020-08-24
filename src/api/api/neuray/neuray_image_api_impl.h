/***************************************************************************************************
 * Copyright (c) 2009-2020, NVIDIA CORPORATION. All rights reserved.
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

/** \file
 ** \brief Implementation of IImage_api
 **
 ** Implements the IImage_api interface
 **/

#ifndef API_API_NEURAY_IMAGE_API_IMPL_H
#define API_API_NEURAY_IMAGE_API_IMPL_H

#include <mi/base/interface_implement.h>
#include <mi/neuraylib/iimage_api.h>

#include <boost/core/noncopyable.hpp>
#include <base/system/main/access_module.h>

namespace mi { class IArray; namespace neuraylib { class INeuray; } }

namespace MI {

namespace IMAGE { class Image_module; }

namespace NEURAY {

class Image_api_impl
  : public mi::base::Interface_implement<mi::neuraylib::IImage_api>,
    public boost::noncopyable
{
public:
    /// Construction of Image_api_impl
    ///
    /// \param neuray   The neuray instance which contains this Image_api_impl
    Image_api_impl( mi::neuraylib::INeuray* neuray);

    /// Destructor of Image_api_impl
    ~Image_api_impl();

    // public API methods

    mi::neuraylib::ICanvas* create_canvas(
        const char* pixel_type,
        mi::Uint32 width,
        mi::Uint32 height,
        mi::Uint32 tile_width,
        mi::Uint32 tile_height,
        mi::Uint32 layers,
        bool is_cubemap,
        mi::Float32 gamma) const;

    mi::neuraylib::ICanvas_cuda* create_canvas_cuda(
        mi::Sint32 cuda_device_id,
        const char* pixel_type,
        mi::Uint32 width,
        mi::Uint32 height,
        mi::Uint32 layers,
        mi::Float32 gamma) const;

    mi::neuraylib::ITile* create_tile(
        const char* pixel_type,
        mi::Uint32 width,
        mi::Uint32 height) const;

    mi::Sint32 read_raw_pixels(
        mi::Uint32 width,
        mi::Uint32 height,
        const mi::neuraylib::ICanvas* canvas,
        mi::Uint32 canvas_x,
        mi::Uint32 canvas_y,
        mi::Uint32 canvas_layer,
        void* buffer,
        bool buffer_topdown,
        const char* buffer_pixel_type,
        mi::Uint32 buffer_padding) const;

    mi::Sint32 write_raw_pixels(
        mi::Uint32 width,
        mi::Uint32 height,
        mi::neuraylib::ICanvas* canvas,
        mi::Uint32 canvas_x,
        mi::Uint32 canvas_y,
        mi::Uint32 canvas_layer,
        const void* buffer,
        bool buffer_topdown,
        const char* buffer_pixel_type,
        mi::Uint32 buffer_padding) const;

    mi::neuraylib::IBuffer* create_buffer_from_canvas(
        const mi::neuraylib::ICanvas* canvas,
        const char* image_format,
        const char* pixel_type,
        const char* quality) const;

    mi::neuraylib::ICanvas* create_canvas_from_buffer(
        const mi::neuraylib::IBuffer* buffer,
        const char* image_format) const;

    bool supports_format_for_decoding(
        const char* image_format, mi::neuraylib::IReader* reader) const;

    bool supports_format_for_encoding( const char* image_format) const;

    mi::neuraylib::ICanvas* convert(
        const mi::neuraylib::ICanvas* canvas, const char* pixel_type) const;

    void adjust_gamma( mi::neuraylib::ICanvas* canvas, mi::Float32 new_gamma) const;

    mi::Uint32 get_components_per_pixel( const char* pixel_type) const;

    mi::Uint32 get_bytes_per_component( const char* pixel_type) const;

    mi::IArray* create_mipmaps(const mi::neuraylib::ICanvas* canvas, mi::Float32 gamma) const;

    // internal methods

    /// Starts this API component.
    ///
    /// The implementation of INeuray::start() calls the #start() method of each API component.
    /// This method performs the API component's specific part of the library start.
    ///
    /// \return            0, in case of success, -1 in case of failure.
    mi::Sint32 start();

    /// Shuts down this API component.
    ///
    /// The implementation of INeuray::shutdown() calls the #shutdown() method of each API
    /// component. This method performs the API component's specific part of the library shutdown.
    ///
    /// \return           0, in case of success, -1 in case of failure
    mi::Sint32 shutdown();


private:

    /// Pointer to INeuray instance
    mi::neuraylib::INeuray* m_neuray;

    /// Access to the IMAGE module
    SYSTEM::Access_module<IMAGE::Image_module> m_image_module;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_IMAGE_API_IMPL_H
