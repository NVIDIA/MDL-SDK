/***************************************************************************************************
 * Copyright (c) 2010-2025, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Header for the IImage implementation.
 **/

#ifndef API_API_NEURAY_NEURAY_IMAGE_IMPL_H
#define API_API_NEURAY_NEURAY_IMAGE_IMPL_H

#include <mi/neuraylib/iimage.h>

#include "neuray_db_element_impl.h"
#include "neuray_attribute_set_impl.h"

#include <string>

namespace MI {

namespace DBIMAGE { class Image; }

namespace NEURAY {

class Image_impl
  : public Attribute_set_impl<Db_element_impl<mi::neuraylib::IImage, DBIMAGE::Image> >
{
public:

    static DB::Element_base* create_db_element(
        mi::neuraylib::ITransaction* transaction,
        mi::Uint32 argc,
        const mi::base::IInterface* argv[]);

    static mi::base::IInterface* create_api_class(
        mi::neuraylib::ITransaction* transaction,
        mi::Uint32 argc,
        const mi::base::IInterface* argv[]);

    // public API methods

    mi::neuraylib::Element_type get_element_type() const;

    mi::Sint32 reset_file( const char* filename, const char* selector);

    mi::Sint32 reset_reader(
        mi::neuraylib::IReader* reader, const char* image_format, const char* selector);

    mi::Sint32 reset_reader( mi::IArray* reader, const char* image_format, const char* selector);

    bool set_from_canvas( const mi::neuraylib::ICanvas* canvas, const char* selector);

    bool set_from_canvas( mi::neuraylib::ICanvas* canvas, const char* selector, bool shared);

    bool set_from_canvas( const mi::IArray* canvas, const char* selector);

    bool set_from_canvas( mi::IArray* canvas, const char* selector, bool shared);

    bool is_animated() const;

    mi::Size get_length() const;

    mi::Size get_frame_number( mi::Size frame_id) const;

    mi::Size get_frame_id( mi::Size frame_number) const;

    bool is_uvtile() const;

    mi::Size get_frame_length( mi::Size frame_id) const;

    mi::Sint32 get_uvtile_uv(
        mi::Size frame_id, mi::Size uvtile_id, mi::Sint32& u, mi::Sint32& v) const;

    mi::Size get_uvtile_id( mi::Size frame_id, mi::Sint32 u, mi::Sint32 v) const;

    void get_uvtile_uv_ranges(
        mi::Size frame_id,
        mi::Sint32& min_u,
        mi::Sint32& min_v,
        mi::Sint32& max_u,
        mi::Sint32& max_v) const;

    const char* get_original_filename() const;

    const char* get_selector() const;

    const char* get_filename( mi::Size frame_id, mi::Size uvtile_id) const;

    const mi::neuraylib::ICanvas* get_canvas(
        mi::Size frame_id, mi::Size uvtile_id, mi::Uint32 level) const;

    const char* get_type( mi::Size frame_id, mi::Size uvtile_id) const;

    mi::Uint32 get_levels( mi::Size frame_id, mi::Size uvtile_id) const;

    mi::Uint32 resolution_x( mi::Size frame_id, mi::Size uvtile_id, mi::Uint32 level) const;

    mi::Uint32 resolution_y( mi::Size frame_id, mi::Size uvtile_id, mi::Uint32 level) const;

    mi::Uint32 resolution_z( mi::Size frame_id, mi::Size uvtile_id, mi::Uint32 level) const;

private:

    mutable std::string m_cached_selector;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_NEURAY_IMAGE_IMPL_H
