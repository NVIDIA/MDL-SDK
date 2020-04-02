/***************************************************************************************************
 * Copyright (c) 2010-2020, NVIDIA CORPORATION. All rights reserved.
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

    mi::Sint32 reset_file( const char* filename);

    mi::Sint32 reset_reader( mi::neuraylib::IReader* reader, const char* image_format);

    Sint32 reset_reader( mi::IArray* reader, const char* image_format);

    const char* get_filename( mi::Uint32 uvtile_id = 0) const;

    const char* get_original_filename() const;

    const char* get_type( mi::Uint32 uvtile_id = 0) const;

    bool set_from_canvas( const mi::neuraylib::ICanvas* canvas);

    bool set_from_canvas( mi::neuraylib::ICanvas* canvas, bool shared);

    bool set_from_canvas( const mi::IArray* canvas);

    bool set_from_canvas( mi::IArray* canvas, bool shared);

    const mi::neuraylib::ICanvas* get_canvas( mi::Uint32 level, mi::Uint32 uvtile_id = 0) const;

    mi::Uint32 get_levels( mi::Uint32 uvtile_id = 0) const;

    mi::Uint32 resolution_x( mi::Uint32 level, mi::Uint32 uvtile_id = 0) const;

    mi::Uint32 resolution_y( mi::Uint32 level, mi::Uint32 uvtile_id = 0) const;

    mi::Uint32 resolution_z( mi::Uint32 level, mi::Uint32 uvtile_id = 0) const;

    bool is_uvtile() const;

    void get_uvtile_uv_ranges(
        mi::Sint32& min_u, mi::Sint32& min_v, mi::Sint32& max_u, mi::Sint32& max_v) const;

    mi::Size get_uvtile_length() const;

    mi::Sint32 get_uvtile_uv( Uint32 uvtile_id, Sint32& u, Sint32& v) const;

    mi::Uint32 get_uvtile_id( Sint32 u, Sint32 v) const;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_NEURAY_IMAGE_IMPL_H
