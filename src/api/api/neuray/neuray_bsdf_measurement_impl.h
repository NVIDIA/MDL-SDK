/***************************************************************************************************
 * Copyright (c) 2013-2020, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Header for the IBsdf_measurement implementation.
 **/

#ifndef API_API_NEURAY_NEURAY_BSDF_MEASUREMENT_IMPL_H
#define API_API_NEURAY_NEURAY_BSDF_MEASUREMENT_IMPL_H

#include <mi/neuraylib/ibsdf_measurement.h>

#include "neuray_db_element_impl.h"
#include "neuray_attribute_set_impl.h"

namespace MI {

namespace BSDFM { class Bsdf_measurement; }

namespace NEURAY {

class Bsdf_measurement_impl
  : public Attribute_set_impl<Db_element_impl<mi::neuraylib::IBsdf_measurement,
                                              BSDFM::Bsdf_measurement> >
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

    mi::Sint32 reset_reader( mi::neuraylib::IReader* reader);

    const char* get_filename() const;

    const char* get_original_filename() const;

    mi::Sint32 set_reflection( const mi::neuraylib::IBsdf_isotropic_data* bsdf_data);

    const mi::base::IInterface* get_reflection() const;

    mi::Sint32 set_transmission( const mi::neuraylib::IBsdf_isotropic_data* bsdf_data);

    const mi::base::IInterface* get_transmission() const;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_NEURAY_BSDF_MEASUREMENT_IMPL_H
