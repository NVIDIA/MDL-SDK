/***************************************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Header for the ICompiled_material implementation.
 **/

#ifndef API_API_NEURAY_COMPILED_MATERIAL_IMPL_H
#define API_API_NEURAY_COMPILED_MATERIAL_IMPL_H

#include <mi/neuraylib/icompiled_material.h>

#include "neuray_db_element_impl.h"
#include "neuray_attribute_set_impl.h"

namespace mi { namespace neuraylib { class INeuray; } }

namespace MI {

namespace MDL { class Mdl_compiled_material; }

namespace NEURAY {

/// This class implements compiled MDL material instances.
///
class Compiled_material_impl
    : public Attribute_set_impl<Db_element_impl<mi::neuraylib::ICompiled_material,
                                                MDL::Mdl_compiled_material> >
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

    mi::neuraylib::Element_type get_element_type() const final;

    const mi::neuraylib::IExpression_direct_call* get_body() const final;

    mi::Size get_temporary_count() const final;

    const mi::neuraylib::IExpression* get_temporary( mi::Size index) const final;

    mi::Float32 get_mdl_meters_per_scene_unit() const final;

    mi::Float32 get_mdl_wavelength_min() const final;

    mi::Float32 get_mdl_wavelength_max() const final;

    bool depends_on_state_transform() const final;

    bool depends_on_state_object_id() const final;

    bool depends_on_global_distribution() const final;

    bool depends_on_uniform_scene_data() const final;

    mi::Size get_referenced_scene_data_count() const final;

    char const *get_referenced_scene_data_name( mi::Size index) const final;

    mi::Size get_parameter_count() const final;

    const char* get_parameter_name( mi::Size index) const final;

    const mi::neuraylib::IValue* get_argument( mi::Size index) const final;

    mi::base::Uuid get_hash() const final;

    mi::base::Uuid get_slot_hash( mi::neuraylib::Material_slot slot) const final;

    const mi::neuraylib::IExpression* lookup_sub_expression( const char* path) const final;

    const mi::IString* get_connected_function_db_name(
        const char* material_instance_name, 
        mi::Size parameter_index, 
        mi::Sint32* errors) const final;

    mi::neuraylib::Material_opacity get_opacity() const final;

    mi::neuraylib::Material_opacity get_surface_opacity() const final;

    bool get_cutout_opacity(mi::Float32 *cutout_opacity) const final;

    bool is_valid(mi::neuraylib::IMdl_execution_context* context) const final;

    // own methods

    /// Swaps the internal DB class with \p rhs.
    void swap( MDL::Mdl_compiled_material& rhs);
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_COMPILED_MATERIAL_IMPL_H
