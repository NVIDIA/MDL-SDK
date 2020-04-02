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
 ** \brief Header for the IMaterial_instance implementation.
 **/

#ifndef API_API_NEURAY_MATERIAL_INSTANCE_IMPL_H
#define API_API_NEURAY_MATERIAL_INSTANCE_IMPL_H

#include <mi/neuraylib/imaterial_instance.h>

#include "neuray_db_element_impl.h"
#include "neuray_attribute_set_impl.h"

namespace MI {

namespace MDL { class Mdl_material_instance; }

namespace NEURAY {

/// This class implements MDL material instances.
class Material_instance_impl final
    : public Attribute_set_impl<Db_element_impl<mi::neuraylib::IMaterial_instance,
    MDL::Mdl_material_instance> >
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

    const char* get_material_definition() const final;

    const char* get_mdl_material_definition() const final;

    mi::Size get_parameter_count() const final;

    const char* get_parameter_name(mi::Size index) const final;

    mi::Size get_parameter_index(const char* name) const final;

    const mi::neuraylib::IType_list* get_parameter_types() const final;

    const mi::neuraylib::IExpression_list* get_arguments() const final;

    mi::Sint32 set_arguments(
        const mi::neuraylib::IExpression_list* arguments) final;

    mi::Sint32 set_argument(
        mi::Size index,
        const mi::neuraylib::IExpression* argument) final;

    mi::Sint32 set_argument(
        const char* name,
        const mi::neuraylib::IExpression* argument) final;

    mi::neuraylib::ICompiled_material* create_compiled_material(
        mi::Uint32 flags,
        mi::neuraylib::IMdl_execution_context* context) const final;

    bool is_default() const final;

    bool is_valid(mi::neuraylib::IMdl_execution_context* context) const final;

    mi_static_assert(sizeof(Compilation_options) == sizeof(mi::Uint32));

    mi::Sint32 repair(
        mi::Uint32 flags,
        mi::neuraylib::IMdl_execution_context* context) final;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_MATERIAL_INSTANCE_IMPL_H
