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
 ** \brief Header for the IModule implementation.
 **/

#ifndef API_API_NEURAY_MODULE_IMPL_H
#define API_API_NEURAY_MODULE_IMPL_H

#include <mi/neuraylib/imodule.h>
#include <mi/neuraylib/imaterial_definition.h>

#include "neuray_db_element_impl.h"
#include "neuray_attribute_set_impl.h"

namespace MI {

namespace MDL { class Mdl_module; }

namespace NEURAY {

/// This class implements an MDL module.
class Module_impl
    : public Attribute_set_impl<Db_element_impl<mi::neuraylib::IModule, MDL::Mdl_module> >
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

    const char* get_filename() const;

    const char* get_mdl_name() const;

    mi::Size get_mdl_package_component_count() const;

    const char* get_mdl_package_component_name( mi::Size index) const;

    const char* get_mdl_simple_name() const;

    mi::neuraylib::Mdl_version get_mdl_version() const;

    mi::Size get_import_count() const;

    const char* get_import( mi::Size index) const;

    const mi::neuraylib::IType_list* get_types() const;

    const mi::neuraylib::IValue_list* get_constants() const;

    mi::Size get_function_count() const;

    const char* get_function( mi::Size index) const;

    mi::Size get_material_count() const;

    const char* get_material( mi::Size index) const;

    const mi::neuraylib::IAnnotation_block* get_annotations() const;

    bool is_standard_module() const;

    bool is_mdle_module() const;

    const mi::IArray* get_function_overloads(
        const char* name, const mi::neuraylib::IExpression_list* arguments) const;

    const mi::IArray* get_function_overloads(
        const char* name, const mi::IArray* parameter_types) const;

    mi::Size get_resources_count() const;

    const mi::neuraylib::IType_resource* get_resource_type(mi::Size index) const;

    const char* get_resource_mdl_file_path(mi::Size index) const;

    const char* get_resource_name(mi::Size index) const;

    mi::Size get_annotation_definition_count() const;

    const mi::neuraylib::IAnnotation_definition* get_annotation_definition(
        mi::Size index) const;

    const mi::neuraylib::IAnnotation_definition* get_annotation_definition(
        const char* name) const;

    bool is_valid( mi::neuraylib::IMdl_execution_context* context) const;

    mi::Sint32 reload(
        bool recursive,
        mi::neuraylib::IMdl_execution_context* context);

    mi::Sint32 reload_from_string(
        const char* module_source,
        bool recursive,
        mi::neuraylib::IMdl_execution_context* context);

    const mi::IArray* deprecated_get_function_overloads(
        const char* name, const char* param_sig) const;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_MODULE_IMPL_H
