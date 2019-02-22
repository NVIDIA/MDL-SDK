/***************************************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
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

#include <cstring>
#include <string>

#include <mi/base/types.h>
#include <mi/mdl/mdl_mdl.h>
#include <mi/mdl/mdl_symbols.h>
#include <mi/mdl/mdl_types.h>
#include <base/data/db/i_db_access.h>
#include <io/scene/mdl_elements/mdl_elements_detail.h> // DETAIL::Type_binder
#include <io/scene/mdl_elements/i_mdl_elements_compiled_material.h>
#include <io/scene/mdl_elements/i_mdl_elements_function_call.h>
#include <io/scene/mdl_elements/i_mdl_elements_function_definition.h>
#include <io/scene/mdl_elements/i_mdl_elements_utilities.h>
#include <render/mdl/runtime/i_mdlrt_resource_handler.h>
#include <render/mdl/backends/backends_target_code.h>
#include <api/api/neuray/neuray_compiled_material_impl.h>
#include <api/api/neuray/neuray_function_call_impl.h>
#include <api/api/neuray/neuray_transaction_impl.h>
#include <api/api/neuray/neuray_mdl_execution_context_impl.h>

#include "mdl_mdl_backend_impl.h"

namespace MI {

namespace MDL {

static DB::Transaction *unwrap(mi::neuraylib::ITransaction *transaction)
{
    NEURAY::Transaction_impl *transaction_impl =
        static_cast<NEURAY::Transaction_impl *>(transaction);
    return transaction_impl->get_db_transaction();
}

static MDL::Mdl_function_call const *unwrap(mi::neuraylib::IFunction_call const *function_call)
{
    NEURAY::Function_call_impl const *function_call_impl =
        static_cast<NEURAY::Function_call_impl const *>(function_call);
     return function_call_impl->get_db_element();
}

static MDL::Mdl_compiled_material const *unwrap(
    mi::neuraylib::ICompiled_material const *compiled_material)
{
    NEURAY::Compiled_material_impl const *compiled_material_impl =
        static_cast<NEURAY::Compiled_material_impl const *>(compiled_material);
     return compiled_material_impl->get_db_element();
}

static BACKENDS::Link_unit const *unwrap(mi::neuraylib::ILink_unit const *lu)
{
    MI::MDL::Link_unit const *lu_impl =
        static_cast<MI::MDL::Link_unit const *>(lu);
    return lu_impl->get_link_unit();
}

static MDL::Execution_context *unwrap_and_clear(mi::neuraylib::IMdl_execution_context *context)
{
    NEURAY::Mdl_execution_context_impl *context_impl =
        static_cast<NEURAY::Mdl_execution_context_impl *>(context);
    if (context_impl) {
        MDL::Execution_context& wrapped_context = context_impl->get_context();
        wrapped_context.clear_messages();
        return &wrapped_context;
    }

    return NULL;
}

// Constructor from an LLVM backend.
Link_unit::Link_unit(
    BACKENDS::Mdl_llvm_backend  &be,
    mi::neuraylib::ITransaction *transaction)
: m_link_unit(be, unwrap(transaction))
{
}


// Add an MDL environment function call as a function to this link unit.
mi::Sint32 Link_unit::deprecated_add_environment(
    mi::neuraylib::IFunction_call const *call,
    char const                          *fname,
    mi::Float32                         mdl_meters_per_scene_unit,
    mi::Float32                         mdl_wavelength_min,
    mi::Float32                         mdl_wavelength_max)
{
    MDL::Execution_context context;
    context.set_option("mdl_meters_per_scene_unit", mdl_meters_per_scene_unit);
    context.set_option("mdl_wavelength_min", mdl_wavelength_min);
    context.set_option("mdl_wavelength_max", mdl_wavelength_max);

    m_link_unit.add_environment(unwrap(call), fname, &context);
    return context.get_result();
}

// Add an MDL environment function call as a function to this link unit.
mi::Sint32 Link_unit::add_environment(
    mi::neuraylib::IFunction_call const *call,
    char const                          *fname,
    mi::neuraylib::IMdl_execution_context* context)
{
    return m_link_unit.add_environment(unwrap(call), fname, unwrap_and_clear(context));
}

// Add an expression that is part of an MDL material instance as a function to this link unit.
mi::Sint32 Link_unit::deprecated_add_material_expression(
    mi::neuraylib::ICompiled_material const *material,
    char const                              *path,
    char const                              *fname)
{
    MDL::Execution_context context;
    m_link_unit.add_material_expression(unwrap(material), path, fname, &context);
    return context.get_result();
}

// Add an expression that is part of an MDL material instance as a function to this link unit.
mi::Sint32 Link_unit::add_material_expression(
    mi::neuraylib::ICompiled_material const *material,
    char const                              *path,
    char const                              *fname,
    mi::neuraylib::IMdl_execution_context   *context)
{
    return m_link_unit.add_material_expression(unwrap(material), path, fname, unwrap_and_clear(context));
}

// Add an MDL distribution function to this link unit.
mi::Sint32 Link_unit::deprecated_add_material_df(
    mi::neuraylib::ICompiled_material const *material,
    char const                              *path,
    char const                              *base_fname,
    bool                                     include_geometry_normal)
{
    MDL::Execution_context context;
    context.set_option(MDL_CTX_OPTION_INCLUDE_GEO_NORMAL, include_geometry_normal);
    m_link_unit.add_material_df(
        unwrap(material), path, base_fname, &context);
    return context.get_result();
}

// Add an MDL distribution function to this link unit.
mi::Sint32 Link_unit::add_material_df(
    mi::neuraylib::ICompiled_material const *material,
    char const                              *path,
    char const                              *base_fname,
    mi::neuraylib::IMdl_execution_context   *context)
{
    return m_link_unit.add_material_df(
        unwrap(material), path, base_fname, unwrap_and_clear(context));
}

mi::Sint32 Link_unit::deprecated_add_material(
    mi::neuraylib::ICompiled_material const    *material,
    mi::neuraylib::Target_function_description *function_descriptions,
    mi::Size                                    description_count,
    bool                                        include_geometry_normal)
{
    MDL::Execution_context context;
    context.set_option(MDL_CTX_OPTION_INCLUDE_GEO_NORMAL, include_geometry_normal);

    mi::Sint32 result = m_link_unit.add_material(
        unwrap(material),
        reinterpret_cast<mi::mdl::ILink_unit::Target_function_description*>(function_descriptions),
        static_cast<size_t>(description_count),
        &context);

    if(result < 0) // store legacy error code
        for(mi::Size i=0; i<description_count; ++i)
            if (function_descriptions[i].return_code == -1) {
                function_descriptions[i].return_code = context.get_result();
                break;
            }
    return result;
}

mi::Sint32 Link_unit::add_material(
    mi::neuraylib::ICompiled_material const    *material,
    mi::neuraylib::Target_function_description *function_descriptions,
    mi::Size                                    description_count,
    mi::neuraylib::IMdl_execution_context      *context)
{
    return m_link_unit.add_material(
        unwrap(material),
        reinterpret_cast<mi::mdl::ILink_unit::Target_function_description*>(function_descriptions),
        static_cast<size_t>(description_count),
        unwrap_and_clear(context));
}

Mdl_llvm_backend::Mdl_llvm_backend(
    mi::neuraylib::IMdl_compiler::Mdl_backend_kind kind,
    mi::mdl::IMDL                *compiler,
    mi::mdl::ICode_generator_jit *jit,
    mi::mdl::ICode_cache         *code_cache,
    bool                         string_ids)
: m_backend(kind, compiler, jit, code_cache, string_ids)
{
}

mi::Sint32 Mdl_llvm_backend::set_option(char const *name, char const *value)
{
    return m_backend.set_option(name, value);
}

mi::Sint32 Mdl_llvm_backend::set_option_binary(char const *name, const char* data, mi::Size size)
{
    return m_backend.set_option_binary(name, data, size);
}

mi::neuraylib::ITarget_code const *Mdl_llvm_backend::deprecated_translate_environment(
    mi::neuraylib::ITransaction         *transaction,
    mi::neuraylib::IFunction_call const *function_call,
    mi::Float32                         mdl_meters_per_scene_unit,
    mi::Float32                         mdl_wavelength_min,
    mi::Float32                         mdl_wavelength_max,
    char const                          *fname,
    mi::Sint32                          *errors)
{
    mi::Sint32 dummy_errors;
    if (errors == NULL)
        errors = &dummy_errors;

    if (transaction == NULL || function_call == NULL) {
        *errors = -1;
        return NULL;
    }
    MDL::Execution_context context;
    context.set_option("mdl_meters_per_scene_unit", mdl_meters_per_scene_unit);
    context.set_option("mdl_wavelength_min", mdl_wavelength_min);
    context.set_option("mdl_wavelength_max", mdl_wavelength_max);

    DB::Transaction *db_transaction = unwrap(transaction);
    MDL::Mdl_function_call const *db_function_call = unwrap(function_call);

    mi::neuraylib::ITarget_code const *target_code = m_backend.translate_environment(
        db_transaction,
        db_function_call,
        fname,
        &context);
    *errors = context.get_result();
    return target_code;
}

mi::neuraylib::ITarget_code const *Mdl_llvm_backend::translate_environment(
    mi::neuraylib::ITransaction           *transaction,
    mi::neuraylib::IFunction_call const   *function_call,
    char const                            *fname,
    mi::neuraylib::IMdl_execution_context *context)
{
    MDL::Execution_context* context_impl = unwrap_and_clear(context);

    if (transaction == NULL || function_call == NULL) {
        if (context_impl) {
            MDL::Message err(mi::base::MESSAGE_SEVERITY_ERROR, "Invalid parameters (NULL pointer).");
            context_impl->add_error_message(err);
            context_impl->add_message(err);
        }
        return NULL;
    }

    DB::Transaction *db_transaction = unwrap(transaction);
    MDL::Mdl_function_call const *db_function_call = unwrap(function_call);
   
    return m_backend.translate_environment(
        db_transaction,
        db_function_call,
        fname,
        context_impl);
}

mi::neuraylib::ITarget_code const *Mdl_llvm_backend::deprecated_translate_material_expression(
    mi::neuraylib::ITransaction             *transaction,
    mi::neuraylib::ICompiled_material const *compiled_material,
    char const                              *path,
    char const                              *fname,
    mi::Sint32                              *errors)
{
    mi::Sint32 dummy_errors;
    if (errors == NULL)
        errors = &dummy_errors;

    if (transaction == NULL || compiled_material == NULL) {
        *errors = -1;
        return NULL;
    }

    MDL::Execution_context context;
    mi::neuraylib::ITarget_code const *tc = m_backend.translate_material_expression(
        unwrap(transaction),
        unwrap(compiled_material),
        path,
        fname,
        &context);

    *errors = context.get_result();
    return tc;
}

mi::neuraylib::ITarget_code const *Mdl_llvm_backend::translate_material_expression(
    mi::neuraylib::ITransaction             *transaction,
    mi::neuraylib::ICompiled_material const *compiled_material,
    char const                              *path,
    char const                              *fname,
    mi::neuraylib::IMdl_execution_context   *context)
{
  
    MDL::Execution_context* context_impl = unwrap_and_clear(context);

    if (transaction == NULL || compiled_material == NULL) {
        if (context_impl) {
            MDL::Message err(mi::base::MESSAGE_SEVERITY_ERROR, 
                "Invalid parameters (NULL pointer).");
            context_impl->add_error_message(err);
            context_impl->add_message(err);
        }
        return NULL;
    }

    return m_backend.translate_material_expression(
        unwrap(transaction), unwrap(compiled_material), path, fname, context_impl);
}

mi::neuraylib::ITarget_code const *Mdl_llvm_backend::deprecated_translate_material_expressions(
    mi::neuraylib::ITransaction             *transaction,
    mi::neuraylib::ICompiled_material const *compiled_material,
    char const * const                      paths[],
    mi::Uint32                              path_cnt,
    char const                              *fname,
    mi::Sint32                              *errors)
{
    mi::Sint32 dummy_errors;
    if (errors == NULL)
        errors = &dummy_errors;

    if (transaction == NULL || compiled_material == NULL) {
        *errors = -1;
        return NULL;
    }

    DB::Transaction *db_transaction = unwrap(transaction);
    MDL::Mdl_compiled_material const *db_compiled_material = unwrap(compiled_material);

    return m_backend.translate_material_expressions(
        db_transaction,
        db_compiled_material,
        paths,
        path_cnt,
        fname,
        errors);
}

mi::neuraylib::ITarget_code const*
Mdl_llvm_backend::deprecated_translate_material_expression_uniform_state(
    mi::neuraylib::ITransaction             *transaction,
    mi::neuraylib::ICompiled_material const *compiled_material,
    char const                              *path,
    char const                              *fname,
    mi::Float32_4_4_struct const            &world_to_obj,
    mi::Float32_4_4_struct const            &obj_to_world,
    mi::Sint32                              object_id,
    mi::Sint32                              *errors)
{
    Sint32 dummy_errors;
    if (errors == NULL)
        errors = &dummy_errors;

    if (transaction == NULL || compiled_material == NULL) {
        *errors = -1;
        return NULL;
    }

    DB::Transaction *db_transaction = unwrap(transaction);
    MDL::Mdl_compiled_material const *db_compiled_material = unwrap(compiled_material);

    return m_backend.translate_material_expression_uniform_state(
        db_transaction,
        db_compiled_material,
        path,
        fname,
        world_to_obj,
        obj_to_world,
        object_id,
        errors);
}

const mi::neuraylib::ITarget_code* Mdl_llvm_backend::deprecated_translate_material_df(
    mi::neuraylib::ITransaction* transaction,
    const mi::neuraylib::ICompiled_material* compiled_material,
    const char* path,
    const char* base_fname,
    bool include_geometry_normal,
    mi::Sint32* errors)
{
    mi::Sint32 dummy_errors;
    if( !errors)
        errors = &dummy_errors;

    if( !transaction || !compiled_material || !path) {
        *errors = -1;
        return 0;
    }

    MDL::Execution_context context;
    context.set_option(MDL_CTX_OPTION_INCLUDE_GEO_NORMAL, include_geometry_normal);

    const mi::neuraylib::ITarget_code* tc =  m_backend.translate_material_df(
        unwrap(transaction), unwrap(compiled_material), path, base_fname, &context);
    *errors = context.get_result();
    return tc;
}

const mi::neuraylib::ITarget_code* Mdl_llvm_backend::translate_material_df(
    mi::neuraylib::ITransaction* transaction,
    const mi::neuraylib::ICompiled_material* compiled_material,
    const char* path,
    const char* base_fname,
    mi::neuraylib::IMdl_execution_context* context)
{
    MDL::Execution_context* context_impl = unwrap_and_clear(context);

    if (!transaction || !compiled_material || !path) {
        if (context_impl) {
            MDL::Message err(mi::base::MESSAGE_SEVERITY_ERROR, "Invalid parameters (NULL pointer).");
            context_impl->add_error_message(err);
            context_impl->add_message(err);
        }
        return 0;
    }

    return m_backend.translate_material_df(
        unwrap(transaction), unwrap(compiled_material), path, base_fname, context_impl);
}

const mi::neuraylib::ITarget_code* Mdl_llvm_backend::deprecated_translate_material(
    mi::neuraylib::ITransaction* transaction,
    const mi::neuraylib::ICompiled_material* material,
    mi::neuraylib::Target_function_description* function_descriptions,
    mi::Size description_count,
    bool include_geometry_normal)
{
    NEURAY::Mdl_execution_context_impl context;
    context.set_option(MDL_CTX_OPTION_INCLUDE_GEO_NORMAL, include_geometry_normal);

    return translate_material(
        transaction, material, function_descriptions, description_count, &context);
}

const mi::neuraylib::ITarget_code* Mdl_llvm_backend::translate_material(
    mi::neuraylib::ITransaction* transaction,
    const mi::neuraylib::ICompiled_material* material,
    mi::neuraylib::Target_function_description* function_descriptions,
    mi::Size description_count,
    mi::neuraylib::IMdl_execution_context* context)
{
    MDL::Execution_context* wrapped_context = unwrap_and_clear(context);

    // reuse link unit based implementation
    BACKENDS::Link_unit link_unit(m_backend, unwrap(transaction));

    if (link_unit.add_material(
        unwrap(material),
        reinterpret_cast<mi::mdl::ILink_unit::Target_function_description*>(function_descriptions),
        static_cast<size_t>(description_count),
        wrapped_context)) {

        return m_backend.translate_link_unit(&link_unit, wrapped_context);
    }
    return NULL;
}

mi::Uint8 const *Mdl_llvm_backend::get_device_library(Size &size) const
{
    return m_backend.get_device_library(size);
}

mi::neuraylib::ILink_unit* Mdl_llvm_backend::deprecated_create_link_unit(
    mi::neuraylib::ITransaction* transaction,
    mi::Sint32* errors)
{
    if (errors != NULL)
        *errors = 0;
    return new Link_unit(m_backend, transaction);
}

mi::neuraylib::ILink_unit* Mdl_llvm_backend::create_link_unit(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_execution_context* )
{
    return new Link_unit(m_backend, transaction);
}

mi::neuraylib::ITarget_code const* Mdl_llvm_backend::deprecated_translate_link_unit(
    mi::neuraylib::ILink_unit const* lu,
    mi::Sint32* errors)
{
    MDL::Execution_context context;
    const mi::neuraylib::ITarget_code* tc =
        m_backend.translate_link_unit(unwrap(lu), &context);

    if (errors)
        *errors = context.get_result();
    return tc;
}

mi::neuraylib::ITarget_code const* Mdl_llvm_backend::translate_link_unit(
    mi::neuraylib::ILink_unit const* lu,
    mi::neuraylib::IMdl_execution_context* context)
{
    return m_backend.translate_link_unit(unwrap(lu), unwrap_and_clear(context));
}


} // namespace MDL

#ifdef ENABLE_ASSERT
namespace IRAY {
    void mi_iray_assertion_failed(char const *expr, char const *file, unsigned int line)
    {
        LOG::report_assertion_failure(SYSTEM::M_IRAY, expr, file, line);
    }
}  // IRAY
#endif

} // namespace MI

