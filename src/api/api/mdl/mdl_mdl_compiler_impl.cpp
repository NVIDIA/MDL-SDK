/***************************************************************************************************
 * Copyright (c) 2012-2020, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Implementation of IMdl_compiler
 **
 ** Implements the IMdl_compiler interface
 **/

#include "pch.h"

#include "mdl_mdl_compiler_impl.h"
#include "mdl_neuray_impl.h"

#include "neuray_impexp_utilities.h"

#include "neuray_module_impl.h"
#include "neuray_string_impl.h"
#include "neuray_transaction_impl.h"
#include "neuray_mdl_execution_context_impl.h"

#include <mi/neuraylib/imdl_backend_api.h>
#include <mi/neuraylib/imdl_impexp_api.h>

#include <mi/mdl/mdl_modules.h>
#include <mi/mdl/mdl_mdl.h>
#include <mi/mdl/mdl_entity_resolver.h>

#include <base/hal/disk/disk_memory_reader_writer_impl.h>
#include <base/lib/log/i_log_logger.h>
#include <base/lib/path/i_path.h>
#include <base/lib/plug/i_plug.h>
#include <mdl/integration/mdlnr/i_mdlnr.h>
#include <io/scene/mdl_elements/i_mdl_elements_module.h>


namespace MI {

namespace MDL {

Mdl_compiler_impl::Mdl_compiler_impl( Neuray_impl* neuray_impl)
  : m_neuray_impl( neuray_impl),
    m_attr_module( true),
    m_mdlc_module( true),
    m_mem_module( false),
    m_path_module( false),
    m_plug_module( false)
{
}

Mdl_compiler_impl::~Mdl_compiler_impl()
{
    m_neuray_impl = 0;
}

mi::Sint32 Mdl_compiler_impl::add_builtin_module(
    const char* module_name, const char* module_source)
{
    if (!module_name || !module_source)
        return -1;

    mi::base::Handle<mi::mdl::IMDL> compiler(m_mdlc_module->get_mdl());
    bool success = compiler->add_builtin_module(
        module_name,
        module_source,
        strlen(module_source),
        /*is_encoded*/ false,
        /*is_native*/ true);
    return success ? 0 : -1;
}

void Mdl_compiler_impl::deprecated_set_logger( mi::base::ILogger* logger)
{
    m_neuray_impl->set_logger( logger);
}

mi::base::ILogger* Mdl_compiler_impl::deprecated_get_logger()
{
    return m_neuray_impl->get_logger();
}

mi::Sint32 Mdl_compiler_impl::deprecated_load_plugin_library( const char* path)
{
    if( !path)
        return -1;

    mi::neuraylib::INeuray::Status status = m_neuray_impl->get_status();
    if( status != mi::neuraylib::INeuray::PRE_STARTING)
        return -1;

    return m_plug_module->load_library( path) ? 0 : -1;
}

mi::Sint32 Mdl_compiler_impl::deprecated_add_module_path( const char* path)
{
    return !path ? -1 : m_path_module->add_path( PATH::MDL, path);
}

mi::Sint32 Mdl_compiler_impl::deprecated_remove_module_path( const char* path)
{
    return !path ? -1 : m_path_module->remove_path( PATH::MDL, path);
}

void Mdl_compiler_impl::deprecated_clear_module_paths()
{
    m_path_module->clear_search_path( PATH::MDL);
}

mi::Size Mdl_compiler_impl::deprecated_get_module_paths_length() const
{
    return m_path_module->get_path_count( PATH::MDL);
}

const mi::IString* Mdl_compiler_impl::deprecated_get_module_path( mi::Size index) const
{
    const std::string& result = m_path_module->get_path( PATH::MDL, index);
    if( result.empty())
        return 0;
    mi::IString* istring = new NEURAY::String_impl();
    istring->set_c_str( result.c_str());
    return istring;
}

mi::Sint32 Mdl_compiler_impl::deprecated_add_resource_path( const char* path)
{
    return !path ? -1 : m_path_module->add_path( PATH::RESOURCE, path);
}

mi::Sint32 Mdl_compiler_impl::deprecated_remove_resource_path( const char* path)
{
    return !path ? -1 : m_path_module->remove_path( PATH::RESOURCE, path);
}

void Mdl_compiler_impl::deprecated_clear_resource_paths()
{
    m_path_module->clear_search_path( PATH::RESOURCE);
}

mi::Size Mdl_compiler_impl::deprecated_get_resource_paths_length() const
{
    return m_path_module->get_path_count( PATH::RESOURCE);
}

const mi::IString* Mdl_compiler_impl::deprecated_get_resource_path( mi::Size index) const
{
    const std::string& result = m_path_module->get_path( PATH::RESOURCE, index);
    if( result.empty())
        return 0;
    mi::IString* istring = new NEURAY::String_impl();
    istring->set_c_str( result.c_str());
    return istring;
}

mi::Sint32 Mdl_compiler_impl::deprecated_load_module(
    mi::neuraylib::ITransaction* transaction,
    const char* module_name,
    mi::neuraylib::IMdl_execution_context* context)
{
    if (!transaction || !module_name)
        return -1;

    NEURAY::Transaction_impl* transaction_impl
        = static_cast<NEURAY::Transaction_impl*>(transaction);
    DB::Transaction* db_transaction = transaction_impl->get_db_transaction();

    MDL::Execution_context default_context;
    return MDL::Mdl_module::create_module(
        db_transaction, module_name, NEURAY::unwrap_and_clear_context(context, default_context));
}

const char* Mdl_compiler_impl::deprecated_get_module_db_name(
    mi::neuraylib::ITransaction* transaction,
    const char* module_name,
    mi::neuraylib::IMdl_execution_context* context)
{
    if (!transaction || !module_name)
        return NULL;

    NEURAY::Transaction_impl* transaction_impl
        = static_cast<NEURAY::Transaction_impl*>(transaction);
    DB::Transaction* db_transaction = transaction_impl->get_db_transaction();

    MDL::Execution_context default_context;
    return MDL::Mdl_module::deprecated_get_module_db_name(
        db_transaction, module_name, NEURAY::unwrap_and_clear_context(context, default_context));
}

mi::Sint32 Mdl_compiler_impl::deprecated_load_module_from_string(
    mi::neuraylib::ITransaction* transaction,
    const char* module_name,
    const char* module_source,
    mi::neuraylib::IMdl_execution_context* context)
{
    if (!transaction || !module_name || !module_source)
        return -1;

    NEURAY::Transaction_impl* transaction_impl
        = static_cast<NEURAY::Transaction_impl*>(transaction);
    DB::Transaction* db_transaction = transaction_impl->get_db_transaction();

    mi::base::Handle<mi::neuraylib::IReader> reader(
        NEURAY::Impexp_utilities::create_reader(module_source, strlen(module_source)));

    MDL::Execution_context default_context;
    return MDL::Mdl_module::create_module(db_transaction, module_name, reader.get(),
        NEURAY::unwrap_and_clear_context(context, default_context));
}

mi::Sint32 Mdl_compiler_impl::deprecated_export_module(
    mi::neuraylib::ITransaction* transaction,
    const char* module_name,
    const char* filename,
    mi::neuraylib::IMdl_execution_context* context)
{
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api>
        mdl_impexp_api(m_neuray_impl->get_api_component<mi::neuraylib::IMdl_impexp_api>());
    if (!mdl_impexp_api)
        return -1;
    return mdl_impexp_api->export_module(transaction, module_name, filename, context);
}

mi::Sint32 Mdl_compiler_impl::deprecated_export_module_to_string(
    mi::neuraylib::ITransaction* transaction,
    const char* module_name,
    mi::IString* exported_module,
    mi::neuraylib::IMdl_execution_context* context)
{
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api>
        mdl_impexp_api(m_neuray_impl->get_api_component<mi::neuraylib::IMdl_impexp_api>());
    if (!mdl_impexp_api)
        return -1;
    return mdl_impexp_api->export_module_to_string(transaction, module_name, exported_module, context);
}

mi::Sint32 Mdl_compiler_impl::deprecated_export_canvas(
    const char* filename, const mi::neuraylib::ICanvas* canvas, mi::Uint32 quality) const
{
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api>
        mdl_impexp_api(m_neuray_impl->get_api_component<mi::neuraylib::IMdl_impexp_api>());
    if (!mdl_impexp_api)
        return -1;
    return mdl_impexp_api->export_canvas(filename, canvas, quality);
}

mi::Sint32 Mdl_compiler_impl::deprecated_export_lightprofile(
    const char* filename, const mi::neuraylib::ILightprofile* lightprofile) const
{
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api>
        mdl_impexp_api(m_neuray_impl->get_api_component<mi::neuraylib::IMdl_impexp_api>());
    if (!mdl_impexp_api)
        return -1;
    return mdl_impexp_api->export_lightprofile(filename, lightprofile);
}

mi::Sint32 Mdl_compiler_impl::deprecated_export_bsdf_data(
    const char* filename,
    const mi::neuraylib::IBsdf_isotropic_data* reflection,
    const mi::neuraylib::IBsdf_isotropic_data* transmission) const
{
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api>
        mdl_impexp_api(m_neuray_impl->get_api_component<mi::neuraylib::IMdl_impexp_api>());
    if (!mdl_impexp_api)
        return -1;
    return mdl_impexp_api->export_bsdf_data(filename, reflection, reflection);
}

static mi::neuraylib::IMdl_backend_api::Mdl_backend_kind translate_backend_kind(
    mi::neuraylib::IMdl_compiler::Mdl_backend_kind kind)
{
    switch (kind) {
    case mi::neuraylib::IMdl_compiler::MB_LLVM_IR:
        return mi::neuraylib::IMdl_backend_api::MB_LLVM_IR;
    case mi::neuraylib::IMdl_compiler::MB_CUDA_PTX:
        return mi::neuraylib::IMdl_backend_api::MB_CUDA_PTX;
    case mi::neuraylib::IMdl_compiler::MB_NATIVE:
        return mi::neuraylib::IMdl_backend_api::MB_NATIVE;
    case mi::neuraylib::IMdl_compiler::MB_HLSL:
        return mi::neuraylib::IMdl_backend_api::MB_HLSL;
    default: break;
    }
    return mi::neuraylib::IMdl_backend_api::MB_FORCE_32_BIT;
}

mi::neuraylib::IMdl_backend* Mdl_compiler_impl::deprecated_get_backend( Mdl_backend_kind kind)
{
    mi::base::Handle<mi::neuraylib::IMdl_backend_api>
        mdl_backend_api(m_neuray_impl->get_api_component<mi::neuraylib::IMdl_backend_api>());
    if (!mdl_backend_api)
        return nullptr;

    return mdl_backend_api->get_backend(translate_backend_kind(kind));
}

const Float32* Mdl_compiler_impl::deprecated_get_df_data_texture(
    mi::neuraylib::Df_data_kind kind,
    Size &rx,
    Size &ry,
    Size &rz) const
{
    mi::base::Handle<mi::neuraylib::IMdl_backend_api>
        mdl_backend_api(m_neuray_impl->get_api_component<mi::neuraylib::IMdl_backend_api>());
    if (!mdl_backend_api)
        return nullptr;

    return mdl_backend_api->get_df_data_texture(kind, rx, ry, rz);
}

mi::Sint32 Mdl_compiler_impl::start()
{
    m_mdlc_module.set();
    m_attr_module.set();

    return 0;
}

mi::Sint32 Mdl_compiler_impl::shutdown()
{
    m_attr_module.reset();
    m_mdlc_module.reset();

    return 0;
}

const mi::IString* Mdl_compiler_impl::deprecated_uvtile_marker_to_string(
    const char* marker,  mi::Sint32 u, mi::Sint32 v) const
{
    if( !marker)
        return nullptr;

    const std::string& result = MDL::uvtile_marker_to_string( marker, u, v);
    return result.empty() ? nullptr : new NEURAY::String_impl( result.c_str());
}

const mi::IString* Mdl_compiler_impl::deprecated_uvtile_string_to_marker(
    const char* str, const char* marker) const
{
   if( !str && !marker)
        return nullptr;

    const std::string& result = MDL::uvtile_string_to_marker( str, marker);
    return result.empty() ? nullptr : new NEURAY::String_impl( result.c_str());
}

} // namespace MDL



} // namespace MI


