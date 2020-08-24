/***************************************************************************************************
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Source for the IMdl_module_transformer implementation.
 **/

#include "pch.h"

#include "neuray_mdl_module_transformer_impl.h"

#include <mi/mdl/mdl_modules.h>
#include <mi/neuraylib/ibuffer.h>
#include <mi/neuraylib/istring.h>

#include <base/hal/disk/disk_file_reader_writer_impl.h>
#include <base/hal/disk/disk_memory_reader_writer_impl.h>
#include <io/scene/mdl_elements/i_mdl_elements_module_transformer.h>
#include <io/scene/mdl_elements/i_mdl_elements_utilities.h>

#include "neuray_mdl_execution_context_impl.h"
#include "neuray_mdl_impexp_api_impl.h"
#include "neuray_transaction_impl.h"

namespace MI {

namespace NEURAY {

Mdl_module_transformer_impl::Mdl_module_transformer_impl(
    mi::neuraylib::ITransaction* transaction,
    const char* module_name,
    const mi::mdl::IModule* mdl_module)
  : m_transaction( transaction, mi::base::DUP_INTERFACE),
    m_db_module_name( module_name)
{
    Transaction_impl* transaction_impl
        = static_cast<Transaction_impl*>( transaction);
    DB::Transaction* db_transaction = transaction_impl->get_db_transaction();

    m_impl.reset( new MDL::Mdl_module_transformer( db_transaction, mdl_module));
}

Mdl_module_transformer_impl::~Mdl_module_transformer_impl()
{
}

mi::Sint32 Mdl_module_transformer_impl::upgrade_mdl_version(
    mi::neuraylib::Mdl_version to_version, mi::neuraylib::IMdl_execution_context* context)
{
    MDL::Execution_context default_context;
    MDL::Execution_context* context_impl = unwrap_and_clear_context( context, default_context);

    return m_impl->upgrade_mdl_version( to_version, context_impl);
}

mi::Sint32 Mdl_module_transformer_impl::use_absolute_import_declarations(
    const char* include_filter,
    const char* exclude_filter,
    mi::neuraylib::IMdl_execution_context* context)
{
    MDL::Execution_context default_context;
    MDL::Execution_context* context_impl = unwrap_and_clear_context( context, default_context);

    return m_impl->use_absolute_import_declarations( include_filter, exclude_filter, context_impl);
}

mi::Sint32 Mdl_module_transformer_impl::use_relative_import_declarations(
    const char* include_filter,
    const char* exclude_filter,
    mi::neuraylib::IMdl_execution_context* context)
{
    MDL::Execution_context default_context;
    MDL::Execution_context* context_impl = unwrap_and_clear_context( context, default_context);

    return m_impl->use_relative_import_declarations( include_filter, exclude_filter, context_impl);
}

mi::Sint32 Mdl_module_transformer_impl::use_absolute_resource_file_paths(
    const char* include_filter,
    const char* exclude_filter,
    mi::neuraylib::IMdl_execution_context* context)
{
    MDL::Execution_context default_context;
    MDL::Execution_context* context_impl = unwrap_and_clear_context( context, default_context);

    return m_impl->use_absolute_resource_file_paths( include_filter, exclude_filter, context_impl);
}

mi::Sint32 Mdl_module_transformer_impl::use_relative_resource_file_paths(
    const char* include_filter,
    const char* exclude_filter,
    mi::neuraylib::IMdl_execution_context* context)
{
    MDL::Execution_context default_context;
    MDL::Execution_context* context_impl = unwrap_and_clear_context( context, default_context);

    return m_impl->use_relative_resource_file_paths( include_filter, exclude_filter, context_impl);
}

mi::Sint32 Mdl_module_transformer_impl::inline_imported_modules(
    const char* include_filter,
    const char* exclude_filter,
    bool omit_anno_origin,
    mi::neuraylib::IMdl_execution_context* context)
{
    MDL::Execution_context default_context;
    MDL::Execution_context* context_impl = unwrap_and_clear_context( context, default_context);

    return m_impl->inline_imported_modules( include_filter, exclude_filter, omit_anno_origin, context_impl);
}

mi::Sint32 Mdl_module_transformer_impl::export_module(
   const char* filename, mi::neuraylib::IMdl_execution_context* context)
{
    MDL::Execution_context default_context;
    MDL::Execution_context* context_impl = unwrap_and_clear_context( context, default_context);

    if( !filename)
        return add_error_message( context_impl, "Invalid parameters (NULL pointer).", -1);

    if( !m_impl->is_module_valid( context_impl))
        return -1;

    mi::base::Handle<const mi::mdl::IModule> module( m_impl->get_module());

    DISK::File_writer_impl writer;
    if( !writer.open( filename))
        return add_error_message( context_impl,
            std::string( "Failed to open \"") + filename + "\" for write operations.", -2);

    return Mdl_impexp_api_impl::export_module_common(
        m_transaction.get(),
        m_db_module_name.c_str(),
        module.get(),
        &writer,
        filename,
        context_impl);
}

mi::Sint32 Mdl_module_transformer_impl::export_module_to_string(
    mi::IString* exported_module, mi::neuraylib::IMdl_execution_context* context)
{
    MDL::Execution_context default_context;
    MDL::Execution_context* context_impl = unwrap_and_clear_context( context, default_context);

    if( !exported_module)
        return add_error_message( context_impl, "Invalid parameters (NULL pointer).", -1);

    if( !m_impl->is_module_valid( context_impl))
        return -1;

    mi::base::Handle<const mi::mdl::IModule> module( m_impl->get_module());
    DISK::Memory_writer_impl writer;

    mi::Sint32 result = Mdl_impexp_api_impl::export_module_common(
        m_transaction.get(),
        m_db_module_name.c_str(),
        module.get(),
        &writer,
        /*filename*/ nullptr,
        context_impl);

    mi::base::Handle<mi::neuraylib::IBuffer> buffer( writer.get_buffer());
    const mi::Uint8* data = buffer->get_data();
    if( data) {
        std::string s( reinterpret_cast<const char*>( data), buffer->get_data_size());
        exported_module->set_c_str( s.c_str());
    } else
        exported_module->set_c_str( "");

    return result;
}

} // namespace NEURAY

} // namespace MI
