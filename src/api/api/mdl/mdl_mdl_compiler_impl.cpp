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
#include "mdl_mdl_backend_impl.h"
#include "mdl_neuray_impl.h"
#include "mdl_mdl_entity_resolver_impl.h"

#include "neuray_impexp_utilities.h"
#include "neuray_lightprofile_impl.h"
#include "neuray_mdl_resource_callback.h"
#include "neuray_module_impl.h"
#include "neuray_string_impl.h"
#include "neuray_transaction_impl.h"
#include "neuray_mdl_execution_context_impl.h"

#include <mi/neuraylib/ibuffer.h>
#include <mi/neuraylib/iexport_result.h>
#include <mi/neuraylib/imap.h>
#include <mi/neuraylib/inumber.h>
#include <mi/mdl/mdl_modules.h>
#include <mi/mdl/mdl_entity_resolver.h>
#include <base/hal/disk/disk_file_reader_writer_impl.h>
#include <base/hal/disk/disk_memory_reader_writer_impl.h>
#include <base/hal/hal/i_hal_ospath.h>
#include <base/lib/log/i_log_logger.h>
#include <base/lib/mem/mem.h>
#include <base/lib/path/i_path.h>
#include <base/lib/plug/i_plug.h>
#include <mdl/integration/mdlnr/i_mdlnr.h>
#include <io/image/image/i_image.h>
#include <io/scene/bsdf_measurement/i_bsdf_measurement.h>
#include <io/scene/lightprofile/i_lightprofile.h>
#include <io/scene/mdl_elements/i_mdl_elements_module.h>
#include <io/scene/mdl_elements/i_mdl_elements_utilities.h>

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

void Mdl_compiler_impl::set_logger( mi::base::ILogger* logger)
{
    m_neuray_impl->set_logger( logger);
}

mi::base::ILogger* Mdl_compiler_impl::get_logger()
{
    return m_neuray_impl->get_logger();
}

mi::Sint32 Mdl_compiler_impl::load_plugin_library( const char* path)
{
    if( !path)
        return -1;

    mi::neuraylib::INeuray::Status status = m_neuray_impl->get_status();
    if( status != mi::neuraylib::INeuray::PRE_STARTING)
        return -1;

    return m_plug_module->load_library( path) ? 0 : -1;
}

mi::Sint32 Mdl_compiler_impl::add_module_path( const char* path)
{
    return !path ? -1 : m_path_module->add_path( PATH::MDL, path);
}

mi::Sint32 Mdl_compiler_impl::remove_module_path( const char* path)
{
    return !path ? -1 : m_path_module->remove_path( PATH::MDL, path);
}

void Mdl_compiler_impl::clear_module_paths()
{
    m_path_module->clear_search_path( PATH::MDL);
}

mi::Size Mdl_compiler_impl::get_module_paths_length() const
{
    return m_path_module->get_path_count( PATH::MDL);
}

const mi::IString* Mdl_compiler_impl::get_module_path( mi::Size index) const
{
    const std::string& result = m_path_module->get_path( PATH::MDL, index);
    if( result.empty())
        return 0;
    mi::IString* istring = new NEURAY::String_impl();
    istring->set_c_str( result.c_str());
    return istring;
}

mi::Sint32 Mdl_compiler_impl::add_resource_path( const char* path)
{
    return !path ? -1 : m_path_module->add_path( PATH::RESOURCE, path);
}

mi::Sint32 Mdl_compiler_impl::remove_resource_path( const char* path)
{
    return !path ? -1 : m_path_module->remove_path( PATH::RESOURCE, path);
}

void Mdl_compiler_impl::clear_resource_paths()
{
    m_path_module->clear_search_path( PATH::RESOURCE);
}

mi::Size Mdl_compiler_impl::get_resource_paths_length() const
{
    return m_path_module->get_path_count( PATH::RESOURCE);
}

const mi::IString* Mdl_compiler_impl::get_resource_path( mi::Size index) const
{
    const std::string& result = m_path_module->get_path( PATH::RESOURCE, index);
    if( result.empty())
        return 0;
    mi::IString* istring = new NEURAY::String_impl();
    istring->set_c_str( result.c_str());
    return istring;
}

mi::Sint32 Mdl_compiler_impl::load_module(
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

const char* Mdl_compiler_impl::get_module_db_name(
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
    return MDL::Mdl_module::get_module_db_name(
        db_transaction, module_name, NEURAY::unwrap_and_clear_context(context, default_context));
}

mi::Sint32 Mdl_compiler_impl::load_module_from_string(
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

mi::Sint32 Mdl_compiler_impl::add_builtin_module(
    const char* module_name, const char* module_source)
{
    if( !module_name || !module_source)
        return -1;

    mi::base::Handle<mi::mdl::IMDL> compiler( m_mdlc_module->get_mdl());
    bool success = compiler->add_builtin_module(
        module_name,
        module_source,
        strlen( module_source),
        /*is_encoded*/ false,
        /*is_native*/ true);
    return success ? 0 : -1;
}

mi::Sint32 Mdl_compiler_impl::export_module(
    mi::neuraylib::ITransaction* transaction,
    const char* module_name,
    const char* filename,
    mi::neuraylib::IMdl_execution_context* context)
{
    if (!transaction || !module_name || !filename)
        return -1;

    DISK::File_writer_impl writer;
    if (!writer.open(filename))
        return -2;
    
    NEURAY::Mdl_execution_context_impl default_context;
    return export_module_common(
        transaction, module_name, &writer, filename,
        context ? context : &default_context);
}

mi::Sint32 Mdl_compiler_impl::export_module_to_string(
    mi::neuraylib::ITransaction* transaction,
    const char* module_name,
    mi::IString* exported_module,
    mi::neuraylib::IMdl_execution_context* context)
{
    if (!transaction || !module_name || !exported_module)
        return -1;

    DISK::Memory_writer_impl writer;
    NEURAY::Mdl_execution_context_impl default_context;
    mi::Sint32 result = export_module_common(
        transaction, module_name, &writer, 0, context ? context : &default_context);

    mi::base::Handle<mi::neuraylib::IBuffer> buffer(writer.get_buffer());
    const mi::Uint8* data = buffer->get_data();
    if (data) {
        std::string s((const char*)data, buffer->get_data_size());
        exported_module->set_c_str(s.c_str());
    }
    else
        exported_module->set_c_str("");

    return result;
}

mi::Sint32 Mdl_compiler_impl::export_canvas(
    const char* filename, const mi::neuraylib::ICanvas* canvas, mi::Uint32 quality) const
{
    if( !filename)
        return -1;

    if( !canvas)
        return -2;

    if( quality > 100)
       return -3;

    SYSTEM::Access_module<IMAGE::Image_module> image_module( false);
    bool result = image_module->export_canvas( canvas, filename, quality);
    return result ? 0 : -4;
}

mi::Sint32 Mdl_compiler_impl::export_lightprofile(
    const char* filename, const mi::neuraylib::ILightprofile* lightprofile) const
{
    if( !filename)
        return -1;

    if( !lightprofile)
        return -2;

    const NEURAY::Lightprofile_impl* lightprofile_impl
        = static_cast<const NEURAY::Lightprofile_impl*>( lightprofile);
    const LIGHTPROFILE::Lightprofile* db_lightprofile = lightprofile_impl->get_db_element();
    bool result = LIGHTPROFILE::export_to_file( db_lightprofile, filename);
    return result ? 0 : -4;
}

mi::Sint32 Mdl_compiler_impl::export_bsdf_data(
    const char* filename,
    const mi::neuraylib::IBsdf_isotropic_data* reflection,
    const mi::neuraylib::IBsdf_isotropic_data* transmission) const
{
    if( !filename)
        return -1;

    bool result = BSDFM::export_to_file( reflection, transmission, filename);
    return result ? 0 : -4;
}

mi::neuraylib::IMdl_backend* Mdl_compiler_impl::get_backend( Mdl_backend_kind kind)
{
    mi::base::Handle<mi::mdl::IMDL> compiler( m_mdlc_module->get_mdl());

    switch( kind) {
        case MB_LLVM_IR:
        case MB_CUDA_PTX:
        case MB_NATIVE:
        case MB_HLSL: {
            mi::base::Handle<mi::mdl::ICode_generator> generator(
                compiler->load_code_generator( "jit"));
            if( !generator)
                return 0;
            mi::base::Handle<mi::mdl::ICode_generator_jit> jit(
                generator->get_interface<mi::mdl::ICode_generator_jit>());
            mi::base::Handle<mi::mdl::ICode_cache> code_cache (m_mdlc_module->get_code_cache());
            return new Mdl_llvm_backend(
                kind,
                compiler.get(),
                jit.get(),
                code_cache.get(),
                /*string_ids=*/true);
        }
        case MB_GLSL:
        case MB_FORCE_32_BIT:
            break;
    }

    return 0;
}

mi::neuraylib::IMdl_entity_resolver* Mdl_compiler_impl::get_entity_resolver() const
{
    return nullptr;
}

void Mdl_compiler_impl::set_external_resolver(mi::mdl::IEntity_resolver *resolver) const
{
}

mi::Sint32 Mdl_compiler_impl::start()
{
    m_mdlc_module.set();
    m_mdlc_module->set_used_with_mdl_sdk( true);
    m_attr_module.set();

    return 0;
}

mi::Sint32 Mdl_compiler_impl::shutdown()
{
    m_attr_module.reset();
    m_mdlc_module.reset();

    return 0;
}

/// Writer output stream.
class Writer_output_stream : public mi::base::Interface_implement<mi::mdl::IOutput_stream>
{
public:

    /// Constructor.
    Writer_output_stream( mi::neuraylib::IWriter* writer) : m_writer( writer), m_error( false) { }

    /// Write a character to the stream.
    virtual void write_char( char c)
    {
        mi::Sint64 out_len = m_writer->write( &c, 1);
        if( 1 != out_len)
            m_error = true;
    }

    /// Write a string to the stream.
    virtual void write( const char* s)
    {
        size_t in_len = strlen( s);
        mi::Sint64 out_len = m_writer->write( s, in_len);
        if( in_len != static_cast<size_t>( out_len))
            m_error = true;
    }

    /// Flush stream.
    virtual void flush()
    {
        if( !m_writer->flush())
            m_error = true;
    }

    /// Remove the last character from output stream if possible.
    ///
    /// \param c  remove this character from the output stream
    ///
    /// \return true if c was the last character in the stream and it was successfully removed,
    /// false otherwise
    virtual bool unput(char c) {
        // unsupported
        return false;
    }

    /// Check error status.
    bool has_error() const { return m_error; }

private:
    mi::neuraylib::IWriter* m_writer;
    bool m_error;
};

mi::Sint32 Mdl_compiler_impl::export_module_common(
    mi::neuraylib::ITransaction* transaction,
    const char* module_name,
    mi::neuraylib::IWriter* writer,
    const char* filename,
    mi::neuraylib::IMdl_execution_context* context)
{
    ASSERT( M_NEURAY_API, module_name);
    ASSERT( M_NEURAY_API, writer);
    ASSERT(M_NEURAY_API, context);

    NEURAY::Mdl_execution_context_impl* context_impl =
        static_cast<NEURAY::Mdl_execution_context_impl*>(context);
    MDL::Execution_context& wrapped_context = context_impl->get_context();
    wrapped_context.clear_messages();

    // get bundle_resources option
    bool bundle_resources = wrapped_context.get_option<bool>(MDL_CTX_OPTION_BUNDLE_RESOURCES);

    // check that the bundle_resources option is not used with string-based exports
    if( !filename && bundle_resources)
        return -6006;

    mi::base::Handle<const mi::neuraylib::IModule> mdl_module(
        transaction->access<mi::neuraylib::IModule>( module_name));
    if( !mdl_module)
        return -6002;

    // check that the MDL module is not a builtin module, their sources are not exported
    const char* old_mdl_module_name = mdl_module->get_mdl_name();
    if( MDL::is_builtin_module( old_mdl_module_name))
        return -6004;

    mi::base::Handle<mi::mdl::IMDL> mdl( m_mdlc_module->get_mdl());

    // check that the module name of the URI is a valid MDL identifier (package names are not
    // checked since we do not know where the search path ends)
    if( filename) {
        std::string path, basename, new_mdl_module_name, ext;
        HAL::Ospath::split( filename, path, basename);
        HAL::Ospath::splitext( basename, new_mdl_module_name, ext);
        if( !MDL::Mdl_module::is_valid_module_name( new_mdl_module_name.c_str(), mdl.get()))
            return -6005;
    }

    // create MDL exporter
    mi::base::Handle<mi::mdl::IMDL_exporter> mdl_exporter( mdl->create_exporter());

    // create the resource callback
    NEURAY::Transaction_impl* transaction_impl
        = static_cast<NEURAY::Transaction_impl*>( transaction);
    DB::Transaction* db_transaction = transaction_impl->get_db_transaction();
    std::string uri
        = filename ? NEURAY::Impexp_utilities::convert_filename_to_uri( filename) : "";
    mi::base::Handle<mi::neuraylib::IExport_result_ext> export_result_ext(
        transaction->create<mi::neuraylib::IExport_result_ext>( "Export_result_ext"));
    NEURAY::Resource_callback resource_callback( db_transaction,
        module_name, !uri.empty() ? uri.c_str() : 0, false, export_result_ext.get());

    // export the MDL module
    Writer_output_stream writer_stream( writer);
    const MI::MDL::Mdl_module* db_mdl_module
        = static_cast<const NEURAY::Module_impl*>( mdl_module.get())->get_db_element();
    mi::base::Handle<const mi::mdl::IModule> mdl_mdl_module( db_mdl_module->get_mdl_module());
    mdl_exporter->export_module( &writer_stream, mdl_mdl_module.get(), &resource_callback);

    if( writer_stream.has_error())
        return -6003;

    return - static_cast<mi::Sint32>( export_result_ext->get_error_number());
}

const mi::IString* Mdl_compiler_impl::uvtile_marker_to_string(
    const char* marker,
    mi::Sint32 u, mi::Sint32 v) const
{
    return NEURAY::Impexp_utilities::uvtile_marker_to_string( marker, u, v);
}

const mi::IString* Mdl_compiler_impl::uvtile_string_to_marker(
    const char* str, const char* marker) const
{
    if( !(str && marker))
        return NULL;
    return NEURAY::Impexp_utilities::uvtile_string_to_marker( str, marker);
}

} // namespace MDL

} // namespace MI

