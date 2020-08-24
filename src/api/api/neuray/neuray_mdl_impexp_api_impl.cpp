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
 ** \brief Source for the IMdl_impexp_api implementation.
 **/

#include "pch.h"

#include "neuray_mdl_impexp_api_impl.h"

#include "neuray_impexp_utilities.h"
#include "neuray_lightprofile_impl.h"
#include "neuray_mdl_execution_context_impl.h"
#include "neuray_mdl_resource_callback.h"
#include "neuray_module_impl.h"
#include "neuray_string_impl.h"
#include "neuray_transaction_impl.h"

#include <base/hal/disk/disk_file_reader_writer_impl.h>
#include <base/hal/disk/disk_memory_reader_writer_impl.h>
#include <base/hal/hal/i_hal_ospath.h>
#include <base/lib/path/i_path.h>
#include <io/image/image/i_image.h>
#include <io/scene/bsdf_measurement/i_bsdf_measurement.h>
#include <io/scene/lightprofile/i_lightprofile.h>
#include <io/scene/mdl_elements/i_mdl_elements_module.h>
#include <io/scene/mdl_elements/i_mdl_elements_utilities.h>
#include <mdl/integration/mdlnr/i_mdlnr.h>

#include <mi/mdl/mdl_mdl.h>
#include <mi/mdl/mdl_modules.h>
#include <mi/neuraylib/ibuffer.h>
#include <mi/neuraylib/iexport_result.h>
#include <mi/neuraylib/imap.h>

namespace MI {
namespace NEURAY {

Mdl_impexp_api_impl::Mdl_impexp_api_impl(mi::neuraylib::INeuray *neuray)
: m_neuray(neuray)
{
}

Mdl_impexp_api_impl::~Mdl_impexp_api_impl()
{
    m_neuray = nullptr;
}

mi::Sint32 Mdl_impexp_api_impl::load_module(
    mi::neuraylib::ITransaction* transaction,
    const char* module_name,
    mi::neuraylib::IMdl_execution_context* context)
{
    if (!transaction || !module_name)
        return -1;

    Transaction_impl* transaction_impl
        = static_cast<Transaction_impl*>(transaction);
    DB::Transaction* db_transaction = transaction_impl->get_db_transaction();

    MDL::Execution_context default_context;
    return MDL::Mdl_module::create_module(
        db_transaction, module_name, unwrap_and_clear_context(context, default_context));
}

mi::Sint32 Mdl_impexp_api_impl::load_module_from_string(
    mi::neuraylib::ITransaction* transaction,
    const char* module_name,
    const char* module_source,
    mi::neuraylib::IMdl_execution_context* context)
{
    if (!transaction || !module_name || !module_source)
        return -1;

    Transaction_impl* transaction_impl
        = static_cast<Transaction_impl*>(transaction);
    DB::Transaction* db_transaction = transaction_impl->get_db_transaction();

    mi::base::Handle<mi::neuraylib::IReader> reader(
        Impexp_utilities::create_reader(module_source, strlen(module_source)));

    MDL::Execution_context default_context;
    return MDL::Mdl_module::create_module(db_transaction, module_name, reader.get(),
        unwrap_and_clear_context(context, default_context));
}

mi::Sint32 Mdl_impexp_api_impl::export_module(
    mi::neuraylib::ITransaction* transaction,
    const char* module_name,
    const char* filename,
    mi::neuraylib::IMdl_execution_context* context)
{
    MDL::Execution_context default_context;
    MDL::Execution_context* context_impl = unwrap_and_clear_context( context, default_context);

    if( !transaction || !module_name || !filename)
        return -1;

    mi::base::Handle<const mi::neuraylib::IModule> module(
        transaction->access<mi::neuraylib::IModule>( module_name));
    if( !module)
        return -6002;

    const MDL::Mdl_module* db_module
        = static_cast<const Module_impl*>( module.get())->get_db_element();
    mi::base::Handle<const mi::mdl::IModule> mdl_module( db_module->get_mdl_module());

    DISK::File_writer_impl writer;
    if( !writer.open( filename))
        return -2;

    return export_module_common(
        transaction, module_name, mdl_module.get(), &writer, filename, context_impl);
}

mi::Sint32 Mdl_impexp_api_impl::export_module_to_string(
    mi::neuraylib::ITransaction* transaction,
    const char* module_name,
    mi::IString* exported_module,
    mi::neuraylib::IMdl_execution_context* context)
{
    MDL::Execution_context default_context;
    MDL::Execution_context* context_impl = unwrap_and_clear_context( context, default_context);

    if( !transaction || !module_name || !exported_module)
        return -1;

    mi::base::Handle<const mi::neuraylib::IModule> module(
        transaction->access<mi::neuraylib::IModule>( module_name));
    if( !module)
        return -6002;

    const MDL::Mdl_module* db_module
        = static_cast<const Module_impl*>( module.get())->get_db_element();
    mi::base::Handle<const mi::mdl::IModule> mdl_module( db_module->get_mdl_module());

    DISK::Memory_writer_impl writer;
    mi::Sint32 result = export_module_common(
        transaction, module_name, mdl_module.get(), &writer, /*filename*/ nullptr, context_impl);

    mi::base::Handle<mi::neuraylib::IBuffer> buffer( writer.get_buffer());
    const mi::Uint8* data = buffer->get_data();
    if( data) {
        std::string s( reinterpret_cast<const char*>( data), buffer->get_data_size());
        exported_module->set_c_str( s.c_str());
    } else
        exported_module->set_c_str( "");

    return result;
}

mi::Sint32 Mdl_impexp_api_impl::export_canvas(
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

mi::Sint32 Mdl_impexp_api_impl::export_lightprofile(
    const char* filename, const mi::neuraylib::ILightprofile* lightprofile) const
{
    if( !filename)
        return -1;

    if( !lightprofile)
        return -2;

    const Lightprofile_impl* lightprofile_impl
        = static_cast<const Lightprofile_impl*>( lightprofile);
    const LIGHTPROFILE::Lightprofile* db_lightprofile = lightprofile_impl->get_db_element();
    DB::Transaction* db_transaction = lightprofile_impl->get_db_transaction();
    bool result = LIGHTPROFILE::export_to_file( db_transaction, db_lightprofile, filename);
    return result ? 0 : -4;
}

mi::Sint32 Mdl_impexp_api_impl::export_bsdf_data(
    const char* filename,
    const mi::neuraylib::IBsdf_isotropic_data* reflection,
    const mi::neuraylib::IBsdf_isotropic_data* transmission) const
{
    if( !filename)
        return -1;

    bool result = BSDFM::export_to_file( reflection, transmission, filename);
    return result ? 0 : -4;
}

const mi::IString* Mdl_impexp_api_impl::uvtile_marker_to_string(
    const char* marker, mi::Sint32 u, mi::Sint32 v) const
{
    if( !marker)
        return nullptr;
    
    const std::string& result = MDL::uvtile_marker_to_string( marker, u, v);
    return result.empty() ? nullptr : new String_impl( result.c_str());
}

const mi::IString* Mdl_impexp_api_impl::uvtile_string_to_marker(
    const char* str, const char* marker) const
{
    if( !str && !marker)
        return nullptr;

    const std::string& result = MDL::uvtile_string_to_marker( str, marker);
    return result.empty() ? nullptr : new String_impl( result.c_str());
}

namespace {

/// Writer output stream.
class Writer_output_stream : public mi::base::Interface_implement<mi::mdl::IOutput_stream>
{
public:

    /// Constructor.
    Writer_output_stream(mi::neuraylib::IWriter* writer) : m_writer(writer), m_error(false) { }

    /// Write a character to the stream.
    virtual void write_char(char c)
    {
        mi::Sint64 out_len = m_writer->write(&c, 1);
        if (1 != out_len)
            m_error = true;
    }

    /// Write a string to the stream.
    virtual void write(const char* s)
    {
        size_t in_len = strlen(s);
        mi::Sint64 out_len = m_writer->write(s, in_len);
        if (in_len != static_cast<size_t>(out_len))
            m_error = true;
    }

    /// Flush stream.
    virtual void flush()
    {
        if (!m_writer->flush())
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

} // end namespace

mi::Sint32 Mdl_impexp_api_impl::export_module_common(
    mi::neuraylib::ITransaction* transaction,
    const char* module_name,
    const mi::mdl::IModule* module,
    mi::neuraylib::IWriter* writer,
    const char* filename,
    MDL::Execution_context* context)
{
    ASSERT(M_NEURAY_API, module);
    ASSERT(M_NEURAY_API, writer);
    ASSERT(M_NEURAY_API, context);

    context->clear_messages();

    // check that the bundle_resources option is not used with string-based exports
    if (!filename && context->get_option<bool>(MDL_CTX_OPTION_BUNDLE_RESOURCES))
        return -6006;

    // check that the MDL module is not a builtin module, their sources are not exported
    const char* old_mdl_module_name = module->get_name();
    if (MDL::is_builtin_module(old_mdl_module_name))
        return -6004;

    // check that the module name of the URI is a valid MDL identifier (package names are not
    // checked since we do not know where the search path ends)
    if (filename) {
        std::string path, basename, new_mdl_module_name, ext;
        HAL::Ospath::split(filename, path, basename);
        HAL::Ospath::splitext(basename, new_mdl_module_name, ext);
        new_mdl_module_name = "::" + new_mdl_module_name;
        if (!MDL::is_valid_module_name(new_mdl_module_name.c_str()))
            return -6005;
    }

    // create MDL exporter
    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module(false);
    mi::base::Handle<mi::mdl::IMDL> mdl(mdlc_module->get_mdl());
    mi::base::Handle<mi::mdl::IMDL_exporter> mdl_exporter(mdl->create_exporter());

    // create the resource callback
    Transaction_impl* transaction_impl
        = static_cast<Transaction_impl*>(transaction);
    DB::Transaction* db_transaction = transaction_impl->get_db_transaction();
    std::string uri
        = filename ? Impexp_utilities::convert_filename_to_uri(filename) : "";
    mi::base::Handle<mi::neuraylib::IExport_result_ext> export_result_ext(
        transaction->create<mi::neuraylib::IExport_result_ext>("Export_result_ext"));
    Resource_callback resource_callback(
        db_transaction,
        module,
        module_name,
        filename,
        context,
        export_result_ext.get());

    // export the MDL module
    Writer_output_stream writer_stream(writer);
    mdl_exporter->export_module(&writer_stream, module, &resource_callback);

    if (writer_stream.has_error())
        return -6003;

    return -static_cast<mi::Sint32>(export_result_ext->get_error_number());
}

mi::Sint32 Mdl_impexp_api_impl::start()
{
    return 0;
}

mi::Sint32 Mdl_impexp_api_impl::shutdown()
{
    return 0;
}

} // namespace NEURAY
} // namespace MI
