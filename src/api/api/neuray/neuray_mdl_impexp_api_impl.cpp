/***************************************************************************************************
 * Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
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
#include "neuray_module_impl.h"
#include "neuray_string_impl.h"
#include "neuray_transaction_impl.h"
#include "neuray_type_impl.h"

#include <boost/algorithm/string/replace.hpp>

#include <base/hal/disk/disk_file_reader_writer_impl.h>
#include <base/hal/disk/disk_memory_reader_writer_impl.h>
#include <base/hal/hal/i_hal_ospath.h>
#include <base/lib/path/i_path.h>
#include <base/system/main/access_module.h>
#include <io/image/image/i_image.h>
#include <io/scene/bsdf_measurement/i_bsdf_measurement.h>
#include <io/scene/lightprofile/i_lightprofile.h>
#include <io/scene/mdl_elements/i_mdl_elements_module.h>
#include <io/scene/mdl_elements/i_mdl_elements_resource_callback.h>
#include <io/scene/mdl_elements/i_mdl_elements_utilities.h>
#include <mdl/integration/mdlnr/i_mdlnr.h>

#include <mi/mdl/mdl_mdl.h>
#include <mi/mdl/mdl_modules.h>
#include <mi/neuraylib/ibuffer.h>
#include <mi/neuraylib/iexport_result.h>
#include <mi/neuraylib/imap.h>

namespace MI {
namespace NEURAY {

Mdl_impexp_api_impl::Mdl_impexp_api_impl( mi::neuraylib::INeuray *neuray)
  : m_neuray( neuray)
{
}

Mdl_impexp_api_impl::~Mdl_impexp_api_impl()
{
    m_neuray = nullptr;
}

mi::Sint32 Mdl_impexp_api_impl::load_module(
    mi::neuraylib::ITransaction* transaction,
    const char* argument,
    mi::neuraylib::IMdl_execution_context* context)
{
    if( !transaction || !argument)
        return -1;

    auto* transaction_impl
        = static_cast<Transaction_impl*>( transaction);
    DB::Transaction* db_transaction = transaction_impl->get_db_transaction();

    MDL::Execution_context default_context;
    MDL::Execution_context* context_impl = unwrap_and_clear_context( context, default_context);
    return MDL::Mdl_module::create_module( db_transaction, argument, context_impl);
}

mi::Sint32 Mdl_impexp_api_impl::load_module_from_string(
    mi::neuraylib::ITransaction* transaction,
    const char* module_name,
    const char* module_source,
    mi::neuraylib::IMdl_execution_context* context)
{
    if( !transaction || !module_name || !module_source)
        return -1;

    auto* transaction_impl
        = static_cast<Transaction_impl*>( transaction);
    DB::Transaction* db_transaction = transaction_impl->get_db_transaction();

    mi::base::Handle<mi::neuraylib::IReader> reader(
        Impexp_utilities::create_reader( module_source, strlen( module_source)));

    MDL::Execution_context default_context;
    MDL::Execution_context* context_impl = unwrap_and_clear_context( context, default_context);
    return MDL::Mdl_module::create_module(
        db_transaction, module_name, reader.get(), context_impl);
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
    mi::base::Handle<const mi::mdl::IModule> core_module( db_module->get_core_module());

    DISK::File_writer_impl writer;
    if( !writer.open( filename))
        return -2;

    return export_module_common(
        transaction, module_name, core_module.get(), &writer, filename, context_impl);
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
    mi::base::Handle<const mi::mdl::IModule> core_module( db_module->get_core_module());

    DISK::Memory_writer_impl writer;
    mi::Sint32 result = export_module_common(
        transaction, module_name, core_module.get(), &writer, /*filename*/ nullptr, context_impl);

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
    const char* filename,
    const mi::neuraylib::ICanvas* canvas,
    const mi::IMap* export_options) const
{
    if( !filename)
        return -1;

    if( !canvas)
        return -2;

    SYSTEM::Access_module<IMAGE::Image_module> image_module( false);
    bool result = image_module->export_canvas( canvas, filename, export_options);
    return result ? 0 : -4;
}

mi::Sint32 Mdl_impexp_api_impl::deprecated_export_canvas(
    const char* filename,
    const mi::neuraylib::ICanvas* canvas,
    mi::Uint32 quality,
    bool force_default_gamma) const
{
    if( quality > 100)
       return -3;

    SYSTEM::Access_module<IMAGE::Image_module> image_module( false);
    mi::base::Handle<mi::IMap> export_options(
        image_module->convert_legacy_options( quality, force_default_gamma));

    return export_canvas( filename, canvas, export_options.get());
}

mi::Sint32 Mdl_impexp_api_impl::export_lightprofile(
    const char* filename, const mi::neuraylib::ILightprofile* lightprofile) const
{
    if( !filename)
        return -1;

    if( !lightprofile)
        return -2;

    const auto* lightprofile_impl
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

const mi::IString* Mdl_impexp_api_impl::frame_uvtile_marker_to_string(
    const char* marker, mi::Size f, mi::Sint32 u, mi::Sint32 v) const
{
    if( !marker)
        return nullptr;

    const std::string& result = MDL::frame_uvtile_marker_to_string( marker, f, u, v);
    return result.empty() ? nullptr : new String_impl( result.c_str());
}

const mi::IString* Mdl_impexp_api_impl::get_mdl_module_name(
    const char* filename, mi::neuraylib::IMdl_impexp_api::Search_option option) const
{
    if( !filename || strlen( filename) == 0)
        return nullptr;

    std::string basename, extension;

    HAL::Ospath::splitext( filename, basename, extension);
    if( extension != ".mdl" && extension != ".mdr")
        return nullptr;
    if( basename.empty())
        return nullptr;

    std::string result = MDL::get_file_path( basename, option);
    boost::replace_all( result, "/", "::");
    result = MDL::encode_module_name( result);

    return new String_impl( result.c_str());
}

const mi::neuraylib::ISerialized_function_name* Mdl_impexp_api_impl::serialize_function_name(
    const char* definition_name,
    const mi::neuraylib::IType_list* argument_types,
    const mi::neuraylib::IType* return_type,
    mi::neuraylib::IMdle_serialization_callback* mdle_callback,
    mi::neuraylib::IMdl_execution_context* context) const
{
    MDL::Execution_context default_context;
    MDL::Execution_context* context_impl = unwrap_and_clear_context( context, default_context);

    if( !definition_name)
        return nullptr;

    mi::base::Handle<const MDL::IType_list> argument_types_int(
        get_internal_type_list( argument_types));
    mi::base::Handle<const MDL::IType> return_type_int(
        get_internal_type( return_type));
    return MDL::serialize_function_name(
        definition_name,
        argument_types_int.get(),
        return_type_int.get(),
        mdle_callback,
        context_impl);
}

mi::neuraylib::IReader* Mdl_impexp_api_impl::create_reader(
    const mi::neuraylib::IBuffer* buffer) const
{
    return buffer ? new DISK::Memory_reader_impl( buffer) : nullptr;
}

mi::neuraylib::IReader* Mdl_impexp_api_impl::create_reader( const char* filename) const
{
    return filename ? Impexp_utilities::create_reader( filename) : nullptr;
}

mi::neuraylib::IWriter* Mdl_impexp_api_impl::create_writer( const char* filename) const
{
    return filename ? Impexp_utilities::create_writer( filename) : nullptr;
}

namespace {

/// Implementation of mi::neuraylib::IDeserialized_function_name which simply wraps
/// MDL::IDeserialized_function_name.
class Deserialized_function_name_impl : public
    mi::base::Interface_implement<mi::neuraylib::IDeserialized_function_name>
{
public:
    Deserialized_function_name_impl(
        Type_factory* tf, const MDL::IDeserialized_function_name* impl)
      : m_tf( tf, mi::base::DUP_INTERFACE)
      , m_impl( impl, mi::base::DUP_INTERFACE)
    { }

    const char* get_db_name() const final { return m_impl->get_db_name(); }

    const mi::neuraylib::IType_list* get_argument_types() const final
    {
        mi::base::Handle<const MDL::IType_list> result_int( m_impl->get_argument_types());
        return m_tf->create_type_list( result_int.get(), /*owner*/ nullptr);
    }

private:
    mi::base::Handle<Type_factory> m_tf;
    mi::base::Handle<const MDL::IDeserialized_function_name> m_impl;
};

} // namespace

const mi::neuraylib::IDeserialized_function_name* Mdl_impexp_api_impl::deserialize_function_name(
    mi::neuraylib::ITransaction* transaction,
    const char* function_name,
    mi::neuraylib::IMdle_deserialization_callback* mdle_callback,
    mi::neuraylib::IMdl_execution_context* context) const
{
    MDL::Execution_context default_context;
    MDL::Execution_context* context_impl = unwrap_and_clear_context( context, default_context);

    if( !transaction || !function_name)
        return nullptr;

    auto* transaction_impl
        = static_cast<Transaction_impl*>( transaction);
    DB::Transaction* db_transaction = transaction_impl->get_db_transaction();

    mi::base::Handle<const MDL::IDeserialized_function_name> result_int(
         MDL::deserialize_function_name(
             db_transaction, function_name, mdle_callback, context_impl));
    if( !result_int)
        return nullptr;

    mi::base::Handle<Type_factory> tf( transaction_impl->get_type_factory());
    return new Deserialized_function_name_impl( tf.get(), result_int.get());
}

const mi::neuraylib::IDeserialized_function_name* Mdl_impexp_api_impl::deserialize_function_name(
    mi::neuraylib::ITransaction* transaction,
    const char* module_name,
    const char* function_name_without_module_name,
    mi::neuraylib::IMdle_deserialization_callback* mdle_callback,
    mi::neuraylib::IMdl_execution_context* context) const
{
    MDL::Execution_context default_context;
    MDL::Execution_context* context_impl = unwrap_and_clear_context( context, default_context);

    if( !transaction || !module_name || !function_name_without_module_name)
        return nullptr;

    auto* transaction_impl
        = static_cast<Transaction_impl*>( transaction);
    DB::Transaction* db_transaction = transaction_impl->get_db_transaction();

    mi::base::Handle<const MDL::IDeserialized_function_name> result_int(
         MDL::deserialize_function_name( db_transaction,
             module_name, function_name_without_module_name, mdle_callback, context_impl));
    if( !result_int)
        return nullptr;

    mi::base::Handle<Type_factory> tf( transaction_impl->get_type_factory());
    return new Deserialized_function_name_impl( tf.get(), result_int.get());
}

const mi::neuraylib::IDeserialized_module_name* Mdl_impexp_api_impl::deserialize_module_name(
    const char* module_name,
    mi::neuraylib::IMdle_deserialization_callback* mdle_callback,
    mi::neuraylib::IMdl_execution_context* context) const
{
    MDL::Execution_context default_context;
    MDL::Execution_context* context_impl = unwrap_and_clear_context( context, default_context);

    if( !module_name)
        return nullptr;

    return MDL::deserialize_module_name( module_name, mdle_callback, context_impl);
}

mi::Sint32 Mdl_impexp_api_impl::export_module_common(
    mi::neuraylib::ITransaction* transaction,
    const char* module_name,
    const mi::mdl::IModule* module,
    mi::neuraylib::IWriter* writer,
    const char* filename,
    MDL::Execution_context* context)
{
    ASSERT( M_NEURAY_API, module);
    ASSERT( M_NEURAY_API, writer);
    ASSERT( M_NEURAY_API, context);

    // check that the bundle_resources option is not used with string-based exports
    if( !filename && context->get_option<bool>( MDL_CTX_OPTION_BUNDLE_RESOURCES))
        return -6006;

    // check that the MDL module is not a builtin module, their sources are not exported
    const char* old_mdl_module_name = module->get_name();
    if( MDL::is_builtin_module( old_mdl_module_name))
        return -6004;

    // check that the module name of the URI is a valid MDL identifier (package names are not
    // checked since we do not know where the search path ends)
    if( filename) {
        std::string path, basename, new_mdl_module_name, ext;
        HAL::Ospath::split( filename, path, basename);
        HAL::Ospath::splitext( basename, new_mdl_module_name, ext);
        new_mdl_module_name = "::" + new_mdl_module_name;
        if( !MDL::is_valid_module_name( new_mdl_module_name))
            return -6005;
    }

    // create MDL exporter
    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module( false);
    mi::base::Handle<mi::mdl::IMDL> mdl( mdlc_module->get_mdl());
    mi::base::Handle<mi::mdl::IMDL_exporter> mdl_exporter( mdl->create_exporter());

    // create the resource callback
    auto* transaction_impl
        = static_cast<Transaction_impl*>( transaction);
    DB::Transaction* db_transaction = transaction_impl->get_db_transaction();
    std::string uri
        = filename ? Impexp_utilities::convert_filename_to_uri( filename) : "";
    mi::base::Handle<mi::neuraylib::IExport_result_ext> export_result_ext(
        transaction->create<mi::neuraylib::IExport_result_ext>( "Export_result_ext"));
    MDL::Resource_callback resource_callback(
        db_transaction,
        module,
        module_name,
        filename,
        context,
        export_result_ext.get());

    // export the MDL module
    mi::base::Handle<MDL::IOutput_stream> output_stream( MDL::get_output_stream( writer));
    mdl_exporter->export_module( output_stream.get(), module, &resource_callback);
    if( output_stream->has_error())
        return -6003;

    return -static_cast<mi::Sint32>( export_result_ext->get_error_number());
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
