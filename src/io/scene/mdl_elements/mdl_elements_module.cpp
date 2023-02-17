/***************************************************************************************************
 * Copyright (c) 2012-2023, NVIDIA CORPORATION. All rights reserved.
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

#include "i_mdl_elements_module.h"

#include "i_mdl_elements_annotation_definition_proxy.h"
#include "i_mdl_elements_compiled_material.h"
#include "i_mdl_elements_function_call.h"
#include "i_mdl_elements_function_definition.h"
#include "i_mdl_elements_module_builder.h"
#include "i_mdl_elements_utilities.h"

#include "mdl_elements_ast_builder.h"
#include "mdl_elements_detail.h"
#include "mdl_elements_expression.h"
#include "mdl_elements_utilities.h"

#include <mutex>
#include <sstream>
#include <thread>

#include <mi/mdl/mdl_code_generators.h>
#include <mi/mdl/mdl_generated_dag.h>
#include <mi/mdl/mdl_mdl.h>
#include <mi/mdl/mdl_modules.h>
#include <mi/mdl/mdl_definitions.h>
#include <mi/mdl/mdl_thread_context.h>
#include <mi/neuraylib/istring.h>
#include <base/system/main/access_module.h>
#include <base/util/string_utils/i_string_utils.h>
#include <boost/core/ignore_unused.hpp>
#include <base/lib/log/i_log_logger.h>
#include <base/data/db/i_db_access.h>
#include <base/data/db/i_db_transaction.h>
#include <base/data/serial/i_serial_buffer_serializer.h>
#include <base/data/serial/i_serializer.h>
#include <base/hal/disk/disk.h>
#include <base/hal/hal/i_hal_ospath.h>
#include <mdl/integration/mdlnr/i_mdlnr.h>
#include <mdl/compiler/compilercore/compilercore_comparator.h>
#include <mdl/compiler/compilercore/compilercore_def_table.h>
#include <mdl/compiler/compilercore/compilercore_modules.h>
#include <mdl/compiler/compilercore/compilercore_tools.h>
#include <io/scene/scene/i_scene_journal_types.h>
#include <io/scene/bsdf_measurement/i_bsdf_measurement.h>
#include <io/scene/lightprofile/i_lightprofile.h>
#include <io/scene/texture/i_texture.h>

namespace MI {

namespace MDL {

namespace {

/// Helper class for dropping module imports at destruction.
class Drop_import_scope
{
public:
    Drop_import_scope( const mi::mdl::IModule* module)
      : m_module( module, mi::base::DUP_INTERFACE)
    {
    }

    ~Drop_import_scope() { m_module->drop_import_entries(); }

private:
    mi::base::Handle<const mi::mdl::IModule> m_module;
};

class Module_loaded_callback : public mi::mdl::IModule_loaded_callback
{
public:
    using Register_internal_func = Sint32 (*)(
        DB::Transaction*,
        mi::mdl::IMDL*,
        const mi::mdl::IModule*,
        Execution_context*,
        Mdl_tag_ident*);

    Module_loaded_callback(
        Register_internal_func func,
        DB::Transaction* t,
        mi::mdl::IMDL* mdl,
        Module_cache* cache,
        Execution_context* c)
        : m_register_internal(func)
        , m_transaction(t)
        , m_mdl(mdl)
        , m_cache(cache)
        , m_context(c)
    {
    }

    bool register_module(
        const mi::mdl::IModule* module) override
    {
        const char* core_name = module->get_name();

        // special handling for built-in modules
        // they are loaded upfront and single-threaded
        if (m_mdl->is_builtin_module(core_name))
        {
            // no notify call here, just registering
            int res = int(m_register_internal(m_transaction, m_mdl, module, m_context, nullptr));
            if (res < 0)
            {
                return int(add_error_message(m_context,
                           STRING::formatted_string(
                           "Failed to register built-in module \"%s\" in DB.",
                           core_name), res));
            }

            m_registered_builtins.insert(core_name);
            return true;
        }

        // processing thread?
        std::string mdl_name = encode_module_name(core_name);
        std::string db_name = get_db_name(mdl_name);
        {
            std::unique_lock<std::mutex> lock(DETAIL::g_transaction_mutex);
            if (m_transaction->name_to_tag(db_name.c_str()).is_invalid() &&
                !m_cache->loading_process_started_in_current_context(core_name))
            {
                add_error_message(m_context,
                    STRING::formatted_string(
                        "Tried to register module in DB on wrong thread: %s\n", core_name), -99);
                        // TODO new error number
                return false;
            }
        }

        ASSERT(M_SCENE, module->is_valid() && "The module to register is invalid");

        // add the module and its content to the database
        int res = int(m_register_internal(m_transaction, m_mdl, module, m_context, nullptr));

        // inform the waiting threads in case of success and case of failure
        m_cache->notify(core_name, res);
        return res >= 0;
    }

    void module_loading_failed(const mi::mdl::IModule_cache_lookup_handle& handle) override
    {
        ASSERT(M_SCENE, m_cache->loading_process_started_in_current_context(
            handle.get_lookup_name()) && "The module loading started on a different context.");

        // inform the waiting threads in case of failure
        m_cache->notify(handle.get_lookup_name(), -1);
    }

    /// Called while loading a module to check if the built-in modules are already registered.
    bool is_builtin_module_registered(const char* absname) const override
    {
        std::unique_lock<std::mutex> lock(DETAIL::g_transaction_mutex);

        // fast check (no db access for recursive loads)
        if (m_registered_builtins.find(absname) != m_registered_builtins.end())
            return true;

        // correct test, executed only once per cache instance and standard module
        std::string db_name = get_db_name(absname);
        return m_transaction->name_to_tag(db_name.c_str()).is_valid();
    }

private:
    Register_internal_func m_register_internal;
    DB::Transaction* m_transaction;
    mi::mdl::IMDL* m_mdl;
    Module_cache* m_cache;
    Execution_context* m_context;
    std::set<std::string> m_registered_builtins;
};

}  // anonymous

const char* Mdl_module::deprecated_get_module_db_name(
    DB::Transaction* transaction,
    const char* argument,
    Execution_context* context)
{
    bool mdle_module = is_mdle(argument);
    std::string mdl_module_name = get_mdl_name_from_load_module_arg(argument, mdle_module);
    if( mdl_module_name.empty()) {
        add_error_message( context,
            STRING::formatted_string( "Invalid module name \"%s\".", argument), -1);
        return nullptr;
    }

    std::string db_module_name = get_db_name(mdl_module_name);

    // Check whether the module exists in the DB.
    DB::Tag db_module_tag = transaction->name_to_tag(db_module_name.c_str());
    if (!db_module_tag)
    {
        // don't consider this an error. This allows to check if a module is loaded.
        return nullptr;
    }

    // Check if the DB entry is a module
    if (transaction->get_class_id(db_module_tag) != Mdl_module::id)
    {
        LOG::mod_log->error(M_SCENE, LOG::Mod_log::C_DATABASE,
            "DB name for module \"%s\" (\"%s\") already in use.",
            mdl_module_name.c_str(), db_module_name.c_str());
        return nullptr;
    }

    // get the name (look up name again instead of using db_module_name to obtain a valid pointer)
    return transaction->tag_to_name(db_module_tag);
}

mi::Sint32 Mdl_module::create_module(
    DB::Transaction* transaction,
    const char* argument,
    Execution_context* context)
{
    ASSERT( M_SCENE, argument);
    ASSERT( M_SCENE, context);

    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module( false);
    mi::base::Handle<mi::mdl::IMDL> mdl( mdlc_module->get_mdl());

    bool mdle_module = is_mdle( argument);

    // Sanity check for non-MDLE module names.
    if( !mdle_module && !starts_with_scope( argument)) {

        if( MDL::get_encoded_names_enabled()) {
            return add_error_message( context, STRING::formatted_string(
                "Invalid module name \"%s\" (does not start with \"::\" and is not an MDLE file "
                "path).", decode_for_error_msg( argument).c_str()), -1);
        } else {
            add_warning_message( context, STRING::formatted_string(
                "The module name \"%s\" does not start with \"::\" (and is not an MDLE file path). "
                "This is deprecated, please add leading double colons.",
                decode_for_error_msg( argument).c_str()));
        }

    }

    // Compute MDL module name
    std::string mdl_module_name = get_mdl_name_from_load_module_arg( argument, mdle_module);
    if( mdl_module_name.empty())
        return add_error_message( context,
            STRING::formatted_string( "Invalid module name \"%s\".", argument), -1);

    // Compute mi::mdl::IMDL::load_module() argument
    std::string core_load_module_arg = decode_module_name( mdl_module_name);
    if( mdle_module) {
        core_load_module_arg = core_load_module_arg.substr( 2);
        core_load_module_arg = remove_slash_in_front_of_drive_letter( core_load_module_arg);
    }

    if( core_load_module_arg.empty())
        return add_error_message( context,
            STRING::formatted_string( "Invalid module name \"%s\".", argument), -1);

    if( MDL::get_encoded_names_enabled()
        && (   (core_load_module_arg.find( '(') != std::string::npos)
            || (core_load_module_arg.find( ')') != std::string::npos)
            || (core_load_module_arg.find( ',') != std::string::npos)))
        return add_error_message( context,
            STRING::formatted_string( "Invalid module name \"%s\" containing parentheses or "
                "commas.", argument), -1);

    std::string db_module_name = get_db_name( mdl_module_name);

    LOG::mod_log->debug( M_SCENE, LOG::Mod_log::C_DATABASE,
        "Module (load_module() argument): \"%s\"", argument);
    LOG::mod_log->debug( M_SCENE, LOG::Mod_log::C_DATABASE,
        "Module (MDL name): \"%s\"", mdl_module_name.c_str());
    LOG::mod_log->debug( M_SCENE, LOG::Mod_log::C_DATABASE,
        "Module (DB name): \"%s\"", db_module_name.c_str());
    LOG::mod_log->debug( M_SCENE, LOG::Mod_log::C_DATABASE,
        "Module (core load_module() argument): \"%s\"", core_load_module_arg.c_str());

    if( mdle_module) {

        // check if the file exists
        if( !DISK::is_file( core_load_module_arg.c_str()))
            return add_error_message( context,
                STRING::formatted_string( "MDLE file \"%s\" does not exist.",
                    core_load_module_arg.c_str()), -1);

    // otherwise we check for valid mdl identifiers
    } else {

        // Reject invalid module names (in particular, names containing slashes and backslashes).
        if ( !is_valid_module_name( core_load_module_arg))
            return add_error_message( context,
                STRING::formatted_string( "The name \"%s\" is not a valid module name",
                    core_load_module_arg.c_str()), -1);
    }

    // Check whether the module exists already in the DB.
    DB::Tag db_module_tag = transaction->name_to_tag( db_module_name.c_str());
    if( db_module_tag) {
        if( transaction->get_class_id( db_module_tag) != Mdl_module::id)
            return add_error_message( context,
                STRING::formatted_string( "DB name for module \"%s\" already in use.",
                    core_load_module_arg.c_str()), -3);
        return 1;
    }

    Mdl_module_wait_queue* wait_queue = mdlc_module->get_module_wait_queue();
    Module_cache module_cache( transaction, wait_queue, {});

    // Set a custom wait handle factory if specified in the context
    mi::base::Handle<const mi::neuraylib::IMdl_loading_wait_handle_factory> factory(
        context->get_interface_option<const mi::neuraylib::IMdl_loading_wait_handle_factory>(
            MDL_CTX_OPTION_LOADING_WAIT_HANDLE_FACTORY));
    if( factory)
        module_cache.set_wait_handle_factory( factory.get());

    // Register a callback to get notified when modules are loaded
    Module_loaded_callback cb(
        &create_module_internal, transaction, mdl.get(), &module_cache, context);
    module_cache.set_module_loading_callback( &cb);

    mi::base::Handle<mi::mdl::IThread_context> ctx( create_thread_context( mdl.get(), context));

    mi::base::Handle<const mi::mdl::IModule> module(
        mdl->load_module( ctx.get(), core_load_module_arg.c_str(), &module_cache));

    // Report messages even when the module is valid (warnings, notes, ...)
    convert_and_log_messages( ctx->access_messages(), context);

    if( !module.is_valid_interface() || !module->is_valid())
        return -2;

    // Even if module loading itself did not fail, DB registration could have failed.
    return context->get_result();
}

mi::Sint32 Mdl_module::create_module(
    DB::Transaction* transaction,
    const char* module_name,
    mi::neuraylib::IReader* module_source,
    Execution_context* context)
{
    ASSERT( M_SCENE, module_name);
    ASSERT( M_SCENE, module_source);
    ASSERT( M_SCENE, context);

    // MDLEs are not supported by this method, hence there is no need to distinguish between the
    // "argument" and the MDL module name, as in the overload above.

    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module( false);
    mi::base::Handle<mi::mdl::IMDL> mdl( mdlc_module->get_mdl());

    // Reject invalid module names (in particular, names containing slashes and backslashes).
    std::string core_module_name = decode_module_name( module_name);
    if( core_module_name.empty())
        return add_error_message( context,
            STRING::formatted_string( "Invalid module name \"%s\".", module_name), -1);

    if( MDL::get_encoded_names_enabled()
        && (   (core_module_name.find( '(') != std::string::npos)
            || (core_module_name.find( ')') != std::string::npos)
            || (core_module_name.find( ',') != std::string::npos)))
        return add_error_message( context,
            STRING::formatted_string( "Invalid module name \"%s\" containing parentheses or "
                "commas.", module_name), -1);

    if( !is_valid_module_name( core_module_name)
        && strcmp( module_name, get_neuray_module_mdl_name()) != 0)
        return add_error_message( context,
            STRING::formatted_string( "The name \"%s\" is not a valid module name",
                core_module_name.c_str()), -1);

    std::string db_module_name = get_db_name( module_name);
    LOG::mod_log->debug( M_SCENE, LOG::Mod_log::C_DATABASE,
        "Module (load_module_from_string() argument): \"%s\"", module_name);
    LOG::mod_log->debug( M_SCENE, LOG::Mod_log::C_DATABASE,
        "Module (MDL name): \"%s\"", module_name);
    LOG::mod_log->debug( M_SCENE, LOG::Mod_log::C_DATABASE,
        "Module (DB name): \"%s\"", db_module_name.c_str());
    LOG::mod_log->debug( M_SCENE, LOG::Mod_log::C_DATABASE,
        "Module (core load_module_from_stream() argument): \"%s\"", core_module_name.c_str());

    // Check whether the module exists already in the DB.
    DB::Tag db_module_tag = transaction->name_to_tag( db_module_name.c_str());
    if( db_module_tag) {
        if( transaction->get_class_id( db_module_tag) != Mdl_module::id)
            return add_error_message( context,
                STRING::formatted_string( "DB name for module \"%s\" already in use.",
                    core_module_name.c_str()), -3);
        return 1;
    }

    mi::base::Handle<mi::mdl::IInput_stream> module_source_stream(
        get_input_stream( module_source, /*filename*/ ""));
    Module_cache module_cache( transaction, mdlc_module->get_module_wait_queue(), {});

    // Set a custom wait handle factory if specified in the context
    mi::base::Handle<const mi::neuraylib::IMdl_loading_wait_handle_factory> factory(
        context->get_interface_option<const mi::neuraylib::IMdl_loading_wait_handle_factory>(
            MDL_CTX_OPTION_LOADING_WAIT_HANDLE_FACTORY));
    if( factory)
        module_cache.set_wait_handle_factory( factory.get());

    // Register a callback to get notified when modules are loaded
    Module_loaded_callback cb(
        &create_module_internal, transaction, mdl.get(), &module_cache, context);
    module_cache.set_module_loading_callback( &cb);

    mi::base::Handle<mi::mdl::IThread_context> ctx( create_thread_context( mdl.get(), context));

    mi::base::Handle<const mi::mdl::IModule> module( mdl->load_module_from_stream(
        ctx.get(), &module_cache, core_module_name.c_str(), module_source_stream.get()));

    // Report messages even when the module is valid (warnings, notes, ...)
    convert_and_log_messages(ctx->access_messages(), context);

    if( !module.is_valid_interface() || !module->is_valid()) {
        return -2;
     }

    // Even if module loading itself did not fail, DB registration could have failed.
    return context->get_result();
}

mi::Sint32 Mdl_module::deprecated_create_module(
    DB::Transaction* transaction,
    const char* module_name,
    Variant_data* variant_data,
    mi::Size variant_count,
    Execution_context* context)
{
    ASSERT( M_SCENE, module_name);
    ASSERT( M_SCENE, variant_data);
    ASSERT( M_SCENE, context);

    // Reject invalid module names (in particular, names containing slashes and backslashes).
    std::string core_module_name = decode_module_name( module_name);
    if( !is_valid_module_name( core_module_name))
        return -1;

    // Check whether the module exists already in the DB.
    std::string db_module_name = get_db_name( module_name);
    DB::Tag db_module_tag = transaction->name_to_tag( db_module_name.c_str());
    if( db_module_tag) {
        if( transaction->get_class_id( db_module_tag) != Mdl_module::id) {
            LOG::mod_log->error( M_SCENE, LOG::Mod_log::C_DATABASE,
                "DB name for module \"%s\" already in use.", core_module_name.c_str());
            return -3;
        }
        return 1;
    }

    // Prepare annotation blocks for module builder (replace nullptr by empty annotation blocks).
    mi::base::Handle<IExpression_factory> ef( get_expression_factory());
    for( mi::Size i = 0; i < variant_count; ++i) {
        Variant_data& pd = variant_data[i];
        if( !pd.m_annotations)
            pd.m_annotations = ef->create_annotation_block();
    }

    // Create builder
    Mdl_module_builder builder(
        transaction,
        db_module_name.c_str(),
        /*min_mdl_version*/ mi::mdl::IMDL::MDL_VERSION_1_0,
        /*max_mdl_version*/ mi::mdl::IMDL::MDL_LATEST_VERSION,
        /*export_to_db*/ true,
        context);
    if( context->get_error_messages_count() > 0)
        return -1;

    // Add variants to module
    for( mi::Size i = 0; i < variant_count; ++i) {
        const Variant_data& vd = variant_data[i];
        if( builder.add_variant(
            vd.m_variant_name.c_str(),
            vd.m_prototype_tag,
            vd.m_defaults.get(),
            vd.m_annotations.get(),
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            context) < 0)
            return -1;
    }

    return context->get_error_messages_count() > 0 ? -1 : 0;
}

bool Mdl_module::supports_reload() const
{
    if( m_module->is_stdlib() || m_module->is_builtins())
        return false;

    const char* name = m_module->get_name();
    if( strcmp( name, "::base") == 0)
        return false;


    return true;
}


namespace {

mi::mdl::IGenerated_code_dag* generate_dag(
    DB::Transaction* transaction,
    mi::mdl::IMDL* mdl,
    const mi::mdl::IModule* module,
    Execution_context* context)
{
    // Compile the module.
    mi::base::Handle<mi::mdl::ICode_generator_dag> generator_dag
        = mi::base::make_handle(mdl->load_code_generator("dag"))
        .get_interface<mi::mdl::ICode_generator_dag>();

    mi::mdl::Options& options = generator_dag->access_options();

    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module(false);

    // We support local entity usage inside MDL materials in neuray, but ...
    options.set_option(MDL_CG_DAG_OPTION_NO_LOCAL_FUNC_CALLS, "false");
    /// ... we need entries for those in the DB, hence generate them
    options.set_option(MDL_CG_DAG_OPTION_INCLUDE_LOCAL_ENTITIES, "true");
    // We enable unsafe math optimizations in neuray
    options.set_option(MDL_CG_DAG_OPTION_UNSAFE_MATH_OPTIMIZATIONS, "true");

    const std::string internal_space =
        context->get_option<std::string>(MDL_CTX_OPTION_INTERNAL_SPACE);
    options.set_option(MDL_CG_OPTION_INTERNAL_SPACE, internal_space.c_str());

    // If configured, we expose names of let expressions as temporaries in neuray
    if (mdlc_module->get_expose_names_of_let_expressions())
        options.set_option(MDL_CG_DAG_OPTION_EXPOSE_NAMES_OF_LET_EXPRESSIONS, "true");

    // Enable target material model mode if requested
    if (context->get_option<bool>(MDL_CTX_OPTION_TARGET_MATERIAL_MODEL_MODE)) {
        options.set_option(MDL_CG_DAG_OPTION_TARGET_MATERIAL_MODE, "true");
    }

    {
        std::unique_lock<std::mutex> lock(DETAIL::g_transaction_mutex);
        Module_cache module_cache(transaction, mdlc_module->get_module_wait_queue(), {});
        if (!module->restore_import_entries(&module_cache)) {
            LOG::mod_log->error(M_SCENE, LOG::Mod_log::C_DATABASE,
                "Failed to restore imports of module \"%s\".", module->get_name());
            context->set_result(-4);
            return nullptr;
        }
    }

    Drop_import_scope scope(module);
    mi::base::Handle<mi::mdl::IGenerated_code> code(generator_dag->compile(module));
    if (!code.is_valid_interface()) {
        context->set_result(-2);
        return nullptr;
    }
    const mi::mdl::Messages& code_messages = code->access_messages();
    convert_and_log_messages(code_messages, context);

    // Treat error messages as compilation failures, e.g., "Call to non-exported function '...' is
    // not allowed in this context".
    if (code_messages.get_error_message_count() > 0) {
        context->set_result(-2);
        return nullptr;
    }

    ASSERT(M_SCENE, code->get_kind() == mi::mdl::IGenerated_code::CK_DAG);
    mi::base::Handle<mi::mdl::IGenerated_code_dag> code_dag(
        code->get_interface<mi::mdl::IGenerated_code_dag>());

    if (context->get_option<bool>(MDL_CTX_OPTION_RESOLVE_RESOURCES)) {

        const char* module_filename = module->get_filename();
        if (module_filename[0] == '\0')
            module_filename = nullptr;

        Mdl_call_resolver_ext resolver(transaction, module);
        Resource_updater updater(
            transaction,
            resolver,
            code_dag.get(),
            module_filename,
            module->get_name(),
            context);
        updater.update_resource_literals();
    }

    code_dag->retain();
    return code_dag.get();
}

} // namespace

mi::Sint32 Mdl_module::create_module_internal(
    DB::Transaction* transaction,
    mi::mdl::IMDL* mdl,
    const mi::mdl::IModule* module,
    Execution_context* context,
    Mdl_tag_ident* module_tag_ident)
{
    std::unique_lock<std::mutex> lock( DETAIL::g_transaction_mutex);

    ASSERT( M_SCENE, mdl);
    ASSERT( M_SCENE, module);
    const char* core_module_name = module->get_name();
    ASSERT( M_SCENE, core_module_name);
    const char* module_filename = module->get_filename();
    if( module_filename[0] == '\0')
        module_filename = nullptr;
    ASSERT( M_SCENE, !mdl->is_builtin_module( core_module_name) || !module_filename);
    ASSERT( M_SCENE, context);

    convert_and_log_messages( module->access_messages(), context);
    if( !module->is_valid())
        return -2;

    // Check whether the module exists already in the DB.
    std::string mdl_module_name = encode_module_name( core_module_name);
    std::string db_module_name = get_db_name( mdl_module_name);
    DB::Tag db_module_tag = transaction->name_to_tag( db_module_name.c_str());
    if( db_module_tag) {
        if( transaction->get_class_id( db_module_tag) != Mdl_module::id) {
            return add_error_message( context,
                STRING::formatted_string( "DB name for module \"%s\" already in use.",
                    core_module_name), -3);
        }
        if( module_tag_ident) {
            DB::Access<Mdl_module> m( db_module_tag, transaction);
            module_tag_ident->first  = db_module_tag;
            module_tag_ident->second = m->get_ident();
        }
        return 1;
    }

    lock.unlock();

    // Compile the module.
    mi::base::Handle<mi::mdl::IGenerated_code_dag> code_dag(
        generate_dag( transaction, mdl, module, context));
    if( context->get_result() != 0)
        return context->get_result();

    lock.lock();

    // Collect tags of imported modules, create DB elements on the fly if necessary.
    mi::Uint32 import_count = module->get_import_count();
    std::vector<Mdl_tag_ident> imports;
    imports.reserve( import_count);

    for( mi::Uint32 i = 0; i < import_count; ++i) {
        mi::base::Handle<const mi::mdl::IModule> import( module->get_import( i));
        std::string core_import_name = import->get_name();
        std::string mdl_import_name  = encode_module_name( core_import_name);
        std::string db_import_name   = get_db_name( mdl_import_name);
        Mdl_tag_ident import_ident;
        DB::Tag import_tag = transaction->name_to_tag( db_import_name.c_str());
        if( !import_tag) {
            // The imported module has to exist in the DB.
            return add_error_message( context,
                STRING::formatted_string( "Failed to initialize imported module \"%s\".",
                    core_import_name.c_str()), -4);
        }

        // Sanity-check for the type of the tag.
        if( transaction->get_class_id( import_tag) != Mdl_module::id)
            return -3;

        DB::Access<Mdl_module> import_module( import_tag, transaction);
        imports.emplace_back( import_tag, import_module->get_ident());
    }

    Mdl_ident module_ident = generate_unique_id();

    // Compute DB names of the annotation definitions in this module.
    mi::Size annotation_definition_count = code_dag->get_annotation_count();
    std::vector<std::string> annotation_names;
    annotation_names.reserve( annotation_definition_count);
    std::vector<DB::Tag> annotation_tags;
    annotation_tags.reserve( annotation_definition_count);

    for( mi::Size i = 0; i < annotation_definition_count; ++i) {
        std::string db_annotation_name = get_db_name_annotation_definition(
            MDL::get_mdl_annotation_name( code_dag.get(), i));
        annotation_names.push_back( db_annotation_name);
        DB::Tag annotation_tag = transaction->name_to_tag( db_annotation_name.c_str());
        if( annotation_tag) {
            return add_error_message( context,
                STRING::formatted_string(
                    "DB name for annotation definition \"%s\" already in use.",
                    db_annotation_name.c_str()), -3);
        }
        LOG::mod_log->debug( M_SCENE, LOG::Mod_log::C_DATABASE,
          "    Annotation %llu (DB name): \"%s\"", i, db_annotation_name.c_str());
        annotation_tags.push_back( transaction->reserve_tag());
    }

    // Compute DB names of the function definitions in this module.
    mi::Size function_count = code_dag->get_function_count();
    std::vector<std::string> function_names;
    function_names.reserve( function_count);
    std::vector<Mdl_tag_ident> function_tags;
    function_tags.reserve( function_count);

    for( mi::Size i = 0; i < function_count; ++i) {
        std::string db_function_name
            = get_db_name( MDL::get_mdl_name( code_dag.get(), /*is_material*/ false, i));
        function_names.push_back( db_function_name);
        DB::Tag function_tag = transaction->name_to_tag( db_function_name.c_str());
        if( function_tag) {
            return add_error_message( context,
                STRING::formatted_string(
                    "DB name for function definition \"%s\" already in use.",
                    db_function_name.c_str()), -3);
        }
        LOG::mod_log->debug( M_SCENE, LOG::Mod_log::C_DATABASE,
          "    Function %llu (DB name): \"%s\"", i, db_function_name.c_str());
        function_tags.emplace_back( transaction->reserve_tag(), module_ident);
    }

    // Compute DB names of the material definitions in this module.
    mi::Size material_count = code_dag->get_material_count();
    std::vector<std::string> material_names;
    material_names.reserve( material_count);
    std::vector<Mdl_tag_ident> material_tags;
    material_tags.reserve( material_count);

    for( mi::Size i = 0; i < material_count; ++i) {
        std::string db_material_name
            = get_db_name( MDL::get_mdl_name( code_dag.get(), /*is_material*/ true, i));
        material_names.push_back( db_material_name);
        DB::Tag material_tag = transaction->name_to_tag( db_material_name.c_str());
        if( material_tag) {
            return add_error_message( context,
                STRING::formatted_string(
                    "DB name for material definition \"%s\" already in use.",
                    db_material_name.c_str()), -3);
        }
        LOG::mod_log->debug( M_SCENE, LOG::Mod_log::C_DATABASE,
          "    Material %llu (DB name): \"%s\"", i, db_material_name.c_str());
        material_tags.emplace_back( transaction->reserve_tag(), module_ident);
    }

    if( !mdl->is_builtin_module( core_module_name)) {
        if( !module_filename)
            add_info_message( context,
                STRING::formatted_string( "Loading module \"%s\".", core_module_name));
        else if( is_container_member( module_filename)) {
            const std::string& container_filename = get_container_filename( module_filename);
            add_info_message( context,
                STRING::formatted_string( "Loading module \"%s\" from \"%s\".",
                    core_module_name, container_filename.c_str()));
        } else
            add_info_message( context,
                STRING::formatted_string( "Loading module \"%s\" from \"%s\".",
                    core_module_name, module_filename));
    }

    Mdl_module* db_module = new Mdl_module(
        transaction, module_ident, mdl, module, code_dag.get(),
        imports, function_tags, material_tags, annotation_tags, context);

    DB::Privacy_level privacy_level = transaction->get_scope()->get_level();

    // Create DB elements for the annotation definition proxies in this module.
    for( mi::Size i = 0; i < annotation_definition_count; ++i) {
        Mdl_annotation_definition_proxy* db_annotation = new Mdl_annotation_definition_proxy(
            mdl_module_name.c_str());
        transaction->store_for_reference_counting(
            annotation_tags[i], db_annotation, annotation_names[i].c_str(), privacy_level);
    }

    bool resolve_resources = context->get_option<bool>( MDL_CTX_OPTION_RESOLVE_RESOURCES);

    // Create DB elements for the function definitions in this module.
    for( mi::Size i = 0; i < function_count; ++i) {
        Mdl_function_definition* db_function = new Mdl_function_definition(
            transaction,
            function_tags[i].first,
            module_ident,
            module,
            code_dag.get(),
            /*is_material*/ false,
            i,
            module_filename,
            mdl_module_name.c_str(),
            resolve_resources);
        transaction->store_for_reference_counting(
            function_tags[i].first, db_function, function_names[i].c_str(), privacy_level);
    }

    // Create DB elements for the material definitions in this module.
    for( mi::Size i = 0; i < material_count; ++i) {
        Mdl_function_definition* db_material = new Mdl_function_definition(
            transaction,
            material_tags[i].first,
            module_ident, module,
            code_dag.get(),
            /*is_material*/ true,
            i,
            module_filename,
            mdl_module_name.c_str(),
            resolve_resources);
        transaction->store_for_reference_counting(
            material_tags[i].first, db_material, material_names[i].c_str(), privacy_level);
    }

    // Store the module in the DB.
    db_module_tag = transaction->reserve_tag();
    transaction->store( db_module_tag, db_module, db_module_name.c_str(), privacy_level);

    if( module_tag_ident) {
        module_tag_ident->first  = db_module_tag;
        module_tag_ident->second = module_ident;
    }

    return 0;
}

IValue_texture* Mdl_module::create_texture(
    DB::Transaction* transaction,
    const char* file_path,
    IType_texture::Shape shape,
    mi::Float32 gamma,
    const char* selector,
    bool shared,
    mi::Sint32* errors)
{
    mi::Sint32 dummy_errors = 0;
    if( !errors)
        errors = &dummy_errors;

    if( !transaction || !file_path) {
        *errors = -1;
        return nullptr;
    }

    if( !is_absolute_mdl_file_path( file_path)) {
        *errors = -2;
        return nullptr;
    }

    DB::Tag tag = DETAIL::mdl_texture_to_tag(
        transaction,
        file_path,
        /*module_filename*/ nullptr,
        /*module_name*/ nullptr,
        gamma,
        selector,
        shared,
        /*context*/ nullptr); // TODO expose execution context
    if( !tag) {
        *errors = -3;
        return nullptr;
    }

    *errors = 0;
    mi::base::Handle<IType_factory> tf( get_type_factory());
    mi::base::Handle<IValue_factory> vf( get_value_factory());
    mi::base::Handle<const IType_texture> t( tf->create_texture( shape));
    return vf->create_texture( t.get(), tag);
}

IValue_light_profile* Mdl_module::create_light_profile(
    DB::Transaction* transaction, const char* file_path, bool shared, mi::Sint32* errors)
{
    mi::Sint32 dummy_errors = 0;
    if( !errors)
        errors = &dummy_errors;

    if( !transaction || !file_path) {
        *errors = -1;
        return nullptr;
    }

    if( !is_absolute_mdl_file_path( file_path)) {
        *errors = -2;
        return nullptr;
    }

    DB::Tag tag = DETAIL::mdl_light_profile_to_tag(
        transaction,
        file_path,
        /*module_filename*/ nullptr,
        /*module_name*/ nullptr,
        shared,
        /*context*/ nullptr); // TODO expose execution context
    if( !tag) {
        *errors = -3;
        return nullptr;
    }

    *errors = 0;
    mi::base::Handle<IValue_factory> vf( get_value_factory());
    return vf->create_light_profile( tag);
}

IValue_bsdf_measurement* Mdl_module::create_bsdf_measurement(
    DB::Transaction* transaction, const char* file_path, bool shared, mi::Sint32* errors)
{
    mi::Sint32 dummy_errors = 0;
    if( !errors)
        errors = &dummy_errors;

    if( !transaction || !file_path) {
        *errors = -1;
        return nullptr;
    }

    if( !is_absolute_mdl_file_path( file_path)) {
        *errors = -2;
        return nullptr;
    }

    DB::Tag tag = DETAIL::mdl_bsdf_measurement_to_tag(
        transaction,
        file_path,
        /*module_filename*/ nullptr,
        /*module_name*/ nullptr,
        shared,
        /*context*/ nullptr); // TODO expose execution context
    if( !tag) {
        *errors = -3;
        return nullptr;
    }

    *errors = 0;
    mi::base::Handle<IValue_factory> vf( get_value_factory());
    return vf->create_bsdf_measurement( tag);
}

Mdl_module::Mdl_module()
  : m_ident( 0)
{
    m_tf = get_type_factory();
    m_vf = get_value_factory();
    m_ef = get_expression_factory();
}

Mdl_module::Mdl_module( const Mdl_module& other)
  : SCENE::Scene_element<Mdl_module, ID_MDL_MODULE>( other),
    m_mdl( other.m_mdl),
    m_module( other.m_module),
    m_code_dag( other.m_code_dag),
    m_tf( other.m_tf),
    m_vf( other.m_vf),
    m_ef( other.m_ef),
    m_mdl_name( other.m_mdl_name),
    m_simple_name( other.m_simple_name),
    m_package_component_names( other.m_package_component_names),
    m_file_name( other.m_file_name),
    m_api_file_name( other.m_api_file_name),
    m_ident(other.m_ident),
    m_imports( other.m_imports),
    m_exported_types( other.m_exported_types),
    m_local_types(other.m_local_types),
    m_constants( other.m_constants),
    m_annotations( other.m_annotations),
    m_annotation_definitions( other.m_annotation_definitions),
    m_functions( other.m_functions),
    m_materials( other.m_materials),
    m_annotation_proxies( other.m_annotation_proxies),
    m_resources( other.m_resources),
    m_function_name_to_index( other.m_function_name_to_index),
    m_material_name_to_index( other.m_material_name_to_index),
    m_annotation_name_to_index( other.m_annotation_name_to_index)
{
}

namespace
{
void convert_type_annotations(
    const mi::mdl::IGenerated_code_dag* code_dag,
    mi::Size index,
    Mdl_annotation_block& annotations,
    Mdl_annotation_block_vector& sub_annotations)
{
    mi::Size annotation_count = code_dag->get_type_annotation_count(index);
    annotations.resize(annotation_count);
    for (mi::Size k = 0; k < annotation_count; ++k)
        annotations[k] = code_dag->get_type_annotation(index, k);

    mi::Size member_count = code_dag->get_type_sub_entity_count(index);
    sub_annotations.resize(member_count);
    for (mi::Size j = 0; j < member_count; ++j) {
        annotation_count = code_dag->get_type_sub_entity_annotation_count(index, j);
        sub_annotations[j].resize(annotation_count);
        for (mi::Size k = 0; k < annotation_count; ++k)
            sub_annotations[j][k] = code_dag->get_type_sub_entity_annotation(index, j, k);
    }
}

} // end namespace

void Mdl_module::init_module( DB::Transaction* transaction, Execution_context* context)
{
    ASSERT(M_SCENE, m_code_dag);

    // convert types
    m_exported_types = m_tf->create_type_list();
    m_local_types    = m_tf->create_type_list();

    mi::Size type_count = m_code_dag->get_type_count();
    for (mi::Size i = 0; i < type_count; ++i) {
        const char* name = m_code_dag->get_type_name(i);
        const mi::mdl::IType* type = m_code_dag->get_type(i);

        Mdl_annotation_block annotations;
        Mdl_annotation_block_vector sub_annotations;
        convert_type_annotations(m_code_dag.get(), i, annotations, sub_annotations);

        mi::base::Handle<const IType> type_int(
            mdl_type_to_int_type(m_tf.get(), type, &annotations, &sub_annotations));
        std::string full_name = m_mdl_name + "::" + name;

        if (m_code_dag->is_type_exported(i))
            m_exported_types->add_type(full_name.c_str(), type_int.get());
        else
            m_local_types->add_type(full_name.c_str(), type_int.get());
    }

    bool resolve_resources = context->get_option<bool>( MDL_CTX_OPTION_RESOLVE_RESOURCES);

    Mdl_dag_converter converter(
        m_ef.get(),
        transaction,
        m_code_dag->get_resource_tagger(),
        m_code_dag.get(),
        /*immutable*/ true,
        /*create_direct_calls*/ false,
        m_mdl_name.c_str(),
        /*prototype_tag*/ DB::Tag(),
        resolve_resources,
        /*user_modules_seen*/ nullptr);

    // convert constants
    m_constants = m_vf->create_value_list();

    mi::Size constant_count = m_code_dag->get_constant_count();
    for (mi::Size i = 0; i < constant_count; ++i) {
        const char* name = m_code_dag->get_constant_name(i);
        const mi::mdl::DAG_constant* constant = m_code_dag->get_constant_value(i);
        const mi::mdl::IValue* value = constant->get_value();
        mi::base::Handle<IValue> value_int(
            converter.mdl_value_to_int_value(/*type_int*/ nullptr, value));
        std::string full_name = m_mdl_name + "::" + name;
        m_constants->add_value(full_name.c_str(), value_int.get());
    }

    // convert annotation definitions
    m_annotation_definitions = m_ef->create_annotation_definition_list();

    mi::Size annotation_definition_count = m_code_dag->get_annotation_count();
    for (mi::Size i = 0; i < annotation_definition_count; ++i) {

        const char* core_name = m_code_dag->get_annotation_name( i);
        std::string name = MDL::get_mdl_annotation_name( m_code_dag.get(), i);
        std::string simple_name = encode_name_without_signature(
            m_code_dag->get_simple_annotation_name( i));

        // convert parameters
        mi::base::Handle<IType_list> parameter_types(m_tf->create_type_list());
        mi::base::Handle<IExpression_list> parameter_defaults(m_ef->create_expression_list());
        std::vector<std::string> parameter_type_names;

        for (mi::Size p = 0, n = m_code_dag->get_annotation_parameter_count(i); p < n; ++p) {

            const char* parameter_name = m_code_dag->get_annotation_parameter_name(i, p);

            // convert types
            const mi::mdl::IType* parameter_type
                = m_code_dag->get_annotation_parameter_type(i, p);
            mi::base::Handle<const IType> type_int(
                mdl_type_to_int_type(m_tf.get(), parameter_type));
            parameter_types->add_type(parameter_name, type_int.get());
            parameter_type_names.emplace_back(encode_name_without_signature(
                m_code_dag->get_annotation_parameter_type_name(i, p)).c_str());

            // convert defaults
            const mi::mdl::DAG_node* default_
                = m_code_dag->get_annotation_parameter_default(i, p);
            if (default_) {
                mi::base::Handle<IExpression> default_int(converter.mdl_dag_node_to_int_expr(
                    default_, type_int.get()));
                ASSERT(M_SCENE, default_int);
                parameter_defaults->add_expression(parameter_name, default_int.get());
            }
        }

        // convert annotations
        mi::Size annotation_annotation_count = m_code_dag->get_annotation_annotation_count(i);
        Mdl_annotation_block annotations(annotation_annotation_count);
        for (mi::Size a = 0; a < annotation_annotation_count; ++a) {
            annotations[a] = m_code_dag->get_annotation_annotation(i, a);
        }
        mi::base::Handle<IAnnotation_block> annotations_int(converter.mdl_dag_node_vector_to_int_annotation_block(
            annotations, m_mdl_name.c_str()));

        bool is_exported = m_code_dag->get_annotation_property(i, mi::mdl::IGenerated_code_dag::AP_IS_EXPORTED);

        mi::neuraylib::IAnnotation_definition::Semantics sema = mdl_semantics_to_ext_annotation_semantics(
            m_code_dag->get_annotation_semantics(i));

        // get version information
        mi::neuraylib::Mdl_version since_version;
        mi::neuraylib::Mdl_version removed_version;
        const mi::mdl::Module* module_impl = mi::mdl::impl_cast<mi::mdl::Module>( m_module.get());
        if( !module_impl->is_stdlib() && !module_impl->is_builtins()) {
            since_version   = convert_mdl_version( module_impl->get_mdl_version());
            removed_version = mi::neuraylib::MDL_VERSION_INVALID;
        } else {
            const mi::mdl::Definition* def = mi::mdl::impl_cast<mi::mdl::Definition>( module_impl->find_signature(
                core_name, /*only_exported*/ !m_module->is_builtins(), /*find_function*/ false));
            if( !def) {
                // ::math::const_expr() and ::...::native() have no definition
                since_version   = convert_mdl_version( module_impl->get_mdl_version());
                removed_version = mi::neuraylib::MDL_VERSION_INVALID;
            } else {
                unsigned flags = def->get_version_flags();
                since_version   = convert_mdl_version_uint32( mi::mdl::mdl_since_version( flags));
                removed_version = convert_mdl_version_uint32( mi::mdl::mdl_removed_version( flags));
            }
        }

        mi::base::Handle<IAnnotation_definition> anno_def(
            m_ef->create_annotation_definition(
                name.c_str(),
                m_mdl_name.c_str(),
                simple_name.c_str(),
                parameter_type_names,
                sema,
                is_exported,
                since_version,
                removed_version,
                parameter_types.get(),
                parameter_defaults.get(),
                annotations_int.get()));

        m_annotation_definitions->add_definition(anno_def.get());
    }

    // convert module annotations
    mi::Size annotation_count = m_code_dag->get_module_annotation_count();
    Mdl_annotation_block annotations(annotation_count);
    for (mi::Size i = 0; i < annotation_count; ++i)
        annotations[i] = m_code_dag->get_module_annotation(i);
    m_annotations = converter.mdl_dag_node_vector_to_int_annotation_block(
        annotations, m_mdl_name.c_str());

    // collect refereced resources

    m_resources.clear();

    bool keep_original_resource_file_paths
        = context->get_option<bool>( MDL_CTX_OPTION_KEEP_ORIGINAL_RESOURCE_FILE_PATHS);
    boost::ignore_unused( keep_original_resource_file_paths);

    // Build map from resource URL to AST index.
    std::map<std::string, size_t> resource_url_2_index;
    size_t n_ast_resources = m_module->get_referenced_resources_count();
    for( size_t i = 0; i < n_ast_resources; ++i)
        resource_url_2_index[m_module->get_referenced_resource_url( i)] = i;

    // Copy the DAG resource table (restricted to resources with URL, and URL known to the AST
    // structure), The AST index is used to retried the mi::mdl::IType for the resource. Using
    // resource_url_2_index assumes that no URL is used for different resource types.
    size_t n_dag_resources = m_code_dag->get_resource_tag_map_entries_count();
    for( size_t i = 0; i < n_dag_resources; ++i) {

        // Skip resources without URL.
        const mi::mdl::Resource_tag_tuple* rtt = m_code_dag->get_resource_tag_map_entry( i);
        if( !rtt->m_url) {
            ASSERT( M_SCENE, !"resource without URL");
            continue;
        }

        // Skip resources not in AST. This can happen if
        // - resources are not resolved (due to a context option),
        // - original resource file paths are kept (due to a context option), or
        // - the DAG contains additional resources (via inlining code from imported modules).
        const auto it = resource_url_2_index.find( rtt->m_url);
        if( it == resource_url_2_index.end())
            continue;

        mi::Size ast_index = it->second;
        const mi::mdl::IType* mdl_type = m_module->get_referenced_resource_type( ast_index);
        mi::base::Handle<const IType> int_type(
            mdl_type_to_int_type<IType_resource>( m_tf.get(), mdl_type));
        IType::Kind type_kind = int_type->get_kind();
        mi::base::Handle<const IType_texture> int_type_texture(
            int_type->get_interface<IType_texture>());
        IType_texture::Shape texture_shape
            = int_type_texture ? int_type_texture->get_shape() : IType_texture::TS_2D;

        m_resources.emplace_back( *rtt, type_kind, texture_shape);
    }
}

Mdl_module::Mdl_module(
    DB::Transaction* transaction,
    Mdl_ident module_ident,
    mi::mdl::IMDL* mdl,
    const mi::mdl::IModule* module,
    mi::mdl::IGenerated_code_dag* code_dag,
    const std::vector<Mdl_tag_ident>& imports,
    const std::vector<Mdl_tag_ident>& functions,
    const std::vector<Mdl_tag_ident>& materials,
    const std::vector<DB::Tag>& annotation_proxies,
    Execution_context* context)
  : m_mdl(mdl, mi::base::DUP_INTERFACE),
    m_module(module, mi::base::DUP_INTERFACE),
    m_code_dag(code_dag, mi::base::DUP_INTERFACE),
    m_tf(get_type_factory()),
    m_vf(get_value_factory()),
    m_ef(get_expression_factory()),
    m_ident(module_ident),
    m_imports(imports),
    m_functions(functions),
    m_materials(materials),
    m_annotation_proxies(annotation_proxies)
{
    ASSERT( M_SCENE, mdl);
    ASSERT( M_SCENE, module);
    ASSERT( M_SCENE, module->get_name());
    ASSERT( M_SCENE, module->get_filename());

    m_mdl_name = encode_module_name( module->get_name());

    const mi::mdl::IQualified_name* qname = module->get_qualified_name();
    size_t n = qname->get_component_count();
    ASSERT( M_SCENE, n > 0);
    m_simple_name = encode( qname->get_component( n-1)->get_symbol()->get_name());
    for( mi::Size i = 0; i < n-1; ++i)
         m_package_component_names.emplace_back(
             encode( qname->get_component( int(i))->get_symbol()->get_name()));

    m_file_name = module->get_filename();
    m_api_file_name = is_container_member( m_file_name.c_str())
        ? get_container_filename( m_file_name.c_str()) : m_file_name;

    init_module( transaction, context);

    m_function_name_to_index.clear();
    for( mi::Size i = 0, n = m_functions.size(); i < n; ++i) {
        const std::string& name
            = get_db_name( MDL::get_mdl_name( code_dag, /*is_material*/ false, i));
        m_function_name_to_index.insert( std::make_pair( name, i));
    }

    m_material_name_to_index.clear();
    for( mi::Size i = 0, n = m_materials.size(); i < n; ++i) {
        const std::string& name
            = get_db_name( MDL::get_mdl_name( code_dag, /*is_material*/ true, i));
        m_material_name_to_index.insert( std::make_pair( name, i));
    }

    m_annotation_name_to_index.clear();
    for( mi::Size i = 0, n = m_annotation_proxies.size(); i < n; ++i) {
        const std::string& name
            = get_db_name_annotation_definition( MDL::get_mdl_annotation_name( code_dag, i));
        m_annotation_name_to_index.insert( std::make_pair( name, i));
    }
}

const char* Mdl_module::get_filename() const
{
    return m_file_name.empty() ? nullptr : m_file_name.c_str();
}

const char* Mdl_module::get_api_filename() const
{
    return m_api_file_name.empty() ? nullptr : m_api_file_name.c_str();
}

const char* Mdl_module::get_mdl_name() const
{
    return m_mdl_name.c_str();
}

const char* Mdl_module::get_mdl_simple_name() const
{
    return m_simple_name.c_str();
}

mi::Size Mdl_module::get_mdl_package_component_count() const
{
    return m_package_component_names.size();
}

const char* Mdl_module::get_mdl_package_component_name( mi::Size index) const
{
    if( index >= m_package_component_names.size())
        return nullptr;
    return m_package_component_names[index].c_str();
}

mi::neuraylib::Mdl_version Mdl_module::get_mdl_version() const
{
    const mi::mdl::Module* impl = mi::mdl::impl_cast<mi::mdl::Module>( m_module.get());
    return MDL::convert_mdl_version( impl->get_mdl_version());
}

mi::Size Mdl_module::get_import_count() const
{
    return m_imports.size();
}

DB::Tag Mdl_module::get_import( mi::Size index) const
{
    if( index >= m_imports.size())
        return DB::Tag( 0);
    return m_imports[index].first;
}

const IType_list* Mdl_module::get_types() const
{
    m_exported_types->retain();
    return m_exported_types.get();
}

const IValue_list* Mdl_module::get_constants() const
{
    m_constants->retain();
    return m_constants.get();
}

mi::Size Mdl_module::get_function_count() const
{
    return m_functions.size();
}

DB::Tag Mdl_module::get_function(mi::Size index) const
{
    if( index >= m_functions.size())
        return DB::Tag( 0);
    return m_functions[index].first;
}

const char* Mdl_module::get_function_name(DB::Transaction* transaction, mi::Size index) const
{
    return index >= m_functions.size() ? nullptr : transaction->tag_to_name(m_functions[index].first);
}

mi::Size Mdl_module::get_material_count() const
{
    return m_materials.size();
}

DB::Tag Mdl_module::get_material(mi::Size index) const
{
    if( index >= m_materials.size())
        return DB::Tag( 0);
    return m_materials[index].first;
}

const IAnnotation_block* Mdl_module::get_annotations() const
{
   if( !m_annotations)
        return nullptr;
    m_annotations->retain();
    return m_annotations.get();
}

mi::Size Mdl_module::get_annotation_definition_count() const
{
    ASSERT(M_SCENE, m_annotation_definitions);
    return m_annotation_definitions->get_size();
}

const IAnnotation_definition* Mdl_module::get_annotation_definition(mi::Size index) const
{
    ASSERT(M_SCENE, m_annotation_definitions);
    return m_annotation_definitions->get_definition(index);
}

const IAnnotation_definition* Mdl_module::get_annotation_definition(const char* name) const
{
    ASSERT(M_SCENE, m_annotation_definitions);
    return m_annotation_definitions->get_definition(name);
}

const char* Mdl_module::get_material_name(DB::Transaction* transaction, mi::Size index) const
{
    return index >= m_materials.size() ? nullptr : transaction->tag_to_name(m_materials[index].first);
}

bool Mdl_module::is_standard_module() const
{
    return m_module->is_stdlib();
}

bool Mdl_module::is_mdle_module() const
{
    return m_module->is_mdle();
}

std::vector<std::string> Mdl_module::get_function_overloads(
    DB::Transaction* transaction, const char* name, const IExpression_list* arguments) const
{
    std::vector<std::string> result;
    if( !name)
        return result;

    std::string name_str( name);
    if( !name_str.empty() && name_str.back() == ')') {
        LOG::mod_log->warning( M_NEURAY_API, LOG::Mod_log::C_MISC,
            "Name of function definition \"%s\" passed to mi::neuraylib::IModule::get_function_"
            "overloads() includes the signature. This is deprecated and may fail in the case of "
            "general Unicode names.", name_str.c_str());
        size_t left = name_str.rfind( '(');
        name_str = name_str.substr( 0, left);
    }

    // strip optional module name
    size_t double_colon = name_str.rfind( "::");
    if( double_colon != std::string::npos) {
        // check that the module matches this module
        if( m_mdl_name == get_builtins_module_mdl_name()) {
            if( double_colon != 3)
                return result;
        } else {
            if( name_str.substr( 0, double_colon) != get_db_name( m_mdl_name))
                return result;
        }
        name_str = name_str.substr( double_colon + 2);
    }

    std::vector<Mdl_tag_ident> candidates;
    candidates.insert( candidates.end(), m_functions.begin(), m_functions.end());
    candidates.insert( candidates.end(), m_materials.begin(), m_materials.end());

    // find overloads
    for( mi::Size i = 0; i < candidates.size(); ++i) {

        DB::Tag tag = candidates[i].first;
        ASSERT( M_SCENE, tag && transaction->get_class_id( tag) == Mdl_function_definition::id);
        DB::Access<Mdl_function_definition> definition( tag, transaction);

        std::string simple_name = definition->get_mdl_simple_name();
        if( simple_name != name_str)
            continue;

        // no arguments provided, don't check for exact match
        if( !arguments) {
            const char* fd_name = transaction->tag_to_name( candidates[i].first);
            result.emplace_back( fd_name);
            continue;
        }

        // arguments provided, check for exact match
        mi::Sint32 errors = 0;
        // TODO check whether we can avoid the function call creation
        Mdl_function_call* call = definition->create_function_call(
            transaction, arguments, &errors);
        if( call && errors == 0) {
            const char* fd_name = transaction->tag_to_name( candidates[i].first);
            result.emplace_back( fd_name);
        }
        delete call;
    }

    return result;
}

std::vector<std::string> Mdl_module::get_function_overloads_by_signature(
    const char* name, const std::vector<const char*>& parameter_types) const
{
    std::vector<std::string> result;
    if( !name)
        return result;

    std::string core_name;
    std::string name_str( name);
    size_t double_colon = name_str.rfind( "::");
    if( double_colon == std::string::npos) {
        // Simple name given, add core module name as prefix.
        core_name = m_mdl_name == get_builtins_module_mdl_name()
            ? std::string( "::") + name_str
            : decode_module_name( m_mdl_name) + "::" + name_str;
    } else {
        // DB name given, strip prefix and decode.
        if( !starts_with_mdl_or_mdle( name_str))
            return result;
        core_name = decode_module_name( strip_mdl_or_mdle_prefix( name_str));
    }

    size_t n = parameter_types.size();
    std::vector<std::string> mdl_parameter_types;
    mdl_parameter_types.reserve( n);
    for( const auto& s: parameter_types)
        mdl_parameter_types.push_back( decode_name_without_signature( s));

    std::vector<const char*> mdl_parameter_types_cstr;
    mdl_parameter_types_cstr.reserve( n);
    for( const auto& s: mdl_parameter_types)
        mdl_parameter_types_cstr.push_back( s.c_str());

    mi::base::Handle<const mi::mdl::IOverload_result_set> set( m_module->find_overload_by_signature(
        core_name.c_str(), mdl_parameter_types_cstr.data(), mdl_parameter_types_cstr.size()));
    if( !set.is_valid_interface())
        return result;

    for( const char* s = set->first_signature(); s != nullptr; s = set->next_signature()) {

        // TODO encoded names: workaround to add missing signature for materials
        std::string name = s;
        if( get_encoded_names_enabled() && (name.back() != ')')) {

            std::string prefix = get_db_name( encode_name_without_signature( name)) + '(';
            size_t prefix_len = prefix.size();
            size_t result_size = result.size();
            boost::ignore_unused( result_size);
            for( const auto& material_name: m_material_name_to_index) {
                if( material_name.first.substr( 0, prefix_len) != prefix)
                    continue;
                result.push_back( material_name.first);
                break;
            }
            ASSERT( M_SCENE, result.size() == result_size + 1);

        } else {

            result.push_back( get_db_name( encode_name_with_signature( s)));

        }
    }

    return result;
}

mi::Size Mdl_module::get_resources_count() const
{
    return m_resources.size();
}

const IValue_resource* Mdl_module::get_resource( mi::Size index) const
{
    if( index >= m_resources.size())
        return nullptr;

    const Resource_tag_tuple_ext& rtt = m_resources[index];
    ASSERT( M_SCENE, rtt.m_tag);

    switch( rtt.m_type_kind) {

        case IType::TK_TEXTURE: {
             mi::base::Handle<const IType_texture> type(
                 m_tf->create_texture( rtt.m_texture_shape));
             return m_vf->create_texture( type.get(), rtt.m_tag);
        }

        case IType::TK_LIGHT_PROFILE:
            return m_vf->create_light_profile( rtt.m_tag);

        case IType::TK_BSDF_MEASUREMENT:
            return m_vf->create_bsdf_measurement( rtt.m_tag);

        case IType::TK_ALIAS:
        case IType::TK_BOOL:
        case IType::TK_INT:
        case IType::TK_ENUM:
        case IType::TK_FLOAT:
        case IType::TK_DOUBLE:
        case IType::TK_STRING:
        case IType::TK_VECTOR:
        case IType::TK_MATRIX:
        case IType::TK_COLOR:
        case IType::TK_ARRAY:
        case IType::TK_STRUCT:
        case IType::TK_BSDF:
        case IType::TK_HAIR_BSDF:
        case IType::TK_EDF:
        case IType::TK_VDF:
        case IType::TK_FORCE_32_BIT:
            ASSERT( M_SCENE, false);
            return nullptr;
    }

    ASSERT( M_SCENE, false);
    return nullptr;
}

const Resource_tag_tuple_ext* Mdl_module::get_resource_tag_tuple( mi::Size index) const
{
    if( index >= m_resources.size())
        return nullptr;

    return &m_resources[index];
}

const mi::mdl::IModule* Mdl_module::get_mdl_module() const
{
    m_module->retain();
    return m_module.get();
}

const mi::mdl::IGenerated_code_dag* Mdl_module::get_code_dag() const
{
    if( !m_code_dag.is_valid_interface())
        return nullptr;
    m_code_dag->retain();
    return m_code_dag.get();
}

bool Mdl_module::is_valid(
    DB::Transaction* transaction,
    Execution_context* context) const
{
    if (m_ident == Mdl_ident(-1))
        return false;

    if (is_standard_module())
        return true;
    for (const auto& import : m_imports) {
        DB::Access<Mdl_module> module(import.first, transaction);
        if (module->get_ident() != import.second) {
            std::string message = "The identifier of the imported module '"
                + get_db_name(module->get_mdl_name())
                + "' has changed. Try to reload this module.";
            add_error_message(context, message, -1);
            return false;
        }
        if (!module->is_valid(transaction, context)) {
            std::string message = "The imported module '"
                + get_db_name(module->get_mdl_name())
                + "' is invalid. Try to reload this module recursively.";
            add_error_message(context, message, -1);
            return false;
        }
    }
    return true;
}

mi::Sint32 Mdl_module::reload(
    DB::Transaction* transaction,
    bool recursive,
    Execution_context* context)
{
    ASSERT(M_SCENE, context);

    if (!supports_reload()) {
        add_error_message(
            context, "Standard and built-in modules cannot be reloaded.", -1);
        return -1;
    }
    if (m_file_name.empty()) {
        // nothing to reload from
        add_error_message(
            context, "Cannot reload memory-based module without new source code.", -1);
        return -1;
    }

    std::string db_name = get_db_name(m_mdl_name);
    DB::Tag tag = transaction->name_to_tag(db_name.c_str());

    if (recursive) {
        std::set<DB::Tag> done;
        mi::Sint32 result = reload_imports(transaction, tag, /*top_level*/ true, done, context);
        if (result != 0)
            return result;
    }

    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module(false);
    mi::base::Handle<mi::mdl::IMDL> mdl(mdlc_module->get_mdl());

    mi::base::Handle<mi::mdl::IThread_context> ctx( create_thread_context( mdl.get(), context));

    Module_cache cache( transaction, mdlc_module->get_module_wait_queue(), { tag });

    std::string core_module_name = decode_module_name( m_mdl_name);
    mi::base::Handle<const mi::mdl::IModule> module(
        mdl->load_module(ctx.get(), core_module_name.c_str(), recursive ? nullptr : &cache));

    // report messages even when the module is valid (warnings, notes, ...)
    convert_and_log_messages(ctx->access_messages(), context);

    if (!module.is_valid_interface() || !module->is_valid()) {
        add_error_message(
            context, "The module failed to compile.", -2);
        return -1;
    }

    return reload_module_internal(transaction, mdl.get(), module.get(), context);
}

mi::Sint32 Mdl_module::reload_from_string(
    DB::Transaction* transaction,
    mi::neuraylib::IReader* module_source,
    bool recursive,
    Execution_context* context)
{
    ASSERT(M_SCENE, context);

    if (!supports_reload()) {
        add_error_message(
            context, "Standard and built-in modules cannot be reloaded.", -1);
        return -1;
    }
    if (!m_file_name.empty()) {
        add_error_message(
            context, "File-based modules cannot be replaced by memory-based modules.", -1);
        return -1;
    }

    std::string db_name = get_db_name(m_mdl_name);
    DB::Tag tag = transaction->name_to_tag(db_name.c_str());

    if (recursive) {
        std::set<DB::Tag> done;
        mi::Sint32 result = reload_imports(transaction, tag, /*top_level*/ true, done, context);
        if (result != 0)
            return result;
    }

    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module(false);
    mi::base::Handle<mi::mdl::IMDL> mdl(mdlc_module->get_mdl());

    mi::base::Handle<mi::mdl::IThread_context> ctx( create_thread_context( mdl.get(), context));

    Module_cache cache( transaction, mdlc_module->get_module_wait_queue(), { tag });

    mi::base::Handle<mi::mdl::IInput_stream> module_source_stream(
        get_input_stream( module_source, /*filename*/ ""));

    std::string core_module_name = decode_module_name( m_mdl_name);
    mi::base::Handle<const mi::mdl::IModule> module(mdl->load_module_from_stream(
        ctx.get(), recursive ? nullptr : &cache, core_module_name.c_str(), module_source_stream.get()));

    // report messages even when the module is valid (warnings, notes, ...)
    convert_and_log_messages(ctx->access_messages(), context);

    if (!module.is_valid_interface() || !module->is_valid()) {
        add_error_message(
            context, "The module failed to compile.", -2);
        return -1;
    }

    return reload_module_internal(transaction, mdl.get(), module.get(), context);
}

mi::Sint32 Mdl_module::reload_imports(
    DB::Transaction* transaction,
    DB::Tag module_tag,
    bool top_level,
    std::set<DB::Tag>& done,
    Execution_context* context)
{
    // No need to do anything if this module has already been reloaded via some other path in the
    // DAG.
    if( done.count( module_tag) > 0)
        return 0;

    // Nothing to be done for modules that do not support reloads (we assume here that this also
    // applies to their imports).
    DB::Access<Mdl_module> module( module_tag, transaction);
    if( !module->supports_reload()) {
        done.insert( module_tag);
        return 0;
    }

    // Reload imports
    mi::Size n = module->get_import_count();
    for( mi::Size i = 0; i < n; ++i) {
        mi::Sint32 result = reload_imports(
            transaction, module->get_import( i), /*top_level*/ false, done, context);
        if( result != 0)
            return result;
    }

    // In the top-level case, the module itself is handled in the caller.
    if( top_level)
        return 0;

    // Skip string-based modules which cannot be reloaded.
    if( !module->get_filename()) {
        done.insert( module_tag);
        return 0;
    }

    // Reload the module itself.
    DB::Edit<Mdl_module> edit_module( module);
    mi::Sint32 result = edit_module->reload( transaction, /*recursive*/ false, context);
    done.insert( module_tag);
    return result;
}

namespace {

bool check_user_types(const mi::mdl::IGenerated_code_dag* code_dag)
{
    mi::base::Handle<IType_factory> tf(get_type_factory());
    mi::Size type_count = code_dag->get_type_count();
    for (mi::Size i = 0; i < type_count; ++i) {

        const mi::mdl::IType* type = code_dag->get_type(i);
        type = type->skip_type_alias();
        if (const mi::mdl::IType_struct* ts = mi::mdl::as<mi::mdl::IType_struct>(type)) {

            std::string name = encode_name_without_signature(ts->get_symbol()->get_name());
            mi::base::Handle<const IType> existing_type(tf->create_struct(name.c_str()));
            if (existing_type) {
                Mdl_annotation_block annotations;
                Mdl_annotation_block_vector sub_annotations;
                convert_type_annotations(code_dag, i, annotations, sub_annotations);

                if (!mdl_type_struct_to_int_type_test(tf.get(), ts, &annotations, &sub_annotations))
                    return false;
            }
        }
        else if (const mi::mdl::IType_enum* te = mi::mdl::as<mi::mdl::IType_enum>(type)) {

            std::string name = encode_name_without_signature(te->get_symbol()->get_name());
            mi::base::Handle<const IType> existing_type(tf->create_enum(name.c_str()));
            if (existing_type) {
                Mdl_annotation_block annotations;
                Mdl_annotation_block_vector sub_annotations;
                convert_type_annotations(code_dag, i, annotations, sub_annotations);

                if (!mdl_type_enum_to_int_type_test(tf.get(), te, &annotations, &sub_annotations))
                    return false;
            }
        }
    }
    return true;
}

} // end namespace

mi::Sint32 Mdl_module::reload_module_internal(
    DB::Transaction* transaction,
    mi::mdl::IMDL* mdl,
    const mi::mdl::IModule* module,
    Execution_context* context)
{
    // check imports
    mi::Uint32 import_count = module->get_import_count();
    std::vector<Mdl_tag_ident> imports( import_count);

    for( mi::Uint32 i = 0; i < import_count; ++i) {

        mi::base::Handle<const mi::mdl::IModule> import( module->get_import(i));
        std::string import_mdl_name = encode_module_name( import->get_name());
        std::string import_db_name  = get_db_name( import_mdl_name);
        DB::Tag import_tag = transaction->name_to_tag( import_db_name.c_str());

        if( import_tag) {

            // Sanity-check for the type of the tag.
            if( transaction->get_class_id( import_tag) != Mdl_module::id) {
                add_error_message(
                    context, "DB name for " + import_db_name + " already in use.", -3);
                return -1;
            }
            DB::Access<Mdl_module> import_db_module( import_tag, transaction);
            imports[i].first  = import_tag;
            imports[i].second = import_db_module->get_ident();

        }  else {

            // The imported module does not yet exist in the DB.
            mi::Sint32 result = create_module_internal(
                transaction, mdl, import.get(), context, &imports[i]);
            if( result < 0) {
                add_error_message(
                    context, "Could not import module '" + import_db_name + "'.", -4);
                m_ident = -1;
                return -1;
            }
        }
    }

    // check whether the module changed at all (including the imports)
    //
    // Just checking the module itself is not sufficient, since changes in the imports might
    // affect the DAG representation, but not the AST of this module.
    if( imports == m_imports && mi::mdl::equal( module, m_module.get()))
        return 0;

    mi::base::Handle<mi::mdl::IGenerated_code_dag> code_dag(
        generate_dag( transaction, mdl, module, context));
    if( context->get_result() != 0)
        return context->get_result();

    // check types
    if( !check_user_types( code_dag.get())) {
        add_error_message(
            context, "The module has incompatible type changes and therefore cannot be reloaded.", -2);
        return -1;
    }

    // check function definitions
    mi::Size function_count = code_dag->get_function_count();
    std::vector<std::string> function_names( function_count);

    for( mi::Size i = 0; i < function_count; ++i) {
        std::string db_function_name
            = get_db_name( MDL::get_mdl_name( code_dag.get(), /*is_material*/ false, i));
        function_names[i] = db_function_name;
        DB::Tag function_tag = transaction->name_to_tag( db_function_name.c_str());
        if( function_tag && transaction->get_class_id( function_tag) != ID_MDL_FUNCTION_DEFINITION) {
            std::string msg = "DB name for function definition '" + db_function_name + "' "
                "already in use and not of type ELEMENT_TYPE_FUNCTION_DEFINITION";
            add_error_message( context, msg, -3);
            return -1;
        }
    }

    // check material definitions
    mi::Size material_count = code_dag->get_material_count();
    std::vector<std::string> material_names( material_count);

    for( mi::Size i = 0; i < material_count; ++i) {
        std::string db_material_name
            = get_db_name( MDL::get_mdl_name( code_dag.get(), /*is_material*/ true, i));
        material_names[i] = db_material_name;
        DB::Tag material_tag = transaction->name_to_tag( db_material_name.c_str());
        if( material_tag && transaction->get_class_id( material_tag) != ID_MDL_FUNCTION_DEFINITION) {
            std::string msg = "DB name for material definition '" + db_material_name + "' "
                "already in use and not of type ELEMENT_TYPE_FUNCTION_DEFINITION";
            add_error_message( context, msg, -3);
            return -1;
        }
    }

    // check annotation definitions
    mi::Size annotation_count = code_dag->get_annotation_count();
    std::vector<std::string> annotation_names(annotation_count);

    for (mi::Size i = 0; i < annotation_count; ++i) {
        std::string db_annotation_name = get_db_name_annotation_definition(
            MDL::get_mdl_annotation_name( code_dag.get(), i));
        annotation_names[i] = db_annotation_name;
        DB::Tag annotation_tag = transaction->name_to_tag(db_annotation_name.c_str());
        if (annotation_tag
            && transaction->get_class_id(annotation_tag) != ID_MDL_ANNOTATION_DEFINITION_PROXY) {
            // Annotation definition proxies do not exist at the API level, but use the enum name
            // as if they did.
            std::string msg = "DB name for annotation definition '" + db_annotation_name + "' "
                "already in use and not of type ELEMENT_TYPE_ANNOTATION_DEFINITION";
            add_error_message(context, msg, -3);
            return -1;
        }
    }

    // initialize module

    m_ident = generate_unique_id();

    m_code_dag = mi::base::make_handle_dup(code_dag.get());
    m_module = mi::base::make_handle_dup(module);
    m_imports = imports;

    init_module( transaction, context);

    DB::Privacy_level privacy_level = transaction->get_scope()->get_level();
    bool resolve_resources = context->get_option<bool>( MDL_CTX_OPTION_RESOLVE_RESOURCES);

    // Update DB elements for the function definitions in this module.
    std::vector<Mdl_tag_ident> new_functions( function_count);
    for( mi::Size i = 0; i < function_count; ++i) {

        const auto& it = m_function_name_to_index.find( function_names[i]);
        if( it == m_function_name_to_index.end()) {

            // does not exist or signature changed, recreate
            DB::Tag new_tag = transaction->reserve_tag();
            Mdl_function_definition* db_function = new Mdl_function_definition(
                transaction, new_tag, m_ident, module, m_code_dag.get(), /*is_material*/ false, i,
                m_file_name.c_str(), m_mdl_name.c_str(), resolve_resources);

            LOG::mod_log->debug( M_SCENE, LOG::Mod_log::C_DATABASE,
              "    Function %llu (reload/new, DB name): \"%s\"", i, function_names[i].c_str());
            new_functions[i] = Mdl_tag_ident( new_tag, m_ident);
            transaction->store_for_reference_counting(
                new_tag, db_function, function_names[i].c_str(), privacy_level);

        } else {

            // exists, prepare to check compatibility
            DB::Tag old_tag = m_functions[it->second].first;
            Mdl_function_definition* db_function = new Mdl_function_definition(
                transaction, old_tag, m_ident, module, m_code_dag.get(), /*is_material*/ false, i,
                m_file_name.c_str(), m_mdl_name.c_str(), resolve_resources);

            DB::Access<Mdl_function_definition> old_db_function( old_tag, transaction);
            if( old_db_function->is_compatible( *db_function)) {
                LOG::mod_log->debug( M_SCENE, LOG::Mod_log::C_DATABASE,
                  "    Function %llu (reload/compatible, DB name): \"%s\"", i, function_names[i].c_str());
                new_functions[i] = Mdl_tag_ident( old_tag, m_functions[it->second].second);
                delete db_function;
            } else {
                LOG::mod_log->debug( M_SCENE, LOG::Mod_log::C_DATABASE,
                  "    Function %llu (reload/replace, DB name): \"%s\"", i, function_names[i].c_str());
                new_functions[i] = Mdl_tag_ident( old_tag, m_ident);
                transaction->store_for_reference_counting(
                    old_tag, db_function, function_names[i].c_str(), privacy_level);
            }
        }
    }
    m_functions = new_functions;

    // Create DB elements for the material definitions in this module.
    std::vector<Mdl_tag_ident> new_materials( material_count);
    for( mi::Size i = 0; i < material_count; ++i) {

        const auto& it = m_material_name_to_index.find( material_names[i]);
        if( it == m_material_name_to_index.end()) {

            // does not exist or signature changed, recreate
            DB::Tag new_tag = transaction->reserve_tag();
            Mdl_function_definition* db_material = new Mdl_function_definition(
                transaction, new_tag, m_ident, module, m_code_dag.get(), /*is_material*/ true, i,
                m_file_name.c_str(), m_mdl_name.c_str(), resolve_resources);

            LOG::mod_log->debug( M_SCENE, LOG::Mod_log::C_DATABASE,
              "    Material %llu (reload/new, DB name): \"%s\"", i, material_names[i].c_str());
            new_materials[i] = Mdl_tag_ident( new_tag, m_ident);
            transaction->store_for_reference_counting(
                new_tag, db_material, material_names[i].c_str(), privacy_level);

        } else {

            // exists, prepare to check compatibility
            DB::Tag old_tag = m_materials[it->second].first;
            Mdl_function_definition* db_material = new Mdl_function_definition(
                transaction, old_tag, m_ident, module, m_code_dag.get(), /*is_material*/ true, i,
                m_file_name.c_str(), m_mdl_name.c_str(), resolve_resources);

            DB::Access<Mdl_function_definition> old_db_material( old_tag, transaction);
            if( old_db_material->is_compatible( *db_material)) {
                LOG::mod_log->debug( M_SCENE, LOG::Mod_log::C_DATABASE,
                  "    Material %llu (reload/compatible, DB name): \"%s\"", i, material_names[i].c_str());
                new_materials[i] = Mdl_tag_ident( old_tag, m_materials[it->second].second);
                delete db_material;
            } else {
                LOG::mod_log->debug( M_SCENE, LOG::Mod_log::C_DATABASE,
                  "    Material %llu (reload/replace, DB name): \"%s\"", i, material_names[i].c_str());
                new_materials[i] = Mdl_tag_ident( old_tag, m_ident);
                transaction->store_for_reference_counting(
                    old_tag, db_material, material_names[i].c_str(), privacy_level);
            }
        }
    }
    m_materials = new_materials;

    // Create DB elements for the annotation definitions in this module.
    std::vector<DB::Tag> new_annotations( annotation_count);
    for( mi::Size i = 0; i < annotation_count; ++i) {

        const auto& it = m_annotation_name_to_index.find( annotation_names[i]);
        if( it == m_annotation_name_to_index.end()) {

            // does not exist or signature changed, recreate
            DB::Tag new_tag = transaction->reserve_tag();
            Mdl_annotation_definition_proxy* db_annotation = new Mdl_annotation_definition_proxy(
                m_mdl_name.c_str());

            LOG::mod_log->debug( M_SCENE, LOG::Mod_log::C_DATABASE,
              "    Annotation %llu (reload/new, DB name): \"%s\"", i, annotation_names[i].c_str());
            new_annotations[i] = new_tag;
            transaction->store_for_reference_counting(
                new_tag, db_annotation, annotation_names[i].c_str(), privacy_level);

        } else {

            // no compatibility checking for annotations, always recreate
            DB::Tag old_tag = m_annotation_proxies[it->second];
            Mdl_annotation_definition_proxy* db_annotation = new Mdl_annotation_definition_proxy(
                m_mdl_name.c_str());

            LOG::mod_log->debug( M_SCENE, LOG::Mod_log::C_DATABASE,
              "    Annotation %llu (reload/replace, DB name): \"%s\"", i, annotation_names[i].c_str());
            new_annotations[i] = old_tag;
            transaction->store_for_reference_counting(
                old_tag, db_annotation, annotation_names[i].c_str(), privacy_level);
        }
    }
    m_annotation_proxies = new_annotations;

    m_function_name_to_index.clear();
    for( mi::Size i = 0, n = m_functions.size(); i < n; ++i)
        m_function_name_to_index[function_names[i]] = i;

    m_material_name_to_index.clear();
    for( mi::Size i = 0, n = m_materials.size(); i < n; ++i)
        m_material_name_to_index[material_names[i]] = i;

    m_annotation_name_to_index.clear();
    for( mi::Size i = 0, n = m_annotation_proxies.size(); i < n; ++i)
        m_annotation_name_to_index[annotation_names[i]] = i;

    return 0;
}

mi::Sint32 Mdl_module::has_definition(
    bool is_material, const std::string& def_name, Mdl_ident def_ident) const
{
    if( is_material) {
        auto it = m_material_name_to_index.find( def_name);
        if( it == m_material_name_to_index.end())
            return -1;
        if( m_materials[it->second].second != def_ident)
            return -2;
        return 0;
    } else {
        auto it = m_function_name_to_index.find( def_name);
        if( it == m_function_name_to_index.end())
            return -1;
        if( m_functions[it->second].second != def_ident)
            return -2;
        return 0;
    }
}

mi::Size Mdl_module::get_definition_index(
    bool is_material, const std::string& def_name, Mdl_ident def_ident) const
{
    if( is_material) {
        const auto& it = m_material_name_to_index.find( def_name);
        if( it == m_material_name_to_index.end())
            return -1;
        if( def_ident == Mdl_ident( -1))
            return it->second;
        if( m_materials[it->second].second != def_ident)
            return -1;
        return it->second;
    } else {
        const auto& it = m_function_name_to_index.find( def_name);
        if( it == m_function_name_to_index.end())
            return -1;
        if( def_ident == Mdl_ident( -1))
            return it->second;
        if( m_functions[it->second].second != def_ident)
            return -1;
        return it->second;
    }
}

DB::Tag Mdl_module::get_definition( bool is_material, mi::Size index) const
{
    return is_material ? get_material( index) : get_function( index);
}

Mdl_ident Mdl_module::get_ident() const
{
    return m_ident;
}

const SERIAL::Serializable* Mdl_module::serialize( SERIAL::Serializer* serializer) const
{
    Scene_element_base::serialize( serializer);

    // m_mdl is never serialized (independent of DB element)
    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module( false);
    mdlc_module->serialize_module( serializer, m_module.get());

    bool has_code = m_code_dag.is_valid_interface();
    SERIAL::write( serializer, has_code);
    if( has_code)
        mdlc_module->serialize_code_dag( serializer, m_code_dag.get());

    SERIAL::write( serializer, m_mdl_name);
    SERIAL::write( serializer, m_simple_name);
    SERIAL::write( serializer, m_package_component_names);
    SERIAL::write( serializer, m_file_name);
    SERIAL::write( serializer, m_api_file_name);

    SERIAL::write( serializer, m_ident);
    SERIAL::write( serializer, m_imports);
    m_tf->serialize_list( serializer, m_exported_types.get());
    m_tf->serialize_list( serializer, m_local_types.get());
    m_vf->serialize_list( serializer, m_constants.get());
    m_ef->serialize_annotation_block( serializer, m_annotations.get());
    m_ef->serialize_annotation_definition_list(serializer, m_annotation_definitions.get());
    SERIAL::write( serializer, m_functions);
    SERIAL::write( serializer, m_materials);
    SERIAL::write( serializer, m_annotation_proxies);
    SERIAL::write( serializer, m_resources);
    SERIAL::write( serializer, m_function_name_to_index);
    SERIAL::write( serializer, m_material_name_to_index);
    SERIAL::write( serializer, m_annotation_name_to_index);

    return this + 1;
}

SERIAL::Serializable* Mdl_module::deserialize( SERIAL::Deserializer* deserializer)
{
    Scene_element_base::deserialize( deserializer);

    // deserialize m_module
    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module( false);
    m_mdl = mdlc_module->get_mdl();
    m_module = mdlc_module->deserialize_module( deserializer);

    bool has_code = false;
    SERIAL::read( deserializer, &has_code);
    if( has_code)
        m_code_dag = mdlc_module->deserialize_code_dag( deserializer);

    SERIAL::read( deserializer, &m_mdl_name);
    SERIAL::read( deserializer, &m_simple_name);
    SERIAL::read( deserializer, &m_package_component_names);
    SERIAL::read( deserializer, &m_file_name);
    SERIAL::read( deserializer, &m_api_file_name);

    SERIAL::read( deserializer, &m_ident);
    SERIAL::read( deserializer, &m_imports);
    m_exported_types = m_tf->deserialize_list( deserializer);
    m_local_types = m_tf->deserialize_list( deserializer);
    m_constants = m_vf->deserialize_list( deserializer);
    m_annotations = m_ef->deserialize_annotation_block( deserializer);
    m_annotation_definitions = m_ef->deserialize_annotation_definition_list(deserializer);
    SERIAL::read( deserializer, &m_functions);
    SERIAL::read( deserializer, &m_materials);
    SERIAL::read( deserializer, &m_annotation_proxies);
    SERIAL::read( deserializer, &m_resources);
    SERIAL::read( deserializer, &m_function_name_to_index);
    SERIAL::read( deserializer, &m_material_name_to_index);
    SERIAL::read( deserializer, &m_annotation_name_to_index);

    return this + 1;
}

void Mdl_module::dump( DB::Transaction* transaction) const
{
    std::ostringstream s;
    s << std::boolalpha;
    mi::base::Handle<const mi::IString> tmp;

    // m_mdl, m_module, m_code_dag missing

    s << "Module MDL name: " << m_mdl_name << std::endl;
    s << "File name: " << m_file_name << std::endl;
    s << "API file name: " << m_api_file_name << std::endl;
    s << "Identifier: " << m_ident << std::endl;

    mi::Size imports_count = m_imports.size();
    for( mi::Size i = 0; i < imports_count; ++i)
        s << "Import " << i << ": "
          << " tag " << m_imports[i].first.get_uint()
          << ", identifier: " << m_imports[i].second << std::endl;

    tmp = m_tf->dump( m_exported_types.get());
    s << "Exported types: " << tmp->get_c_str() << std::endl;

    tmp = m_tf->dump( m_local_types.get());
    s << "Local types: " << tmp->get_c_str() << std::endl;

    tmp = m_vf->dump( transaction, m_constants.get(), /*name*/ nullptr);
    s << "Constants: " << tmp->get_c_str() << std::endl;

    // m_annotations, m_annotation_definitions missing

    mi::Size resources_count = m_resources.size();
    for( mi::Size i = 0; i < resources_count; ++i)
        s << "Resource " << i << ": "
          << "MDL file path " << m_resources[i].m_mdl_file_path
          << ", tag " << m_resources[i].m_tag.get_uint()
          << ", kind " << m_resources[i].m_kind
          << ", selector " << m_resources[i].m_selector
          << ", type kind " <<  m_resources[i].m_type_kind
          << ", texture shape " <<  m_resources[i].m_texture_shape << std::endl;

    mi::Size function_count = m_functions.size();
    for( mi::Size i = 0; i < function_count; ++i)
        s << "Function definition " << i << ": " << m_functions[i].first.get_uint() << std::endl;

    mi::Size material_count = m_materials.size();
    for( mi::Size i = 0; i < material_count; ++i)
        s << "Material definition " << i << ": " << m_materials[i].first.get_uint() << std::endl;

    mi::Size annotation_proxies_count = m_annotation_proxies.size();
    for( mi::Size i = 0; i < annotation_proxies_count; ++i)
        s << "Annotation definition " << i << ": " << m_annotation_proxies[i].get_uint() << std::endl;

    s << std::endl;
    LOG::mod_log->info( M_SCENE, LOG::Mod_log::C_DATABASE, "%s", s.str().c_str());
}

size_t Mdl_module::get_size() const
{
    return sizeof( *this)
        + SCENE::Scene_element<Mdl_module, Mdl_module::id>::get_size()
            - sizeof( SCENE::Scene_element<Mdl_module, Mdl_module::id>)
        + m_module->get_memory_size()
        + (m_code_dag ? m_code_dag->get_memory_size() : 0)
        + dynamic_memory_consumption( m_mdl_name)
        + dynamic_memory_consumption( m_file_name)
        + dynamic_memory_consumption( m_api_file_name)
        + dynamic_memory_consumption( m_imports)
        + dynamic_memory_consumption( m_exported_types)
        + dynamic_memory_consumption( m_local_types)
        + dynamic_memory_consumption( m_constants)
        + dynamic_memory_consumption( m_annotations)
        + dynamic_memory_consumption( m_annotation_definitions)
        + dynamic_memory_consumption( m_functions)
        + dynamic_memory_consumption( m_materials)
        + dynamic_memory_consumption( m_annotation_proxies)
        + dynamic_memory_consumption( m_resources)
        + dynamic_memory_consumption( m_function_name_to_index)
        + dynamic_memory_consumption( m_material_name_to_index)
        + dynamic_memory_consumption( m_annotation_name_to_index);
}

DB::Journal_type Mdl_module::get_journal_flags() const
{
    return SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE;
}

Uint Mdl_module::bundle( DB::Tag* results, Uint size) const
{
    return 0;
}

void Mdl_module::get_scene_element_references( DB::Tag_set* result) const
{
    for( const auto& imp: m_imports)
        result->insert( imp.first);

    for( const auto& fct: m_functions)
        result->insert( fct.first);

    for( const auto& mat: m_materials)
        result->insert( mat.first);

    for( const auto& ad: m_annotation_proxies)
        result->insert( ad);

    collect_references( m_annotations.get(), result);

    for( const auto& res: m_resources)
        if( res.m_tag)
            result->insert( res.m_tag);
}

} // namespace MDL

} // namespace MI

