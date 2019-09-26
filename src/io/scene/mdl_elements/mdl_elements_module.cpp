/***************************************************************************************************
 * Copyright (c) 2012-2019, NVIDIA CORPORATION. All rights reserved.
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

#include "i_mdl_elements_compiled_material.h"
#include "i_mdl_elements_function_call.h"
#include "i_mdl_elements_function_definition.h"
#include "i_mdl_elements_material_definition.h"
#include "i_mdl_elements_material_instance.h"
#include "i_mdl_elements_module_builder.h"
#include "i_mdl_elements_utilities.h"
#include "mdl_elements_ast_builder.h"
#include "mdl_elements_detail.h"
#include "mdl_elements_expression.h"
#include "mdl_elements_utilities.h"

#include <sstream>
#include <mi/mdl/mdl_code_generators.h>
#include <mi/mdl/mdl_generated_dag.h>
#include <mi/mdl/mdl_mdl.h>
#include <mi/mdl/mdl_modules.h>
#include <mi/mdl/mdl_module_transformer.h>
#include <mi/mdl/mdl_definitions.h>
#include <mi/mdl/mdl_thread_context.h>
#include <mi/neuraylib/istring.h>
#include <base/system/main/access_module.h>
#include <base/util/string_utils/i_string_utils.h>
#include <boost/core/ignore_unused.hpp>
#include <base/lib/log/i_log_logger.h>
#include <base/data/db/i_db_access.h>
#include <base/data/db/i_db_transaction.h>
#include <base/data/serial/i_serializer.h>
#include <base/hal/disk/disk.h>
#include <base/hal/hal/i_hal_ospath.h>
#include <mdl/integration/mdlnr/i_mdlnr.h>
#include <mdl/compiler/compilercore/compilercore_modules.h>
#include <io/scene/scene/i_scene_journal_types.h>
#include <io/scene/bsdf_measurement/i_bsdf_measurement.h>
#include <io/scene/lightprofile/i_lightprofile.h>
#include <io/scene/texture/i_texture.h>

#include "mdl/compiler/compilercore/compilercore_modules.h"
#include "mdl/compiler/compilercore/compilercore_def_table.h"
#include "mdl/compiler/compilercore/compilercore_tools.h"

namespace MI {

namespace MDL {


// Interface to access Material instances and Function calls in a uniform way
class ICall {
public:
    /// Get the absolute name of the entity.
    virtual char const *get_abs_name() const = 0;

    /// Get the argument list.
    virtual const IExpression_list* get_arguments() const = 0;

    /// Get the parameter types.
    virtual const IType_list* get_parameter_types() const = 0;
};

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

// Implements the ICall interface for a function call.
class Function_call : public ICall
{
    typedef ICall Base;
public:
    /// Get the absolute name of the entity.
    virtual char const *get_abs_name() const
    {
        // so far it is always "material"
        return "material";
    }

    /// Get the argument list.
    virtual const IExpression_list* get_arguments() const
    {
        return m_call.get_arguments();
    }

    /// Get the parameter types.
    virtual const IType_list* get_parameter_types() const
    {
        return m_call.get_parameter_types();
    }

public:
    /// Constructor.
    ///
    /// \param call  a MDL function call
    Function_call(Mdl_function_call const &call)
    : Base()
    , m_call(call)
    {
    }

private:
    // The mdl function call.
    Mdl_function_call const &m_call;
};

// Implements the ICall interface for a material instance.
class Material_call : public ICall
{
    typedef ICall Base;
public:
    /// Get the absolute name of the entity.
    virtual char const *get_abs_name() const
    {
        return m_inst.get_mdl_material_definition();
    }

    /// Get the argument list.
    virtual const IExpression_list* get_arguments() const
    {
        return m_inst.get_arguments();
    }

    /// Get the parameter types.
    virtual const IType_list* get_parameter_types() const
    {
        return m_inst.get_parameter_types();
    }

public:
    /// Constructor.
    ///
    /// \param inst  a MDL material instance
    Material_call(Mdl_material_instance const &inst)
    : Base()
    , m_inst(inst)
    {
    }

private:
    // The MDL material instance.
    Mdl_material_instance const &m_inst;
};

/// Checks, if the given module name has an .mdle extension.
bool is_mdle(const std::string& module_name)
{
    size_t n = module_name.length();
    if (n > 5) {
        return STRING::to_lower(module_name.substr(n - 5)) == ".mdle";
    }
    return false;
}

// Makes a module name standard conform.
// For standard MDL module names, it is ensured that there is a leading "::".
// For MDLE, the path is normalized, meaning absolute, no "/../" and using forward slashes.
// After execution, 'out_is_mdle' is set to indicate if the module is an MDLE module or not.
std::string normalize_module_name(const std::string& module_name, bool& out_is_mdle)
{
    out_is_mdle = is_mdle(module_name);
    if (out_is_mdle)
    {
        std::string res = module_name;

        // make path absolute
        if (!DISK::is_path_absolute(res.c_str()))
            res = HAL::Ospath::join(DISK::get_cwd(), res);

        // normalize path to make it unique
        res = HAL::Ospath::normpath(res);

        // use forward slashes
        res = HAL::Ospath::convert_to_forward_slashes(res);
        return res;
    }
    else
    {
        // make sure there are leading "::"
        if (module_name[0] == ':' && module_name[1] == ':')
            return module_name;
        else
            return "::" + module_name;
    }

}

std::string normalized_module_name_to_db_name(const std::string& module_name, bool is_mdle)
{
    if(is_mdle)
        // add prefix and insert a leading slash if there is none
        return (module_name[0] == '/' ? "mdle::" : "mdle::/") + module_name;

    return "mdl" + module_name;
}

}  // anonymous

const char* Mdl_module::get_module_db_name(
    DB::Transaction* transaction,
    const char* module_name,
    Execution_context* context)
{
    context->clear_messages();


    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module(false);
    mi::base::Handle<mi::mdl::IMDL> mdl(mdlc_module->get_mdl());

    bool mdle = false;
    std::string normalized_module_name = normalize_module_name(module_name, mdle);
    std::string db_module_name = normalized_module_name_to_db_name(normalized_module_name, mdle);

    // Check whether the module exists in the DB.
    DB::Tag db_module_tag = transaction->name_to_tag(db_module_name.c_str());
    if (!db_module_tag)
    {
        // don't consider this an error. This allows to check if a module is loaded.
        return NULL;
    }

    // Check if the DB entry is a module
    if (transaction->get_class_id(db_module_tag) != Mdl_module::id)
    {
        LOG::mod_log->error(M_SCENE, LOG::Mod_log::C_DATABASE,
            "DB name for module \"%s\" (\"%s\") already in use.", module_name, db_module_name.c_str());
        return NULL;
    }

    // get the name 
    return transaction->tag_to_name(db_module_tag);
}


mi::Sint32 Mdl_module::create_module(
    DB::Transaction* transaction,
    const char* module_name,
    Execution_context* context)
{
    ASSERT(M_SCENE, module_name);
    ASSERT(M_SCENE, context);

    context->clear_messages();

    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module( false);
    mi::base::Handle<mi::mdl::IMDL> mdl( mdlc_module->get_mdl());

   
    bool mdle = false;
    std::string normalized_module_name = normalize_module_name(module_name, mdle);
    std::string db_module_name = normalized_module_name_to_db_name(normalized_module_name, mdle);

    if (mdle) {

        // check if the file exists
        if (!DISK::is_file( normalized_module_name.c_str())) {
            LOG::mod_log->error( M_SCENE, LOG::Mod_log::C_DISK,
                "MDLe file \"%s\" does not exist.", normalized_module_name.c_str());
            return -1;
        }

    // otherwise we check for valid mdl identifiers
    } else {
    
        // Reject invalid module names (in particular, names containing slashes and backslashes).
        if (!is_valid_module_name(module_name, mdl.get()))
            return -1;
    }

    // Check whether the module exists already in the DB.
    DB::Tag db_module_tag = transaction->name_to_tag( db_module_name.c_str());
    if( db_module_tag) {
        if( transaction->get_class_id( db_module_tag) != Mdl_module::id) {
            LOG::mod_log->error( M_SCENE, LOG::Mod_log::C_DATABASE,
                "DB name for module \"%s\" already in use.", db_module_name.c_str());
            return -3;
        }
        return 1;
    }

    Module_cache module_cache( transaction);
    mi::base::Handle<mi::mdl::IThread_context> ctx( mdl->create_thread_context());
    mi::mdl::Options& options = ctx->access_options();
    options.set_option(MDL_OPTION_EXPERIMENTAL_FEATURES,
        context->get_option<bool>(MDL_CTX_OPTION_EXPERIMENTAL) ? "true" : "false");
    options.set_option(MDL_OPTION_RESOLVE_RESOURCES,
        context->get_option<bool>(MDL_CTX_OPTION_RESOLVE_RESOURCES) ? "true" : "false");

    mi::base::Handle<const mi::mdl::IModule> module(
        mdl->load_module( ctx.get(), normalized_module_name.c_str(), &module_cache));

    // report messages even when the module is valid (warnings, notes, ...)
    report_messages(ctx->access_messages(), context);

    if( !module.is_valid_interface())
        return -2;

    mi::Sint32 result
        = create_module_internal(transaction, mdl.get(), module.get(), context);
    if( result < 0)
        return result;

    return result;
}

namespace {

// Wraps a mi::neuraylib::IReader as mi::mdl::IInput_stream.
class Input_stream : public mi::base::Interface_implement<mi::mdl::IInput_stream>
{
public:
    Input_stream( mi::neuraylib::IReader* reader) : m_reader( reader, mi::base::DUP_INTERFACE) { }
    int read_char()
    { char c; mi::Sint64 result = m_reader->read( &c, 1); return result <= 0 ? -1 : c; }
    const char* get_filename() { return 0; }
private:
    mi::base::Handle<mi::neuraylib::IReader> m_reader;
};

} // namespace

mi::Sint32 Mdl_module::create_module(
    DB::Transaction* transaction,
    const char* module_name,
    mi::neuraylib::IReader* module_source,
    Execution_context* context)
{
    ASSERT( M_SCENE, module_name);
    ASSERT( M_SCENE, module_source);
    ASSERT( M_SCENE, context);

    context->clear_messages();

    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module( false);
    mi::base::Handle<mi::mdl::IMDL> mdl( mdlc_module->get_mdl());

    // Reject invalid module names (in particular, names containing slashes and backslashes).
    if( !is_valid_module_name( module_name, mdl.get()) && strcmp( module_name, "::<neuray>") != 0)
        return -1;

    // Check whether the module exists already in the DB.
    std::string db_module_name = add_mdl_db_prefix( module_name);
    DB::Tag db_module_tag = transaction->name_to_tag( db_module_name.c_str());
    if( db_module_tag) {
        if( transaction->get_class_id( db_module_tag) != Mdl_module::id) {
            LOG::mod_log->error( M_SCENE, LOG::Mod_log::C_DATABASE,
                "DB name for module \"%s\" already in use.", db_module_name.c_str());
            return -3;
        }

        // module already exists
        return 1;
    }

    Input_stream module_source_stream( module_source);
    Module_cache module_cache( transaction);
    mi::base::Handle<mi::mdl::IThread_context> ctx( mdl->create_thread_context());
    mi::base::Handle<const mi::mdl::IModule> module( mdl->load_module_from_stream(
        ctx.get(), &module_cache, module_name, &module_source_stream));
    if( !module.is_valid_interface()) {
        report_messages( ctx->access_messages(), context);
        return -2;
     }

    return create_module_internal(transaction, mdl.get(), module.get(), context);
}

/// Get the version of an module
static mi::mdl::IMDL::MDL_version get_version(
    mi::base::Handle<mi::mdl::IModule const> mdl_module)
{
    mi::mdl::Module const *mod = mi::mdl::impl_cast<mi::mdl::Module>(mdl_module.get());
    return mod->get_mdl_version();
}

/// Get the version of a stdlib entity.
static mi::mdl::IMDL::MDL_version get_stdlib_version(
    mi::base::Handle<mi::mdl::IModule const>  mdl_module,
    DB::Access<Mdl_material_definition> const &material)
{
    // materials should not be defined in the standard library
    ASSERT(M_SCENE, !"unexpected material definition found in stdlib");
    return get_version(mdl_module);
}

/// Get the version of a stdlib entity.
static mi::mdl::IMDL::MDL_version get_stdlib_version(
    mi::base::Handle<mi::mdl::IModule const>  mdl_module,
    DB::Access<Mdl_function_definition> const &func)
{
    if (func->get_semantic() == mi::neuraylib::IFunction_definition::DS_CAST)
        return mi::mdl::IMDL::MDL_VERSION_1_5;
    else if (mi::mdl::semantic_is_operator(func->get_mdl_semantic()) ||
        (func->get_semantic() >= mi::neuraylib::IFunction_definition::DS_INTRINSIC_DAG_FIRST &&
         func->get_semantic() <= mi::neuraylib::IFunction_definition::DS_INTRINSIC_DAG_LAST))
        return mi::mdl::IMDL::MDL_VERSION_1_0;

    mi::mdl::Module const     *mod = mi::mdl::impl_cast<mi::mdl::Module>(mdl_module.get());
    mi::mdl::Definition const *def = mi::mdl::impl_cast<mi::mdl::Definition>(mod->find_signature(
        func->get_mdl_name(), /*only_exported=*/!mod->is_builtins()));

    if (def == NULL) {
        // should not happen
        ASSERT(M_SCENE, !"Function definition in stdlib was not found");
        return get_version(mdl_module);
    }

    unsigned flags = def->get_version_flags();

    return mi::mdl::IMDL::MDL_version(mi::mdl::mdl_since_version(flags));
}

/// Get the smallest version necessary for a given function definition.
template<typename Definition>
static mi::mdl::IMDL::MDL_version get_version(
    DB::Transaction                       *transaction,
    typename DB::Access<Definition> const &def)
{
    DB::Tag module_tag = def->get_module(transaction);

    DB::Access<Mdl_module> module(module_tag, transaction);
    mi::base::Handle<mi::mdl::IModule const> mdl_module(module->get_mdl_module());

    if (mdl_module->is_stdlib() || mdl_module->is_builtins()) {
        return get_stdlib_version(mdl_module, def);
    } else {
        // get the version of the module
        return get_version(mdl_module);
    }
}

/// Get the smallest version necessary for a given expression list.
static mi::mdl::IMDL::MDL_version get_version(
    DB::Transaction                          *transaction,
    mi::base::Handle<IExpression_list const> expr_list);

/// Get the smallest version necessary for a given expression.
static mi::mdl::IMDL::MDL_version get_version(
    DB::Transaction                     *transaction,
    mi::base::Handle<IExpression const> expr)
{
    switch (expr->get_kind()) {
    case IExpression::EK_FORCE_32_BIT:
    case IExpression::EK_CONSTANT:
    case IExpression::EK_PARAMETER:
        // smallest version is fine
        return mi::mdl::IMDL::MDL_VERSION_1_0;
    case IExpression::EK_CALL:
        {
            mi::base::Handle<IExpression_call const> call(
                expr->get_interface<IExpression_call>());
            DB::Tag call_tag = call->get_call();
            SERIAL::Class_id class_id = transaction->get_class_id(call_tag);
            mi::mdl::IMDL::MDL_version version = mi::mdl::IMDL::MDL_VERSION_1_0;
            if (class_id == ID_MDL_MATERIAL_INSTANCE) {
                DB::Access<Mdl_material_instance> minst(call_tag, transaction);
                DB::Tag def_tag = minst->get_material_definition();
                DB::Access<Mdl_material_definition> mdef(def_tag, transaction);

                version = get_version(transaction, mdef);
                mi::base::Handle<IExpression_list const> args(minst->get_arguments());

                mi::mdl::IMDL::MDL_version v = get_version(transaction, args);
                if (v > version)
                    version = v;
            } else if (class_id == ID_MDL_FUNCTION_CALL) {
                DB::Access<Mdl_function_call> fcall(call_tag, transaction);
                DB::Tag def_tag = fcall->get_function_definition();
                DB::Access<Mdl_function_definition> fdef(def_tag, transaction);

                version = get_version(transaction, fdef);
                mi::base::Handle<IExpression_list const> args(fcall->get_arguments());

                mi::mdl::IMDL::MDL_version v = get_version(transaction, args);
                if (v > version)
                    version = v;
            } else {
                ASSERT(M_SCENE, !"call to unknown entity class");
            }
            return version;
        }
        break;
    case IExpression::EK_DIRECT_CALL:
        {
            mi::base::Handle<IExpression_direct_call const> call(
                expr->get_interface<IExpression_direct_call>());
            DB::Tag def_tag = call->get_definition();
            SERIAL::Class_id class_id = transaction->get_class_id(def_tag);
            mi::mdl::IMDL::MDL_version version = mi::mdl::IMDL::MDL_VERSION_1_0;
            if (class_id == ID_MDL_MATERIAL_DEFINITION) {
                DB::Access<Mdl_material_definition> mdef(def_tag, transaction);
                version = get_version(transaction, mdef);
            } else if (class_id == ID_MDL_FUNCTION_DEFINITION) {
                DB::Access<Mdl_function_definition> fdef(def_tag, transaction);
                version = get_version(transaction, fdef);
            } else {
                ASSERT(M_SCENE, !"call to unknown entity class");
            }

            mi::base::Handle<IExpression_list const> args(call->get_arguments());
            mi::mdl::IMDL::MDL_version v = get_version(transaction, args);
            if (v > version)
                version = v;
            return version;
        }
        break;
    case IExpression::EK_TEMPORARY:
        break;
    }
    // should not happen here
    ASSERT(M_SCENE, !"Unsupported expression kind");
    return mi::mdl::IMDL::MDL_VERSION_1_0;
}

/// Get the smallest version necessary for a given expression list.
static mi::mdl::IMDL::MDL_version get_version(
    DB::Transaction                          *transaction,
    mi::base::Handle<IExpression_list const> expr_list)
{
    mi::mdl::IMDL::MDL_version version = mi::mdl::IMDL::MDL_VERSION_1_0;

    for (mi::Size i = 0, n = expr_list->get_size(); i < n; ++i) {
        mi::base::Handle<IExpression const> expr(expr_list->get_expression(i));

        mi::mdl::IMDL::MDL_version v = get_version(transaction, expr);

        if (v > version)
            version = v;
    }
    return version;
}

mi::Sint32 Mdl_module::create_module(
    DB::Transaction* transaction,
    const char* module_name,
    const Variant_data* variant_data,
    mi::Size variant_count,
    Execution_context* context)
{
    ASSERT( M_SCENE, module_name);
    ASSERT( M_SCENE, variant_data);
    ASSERT( M_SCENE, context);

    context->clear_messages();

    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module( false);
    mi::base::Handle<mi::mdl::IMDL> mdl( mdlc_module->get_mdl());

    // Reject invalid module names (in particular, names containing slashes and backslashes).
    if( !is_valid_module_name( module_name, mdl.get()))
        return -1;

    // Check whether the module exists already in the DB.
    std::string db_module_name = add_mdl_db_prefix( module_name);
    DB::Tag db_module_tag = transaction->name_to_tag( db_module_name.c_str());
    if( db_module_tag) {
        if( transaction->get_class_id( db_module_tag) != Mdl_module::id) {
            LOG::mod_log->error( M_SCENE, LOG::Mod_log::C_DATABASE,
                "DB name for module \"%s\" already in use.", db_module_name.c_str());
            return -3;
        }
        return 1;
    }

    // Detect the MDL version we need.
    mi::mdl::IMDL::MDL_version version = mi::mdl::IMDL::MDL_VERSION_1_0;
    for (mi::Size i = 0; i < variant_count; ++i) {
        const Variant_data& pd = variant_data[i];
        SERIAL::Class_id class_id = transaction->get_class_id(pd.m_prototype_tag);
        DB::Tag module_tag;
        if (class_id == ID_MDL_MATERIAL_DEFINITION) {
            DB::Access<Mdl_material_definition> prototype(pd.m_prototype_tag, transaction);
            mi::mdl::IMDL::MDL_version v = get_version(transaction, prototype);
            if (v > version)
                version = v;
        } else if (class_id == ID_MDL_FUNCTION_DEFINITION) {
            // function presets require at least mdl 1.3
            if (version < mi::mdl::IMDL::MDL_VERSION_1_3)
                version = mi::mdl::IMDL::MDL_VERSION_1_3;
            DB::Access<Mdl_function_definition> prototype(pd.m_prototype_tag, transaction);
            mi::mdl::IMDL::MDL_version v = get_version(transaction, prototype);
            if (v > version)
                version = v;
        } else {
            return -5;
        }

        if (pd.m_defaults) {
            // also check new defaults
            for (mi::Size j = 0, n = pd.m_defaults->get_size(); j < n; ++j) {
                mi::base::Handle<IExpression const> expr(pd.m_defaults->get_expression(j));

                mi::mdl::IMDL::MDL_version v = get_version(transaction, expr);
                if (v > version)
                    version = v;
            }
        }
    }

    // Create module
    mi::base::Handle<mi::mdl::IModule> module(
        mdl->create_module( /*context=*/NULL, module_name, version));

    Symbol_importer symbol_importer( module.get());

    bool allow_cast = mdlc_module->get_implicit_cast_enabled();

    // Add variants to module
    for( mi::Size i = 0; i < variant_count; ++i) {

        const Variant_data& pd = variant_data[i];
        SERIAL::Class_id class_id = transaction->get_class_id( pd.m_prototype_tag);

        if( class_id == ID_MDL_MATERIAL_DEFINITION) {
            DB::Access<Mdl_material_definition> prototype( pd.m_prototype_tag, transaction);
            mi::Sint32 result = add_variant( symbol_importer, transaction, module.get(),
                prototype, pd.m_variant_name.c_str(), pd.m_defaults.get(),
                pd.m_annotations.get(), allow_cast, context);
            if( result != 0) {
                LOG::mod_log->error( M_SCENE, LOG::Mod_log::C_DATABASE,
                    "Failed to add variant \"%s\" to the module \"%s\".",
                    pd.m_variant_name.c_str(), module_name);
                return result;
            }

        } else if ( class_id == ID_MDL_FUNCTION_DEFINITION) {
            DB::Access<Mdl_function_definition> prototype( pd.m_prototype_tag, transaction);
            mi::Sint32 result = add_variant( symbol_importer, transaction, module.get(),
                prototype, pd.m_variant_name.c_str(), pd.m_defaults.get(),
                pd.m_annotations.get(), allow_cast, context);
            if( result != 0) {
                LOG::mod_log->error( M_SCENE, LOG::Mod_log::C_DATABASE,
                    "Failed to add variant \"%s\" to the module \"%s\".",
                    pd.m_variant_name.c_str(), module_name);
                return result;
            }

        } else {
            ASSERT( M_SCENE, false);
            return -5;
        }
    }

    // add all collected imports
    symbol_importer.add_imports();

    Module_cache module_cache( transaction);
    module->analyze( &module_cache, /*ctx=*/NULL);
    if( !module->is_valid()) {
        LOG::mod_log->error( M_SCENE, LOG::Mod_log::C_DATABASE,
            "Failed to create valid module \"%s\".", module_name);
        report_messages( module->access_messages(), context);
        return -8;
    }

    mi::base::Handle <const mi::mdl::IModule> cmodule(module);

    // inline MDLE
    if (symbol_importer.imports_mdle()) {
        mi::base::Handle<mi::mdl::IMDL_module_transformer> module_transformer(
            mdl->create_module_transformer());

        Module_cache module_cache(transaction);
        if (!module->restore_import_entries(&module_cache)) {
            LOG::mod_log->error(M_SCENE, LOG::Mod_log::C_DATABASE,
                "Failed to restore imports of module \"%s\".", module->get_name());
            return -4;
        }
        Drop_import_scope scope(module.get());

        cmodule = module_transformer->inline_mdle(module.get());
        if (!cmodule) {
            LOG::mod_log->error(M_SCENE, LOG::Mod_log::C_DATABASE,
                "Inlining MDLEs for module \"%s\" failed.", module_name);
            report_messages(module_transformer->access_messages(), context);
            return -8;
        }
    }

    return create_module_internal( transaction, mdl.get(), cmodule.get(), context);
}


// create return type for variant
template <class T>
static const mi::mdl::IType_name* create_return_type_name(
    DB::Transaction* transaction,
    mi::mdl::IModule* module,
    DB::Access<T> prototype);

template <>
const mi::mdl::IType_name* create_return_type_name(
    DB::Transaction* transaction,
    mi::mdl::IModule* module,
    DB::Access<Mdl_material_definition> prototype)
{
    mi::mdl::IName_factory &nf = *module->get_name_factory();

    const mi::mdl::ISymbol* return_type_symbol = nf.create_symbol( "material");
    const mi::mdl::ISimple_name* return_type_simple_name
        = nf.create_simple_name( return_type_symbol);
    mi::mdl::IQualified_name* return_type_qualified_name =nf.create_qualified_name();
    return_type_qualified_name->add_component( return_type_simple_name);
    return nf.create_type_name( return_type_qualified_name);
}

template <>
const mi::mdl::IType_name* create_return_type_name(
    DB::Transaction* transaction,
    mi::mdl::IModule* module,
    DB::Access<Mdl_function_definition> prototype)
{
    const mi::mdl::IType* ret_type = prototype->get_mdl_return_type( transaction);
    return mi::mdl::create_type_name( ret_type, module);
}

template <class T>
mi::Sint32 Mdl_module::add_variant(
    Symbol_importer& symbol_importer,
    DB::Transaction* transaction,
    mi::mdl::IModule* module,
    DB::Access<T> prototype,
    const char* variant_name,
    const IExpression_list* defaults,
    const IAnnotation_block* annotation_block,
    bool allow_cast,
    Execution_context*)
{
    mi::base::Handle<IType_factory> tf( get_type_factory());
    mi::base::Handle<IValue_factory> vf( get_value_factory());
    mi::base::Handle<IExpression_factory> ef( get_expression_factory());

    // dereference prototype references
    DB::Tag dereferenced_prototype_tag = prototype->get_prototype();
    if( dereferenced_prototype_tag) {
        SERIAL::Class_id class_id = transaction->get_class_id( dereferenced_prototype_tag);
        ASSERT( M_SCENE, class_id == T::id);
        boost::ignore_unused( class_id);
        prototype.set( dereferenced_prototype_tag, transaction);
    }

    // check that the provided arguments are parameters of the material definition
    // and that their types match the expected types
    std::vector<bool> needs_cast(prototype->get_parameter_count(), false);

    mi::base::Handle<const IType_list> expected_types( prototype->get_parameter_types());
    for( mi::Size i = 0; defaults && i < defaults->get_size(); ++i) {
        const char* name = defaults->get_name( i);
        mi::Size index = expected_types->get_index(name);
        mi::base::Handle<const IType> expected_type( expected_types->get_type(index));
        if( !expected_type)
            return -6;
        mi::base::Handle<const IExpression> default_( defaults->get_expression( i));
        mi::base::Handle<const IType> actual_type( default_->get_type());

        bool needs_cast_tmp = false;
        if( !argument_type_matches_parameter_type(
            tf.get(), actual_type.get(),expected_type.get(), allow_cast, needs_cast_tmp))
            return -7;
        needs_cast[index] = needs_cast_tmp;
    }

    // create call expression for variant
    const char* prototype_name = prototype->get_mdl_name();
    const mi::mdl::IExpression_reference* prototype_ref
        = signature_to_reference( module, prototype_name);
    mi::mdl::IExpression_factory* expr_factory = module->get_expression_factory();
    mi::mdl::IExpression_call* variant_call = expr_factory->create_call( prototype_ref);

    mi::mdl::IName_factory &nf = *module->get_name_factory();

    // create defaults for variant
    if (defaults != NULL) {
        for (mi::Size i = 0, n = prototype->get_parameter_count(); i < n; ++i) {
            const char* arg_name = prototype->get_parameter_name(i);
            mi::base::Handle<const IExpression> default_(defaults->get_expression(arg_name));
            if (!default_)
                continue;
            if (needs_cast[i]) {

                mi::base::Handle<const IType> expected_type(expected_types->get_type(i));
                mi::Sint32 errors;
                mi::base::Handle<IExpression> expr(
                    ef->clone(default_.get(), transaction, /*copy_immutable_calls=*/false));
                default_ = ef->create_cast(
                    transaction,
                    expr.get(),
                    expected_type.get(),
                    /*cast_db_name=*/nullptr,
                    /*force_cast=*/false,
                    /*direct_call=*/false,
                    &errors);
            }
            const mi::mdl::ISymbol* arg_symbol = nf.create_symbol(arg_name);
            const mi::mdl::ISimple_name* arg_simple_name = nf.create_simple_name(arg_symbol);
            const mi::mdl::IType* arg_type
                = prototype->get_mdl_parameter_type(transaction, static_cast<mi::Uint32>(i));
            const mi::mdl::IExpression* arg_expr = int_expr_to_mdl_ast_expr(
                transaction, module, arg_type, default_.get());
            arg_expr = promote_expressions_to_mdl_version(module, arg_expr);
            if (!arg_expr)
                return -8;
            const mi::mdl::IArgument* argument
                = expr_factory->create_named_argument(arg_simple_name, arg_expr);
            variant_call->add_argument(argument);
            symbol_importer.collect_imports(arg_expr);
        }
    }

    // create annotations for variant
    mi::mdl::IAnnotation_block* mdl_annotation_block = 0;
    mi::Sint32 result = create_annotations(
        transaction, module, annotation_block, &symbol_importer, mdl_annotation_block);
    if( result != 0)
        return result;

    // add imports required by defaults
    symbol_importer.collect_imports( variant_call);

    // create return type for variant
    const mi::mdl::IType_name* return_type_type_name
        = create_return_type_name( transaction, module, prototype);
    symbol_importer.collect_imports(return_type_type_name);

    // create body for variant
    mi::mdl::IStatement_factory* stat_factory = module->get_statement_factory();
    const mi::mdl::IStatement_expression* variant_body
        = stat_factory->create_expression( variant_call);

    const mi::mdl::ISymbol* variant_symbol = nf.create_symbol( variant_name);
    const mi::mdl::ISimple_name* variant_simple_name = nf.create_simple_name( variant_symbol);
    mi::mdl::IDeclaration_factory* decl_factory = module->get_declaration_factory();
    mi::mdl::IDeclaration_function* variant_declaration = decl_factory->create_function(
        return_type_type_name, /*ret_annotations*/ 0, variant_simple_name, /*is_clone*/ true,
        variant_body, mdl_annotation_block, /*is_exported*/ true);

    // add declaration to module
    module->add_declaration( variant_declaration);
    return 0;
}

mi::Sint32 Mdl_module::create_annotations(
    DB::Transaction* transaction,
    mi::mdl::IModule* module,
    const IAnnotation_block* annotation_block,
    Symbol_importer* symbol_importer,
    mi::mdl::IAnnotation_block* &mdl_annotation_block)
{
    if( !annotation_block) {
        mdl_annotation_block = 0;
        return 0;
    }

    mi::mdl::IAnnotation_factory* annotation_factory = module->get_annotation_factory();
    mdl_annotation_block = annotation_factory->create_annotation_block();

    for( mi::Size i = 0; i < annotation_block->get_size(); ++i) {
        mi::base::Handle<const IAnnotation> anno( annotation_block->get_annotation( i));
        const char* anno_name = anno->get_name();
        mi::base::Handle<const IExpression_list> anno_args( anno->get_arguments());
        mi::Sint32 result = add_annotation(
            transaction, module, mdl_annotation_block, anno_name, anno_args.get());
        if( result != 0)
            return result;
    }

    symbol_importer->collect_imports( mdl_annotation_block);
    return 0;
}

mi::Sint32 Mdl_module::add_annotation(
    DB::Transaction* transaction,
    mi::mdl::IModule* module,
    mi::mdl::IAnnotation_block* mdl_annotation_block,
    const char* annotation_name,
    const IExpression_list* annotation_args)
{
    if( strncmp( annotation_name, "::", 2) != 0)
        return -10;
    std::string annotation_name_str = add_mdl_db_prefix( annotation_name);

    // compute DB name of module containing the annotation
    std::string anno_db_module_name = annotation_name_str;
    size_t left_paren = anno_db_module_name.find( '(');
    if( left_paren == std::string::npos)
        return -10;
    anno_db_module_name = anno_db_module_name.substr( 0, left_paren);
    size_t last_double_colon = anno_db_module_name.rfind( "::");
    if( last_double_colon == std::string::npos)
        return -10;
    anno_db_module_name = anno_db_module_name.substr( 0, last_double_colon);

    // get definition of the annotation
    DB::Tag anno_db_module_tag = transaction->name_to_tag( anno_db_module_name.c_str());
    if( !anno_db_module_tag)
        return -10;
    DB::Access<Mdl_module> anno_db_module( anno_db_module_tag, transaction);
    mi::base::Handle<const mi::mdl::IModule> anno_mdl_module( anno_db_module->get_mdl_module());
    std::string annotation_name_wo_signature = annotation_name_str.substr( 3, left_paren-3);
    std::string signature = annotation_name_str.substr(
        left_paren+1, annotation_name_str.size()-left_paren-2);
    const mi::mdl::IDefinition* definition = anno_mdl_module->find_annotation(
        annotation_name_wo_signature.c_str(), signature.c_str());
    if( !definition)
        return -10;

    mi::mdl::IName_factory &nf = *module->get_name_factory();

    // compute IQualified_name for annotation name
    mi::mdl::IQualified_name* anno_qualified_name = nf.create_qualified_name();
    anno_qualified_name->set_absolute();
    size_t start = 5; // skip leading "mdl::"
    while( true) {
        size_t end = annotation_name_str.find( "::", start);
        if( end == std::string::npos || end >= left_paren)
            end = left_paren;
        const mi::mdl::ISymbol* anno_symbol
            = nf.create_symbol( annotation_name_str.substr( start, end-start).c_str());
        const mi::mdl::ISimple_name* anno_simple_name = nf.create_simple_name( anno_symbol);
        anno_qualified_name->add_component( anno_simple_name);
        if( end == left_paren)
            break;
        start = end + 2;
    }

    // create annotation
    mi::mdl::IAnnotation_factory* anno_factory = module->get_annotation_factory();
    mi::mdl::IAnnotation* anno = anno_factory->create_annotation( anno_qualified_name);

    // store parameter types from annotation definition in a map by parameter name
    const mi::mdl::IType* type = definition->get_type();
    ASSERT( M_SCENE, type->get_kind() == mi::mdl::IType::TK_FUNCTION);
    const mi::mdl::IType_function* type_function = mi::mdl::as<mi::mdl::IType_function>( type);
    std::map<std::string, const mi::mdl::IType*> parameter_types;
    int parameter_count = type_function->get_parameter_count();
    for( int i = 0; i < parameter_count; ++i) {
        const mi::mdl::IType* parameter_type;
        const mi::mdl::ISymbol* parameter_name;
        type_function->get_parameter( i, parameter_type, parameter_name);
        parameter_types[parameter_name->get_name()] = parameter_type;
    }

    // convert arguments
    mi::mdl::IType_factory* type_factory = module->get_type_factory();
    mi::mdl::IValue_factory* value_factory = module->get_value_factory();
    mi::mdl::IExpression_factory* expression_factory = module->get_expression_factory();
    mi::Size argument_count = annotation_args->get_size();
    for( mi::Size i = 0; i < argument_count; ++i) {

        const char* arg_name = annotation_args->get_name( i);

        mi::base::Handle<const IExpression_constant> arg_expr(
            annotation_args->get_expression<IExpression_constant>( i));
        if( !arg_expr)
            return -9;
        mi::base::Handle<const IValue> arg_value( arg_expr->get_value());
        mi::base::Handle<const IType> arg_type( arg_value->get_type());

        // The legacy API always provides "argument" as argument name. Since it supports only single
        // string arguments we map that argument name to the correct one if all these conditions are
        // met -- even for the non-legacy API.
        if( i == 0
            && parameter_count == 1
            && argument_count == 1
            && strcmp( arg_name, "argument") == 0
            && arg_type->get_kind() == IType::TK_STRING) {
            arg_name = parameter_types.begin()->first.c_str();
        }

        const mi::mdl::IType* mdl_parameter_type = parameter_types[arg_name];
        if( !mdl_parameter_type)
            return -9;
        mdl_parameter_type = type_factory->import( mdl_parameter_type);
        const mi::mdl::IValue* mdl_arg_value = int_value_to_mdl_value(
            transaction, value_factory, mdl_parameter_type, arg_value.get());
        if( !mdl_arg_value)
            return -9;

        const mi::mdl::IExpression* mdl_arg_expr
            = expression_factory->create_literal( mdl_arg_value);
        const mi::mdl::ISymbol* arg_symbol = nf.create_symbol( arg_name);
        const mi::mdl::ISimple_name* arg_simple_name = nf.create_simple_name( arg_symbol);
        const mi::mdl::IArgument* mdl_arg
            = expression_factory->create_named_argument( arg_simple_name, mdl_arg_expr);
        anno->add_argument( mdl_arg);
    }

    mdl_annotation_block->add_annotation( anno);
    return 0;
}

mi::Sint32 Mdl_module::create_module_internal(
    DB::Transaction* transaction,
    mi::mdl::IMDL* mdl,
    const mi::mdl::IModule* module,
    Execution_context* context,
    DB::Tag* module_tag)
{
    ASSERT( M_SCENE, mdl);
    ASSERT( M_SCENE, module);
    const char* module_name     = module->get_name();
    ASSERT( M_SCENE, module_name);
    const char* module_filename = module->get_filename();
    if( module_filename[0] == '\0')
        module_filename = 0;
    ASSERT( M_SCENE, !mdl->is_builtin_module( module_name) || !module_filename);

    report_messages( module->access_messages(), context);
    if( !module->is_valid())
        return -2;

    // Check whether the module exists already in the DB.
    std::string db_module_name = add_mdl_db_prefix( module->get_name());
    DB::Tag db_module_tag = transaction->name_to_tag( db_module_name.c_str());
    if( db_module_tag) {
        if( transaction->get_class_id( db_module_tag) != Mdl_module::id) {
            LOG::mod_log->error( M_SCENE, LOG::Mod_log::C_DATABASE,
                "DB name for module \"%s\" already in use.", db_module_name.c_str());
            return -3;
        }
        if( module_tag)
            *module_tag = db_module_tag;
        return 1;
    }

    // Compile the module.
    mi::base::Handle<mi::mdl::ICode_generator_dag> generator_dag
        = mi::base::make_handle( mdl->load_code_generator( "dag"))
            .get_interface<mi::mdl::ICode_generator_dag>();

    mi::mdl::Options& options = generator_dag->access_options();

    // We support local entity usage inside MDL materials in neuray, but ...
    options.set_option( MDL_CG_DAG_OPTION_NO_LOCAL_FUNC_CALLS, "false");
    /// ... we need entries for those in the DB, hence generate them
    options.set_option( MDL_CG_DAG_OPTION_INCLUDE_LOCAL_ENTITIES, "true");

    const std::string internal_space =
        context->get_option<std::string>(MDL_CTX_OPTION_INTERNAL_SPACE);
    options.set_option(MDL_CG_OPTION_INTERNAL_SPACE, internal_space.c_str());

   
    Module_cache module_cache( transaction);
    if( !module->restore_import_entries( &module_cache)) {
        LOG::mod_log->error( M_SCENE, LOG::Mod_log::C_DATABASE,
            "Failed to restore imports of module \"%s\".", module->get_name());
        return -4;
    }
    Drop_import_scope scope( module);
    mi::base::Handle<mi::mdl::IGenerated_code> code( generator_dag->compile( module));
    if( !code.is_valid_interface())
        return -2;

    const mi::mdl::Messages& code_messages = code->access_messages();
    report_messages( code_messages, context);

    // Treat error messages as compilation failures, e.g., "Call to unexported function '...' is
    // not allowed in this context".
    if( code_messages.get_error_message_count() > 0)
        return -2;

    ASSERT( M_SCENE, code->get_kind() == mi::mdl::IGenerated_code::CK_DAG);
    mi::base::Handle<mi::mdl::IGenerated_code_dag> code_dag(
        code->get_interface<mi::mdl::IGenerated_code_dag>());

    bool load_resources = context->get_option<bool>(MDL_CTX_OPTION_RESOLVE_RESOURCES);
    if (load_resources) {
        update_resource_literals(transaction, code_dag.get(), module_filename, module_name);
    }
    else {
        LOG::mod_log->info(M_SCENE, LOG::Mod_log::C_DATABASE,
            "Resource loading has been disabled for module '%s'. "
            "Resource data will be unavailable.", module->get_name());
    }

    // Collect tags of imported modules, create DB elements on the fly if necessary.
    mi::Uint32 import_count = module->get_import_count();
    std::vector<DB::Tag> imports;
    imports.reserve( import_count);

    for( mi::Uint32 i = 0; i < import_count; ++i) {
        mi::base::Handle<const mi::mdl::IModule> import( module->get_import( i));
        std::string db_import_name = add_mdl_db_prefix( import->get_name());
        DB::Tag import_tag = transaction->name_to_tag( db_import_name.c_str());
        if( import_tag) {
            // Sanity-check for the type of the tag.
            if( transaction->get_class_id( import_tag) != Mdl_module::id)
                return -3;
        } else {
            // The imported module does not yet exist in the DB.
            mi::Sint32 result = create_module_internal(
                transaction, mdl, import.get(), context, &import_tag);
            if( result < 0) {
                LOG::mod_log->error( M_SCENE, LOG::Mod_log::C_DATABASE,
                    "Failed to initialize imported module \"%s\".", import->get_name());
                return -4;
            }
        }
        imports.push_back( import_tag);
    }

    // Compute DB names of the function definitions in this module.
    mi::Uint32 function_count = code_dag->get_function_count();
    std::vector<std::string> function_names;
    function_names.reserve( function_count);
    std::vector <DB::Tag> function_tags;
    function_tags.reserve(function_count);
    for( mi::Uint32 i = 0; i < function_count; ++i) {
        std::string db_function_name = add_mdl_db_prefix( code_dag->get_function_name( i));
        function_names.push_back( db_function_name);
        DB::Tag function_tag = transaction->name_to_tag( db_function_name.c_str());
        if( function_tag) {
            LOG::mod_log->error( M_SCENE, LOG::Mod_log::C_DATABASE,
                "DB name for function definition \"%s\" already in use.", db_function_name.c_str());
            return -3;
        }
        function_tags.push_back(transaction->reserve_tag());
    }

    // Compute DB names of the material definitions in this module.
    mi::Uint32 material_count = code_dag->get_material_count();
    std::vector<std::string> material_names;
    material_names.reserve( material_count);

    std::vector <DB::Tag> material_tags;
    material_tags.reserve(material_count);
    for( mi::Uint32 i = 0; i < material_count; ++i) {
        std::string db_material_name = add_mdl_db_prefix( code_dag->get_material_name( i));
        material_names.push_back( db_material_name);
        DB::Tag material_tag = transaction->name_to_tag( db_material_name.c_str());
        if( material_tag) {
            LOG::mod_log->error( M_SCENE, LOG::Mod_log::C_DATABASE,
               "DB name for material definition \"%s\" already in use.", db_material_name.c_str());
            return -3;
        }
        material_tags.push_back(transaction->reserve_tag());
    }

    if( !mdl->is_builtin_module( module_name)) {
        if( !module_filename)
            LOG::mod_log->info( M_SCENE, LOG::Mod_log::C_IO,
                "Loading module \"%s\".", module_name);
        else if( DETAIL::is_container_member( module_filename)) {
            const std::string& container_filename = DETAIL::get_container_filename( module_filename);
            LOG::mod_log->info( M_SCENE, LOG::Mod_log::C_IO,
                "Loading module \"%s\" from \"%s\".", module_name, container_filename.c_str());
        } else
            LOG::mod_log->info( M_SCENE, LOG::Mod_log::C_IO,
                "Loading module \"%s\" from \"%s\".", module_name, module_filename);
    }

    db_module_tag = transaction->reserve_tag();
    Mdl_module* db_module = new Mdl_module(transaction,
        mdl, module, code_dag.get(), imports, function_tags, material_tags, load_resources);

    DB::Privacy_level privacy_level = transaction->get_scope()->get_level();

    // Create DB elements for the function definitions in this module.
    for( mi::Uint32 i = 0; i < function_count; ++i) {
        Mdl_function_definition* db_function = new Mdl_function_definition(transaction,
            db_module_tag, function_tags[i], code_dag.get(), i, module_filename, module_name,
            load_resources);
        transaction->store_for_reference_counting(
            function_tags[i], db_function, function_names[i].c_str(), privacy_level);
    }

    // Create DB elements for the material definitions in this module.
    for( mi::Uint32 i = 0; i < material_count; ++i) {
        Mdl_material_definition* db_material = new Mdl_material_definition(transaction,
            db_module_tag, material_tags[i], code_dag.get(), i, module_filename, module_name,
            load_resources);
        transaction->store_for_reference_counting(
            material_tags[i], db_material, material_names[i].c_str(), privacy_level);
    }

    // Store the module in the DB.
    transaction->store(db_module_tag, db_module, db_module_name.c_str(), privacy_level);

    // Do not use the pointer to the DB element anymore after store().
    db_module = 0;
    if( module_tag)
        *module_tag = db_module_tag;
    return 0;
}

IValue_texture* Mdl_module::create_texture(
    DB::Transaction* transaction,
    const char* file_path,
    IType_texture::Shape shape,
    mi::Float32 gamma,
    bool shared,
    mi::Sint32* errors)
{
    mi::Sint32 dummy_errors = 0;
    if( !errors)
        errors = &dummy_errors;

    if( !transaction || !file_path) {
        *errors = -1;
        return 0;
    }

    if( file_path[0] != '/') {
        *errors = -2;
        return 0;
    }

    DB::Tag tag = DETAIL::mdl_texture_to_tag(
        transaction, file_path, /*module_filename*/ 0, /*module_name*/ 0, shared, gamma);
    if( !tag) {
        *errors = -3;
        return 0;
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
        return 0;
    }

    if( file_path[0] != '/') {
        *errors = -2;
        return 0;
    }

    DB::Tag tag = DETAIL::mdl_light_profile_to_tag(
        transaction, file_path, /*module_filename*/ 0, /*module_name*/ 0, shared);
    if( !tag) {
        *errors = -3;
        return 0;
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
        return 0;
    }

    if( file_path[0] != '/') {
        *errors = -2;
        return 0;
    }

    DB::Tag tag = DETAIL::mdl_bsdf_measurement_to_tag(
        transaction, file_path, /*module_filename*/ 0, /*module_name*/ 0, shared);
    if( !tag) {
        *errors = -3;
        return 0;
    }

    *errors = 0;
    mi::base::Handle<IValue_factory> vf( get_value_factory());
    return vf->create_bsdf_measurement( tag);
}

Mdl_module::Mdl_module()
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
    m_name( other.m_name),
    m_file_name( other.m_file_name),
    m_api_file_name( other.m_api_file_name),
    m_imports( other.m_imports),
    m_exported_types( other.m_exported_types),
    m_local_types(other.m_local_types),
    m_constants( other.m_constants),
    m_annotations( other.m_annotations),
    m_annotation_definitions( other.m_annotation_definitions),
    m_functions( other.m_functions),
    m_materials( other.m_materials),
    m_resource_reference_tags(other.m_resource_reference_tags)
{
}

Mdl_module::Mdl_module(
    DB::Transaction* transaction,
    mi::mdl::IMDL* mdl,
    const mi::mdl::IModule* module,
    mi::mdl::IGenerated_code_dag* code_dag,
    const std::vector<DB::Tag>& imports,
    const std::vector<DB::Tag>& functions,
    const std::vector<DB::Tag>& materials,
    bool load_resources) 
    : m_mdl(make_handle_dup(mdl)),
    m_module(make_handle_dup(module)),
    m_code_dag(make_handle_dup(code_dag)),
    m_tf(get_type_factory()),
    m_vf(get_value_factory()),
    m_ef(get_expression_factory()),
    m_imports(imports),
    m_exported_types(m_tf->create_type_list()),
    m_local_types(m_tf->create_type_list()),
    m_constants(m_vf->create_value_list()),
    m_annotation_definitions(m_ef->create_annotation_definition_list()),
    m_functions(functions),
    m_materials(materials)
{
    ASSERT( M_SCENE, mdl);
    ASSERT( M_SCENE, module);
    ASSERT( M_SCENE, module->get_name());
    ASSERT( M_SCENE, module->get_filename());

    m_name = module->get_name();
    m_file_name = module->get_filename();
    m_api_file_name = DETAIL::is_container_member( m_file_name.c_str())
        ? DETAIL::get_container_filename( m_file_name.c_str()) : m_file_name;

    // convert types
    mi::Uint32 type_count = code_dag->get_type_count();
    for( mi::Uint32 i = 0; i < type_count; ++i) {
        const char* name = code_dag->get_type_name( i);
        const mi::mdl::IType* type = code_dag->get_type( i);

        mi::Uint32 annotation_count = code_dag->get_type_annotation_count( i);
        Mdl_annotation_block annotations( annotation_count);
        for( mi::Uint32 k = 0; k < annotation_count; ++k)
            annotations[k] = code_dag->get_type_annotation( i, k);

        mi::Uint32 member_count = code_dag->get_type_sub_entity_count( i);
        Mdl_annotation_block_vector sub_annotations( member_count);
        for( mi::Uint32 j = 0; j < member_count; ++j) {
            annotation_count = code_dag->get_type_sub_entity_annotation_count( i, j);
            sub_annotations[j].resize( annotation_count);
            for( mi::Uint32 k = 0; k < annotation_count; ++k)
                sub_annotations[j][k] = code_dag->get_type_sub_entity_annotation( i, j, k);
        }

        mi::base::Handle<const IType> type_int(
            mdl_type_to_int_type( m_tf.get(), type, &annotations, &sub_annotations));
        std::string full_name = m_name + "::" + name;

        if( code_dag->is_type_exported( i))
            m_exported_types->add_type( full_name.c_str(), type_int.get());
        else
            m_local_types->add_type( full_name.c_str(), type_int.get());
    }

    // convert constants
    mi::Uint32 constant_count = code_dag->get_constant_count();
    for( mi::Uint32 i = 0; i < constant_count; ++i) {
        const char* name = code_dag->get_constant_name( i);
        const mi::mdl::DAG_constant* constant = code_dag->get_constant_value( i);
        const mi::mdl::IValue* value = constant->get_value();
        mi::base::Handle<IValue> value_int( mdl_value_to_int_value(
            m_vf.get(), transaction, /*type_int*/ 0, value, m_file_name.c_str(), m_name.c_str(), load_resources));
        std::string full_name = m_name + "::" + name;
        m_constants->add_value( full_name.c_str(), value_int.get());
    }

    Mdl_dag_converter anno_converter(
        m_ef.get(),
        transaction,
        /*immutable*/ true,
        /*create_direct_calls*/ false,
        m_file_name.c_str(),
        m_name.c_str(),
        /*prototype_tag*/ DB::Tag(),
        load_resources);

    // convert annotation definitions
    mi::Uint32 annotation_definition_count = code_dag->get_annotation_count();
    for (mi::Uint32 i = 0; i < annotation_definition_count; ++i) {

        const char* name = code_dag->get_annotation_name(i);

        // convert parameters
        mi::base::Handle<IType_list> parameter_types(m_tf->create_type_list());
        mi::base::Handle<IExpression_list> parameter_defaults(m_ef->create_expression_list());

        for (int p = 0, n = code_dag->get_annotation_parameter_count(i); p < n; ++p) {

            const char* parameter_name = code_dag->get_annotation_parameter_name(i, p);

            // convert types
            const mi::mdl::IType* parameter_type
                = code_dag->get_annotation_parameter_type(i, p);

            mi::base::Handle<const IType> type_int(
                mdl_type_to_int_type(m_tf.get(), parameter_type));
            parameter_types->add_type(parameter_name, type_int.get());

            // convert defaults
            const mi::mdl::DAG_node* default_
                = code_dag->get_annotation_parameter_default(i, p);
            if (default_) {
                mi::base::Handle<IExpression> default_int(anno_converter.mdl_dag_node_to_int_expr(
                    default_, type_int.get()));
                ASSERT(M_SCENE, default_int);
                parameter_defaults->add_expression(parameter_name, default_int.get());
            }
        }
        // convert annotations
        int annotation_annotation_count = code_dag->get_annotation_annotation_count(i);
        Mdl_annotation_block annotations(annotation_annotation_count);
        for (int a = 0; a < annotation_annotation_count; ++a) {
            annotations[a] = code_dag->get_annotation_annotation(i, a);
        }
        mi::base::Handle<IAnnotation_block> annotations_int(anno_converter.mdl_dag_node_vector_to_int_annotation_block(
            annotations, m_name.c_str()));

        bool is_exported = code_dag->get_annotation_property(i, mi::mdl::IGenerated_code_dag::AP_IS_EXPORTED);

        mi::neuraylib::IAnnotation_definition::Semantics sema = mdl_semantics_to_ext_annotation_semantics(
            code_dag->get_annotation_semantics(i));

        mi::base::Handle<IAnnotation_definition> anno_def(
            m_ef->create_annotation_definition(
                name,
                sema,
                is_exported,
                parameter_types.get(),
                parameter_defaults.get(),
                annotations_int.get()));

        m_annotation_definitions->add_definition(anno_def.get());
    }

    // convert module annotations
    mi::Uint32 annotation_count = code_dag->get_module_annotation_count();
    Mdl_annotation_block annotations( annotation_count);
    for( mi::Uint32 i = 0; i < annotation_count; ++i)
        annotations[i] = code_dag->get_module_annotation( i);
    m_annotations = anno_converter.mdl_dag_node_vector_to_int_annotation_block(
        annotations, m_name.c_str());

    if (module->get_referenced_resources_count() > 0) {
        std::map <std::string, mi::Size> resource_url_2_index;
        for (mi::Size i = 0; i < module->get_referenced_resources_count(); ++i)
            resource_url_2_index.insert(std::make_pair(module->get_referenced_resource_url(i), i));

        // update resource references
        std::set<const mi::mdl::IValue_resource*> resources;
        collect_resource_references(code_dag, resources);

        m_resource_reference_tags.resize(module->get_referenced_resources_count());
        for (const auto r : resources) {
            const char *key = r->get_string_value();
            const auto& it = resource_url_2_index.find(key);
            if (it != resource_url_2_index.end())
                m_resource_reference_tags[it->second].push_back(DB::Tag(r->get_tag_value()));
        }
    }
}

const char* Mdl_module::get_filename() const
{
    return m_file_name.empty() ? 0 : m_file_name.c_str();
}

const char* Mdl_module::get_api_filename() const
{
    return m_api_file_name.empty() ? 0 : m_api_file_name.c_str();
}

const char* Mdl_module::get_mdl_name() const
{
    return m_name.c_str();
}

mi::Size Mdl_module::get_import_count() const
{
    return m_imports.size();
}

DB::Tag Mdl_module::get_import( mi::Size index) const
{
    if( index >= m_imports.size())
        return DB::Tag( 0);
    return m_imports[index];
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
    return m_functions[index];
}

const char* Mdl_module::get_function_name(DB::Transaction* transaction, mi::Size index) const
{
    return index >= m_functions.size() ? 0 : transaction->tag_to_name(m_functions[index]);
}

mi::Size Mdl_module::get_material_count() const
{
    return m_materials.size();
}

DB::Tag Mdl_module::get_material(mi::Size index) const
{
    if( index >= m_materials.size())
        return DB::Tag( 0);
    return m_materials[index];
}

const IAnnotation_block* Mdl_module::get_annotations() const
{
   if( !m_annotations)
        return 0;
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
    return index >= m_materials.size() ? 0 : transaction->tag_to_name(m_materials[index]);
}

bool Mdl_module::is_standard_module() const
{
    return m_module->is_stdlib();
}

const std::vector<std::string> Mdl_module::get_function_overloads(
    DB::Transaction* transaction,
    const char* name,
    const IExpression_list* arguments) const
{
    std::vector<std::string> result;
    if( !name)
        return result;

    // compute prefix length (without signature)
    const char* pos = strchr( name,'(');
    size_t prefix_len = pos ? pos - name : strlen( name);

    // find overloads
    for( mi::Size i = 0; i < m_functions.size(); ++i) {

        const char* f = transaction->tag_to_name(m_functions[i]);
        if( strncmp( f, name, prefix_len) != 0)
            continue;

        const char next = f[prefix_len];
        if( next != '\0' && next != '(')
            continue;

        // no arguments provided, don't check for exact match
        if( !arguments) {
            result.push_back(f);
            continue;
        }
        // arguments provided, check for exact match
        DB::Tag tag = m_functions[i];
        ASSERT(M_SCENE, tag && transaction->get_class_id(tag) == Mdl_function_definition::id);

        DB::Access<Mdl_function_definition> definition( tag, transaction);
        mi::Sint32 errors = 0;
        // TODO check whether we can avoid the function call creation
        Mdl_function_call* call = definition->create_function_call(
            transaction, arguments, &errors);
        if (call && errors == 0) {
            result.push_back(f);
        }
        delete call;
    }

    return result;
}

const std::vector<std::string> Mdl_module::get_function_overloads_by_signature(
    DB::Transaction* transaction,
    const char* name,
    const char* param_sig) const
{
    std::vector<std::string> result;
    if( !name)
        return result;

    // reject names that do no start with the "mdl" prefix
    if( strncmp( name, "mdl", 3) != 0)
        return result;

    mi::base::Handle<const mi::mdl::IOverload_result_set> set(
        m_module->find_overload_by_signature( name + 3, param_sig));
    if( !set.is_valid_interface())
        return result;

    for( char const* name = set->first_signature(); name != NULL; name = set->next_signature())
        result.push_back( add_mdl_db_prefix( name));

    return result;
}

mi::Size Mdl_module::get_resources_count() const
{
    return m_module->get_referenced_resources_count();
}

const char* Mdl_module::get_resource_mdl_file_path(mi::Size index) const
{
    if (index >= m_module->get_referenced_resources_count())
        return nullptr;

    return m_module->get_referenced_resource_url(index);
}

DB::Tag Mdl_module::get_resource_tag(mi::Size index) const
{
    if (index >= m_resource_reference_tags.size())
        return DB::Tag(0);

    // for now, only give access to the first element
    if (m_resource_reference_tags[index].size() == 0)
        return DB::Tag(0);

    return m_resource_reference_tags[index][0];
}

const IType_resource* Mdl_module::get_resource_type(mi::Size index) const
{
    if (index >= m_module->get_referenced_resources_count())
        return nullptr;

    const mi::mdl::IType* t = m_module->get_referenced_resource_type(index);
    return mdl_type_to_int_type<IType_resource>(m_tf.get(), t);
}

const mi::mdl::IModule* Mdl_module::get_mdl_module() const
{
    m_module->retain();
    return m_module.get();
}

const mi::mdl::IGenerated_code_dag* Mdl_module::get_code_dag() const
{
    if( !m_code_dag.is_valid_interface())
        return 0;
    m_code_dag->retain();
    return m_code_dag.get();
}

bool Mdl_module::is_valid_module_name( const char* name, const mi::mdl::IMDL* mdl)
{
    if( !name)
        return false;
    if( name[0] == ':' && name[1] == ':') {
        // skip "::" scope
        name += 2;
    }

    for( ;;) {
        const char *scope = strstr( name, "::");

        std::string ident;
        if( scope)
            ident = std::string( name, scope - name);
        else
            ident = name;

        // the compiler checks an identifier only
        if( !mdl->is_valid_mdl_identifier( ident.c_str()))
            return false;

        if( scope)
            name = scope + 2;
        else
            break;
    }
    return true;
}

const SERIAL::Serializable* Mdl_module::serialize( SERIAL::Serializer* serializer) const
{
    Scene_element_base::serialize( serializer);

    // m_mdl is never serialized (independent of DB element)
    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module( false);
    mdlc_module->serialize_module( serializer, m_module.get());

    bool has_code = m_code_dag.is_valid_interface();
    serializer->write( has_code);
    if( has_code)
        mdlc_module->serialize_code_dag( serializer, m_code_dag.get());

    serializer->write( m_name);
    serializer->write( m_file_name);
    serializer->write( m_api_file_name);
    SERIAL::write( serializer, m_imports);
    m_tf->serialize_list( serializer, m_exported_types.get());
    m_tf->serialize_list( serializer, m_local_types.get());
    m_vf->serialize_list( serializer, m_constants.get());
    m_ef->serialize_annotation_block( serializer, m_annotations.get());
    m_ef->serialize_annotation_definition_list(serializer, m_annotation_definitions.get());
    SERIAL::write( serializer, m_functions);
    SERIAL::write( serializer, m_materials);
    SERIAL::write(serializer, m_resource_reference_tags);
    
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
    deserializer->read( &has_code);
    if( has_code)
        m_code_dag = mdlc_module->deserialize_code_dag( deserializer);

    deserializer->read( &m_name);
    deserializer->read( &m_file_name);
    deserializer->read( &m_api_file_name);
    SERIAL::read( deserializer, &m_imports);
    m_exported_types = m_tf->deserialize_list( deserializer);
    m_local_types = m_tf->deserialize_list( deserializer);
    m_constants = m_vf->deserialize_list( deserializer);
    m_annotations = m_ef->deserialize_annotation_block( deserializer);
    m_annotation_definitions = m_ef->deserialize_annotation_definition_list(deserializer);
    SERIAL::read( deserializer, &m_functions);
    SERIAL::read( deserializer, &m_materials);
    SERIAL::read( deserializer, &m_resource_reference_tags);

    return this + 1;
}

void Mdl_module::dump( DB::Transaction* transaction) const
{
    std::ostringstream s;
    s << std::boolalpha;
    mi::base::Handle<const mi::IString> tmp;

    // m_mdl, m_module, m_code_dag missing

    s << "Module MDL name: " << m_name << std::endl;
    s << "File name: " << m_file_name << std::endl;
    s << "API file name: " << m_api_file_name << std::endl;

    s << "Imports: ";
    mi::Size imports_count = m_imports.size();
    for( mi::Size i = 0; i+1 < imports_count; ++i)
        s << "tag " << m_imports[i].get_uint() << ", ";
    if( imports_count > 0)
        s << "tag " << m_imports[imports_count-1].get_uint();
    s << std::endl;

    tmp = m_tf->dump( m_exported_types.get());
    s << "Exported types: " << tmp->get_c_str() << std::endl;

    tmp = m_tf->dump( m_local_types.get());
    s << "Local types: " << tmp->get_c_str() << std::endl;

    tmp = m_vf->dump( transaction, m_constants.get(), /*name*/ 0);
    s << "Constants: " << tmp->get_c_str() << std::endl;

    // m_annotations, m_annotation_definitions, m_resource_references missing

    mi::Size function_count = m_functions.size();
    for( mi::Size i = 0; i < function_count; ++i)
        s << "Function definition " << i << ": " << m_functions[i] << std::endl;

    mi::Size material_count = m_materials.size();
    for( mi::Size i = 0; i < material_count; ++i)
        s << "Material definition " << i << ": " << m_materials[i] << std::endl;

    LOG::mod_log->info( M_SCENE, LOG::Mod_log::C_DATABASE, "%s", s.str().c_str());
}

size_t Mdl_module::get_size() const
{
    return sizeof( *this)
        + SCENE::Scene_element<Mdl_module, Mdl_module::id>::get_size()
            - sizeof( SCENE::Scene_element<Mdl_module, Mdl_module::id>)
        + dynamic_memory_consumption( m_name)
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
        + m_module->get_memory_size()
        + (m_code_dag ? m_code_dag->get_memory_size() : 0);
}

DB::Journal_type Mdl_module::get_journal_flags() const
{
    return DB::JOURNAL_NONE;
}

Uint Mdl_module::bundle( DB::Tag* results, Uint size) const
{
    return 0;
}

void Mdl_module::get_scene_element_references( DB::Tag_set* result) const
{
    result->insert(m_imports.begin(), m_imports.end());
    result->insert(m_functions.begin(), m_functions.end());
    result->insert(m_materials.begin(), m_materials.end());
    collect_references(m_annotations.get(), result);

    for (const auto& tags : m_resource_reference_tags)
    {
        for(const auto& tag: tags)
            if (tag.is_valid())
                result->insert(tag);
    }
}

} // namespace MDL

} // namespace MI

