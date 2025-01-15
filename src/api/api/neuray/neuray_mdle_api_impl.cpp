/***************************************************************************************************
 * Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Source for the IMdle_api implementation.
 **/

#include "pch.h"

#include "neuray_mdle_api_impl.h"

#include <cassert>

#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>
#include <mi/neuraylib/iarray.h>
#include <mi/neuraylib/ibuffer.h>
#include <mi/neuraylib/istring.h>
#include <mi/neuraylib/istructure.h>
#include <mi/neuraylib/version.h>
#include <mi/mdl/mdl_archiver.h>
#include <mi/mdl/mdl_encapsulator.h>
#include <mi/mdl/mdl_module_transformer.h>
#include <mi/mdl/mdl_values.h>

#include <base/data/db/i_db_transaction.h>
#include <base/hal/hal/i_hal_ospath.h>
#include <base/hal/disk/disk_memory_reader_writer_impl.h>
#include <base/system/version/i_version.h>

#include <io/scene/bsdf_measurement/i_bsdf_measurement.h>
#include <io/scene/dbimage/i_dbimage.h>
#include <io/image/image/i_image.h>
#include <io/scene/lightprofile/i_lightprofile.h>
#include <io/scene/mdl_elements/i_mdl_elements_function_call.h>
#include <io/scene/mdl_elements/i_mdl_elements_function_definition.h>
#include <io/scene/mdl_elements/i_mdl_elements_module.h>
#include <io/scene/mdl_elements/i_mdl_elements_module_builder.h>
#include <io/scene/mdl_elements/i_mdl_elements_utilities.h>
#include <io/scene/mdl_elements/i_mdl_elements_expression.h>
#include <io/scene/mdl_elements/i_mdl_elements_type.h>
#include <io/scene/mdl_elements/i_mdl_elements_value.h>
#include <io/scene/texture/i_texture.h>

#include <mdl/integration/mdlnr/i_mdlnr.h>
#include <mdl/compiler/compilercore/compilercore_allocator.h>
#include <mdl/compiler/compilercore/compilercore_file_resolution.h>
#include <mdl/compiler/compilercore/compilercore_file_utils.h>
#include <mdl/compiler/compilercore/compilercore_zip_utils.h>
#include <mdl/compiler/compilercore/compilercore_encapsulator.h>

#include "neuray_transaction_impl.h"
#include "neuray_expression_impl.h"
#include "neuray_mdl_execution_context_impl.h"

namespace MI {
namespace NEURAY {


namespace {

mi::mdl::UDIM_mode get_uvtile_marker(const std::string& str)
{
    if (str.find("<UDIM>") != std::string::npos)
        return mi::mdl::UM_MARI;

    if (str.find("<UVTILE0>") != std::string::npos)
        return mi::mdl::UM_ZBRUSH;

    if (str.find("<UVTILE1>") != std::string::npos)
        return mi::mdl::UM_MUDBOX;

    return mi::mdl::NO_UDIM;
}

std::string replace_uvtile_base(
    const std::string& mdle_name_mask,      // e.g.     some_path_x/base_name_x.<UDIM>.png
    const std::string& resolved_file_name)  // e.g.     some_path_y/base_name_y.1001.png
                                            // returns  some_path_x/base_name_x.1001.png
{
    // get everything before the mask
    size_t p_base_name = mdle_name_mask.rfind('<');
    mi::mdl::UDIM_mode udim_mode = get_uvtile_marker(mdle_name_mask);
    std::string result;

    switch (udim_mode) {
        case mi::mdl::NO_UDIM:
            result = mdle_name_mask;
            break;
        case mi::mdl::UM_MARI:
        {
            // find second last dot to replace mask and extension
            size_t p = resolved_file_name.rfind('.');

            // if there is a point in front of the mask, we consider it part of the mask
            if (resolved_file_name.length() > 5 && resolved_file_name[p - 5] == '.') {
                p -= 1;
                p_base_name -= 1;
            }

            std::string resolved_mask_and_ext = resolved_file_name.substr(p - 4);

            // combine base name, pattern, extension
            result = mdle_name_mask.substr(0, p_base_name);
            result.append(resolved_mask_and_ext);
            break;
        }
        case mi::mdl::UM_ZBRUSH:
        case mi::mdl::UM_MUDBOX:
        {
            // find second last _ to replace mask and extension
            size_t p = resolved_file_name.rfind('_');
            p = resolved_file_name.rfind('_', p - 1);
            std::string resolved_mask_and_ext = resolved_file_name.substr(p);

            // combine base name, pattern, extension
            result = mdle_name_mask.substr(0, p_base_name);
            result.append(resolved_mask_and_ext);
            break;
        }
        default:
            assert(false);
    }
    return result;
}


} // anonymous


Mdle_resource_mapper::Mdle_resource_mapper(
    mi::mdl::IMDL* mdl,
    DB::Transaction* transaction,
    MDL::Execution_context* context)
    : Base(transaction, /*module*/ nullptr, "mdl::main", "main.mdl", context, /*result*/ nullptr)
    , m_mdl(mi::base::make_handle_dup(mdl))
    , m_resolver(m_mdl->get_entity_resolver(/*module_cache*/ nullptr))
    , m_context(context)
{
}

// Retrieve the "resource name" of an MDL resource value.
const char* Mdle_resource_mapper::get_resource_name(
    const mi::mdl::IValue_resource* v,
    bool support_strict_relative_path)
{
    // handle resources that have to be exported or copied here,
    // basically all resources that are not on disk and in a valid search path
    std::unordered_map<std::string, mi::base::Handle<mi::neuraylib::IBuffer>> in_memory_resources;
    Resource_callback::Buffer_callback callback = [&](
        mi::neuraylib::IBuffer* buffer,
        const char* suggested_file_name)
    {
        std::string name = suggested_file_name;
        size_t p = name.find_last_of("/\\:");
        ASSERT(M_NEURAY_API, p != std::string::npos);
        ASSERT(M_NEURAY_API, name.substr(p+1, 5) == "main_");
        name = name.substr(p + 6); // strip away the leading "<path>/main_"

        // store the buffer
        in_memory_resources[name] = mi::base::make_handle_dup(buffer);
        return "./" + name;
    };

    // Use the base class implementation the handle all the mappings and exporting of in-memory
    // resources. This also takes care of duplicated resources (the ones with an equal tag).
    const char* result = Base::get_resource_name(v, support_strict_relative_path, &callback);
    if (!result || strcmp(result, "") == 0)
    {
        std::string message = "Unable to resolve resource: " + std::string(v->get_string_value())
                            + " (Tag: " + std::to_string( v->get_tag_value()) + ").";
        add_error_message(m_context, message, -1);
        return "";
    }

    std::string resolved_name = result;

    // if this resource is already in the map, it can be skipped
    auto found = m_resource_names_resolved2mdle.find(resolved_name);
    if (found != m_resource_names_resolved2mdle.end()) {
        const char* result = found->second.c_str();
        return result;
    }

    // the resource was not handled yet
    // keep track of that resource in order to provide data to the MDLE write
    std::string mdle_name = generate_mdle_name(HAL::Ospath::basename(resolved_name));
    m_resource_names_resolved2mdle[resolved_name] = mdle_name;

    // keep a list of all files

    // handle in-memory resources
    if (!in_memory_resources.empty())
    {
        Resource_desc desc;
        desc.mdle_file_name_mask = mdle_name;
        desc.resolved_file_name = "";

        for (auto&& entry : in_memory_resources)
        {
            // use the uv-components (if there are any) of the resolved file
            desc.mdle_file_name = replace_uvtile_base(mdle_name, entry.first);

            // keep the buffer
            desc.in_memory_buffer = make_handle_dup(entry.second.get());
            m_resources.push_back(desc);
        }
    }

    // handle file-based resources
    else
    {
        // use the entity resolver to get the file path on disk
        mi::base::Handle<mi::mdl::IThread_context> ctx( MDL::create_thread_context( m_context));
        mi::base::Handle<mi::mdl::IMDL_resource_set> res_set(m_resolver->resolve_resource_file_name(
            resolved_name.c_str(),
            /*owner_file_path*/ nullptr,
            /*owner_name*/ nullptr,
            /*pos*/ nullptr,
            ctx.get()));

        for (size_t fi = 0, fi_n = res_set->get_count(); fi < fi_n; ++fi)
        {
            mi::base::Handle<mi::mdl::IMDL_resource_element const> elem(res_set->get_element(fi));
            for (size_t i = 0, el_n = elem->get_count(); i < el_n; ++i) {
                Resource_desc desc;
                desc.mdle_file_name_mask = mdle_name;
                desc.resolved_file_name = HAL::Ospath::normpath_v2(elem->get_filename(i));
                desc.in_memory_buffer = nullptr;

                // use the uv-components (if there are any) of the resolved file
                desc.mdle_file_name = replace_uvtile_base(mdle_name, desc.resolved_file_name);

                m_resources.push_back(desc);
            }
        }
    }
    // return the name that will be written into the exported module
    result = m_resources.back().mdle_file_name_mask.c_str();
    return result;
}

// Number of resources files to encapsulate.
size_t Mdle_resource_mapper::get_resource_count() const
{
    return m_resources.size();
}

// Get the resource path that should be used in the MDLE main module.
const char* Mdle_resource_mapper::get_mlde_resource_path(size_t index) const
{
    if (index >= m_resources.size())
        return nullptr;

    return m_resources[index].mdle_file_name.c_str();
}


// Get a stream reader interface that gives access to the requested resource data.
mi::mdl::IMDL_resource_reader* Mdle_resource_mapper::get_resource_reader(size_t index) const
{
    if (index >= m_resources.size())
        return nullptr;

    // in-memory
    if (m_resources[index].in_memory_buffer) {
        mi::base::Handle<mi::neuraylib::IReader> reader(
            new DISK::Memory_reader_impl( m_resources[index].in_memory_buffer.get()));
        return MDL::get_resource_reader(
            reader.get(), /*filepath*/ {}, /*filename*/ {}, /*hash*/ {});
    }

    // handle archives
    std::string fn = m_resources[index].resolved_file_name;
    auto p_mdr = fn.find(".mdr:");
    auto p_mdle = fn.find(".mdle:");
    if (p_mdr != std::string::npos/* && p_mdr != fn.size() - 5*/)
    {
        mi::base::Handle<mi::mdl::IArchive_tool> archive_tool(m_mdl->create_archive_tool());
        mi::base::Handle<mi::mdl::IInput_stream> input_stream(archive_tool->get_file_content(
            fn.substr(0, p_mdr + 4).c_str(), fn.substr(p_mdr + 5).c_str()));

        const mi::mdl::Messages& messages = archive_tool->access_messages();
        MDL::convert_and_log_messages(messages, /*context*/ nullptr);
        if (!input_stream || messages.get_error_message_count() > 0)
            return nullptr;

        return input_stream->get_interface<mi::mdl::IMDL_resource_reader>();
    }

    // handle mdle
    if (p_mdle != std::string::npos)
    {
        mi::base::Handle<mi::mdl::IEncapsulate_tool> mdle_tool(m_mdl->create_encapsulate_tool());
        mi::base::Handle<mi::mdl::IInput_stream> input_stream(mdle_tool->get_file_content(
            fn.substr(0, p_mdle + 5).c_str(), fn.substr(p_mdle + 6).c_str()));

        const mi::mdl::Messages& messages = mdle_tool->access_messages();
        MDL::convert_and_log_messages(messages, /*context*/ nullptr);
        if (!input_stream || messages.get_error_message_count() > 0)
            return nullptr;

        return input_stream->get_interface<mi::mdl::IMDL_resource_reader>();
    }

    // handle files on disk (and also already exported in-memory files)
    mi::mdl::MDL_zip_container_error_code  err;
    mi::mdl::File_handle* file = mi::mdl::File_handle::open(
        m_mdl->get_mdl_allocator(), fn.c_str(), err);

    if (file == nullptr || file->get_kind() != mi::mdl::File_handle::FH_FILE)
        return nullptr;

    mi::mdl::Allocator_builder builder(m_mdl->get_mdl_allocator());
    return builder.create<mi::mdl::File_resource_reader>(
        m_mdl->get_mdl_allocator(), file, fn.c_str(), "");
}

// Get a stream reader interface that gives access to the requested addition data file.
mi::mdl::IMDL_resource_reader* Mdle_resource_mapper::get_additional_data_reader(const char* path)
{
    // resolved / absolute file path
    mi::mdl::string absolute_path(m_mdl->get_mdl_allocator());

    // if the path is an absolute file system path use that
    if (mi::mdl::is_path_absolute(path)) {
        absolute_path = path;
    } else {
        // try to resolve from mdl search paths next
        mi::base::Handle<mi::mdl::IThread_context> ctx( MDL::create_thread_context( m_context));
        mi::base::Handle<mi::mdl::IMDL_resource_set> res_set(
            m_resolver->resolve_resource_file_name(
                path,
                /*owner_file_path*/ nullptr,
                /*owner_name*/ nullptr,
                /*pos*/ nullptr,
                ctx.get()));

        if (res_set && res_set->get_count() > 0) {
            mi::base::Handle<mi::mdl::IMDL_resource_element const> elem(res_set->get_element(0));
            absolute_path = elem->get_filename(0);
        }
    }

    // as fall-back, check the file system for relative paths
    if (absolute_path.empty()) {
        // add the current working directory
        absolute_path = mi::mdl::join_path(
            mi::mdl::get_cwd(m_mdl->get_mdl_allocator()),
            mi::mdl::convert_slashes_to_os_separators(absolute_path));
    }

    mi::mdl::MDL_zip_container_error_code  err;
    mi::mdl::File_handle* file = mi::mdl::File_handle::open(
        m_mdl->get_mdl_allocator(), absolute_path.c_str(), err);

    if (!file)
        return nullptr;

    mi::mdl::Allocator_builder builder(m_mdl->get_mdl_allocator());
    return builder.create<mi::mdl::File_resource_reader>(
        m_mdl->get_mdl_allocator(), file, path, "");
}

// Avoid file name collisions inside the MDLE.
std::string Mdle_resource_mapper::generate_mdle_name(const std::string& base_name)
{
    // use mask as part of the extension
    size_t pos_ext = base_name.rfind('<');
    if (pos_ext != std::string::npos)
        pos_ext--; // include the dot
    else
        pos_ext = base_name.rfind('.');

    // split
    std::string name = base_name.substr(0, pos_ext);
    std::string ext = base_name.substr(pos_ext);

    // find a unused name
    static int counter = 0;
    std::string n_test = "./resources/" + name;
    while (m_reserved_mdle_names.find(n_test) != m_reserved_mdle_names.end())
        n_test = "./resources/" + name + "_" + std::to_string( counter++);

    m_reserved_mdle_names.insert(n_test);
    return n_test + ext;
}


Mdle_api_impl::Mdle_api_impl(mi::neuraylib::INeuray* neuray)
: m_neuray(neuray)
, m_mdlc_module(true)
{
}

Mdle_api_impl::~Mdle_api_impl()
{
    m_neuray = nullptr;
}

namespace {

/// Helper class for dropping module imports at destruction.
class Drop_import_scope
{
public:
    Drop_import_scope(const mi::mdl::IModule* module)
    : m_module(module, mi::base::DUP_INTERFACE)
    {
    }

    ~Drop_import_scope() { m_module->drop_import_entries(); }

private:
    mi::base::Handle<const mi::mdl::IModule> m_module;
};

}  // anonymous

mi::Sint32 Mdle_api_impl::export_mdle(
    mi::neuraylib::ITransaction* transaction,
    const char* file_name,
    const mi::IStructure* mdle_data,
    mi::neuraylib::IMdl_execution_context* context) const
{
    MDL::Execution_context default_context;
    MDL::Execution_context* mdl_context = unwrap_and_clear_context(context, default_context);

    if (!file_name || !mdle_data || !transaction) {
        add_error_message(mdl_context, "Invalid parameters (NULL pointer).", -1);
        return -1;
    }

    auto* transaction_impl = static_cast<Transaction_impl*>(transaction);
    DB::Transaction* db_transaction = transaction_impl->get_db_transaction();
    mi::base::Handle<MDL::IValue_factory> vf( MDL::get_value_factory());
    mi::base::Handle<MDL::IExpression_factory> ef( MDL::get_expression_factory());

    // get the prototype
    mi::base::Handle<const mi::IString> prototype_name(
        mdle_data->get_value<mi::IString>("prototype_name"));
    if (!prototype_name) {
        add_error_message(mdl_context, "Invalid parameters (NULL pointer).", -1);
        return -1;
    }
    DB::Tag prototype_tag = db_transaction->name_to_tag(prototype_name->get_c_str());
    if (prototype_tag.is_invalid()) {
        add_error_message(
            mdl_context,
            "Prototype name cannot be found in database.", -1);
        return -1;
    }
    SERIAL::Class_id class_id = db_transaction->get_class_id(prototype_tag);
    if (class_id != MDL::ID_MDL_FUNCTION_DEFINITION &&
        class_id != MDL::ID_MDL_FUNCTION_CALL) {
        add_error_message(
            mdl_context,
            "Prototype is neither a function definition "
            "nor a function call.", -1);
        return -1;
    }
    mi::base::Handle<const mi::neuraylib::IExpression_list> defaults(
        mdle_data->get_value<mi::neuraylib::IExpression_list>("defaults"));
    mi::base::Handle<const MDL::IExpression_list> defaults_int(
        get_internal_expression_list(defaults.get()));

    mi::base::Handle<const mi::neuraylib::IAnnotation_block> annotations(
        mdle_data->get_value<mi::neuraylib::IAnnotation_block>("annotations"));

    mi::base::Handle<const MDL::IAnnotation_block> annotations_int(
        get_internal_annotation_block(annotations.get()));

    mi::base::Handle<mi::mdl::IMDL> imdl(m_mdlc_module->get_mdl());

    switch (class_id)
    {
        case MDL::ID_MDL_FUNCTION_CALL:
        {
            DB::Access<MDL::Mdl_function_call> func_call(prototype_tag, db_transaction);
            prototype_tag = func_call->get_function_definition(db_transaction);
            if (!prototype_tag.is_valid()) {
                add_error_message(
                    mdl_context,
                    "Function call is invalid.", -1);
                return -1;
            }

            DB::Access<MDL::Mdl_function_definition> func_def(prototype_tag, db_transaction);
            // skip presets
            DB::Tag def_prototype = func_def->get_prototype();
            if (def_prototype.is_valid())
                prototype_tag = def_prototype;

            if (!defaults_int)
                defaults_int = func_call->get_arguments();

            if (!annotations_int)
                annotations_int = func_def->get_annotations();
            break;
        }

        case MDL::ID_MDL_FUNCTION_DEFINITION:
        {
            DB::Access<MDL::Mdl_function_definition> func_def(prototype_tag, db_transaction);
            // skip presets
            DB::Tag def_prototype = func_def->get_prototype();
            if (def_prototype.is_valid())
                prototype_tag = def_prototype;

            if (!defaults_int)
                defaults_int = func_def->get_defaults();

            if (!annotations_int)
                annotations_int = func_def->get_annotations();
            break;
        }

        default:
            assert(false); // handled above
    }

    // collect files that will be included in the mdle
    std::vector<const char*> additional_file_source_paths;
    std::vector<const char*> additional_file_target_paths;


    // compute (filtered) annotations
    //---------------------------------------------------------------------------------------------
    bool has_origin = false;
    mi::Size n = annotations_int ? annotations_int->get_size() : 0;
    mi::base::Handle<MDL::IAnnotation_block> new_annotations_int(ef->create_annotation_block( n));
    for(size_t a = 0; a < n; ++a) {

        mi::base::Handle<const MDL::IAnnotation> anno(annotations_int->get_annotation(a));

        // skip hidden as we want the main material be visible in any case
        if (strcmp(anno->get_name(), "::anno::hidden()") == 0)
            continue;

        // has to be set manually on Mdle_data struct
        if (strcmp(anno->get_name(), "::anno::thumbnail(string)") == 0) {
            continue;
        }

        if (strcmp(anno->get_name(), "::anno::origin(string)") == 0) {
            has_origin = true;
        }

        new_annotations_int->add_annotation(anno.get());
    }
    if (!has_origin) {
        // add origin annotation
        std::string definiton_name = db_transaction->tag_to_name(prototype_tag);
        // strip mdl::/mdle:: prefix
        definiton_name = definiton_name.substr(definiton_name.find("::"));
        // strip signature
        definiton_name = definiton_name.substr(0, definiton_name.rfind('('));

        mi::base::Handle<MDL::IValue> anno_value(
            vf->create_string(definiton_name.c_str()));
        mi::base::Handle<MDL::IExpression> anno_expr(ef->create_constant(anno_value.get()));
        mi::base::Handle<MDL::IExpression_list> anno_expr_list(
            ef->create_expression_list( /*initial_capacity*/ 1));
        anno_expr_list->add_expression("name", anno_expr.get());

        mi::base::Handle<MDL::IAnnotation> anno(ef->create_annotation(
            db_transaction, "::anno::origin(string)", anno_expr_list.get()));
        new_annotations_int->add_annotation(anno.get());
    }

    // add thumbnail
    //---------------------------------------------------------------------------------------------

    // user specified thumbnail overrides the existing one
    mi::base::Handle<const mi::IString> thumbnail(
        mdle_data->get_value<mi::IString>("thumbnail_path"));

    std::string thumbnail_path;
    if (thumbnail && strcmp(thumbnail->get_c_str(), "") != 0) {
        thumbnail_path = thumbnail->get_c_str();

        // add the file to the list
        additional_file_source_paths.push_back(thumbnail_path.c_str());
        additional_file_target_paths.push_back("thumbnails/main.main.png");

        // create the thumbnail annotation
        mi::base::Handle<MDL::IValue> anno_value(
            vf->create_string(additional_file_target_paths.back()));
        mi::base::Handle<MDL::IExpression> anno_expr(ef->create_constant(anno_value.get()));
        mi::base::Handle<MDL::IExpression_list> anno_expr_list(
            ef->create_expression_list( /*initial_capacity*/ 1));
        anno_expr_list->add_expression("name", anno_expr.get());

        mi::base::Handle<MDL::IAnnotation> anno(ef->create_annotation(
            db_transaction, "::anno::thumbnail(string)", anno_expr_list.get()));
        new_annotations_int->add_annotation(anno.get());
    }

    // create a new module builder
    MDL::Mdl_module_builder builder(
        db_transaction,
        "mdl::tmp_mdle_export",
        /*min_mdl_version*/ mi::mdl::IMDL::MDL_LATEST_VERSION,
        /*max_mdl_version*/ mi::mdl::IMDL::MDL_LATEST_VERSION,
        /*export_to_db*/ false,
        mdl_context);
    if (mdl_context->get_error_messages_count() != 0)
        return -1;

    // add main material
    //---------------------------------------------------------------------------------------------
    mi::base::Handle<MDL::IAnnotation_block> empty_block(ef->create_annotation_block( 0));
    mi::base::Handle<MDL::IAnnotation_list> empty_list(ef->create_annotation_list( 0));
    builder.add_function(
        "main",                    // function/material name is "main"
        prototype_tag,             // stored in the database with this tag
        defaults_int.get(),        // use the following defaults
        empty_list.get(),          // delete all parameter annotations
        new_annotations_int.get(), // origin and thumbnail
        empty_block.get(),         // delete all return annotations
        /*is_exported*/ true,      // let the module export this function
        /*is_declarative*/false,   // no need to enforce the declarative flag
        MDL::IType::MK_NONE,
        mdl_context);
    if (mdl_context->get_error_messages_count() != 0)
        return -1;

    mi::base::Handle<const mi::mdl::IModule> tmp_module(builder.get_module());
    ASSERT( M_NEURAY_API, tmp_module);

    // in-line the temp-module
    //---------------------------------------------------------------------------------------------
    mi::base::Handle<mi::mdl::IMDL_module_transformer> module_transformer(
        imdl->create_module_transformer());
    mi::base::Handle<mi::mdl::IModule const> inlined_module;

    {
        MDL::Module_cache module_cache(db_transaction, m_mdlc_module->get_module_wait_queue(), {});
        if (!tmp_module->restore_import_entries(&module_cache)) {
            add_error_message(mdl_context, "Restore import entries failed.", -1);
            return -1;
        }
        Drop_import_scope scope(tmp_module.get());

        inlined_module = mi::base::make_handle(
            module_transformer->inline_imports(tmp_module.get()));

        if (!inlined_module.is_valid_interface()) {

            MDL::convert_and_log_messages(module_transformer->access_messages(), mdl_context);
            return -1;
        }
    }

    // add user files
    //---------------------------------------------------------------------------------------------
    mi::base::Handle<const mi::IArray> user_files(mdle_data->get_value<mi::IArray>("user_files"));

    if (user_files.is_valid_interface()) {

        size_t n = user_files->get_length();
        for (mi::Size i = 0; i < n; ++i) {

            mi::base::Handle<const mi::IStructure> user_file(
                user_files->get_element<mi::IStructure>(i));

            mi::base::Handle<const mi::IString> source_path(
                user_file->get_value<mi::IString>("source_path"));
            additional_file_source_paths.push_back(source_path->get_c_str());

            mi::base::Handle<const mi::IString> target_path(
                user_file->get_value<mi::IString>("target_path"));
            additional_file_target_paths.push_back(target_path->get_c_str());
        }
    }

    // encapsulate module
    //---------------------------------------------------------------------------------------------
    mi::base::Handle<mi::mdl::IEncapsulate_tool> encapsulator(
        imdl->create_encapsulate_tool());
    encapsulator->access_options().set_option(MDL_ENCAPS_OPTION_OVERWRITE, "true");

    // setup MDLe export
    mi::mdl::IEncapsulate_tool::Mdle_export_description desc;

    // resource handler
    Mdle_resource_mapper resource_collector(imdl.get(), db_transaction, mdl_context);
    desc.resource_callback = &resource_collector;
    desc.resource_collector = &resource_collector; // implements both interfaces
    desc.additional_file_source_paths = additional_file_source_paths.data();
    desc.additional_file_target_paths = additional_file_target_paths.data();
    desc.additional_file_count = additional_file_source_paths.size();
    MDL_ASSERT(additional_file_source_paths.size() == additional_file_target_paths.size());

    // add authoring tool
    // add zip file comments
    std::string author("MDL SDK ");
    author.append(MI_NEURAYLIB_PRODUCT_VERSION_STRING);
    author.append(" (build ");
    author.append(VERSION::get_platform_version());
    author.append(", ");
    author.append(VERSION::get_platform_date());
    author.append(", ");
    author.append(VERSION::get_platform_os());
    author.append(")");
    desc.authoring_tool_name_and_version = author.c_str();

    // write module to disk
    std::string file, dir;
    HAL::Ospath::split(file_name, dir, file);

    encapsulator->create_encapsulated_module(
        inlined_module.get(),
        file.substr(0, file.size() - 5).c_str(), // encapsulator adds extension (todo: fix)
        dir.c_str(),
        desc);

    MDL::convert_and_log_messages(encapsulator->access_messages(), mdl_context);

    return mdl_context->get_error_messages_count() == 0 ? 0 : -1;
}

// Checks the integrity of an MDLE based on MD5 hashes that are stored for the contained files.
mi::Sint32 Mdle_api_impl::validate_mdle(
    const char* file_name,
    mi::neuraylib::IMdl_execution_context* context) const
{
    MDL::Execution_context default_context;
    MDL::Execution_context* mdl_context = unwrap_and_clear_context(context, default_context);

    mi::base::Handle<mi::mdl::IMDL> imdl(m_mdlc_module->get_mdl());
    mi::base::Handle<mi::mdl::IEncapsulate_tool> encapsulator(imdl->create_encapsulate_tool());
    bool valid = encapsulator->check_integrity(file_name);

    MDL::convert_and_log_messages(encapsulator->access_messages(), mdl_context);
    MDL_ASSERT(valid == (mdl_context->get_error_messages_count() == 0));
    return valid ? 0 : -1;
}

mi::neuraylib::IReader* Mdle_api_impl::get_user_file(
    const char* mlde_file_name,
    const char* user_file_name,
    mi::neuraylib::IMdl_execution_context* context) const
{
    MDL::Execution_context default_context;
    MDL::Execution_context* mdl_context = unwrap_and_clear_context(context, default_context);

    if (!mlde_file_name || !user_file_name)
    {
        add_error_message(
            mdl_context,
            "Failed to get user file because of invalid parameter have been provided.", -1);
        return nullptr;
    }

    // try to read the file
    mi::base::Handle<mi::mdl::IMDL> imdl(m_mdlc_module->get_mdl());
    mi::base::Handle<mi::mdl::IEncapsulate_tool> encapsulator(imdl->create_encapsulate_tool());
    mi::base::Handle<mi::mdl::IInput_stream> reader(
        encapsulator->get_file_content(mlde_file_name, user_file_name));

    // check for errors
    MDL::convert_and_log_messages(encapsulator->access_messages(), mdl_context);
    if (!reader && mdl_context->get_error_messages_count() != 0)
        return nullptr;

    // wrap in a IMDL_resource_reader
    mi::base::Handle<mi::mdl::IMDL_resource_reader> file_random_access(
        reader->get_interface<mi::mdl::IMDL_resource_reader>());
    ASSERT(M_NEURAY_API, file_random_access.get());
    return MDL::get_reader(file_random_access.get());
}

mi::Sint32 Mdle_api_impl::compare_mdle(
    const char* mdle_file_name_a,
    const char* mdle_file_name_b,
    mi::neuraylib::IMdl_execution_context* context) const
{
    MDL::Execution_context default_context;
    MDL::Execution_context* mdl_context = unwrap_and_clear_context(context, default_context);
    mi::base::Handle<mi::mdl::IMDL> imdl(m_mdlc_module->get_mdl());
    mi::base::Handle<mi::mdl::IEncapsulate_tool> encapsulator(imdl->create_encapsulate_tool());
    mi::mdl::MDL_zip_container_mdle* mdle_a = nullptr;
    mi::mdl::MDL_zip_container_mdle* mdle_b = nullptr;

    auto after_cleanup = [&](bool result)
    {
        if (mdle_a) mdle_a->close();
        if (mdle_b) mdle_b->close();

        MDL::convert_and_log_messages(encapsulator->access_messages(), mdl_context);
        return (result && mdl_context->get_error_messages_count() == 0) ? 0 : -1;
    };

    // open mdle files
    mdle_a = encapsulator->open_encapsulated_module(mdle_file_name_a);
    if (!mdle_a) return after_cleanup(false);

    mdle_b = encapsulator->open_encapsulated_module(mdle_file_name_b);
    if (!mdle_b) return after_cleanup(false);

    unsigned char hash_a[16];
    if (!mdle_a->get_hash(hash_a))
    {
        mi::mdl::string message("MDLE comparison failed.", imdl->get_mdl_allocator());
        message.append(" Failed to get hash of '");
        message.append(mdle_file_name_a);
        message.append("'.");
        add_error_message(mdl_context, message.c_str(), -1);
        return after_cleanup(false);
    }
    unsigned char hash_b[16];
    if (!mdle_b->get_hash(hash_b))
    {
        mi::mdl::string message("MDLE comparison failed.", imdl->get_mdl_allocator());
        message.append(" Failed to get hash of '");
        message.append(mdle_file_name_b);
        message.append("'.");
        add_error_message(mdl_context, message.c_str(), -1);
        return after_cleanup(false);
    }

    // compare the hashes
    for (size_t i = 0; i < 16; ++i)
    {
        if (hash_a[i] != hash_b[i])
        {
            return after_cleanup(false);
        }
    }

    return after_cleanup(true);
}

mi::Sint32 Mdle_api_impl::get_hash(
    const char* mdle_file_name,
    mi::base::Uuid& hash,
    mi::neuraylib::IMdl_execution_context* context) const
{
    MDL::Execution_context default_context;
    MDL::Execution_context* mdl_context = unwrap_and_clear_context(context, default_context);

    mi::base::Handle<mi::mdl::IMDL> imdl(m_mdlc_module->get_mdl());
    mi::base::Handle<mi::mdl::IEncapsulate_tool> encapsulator(imdl->create_encapsulate_tool());
    mi::mdl::MDL_zip_container_mdle* mdle = nullptr;

    // reset
    hash.m_id1 = 0;
    hash.m_id2 = 0;
    hash.m_id3 = 0;
    hash.m_id4 = 0;

    // open mdle file
    mdle = encapsulator->open_encapsulated_module(mdle_file_name);
    if (!mdle) {
        MDL::convert_and_log_messages(encapsulator->access_messages(), mdl_context);
        return -1;
    }

    // get hash
    unsigned char h[16];
    if (!mdle->get_hash(h))
    {
        mi::mdl::string message("Failed to get hash of '", imdl->get_mdl_allocator());
        message.append(mdle_file_name);
        message.append("'.");
        add_error_message(mdl_context, message.c_str(), -1);
        mdle->close();

        return -1;
    }
    mdle->close();

    hash = MDL::convert_hash(h);

    return 0;
}

mi::Sint32 Mdle_api_impl::start()
{
    m_mdlc_module.set();
    return 0;
}

mi::Sint32 Mdle_api_impl::shutdown()
{
    m_mdlc_module.reset();
    return 0;
}

} // namespace NEURAY
} // namespace MI
