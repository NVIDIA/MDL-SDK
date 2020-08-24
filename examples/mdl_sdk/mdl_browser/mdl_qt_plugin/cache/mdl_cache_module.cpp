/******************************************************************************
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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
 *****************************************************************************/


#include "mdl_cache_module.h"
#include <mi/mdl_sdk.h>
#include <iostream>
#include "mdl_cache_material.h"
#include "mdl_cache_function.h"
#include "../utilities/platform_helper.h"
#include <set>


bool Mdl_cache_module::update(mi::neuraylib::INeuray* neuray, 
                              mi::neuraylib::ITransaction* transaction, 
                              const mi::base::IInterface* node)
{
    const mi::base::Handle<const mi::neuraylib::IMdl_module_info> module_info(
        node->get_interface<const mi::neuraylib::IMdl_module_info>());
    if (!module_info)
        return false;

    // get the resolved path of the file this module is defined in
    // in case of modules in archives, we need the file path of the archive
    const mi::base::Handle<const mi::IString> resolved_path(module_info->get_resolved_path());
    std::string current_path = resolved_path->get_c_str();
    if(module_info->in_archive())
    {
        const size_t pos = current_path.find(".mdr:");
        current_path = current_path.substr(0, pos + 4);
    }

    // if the search path is one we stored in the cache, we check the date
    const mi::Uint64 timestamp = static_cast<time_t>(
        Platform_helper::get_file_change_time(current_path));

    // maybe, this module is from another search path
    const char* cached_path = get_file_path();
    if(cached_path && strcmp(current_path.c_str(), cached_path) == 0)
    {
        // return true here if the file has not changed
        if (timestamp > 0 && timestamp <= get_timestamp())
        {
            /*
            std::cerr << "[Mdl_cache_module] update: skipped unchanged module: "
                      << get_entity_name() << "\n";
            */
            return true;
        }
    }

    // load the selected module
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
        neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());

    if (!mdl_impexp_api || mdl_impexp_api->load_module(transaction, module_info->get_qualified_name()) < 0)
    {
        std::cerr << "[Mdl_cache_module] update: Failed to load module: " 
                  << get_entity_name() << "\n";
        return false;
    }

    mi::base::Handle<const mi::neuraylib::IModule> mdl_module(
        transaction->access<mi::neuraylib::IModule>(
        (std::string("mdl") + module_info->get_qualified_name()).c_str()));

    if (!mdl_module)
    {
        std::cerr << "[Mdl_cache_module] update: Failed to load module: " 
                  << get_entity_name() << "\n";
        return false;
    }

    // get infos from annotations
    const mi::base::Handle<const mi::neuraylib::IAnnotation_block> anno_block(
        mdl_module->get_annotations());

    // are there annotations?
    if (anno_block)
    {
        const mi::neuraylib::Annotation_wrapper annotations(anno_block.get());

        if (annotations.get_annotation_index("::anno::hidden()") != static_cast<mi::Size>(-1))
            set_is_hidden(true);
    }

    // keep track of the existence of children to allow clean-up
    std::set<Child_map_key> not_present_children;
    for (const auto& c : get_children())
        not_present_children.insert(c.first);

    bool success = true;

    // Iterate over all materials exported by the module.
    for (mi::Size i = 0, n = mdl_module->get_material_count(); i < n; ++i)
    {
        mi::base::Handle<const mi::neuraylib::IMaterial_definition> material_definition(
            transaction->access<mi::neuraylib::IMaterial_definition>(mdl_module->get_material(i)));
        std::string simple_name = material_definition->get_mdl_simple_name();
        std::string qualified_name = material_definition->get_mdl_name();
        const Child_map_key key{IMdl_cache_item::CK_MATERIAL, simple_name};

        // "mark" as present by removing from not_present_children
        const auto& pos = not_present_children.find(key);
        if (pos != not_present_children.end())
            not_present_children.erase(pos);

        // insert new child if not already available
        IMdl_cache_item* child = get_child(key);
        if (!child)
        {
            child = get_cache()->create(
                CK_MATERIAL, simple_name.c_str(), simple_name.c_str(), qualified_name.c_str());
            add_child(child);
        }

        // update recursively
        if (!dynamic_cast<Mdl_cache_material*>(child)->update(
            neuray, transaction, mdl_module.get()))
        {
            std::cerr << "[Mdl_cache_module] update: Failed to update material: " 
                      << get_qualified_name() << "\n";
            success = false;
        }

        // pass down the hidden property
        if (get_is_hidden())
            dynamic_cast<Mdl_cache_material*>(child)->set_is_hidden(true);
    }

    // Iterate over all materials exported by the module.
    for (mi::Size i = 0, n = mdl_module->get_function_count(); i < n; ++i)
    {
        mi::base::Handle<const mi::neuraylib::IFunction_definition> function_definition(
            transaction->access<mi::neuraylib::IFunction_definition>(mdl_module->get_function(i)));
        std::string simple_name = function_definition->get_mdl_simple_name();
        std::string qualified_name = function_definition->get_mdl_name();
        const Child_map_key key{IMdl_cache_item::CK_FUNCTION, simple_name};

        // compute entity name by adding the parameter types
        std::string entity_name = simple_name + '(';
        mi::Size j = 0;
        while (true) {
            const char* type_name = function_definition->get_mdl_parameter_type_name(j++);
            if (!type_name)
                break;
            if (j > 1)
                entity_name += ',';
            entity_name += type_name;
        }
        entity_name += ')';

        // "mark" as present by removing from not_present_children
        const auto& pos = not_present_children.find(key);
        if (pos != not_present_children.end())
            not_present_children.erase(pos);

        // insert new child if not already available
        IMdl_cache_item* child = get_child(key);
        if (!child)
        {
            child = get_cache()->create(
                CK_FUNCTION, entity_name.c_str(), simple_name.c_str(), qualified_name.c_str());
            add_child(child);
        }

        // update recursively
        if (!dynamic_cast<Mdl_cache_function*>(child)->update(
            neuray, transaction, mdl_module.get()))
        {
            std::cerr << "[Mdl_cache_module] update: Failed to update function: " 
                      << get_qualified_name() << "\n";
            success = false;
        }

        // pass down the hidden property
        if (get_is_hidden())
            dynamic_cast<Mdl_cache_function*>(child)->set_is_hidden(true);
    }

    // remove all cached modules that are not present anymore
    for (const auto& c : not_present_children)
    {
        IMdl_cache_item* item = remove_child(c);
        get_cache()->erase(item);
    }
        
    // keep the timestamp if everything went fine
    set_timestamp(success ? timestamp : 0);
    set_file_path(current_path.c_str()); // store the search path, too
    set_located_in_archive(module_info->in_archive());

    return success;
}

bool Mdl_cache_module::get_located_in_archive() const
{
    const char* value = Base::get_cache_data("LocatedInArchive");
    return value && strcmp(value, "true") == 0;
}

void Mdl_cache_module::set_located_in_archive(bool value)
{
    Base::set_cache_data("LocatedInArchive", value ? "true" : "false");
}

const char* Mdl_cache_module::get_file_path() const
{
    return Base::get_cache_data("FilePath");
}

void Mdl_cache_module::set_file_path(const char* search_path)
{
    Base::set_cache_data("FilePath", search_path);
}
