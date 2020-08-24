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


#include "mdl_browser_node.h"
#include <mi/mdl_sdk.h>
#include <iostream>
#include "cache/mdl_cache.h"

Mdl_browser_tree::Mdl_browser_tree() :
    m_root(nullptr)
{
}

Mdl_browser_tree::~Mdl_browser_tree()
{
    delete m_root;
    m_map.clear();
}

Mdl_browser_tree* Mdl_browser_tree::build(
    const mi::neuraylib::IMdl_discovery_result* discovery_result,
    const IMdl_cache_node* cache_root_node)
{
    Mdl_browser_tree* tree = new Mdl_browser_tree();
    
    // build tree
    mi::base::Handle<const mi::neuraylib::IMdl_package_info> discovery_root_package(
        discovery_result->get_graph());

    tree->m_root = Mdl_browser_node::build_tree(discovery_root_package.get(), cache_root_node);

    // build map and collect some statistics
    mi::Size package_count = 0;
    mi::Size module_count = 0;
    mi::Size material_count = 0;
    mi::Size function_count = 0;
    Mdl_browser_tree::Child_node_map& map = tree->m_map;
    std::function<void(Mdl_browser_node*)> action = [&](Mdl_browser_node* node)
    {
        const IMdl_cache_node::Child_map_key key
        {
            node->get_cache_node()->get_kind(),
            node->get_qualified_name()
        };
        map[key] = node;

        // statistics 
        if(node->get_is_module())
        {
            module_count++;
            const auto cache_node = node->get_cache_node();
            for (mi::Size i = 0, n = cache_node->get_child_count(); i<n; ++i)
            {
                const auto child = cache_node->get_child(*cache_node->get_child_key(i));
                if (child->get_kind() == IMdl_cache_item::CK_MATERIAL)
                    material_count++;
                else if (child->get_kind() == IMdl_cache_item::CK_FUNCTION)
                    function_count++;
            }
        }
        else
            package_count++;

    };
    traverse_down(tree->m_root, action);

    std::cerr << "[INFO] Mdl_browser_tree::build: package count: " << package_count << "\n";
    std::cerr << "[INFO] Mdl_browser_tree::build: module count: " << module_count << "\n";
    std::cerr << "[INFO] Mdl_browser_tree::build: material count: " << material_count << "\n";
    std::cerr << "[INFO] Mdl_browser_tree::build: function count: " << function_count << "\n";


    return tree;
}

void Mdl_browser_tree::reset_presentation_counters()
{
    for (auto& kv : m_map)
        kv.second->m_presentation_count = 0;
}

void Mdl_browser_tree::gather_presentation_counters()
{
    std::function<void(Mdl_browser_node*)> action = [](Mdl_browser_node* node)
    {
        mi::Size count = 0;
        for (const auto& kv : node->m_children)
            count += kv.second->m_presentation_count;

        node->m_presentation_count += count;
    };
    traverse_up(m_root, action);
}

void Mdl_browser_tree::increment_presentation_counter(const char* qualified_module_name)
{
    // only increment counter of modules, packages are collected by the gather traversal
    const auto it = m_map.find({IMdl_cache_item::CK_MODULE, qualified_module_name});
    if (it != m_map.end())
        it->second->m_presentation_count++;
}


void Mdl_browser_tree::traverse_down(Mdl_browser_node* traversal_root,
                                     std::function<void(Mdl_browser_node*)>& action)
{
    action(traversal_root);

    for (auto& kv : traversal_root->m_children)
        traverse_down(kv.second, action);
}

void Mdl_browser_tree::traverse_up(Mdl_browser_node* traversal_root,
                                   std::function<void(Mdl_browser_node*)>& action)
{
    for (auto& kv : traversal_root->m_children)
        traverse_up(kv.second, action);

    action(traversal_root);
}

void Mdl_browser_tree::traverse(Mdl_browser_node* traversal_root,
                                std::function<void(Mdl_browser_node*)>& action_down,
                                std::function<void(Mdl_browser_node*)>& action_up)
{
    action_down(traversal_root);

    for (auto& kv : traversal_root->m_children)
        traverse(kv.second, action_down, action_up);

    action_up(traversal_root);
}



Mdl_browser_node* Mdl_browser_node::build_tree(
    const mi::neuraylib::IMdl_package_info* discovery_root_package, 
    const IMdl_cache_node* cache_root_node)
{
    // create the current node
    Mdl_browser_node* node = new Mdl_browser_node(discovery_root_package, cache_root_node);

    // iterate over children 
    for (mi::Size c = 0, n = discovery_root_package->get_child_count(); c < n; c++)
    {
        // discovery child
        const mi::base::Handle<const mi::neuraylib::IMdl_info> c_discovery(
            discovery_root_package->get_child(c));

        switch (c_discovery->get_kind())
        {
            case mi::neuraylib::IMdl_info::DK_PACKAGE:
            {
                const mi::base::Handle<const mi::neuraylib::IMdl_package_info> c_package(
                    c_discovery.get_interface<const mi::neuraylib::IMdl_package_info>());

                const std::string c_name = c_package->get_simple_name();
                const IMdl_cache_node::Child_map_key key{IMdl_cache_item::CK_PACKAGE, c_name};

                // cache child with same name
                const IMdl_cache_item* c_cache = cache_root_node->get_child(key);
                const auto c_cache_package = dynamic_cast<const IMdl_cache_package*>(c_cache);
                if (!c_cache_package)
                {
                    std::cerr << "[Mdl_browser_node] build_tree: Cache not found for package: "
                              << c_package->get_simple_name() << ".\n";
                    continue; // do not add this node
                }

                // build hierarchy recursively
                Mdl_browser_node* c_node = build_tree(c_package.get(), c_cache_package);
                node->m_children[key] = c_node;
                break;
            }

            case mi::neuraylib::IMdl_info::DK_MODULE:
            {
                const mi::base::Handle<const mi::neuraylib::IMdl_module_info> c_module(
                    c_discovery.get_interface<const mi::neuraylib::IMdl_module_info>());

                const std::string c_name = c_module->get_simple_name();
                const IMdl_cache_node::Child_map_key key{IMdl_cache_item::CK_MODULE, c_name};

                // cache child with same name
                const IMdl_cache_item* c_cache = cache_root_node->get_child(key);
                const auto c_cache_module = dynamic_cast<const IMdl_cache_module*>(c_cache);
                if (!c_cache_module)
                {
                    std::cerr << "[Mdl_browser_node] build_tree: Cache not found for module: "
                        << c_module->get_simple_name() << ".\n";
                    continue; // do not add this node
                }

                // add the leafs of this tree
                Mdl_browser_node* c_node = new Mdl_browser_node(c_module.get(), c_cache_module);
                node->m_children[key] = c_node;
                break;
            }

            default:
                std::cerr << "[Mdl_browser_node] build_tree: discovery info kind no handled: "
                          << c_discovery->get_simple_name() << ".\n";
                break;
        }
    }

    return node;
}

Mdl_browser_node::Mdl_browser_node(const mi::neuraylib::IMdl_info* discovery_info,
                           const IMdl_cache_node* cache_node) :
    m_discovery_info(mi::base::make_handle_dup(discovery_info)),
    m_cache_node(cache_node),
    m_presentation_count(0)
{
}

Mdl_browser_node::~Mdl_browser_node()
{
    for (auto& kv : m_children)
        delete kv.second;

    m_children.clear();
}

const char* Mdl_browser_node::get_entity_name() const
{
    return m_cache_node->get_entity_name();
}

const char* Mdl_browser_node::get_qualified_name() const
{
    return m_cache_node->get_qualified_name();
}

bool Mdl_browser_node::get_is_module() const
{
    return m_discovery_info->get_kind() == mi::neuraylib::IMdl_info::DK_MODULE;
}

mi::Size Mdl_browser_node::get_module_shadow_count() const
{
    const mi::base::Handle<const mi::neuraylib::IMdl_module_info> module_info(
        m_discovery_info.get_interface<const mi::neuraylib::IMdl_module_info>());

    // if this is not a module, it has no shadows
    return module_info.is_valid_interface() ? module_info->get_shadows_count() : 0;
}


std::string Mdl_browser_node::get_module_search_path() const
{
    // current module
    const mi::base::Handle<const mi::neuraylib::IMdl_module_info> module_info(
        m_discovery_info.get_interface<const mi::neuraylib::IMdl_module_info>());

    // if this is not a module, it has no shadows
    return module_info.is_valid_interface() ? module_info->get_search_path() : "";
}


std::string Mdl_browser_node::get_module_shadow_search_path(mi::Size shadow_index) const
{
    // current module
    const mi::base::Handle<const mi::neuraylib::IMdl_module_info> module_info(
        m_discovery_info.get_interface<const mi::neuraylib::IMdl_module_info>());

    // if this is not a module, it has no shadows
    if (!module_info.is_valid_interface()) return "";

    const mi::base::Handle<const mi::neuraylib::IMdl_module_info> shadow_module_info(
        module_info->get_shadow(shadow_index));

    return shadow_module_info->get_search_path();
}
