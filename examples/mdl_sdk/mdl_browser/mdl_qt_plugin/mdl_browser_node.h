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

/// \file
/// \brief Main data structure of the browser to represent the package tree.


#ifndef MDL_SDK_EXAMPLES_MDL_BROWSER_MDL_BROWSER_NODE_H 
#define MDL_SDK_EXAMPLES_MDL_BROWSER_MDL_BROWSER_NODE_H

#include <map>
#include <string>

#include <mi/base/handle.h>
#include <functional>
#include "cache/imdl_cache.h"

namespace mi
{
    namespace base
    {
        class IInterface;
    }

    namespace neuraylib
    {
        class IMdl_discovery_result;
        class IMdl_package_info;
        class IMdl_module_info;
        class IMdl_info;
    }
}

class Mdl_browser_node;

class Mdl_browser_tree
{
public:
    typedef std::map<IMdl_cache_node::Child_map_key, Mdl_browser_node*,
                     IMdl_cache_node::Child_map_key_hash> Child_node_map;

    static Mdl_browser_tree* build(const mi::neuraylib::IMdl_discovery_result* discovery_result,
                                   const IMdl_cache_node* cache_root_node);
    virtual ~Mdl_browser_tree();

    // root that corresponds to the union of all mdl search paths
    Mdl_browser_node* get_root() { return m_root; }

    // called by the selection proxy model just before filtering is changed
    void reset_presentation_counters();

    // called by the selection proxy model during the filtering
    void increment_presentation_counter(const char* qualified_module_name);

    // called by the selection proxy model just after filtering has changed
    void gather_presentation_counters();


    // traverse and apply a function to each node
    static void traverse_down(Mdl_browser_node* traversal_root,
                              std::function<void(Mdl_browser_node*)>& action);

    // traverse and apply a function to each node
    static void traverse_up(Mdl_browser_node* traversal_root,
                            std::function<void(Mdl_browser_node*)>& action);

    // traverse and apply a function to each node
    static void traverse(Mdl_browser_node* traversal_root,
                         std::function<void(Mdl_browser_node*)>& action_down,
                         std::function<void(Mdl_browser_node*)>& action_up);

private:
    Mdl_browser_tree();
    Mdl_browser_node* m_root;
    Child_node_map m_map;
};


class Mdl_browser_node
{
    friend class Mdl_browser_tree;

public:

    virtual ~Mdl_browser_node();

    // simple name of the package or module
    const char* get_entity_name() const;

    // qualified name of the package or module
    const char* get_qualified_name() const;

    // true if this node represents a module rather than a package
    bool get_is_module() const;

    // get the search path this module was in
    std::string get_module_search_path() const;

    // get the number of shadows this module has
    mi::Size get_module_shadow_count() const;

    // get the search path of a shadow of this module
    std::string get_module_shadow_search_path(mi::Size shadow_index) const;

    // properties for presentation
    mi::Size get_presentation_count() const { return m_presentation_count; }

    // get child nodes
    const Mdl_browser_tree::Child_node_map& get_children() const { return m_children; }

    // get the cache node that corresponds to this browser node
    const IMdl_cache_node* get_cache_node() const { return m_cache_node; }


private:

    // called recursively starting from Mdl_browser_tree::build_tree(...)
    static Mdl_browser_node* build_tree(
        const mi::neuraylib::IMdl_package_info* discovery_root_package,
        const IMdl_cache_node* cache_root_node);

    // called during build_tree 
    explicit Mdl_browser_node(const mi::neuraylib::IMdl_info* discovery_info,
                              const IMdl_cache_node* cache_node);

    // MDL discovery
    mi::base::Handle<const mi::neuraylib::IMdl_info> m_discovery_info;

    // MDL cache
    const IMdl_cache_node* m_cache_node;

    // hierarchy
    Mdl_browser_tree::Child_node_map m_children;

    // presentation
    mi::Size m_presentation_count;
};

#endif