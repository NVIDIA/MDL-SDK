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


#include "mdl_cache.h"
#include <mi/mdl_sdk.h>

#include "mdl_cache_package.h"
#include "mdl_cache_module.h"
#include "mdl_cache_material.h"
#include "mdl_cache_function.h"
#include "../index/index_cache_elements.h"
#include "../utilities/platform_helper.h"

using namespace tinyxml2;

Mdl_cache::Mdl_cache() :
    m_cache_root(nullptr),
    m_index(new Index_cache_elements()),
    m_locale("")
{
    m_cache_root = dynamic_cast<IMdl_cache_package*>(
        Mdl_cache::create(IMdl_cache_item::CK_PACKAGE, "::", "::", "::"));
}

Mdl_cache::~Mdl_cache()
{
    delete m_index;
    delete m_cache_root;
    // m_item_map stores weak pointers (no ownership)
}

IMdl_cache_item* Mdl_cache::create(
    IMdl_cache_item::Kind kind,
    const char* entity_name,
    const char* simple_name,
    const char* qualified_name)
{
    IMdl_cache_item* item = nullptr;
    switch (kind)
    {
        case IMdl_cache_item::CK_PACKAGE:
        {
            auto item_impl = new Mdl_cache_package();
            item_impl->initialize(this, entity_name, entity_name, qualified_name);
            item = item_impl;
            break;
        }

        case IMdl_cache_item::CK_MODULE:
        {
            auto item_impl = new Mdl_cache_module();
            item_impl->initialize(this, entity_name, entity_name, qualified_name);
            item = item_impl;
            break;
        }

        case IMdl_cache_item::CK_MATERIAL:
        {
            auto item_impl = new Mdl_cache_material();
            item_impl->initialize(this, entity_name, entity_name, qualified_name);
            item = item_impl;
            break;
        }

        case IMdl_cache_item::CK_FUNCTION:
        {
            auto item_impl = new Mdl_cache_function();
            item_impl->initialize(this, entity_name, entity_name, qualified_name);
            item = item_impl;
            break;
        }

        default:
            break;
    }

    m_item_map[{kind, qualified_name}] = item;
    return item;
}

bool Mdl_cache::erase(IMdl_cache_item* item)
{
    const auto found = m_item_map.find({item->get_kind(), item->get_qualified_name()});
    if (found == m_item_map.end())
    {
        std::cerr << "[Mdl_cache] erase: it to delete not found: " << item->get_qualified_name() 
                  << std::endl;
        return false;
    }

    m_item_map.erase(found);
    delete item;
    return true;
}


bool Mdl_cache::update(mi::neuraylib::INeuray* neuray, mi::neuraylib::ITransaction* transaction)
{
    // run discovery api
    Platform_helper::tic_toc_log("Discover Packages and Modules: ", [&]()
    {
        // keep a handle to the root to make sure we always deal with the structure
        // at a certain point in time
        const mi::base::Handle<const mi::neuraylib::IMdl_discovery_api> discover(
            neuray->get_api_component<const mi::neuraylib::IMdl_discovery_api>());
        m_discovery_result = mi::base::make_handle_dup(discover->discover(
            mi::neuraylib::IMdl_info::DK_PACKAGE | mi::neuraylib::IMdl_info::DK_MODULE));
    });

    bool updated = false;

    // update the cache nodes
    Platform_helper::tic_toc_log("Update Cache: ", [&]()
    {
        mi::base::Handle<const mi::neuraylib::IMdl_package_info> root(
            m_discovery_result->get_graph());

        updated = dynamic_cast<Mdl_cache_package*>(m_cache_root)->update(
            neuray, transaction, root.get());
    });

    // build up inverse index for searching
    Platform_helper::tic_toc_log("Update Index: ", [&]()
    {
        updated &= m_index->build(this);
    });

    return updated;
}

const IMdl_cache_package* Mdl_cache::get_cache_root() const
{
    return m_cache_root;
}

const mi::neuraylib::IMdl_discovery_result* Mdl_cache::get_discovery_result() const
{
    m_discovery_result->retain();
    return m_discovery_result.get();
}


const IMdl_cache_item* Mdl_cache::get_cache_item(const IMdl_cache_node::Child_map_key& key) const
{
    const auto found = m_item_map.find(key);
    if (found == m_item_map.end()) return nullptr;
    return found->second;
}

bool Mdl_cache::save_to_disk(const IMdl_cache_serializer& serializer, 
                             const std::string& path) const
{
    return serializer.serialize(this, path.c_str());
}

bool Mdl_cache::load_from_disk(const IMdl_cache_serializer& serializer, 
                               const std::string& path)
{
    IMdl_cache_item* deserialized = serializer.deserialize(this, path.c_str());

    if (!deserialized)
        return false;

    m_cache_root = dynamic_cast<Mdl_cache_package*>(deserialized);
    return m_cache_root != nullptr;
}


