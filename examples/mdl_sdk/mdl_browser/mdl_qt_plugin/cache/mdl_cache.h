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
/// \brief Implementation of the IMdl_cache interface.


#ifndef MDL_SDK_EXAMPLES_MDL_BROWSER_MDL_CACHE_H
#define MDL_SDK_EXAMPLES_MDL_BROWSER_MDL_CACHE_H

#include "example_shared.h"
#include "imdl_cache.h"

#include <string>
#include <unordered_map>

namespace mi
{
    namespace base
    {
        class IInterface;
    }

    namespace neuraylib
    {
        class IMdl_module_info;
        class IMdl_package_info;
        class INeuray;
        class ITransaction;
    }
}
namespace tinyxml2
{
    class XMLElement;
}

class Mdl_cache_package;
class Index_cache_elements;

class Mdl_cache : public IMdl_cache
{
public:

    // child map, indexed by kind and unique identifier
    typedef std::unordered_map<IMdl_cache_node::Child_map_key,
                               IMdl_cache_item*,
                               IMdl_cache_node::Child_map_key_hash> Child_map;

    explicit Mdl_cache();
    virtual ~Mdl_cache();

    IMdl_cache_item* create(
        IMdl_cache_item::Kind kind,
        const char* entity_name,
        const char* simple_name,
        const char* qualified_name) override;
    bool erase(IMdl_cache_item* item) override;

    const char* get_locale() const override {return m_locale.empty() ? nullptr : m_locale.c_str();}
    void set_locale(const char* locale) override { m_locale = locale ? locale : ""; }

    const IMdl_cache_item* get_cache_item(
        const IMdl_cache_node::Child_map_key& key) const override;

    const IMdl_cache_package* get_cache_root() const override;
    const mi::neuraylib::IMdl_discovery_result* get_discovery_result() const override;

    bool save_to_disk(const IMdl_cache_serializer& serializer, const std::string& path) const;
    bool load_from_disk(const IMdl_cache_serializer& serializer, const std::string& path);

    // Updates the cache structure with the info from all search paths.
    // Note, this fails when no valid search path was found.
    bool update(mi::neuraylib::INeuray* neuray, mi::neuraylib::ITransaction* transaction);
    const Index_cache_elements* get_search_index() const { return m_index; }

private:
    mi::base::Handle<const mi::neuraylib::IMdl_discovery_result> m_discovery_result;
    IMdl_cache_package* m_cache_root;
    Index_cache_elements* m_index;
    Child_map m_item_map;
    std::string m_locale;
};


#endif



