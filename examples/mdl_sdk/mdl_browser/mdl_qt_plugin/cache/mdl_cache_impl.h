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
/// \brief Implementation of the IMdl_cache_item, -node, and -element interface.


#ifndef MDL_SDK_EXAMPLES_MDL_BROWSER_MDL_CACHE_IMPL_H
#define MDL_SDK_EXAMPLES_MDL_BROWSER_MDL_CACHE_IMPL_H

#include "imdl_cache.h"
#include <string>
#include <string.h> 
#include <sstream>
#include <iostream>
#include <cassert>
#include <map>
#include <unordered_map>
#include <vector>
#include <mi/base/atom.h>


namespace mi
{
    namespace base
    {
        class IInterface;
    }

    namespace neuraylib
    {
        class INeuray;
        class ITransaction;
    }
}

class Mdl_Cache;

// generic item class to realize (multiple) interface inheritance 
template<class T_item>
class Mdl_cache_item :
    public T_item   // class is expected to implement (at least) IMdl_cache_item
{
public:
    explicit Mdl_cache_item()
        : m_cache(nullptr)
        , m_entity_name("")
        , m_simple_name("")
        , m_qualified_name("")
        , m_hidden(false)
        , m_hidden_valid(false)
    { }
    virtual ~Mdl_cache_item<T_item>() = default;

    const char* get_entity_name() const override { return m_entity_name.c_str(); }
    const char* get_simple_name() const override { return m_simple_name.c_str(); }
    const char* get_qualified_name() const override { return m_qualified_name.c_str(); }

    mi::Size get_cache_element_count() const override { return m_cached_data.size(); }
    const char* get_cache_key(mi::Size index) const override;
    const char* get_cache_data(const char* key) const override;
    void set_cache_data(const char* key, const char* value) override;

    IMdl_cache* get_cache() override { return m_cache; }
    const IMdl_cache* get_cache() const override { return m_cache; }

    // used while building the cache
    void initialize(
        IMdl_cache* cache,
        const std::string& entity_name,
        const std::string& simple_name,
        const std::string& qualified_name);

    // used while building the cache
    virtual bool update(mi::neuraylib::INeuray* neuray, 
                        mi::neuraylib::ITransaction* transaction, 
                        const mi::base::IInterface* node) = 0;

    bool get_is_hidden() const override;
    void set_is_hidden(bool value);

protected:
    typedef Mdl_cache_item<T_item> Base;
    std::map<std::string, IMdl_cache_item*> m_children;

private:
    IMdl_cache* m_cache;
    std::string m_entity_name;
    std::string m_simple_name;
    std::string m_qualified_name;
    std::vector<std::string> m_keys; // to be able to iterate using the interface
    std::unordered_map<std::string, std::string> m_cached_data;
    mutable bool m_hidden;
    mutable bool m_hidden_valid;
};



// generic item class to realize (multiple) interface inheritance 
template<class T_node>
class Mdl_cache_node :
    public Mdl_cache_item<T_node>   // class is expected to implement (at least) IMdl_cache_node
{
public:

    // child map, indexed by kind and unique identifier
    typedef std::unordered_map<IMdl_cache_node::Child_map_key,
        IMdl_cache_item*,
        IMdl_cache_node::Child_map_key_hash> Child_map;

    explicit Mdl_cache_node<T_node>() 
        : m_timestamp(0)
    {
    }
    virtual ~Mdl_cache_node<T_node>()
    {
        for (auto& kv : m_children)
            Base::get_cache()->erase(kv.second);
        m_children.clear();
        m_child_names.clear();
    }

    mi::Uint64 get_timestamp() const override;

    mi::Size get_child_count() const override { return m_children.size(); }
    const IMdl_cache_node::Child_map_key* get_child_key(mi::Size index) const override;
    IMdl_cache_item* get_child(const IMdl_cache_node::Child_map_key& key) override;
    const IMdl_cache_item* get_child(const IMdl_cache_node::Child_map_key& key) const override;

    bool add_child(IMdl_cache_item* child) override;
    bool remove_child(IMdl_cache_item* child) override;
    IMdl_cache_item* remove_child(const IMdl_cache_node::Child_map_key& key) override;

protected:
    typedef Mdl_cache_item<T_node> Base;

    const Child_map& get_children() const { return m_children; }

    void set_timestamp(mi::Uint64 value);
    mutable mi::Uint64 m_timestamp;

private:
    Child_map m_children;

    std::vector<IMdl_cache_node::Child_map_key> m_child_names;
};


// generic item class to realize (multiple) interface inheritance 
template<class T_element>
class Mdl_cache_element :
    public Mdl_cache_item<T_element> // expected to implement (at least) IMdl_cache_element
{
public:
    explicit Mdl_cache_element<T_element>()
        : m_module("")
    { }
    virtual ~Mdl_cache_element<T_element>() = default;

    const char* get_module() const override;

protected:
    typedef Mdl_cache_item<T_element> Base;

private:
    mutable std::string m_module;
};


// ------------------------------------------------------------------------------------------------
// Implementation: ITEM
// ------------------------------------------------------------------------------------------------

template<class T_item>
void Mdl_cache_item<T_item>::initialize(IMdl_cache* cache, const std::string& entity_name, const std::string& simple_name, const std::string& qualified_name)
{
    m_cache = cache;
    m_entity_name = entity_name;
    m_simple_name = simple_name;
    m_qualified_name = qualified_name;
}

template<class T_item>
const char* Mdl_cache_item<T_item>::get_cache_key(mi::Size index) const
{
    if (index >= m_keys.size()) 
        return nullptr;

    return m_keys[index].c_str();
}



template <class T_item>
const char* Mdl_cache_item<T_item>::get_cache_data(const char* key) const
{
    const auto it = m_cached_data.find(key);
    if (it == m_cached_data.end())
        return nullptr;

    return it->second.data();
}

template <class T_item>
void Mdl_cache_item<T_item>::set_cache_data(const char* key, 
                                            const char* value)
{
    const auto it = m_cached_data.find(key);
    if (it == m_cached_data.end())
        m_keys.push_back(key);

    m_cached_data[key] = value;
}


template<class T_item>
bool Mdl_cache_item<T_item>::get_is_hidden() const
{
    if (!m_hidden_valid)
    {
        const char* value = Base::get_cache_data("Hidden");
        m_hidden = value && strcmp(value, "true") == 0;
        m_hidden_valid = true;
    }
    return m_hidden;
}

template<class T_item>
void Mdl_cache_item<T_item>::set_is_hidden(bool value)
{
    if (m_hidden_valid && m_hidden == value) return;

    Base::set_cache_data("Hidden", value ? "true" : "false");
    m_hidden = value;
    m_hidden_valid = true;
}


// ------------------------------------------------------------------------------------------------
// Implementation: NODE
// ------------------------------------------------------------------------------------------------

template<class T_node>
mi::Uint64 Mdl_cache_node<T_node>::get_timestamp() const
{
    if (m_timestamp == 0)
    {
        const char* value = Base::get_cache_data("Timestamp");
        m_timestamp = value ? static_cast<mi::Uint64>(std::stoll(value, nullptr, 0)) : 0;
    }

    return m_timestamp;
}

template<class T_node>
void Mdl_cache_node<T_node>::set_timestamp(mi::Uint64 value)
{
    if (m_timestamp != value)
    {
        m_timestamp = value;
        std::stringstream ss;
        ss << m_timestamp;
        const std::string s = ss.str();
        Base::set_cache_data("Timestamp", s.c_str());
    }
}


template <class T_node>
const IMdl_cache_node::Child_map_key* Mdl_cache_node<T_node>::get_child_key(mi::Size index) const
{
    if (index >= m_child_names.size())
        return nullptr;

    return &m_child_names[index];
}

template <class T_node>
IMdl_cache_item* Mdl_cache_node<T_node>::get_child(const IMdl_cache_node::Child_map_key& key)
{
    auto it = m_children.find(key);
    return it == m_children.end() ? nullptr : it->second;
}

template <class T_node>
const IMdl_cache_item* Mdl_cache_node<T_node>::get_child(
    const IMdl_cache_node::Child_map_key& key) const
{
    const auto it = m_children.find(key);
    return it == m_children.end() ? nullptr : it->second;
}

template<class T_node>
bool Mdl_cache_node<T_node>::add_child(IMdl_cache_item* child)
{
    const IMdl_cache_node::Child_map_key key = {child->get_kind(), child->get_entity_name()};
    const auto it = m_children.find(key);
    if (it != m_children.end())
        return false;

    m_child_names.push_back(key);
    m_children[key] = child;
    return true;
}

template<class T_node>
IMdl_cache_item* Mdl_cache_node<T_node>::remove_child(const IMdl_cache_node::Child_map_key& key)
{
    const auto it = m_children.find(key);
    if (it == m_children.end())
        return nullptr;

    IMdl_cache_item* item = it->second;
    m_children.erase(it);

    const auto it2 = std::find(m_child_names.begin(), m_child_names.end(), key);
    assert(it2 != m_child_names.end());
    m_child_names.erase(it2);
    return item;
}

template<class T_node>
bool Mdl_cache_node<T_node>::remove_child(IMdl_cache_item* child)
{
    const auto removed = remove_child({child->get_kind(), child->get_entity_name()});
    return removed != nullptr;
}


// ------------------------------------------------------------------------------------------------
// Implementation: ELEMENT
// ------------------------------------------------------------------------------------------------

template<class T_element>
const char* Mdl_cache_element<T_element>::get_module() const
{
    if (m_module.empty())
    {
        std::string qualified_name = Base::get_qualified_name();
        m_module = qualified_name.substr(0, qualified_name.rfind(Base::get_entity_name()) - 2);
    }
    return m_module.c_str();
}

#endif
