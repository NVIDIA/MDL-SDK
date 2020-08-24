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
/// \brief Base class of the cache hierarchy.
///        Stores a list of key-values pairs to hold information on particular items.
///        Follows the interface style of the MDL SDK.


#ifndef MDL_SDK_EXAMPLES_MDL_BROWSER_IMDL_CACHE_ITEM_H
#define MDL_SDK_EXAMPLES_MDL_BROWSER_IMDL_CACHE_ITEM_H

#include <mi/base/atom.h>
#include <string>
#include <functional>

namespace mi
{
    namespace neuraylib
    {
        class IMdl_package_info;
        class IMdl_discovery_result;
    }
}

class IMdl_cache;
class IMdl_cache_package;
class IMdl_cache_module;
class IMdl_cache_material;
class IMdl_cache_function;

class IMdl_cache_item
{
public:

    enum Kind
    {
        CK_PACKAGE,
        CK_MODULE,
        CK_MATERIAL,
        CK_FUNCTION,
    };


    virtual ~IMdl_cache_item() = default;

    // get the kind this item
    virtual Kind get_kind() const = 0;

    // name of the current item (unique in the child list of the parent item)
    // usually equal to the simple name, but e.g. includes the parameter types for functions
    virtual const char* get_entity_name() const = 0;

    // simple MDL name of the item, \em not necessarily unique in the child list of the parent item
    virtual const char* get_simple_name() const = 0;

    // qualified MDL name of the item, unique for the entire graph
    virtual const char* get_qualified_name() const = 0;

    // number of key-value pairs stored for this item
    virtual mi::Size get_cache_element_count() const = 0;

    // get the name a key.
    // index \in [0, max(0, get_cache_element_count()-1)]
    virtual const char* get_cache_key(mi::Size index) const = 0;

    // get the value that is stored for a given key
    virtual const char* get_cache_data(const char* key) const = 0;

    // set the value for a given key
    virtual void set_cache_data(const char* key, const char* value) = 0;

    // Get the cache this item is in, which allows to access the parent or other items
    virtual IMdl_cache* get_cache() = 0;
    virtual const IMdl_cache* get_cache() const = 0;

    // respect the hidden annotation of materials, functions, ...
    // is true if this or any parent node in the hierarchy is hidden
    virtual bool get_is_hidden() const = 0;
};



// item with children, i.e., packages and modules
class IMdl_cache_node : public IMdl_cache_item
{
public:

    // kind-name pair to identify children
    struct Child_map_key
    {
        IMdl_cache_item::Kind kind;
        std::string name; /* depending on the use case, this can be simple or qualified */

        bool operator==(const Child_map_key& other) const
        {
            return kind == other.kind && name == other.name;
        }

        friend bool operator <(const Child_map_key& lhs, const Child_map_key& rhs)
        {
            if (lhs.kind != rhs.kind) return lhs.kind < rhs.kind;
            return lhs.name < rhs.name;
        }
    };

    // hashing and comparison
    struct Child_map_key_hash
    {

        // hash function
        std::size_t operator () (const IMdl_cache_node::Child_map_key& p) const
        {
            // TODO replace by a more robust hash combination
            return std::hash<size_t>()(static_cast<size_t>(p.kind))
                 ^ std::hash<std::string>()(p.name) << 1;
        }

        // less then operator
        bool operator ()(const IMdl_cache_node::Child_map_key& lhs,
                         const IMdl_cache_node::Child_map_key& rhs) const
        {
            return lhs < rhs;
        }
    };

    virtual ~IMdl_cache_node() = default;

    // get the number of children this node has
    virtual mi::Size get_child_count() const = 0;

    // the name of a child base, which equals its simple_name.
    // index \in [0, max(0, get_child_count()-1)]
    virtual const IMdl_cache_node::Child_map_key* get_child_key(mi::Size index) const = 0;

    // get a child based on its simple_name 
    virtual IMdl_cache_item* get_child(const Child_map_key& key) = 0;
   
    // get a child based on its simple_name 
    virtual const IMdl_cache_item* get_child(const Child_map_key& key) const = 0;

    // Add an item to the child list.
    virtual bool add_child(IMdl_cache_item* child) = 0;
   
    // Removes an item from the child list.
    // This will not destruct the child.
    virtual bool remove_child(IMdl_cache_item* child) = 0;

    // Removes an item from the child list.
    // Returns the removed item or NULL if not found.
    // This will not destruct the child.
    // key is the combination of the kind and the simple_name
    virtual IMdl_cache_item* remove_child(const Child_map_key& key) = 0;

    // time of last modification
    // this allows to stop the updates of the cache earlier.
    virtual mi::Uint64 get_timestamp() const = 0;
};


// class to represent packages in the cache
class IMdl_cache_package : public IMdl_cache_node
{
public:
    virtual ~IMdl_cache_package() = default;
};

// class to represent modules in the cache
class IMdl_cache_module : public IMdl_cache_node
{
public:
    virtual ~IMdl_cache_module() = default;

    // true if the module is located in an MDL archive
    virtual bool get_located_in_archive() const = 0;
};



// item without children, that belongs to a module
class IMdl_cache_element : public IMdl_cache_item
{
public:
    virtual ~IMdl_cache_element() = default;

    // qualified name of the module this element is defined in
    virtual const char* get_module() const = 0;
};

// class to represent materials in the cache
class IMdl_cache_material : public IMdl_cache_element
{
public:
    virtual ~IMdl_cache_material() = default;
};

// class to represent functions in the cache
class IMdl_cache_function : public IMdl_cache_element
{
public:
    virtual ~IMdl_cache_function() = default;
};



// cache component
// get access to the cache tree, a searchable index structure for this cache, 
// and the MDL discovery API result, 
class IMdl_cache
{
public:
    // get elements by kind and qualified name
    virtual const IMdl_cache_item* get_cache_item(
        const IMdl_cache_node::Child_map_key& key) const = 0;

    // get element hierarchy
    //
    // could also be implemented using the map instead of child pointers
    // in this case we would use an ordered map which leads to log complexity for 'get_cache_item'.
    virtual const IMdl_cache_package* get_cache_root() const = 0;

    // the cache is created for the mdl package environment at a given point in time
    // the packages and modules available at that time a stored in this graph
    virtual const mi::neuraylib::IMdl_discovery_result* get_discovery_result() const = 0;

    // Factory method, used e.g. during deserialization, or while building the cache from scratch
    // use the corresponding erase to delete the created object
    virtual IMdl_cache_item* create(
        IMdl_cache_item::Kind kind,
        const char* entity_name,
        const char* simple_name,
        const char* qualified_name) = 0;

    // delete a cache item
    // this will trigger a recursive deleting of child elements, too.
    virtual bool erase(IMdl_cache_item* item) = 0;

    // get the local this cache was constructed for or null of translation was disabled.
    virtual const char* get_locale() const = 0;

    // set the local of the cache. 
    // note, that this will not translate the containing data.
    virtual void set_locale(const char* locale) = 0;
};


// interface to handle persistent storage of the cache
class IMdl_cache_serializer
{
public:
    virtual ~IMdl_cache_serializer() = default;

    // store the cache persistently under a given file path
    virtual bool serialize(const IMdl_cache* cache, const char* file_path) const = 0;

    // loads the presently stored cache from a given file path
    virtual IMdl_cache_package* deserialize(IMdl_cache* cache, const char* file_path) const = 0;
};

// class to store and load a cache to and from xml
class IMdl_cache_serializer_xml : public IMdl_cache_serializer
{
public:
    virtual ~IMdl_cache_serializer_xml() = default;
};

#endif
