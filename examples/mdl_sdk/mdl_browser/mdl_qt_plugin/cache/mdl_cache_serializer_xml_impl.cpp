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


#include "mdl_cache_serializer_xml_impl.h"
#include <mi/mdl_sdk.h>
#include "imdl_cache.h"
#include "../utilities/xml_helper.h"
#include <fstream>
#include <iostream>
using namespace tinyxml2;

Mdl_cache_serializer_xml_impl::Mdl_cache_serializer_xml_impl()
{
}

bool Mdl_cache_serializer_xml_impl::serialize(const IMdl_cache* cache,
                                              const char* file_path) const
{
    XMLDocument* doc = Xml_helper::create_document();
    XMLElement* cache_node = Xml_helper::attach_element(doc, "MdlCache");

    if (cache->get_locale())
        cache_node->SetAttribute("Locale", cache->get_locale());
    
    serialize(cache_node, cache->get_cache_root());

    XMLPrinter printer;
    doc->Print(&printer);

    std::ofstream file_stream;
    file_stream.open(file_path);
    if (file_stream)
    {
        file_stream << printer.CStr();
        file_stream.close();
        return true;
    }
    return false;
}

XMLElement* Mdl_cache_serializer_xml_impl::serialize(XMLElement* parent, 
                                                     const IMdl_cache_item* item) const
{
    const char* xml_node_name = nullptr;

    if (dynamic_cast<const IMdl_cache_package*>(item))
        xml_node_name = "Package";
    else if (dynamic_cast<const IMdl_cache_module*>(item))
        xml_node_name = "Module";
    else if (dynamic_cast<const IMdl_cache_material*>(item))
        xml_node_name = "Material";
    else if (dynamic_cast<const IMdl_cache_function*>(item))
        xml_node_name = "Function";

    XMLElement* element = Xml_helper::attach_element(parent, xml_node_name);

    // attributes
    element->SetAttribute("EntityName", item->get_entity_name());
    element->SetAttribute("SimpleName", item->get_simple_name());
    element->SetAttribute("QualifiedName", item->get_qualified_name());

    // Cached data
    tinyxml2::XMLElement* cache_node = Xml_helper::attach_element(element, "CacheData");
    for (mi::Size i = 0, n = item->get_cache_element_count(); i<n; ++i)
    {
        const char* key = item->get_cache_key(i);
        Xml_helper::attach_element(
            cache_node, 
            key,
            item->get_cache_data(key));

        // xml will not work with general binary data.. 
        // in that case, we could use base64 encoding
    }


    // add children
    const IMdl_cache_node* node = dynamic_cast<const IMdl_cache_node*>(item);
    if(node)
    {
        for (mi::Size i = 0, n = node->get_child_count(); i<n; ++i)
            serialize(element, node->get_child(*node->get_child_key(i)));

    }

    return element;
}


IMdl_cache_package* Mdl_cache_serializer_xml_impl::deserialize(
    IMdl_cache* cache, const char* file_path) const
{
    XMLDocument* document = new XMLDocument();
    if (XML_SUCCESS != document->LoadFile(file_path))
        return nullptr;

    std::string locale = Xml_helper::query_text_attribute(document->RootElement(), "Locale", "");
    if (!locale.empty())
        cache->set_locale(locale.c_str());

    XMLElement* root = document->RootElement()->FirstChildElement("Package");
    IMdl_cache_item* cache_root = deserialize(cache, nullptr, root);
    
    return dynamic_cast<IMdl_cache_package*>(cache_root);
}

IMdl_cache_item* Mdl_cache_serializer_xml_impl::deserialize(
    IMdl_cache* cache, const IMdl_cache_node* parent, const tinyxml2::XMLElement* item) const
{
    const std::string node_name = item->Value();
    IMdl_cache_item* cache_item = nullptr;
    IMdl_cache_node* cache_node = nullptr;
    
    std::string entity_name = Xml_helper::query_text_attribute(item, "EntityName");
    std::string simple_name = Xml_helper::query_text_attribute(item, "SimpleName");
    std::string qualified_name = Xml_helper::query_text_attribute(item, "QualifiedName");

    if (node_name == "Package")
    {
        cache_item = cache->create(IMdl_cache_item::CK_PACKAGE,
            entity_name.c_str(), simple_name.c_str(), qualified_name.c_str());
        cache_node = dynamic_cast<IMdl_cache_node*>(cache_item);
    }
    else if (node_name == "Module")
    {
        cache_item = cache->create(IMdl_cache_item::CK_MODULE,
            entity_name.c_str(), simple_name.c_str(), qualified_name.c_str());
        cache_node = dynamic_cast<IMdl_cache_node*>(cache_item);
    }
    else if (node_name == "Material")
    {
        cache_item = cache->create(IMdl_cache_item::CK_MATERIAL,
            entity_name.c_str(), simple_name.c_str(), qualified_name.c_str());
    }
    else if (node_name == "Function")
    {
        cache_item = cache->create(IMdl_cache_item::CK_FUNCTION,
            entity_name.c_str(), simple_name.c_str(), qualified_name.c_str());
    }

    // process child elements
    for (const XMLElement* child = item->FirstChildElement(); 
         child != nullptr; 
         child = child->NextSiblingElement())
    {
        const std::string child_name = child->Value();

        // data stored as key value pairs
        if (child_name == "CacheData")
        {
            for (const XMLElement* data = child->FirstChildElement(); 
                 data != nullptr; 
                 data = data->NextSiblingElement())
            {
                if (data && data->FirstChild())
                {
                    const char* key = data->Value();
                    const char* value = data->FirstChild()->Value();
                    cache_item->set_cache_data(key, value);
                }
            }
            continue;
        }

        // actual child items in the hierarchy
        IMdl_cache_item* cache_child = deserialize(cache, cache_node, child);
        if(!cache_child)
        {
            std::cerr << "[Mdl_cache_serializer_xml_impl] deserialize: " 
                      << "Failed to deserialize children of: " << qualified_name << "\n";
            delete cache_item;
            return nullptr;
        }
        cache_node->add_child(cache_child);
    }
    return cache_item;
}
