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


#include "application_settings_serializer_xml.h"
#include "xml_helper.h"
#include <fstream>

using namespace tinyxml2;

bool Application_settings_serializer_xml::serialize(
    const Application_settings_base& settings, const std::string& file_path) const
{
    XMLDocument* doc = Xml_helper::create_document();
    XMLElement* settings_node = Xml_helper::attach_element(doc, "Settings");

    for (const auto& kv : get_key_value_storage(settings))
        Xml_helper::attach_element(settings_node, kv.first, kv.second);    

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

bool Application_settings_serializer_xml::deserialize(
    Application_settings_base& settings, const std::string& file_path) const
{
    XMLDocument* document = new XMLDocument();
    if (XML_SUCCESS != document->LoadFile(file_path.c_str()))
        return false;

    XMLElement* settings_node = document->FirstChildElement("Settings");

    auto& kv_map = get_key_value_storage(settings);
    for (const XMLElement* data = settings_node->FirstChildElement(); 
         data != nullptr; 
         data = data->NextSiblingElement())
    {
        if (data && data->FirstChild())
        {
            const char* key = data->Value();
            const char* value = data->FirstChild()->Value();
            kv_map[key] = value;
        }
    }

    return true;
}
