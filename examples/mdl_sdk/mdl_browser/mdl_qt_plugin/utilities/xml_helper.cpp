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


#include "xml_helper.h"
#include <sstream>
#include <vector>

using namespace tinyxml2;

std::string Xml_helper::query_text(XMLElement* element, const std::string& defaultValue)
{
    if (!element) return defaultValue;
    if (element->FirstChild() && element->FirstChild()->ToText())
        return std::string(element->FirstChild()->Value());
    return "";
}


std::string Xml_helper::query_text_element(XMLElement* element, const std::string& element_name, 
                                           const std::string& defaultValue)
{
    if (!element) return defaultValue;
    XMLElement* child = element->FirstChildElement(element_name.c_str());

    if (child && child->FirstChild())
        return std::string(child->FirstChild()->Value());

    return defaultValue;
}

std::string Xml_helper::query_text_attribute(const XMLElement* element, 
                                             const std::string& attribute_name, 
                                             const std::string& defaultValue)
{
    if (!element) return defaultValue;
    const char* value = element->Attribute(attribute_name.c_str());
    if (value)
        return value;

    return defaultValue;
}

XMLDocument* Xml_helper::create_document()
{
    XMLDocument* doc = new XMLDocument();
    doc->InsertFirstChild(doc->NewDeclaration());
    return doc;
}

XMLElement* Xml_helper::create_element(XMLNode* parent, const std::string& name)
{
    XMLDocument* doc = parent->GetDocument();
    XMLElement* element = doc->NewElement(name.c_str());
    return element;
}

void Xml_helper::attach_element(XMLNode* parent, XMLElement* element)
{
    parent->InsertEndChild(element);
}


XMLElement* Xml_helper::attach_element(XMLDocument* doc, const std::string& name)
{
    XMLElement* element = doc->NewElement(name.c_str());
    doc->InsertEndChild(element);
    return element;
}


XMLElement* Xml_helper::attach_element(XMLNode* parent, const std::string& name)
{
    XMLDocument* doc = parent->GetDocument();
    XMLElement* element = doc->NewElement(name.c_str());
    parent->InsertEndChild(element);
    return element;
}


XMLElement* Xml_helper::attach_element(XMLNode* parent, const std::string& name, 
                                       const std::string& text)
{
    XMLElement* element = create_element(parent, name);
    element->SetText(text.c_str());
    attach_element(parent, element);
    return element;
}

