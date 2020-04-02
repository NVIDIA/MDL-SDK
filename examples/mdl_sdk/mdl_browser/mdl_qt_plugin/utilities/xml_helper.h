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
/// \brief Helper class for handling xml operations.


#ifndef MDL_SDK_EXAMPLES_MDL_BROWSER_XML_HELPER_H
#define MDL_SDK_EXAMPLES_MDL_BROWSER_XML_HELPER_H

#include <tinyxml2.h>
#include <string>
#include <vector>


class Xml_helper
{
public:
    // reading
    //---------------------------------------------------------------------
    static std::string query_text(tinyxml2::XMLElement* element, 
                                  const std::string& defaultValue = "");

    static std::string query_text_element(tinyxml2::XMLElement* element, 
                                          const std::string& element_name, 
                                          const std::string& defaultValue = "");

    static std::string query_text_attribute(const tinyxml2::XMLElement* element, 
                                            const std::string& attribute_name, 
                                            const std::string& defaultValue = "");


    // writing
    //---------------------------------------------------------------------
    static tinyxml2::XMLDocument* create_document();
    static tinyxml2::XMLElement* attach_element(tinyxml2::XMLDocument* doc, 
                                                const std::string& name);

    static tinyxml2::XMLElement* attach_element(tinyxml2::XMLNode* parent, 
                                                const std::string& name);

    static tinyxml2::XMLElement* attach_element(tinyxml2::XMLNode* parent, 
                                                const std::string& name, 
                                                const std::string& text);


private:
    static tinyxml2::XMLElement* create_element(tinyxml2::XMLNode* parent, 
                                                const std::string& name);

    static void attach_element(tinyxml2::XMLNode* parent, 
                               tinyxml2::XMLElement* element);
};

#endif