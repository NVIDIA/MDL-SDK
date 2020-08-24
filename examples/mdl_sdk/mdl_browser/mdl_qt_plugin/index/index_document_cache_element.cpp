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


#include "index_document_cache_element.h"
#include "../cache/mdl_cache_material.h"
#include "../cache/mdl_cache_function.h"
#include "tokenizer.h"
#include "../utilities/string_helper.h"


class Mdl_cache_function;

Index_document_cache_element::Index_document_cache_element(const IMdl_cache_element* cache_item) 
    : m_cache_element(cache_item)
{
}

Index_document::word_list Index_document_cache_element::get_words(const Tokenizer* tokenizer) const
{
    // rating weights for different properties
    const float weight_name =           4.0f; // mdl name and display name
    const float weight_keywords =       3.0f;
    const float weight_module =         2.0f;
    const float weight_author =         2.0f;
    const float weight_description =    1.0f;

    Index_document::word_list words;

    // name without path and parameters
    std::string name = m_cache_element->get_simple_name();
    insert(words, tokenizer->tokenize(name), weight_name);

    // module
    const std::string qual_name = String_helper::replace(m_cache_element->get_module(), "::", " ");
    insert(words, tokenizer->tokenize(qual_name), weight_module);

    // Description
    const char* desc = m_cache_element->get_cache_data("Description");
    if(desc)
        insert(words, tokenizer->tokenize(desc), weight_description);

    // Keywords
    const char* keywords = m_cache_element->get_cache_data("Keywords");
    if (keywords)
        insert(words, tokenizer->tokenize(keywords), weight_keywords);

    if (dynamic_cast<const Mdl_cache_material*>(m_cache_element) != nullptr)
    {
        // Display Name
        const char* disp_name = m_cache_element->get_cache_data("DisplayName");
        if (disp_name)
            insert(words, tokenizer->tokenize(disp_name), weight_name);

        // Display Name
        const char* author_name = m_cache_element->get_cache_data("Author");
        if (author_name)
            insert(words, tokenizer->tokenize(author_name), weight_author);
    }

    if (dynamic_cast<const Mdl_cache_function*>(m_cache_element) != nullptr)
    {
        // the browser does not focus on functions at the moment.
    }

    return words;
}
