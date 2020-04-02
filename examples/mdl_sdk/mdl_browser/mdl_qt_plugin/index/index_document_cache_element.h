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
/// \brief indexable document that is created from an element of the cache that stores 
///        informations about materials, functions, ...


#ifndef MDL_SDK_EXAMPLES_MDL_BROWSER_DOCUMENT_MDL_ELEMENT_H
#define MDL_SDK_EXAMPLES_MDL_BROWSER_DOCUMENT_MDL_ELEMENT_H

#include "index_document.h"

class IMdl_cache_element;
class Tokenizer;

// indexable document that is created from an element of the cache that stores informations about 
// materials, functions, ...
class Index_document_cache_element : public Index_document
{
public:
    explicit Index_document_cache_element(const IMdl_cache_element* cache_item);
    virtual ~Index_document_cache_element() = default;

    // the words to be indexed for this document
    word_list get_words(const Tokenizer* tokenizer) const override;

    // the cache element this documented is created from
    const IMdl_cache_element* get_cache_element() const { return m_cache_element; }

private:
    const IMdl_cache_element* m_cache_element;
};

#endif
