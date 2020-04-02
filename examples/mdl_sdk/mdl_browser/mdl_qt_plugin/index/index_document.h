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
/// \brief Document for an inverse index search structure.


#ifndef MDL_SDK_EXAMPLES_MDL_BROWSER_DOCUMENT_BASE_H
#define MDL_SDK_EXAMPLES_MDL_BROWSER_DOCUMENT_BASE_H

#include <vector>
#include <string>

class Tokenizer;

class Index_document
{
public:
    // word-rating pair to be returned by get_words and in the index class.
    typedef std::vector<std::pair<std::string, float>> word_list;

    explicit Index_document();
    virtual ~Index_document() = default;

    // the words to be indexed for this document
    virtual word_list get_words(const Tokenizer* tokenizer) const = 0;

    // identifier of this document that is used in the posting list of the index.
    virtual uint32_t get_document_id() const;

protected:

    // helper function to be used in get_words.
    // allows to add multiple words with a certain ranking to the word_list.
    static void insert(std::vector<std::pair<std::string, float>>& destination,
                       const std::vector<std::string>& source, 
                       float ranking);

private:
    // As we build up the index on startup, we can use a simple running index
    // to identify documents. In case the index should be saved to disk, you
    // need to something more sophisticated or swtich to a real data base.
    static uint32_t s_id_counter;

    const uint32_t m_id;
};

#endif