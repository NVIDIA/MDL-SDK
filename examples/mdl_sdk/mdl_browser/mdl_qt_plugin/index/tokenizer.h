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
/// \brief Spits a text string into single words and applies certain normalization rules to limit 
///        the size of the index and thereby to increase search performance


#ifndef MDL_SDK_EXAMPLES_MDL_BROWSER_TOKENIZER_H
#define MDL_SDK_EXAMPLES_MDL_BROWSER_TOKENIZER_H

#include <vector>
#include <string>
#include <set>

// spits a text string into single words and applies certain normalization rules to limit the size
// of the index and thereby to increase search performance
class Tokenizer
{
public:
    explicit Tokenizer();
    virtual ~Tokenizer() = default;

    // normalization applied to the entire text
    // this typically splits concatenated words, e.g., by replacing underscores with spaces
    virtual std::string normalize_sentence(const std::string& input) const;

    // normalization applied to single words
    // removes invalid characters, to lower case, ..., potentially stemming
    virtual std::string normalize_word(const std::string& input, 
                                       bool allow_operators = false) const;

    // splits an input text into tokens that appear in the index
    // runs normalization and excludes very common words using black listing
    virtual std::vector<std::string> tokenize(const std::string& input, 
                                              bool allow_operators = false) const;

    // some characters, e.g., the minus, are important to interpret the search query
    // this simply checks the a character if it is such a character
    bool is_operator(char character) const;

private:
    std::set<std::string> m_word_blacklist;
    std::vector<char> m_operators;
    size_t m_min_token_length;
};

#endif