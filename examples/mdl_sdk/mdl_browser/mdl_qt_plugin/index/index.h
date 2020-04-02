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
/// \brief A simple inverse index structure to support searching.
///        The index is currently build on startup every time.
///        It is not possible to update documents at that point.


#ifndef MDL_SDK_EXAMPLES_MDL_BROWSER_INDEX_H
#define MDL_SDK_EXAMPLES_MDL_BROWSER_INDEX_H

#include <map>
#include <vector>
#include <regex>
#include <set>
#include "tokenizer.h"
#include "index_document.h"
#include <cstdlib>

// list element stored for each document that contains the current word 
template<class T_document>
struct PostingListItem
{
    const T_document* Document;     // the document that contains the word
    float Ranking_weight;           // the importance of the word for the document
};

// list element returned with a query
template<class T_document>
struct ResultListItem
{
    ResultListItem<T_document>(const T_document* document, float ranking) 
        : Document(document)
        , Ranking(ranking)
    {  }

    const T_document* Document; // the document that fits the query
    float Ranking;              // the rating this document has for the query                      
};

template<class T_document>
class Index
{
public:
    typedef PostingListItem<T_document> PostingItem;
    typedef ResultListItem<T_document> ResultItem;
    typedef std::map<uint32_t, PostingListItem<T_document>> PostingList;

    explicit Index()
    {
        m_tokenizer = new Tokenizer();
    }

    virtual ~Index()
    {
        for (auto it : m_documents)
            delete it;

        m_documents.clear();
        delete m_tokenizer;
    };

    // add new document to the index.
    // is called during build up.
    void add_document(const T_document* doc)
    {
        uint32_t id = doc->get_document_id();
        Index_document::word_list words = doc->get_words(m_tokenizer);
        for (const auto& p : words)
        {
            // make sure there is a posting list for this word
            if (m_index.find(p.first) == m_index.end())
                m_index[p.first] = PostingList();

            // add new posting or increment frequency counter
            auto& list = m_index[p.first];
            if (list.find(id) == list.end())
                list[id] = {doc,  p.second};
            else
                list[id].Ranking_weight += p.second; // accumulate single contributions (simple)
        }
        m_documents.push_back(doc);
    }

    // find documents for a search term (query)
    // called at run time.
    std::vector<ResultItem> find(const std::string& query) const
    {
        std::vector<ResultItem> results;
        const std::regex expression("(.*)(" + query + ")(.*)");
        std::smatch matches;

        for (const auto& entry : m_index)
        {
            std::regex_match(entry.first, matches, expression);
            if (!matches.empty())
            {
                for (const auto& p : entry.second)
                {
                    ResultItem r(p.second.Document, rank(p.second, entry.first, query));
                    results.push_back(r);
                }
            }
        }
        return results;
    }

    // we use one tokenizer for generating the index as well as for processing the user 
    // input, this way we make sure the same rules are applied (casing, special chars, ...)
    const Tokenizer* get_tokenizer() const { return m_tokenizer; }

protected:

    // computes the ranking for a specific result item.
    // simple approach to rank fuzzy search results,
    // but here we rank the regex match.. totally equal vs. only a small part of the word.
    virtual float rank(const PostingItem& item, // entry in the index
                       const std::string& indexed_word, // the word found in the index
                       const std::string& query) const // the queried word
    {
        float dist = static_cast<float>(std::abs(static_cast<int>(indexed_word.length())
                                               - static_cast<int>(query.length())));
        return item.Ranking_weight / (dist + 1.0f);
    }

private:
    std::map<std::string, PostingList> m_index; // posting list for each word 
    std::vector<const T_document*> m_documents; // keep track of all documents (for cleanup only)
    Tokenizer* m_tokenizer;
};



#endif