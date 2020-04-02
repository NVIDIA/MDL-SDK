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


#include "selection_filter.h"
#include "../../cache/imdl_cache.h"
#include "../../index/index.h"
#include "../../index/index_document_cache_element.h"
#include "../../index/index_cache_elements.h"
#include "../../mdl_browser_node.h"
#include "../navigation/vm_nav_package.h"
#include "vm_sel_proxy_model.h"
#include "vm_sel_element.h"


Selection_filter_base::Selection_filter_base() :
    m_is_enabled(true),
    m_is_negated(false),
    m_model(nullptr)
{
}

bool Selection_filter_base::evaluate(const VM_sel_element* element) const
{
    if (!m_is_enabled) return true;
    return m_is_negated ^ evaluate_intern(element);
}

void Selection_filter_base::set_enabled(bool value)
{
    if (m_is_enabled == value) return;
    m_is_enabled = value;
    invalidate_filter();
}

void Selection_filter_base::set_negated(bool value)
{
    if (m_is_negated == value) return;
    m_is_negated = value;
    invalidate_filter();
}

void Selection_filter_base::invalidate_filter()
{
    if(m_model) m_model->invalidate();
}


//-------------------------------------------------------------------------------------------------
// Logic Combiners
//-------------------------------------------------------------------------------------------------


Selection_filter_operator::Selection_filter_operator(FilterOperator op)
    : m_operator(op)
{
}

Selection_filter_operator::~Selection_filter_operator()
{
    for (auto it : m_filters)
        delete it;

    m_filters.clear();
}

bool Selection_filter_operator::evaluate_intern(const VM_sel_element* element) const
{
    switch (m_operator)
    {
    case FO_AND:
    {
        for (const auto& it : m_filters)
            if (!it->evaluate(element))
                return false;
        return true;
    }

    case FO_OR:
    {
        // it would make sense to sort the filters by complexity and start with cheep 
        for (const auto& it : m_filters)
            if (it->evaluate(element))
                return true;
        return false;
    }
    }
    return true; // should not happen
}

void Selection_filter_operator::add(const Selection_filter_base* filter)
{
    m_filters.push_back(filter);
    invalidate_filter();
}

void Selection_filter_operator::remove(const Selection_filter_base* filter)
{
    const auto it = std::find(m_filters.begin(), m_filters.end(), filter);
    if (it != m_filters.end())
        m_filters.erase(it);
    invalidate_filter();
}

void Selection_filter_operator::set_operator(FilterOperator value)
{
    if (m_operator == value) return;
    m_operator = value;
    invalidate_filter();
}


//-------------------------------------------------------------------------------------------------
// Element Selected Module(s)
//-------------------------------------------------------------------------------------------------

Selection_filter_selected_module::Selection_filter_selected_module():
    m_selected_module(nullptr)
{
}

bool Selection_filter_selected_module::evaluate_intern(const VM_sel_element* element) const
{
    if (m_selected_module == nullptr) return true;

    const char* sel_name = m_selected_module->get_browser_node()->get_qualified_name();
    const char* cur_name = element->get_cache_element()->get_module();

    return strcmp(cur_name, sel_name) == 0;
}

void Selection_filter_selected_module::set_selected_module(const VM_nav_package* value)
{
    if (m_selected_module == value) return;
    m_selected_module = value;
    invalidate_filter();
}

//-------------------------------------------------------------------------------------------------
// Element Type
//-------------------------------------------------------------------------------------------------

Selection_filter_element_type::Selection_filter_element_type(VM_sel_element::Element_type mask)
    : m_type_mask(mask)
{
}

bool Selection_filter_element_type::evaluate_intern(const VM_sel_element* element) const
{
    return (element->get_type() & m_type_mask) != 0;
}

void Selection_filter_element_type::get_type_mask(VM_sel_element::Element_type value)
{
    if (m_type_mask == value) return;
    m_type_mask = value;
    invalidate_filter();
}

//-------------------------------------------------------------------------------------------------
// Simple Search
//-------------------------------------------------------------------------------------------------

Selection_filter_single_token_search::Selection_filter_single_token_search(
    const Index_cache_elements* index) :
    m_index(index),
    m_query("")
{
}

bool Selection_filter_single_token_search::evaluate_intern(const VM_sel_element* element) const
{
    const IMdl_cache_element* item = element->get_cache_element();

    bool found = false;
    float found_ranking = 1.0f;
    for (const auto& it : m_query_results)
        if (it.Document->get_cache_element() == item)
        {
            found = true;
            found_ranking = it.Ranking;
            break;
        }

    if (found ^ get_negated())
        element->set_search_ranking({}, element->get_search_ranking() * found_ranking);

    return found;
}

void Selection_filter_single_token_search::set_query(const std::string& value)
{
    if (m_query == value) return;
    m_query = value;

    // negated search?
    bool negated = false;
    if (m_query[0] == '-')
    {
        m_query_results = m_index->find(value.substr(1));
        negated = true;
    }
    else
    {
        m_query_results = m_index->find(value);
    }
    
    // invalidate only once
    if(get_negated() == negated)
        invalidate_filter();
    else
        set_negated(negated); // this will call invalidate_filter();
}


Selection_filter_composed_search::Selection_filter_composed_search(
    const Index_cache_elements* index) :
    m_index(index),
    m_query("")
{
}

bool Selection_filter_composed_search::evaluate_intern(const VM_sel_element* element) const
{
    element->set_search_ranking({}, 1.0f);
    return m_main.evaluate(element);
}

void Selection_filter_composed_search::set_query(const std::string& value)
{
    if (m_query == value) return;
    m_query = value;

    // get query tokens and remove duplicates
    std::vector<std::string> toAdd = m_index->get_tokenizer()->tokenize(m_query, true);
    std::sort(toAdd.begin(), toAdd.end());
    toAdd.erase(std::unique(toAdd.begin(), toAdd.end()), toAdd.end());

    // find existing queries and mark changed
    std::set<std::string> to_remove;
    for (const auto& it : m_token_queries)
        to_remove.insert(it.first);

    // iterate over tokens and identify unchanged ones
    for (int i = static_cast<int>(toAdd.size()) - 1; i >= 0; --i)
    {
        const auto existing = to_remove.find(toAdd[i]);
        if (existing != to_remove.end())
        {
            to_remove.erase(existing);
            toAdd.erase(toAdd.begin() + i);
        }
    }

    // reuse existing outdated queries
    while(!toAdd.empty() && !to_remove.empty())
    {
        // get to first free
        const std::string old_key = *to_remove.begin();
        Selection_filter_single_token_search* to_change = m_token_queries[old_key];
        m_token_queries.erase(old_key);
        to_remove.erase(to_remove.begin());

        // use it for a new or changed query
        const std::string new_key = toAdd.back();
        to_change->set_query(new_key);
        m_token_queries[new_key] = to_change;
        toAdd.pop_back();
    }

    // free existing unused queries
    for (int i = static_cast<int>(to_remove.size()) - 1; i >= 0; --i)
    {
        Selection_filter_single_token_search* to_delete = m_token_queries[*to_remove.begin()];
        m_token_queries.erase(*to_remove.begin());
        to_remove.erase(to_remove.begin());
        m_main.remove(to_delete);
        delete to_delete;
    }

    // create new ones
    for (size_t i = 0, n = toAdd.size(); i<n; ++i)
    {
        Selection_filter_single_token_search* to_create = new Selection_filter_single_token_search(
            m_index);
        to_create->set_query(toAdd[i]);
        m_token_queries[toAdd[i]] = to_create;
        m_main.add(to_create);
    }

    // mark filter changed
    // note that the child filters have not set a model, 
    // so their changes will not be reflected directly (instead only here)
    invalidate_filter();
}
