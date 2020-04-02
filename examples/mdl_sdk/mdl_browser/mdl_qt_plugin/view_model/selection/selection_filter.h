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
/// \brief Classes for filtering selectable elements, e.g., materials, functions, ...


#ifndef MDL_SDK_EXAMPLES_MDL_BROWSER_SELECTION_FILTER_H
#define MDL_SDK_EXAMPLES_MDL_BROWSER_SELECTION_FILTER_H

#include <vector>
#include "vm_sel_element.h"
#include "../../index/index.h"

class Index_cache_elements;
class VM_sel_proxy_model;
class VM_nav_package;
class Index_document_cache_element;

// base class for filtering selectable elements.
class Selection_filter_base
{
    friend class VM_sel_proxy_model;

public:
    explicit Selection_filter_base();
    virtual ~Selection_filter_base() = default;
    
    // evaluate this filter.
    // this will respect the 'enabled' and 'negated' property.
    // called by the proxy model
    bool evaluate(const VM_sel_element* element) const;

    // enable/disable this filter
    bool get_enabled() const { return m_is_enabled; }
    void set_enabled(bool value);

    // invert the result of the filter
    bool get_negated() const { return m_is_negated; }
    void set_negated(bool value);

protected:
    // implementation of the actual filter criteria
    virtual bool evaluate_intern(const VM_sel_element* element) const = 0;
    
    // informs the proxy model in case of changes of the filter settings 
    // if properties of the filter are changed that would change the overall filtering result
    void invalidate_filter();

private:
    // keep track of the proxy model in order to inform it about changes
    void set_proxy_model(VM_sel_proxy_model* model) { m_model = model; }

    bool m_is_enabled;
    bool m_is_negated;
    VM_sel_proxy_model* m_model;
};

//-------------------------------------------------------------------------------------------------
// Logic Combiners
//-------------------------------------------------------------------------------------------------

// simple logical operators to combine filters
class Selection_filter_operator 
    : public Selection_filter_base
{
public:
    enum FilterOperator
    {
        FO_AND = 0,
        FO_OR = 1
    };

    explicit Selection_filter_operator(FilterOperator op = FO_AND);
    virtual ~Selection_filter_operator();
    
    // evaluate the logical expression
    bool evaluate_intern(const VM_sel_element* element) const override;

    // added filters are owned
    void add(const Selection_filter_base* filter);

    // removed filters are not owned anymore
    void remove(const Selection_filter_base* filter);

    // operator used to combine added filters
    FilterOperator get_operator() const { return m_operator; }
    void set_operator(FilterOperator value);

private:
    FilterOperator m_operator;
    std::vector<const Selection_filter_base*> m_filters;
};

//-------------------------------------------------------------------------------------------------
// Element Selected Module(s)
//-------------------------------------------------------------------------------------------------

// filtering based on the parenting module
class Selection_filter_selected_module 
    : public Selection_filter_base
{
public:
    explicit Selection_filter_selected_module();
    virtual ~Selection_filter_selected_module() = default;

    // evaluate the filter
    bool evaluate_intern(const VM_sel_element* element) const override;

    // the parent model to filter by
    void set_selected_module(const VM_nav_package* value);
    const VM_nav_package* get_selected_module() { return m_selected_module; }

private:
    const VM_nav_package* m_selected_module;
};


//-------------------------------------------------------------------------------------------------
// Element Type
//-------------------------------------------------------------------------------------------------

// filter based on the element type, e.g., material, function, ..
class Selection_filter_element_type 
    : public Selection_filter_base
{
public:
    explicit Selection_filter_element_type(
        VM_sel_element::Element_type mask = VM_sel_element::ET_All);
    virtual ~Selection_filter_element_type() = default;

    // evaluate the filter
    bool evaluate_intern(const VM_sel_element* element) const override;

    // selection mask of types that will pass the filter
    VM_sel_element::Element_type get_type_mask() const { return m_type_mask; }
    void get_type_mask(VM_sel_element::Element_type value);

private:
    VM_sel_element::Element_type m_type_mask;
};


//-------------------------------------------------------------------------------------------------
// Simple Search
//-------------------------------------------------------------------------------------------------

// filter based on single token of a search query
class Selection_filter_single_token_search 
    : public Selection_filter_base
{
public:
    explicit Selection_filter_single_token_search(const Index_cache_elements* index);
    virtual ~Selection_filter_single_token_search() = default;

    // evaluate the filter
    // returns true if the element (material, function) appeared in the search result for 
    // queried token
    // this also handles the 'minus' prefix in order to negate the result
    bool evaluate_intern(const VM_sel_element* element) const override;

    // one token of a (maybe more complex) search query
    // setting the query will initiate a search, the result is stored and the proxy model is
    // invalidated, which in turn courses a re-evaluation
    const std::string& get_query() const { return m_query; }
    void set_query(const std::string& value);

private:
    const Index_cache_elements* m_index;
    std::string m_query;
    std::vector<ResultListItem<Index_document_cache_element>> m_query_results;
};

// filter based on more complex multi-token search query 
class Selection_filter_composed_search 
    : public Selection_filter_base
{
public:
    explicit Selection_filter_composed_search(const Index_cache_elements* index);
    virtual ~Selection_filter_composed_search() = default;

    // evaluates the filter that is composed of multiple single token search queries
    bool evaluate_intern(const VM_sel_element* element) const override;

    // the more complex query (entered by the user)
    const std::string& get_query() const { return m_query; }
    void set_query(const std::string& value);

private:
    const Index_cache_elements* m_index;
    std::map<std::string, Selection_filter_single_token_search*> m_token_queries;
    Selection_filter_operator m_main;
    std::string m_query;
};

#endif
