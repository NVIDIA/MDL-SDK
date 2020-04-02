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


#include "vm_sel_proxy_model.h"
#include "../navigation/vm_nav_package.h"
#include "vm_sel_element.h"
#include "vm_sel_model.h"
#include <iostream>
#include "../../cache/mdl_cache.h"
#include "../../mdl_browser_node.h"
#include "../../mdl_browser_settings.h"
#include "selection_sorter.h"
#include <cassert>

VM_sel_proxy_model::VM_sel_proxy_model(QObject* parent) :
    m_model(nullptr),
    m_root_filter(new Selection_filter_operator()),
    m_selected_module_filter(new Selection_filter_selected_module())
{
    m_root_filter->set_proxy_model(this);
    m_selected_module_filter->set_proxy_model(this);
}

VM_sel_proxy_model::~VM_sel_proxy_model()
{
    delete m_root_filter;
    delete m_selected_module_filter;

    for (auto sorter : m_sorter_list)
        delete sorter;
    m_sorter_list.clear();
}

void VM_sel_proxy_model::setSourceModel(QAbstractItemModel* sourceModel)
{
    m_model = dynamic_cast<VM_sel_model*>(sourceModel);
    QSortFilterProxyModel::setSourceModel(sourceModel);
}

void VM_sel_proxy_model::set_browser_package_filter(VM_nav_package* package)
{
}

void VM_sel_proxy_model::add_filter(Selection_filter_base* filter)
{
    filter->set_proxy_model(this);
    m_root_filter->add(filter);
    invalidate();
}

void VM_sel_proxy_model::remove_filter(Selection_filter_base* filter)
{
    m_root_filter->remove(filter);
    filter->set_proxy_model(nullptr);
    invalidate();
}

void VM_sel_proxy_model::add_sorter(Selection_sorter_base* sorter)
{
    if(std::find(m_sorter_list.begin(), m_sorter_list.end(), sorter) != m_sorter_list.end())
    {
        std::cerr << "[ERROR] VM_sel_proxy_model::add_sorter: sorted name already in use.\n";
        return;
    }
    m_sorter_list.push_back(sorter);


    if (m_model->columnCount() < static_cast<int>(m_sorter_list.size()))
    {
        // TODO increase column count in the model .. hard-coded 3
        assert(false);
    }
}

void VM_sel_proxy_model::remove_sorter(Selection_sorter_base* sorter)
{
    auto found = std::find(m_sorter_list.begin(), m_sorter_list.end(), sorter);
    if (found == m_sorter_list.end())
    {
        std::cerr << "[ERROR] VM_sel_proxy_model::remove_sorter: sorted name already in use.\n";
        return;
    }
    m_sorter_list.erase(found);
}

int VM_sel_proxy_model::get_sorter_index(const std::string& name)
{
    auto found = std::find_if(m_sorter_list.begin(), m_sorter_list.end(),
        [&name](Selection_sorter_base* const& item)
        { return item->get_name() == name; });

    if (found == m_sorter_list.end()) return -1;
    return found - m_sorter_list.begin();
}

void VM_sel_proxy_model::set_selected_root_package(VM_nav_package* p)
{
    if (m_model && !p->get_is_module())
    {
        m_model->set_browser_package_filter(p);
        m_selected_module_filter->set_selected_module(nullptr);
    }
}

void VM_sel_proxy_model::set_selected_module(VM_nav_package* m)
{
    if (m_model && m && m->get_is_module())
        m_selected_module_filter->set_selected_module(m);
    else
        m_selected_module_filter->set_selected_module(nullptr);
}

void VM_sel_proxy_model::set_sort_mode(QString sortBy, bool sortAscending)
{
    // save this setting for the next start
    // TODO this does not save the order of sorters
    {
        // keep track of this in the settings
        m_settings->last_sort_critereon = sortBy.toUtf8().constData();

        if (sortBy == "Relevance")
            m_settings->sort_by_relevance_ascending = sortAscending;
        else if (sortBy == "Name")
            m_settings->sort_by_name_ascending = sortAscending;
        else if (sortBy == "Modification Date")
            m_settings->sort_by_date_ascending = sortAscending;
    }

    const std::string name = sortBy.toUtf8().constData();
    const int sorter_index = get_sorter_index(name);
    if (sorter_index < 0)
    {
        std::cerr << "[ERROR] VM_sel_proxy_model::set_sort_mode: unknown sorter: " 
                  << name << ".\n";
        return;
    }

    // encode the sorter index as column to sort
    sort(sorter_index, sortAscending ? Qt::AscendingOrder : Qt::DescendingOrder);

    // keep track of that for saving
    m_sorter_list[sorter_index]->set_ascending(sortAscending);
}

bool VM_sel_proxy_model::filterAcceptsRow(int sourceRow, const QModelIndex& sourceParent) const
{
    if (sourceRow == 0)
        emit filtering_about_to_start();

    QModelIndex index = sourceModel()->index(sourceRow, 0, sourceParent);
    VM_sel_element* element = m_model->get_element(sourceRow);

    // evaluate the filter tree
    const FFilter_level accepted_first = m_root_filter->evaluate(element) 
        ? FFilter_level::FL_MAIN_FILTER_EXPRESSION : FFilter_level::FL_NONE;

    // second level filtering 
    const FFilter_level accepted_second = m_selected_module_filter->evaluate(element)
        ? FFilter_level::FL_SELECTION_IN_NAVIGATION : FFilter_level::FL_NONE;

    // respect the hidden annotation
    const FFilter_level accepted_thrid = !element->get_is_hidden()
        ? FFilter_level::FL_HIDDEN : FFilter_level::FL_NONE;

    // combine
    const FFilter_level accepted = static_cast<const FFilter_level>(
          static_cast<size_t>(accepted_first) 
        + static_cast<const size_t>(accepted_second)
        + static_cast<const size_t>(accepted_thrid));

    emit filtered_element(element, accepted);

    if (sourceRow == m_model->rowCount() - 1)
        emit filtering_finshed();

    // passed the filter or not
    return accepted == FFilter_level::FL_All;
}

bool VM_sel_proxy_model::lessThan(const QModelIndex& left, const QModelIndex& right) const
{
    VM_sel_element* element_left = m_model->get_element(left.row());
    VM_sel_element* element_right = m_model->get_element(right.row());

    // sort criteria is encoded by the column index
    return m_sorter_list[left.column()]->compare(element_left, element_right) < 0.0f;
}