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


#include "vm_sel_model.h"
#include "../navigation/vm_nav_package.h"
#include "../../cache/imdl_cache.h"
#include "../../mdl_browser_node.h"
#include <mi/mdl_sdk.h>
#include <iostream>

void create_full_element_list_recursively(std::vector<const IMdl_cache_element*>& list, 
                                          const IMdl_cache_node* node)
{
    for (mi::Size i = 0, n = node->get_child_count(); i<n; ++i)
    {
        // nodes that can have "grand" children  (packages only)
        const IMdl_cache_node::Child_map_key* child_key = node->get_child_key(i);
        const IMdl_cache_item* child_item = node->get_child(*child_key);
        const auto child_nore = dynamic_cast<const IMdl_cache_node*>(child_item);
        if (child_nore)
        {
            create_full_element_list_recursively(list, child_nore);
            continue;
        }

        // elements that can be selected
        const auto child_element = dynamic_cast<const IMdl_cache_element*>(child_item);
        if (child_element)
        {
            list.push_back(child_element);
            continue;
        }

        std::cerr << "[Mdl_cache_item] create_full_element_list_recursively: missing case.\n";
    }
}

std::vector<const IMdl_cache_element*> create_full_element_list(const Mdl_browser_node* node)
{
    std::vector<const IMdl_cache_element*> list;
    create_full_element_list_recursively(list, node->get_cache_node());
    return list;
}

VM_sel_model::VM_sel_model(QObject* parent) :
    QAbstractListModel(parent),
    m_elements()
{
}

VM_sel_model::~VM_sel_model()
{
    m_elements.clear();
}

int VM_sel_model::rowCount(const QModelIndex&) const
{
    return m_elements.count();
}

int VM_sel_model::columnCount(const QModelIndex&) const
{
    // ATTENTION: this should be large enough to support sorting
    // so increase it in case there are more than 3 sorting criteria.
    return 3; 
}

QVariant VM_sel_model::data(const QModelIndex& index, int role) const
{
    if (!index.isValid())
        return QVariant();

    if (index.row() >= rowCount() || index.row() < 0)
        return QVariant();

    if(index.column() > 0)
        return QVariant();

    return m_elements[index.row()]->data(role);
}

Q_INVOKABLE VM_sel_element* VM_sel_model::get_element(int index) const
{
    if (index < 0 || index >= rowCount())
        return nullptr;

    return m_elements[index];
}

void VM_sel_model::set_browser_package_filter(VM_nav_package* package)
{
    // clear the list (for now)
    if (!m_elements.empty())
    {
        beginRemoveRows(QModelIndex(), 0, m_elements.size() - 1);
        for (auto e : m_elements)
            push_pooled(e);

        m_elements.clear();
        endRemoveRows();
    }

    // add new ones
    auto cache_elements = create_full_element_list(package->get_browser_node());
    m_elements.reserve(static_cast<int>(cache_elements.size()));
    if (!cache_elements.empty())
    {
        beginInsertRows(QModelIndex(), 0, static_cast<int>(cache_elements.size()) - 1);
        for (auto e : cache_elements)
        {
            VM_sel_element* pooled = pull_pooled();
            pooled->setup(e);
            m_elements.append(pooled);
        }
        endInsertRows();
    }
    emit dataChanged(index(0, 0), index(rowCount() - 1, 0));
    emit countChanged();
}

VM_sel_element* VM_sel_model::pull_pooled()
{
    // create a fixed number of instances if the pool is empty
    if(m_pool.empty())
        for (int i = 0; i < 128; ++i)
            m_pool.push(new VM_sel_element(this));

    VM_sel_element* element = m_pool.top();
    m_pool.pop();
    return element;
}

void VM_sel_model::push_pooled(VM_sel_element* element)
{
    m_pool.push(element);
}
