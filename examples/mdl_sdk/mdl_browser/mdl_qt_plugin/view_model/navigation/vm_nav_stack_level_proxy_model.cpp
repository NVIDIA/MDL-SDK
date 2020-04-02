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


#include "vm_nav_stack_level_proxy_model.h"
#include "vm_nav_stack_level_model.h"

VM_nav_stack_level_proxy_model::VM_nav_stack_level_proxy_model(QObject* parent)
    : QSortFilterProxyModel(parent)
    , m_model(nullptr)
{
}

void VM_nav_stack_level_proxy_model::setSourceModel(QAbstractItemModel* source_model)
{
    m_model = dynamic_cast<VM_nav_stack_level_model*>(source_model);
    QSortFilterProxyModel::setSourceModel(source_model);

    // encode the sorter index as column to sort
    sort(0, Qt::AscendingOrder);
}

Q_INVOKABLE VM_nav_package* VM_nav_stack_level_proxy_model::get_package(int index) const
{
    const QModelIndex proxy_index = QSortFilterProxyModel::index(index, 0); // proxy index
    const QModelIndex source_index = mapToSource(proxy_index); // source index
    return m_model->get_package(source_index.row());
}

VM_nav_package* VM_nav_stack_level_proxy_model::get_current_package()
{
    return m_model->get_current_package();
}
bool VM_nav_stack_level_proxy_model::filterAcceptsRow(int source_row,
                                                      const QModelIndex &source_parent) const
{
    // only show packages and modules with selectable elements
    return m_model->get_package(source_row)->get_presentation_count() > 0;
}
