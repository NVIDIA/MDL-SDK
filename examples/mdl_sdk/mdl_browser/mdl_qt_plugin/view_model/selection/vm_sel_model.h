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
/// \brief View model class for element selection.
///        This class contains the data model for the set of potentially selectable elements.
///        The data is unfiltered. For filtering, the corresponding proxy model is used.


#ifndef MDL_SDK_EXAMPLES_MDL_BROWSER_VM_SEL_MODEL_H
#define MDL_SDK_EXAMPLES_MDL_BROWSER_VM_SEL_MODEL_H

#include <QObject>
#include <qabstractitemmodel.h>
#include "vm_sel_element.h"
#include <stack>

class VM_nav_package;

class VM_sel_model : public QAbstractListModel
{
    Q_OBJECT

public:
    explicit VM_sel_model(QObject* parent = nullptr);
    virtual ~VM_sel_model();

    // number of elements in this list
    Q_PROPERTY(int count READ rowCount NOTIFY countChanged)
        int rowCount(const QModelIndex& = QModelIndex()) const override;

    // no further columns
    int columnCount(const QModelIndex& = QModelIndex()) const override;

    // get and set data from QML
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    QHash<int, QByteArray> roleNames() const override { return VM_sel_element::roleNames(); }

    // get an item that is potentially selectable (before filtering)
    Q_INVOKABLE VM_sel_element* get_element(int index) const;

    // filter the shown elements based on the selected package(s)
    // in case multiple packages are added, they are logically concatenated with a OR
    Q_INVOKABLE void set_browser_package_filter(VM_nav_package* package);

signals:
    void countChanged();

private:
    VM_sel_element* pull_pooled();
    void push_pooled(VM_sel_element* element);

    QVector<VM_sel_element*> m_elements;
    std::stack<VM_sel_element*> m_pool;

};

#endif
