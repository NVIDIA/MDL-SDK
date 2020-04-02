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
/// \brief View model class for package navigation.
///        This class contains the data model for the set of packages and modules at this level.
///        The data is unfiltered. For filtering, the corresponding proxy model is used.


#ifndef MDL_SDK_EXAMPLES_MDL_BROWSER_VM_NAV_STACK_LEVEL_H
#define MDL_SDK_EXAMPLES_MDL_BROWSER_VM_NAV_STACK_LEVEL_H

#include <QObject>
#include <qabstractitemmodel.h>
#include "vm_nav_package.h"

class Mdl_browser_node;
class VM_nav_stack;

class VM_nav_stack_level_model : public QAbstractListModel
{
    Q_OBJECT

public:
    explicit VM_nav_stack_level_model(QObject* parent = nullptr, Mdl_browser_node* node = nullptr);
    virtual ~VM_nav_stack_level_model();

    // number of elements in this list
    Q_PROPERTY(int count        READ rowCount
        NOTIFY countChanged)
        int rowCount(const QModelIndex& = QModelIndex()) const override;

    // no further columns
    int columnCount(const QModelIndex& = QModelIndex()) const override { return 1; }

    // get and set data from QML
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    bool setData(const QModelIndex& index, const QVariant& value, int role = Qt::EditRole)override;
    QHash<int, QByteArray> roleNames() const override { return VM_nav_package::roleNames(); }
    
    // get the corresponds parenting package of this level (QML binding in the proxy model)
    VM_nav_package* get_current_package() { return m_current_package; }

    // get an item of this level (the VM_nav_package can represent both, packages and modules)
    VM_nav_package* get_package(int index) const;

    // called from the model proxy after the set of selectable elements changed.
    // the call is forwarded to all elements in the list
    void update_presentation_counters();

signals:
    void countChanged();

private:

    Mdl_browser_node* m_current_node;
    VM_nav_package* m_current_package;
    QList<VM_nav_package*> m_child_packages;
};

#endif
