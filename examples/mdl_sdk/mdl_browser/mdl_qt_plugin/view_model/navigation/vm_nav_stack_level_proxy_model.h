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
/// \brief View model proxy class for package navigation.
///        This class filters the data model of the VM_nav_stack_level_model.
///        It stores no own data but influences the visual presentation.
///        Belongs to the ListView in NavStack.qml which defines the visual presentation.


#ifndef MDL_SDK_EXAMPLES_MDL_BROWSER_VM_NAV_STACK_LEVEL_PROXY_MODEL_H
#define MDL_SDK_EXAMPLES_MDL_BROWSER_VM_NAV_STACK_LEVEL_PROXY_MODEL_H

#include <QObject>
#include <QSortFilterProxyModel>


class VM_nav_package;
class VM_nav_stack_level_model;

class VM_nav_stack_level_proxy_model : public QSortFilterProxyModel
{
    Q_OBJECT

public:
    explicit VM_nav_stack_level_proxy_model(QObject* parent = nullptr);
    virtual ~VM_nav_stack_level_proxy_model() = default;

    //  item of this level (the VM_nav_package can represent both, packages and modules)
    Q_PROPERTY(VM_nav_package* currentPackage   READ get_current_package)
        VM_nav_package* get_current_package();

    // get an item of this level (the VM_nav_package can represent both, packages and modules)
    Q_INVOKABLE VM_nav_package* get_package(int index) const;

    // sets the underlaying VM_nav_stack_level_model
    // called by the VM_nav_stack
    void setSourceModel(QAbstractItemModel* source_model) override;

protected:
    // implements the filtering of packages and modules
    bool filterAcceptsRow(int source_row, const QModelIndex &source_parent) const override;

private:
    VM_nav_stack_level_model* m_model;
};

#endif
