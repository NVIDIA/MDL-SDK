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
///        Provides the view models for navigation levels while traversing.
///        Belongs to the NavStack.qml which defines the visual presentation.


#ifndef MDL_SDK_EXAMPLES_MDL_BROWSER_VM_NAV_STACK_H
#define MDL_SDK_EXAMPLES_MDL_BROWSER_VM_NAV_STACK_H

#include <QObject>

class Mdl_browser_tree;
class VM_nav_stack_level_proxy_model;
class VM_nav_package;


class VM_nav_stack : public QObject
{
    Q_OBJECT
public:
    explicit VM_nav_stack(QObject* parent = nullptr, Mdl_browser_tree* browser_tree = nullptr);
    virtual ~VM_nav_stack() = default;

    // called from QML when a module was selected or deselected (nullptr in the latter case)
    // independent of the type of navigation used
    Q_INVOKABLE void set_selected_module(VM_nav_package* package);

    // called from QML to get a initial level to display (first item on the stack)
    Q_INVOKABLE VM_nav_stack_level_proxy_model* create_root_level();

    // called from QML while navigating to a deeper level 
    Q_INVOKABLE VM_nav_stack_level_proxy_model* expand_package(VM_nav_package* package);

    // is called from QML at the end of the navigation to a deeper level or after going back
    Q_INVOKABLE void set_current_level(VM_nav_stack_level_proxy_model* current);


    Q_INVOKABLE void dispose_level(VM_nav_stack_level_proxy_model* level);

signals:
    // signal emitted by 'set_current_level'
    // used to inform the 'selection' (connection is made by the View_model)
    void selected_package_changed(VM_nav_package* p);

    // signal emitted by 'set_selected_module'
    // used to inform the 'selection' (connection is made by the View_model)
    void selected_module_changed(VM_nav_package* m);

public slots:
    // after the filtering changed, we update local counters to eventually hide irrelevant
    // packages and modules (connection is made by the View_model)
    void update_presentation_counters();

private:
    Mdl_browser_tree* m_browser_tree;
    VM_nav_stack_level_proxy_model* m_current_level;
};

#endif
