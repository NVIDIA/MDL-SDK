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
/// \brief Main view model class that can be globally accessed from QML


#ifndef MDL_SDK_EXAMPLES_MDL_BROWSER_VIEW_MODEL_H
#define MDL_SDK_EXAMPLES_MDL_BROWSER_VIEW_MODEL_H

#include <QObject>
#include <mi/mdl_sdk.h>
#include "selection/selection_filter.h"
#include "../mdl_browser_settings.h"

struct Mdl_browser_callbacks;
class Mdl_browser_tree;

class VM_nav_stack;
class VM_sel_model;
class VM_sel_proxy_model;

class Mdl_cache;

class View_model : public QObject
{
Q_OBJECT

public:
    explicit View_model() {};
    explicit View_model(mi::neuraylib::INeuray* neuray,
                        mi::neuraylib::ITransaction* transaction,
                        Mdl_browser_callbacks* callbacks,
                        bool cache_rebuild,
                        const char* application_folder);

    virtual ~View_model();

    // navigation component
    Q_PROPERTY(VM_nav_stack* navigation             READ get_navigation
                                                    NOTIFY navigation_changed)
        VM_nav_stack* get_navigation() const { return m_navigation; }

    // selection component
    Q_PROPERTY(VM_sel_proxy_model* selectionModel   READ get_selection_model
                                                    NOTIFY selection_model_changed)
        VM_sel_proxy_model* get_selection_model() const { return m_selection_proxy_model; }

    // persistent application settings
    Q_PROPERTY(Mdl_browser_settings* settings       READ get_settings
                                                    NOTIFY settings_changed)
        Mdl_browser_settings* get_settings() { return &m_settings; }

    // navigation component
    Q_PROPERTY(QString application_folder          READ get_application_folder
               NOTIFY navigation_changed)
        QString get_application_folder() const { return m_application_folder; }

    // header is quite simple at the moment, so we don't add an extra component

    // invoked when search query changed
    Q_INVOKABLE void update_user_filter(const QString& text);

    // invoked when the sorting is changed (button, or when hitting enter in the search field)
    Q_INVOKABLE void set_sort_mode(QString sortBy, bool sortAscending);


    // passing configurations from the application an returning results is currently kept simple

    // called when the user confirms the selection after that, the window is closed
    Q_INVOKABLE void set_result_and_close(const QString& text);

    // called from the entry-function that opened the browser to get the result
    // and also from connecting qml components
    Q_INVOKABLE const QString get_result() const { return m_result; }

    // called when a selection was made
    Q_INVOKABLE void accept();

    // called when the browser was closed without accepting
    Q_INVOKABLE void reject();

signals:
    void navigation_changed();
    void selection_model_changed();
    void settings_changed();

    // signals before and after the search query changed
    void user_filter_is_about_to_change(); // TODO required?
    void user_filter_changed();

    void close_window(); // TODO required?

private:
    mi::base::Handle<mi::neuraylib::INeuray> m_neuray;
    mi::base::Handle<mi::neuraylib::ITransaction> m_transaction;
    Mdl_cache* m_cache;

    Mdl_browser_tree* m_browser_tree;
    VM_nav_stack* m_navigation;
    VM_sel_model* m_selection_model;
    VM_sel_proxy_model* m_selection_proxy_model;
    Selection_filter_composed_search* m_user_filter;

    Mdl_browser_settings m_settings;
    QString m_application_folder;
    QString m_result;
    Mdl_browser_callbacks* m_callbacks;
};

#endif
