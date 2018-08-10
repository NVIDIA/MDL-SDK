/******************************************************************************
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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


#include "view_model.h"
#include <QtCore/QString>

#include "navigation/vm_nav_stack.h"
#include "selection/vm_sel_model.h"
#include "selection/vm_sel_proxy_model.h"
#include "../mdl_sdk_wrapper.h"
#include "../cache/mdl_cache.h"
#include "../cache/imdl_cache.h"
#include "../mdl_browser_node.h"
#include "selection/selection_sorter.h"
#include <mi/mdl_sdk.h>

View_model::View_model(Mdl_sdk_wrapper* sdk, const char* application_folder)
    : m_navigation(nullptr)
    , m_selection_model(nullptr)
    , m_selection_proxy_model(nullptr)
    , m_settings("settings.xml")
{
    // use the discovered tree of the cache to be consistent regards to changes since then
    Mdl_cache* cache = sdk->get_cache();
    mi::base::Handle<const mi::neuraylib::IMdl_discovery_result> discovery_result(
        cache->get_discovery_result());
    m_browser_tree = Mdl_browser_tree::build(discovery_result.get(), cache->get_cache_root());

    // setup components
    m_navigation = new VM_nav_stack(this, m_browser_tree);
    m_selection_model = new VM_sel_model(this);
    m_selection_proxy_model = new VM_sel_proxy_model(this);
    m_selection_proxy_model->setSourceModel(m_selection_model);
    m_selection_proxy_model->set_settings(&m_settings);
    m_selection_proxy_model->setDynamicSortFilter(false);

    // filter based on the UI input during runtime
    m_user_filter = new Selection_filter_composed_search(sdk->get_cache()->get_search_index());
    m_selection_proxy_model->add_filter(m_user_filter);

    // filter on developer level
    m_selection_proxy_model->add_filter(
        new Selection_filter_element_type(VM_sel_element::ET_Material));

    // sorters
    m_selection_proxy_model->add_sorter(
        new Selection_sorter_ranking("Relevance", m_settings.sort_by_relevance_ascending));

    m_selection_proxy_model->add_sorter(
        new Selection_sorter_name("Name", m_settings.sort_by_name_ascending));

    m_selection_proxy_model->add_sorter(
        new Selection_sorter_date("Modification Date", m_settings.sort_by_date_ascending));

    // we want to count the elements that are not filter on a per package/module base
    // therefore we reset the counter "before" filtering
    connect(m_selection_proxy_model, &VM_sel_proxy_model::filtering_about_to_start,
            [this]() 
            { 
                m_browser_tree->reset_presentation_counters(); 
            });

    // increase the counter for each accepted element at the corresponding package/module
    connect(m_selection_proxy_model, &VM_sel_proxy_model::filtered_element,
            [this](VM_sel_element* element, FFilter_level level) 
            { 
                if((static_cast<size_t>(level) 
                   | static_cast<size_t>(FFilter_level::FL_SELECTION_IN_NAVIGATION))
                    == static_cast<size_t>(FFilter_level::FL_All))
                    m_browser_tree->increment_presentation_counter(
                        element->get_cache_element()->get_module()); 
            });

    // and update the recursive count after the last element has been filtered
    connect(m_selection_proxy_model, &VM_sel_proxy_model::filtering_finshed,
            [this]() 
            { 
                m_browser_tree->gather_presentation_counters(); 
            });

    // every time the query changed, the filter changed and we need to update the view model
    // this means the number of found elements per package/module
    connect(this, SIGNAL(user_filter_changed()),
            m_navigation, SLOT(update_presentation_counters()),
            Qt::ConnectionType::DirectConnection);

    // notify the selection when the browser packed a new package
    connect(m_navigation, SIGNAL(selected_package_changed(VM_nav_package*)),
            m_selection_proxy_model, SLOT(set_selected_root_package(VM_nav_package*)),
            Qt::ConnectionType::DirectConnection);

    // notify the selection when the browser packed a new module
    connect(m_navigation, SIGNAL(selected_module_changed(VM_nav_package*)),
            m_selection_proxy_model, SLOT(set_selected_module(VM_nav_package*)),
            Qt::ConnectionType::DirectConnection);



    m_application_folder = QString(application_folder);
    m_application_folder.replace('\\', '/');
    m_application_folder.append('/');
}

View_model::~View_model()
{
    m_user_filter = nullptr; // owned by the proxy model
    delete m_selection_model;
    delete m_selection_proxy_model;
    delete m_navigation;
    delete m_browser_tree;
}

void View_model::update_user_filter(const QString& text)
{
    // we want to count the elements that are not filter on a per package/module base
    // therefore we reset the counter "before" filtering
    emit user_filter_is_about_to_change();
    m_user_filter->set_query(text.toUtf8().constData());
    emit user_filter_changed();
}

void View_model::set_sort_mode(QString sortBy, bool sortAscending)
{
    m_selection_proxy_model->set_sort_mode(sortBy, sortAscending);
}

Q_INVOKABLE void View_model::set_result_and_close(const QString& text)
{
    m_result = text;
    emit close_window();
}
