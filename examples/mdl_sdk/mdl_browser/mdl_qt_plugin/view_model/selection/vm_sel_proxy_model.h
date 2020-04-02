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
///        This class sorts and filters the data model of the VM_sel_model.
///        It stores no own data but influences the visual presentation.
///        Belongs to the ListView in SelList or SelGrid.qml which defines the visual presentation.


#ifndef MDL_SDK_EXAMPLES_MDL_BROWSER_VM_SEL_PROXY_MODEL_H
#define MDL_SDK_EXAMPLES_MDL_BROWSER_VM_SEL_PROXY_MODEL_H

#include <QObject>
#include <QSortFilterProxyModel>
#include "selection_filter.h"
#include <vector>

class VM_nav_package;
class VM_sel_element;
class VM_sel_model;
class Index_cache_elements;
class Mdl_browser_tree;

class Selection_sorter_base;
class Mdl_browser_settings;

// Flags
enum class FFilter_level : size_t
{
    FL_NONE = 0,
    FL_MAIN_FILTER_EXPRESSION = 1,      // all filter, like type and filtering based on search
    FL_SELECTION_IN_NAVIGATION = 2,     // narrowing down the result by picking a module
    FL_HIDDEN = 4,
    // next = 8
    FL_All = (FL_HIDDEN* 2 -1)
};

class VM_sel_proxy_model : public QSortFilterProxyModel
{
    friend class Selection_filter_base; // in order to invalidate after filter changes

    Q_OBJECT

public:
    explicit VM_sel_proxy_model(QObject* parent = nullptr);
    virtual ~VM_sel_proxy_model();

    // sets the underlaying VM_sel_model.
    // called by the View_model class on startup.
    void setSourceModel(QAbstractItemModel* sourceModel) override;

    // sets the shared settings application settings to restore previous sorting settings.
    // called by the View_model class on startup.
    void set_settings(Mdl_browser_settings* settings) { m_settings = settings; }

    // filter the shown elements based on the selected package or module in the navigation panel.
    Q_INVOKABLE void set_browser_package_filter(VM_nav_package* package);

    // add a filter (the filter is owned by this proxy model then)
    void add_filter(Selection_filter_base* filter);
    void remove_filter(Selection_filter_base* filter);

    // add a sorter (the filter is owned by this proxy model then)
    void add_sorter(Selection_sorter_base* sorter);
    void remove_sorter(Selection_sorter_base* sorter);

    // get the sorter index by name
    int get_sorter_index(const std::string& name);

signals:
    // events are used to track the number of relevant elements per package/module
    // the events are wired up in the View_model class on startup

    // fired before elements are filtered
    void filtering_about_to_start() const;
    // fired after the decision of each element
    void filtered_element(VM_sel_element* element, FFilter_level accepted) const;
    // fired when filtering is done
    void filtering_finshed() const;

public slots:

    // invoked, when the used picked a package in the navigation panel (connected by View_model)
    // this fills the (unfiltered) model with all elements under the selected package 
    void set_selected_root_package(VM_nav_package* p);

    // invoked, when the used picked a module in the navigation panel (connected by View_model)
    // this set the FL_SELECTION_IN_NAVIGATION to show only elements of the selected module
    void set_selected_module(VM_nav_package* m);

    // invoked when the sorting changed (from HeaderBar.qml, connected by View_model)
    void set_sort_mode(QString sortBy, bool sortAscending);

protected:

    // implements the filtering of packages and modules
    bool filterAcceptsRow(int sourceRow, const QModelIndex &sourceParent) const override;

    // implements the sorting of packages and modules
    bool lessThan(const QModelIndex &left, const QModelIndex &right) const override;

private:
    VM_sel_model* m_model;
    Selection_filter_operator* m_root_filter;
    Selection_filter_selected_module* m_selected_module_filter;
    Mdl_browser_tree* m_browser_tree;
    std::vector<Selection_sorter_base*> m_sorter_list;
    Mdl_browser_settings* m_settings;
};

#endif
