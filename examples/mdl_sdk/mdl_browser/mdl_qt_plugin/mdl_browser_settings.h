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
/// \brief Persistent settings of this application


#ifndef MDL_SDK_EXAMPLES_MDL_BROWSER_MDL_BROWSER_SETTINGS_H
#define MDL_SDK_EXAMPLES_MDL_BROWSER_MDL_BROWSER_SETTINGS_H

#include "utilities/application_settings.h"
#include <QObject>
#include <qhash.h>
#include <QVariant>
#include <QMetaType>
#include <string>

class Mdl_browser_settings 
    : public QObject
    , public Application_settings_base
{
    Q_OBJECT

public:

    explicit Mdl_browser_settings(const std::string& auto_save_path = "");
    virtual ~Mdl_browser_settings() = default;


    // the value of the sorting criteria selected last (time the browser was used)
    Q_PROPERTY(QString last_sort_critereon          READ get_last_sort_critereon)
        QString get_last_sort_critereon() const { return last_sort_critereon.get().c_str(); }

    // to order when sorting by relevance
    Q_PROPERTY(bool sort_by_relevance_ascending     READ get_sort_by_relevance_ascending)
        bool get_sort_by_relevance_ascending() const { return sort_by_relevance_ascending; }

    // to order when sorting by name
    Q_PROPERTY(bool sort_by_name_ascending          READ get_sort_by_name_ascending)
        bool get_sort_by_name_ascending() const { return sort_by_name_ascending; }

    // to order when sorting by date
    Q_PROPERTY(bool sort_by_date_ascending          READ get_sort_by_date_ascending)
        bool get_sort_by_date_ascending() const { return sort_by_date_ascending; }

    // the selected view mode, grid or list, selected last (time the browser was used)
    Q_PROPERTY(QString last_view_mode               READ get_last_view_mode
                                                    WRITE set_last_view_mode 
                                                    NOTIFY last_view_mode_changed)
        QString get_last_view_mode() const { return last_view_mode.get().c_str(); }
        void set_last_view_mode(const QString& value);

    // persistent properties
    SettingString last_navigation_package;
    SettingString last_user_query;
    SettingString last_sort_critereon;
    SettingBool sort_by_relevance_ascending;
    SettingBool sort_by_name_ascending;
    SettingBool sort_by_date_ascending;
    SettingString last_view_mode;


signals:
    void last_view_mode_changed();
};

#endif 
