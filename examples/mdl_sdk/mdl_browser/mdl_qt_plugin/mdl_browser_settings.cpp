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


#include "mdl_browser_settings.h"
#include "utilities/application_settings_serializer_xml.h"

Mdl_browser_settings::Mdl_browser_settings(const std::string& auto_save_path)
    : Application_settings_base(
        new Application_settings_serializer_xml(), // change for other serialization interface
        auto_save_path  // if true, setting a value will update the file
)
, last_navigation_package(*this, "LastNavigationPackage", "::")
, last_user_query(*this, "LastUserQuery", "")
, last_sort_critereon(*this, "LastSortCriterion", "Relevance")
, sort_by_relevance_ascending(*this, "SortByRelevanceAscending", false)
, sort_by_name_ascending(*this, "SortByNameAscending", true)
, sort_by_date_ascending(*this, "SortByDateAscending", false)
, last_view_mode(*this, "LastViewMode", "List")
{
}

void Mdl_browser_settings::set_last_view_mode(const QString& value)
{
    last_view_mode = value.toUtf8().constData();
    emit last_view_mode_changed();
}
