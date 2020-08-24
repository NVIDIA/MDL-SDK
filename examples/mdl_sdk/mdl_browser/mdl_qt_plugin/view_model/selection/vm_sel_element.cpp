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


#include "vm_sel_element.h"
#include "../../cache/imdl_cache.h"
#include "../../cache/mdl_cache_material.h"
#include "../../cache/mdl_cache_function.h"
#include <ctime>

QString query_cache_string(const IMdl_cache_element* element, const char* key, 
                           const char* default_value = "")
{
    const char* data = element->get_cache_data(key);
    return QString(data ? data : default_value);
}

VM_sel_element::VM_sel_element(QObject* parent) :
    QObject(parent),
    m_cache_element(nullptr),
    m_type(ET_Undefined),
    m_search_ranking(1.0)
{
}

QHash<int, QByteArray> VM_sel_element::roleNames()
{
    static const QHash<int, QByteArray> roles{
        {NameRole, "elementName"},
        {ModuleNameRole, "elementModuleName"},
        {TypeRole, "elementType"},
        {DisplayNameRole, "elementDisplayName"},
        {AuthorRole, "elementAuthor"},
        {DescriptionRole, "elementDescription"},
        {KeywordsRole, "elementKeywords"},
        {LocatedInArchiveRole, "elementLocatedInArchive"},
        {ModuleHintRole, "elementModuleHint"},
        {ThumbnailRole, "elementThumbnail"},
        {ModificationDateRole, "elementModification"},
        {SearchRankingRole, "elementSearchRanking"}
    };
    return roles;
}

const IMdl_cache_module* VM_sel_element::get_parent_module_cache_element() const
{
    const auto cache_item = get_cache_element();
    return dynamic_cast<const IMdl_cache_module*>(cache_item->get_cache()->get_cache_item(
        {IMdl_cache_item::CK_MODULE, cache_item->get_module()}));
}

bool VM_sel_element::get_is_hidden() const
{
    return m_cache_element->get_is_hidden();
}

void VM_sel_element::setup(const IMdl_cache_element* cache_element)
{
    m_cache_element = cache_element;
    
    if(!m_cache_element)
    {
        m_type = ET_Undefined;
        return;
    }

    if (dynamic_cast<const Mdl_cache_material*>(cache_element) != nullptr)
        m_type = ET_Material;

    if (dynamic_cast<const Mdl_cache_function*>(cache_element) != nullptr)
        m_type = ET_Function;
}

QVariant VM_sel_element::data(int role) const
{
    switch (role)
    {
        case VM_sel_element::NameRole: return get_name();
        case VM_sel_element::ModuleNameRole: return get_module_name();
        case VM_sel_element::TypeRole: return get_type();
        case VM_sel_element::DisplayNameRole: return get_display_name();
        case VM_sel_element::AuthorRole: return get_author();
        case VM_sel_element::DescriptionRole: return get_description();
        case VM_sel_element::KeywordsRole: return get_keywords();
        case VM_sel_element::LocatedInArchiveRole: return get_located_in_archive();
        case VM_sel_element::ModuleHintRole: return get_module_hint();
        case VM_sel_element::ThumbnailRole: return get_thumbnail();
        case VM_sel_element::ModificationDateRole: return get_modification();
        case VM_sel_element::SearchRankingRole: return get_search_ranking();
        default: return QVariant();
    }
}

QString VM_sel_element::get_name() const
{
    return m_cache_element->get_simple_name();
}

QString VM_sel_element::get_module_name() const
{
    return m_cache_element->get_module();
}

QString VM_sel_element::get_display_name() const
{
    return query_cache_string(m_cache_element, "DisplayName");
}

QString VM_sel_element::get_keywords() const
{
    return query_cache_string(m_cache_element, "Keywords");
}

QString VM_sel_element::get_author() const
{
    return query_cache_string(m_cache_element, "Author");
}

QString VM_sel_element::get_description() const
{
    return query_cache_string(m_cache_element, "Description");
}

bool VM_sel_element::get_located_in_archive() const
{
    const IMdl_cache_module* parent_module = get_parent_module_cache_element();
    if (!parent_module)
        return false;

    return parent_module->get_located_in_archive();
}

QString VM_sel_element::get_module_hint() const
{
    QString result = "<b>Name in the MDL module:</b><br>" 
                   + QString(m_cache_element->get_module()) + "::" + get_name() + "<br>";

    result += "<br><b>Located in an MDL Archive:</b><br>" 
            + QString(get_located_in_archive() ? "true" : "false") + "<br>";

    QString author = get_author();
    if (!author.isEmpty())
        result += "<br><b>Author:</b><br>" + author + "<br>";

    QString desc = get_description();
    if (!desc.isEmpty())
        result += "<br><b>Description:</b><br>" + desc + "<br>";

    QString keywords = get_keywords();
    if (!keywords.isEmpty())
        result += "<br><b>Keywords:</b><br>" + keywords + "<br>";

    result += "<br><b>Current search filter rank:</b><br>" 
            + QString::number(get_search_ranking(), 10, 3) + "<br>";

    result += "<br><b>Last modified:</b><br>" + get_modification();
    return result;
}

QString VM_sel_element::get_thumbnail() const
{
    return query_cache_string(m_cache_element, "Thumbnail");
}

QString VM_sel_element::get_modification() const
{
    const IMdl_cache_module* parent_module = get_parent_module_cache_element();
    if (!parent_module)
        return "";

    const time_t date_changed = parent_module->get_timestamp();

    struct std::tm* ptm = nullptr;

    #pragma warning( disable : 4996 )
    ptm = localtime(&date_changed);
    #pragma warning( default : 4996 )

    char buffer[64];
    // Format: Mo, 15 Aug 2009 - 08:20:00 PM
    std::strftime(buffer, 32, "%a, %b %d %Y - %I:%M:%S %p", ptm);
    return buffer;
}

float VM_sel_element::get_search_ranking() const
{
    return m_search_ranking;
}

void VM_sel_element::set_search_ranking(ChangeSearchRankingAccess /*access*/, float value) const
{
    m_search_ranking = value;
    emit search_ranking_changed();
    emit module_hint_changed();
}
