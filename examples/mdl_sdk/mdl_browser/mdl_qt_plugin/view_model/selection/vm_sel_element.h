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
/// \brief View model class to represent an element (material, function, ...) 
///        in the selection panel.
///        Belongs to the SelElement.qml which defines the visual presentation.


#ifndef MDL_SDK_EXAMPLES_MDL_BROWSER_VM_SEL_ELEMENT_H
#define MDL_SDK_EXAMPLES_MDL_BROWSER_VM_SEL_ELEMENT_H

#include <QObject>
#include <qhash.h>
#include <QVariant>

class IMdl_cache_element;
class IMdl_cache_module;
class Selection_filter_single_token_search;
class Selection_filter_composed_search;

// key pass pattern
// we want to control the access to the ranking of the search results
class ChangeSearchRankingAccess
{
    friend class Selection_filter_single_token_search;
    friend class Selection_filter_composed_search;

    ChangeSearchRankingAccess() {}
    ChangeSearchRankingAccess(const ChangeSearchRankingAccess& other) = delete;
};


class VM_sel_element : public QObject
{
    Q_OBJECT

public:

    enum Element_type
    {
        ET_Undefined = 0,
        ET_Material = 1,
        ET_Function = 2,     // not implemented completely
        ET_Constant = 4,     // not implemented
        ET_All = ET_Constant * 2 - 1
    };
    Q_ENUM(Element_type)

    enum NodeRole
    {
        NameRole = Qt::UserRole,
        ModuleNameRole,
        TypeRole,
        DisplayNameRole,
        AuthorRole,
        DescriptionRole,
        KeywordsRole,
        LocatedInArchiveRole,
        ModuleHintRole,
        ThumbnailRole,
        ModificationDateRole,
        SearchRankingRole
    };
    Q_ENUM(NodeRole)


    explicit VM_sel_element(QObject* parent = nullptr);
    virtual ~VM_sel_element() = default;


    // name of the material, function, ...
    Q_PROPERTY(QString elementName              READ get_name)
        QString get_name() const;

    // name of module this element is exported by
    Q_PROPERTY(QString elementModuleName        READ get_module_name)
        QString get_module_name() const;

    // type of element (material, function, ...)
    Q_PROPERTY(Element_type elementType         READ get_type)
        Element_type get_type() const { return m_type; }

    // display name specified in the annotation
    Q_PROPERTY(QString elementDisplayName       READ get_display_name)
        QString get_display_name() const;

    // comma separated list of keywords specified in the annotation
    Q_PROPERTY(QString elementKeywords          READ get_keywords)
        QString get_keywords() const;

    // author specified in the annotation
    Q_PROPERTY(QString elementAuthor            READ get_author)
        QString get_author() const;

    // description specified in the annotation
    Q_PROPERTY(QString elementDescription       READ get_description)
        QString get_description() const;

    // true if the module this element is exported by is located in an mdl archive
    Q_PROPERTY(bool elementLocatedInArchive     READ get_located_in_archive)
        bool get_located_in_archive() const;

    // tool tip text that shown while hovering over the info icon
    Q_PROPERTY(QString elementModuleHint        READ get_module_hint 
                                                NOTIFY module_hint_changed)
        QString get_module_hint() const;

    // thumbnail path for materials (if available)
    Q_PROPERTY(QString elementThumbnail         READ get_thumbnail)
        QString get_thumbnail() const;

    // date of last modification
    Q_PROPERTY(QString elementModification      READ get_modification)
        QString get_modification() const;

    // ranking of the element in the current search result
    Q_PROPERTY(float elementSearchRanking       READ get_search_ranking
                                                /*no write access from QML*/
                                                NOTIFY search_ranking_changed)
        float get_search_ranking() const;
        void set_search_ranking(ChangeSearchRankingAccess /*access*/, float value) const;


    // get and set data from QML
    void setup(const IMdl_cache_element* cache_element);
    QVariant data(int role) const;
    static QHash<int, QByteArray> roleNames();

    // data object behind this view model
    const IMdl_cache_element* get_cache_element() const { return m_cache_element; }

    // module this material, function or, ... is exporeted from
    const IMdl_cache_module* get_parent_module_cache_element() const;

    // hide the element if it has an anno::hidden
    bool get_is_hidden() const;

signals:
    void module_hint_changed() const;
    void search_ranking_changed() const;

private:
    const IMdl_cache_element* m_cache_element;
    Element_type m_type;
    mutable float m_search_ranking; // not part of the data directly so it can be changed
};

#endif
