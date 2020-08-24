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


#include "vm_nav_package.h"
#include "../../mdl_browser_node.h"

VM_nav_package::VM_nav_package(QObject* parent, Mdl_browser_node* node)
    : QObject(parent)
    , m_node(node)
    , m_shortMode(false)
    , m_shortModeSaving(0.0f)
{
}

QHash<int, QByteArray> VM_nav_package::roleNames()
{
    static const QHash<int, QByteArray> roles{
        {NameRole, "packageName"},
        {IsModuleRole, "packageIsModule"},
        {IsShadowing, "moduleIsShadowing"},
        {ShadowHintRole, "moduleShadowHint"},
        {ShortModeRole, "packageShortMode"},
        {ShortModeSavingRole, "packageShortModeSaving"},
        {PresentationCountRole, "packagePresentationCount"},
    };
    return roles;
}

QVariant VM_nav_package::data(int role) const
{
    switch (role)
    {
        case VM_nav_package::NameRole: return get_name();
        case VM_nav_package::IsModuleRole: return get_is_module();
        case VM_nav_package::IsShadowing: return get_is_shadowing();
        case VM_nav_package::ShadowHintRole: return get_shadow_hint();
        case VM_nav_package::ShortModeRole: return get_short_mode();
        case VM_nav_package::ShortModeSavingRole: return get_short_mode_saving();
        case VM_nav_package::PresentationCountRole: return get_presentation_count();
        default: return QVariant();
    }
}

QString VM_nav_package::get_name() const 
{
    return m_node->get_entity_name(); 
}

bool VM_nav_package::get_is_module() const 
{ 
    return m_node->get_is_module(); 
}

bool VM_nav_package::get_is_shadowing() const 
{ 
    return m_node->get_module_shadow_count() > 0; 
}

int VM_nav_package::get_presentation_count() const
{
    return static_cast<int>(m_node->get_presentation_count());
}

void VM_nav_package::set_short_mode(bool value)
{
    m_shortMode = value;
    emit short_mode_changed();
}

void VM_nav_package::set_short_mode_saving(float value)
{
    m_shortModeSaving = value;
    emit short_mode_saving_changed();
}

QString VM_nav_package::get_shadow_hint() const
{
    QString text = "<b>This module is shadowing other modules.</b>";

    text += "<br><br>Qualified Module Name:<br>" 
          + QString(m_node->get_qualified_name());

    text += "<br><br>found in the Search Path:<br>" 
          + QString(m_node->get_module_search_path().c_str());

    size_t n = m_node->get_module_shadow_count();
    QString s = QString((n > 1) ? "s" : "");
    text += "<br><br>shadows the module" + s + " in the Search Path" + s + ":";

    for (int i = 0; i < n; ++i)
        text += "<br>" + QString(m_node->get_module_shadow_search_path(i).c_str());

    return text;
}

bool VM_nav_package::setData(const QVariant& value, int role)
{
    switch (role)
    {
        case VM_nav_package::NameRole:
        case VM_nav_package::IsModuleRole:
        case VM_nav_package::IsShadowing:
        case VM_nav_package::ShadowHintRole:
            return false;

        case VM_nav_package::ShortModeRole: 
            set_short_mode(value.toBool());
            return true;

        case VM_nav_package::ShortModeSavingRole: 
            set_short_mode_saving(value.toFloat()); 
            return true;

        case VM_nav_package::PresentationCountRole:
            return false;
    }
    return false;
}
