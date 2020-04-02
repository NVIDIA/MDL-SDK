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


#include "vm_nav_stack.h"
#include "vm_nav_stack_level_model.h"
#include "vm_nav_package.h"

#include <mi/mdl_sdk.h>
#include "../../cache/mdl_cache_package.h"
#include "../../mdl_browser_node.h"
#include "vm_nav_stack_level_proxy_model.h"


VM_nav_stack::VM_nav_stack(QObject* parent, Mdl_browser_tree* browser_tree) 
    : QObject(parent)
    , m_browser_tree(browser_tree)
{
}

Q_INVOKABLE VM_nav_stack_level_proxy_model* VM_nav_stack::create_root_level()
{
    auto proxy = new VM_nav_stack_level_proxy_model(this);
    proxy->setSourceModel(new VM_nav_stack_level_model(this, m_browser_tree->get_root()));
    return proxy;
}

Q_INVOKABLE VM_nav_stack_level_proxy_model* VM_nav_stack::expand_package(VM_nav_package* package)
{
    auto proxy = new VM_nav_stack_level_proxy_model(this);
    proxy->setSourceModel(new VM_nav_stack_level_model(this, package->get_browser_node()));
    return proxy;
}

Q_INVOKABLE void VM_nav_stack::dispose_level(VM_nav_stack_level_proxy_model* level)
{
    // because of the passed this pointer in the methods above,
    // we don't need to free levels
    // to save memory in case of heavy use, we should do it manually
}

Q_INVOKABLE void VM_nav_stack::set_current_level(VM_nav_stack_level_proxy_model* current)
{
    if (!current)
    {
        std::cerr << "[VM_nav_stack] set_current: passing a 'nullptr' is not allowed.\n";
        return;
    }

    m_current_level = current;

    auto model = dynamic_cast<VM_nav_stack_level_model*>(m_current_level->sourceModel());

    //std::cout << "[VM_nav_stack] set_current_level: current level is: " 
    //          << model->get_current_package()->get_name().toUtf8().data() << "\n";
    emit selected_package_changed(model->get_current_package());
}

Q_INVOKABLE void VM_nav_stack::set_selected_module(VM_nav_package* m)
{
    if (!m)
    {
        //std::cout << "[VM_nav_stack] set_selected_module: to none\n";
        emit selected_module_changed(nullptr);
        return;
    }
    
    //std::cout << "[VM_nav_stack] set_selected_module: current module is: " 
    //          << m->get_name().toUtf8().data() << "\n";

    emit selected_module_changed(m);
}

Q_INVOKABLE void VM_nav_stack::update_presentation_counters()
{
    if (!m_current_level) return;

    auto model = dynamic_cast<VM_nav_stack_level_model*>(m_current_level->sourceModel());
    model->update_presentation_counters();
    m_current_level->invalidate();
}
