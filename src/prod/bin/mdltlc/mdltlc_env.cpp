/******************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION. All rights reserved.
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

#include "pch.h"

#include "mdltlc_env.h"

bool Environment::bind(Symbol const *name, Type *type) {
    Type_map::iterator it = m_type_map.find(name);
    if (it == m_type_map.end()) {
        Type_list types(m_arena.get_allocator());
        types.push_back(type);
        m_type_map.insert({name, types});
        return true;
    }
    it->second.push_back(type);
    return false;
}

Environment::Type_list *Environment::find(
    Symbol const *name,
    Environment **binding_env)
{
    Type_map::iterator it = m_type_map.find(name);
    if (it == m_type_map.end()) {
        if (m_enclosing)
            return m_enclosing->find(name, binding_env);
        return nullptr;
    }
    if (binding_env)
        *binding_env = this;
    return &it->second;
}

void Environment::pp(pp::Pretty_print &p) {
    switch (m_kind) {
    case ENV_BUILTIN:
        p.string("-| builtin environment |-");
        break;
    case ENV_ATTRIBUTE:
        p.string("-| attribute environment |-");
        break;
    case ENV_LOCAL:
        p.string("-| local environment |-");
        break;
    }
    p.with_indent([&] (pp::Pretty_print &p) {
            p.nl();
            for (Type_map::iterator it = m_type_map.begin();
                 it != m_type_map.end();
                 ++it) {
                p.string(it->first->get_name());
                p.colon();
                p.space();
                p.with_indent([&] (pp::Pretty_print &p) {
                        bool first = true;
                        for (Type_list::iterator i = it->second.begin();
                             i != it->second.end();
                             ++i) {
                            if (first) {
                                first = false;
                            } else {
                                p.comma();
                                p.space();
                            }
                            (*i)->pp(p);
                        };
                    });
                p.nl();
            }
        });
}

Environment *Environment::attribute_environment() {
    if (m_kind == Kind::ENV_ATTRIBUTE)
        return this;
    if (m_enclosing)
        return m_enclosing->attribute_environment();
    return nullptr;
}

