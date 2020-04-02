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
 ******************************************************************************/
#pragma once

#include <mi/mdl_sdk.h>

namespace i18n
{
    // A base class to gather information about annotations when traversing MDL elements
    class Traversal_context
    {
    public:
        virtual void push_annotation(const char* name, const char* value, const char* note = NULL)
        {}
        virtual void push_qualified_name(const char* name)
        {}
        virtual void pop_qualified_name()
        {}
        virtual const char* top_qualified_name() const
        {
            return NULL;
        }
    };

    // A helper class to traverse MDL elements
    class Annotation_traversal
    {
        mi::base::Handle<mi::neuraylib::ITransaction> m_transaction;
        Traversal_context * m_context;
    public:
        Annotation_traversal(mi::neuraylib::ITransaction * transaction)
            : m_transaction(mi::base::make_handle_dup(transaction))
            , m_context(NULL)
        {}
        void set_context(Traversal_context * context)
        {
            m_context = context;
        }
        void handle_module(const mi::neuraylib::IModule* module);
        void handle_material_definition(const mi::neuraylib::IMaterial_definition* o);
        void handle_function_definition(const mi::neuraylib::IFunction_definition* o);
        void handle_annotation_block(const mi::neuraylib::IAnnotation_block* ablock);
        void handle_annotation_list(const mi::neuraylib::IAnnotation_list* o);
        void handle_annotation(const mi::neuraylib::IAnnotation* anno);
        void handle_type(const mi::neuraylib::IType* o, const char * name);

    };

} // namespace i18n
