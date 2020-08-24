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
#include "traversal.h"
#include "errors.h"
#include <vector>
#include <string>

using namespace i18n;
using mi::base::Handle;
using mi::neuraylib::INeuray;
using mi::neuraylib::IModule;
using mi::neuraylib::IAnnotation;
using mi::neuraylib::IAnnotation_list;
using mi::neuraylib::IAnnotation_block;
using mi::neuraylib::IExpression_list;
using mi::neuraylib::IExpression;
using mi::neuraylib::IFunction_definition;
using mi::neuraylib::IMaterial_definition;
using mi::neuraylib::IValue_string;
using mi::neuraylib::IValue;
using mi::neuraylib::IExpression_constant;
using mi::neuraylib::IType;
using std::string;
using std::vector;

void Annotation_traversal::handle_annotation(const IAnnotation* anno)
{
    const char* name = anno->get_name();
    if (!name)
    {
        return;
    }

    Handle<const IExpression_list> elist(anno->get_arguments());

    for (mi::Size i = 0; i < elist->get_size(); i++)
    {
        Handle<const IExpression> expr(elist->get_expression(mi::Size(i)));
        check_success(expr);
        const IExpression::Kind kind = expr->get_kind();
        if (kind == IExpression::EK_CONSTANT)
        {
            Handle<const IExpression_constant> expr_constant(
                expr->get_interface<IExpression_constant>());
            check_success(expr_constant);
            Handle<const IValue> value(expr_constant->get_value());
            check_success(value);

            if (value->get_kind() == IValue::VK_STRING)
            {
                Handle<const IValue_string> value_string(value->get_interface<IValue_string>());
                check_success(value_string);
                const char* char_value = value_string->get_value();
                check_success(char_value);
                if (m_context)
                {
                    string note(name);
                    const char * qname(m_context->top_qualified_name());
                    if (qname)
                    {
                        note += string(" ") + qname;
                    }
                    m_context->push_annotation(name, char_value, note.c_str());
                }
            }
        }
    }
}

void Annotation_traversal::handle_annotation_block(const IAnnotation_block* ablock)
{
    for (mi::Size i = 0; i < ablock->get_size(); i++)
    {
        Handle<const IAnnotation> anno(ablock->get_annotation(i));
        if (anno)
        {
            handle_annotation(anno.get());
        }
    }
}

void Annotation_traversal::handle_annotation_list(const IAnnotation_list* o)
{
    if (o)
    {
        for (mi::Size i = 0; i < o->get_size(); i++)
        {
            Handle<const IAnnotation_block> anno(o->get_annotation_block(i));
            if (anno)
            {
                handle_annotation_block(anno.get());
            }
        }
    }
}

void Annotation_traversal::handle_function_definition(const IFunction_definition* o)
{
    const char * name(o->get_mdl_name());
    if (!name)
    {
        return;
    }

    // Filter out conversion oprators (e.g. int() for enums)
    const IFunction_definition::Semantics s(o->get_semantic());
    if(s == IFunction_definition::DS_CONV_OPERATOR)
    {
        return;
    }

    if (m_context)
    {
        m_context->push_qualified_name(name);
    }

    {
        Handle<const IAnnotation_block> ablock(o->get_annotations());
        if (ablock)
        {
            handle_annotation_block(ablock.get());
        }
    }
    {
        Handle<const IAnnotation_block> ablock(o->get_return_annotations());
        if (ablock)
        {
            handle_annotation_block(ablock.get());
        }
    }
    {
        Handle<const IAnnotation_list> alist(o->get_parameter_annotations());
        if (alist)
        {
            handle_annotation_list(alist.get());
        }
    }

    if (m_context)
    {
        m_context->pop_qualified_name();
    }
}

void Annotation_traversal::handle_material_definition(const IMaterial_definition* o)
{
    if (m_context)
    {
        m_context->push_qualified_name(o->get_mdl_name());
    }

    {
        Handle<const IAnnotation_block> ablock(o->get_annotations());
        if (ablock)
        {
            handle_annotation_block(ablock.get());
        }
    }
    {
        Handle<const IAnnotation_list> alist(o->get_parameter_annotations());
        if (alist)
        {
            handle_annotation_list(alist.get());
        }
    }
    if (m_context)
    {
        m_context->pop_qualified_name();
    }
}

void Annotation_traversal::handle_type(const IType* o, const char * name)
{
    if (m_context)
    {
        if (name)
        {
            m_context->push_qualified_name(name);
        }
    }
    Handle<const mi::neuraylib::IType_enum> type_enum(
        o->get_interface<mi::neuraylib::IType_enum>());
    if (type_enum)
    {
        Handle<const IAnnotation_block> ab(type_enum->get_annotations());
        if (ab)
        {
            handle_annotation_block(ab.get());
        }
        mi::Size vcount(type_enum->get_size());
        for (mi::Size i = 0; i < vcount; i++)
        {
            Handle<const IAnnotation_block> av(type_enum->get_value_annotations(i));
            if (av)
            {
                handle_annotation_block(av.get());
            }
        }
    }
    Handle<const mi::neuraylib::IType_struct> type_struct(
        o->get_interface<mi::neuraylib::IType_struct>());
    if (type_struct)
    {
        Handle<const IAnnotation_block> ab(type_struct->get_annotations());
        if (ab)
        {
            handle_annotation_block(ab.get());
        }
        mi::Size vcount(type_struct->get_size());
        for (mi::Size i = 0; i < vcount; i++)
        {
            Handle<const IAnnotation_block> af(type_struct->get_field_annotations(i));
            if (af)
            {
                handle_annotation_block(af.get());
            }
        }
    }
    if (m_context)
    {
        m_context->pop_qualified_name();
    }
}

void Annotation_traversal::handle_module(const IModule* module)
{
    check_success(module);
    check_success(m_transaction);

    Handle<const IAnnotation_block> ablock(module->get_annotations());
    if (ablock)
    {
        handle_annotation_block(ablock.get());
    }

    /// Returns the types exported by this module.
    Handle<const mi::neuraylib::IType_list> types(module->get_types());
    if (types)
    {
        mi::Size count(types->get_size());
        for (mi::Size i = 0; i < count; i++)
        {
            const char* name(types->get_name(i));
            Handle<const mi::neuraylib::IType> type(types->get_type(i));
            if (type)
            {
                handle_type(type.get(), name);
            }
        }
    }

    /// Returns the constants exported by this module.
    Handle<const mi::neuraylib::IValue_list> constants(module->get_constants());
    if (constants)
    {
        mi::Size count(constants->get_size());
        for (mi::Size i = 0; i < count; i++)
        {
            const char* name(constants->get_name(i));
            if (name)
            {
                // Nothing to do 
            }
        }
    }

    {
        mi::Size count(module->get_function_count());
        for (mi::Size i = 0; i < count; i++)
        {
            const char * name(module->get_function(i));
            if (name)
            {
                mi::base::Handle<const IFunction_definition> o(
                    m_transaction->access<IFunction_definition>(name));
                check_success(o);
                handle_function_definition(o.get());
            }
        }
    }

    for (mi::Size i = 0; i < module->get_material_count(); i++)
    {
        const char * name(module->get_material(i));
        if (name)
        {
            mi::base::Handle<const IMaterial_definition> o(
                m_transaction->access<IMaterial_definition>(name));
            check_success(o);
            handle_material_definition(o.get());
        }
    }
}
