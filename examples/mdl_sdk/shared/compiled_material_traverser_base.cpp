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

#include "compiled_material_traverser_base.h"
#include <sstream>

void Compiled_material_traverser_base::traverse(
    const mi::neuraylib::ICompiled_material* material, void* context)
{
    mi::base::Handle<const mi::neuraylib::IExpression_direct_call> body(material->get_body());

    // parameter stage
    stage_begin(material, ES_PARAMETERS, context);
    const mi::Size param_count = material->get_parameter_count();
    for (mi::Size i = 0; i < param_count; ++i)
    {
        const mi::base::Handle<const mi::neuraylib::IValue> arg(material->get_argument(i));
        Parameter parameter(arg.get());
        traverse(material, Traversal_element(&parameter, param_count, i), context);
    }
    stage_end(material, ES_PARAMETERS, context);

    // temporary stage
    stage_begin(material, ES_TEMPORARIES, context);
    const mi::Size temp_count = material->get_temporary_count();
    for (mi::Size i = 0; i < temp_count; ++i)
    {
        const mi::base::Handle<const mi::neuraylib::IExpression> referenced_expression(
            material->get_temporary(i));
        Temporary temporary(referenced_expression.get());
        traverse(material, Traversal_element(&temporary, temp_count, i), context);
    }
    stage_end(material, ES_TEMPORARIES, context);

    // body stage
    stage_begin(material, ES_BODY, context);
    traverse(material, Traversal_element(body.get()), context);
    stage_end(material, ES_BODY, context);

    // mark finished
    stage_begin(material, ES_FINISHED, context);
}

void Compiled_material_traverser_base::traverse(const mi::neuraylib::ICompiled_material* material,
                                                const Traversal_element& element,
                                                void* context)
{
    visit_begin(material, element, context);

    // major cases: parameter, temporary, expression or value
    if (element.expression)
    {
        switch (element.expression->get_kind())
        {
            case mi::neuraylib::IExpression::EK_CONSTANT:
            {
                const mi::base::Handle<const mi::neuraylib::IExpression_constant> expr_const(
                    element.expression->get_interface<const mi::neuraylib::IExpression_constant
                    >());
                const mi::base::Handle<const mi::neuraylib::IValue> value(expr_const->get_value());

                traverse(material, Traversal_element(value.get()), context);
                break;
            }

            case mi::neuraylib::IExpression::EK_DIRECT_CALL:
            {
                const mi::base::Handle<const mi::neuraylib::IExpression_direct_call> expr_dcall(
                    element.expression->get_interface<const mi::neuraylib::IExpression_direct_call
                    >());
                const mi::base::Handle<const mi::neuraylib::IExpression_list> arguments(
                    expr_dcall->get_arguments());

                const mi::Size arg_count = arguments->get_size();
                for (mi::Size i = 0; i < arg_count; ++i)
                {
                    mi::base::Handle<const mi::neuraylib::IExpression> expr(
                        arguments->get_expression(i));

                    visit_child(material, element, arg_count, i, context);
                    traverse(material, Traversal_element(expr.get(), arg_count, i), context);
                }
                break;
            }

            case mi::neuraylib::IExpression::EK_PARAMETER: // nothing special to do
            case mi::neuraylib::IExpression::EK_TEMPORARY: // nothing special to do
            case mi::neuraylib::IExpression::EK_CALL: // will not happen for compiled materials
            case mi::neuraylib::IExpression::EK_FORCE_32_BIT: // not a valid value
                break;
        }
    }

    // major cases: parameter, temporary, expression or value
    else if (element.value)
    {
        switch (element.value->get_kind())
        {
            case mi::neuraylib::IValue::VK_BOOL:
            case mi::neuraylib::IValue::VK_INT:
            case mi::neuraylib::IValue::VK_FLOAT:
            case mi::neuraylib::IValue::VK_DOUBLE:
            case mi::neuraylib::IValue::VK_STRING:
            case mi::neuraylib::IValue::VK_ENUM:
            case mi::neuraylib::IValue::VK_INVALID_DF:
            case mi::neuraylib::IValue::VK_TEXTURE:
            case mi::neuraylib::IValue::VK_LIGHT_PROFILE:
            case mi::neuraylib::IValue::VK_BSDF_MEASUREMENT:
                break; // nothing to do here

            // the following values have children
            case mi::neuraylib::IValue::VK_VECTOR:
            case mi::neuraylib::IValue::VK_MATRIX:
            case mi::neuraylib::IValue::VK_COLOR:
            case mi::neuraylib::IValue::VK_ARRAY:
            case mi::neuraylib::IValue::VK_STRUCT:
            {
                const mi::base::Handle<const mi::neuraylib::IValue_compound> value_compound(
                    element.value->get_interface<const mi::neuraylib::IValue_compound>());

                mi::Size compound_size = value_compound->get_size();
                for (mi::Size i = 0; i < compound_size; ++i)
                {
                    const mi::base::Handle<const mi::neuraylib::IValue> compound_element(
                        value_compound->get_value(i));

                    visit_child(material, element, compound_size, i, context);
                    traverse(material,
                             Traversal_element(compound_element.get(), compound_size, i),
                             context);
                }
                break;
            }

            case mi::neuraylib::IValue::VK_FORCE_32_BIT:  
                break; // not a valid value
        }
    }

    // major cases: parameter, temporary, expression or value
    else if (element.parameter)
    {
        traverse(material, Traversal_element(element.parameter->value), context);
    }

    // major cases: parameter, temporary, expression or value
    else if (element.temporary)
    {
        traverse(material, Traversal_element(element.temporary->expression), context);
    }

    visit_end(material, element, context);
}

std::string Compiled_material_traverser_base::get_parameter_name(
    const mi::neuraylib::ICompiled_material* material, mi::Size index,
    bool* out_generated) const
{
    std::string name = material->get_parameter_name(index);

    // dots in parameter names are not allowed in mdl, so these are the compiler generated ones.
    if (out_generated) *out_generated = name.find('.') != std::string::npos;
    return name;
}

std::string Compiled_material_traverser_base::get_temporary_name(
    const mi::neuraylib::ICompiled_material* /*material*/,
    mi::Size index) const
{
    std::stringstream s;
    s << "temporary_" << index;
    return s.str();
}
