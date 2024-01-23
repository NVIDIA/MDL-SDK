/******************************************************************************
 * Copyright (c) 2018-2024, NVIDIA CORPORATION. All rights reserved.
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

// Code shared by distilling MDL SDK examples

#ifndef EXAMPLE_DISTILLING_SHARED_H
#define EXAMPLE_DISTILLING_SHARED_H

#include "example_shared.h"
#include <string>

// Create IData value
template <typename T>
mi::IData* create_value(
    mi::neuraylib::ITransaction* transaction,
    const char* type_name,
    const T& default_value)
{
    mi::base::Handle<mi::base::IInterface> value(
        transaction->create(type_name));

    mi::base::Handle<mi::IData> data(value->get_interface<mi::IData>());
    mi::set_value(data.get(), default_value);

    data->retain();
    return data.get();
}

// Converts the given expression to a direct_call, thereby resolving temporaries
// Returns nullptr if the passed expression is not a direct call
const mi::neuraylib::IExpression_direct_call* to_direct_call(
    const mi::neuraylib::IExpression* expr,
    const mi::neuraylib::ICompiled_material* cm)
{
    if (!expr) return nullptr;
    switch (expr->get_kind())
    {
    case mi::neuraylib::IExpression::EK_DIRECT_CALL:
        return expr->get_interface<mi::neuraylib::IExpression_direct_call>();
    case mi::neuraylib::IExpression::EK_TEMPORARY:
    {
        mi::base::Handle<const mi::neuraylib::IExpression_temporary> expr_temp(
            expr->get_interface<const mi::neuraylib::IExpression_temporary>());
        mi::base::Handle<const mi::neuraylib::IExpression_direct_call> ref_expr(
            cm->get_temporary<const mi::neuraylib::IExpression_direct_call>(
                expr_temp->get_index()));
        ref_expr->retain();
        return ref_expr.get();
    }
    default:
        break;
    }
    return nullptr;
}

// Returns the semantic of the function definition which corresponds to
// the given call or DS::UNKNOWN in case the expression
// is nullptr
mi::neuraylib::IFunction_definition::Semantics get_call_semantic(
    mi::neuraylib::ITransaction* transaction,
    const mi::neuraylib::IExpression_direct_call* call)
{
    if (!(call))
        return mi::neuraylib::IFunction_definition::DS_UNKNOWN;

    mi::base::Handle<const mi::neuraylib::IFunction_definition> mdl_def(
        transaction->access<const mi::neuraylib::IFunction_definition>(
            call->get_definition()));
    check_success(mdl_def.is_valid_interface());

    return mdl_def->get_semantic();
}

// Returns the argument 'argument_name' of the given call
const mi::neuraylib::IExpression_direct_call* get_argument_as_call(
    const mi::neuraylib::ICompiled_material* cm,
    const mi::neuraylib::IExpression_direct_call* call,
    const char* argument_name)
{
    if (!call) return nullptr;

    mi::base::Handle<const mi::neuraylib::IExpression_list> call_args(
        call->get_arguments());
    mi::base::Handle<const mi::neuraylib::IExpression> arg(
        call_args->get_expression(argument_name));
    return to_direct_call(arg.get(), cm);
}

// Looks up the sub expression 'path' within the compiled material
// starting at parent_call. If parent_call is nullptr,the
// material will be traversed from the root
const mi::neuraylib::IExpression_direct_call* lookup_call(
    const std::string& path,
    const mi::neuraylib::ICompiled_material* cm,
    const mi::neuraylib::IExpression_direct_call *parent_call = nullptr
)
{
    mi::base::Handle<const mi::neuraylib::IExpression_direct_call>
        result_call;

    if (parent_call == nullptr)
    {
        mi::base::Handle<const mi::neuraylib::IExpression> expr(
            cm->lookup_sub_expression(path.c_str()));
        result_call = to_direct_call(expr.get(), cm);
    }
    else
    {
        result_call = mi::base::make_handle_dup(parent_call);

        std::string remaining_path = path;
        std::size_t p = 0;
        do
        {
            std::string arg = remaining_path;
            p = remaining_path.find(".");
            if (p != std::string::npos)
            {
                arg = remaining_path.substr(0, p);
                remaining_path = remaining_path.substr(p + 1, remaining_path.size() - p - 1);
            }
            result_call = get_argument_as_call(cm, result_call.get(), arg.c_str());
            if (!result_call)
                return nullptr;
        } while (p != std::string::npos);
    }
    result_call->retain();
    return result_call.get();
}

#endif // EXAMPLE_DISTILLING_SHARED_H
