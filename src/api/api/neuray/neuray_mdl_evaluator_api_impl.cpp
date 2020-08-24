/***************************************************************************************************
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
 **************************************************************************************************/

/** \file
 ** \brief Source for the IMdl_evaluator_api implementation.
 **/

#include "pch.h"

#include <string>
#include <map>

#include <mi/neuraylib/iexpression.h>
#include <mi/neuraylib/itype.h>
#include <mi/neuraylib/ivalue.h>
#include <mi/neuraylib/itransaction.h>

#include <api/api/neuray/neuray_expression_impl.h>

#include <mi/base/handle.h>

#include <base/lib/log/i_log_assert.h>

#include <io/scene/mdl_elements/i_mdl_elements_utilities.h>
#include <io/scene/mdl_elements/i_mdl_elements_function_definition.h>
#include <io/scene/mdl_elements/i_mdl_elements_function_call.h>
#include <io/scene/mdl_elements/i_mdl_elements_expression.h>
#include <io/scene/mdl_elements/i_mdl_elements_material_instance.h>

#include <base/data/db/i_db_access.h>
#include <base/data/db/i_db_transaction.h>

#include <mdl/integration/mdlnr/i_mdlnr.h>

#include <mdl/compiler/compilercore/compilercore_mdl.h>
#include <mdl/compiler/compilercore/compilercore_allocator.h>
#include <mdl/compiler/compilercore/compilercore_factories.h>
#include <mdl/compiler/compilercore/compilercore_symbols.h>
#include <mdl/compiler/compilercore/compilercore_tools.h>

#include "neuray_mdl_evaluator_api_impl.h"
#include "neuray_transaction_impl.h"
#include "neuray_material_instance_impl.h"
#include "neuray_function_call_impl.h"

namespace MI {
namespace NEURAY {

namespace {

class IParameter_helper {
public:
    /// Get the argument for the parameter with given index.
    virtual MI::MDL::IExpression const *get_parameter_argument(size_t index) = 0;
};

class Parameter_helper final : public IParameter_helper {
public:
    /// Constructor.
    template<typename T>
    Parameter_helper(
        T const *entity)
    : m_arguments(entity->get_arguments())
    {
    }

    /// Get the argument for the parameter with given index.
    virtual MI::MDL::IExpression const *get_parameter_argument(size_t index) final
    {
        return m_arguments->get_expression(index);
    }

private:
    mi::base::Handle<MI::MDL::IExpression_list const> m_arguments;
};

/// Helper class to evaluate a neuray expression.
class Evaluator {
public:
    enum Err_codes {
        EC_OK                = 0,
        EC_TEMPORARY         = 1,   ///< temporaries not supported
        EC_NOT_CONSTANT      = 2,   ///< not a enhanced constant expression
        EC_UNSUPPORTED       = 3,   ///< unsupported expression
        EC_NON_FUNCTION_CALL = 4,   ///< called object is not a function
        EC_PARAMETER         = 5,   ///< parameter argument could not be found
        EC_MEMORY_EXHAUSTED  = 6,   ///< Memory size threshold reached
        EC_CYCLES_EXHAUSTED  = 7,   ///< computational time threshold reached
        EC_INVALID_CALL      = 8,   ///< call points to invalid definition
    };

public:
    /// Constructor.
    ///
    /// \param compiler  the MDL compiler
    /// \param trans     the current transaction
    /// \param param_cb  if non-NULL, a callback to retrieve parameter arguments
    explicit Evaluator(
        mi::mdl::IMDL               *compiler,
        mi::neuraylib::ITransaction *trans,
        IParameter_helper           *param_cb)
    : m_compiler(mi::mdl::impl_cast<mi::mdl::MDL>(compiler))
    , m_trans(nullptr)
    , m_param_cp(param_cb)
    , m_arena(m_compiler->get_allocator())
    , m_sym_tab(m_arena)
    , m_type_fact(m_arena, m_compiler, &m_sym_tab)
    , m_value_fact(m_arena, m_type_fact)
    , m_user_types()
    , m_max_size(8*1024*1024)
    , m_max_cycles(1024)
    , m_error(EC_OK)
    {
        Transaction_impl *transaction_impl = static_cast<Transaction_impl*>(trans);
        m_trans = transaction_impl->get_db_transaction();
    }

    /// Evaluate a function call.
    mi::mdl::IValue const *evaluate_call(
        MI::MDL::Mdl_function_definition const *def,
        MI::MDL::IExpression_list const        *args)
    {
        if (m_max_cycles == 0) {
            set_error(EC_CYCLES_EXHAUSTED);
            return m_value_fact.create_bad();
        }
        --m_max_cycles;

        mi::mdl::IDefinition::Semantics sema = def->get_mdl_semantic();

        bool strict = true;

        // check if we can evaluate known semantics
        switch (sema) {
        case mi::mdl::IDefinition::DS_INTRINSIC_TEX_TEXTURE_ISVALID:
        case mi::mdl::IDefinition::DS_INTRINSIC_DF_LIGHT_PROFILE_ISVALID:
        case mi::mdl::IDefinition::DS_INTRINSIC_DF_BSDF_MEASUREMENT_ISVALID:
            // all *_isvalid functions are supported, but strict
            strict = true; //-V1048 //-V1037 PVS
            break;
        case mi::mdl::IDefinition::DS_CONV_OPERATOR:
            // type conversion is supported, but strict
            strict = true; //-V1048 PVS
            break;
        default:
            if (mi::mdl::semantic_is_operator(sema)) {
                // operators are constexpr, some are non-strict
                strict = false;
                break;
            }
            if (mi::mdl::is_math_semantics(sema)) {
                // all math functions are constexpr, but strict
                strict = true; //-V1048 PVS
                break;
            }
            // unsupported so far
            set_error(EC_UNSUPPORTED);
            return m_value_fact.create_bad();
        }

        size_t n_args = args->get_size();
        MI::MDL::Small_VLA<mi::mdl::IValue const *, 8> values(n_args);

        for (size_t i = 0; i < n_args; ++i) {
            mi::base::Handle<MI::MDL::IExpression const> arg(args->get_expression(i));

            mi::mdl::IValue const *v = evaluate(arg.get());

            if (strict && mi::mdl::is<mi::mdl::IValue_bad>(v)) {
                // error already reported
                return v;
            }
            values[i] = v;
        }

        switch (sema) {
        case mi::mdl::IDefinition::DS_INTRINSIC_TEX_TEXTURE_ISVALID:
        case mi::mdl::IDefinition::DS_INTRINSIC_DF_LIGHT_PROFILE_ISVALID:
        case mi::mdl::IDefinition::DS_INTRINSIC_DF_BSDF_MEASUREMENT_ISVALID:
            // just check if we have a valid resource
            ASSERT(M_MDLC, n_args == 1);
            if (mi::mdl::is<mi::mdl::IValue_invalid_ref>(values[0])) {
                // invalid
                return m_value_fact.create_bool(false);
            } else {
                // valid resource
                return m_value_fact.create_bool(true);
            }
            break;
        case mi::mdl::IDefinition::DS_CONV_OPERATOR:
        case mi::mdl::IDefinition::DS_CONV_CONSTRUCTOR:
            ASSERT(M_MDLC, n_args == 1);
            return values[0]->convert(&m_value_fact, def->get_mdl_return_type(m_trans));
        default:
            break;
        }

        if (mi::mdl::semantic_is_operator(sema)) {
            mi::mdl::IExpression::Operator op = mi::mdl::semantic_to_operator(sema);

            switch (op) {
            // unary ops
            case mi::mdl::IExpression::OK_BITWISE_COMPLEMENT:
                ASSERT(M_MDLC, n_args == 1);
                return values[0]->bitwise_not(&m_value_fact);
            case mi::mdl::IExpression::OK_LOGICAL_NOT:
                ASSERT(M_MDLC, n_args == 1);
                return values[0]->logical_not(&m_value_fact);
            case mi::mdl::IExpression::OK_POSITIVE:
                ASSERT(M_MDLC, n_args == 1);
                return values[0];
            case mi::mdl::IExpression::OK_NEGATIVE:
                ASSERT(M_MDLC, n_args == 1);
                return values[0]->minus(&m_value_fact);
            case mi::mdl::IExpression::OK_PRE_INCREMENT:
            case mi::mdl::IExpression::OK_PRE_DECREMENT:
            case mi::mdl::IExpression::OK_POST_INCREMENT:
            case mi::mdl::IExpression::OK_POST_DECREMENT:
                // these should not occur here
                ASSERT(M_MDLC, !"unexpected increment/decrement operators");
                return m_value_fact.create_bad();
            case mi::mdl::IExpression::OK_CAST:
                // unfortunately the interface does not give the MDL result type,
                // however, the DAR-IR always handles casts
                return m_value_fact.create_bad();

            // binary ops
            case mi::mdl::IExpression::OK_SELECT:
                // these should not occur here
                ASSERT(M_MDLC, !"unexpected select");
                return m_value_fact.create_bad();

            case mi::mdl::IExpression::OK_ARRAY_INDEX:
                ASSERT(M_MDLC, n_args == 2);
                {
                    mi::mdl::IValue_array const *a_v =
                        mi::mdl::as<mi::mdl::IValue_array>(values[0]);
                    mi::mdl::IValue_int const *i_v =
                        mi::mdl::as<mi::mdl::IValue_int>(values[1]);

                    if (a_v == nullptr || i_v == nullptr) {
                        return m_value_fact.create_bad();
                    }
                    int index = i_v->get_value();

                    if (index < 0 || index >= a_v->get_component_count()) {
                        // out of bounds
                        return m_value_fact.create_bad();
                    }
                    return a_v->get_value(index);
                }
            case mi::mdl::IExpression::OK_MULTIPLY:
                ASSERT(M_MDLC, n_args == 2);
                return values[0]->multiply(&m_value_fact, values[1]);
            case mi::mdl::IExpression::OK_DIVIDE:
                ASSERT(M_MDLC, n_args == 2);
                return values[0]->divide(&m_value_fact, values[1]);
            case mi::mdl::IExpression::OK_MODULO:
                ASSERT(M_MDLC, n_args == 2);
                return values[0]->modulo(&m_value_fact, values[1]);
            case mi::mdl::IExpression::OK_PLUS:
                ASSERT(M_MDLC, n_args == 2);
                return values[0]->add(&m_value_fact, values[1]);
            case mi::mdl::IExpression::OK_MINUS:
                ASSERT(M_MDLC, n_args == 2);
                return values[0]->sub(&m_value_fact, values[1]);
            case mi::mdl::IExpression::OK_SHIFT_LEFT:
                ASSERT(M_MDLC, n_args == 2);
                return values[0]->shl(&m_value_fact, values[1]);
            case mi::mdl::IExpression::OK_SHIFT_RIGHT:
                ASSERT(M_MDLC, n_args == 2);
                return values[0]->asr(&m_value_fact, values[1]);
            case mi::mdl::IExpression::OK_UNSIGNED_SHIFT_RIGHT:
                ASSERT(M_MDLC, n_args == 2);
                return values[0]->lsr(&m_value_fact, values[1]);
            case mi::mdl::IExpression::OK_LESS:
                {
                    ASSERT(M_MDLC, n_args == 2);
                    mi::mdl::IValue::Compare_results cr = values[0]->compare(values[1]);
                    return m_value_fact.create_bool(
                        cr == mi::mdl::IValue::CR_LT);
                }
            case mi::mdl::IExpression::OK_LESS_OR_EQUAL:
                {
                    ASSERT(M_MDLC, n_args == 2);
                    mi::mdl::IValue::Compare_results cr = values[0]->compare(values[1]);
                    return m_value_fact.create_bool(
                        (cr & mi::mdl::IValue::CR_LE) != 0 && (cr & mi::mdl::IValue::CR_UO) == 0);
                }
            case mi::mdl::IExpression::OK_GREATER_OR_EQUAL:
                {
                    ASSERT(M_MDLC, n_args == 2);
                    mi::mdl::IValue::Compare_results cr = values[0]->compare(values[1]);
                    return m_value_fact.create_bool(
                        (cr & mi::mdl::IValue::CR_GE) != 0 && (cr & mi::mdl::IValue::CR_UO) == 0);
                }
            case mi::mdl::IExpression::OK_GREATER:
                {
                    ASSERT(M_MDLC, n_args == 2);
                    mi::mdl::IValue::Compare_results cr = values[0]->compare(values[1]);
                    return m_value_fact.create_bool(
                        cr == mi::mdl::IValue::CR_GT);
                }
            case mi::mdl::IExpression::OK_EQUAL:
                {
                    ASSERT(M_MDLC, n_args == 2);
                    mi::mdl::IValue::Compare_results cr = values[0]->compare(values[1]);
                    return m_value_fact.create_bool(
                        cr == mi::mdl::IValue::CR_EQ);
                }
            case mi::mdl::IExpression::OK_NOT_EQUAL:
                {
                    ASSERT(M_MDLC, n_args == 2);
                    mi::mdl::IValue::Compare_results cr = values[0]->compare(values[1]);
                    return m_value_fact.create_bool(
                        (cr & mi::mdl::IValue::CR_UEQ) == 0);
                }
            case mi::mdl::IExpression::OK_BITWISE_AND:
                ASSERT(M_MDLC, n_args == 2);
                return values[0]->bitwise_and(&m_value_fact, values[1]);
            case mi::mdl::IExpression::OK_BITWISE_XOR:
                ASSERT(M_MDLC, n_args == 2);
                return values[0]->bitwise_xor(&m_value_fact, values[1]);
            case mi::mdl::IExpression::OK_BITWISE_OR:
                ASSERT(M_MDLC, n_args == 2);
                return values[0]->bitwise_or(&m_value_fact, values[1]);
            case mi::mdl::IExpression::OK_LOGICAL_AND:
                ASSERT(M_MDLC, n_args == 2);
                return values[0]->logical_and(&m_value_fact, values[1]);
            case mi::mdl::IExpression::OK_LOGICAL_OR:
                ASSERT(M_MDLC, n_args == 2);
                return values[0]->logical_or(&m_value_fact, values[1]);

            case mi::mdl::IExpression::OK_ASSIGN:
            case mi::mdl::IExpression::OK_MULTIPLY_ASSIGN:
            case mi::mdl::IExpression::OK_DIVIDE_ASSIGN:
            case mi::mdl::IExpression::OK_MODULO_ASSIGN:
            case mi::mdl::IExpression::OK_PLUS_ASSIGN:
            case mi::mdl::IExpression::OK_MINUS_ASSIGN:
            case mi::mdl::IExpression::OK_SHIFT_LEFT_ASSIGN:
            case mi::mdl::IExpression::OK_SHIFT_RIGHT_ASSIGN:
            case mi::mdl::IExpression::OK_UNSIGNED_SHIFT_RIGHT_ASSIGN:
            case mi::mdl::IExpression::OK_BITWISE_OR_ASSIGN:
            case mi::mdl::IExpression::OK_BITWISE_XOR_ASSIGN:
            case mi::mdl::IExpression::OK_BITWISE_AND_ASSIGN:
                // assignments should not occur
                ASSERT(M_MDLC, !"unexpected assignment operator");
                return m_value_fact.create_bad();
            case mi::mdl::IExpression::OK_SEQUENCE:
                // sequence should not occur
                ASSERT(M_MDLC, !"unexpected sequence operator");
                return m_value_fact.create_bad();

            // ternary
            case mi::mdl::IExpression::OK_TERNARY:
                ASSERT(M_MDLC, n_args == 3);
                if (mi::mdl::IValue_bool const *b = mi::mdl::as<mi::mdl::IValue_bool>(values[0])) {
                    return b->get_value() ? values[1] : values[2];
                }
                return m_value_fact.create_bad();

            // variadic
            case mi::mdl::IExpression::OK_CALL:
                // call operator should not occur
                ASSERT(M_MDLC, !"unexpected call operator");
                return m_value_fact.create_bad();
            }
        } else if (mi::mdl::is_math_semantics(sema)) {
            return mi::mdl::evaluate_intrinsic_function(
                &m_value_fact,
                sema,
                values.data(),
                values.size());
        }

        // should not reach this
        ASSERT(M_MDLC, !"unreachable code reached");
        set_error(EC_UNSUPPORTED);
        return m_value_fact.create_bad();
    }

    /// Evaluate a neuray expression.
    mi::mdl::IValue const *evaluate(mi::neuraylib::IExpression const *expr)
    {
        switch (expr->get_kind()) {
        case mi::neuraylib::IExpression::EK_CONSTANT:
            {
                mi::base::Handle<mi::neuraylib::IExpression_constant const> c(
                    expr->get_interface<mi::neuraylib::IExpression_constant>());
                Expression_constant const *c_imp =
                    static_cast<Expression_constant const *>(c.get());

                return evaluate(c_imp);
            }
            break;
        case mi::neuraylib::IExpression::EK_CALL:
            {
                mi::base::Handle<mi::neuraylib::IExpression_call const> call(
                    expr->get_interface<mi::neuraylib::IExpression_call>());
                Expression_call const *c_impl =
                    static_cast<Expression_call const *>(call.get());
                return evaluate(c_impl);
            }
        case mi::neuraylib::IExpression::EK_PARAMETER:
            {
                mi::base::Handle<mi::neuraylib::IExpression_parameter const> param(
                    expr->get_interface<mi::neuraylib::IExpression_parameter>());
                Expression_parameter const *p_impl =
                    static_cast<Expression_parameter const *>(param.get());
                return evaluate(p_impl);
            }
        case mi::neuraylib::IExpression::EK_DIRECT_CALL:
            {
                mi::base::Handle<mi::neuraylib::IExpression_direct_call const> dcall(
                    expr->get_interface<mi::neuraylib::IExpression_direct_call>());
                Expression_direct_call const *d_impl =
                    static_cast<Expression_direct_call const *>(dcall.get());
                return evaluate(d_impl);
            }
        case mi::neuraylib::IExpression::EK_TEMPORARY:
            {
                mi::base::Handle<mi::neuraylib::IExpression_temporary const> temp(
                    expr->get_interface<mi::neuraylib::IExpression_temporary>());
                Expression_temporary const *t_impl =
                    static_cast<Expression_temporary const *>(temp.get());
                return evaluate(t_impl);
            }
        case mi::neuraylib::IExpression::EK_FORCE_32_BIT:
            break;
        }
        ASSERT(M_MDLC, !"unsupported expression kind");
        set_error(EC_UNSUPPORTED);
        return m_value_fact.create_bad();
    }

    /// Get a parameter argument expression.
    MI::MDL::IExpression const *get_parameter_argument(size_t index)
    {
        if (m_param_cp != nullptr)
            return m_param_cp->get_parameter_argument(index);

        // no callback provided
        return nullptr;
    }

    /// Evaluate an internal neuray expression.
    mi::mdl::IValue const *evaluate(MI::MDL::IExpression const *expr)
    {
        if (m_arena.get_chunks_size() > m_max_size) {
            set_error(EC_MEMORY_EXHAUSTED);
            return m_value_fact.create_bad();
        }
        switch (expr->get_kind()) {
        case MI::MDL::IExpression::EK_CONSTANT:
            {
                mi::base::Handle<MI::MDL::IExpression_constant const> c(
                    expr->get_interface<MI::MDL::IExpression_constant>());
                mi::base::Handle<MI::MDL::IValue const> v(c->get_value());

                return evaluate(v.get());
            }
            break;
        case MI::MDL::IExpression::EK_CALL:
            {
                mi::base::Handle<MI::MDL::IExpression_call const> call(
                    expr->get_interface<MI::MDL::IExpression_call>());
                DB::Tag tag = call->get_call();
                SERIAL::Class_id class_id = m_trans->get_class_id(tag);

                if (class_id != MI::MDL::Mdl_function_call::id) {
                    set_error(EC_NON_FUNCTION_CALL);
                    return m_value_fact.create_bad();
                }
                DB::Access<MI::MDL::Mdl_function_call> fcall(tag, m_trans);
                DB::Tag def_tag = fcall->get_function_definition(m_trans);
                if (!def_tag.is_valid()) {
                    set_error(EC_INVALID_CALL);
                    return m_value_fact.create_bad();
                }
                DB::Access<MI::MDL::Mdl_function_definition> def(def_tag, m_trans);

                mi::base::Handle<MI::MDL::IExpression_list const> args(
                    fcall->get_arguments());

                return evaluate_call(def.get_ptr(), args.get());
            }
        case MI::MDL::IExpression::EK_PARAMETER:
            {
                mi::base::Handle<MI::MDL::IExpression_parameter const> param(
                    expr->get_interface<MI::MDL::IExpression_parameter>());
                size_t index = param->get_index();

                mi::base::Handle<MI::MDL::IExpression const> expr(
                    get_parameter_argument(index));

                if (expr.is_valid_interface())
                    return evaluate(expr.get());
                set_error(EC_PARAMETER);
                return m_value_fact.create_bad();
            }
            break;
        case MI::MDL::IExpression::EK_DIRECT_CALL:
            {
                mi::base::Handle<MI::MDL::IExpression_direct_call const> dcall(
                    expr->get_interface<MI::MDL::IExpression_direct_call>());
                DB::Tag tag = dcall->get_definition(m_trans);
                if (!tag.is_valid()) {
                    set_error(EC_INVALID_CALL);
                    return m_value_fact.create_bad();
                }
                SERIAL::Class_id class_id = m_trans->get_class_id(tag);

                if (class_id != MI::MDL::Mdl_function_definition::id) {
                    set_error(EC_NON_FUNCTION_CALL);
                    return m_value_fact.create_bad();
                }
                DB::Access<MI::MDL::Mdl_function_definition> def(tag, m_trans);
                mi::base::Handle<MI::MDL::IExpression_list const> args(
                    dcall->get_arguments());

                return evaluate_call(def.get_ptr(), args.get());
            }
        case MI::MDL::IExpression::EK_TEMPORARY:
            // should not happen, but if does cannot be evaluated
            set_error(EC_TEMPORARY);
            return m_value_fact.create_bad();
        case MI::MDL::IExpression::EK_FORCE_32_BIT:
            break;
        }
        ASSERT(M_MDLC, !"unsupported expression kind");
        set_error(EC_UNSUPPORTED);
        return m_value_fact.create_bad();
    }

    /// Evaluate a neuray value.
    mi::mdl::IValue const *evaluate(MI::MDL::IValue const *value)
    {
        switch (value->get_kind()) {
        case MI::MDL::IValue::VK_BOOL:
            {
                mi::base::Handle<MI::MDL::IValue_bool const> b(
                    value->get_interface<MI::MDL::IValue_bool>());

                return m_value_fact.create_bool(b->get_value());
            }
        case MI::MDL::IValue::VK_INT:
            {
                mi::base::Handle<MI::MDL::IValue_int const> i(
                    value->get_interface<MI::MDL::IValue_int>());

                return m_value_fact.create_int(i->get_value());
            }
        case MI::MDL::IValue::VK_ENUM:
            {
                mi::base::Handle<MI::MDL::IValue_enum const> e(
                    value->get_interface<MI::MDL::IValue_enum>());
                mi::base::Handle<MI::MDL::IType const> tp(e->get_type());

                mi::mdl::IType_enum const *e_tp = mi::mdl::as<mi::mdl::IType_enum>(
                    translate(tp.get()));

                if (e_tp != nullptr) {
                    return m_value_fact.create_enum(e_tp, e->get_index());
                }
                set_error(EC_UNSUPPORTED);
                return m_value_fact.create_bad();
            }
        case MI::MDL::IValue::VK_FLOAT:
            {
                mi::base::Handle<MI::MDL::IValue_float const> f(
                    value->get_interface<MI::MDL::IValue_float>());

                return m_value_fact.create_float(f->get_value());
            }
            break;
        case MI::MDL::IValue::VK_DOUBLE:
            {
                mi::base::Handle<MI::MDL::IValue_double const> d(
                    value->get_interface<MI::MDL::IValue_double>());

                return m_value_fact.create_double(d->get_value());
            }
            break;
        case MI::MDL::IValue::VK_STRING:
            {
                {
                    mi::base::Handle<MI::MDL::IValue_string_localized const> s(
                        value->get_interface<MI::MDL::IValue_string_localized >());
                    if (s) {
                        return m_value_fact.create_string(s->get_original_value());
                    }
                }
                mi::base::Handle<MI::MDL::IValue_string const> s(
                    value->get_interface<MI::MDL::IValue_string>());

                return m_value_fact.create_string(s->get_value());
            }
            break;
        case MI::MDL::IValue::VK_VECTOR:
            {
                mi::base::Handle<MI::MDL::IValue_vector const> v(
                    value->get_interface<MI::MDL::IValue_vector>());

                mi::base::Handle<MI::MDL::IType const> t(v->get_type());
                mi::mdl::IType_vector const *v_tp =
                    mi::mdl::as<mi::mdl::IType_vector>(translate(t.get()));

                if (v_tp != nullptr) {
                    size_t n = v->get_size();
                    MI::MDL::Small_VLA<mi::mdl::IValue const *, 4> values(n);
                    for (size_t i = 0; i < n; ++i) {
                        mi::base::Handle<MI::MDL::IValue const> e(v->get_value(i));
                        mi::mdl::IValue const *e_v = evaluate(e.get());

                        if (mi::mdl::is<mi::mdl::IValue_bad>(e_v)) {
                            // already error
                            return e_v;
                        }
                        values[i] = e_v;
                    }
                    return m_value_fact.create_vector(v_tp, values.data(), values.size());
                }
                set_error(EC_UNSUPPORTED);
                return m_value_fact.create_bad();
            }
        case MI::MDL::IValue::VK_MATRIX:
            {
                mi::base::Handle<MI::MDL::IValue_matrix const> m(
                    value->get_interface<MI::MDL::IValue_matrix>());

                mi::base::Handle<MI::MDL::IType const> t(m->get_type());
                mi::mdl::IType_matrix const *m_tp =
                    mi::mdl::as<mi::mdl::IType_matrix>(translate(t.get()));

                if (m_tp != nullptr) {
                    size_t n = m->get_size();
                    MI::MDL::Small_VLA<mi::mdl::IValue const *, 4> values(n);
                    for (size_t i = 0; i < n; ++i) {
                        mi::base::Handle<MI::MDL::IValue const> e(m->get_value(i));
                        mi::mdl::IValue const *e_v = evaluate(e.get());

                        if (mi::mdl::is<mi::mdl::IValue_bad>(e_v)) {
                            // already error
                            return e_v;
                        }
                        values[i] = e_v;
                    }
                    return m_value_fact.create_matrix(m_tp, values.data(), values.size());
                }
                set_error(EC_UNSUPPORTED);
                return m_value_fact.create_bad();
            }
        case MI::MDL::IValue::VK_COLOR:
            {
                mi::base::Handle<MI::MDL::IValue_color const> m(
                    value->get_interface<MI::MDL::IValue_color>());

                mi::base::Handle<MI::MDL::IType const> t(m->get_type());
                mi::mdl::IType_color const *m_tp =
                    mi::mdl::as<mi::mdl::IType_color>(translate(t.get()));

                if (m_tp != nullptr) {
                    mi::mdl::IValue_float const *values[3];
                    for (size_t i = 0; i < 3; ++i) {
                        mi::base::Handle<MI::MDL::IValue const> e(m->get_value(i));
                        mi::mdl::IValue const *e_v = evaluate(e.get());

                        if (!mi::mdl::is<mi::mdl::IValue_float>(e_v)) {
                            set_error(EC_UNSUPPORTED);
                            return m_value_fact.create_bad();
                        }
                        values[i] = mi::mdl::as<mi::mdl::IValue_float>(e_v);
                    }
                    return m_value_fact.create_rgb_color(values[0], values[1], values[2]);
                }
                set_error(EC_UNSUPPORTED);
                return m_value_fact.create_bad();
            }
        case MI::MDL::IValue::VK_ARRAY:
            {
                mi::base::Handle<MI::MDL::IValue_array const> a(
                    value->get_interface<MI::MDL::IValue_array>());

                mi::base::Handle<MI::MDL::IType const> t(a->get_type());
                mi::mdl::IType_array const *a_tp =
                    mi::mdl::as<mi::mdl::IType_array>(translate(t.get()));

                if (a_tp != nullptr && a_tp->is_immediate_sized()) {
                    mi::mdl::IType const *e_tp = a_tp->get_element_type();
                    size_t n = a->get_size();
                    MI::MDL::Small_VLA<mi::mdl::IValue const *, 4> values(n);
                    for (size_t i = 0; i < n; ++i) {
                        mi::base::Handle<MI::MDL::IValue const> e(a->get_value(i));
                        mi::mdl::IValue const *e_v = evaluate(e.get());

                        if (e_v->get_type() != e_tp) {
                            set_error(EC_UNSUPPORTED);
                            return m_value_fact.create_bad();
                        }
                        values[i] = e_v;
                    }
                    return m_value_fact.create_array(a_tp, values.data(), values.size());
                }
                set_error(EC_UNSUPPORTED);
                return m_value_fact.create_bad();
            }
        case MI::MDL::IValue::VK_STRUCT:
            {
                mi::base::Handle<MI::MDL::IValue_struct const> s(
                    value->get_interface<MI::MDL::IValue_struct>());
                mi::base::Handle<MI::MDL::IType const> tp(s->get_type());
                mi::mdl::IType_struct const *s_tp =
                    mi::mdl::as<mi::mdl::IType_struct>(translate(tp.get()));

                if (s_tp != nullptr) {
                    size_t n = s->get_size();
                    MI::MDL::Small_VLA<mi::mdl::IValue const *, 8> values(n);
                    for (size_t i = 0; i < n; ++i) {
                        mi::base::Handle<MI::MDL::IValue const> e(s->get_value(i));
                        mi::mdl::IValue const *e_v = evaluate(e.get());

                        if (mi::mdl::is<mi::mdl::IValue_bad>(e_v)) {
                            // already error
                            return e_v;
                        }
                        values[i] = e_v;
                    }
                    return m_value_fact.create_struct(s_tp, values.data(), values.size());
                }
                set_error(EC_UNSUPPORTED);
                return m_value_fact.create_bad();
            }
        case MI::MDL::IValue::VK_INVALID_DF:
            {
                mi::base::Handle<MI::MDL::IType const> type(value->get_type());
                mi::mdl::IType_reference const *ref_type =
                    mi::mdl::as<mi::mdl::IType_reference>(translate(type.get()));
                if (ref_type != nullptr)
                    return m_value_fact.create_invalid_ref(ref_type);
                set_error(EC_UNSUPPORTED);
                return m_value_fact.create_bad();
            }
        case MI::MDL::IValue::VK_TEXTURE:
            {
                mi::base::Handle<MI::MDL::IValue_texture const> tex(
                    value->get_interface<MI::MDL::IValue_texture>());
                mi::base::Handle<MI::MDL::IType const> tp(tex->get_type());

                mi::mdl::IType_texture const *tex_type =
                    mi::mdl::as<mi::mdl::IType_texture>(translate(tp.get()));

                if (tex_type != nullptr) {
                    DB::Tag resource_tag = tex->get_value();
                    if (resource_tag.is_valid()) {
                        // treat this as a valid resource
                        return m_value_fact.create_texture(
                            tex_type,
                            "",
                            mi::mdl::IValue_texture::gamma_default,
                            resource_tag.get_uint(),
                            0);
                    }
                    // invalid texture
                    return m_value_fact.create_invalid_ref(tex_type);
                }
                set_error(EC_UNSUPPORTED);
                return m_value_fact.create_bad();
            }
        case MI::MDL::IValue::VK_LIGHT_PROFILE:
            {
                mi::base::Handle<MI::MDL::IValue_light_profile const> lp(
                    value->get_interface<MI::MDL::IValue_light_profile>());
                mi::base::Handle<MI::MDL::IType const> tp(lp->get_type());

                mi::mdl::IType_light_profile const *lp_type =
                    mi::mdl::as<mi::mdl::IType_light_profile>(translate(tp.get()));

                if (lp_type != nullptr) {
                    DB::Tag resource_tag = lp->get_value();
                    if (resource_tag.is_valid()) {
                        // treat this as a valid resource
                        return m_value_fact.create_light_profile(
                            lp_type, "", resource_tag.get_uint(), 0);
                    }
                    // invalid texture
                    return m_value_fact.create_invalid_ref(lp_type);
                }
                set_error(EC_UNSUPPORTED);
                return m_value_fact.create_bad();
            }
        case MI::MDL::IValue::VK_BSDF_MEASUREMENT:
            {
                mi::base::Handle<MI::MDL::IValue_bsdf_measurement const> bm(
                    value->get_interface<MI::MDL::IValue_bsdf_measurement>());
                mi::base::Handle<MI::MDL::IType const> tp(bm->get_type());

                mi::mdl::IType_bsdf_measurement const *bm_type =
                    mi::mdl::as<mi::mdl::IType_bsdf_measurement>(translate(tp.get()));

                if (bm_type != nullptr) {
                    DB::Tag resource_tag = bm->get_value();
                    if (resource_tag.is_valid()) {
                        // treat this as a valid resource
                        return m_value_fact.create_bsdf_measurement(
                            bm_type, "", resource_tag.get_uint(), 0);
                    }
                    // invalid texture
                    return m_value_fact.create_invalid_ref(bm_type);
                }
                set_error(EC_UNSUPPORTED);
                return m_value_fact.create_bad();
            }
        case MI::MDL::IValue::VK_FORCE_32_BIT:
            break;
        }
        ASSERT(M_MDLC, !"unsupported value kind");
        set_error(EC_UNSUPPORTED);
        return m_value_fact.create_bad();
    }

    /// Creates a new user enum type.
    mi::mdl::IType const *create_enum(MI::MDL::IType_enum const *e_tp)
    {
        char const *name = e_tp->get_symbol();

        mi::mdl::ISymbol const *sym = m_sym_tab.create_user_type_symbol(name);
        mi::mdl::IType_enum    *res = m_type_fact.create_enum(sym);

        for (size_t i = 0, n = e_tp->get_size(); i < n; ++i) {
            char const *v_name = e_tp->get_value_name(i);
            mi::Sint32 v_code  = e_tp->get_value_code(i);

            res->add_value(m_sym_tab.get_symbol(v_name), v_code);
        }

        m_user_types.insert(User_type_map::value_type(name, res));
        return res;
    }

    /// Creates a new user struct type.
    mi::mdl::IType const *create_struct(MI::MDL::IType_struct const *s_tp)
    {
        size_t n = s_tp->get_size();
        MI::MDL::Small_VLA <mi::mdl::IType const *, 8> elem(n);

        for (size_t i = 0; i < n; ++i) {
            mi::base::Handle<MI::MDL::IType const> e(s_tp->get_field_type(i));

            mi::mdl::IType const *e_tp = translate(e.get());

            if (mi::mdl::is<mi::mdl::IType_error>(e_tp)) {
                // already error
                return e_tp;
            }
            elem[i] = e_tp;
        }

        char const *name = s_tp->get_symbol();

        mi::mdl::ISymbol const *sym = m_sym_tab.create_user_type_symbol(name);
        mi::mdl::IType_struct  *res = m_type_fact.create_struct(sym);

        for (size_t i = 0; i < n; ++i) {
            char const *fname = s_tp->get_field_name(i);

            res->add_field(elem[i], m_sym_tab.get_symbol(fname));
        }
        m_user_types.insert(User_type_map::value_type(name, res));
        return res;
    }

    /// Translate a type-
    mi::mdl::IType const *translate(MI::MDL::IType const *type)
    {
        switch (type->get_kind()) {
        case MI::MDL::IType::TK_ALIAS:
            {
                // here we ignore alias types
                mi::base::Handle<MI::MDL::IType_alias const> a(
                    type->get_interface<MI::MDL::IType_alias>());
                mi::base::Handle<MI::MDL::IType const> t(a->get_aliased_type());
                return translate(t.get());
            }
        case MI::MDL::IType::TK_BOOL:
            return m_type_fact.create_bool();
        case MI::MDL::IType::TK_INT:
            return m_type_fact.create_int();
        case MI::MDL::IType::TK_ENUM:
            {
                mi::base::Handle<MI::MDL::IType_enum const> e(
                    type->get_interface<MI::MDL::IType_enum>());

                switch (e->get_predefined_id()) {
                case MI::MDL::IType_enum::EID_USER:
                    {
                        char const *sym = e->get_symbol();
                        User_type_map::const_iterator it(m_user_types.find(sym));

                        if (it != m_user_types.end()) {
                            return it->second;
                        }
                        return create_enum(e.get());
                    }
                case MI::MDL::IType_enum::EID_TEX_GAMMA_MODE:
                    return m_type_fact.get_predefined_enum(mi::mdl::IType_enum::EID_TEX_GAMMA_MODE);
                case MI::MDL::IType_enum::EID_INTENSITY_MODE:
                    return m_type_fact.get_predefined_enum(mi::mdl::IType_enum::EID_INTENSITY_MODE);
                case MI::MDL::IType_enum::EID_FORCE_32_BIT:
                    break;
                }
                ASSERT(M_MDLC, !"Unsupported predefined enum");
                set_error(EC_UNSUPPORTED);
                return m_type_fact.create_error();
            }
            break;
        case MI::MDL::IType::TK_FLOAT:
            return m_type_fact.create_float();
        case MI::MDL::IType::TK_DOUBLE:
            return m_type_fact.create_double();
        case MI::MDL::IType::TK_STRING:
            return m_type_fact.create_string();
        case MI::MDL::IType::TK_VECTOR:
            {
                mi::base::Handle<MI::MDL::IType_vector const> v(
                    type->get_interface<MI::MDL::IType_vector>());

                mi::base::Handle<MI::MDL::IType_atomic const> e(v->get_element_type());
                mi::mdl::IType_atomic const *e_tp =
                    mi::mdl::as<mi::mdl::IType_atomic>(translate(e.get()));

                if (e_tp != nullptr) {
                    return m_type_fact.create_vector(e_tp, int(v->get_size()));
                }
                set_error(EC_UNSUPPORTED);
                return m_type_fact.create_error();
            }
        case MI::MDL::IType::TK_MATRIX:
            {
                mi::base::Handle<MI::MDL::IType_matrix const> m(
                    type->get_interface<MI::MDL::IType_matrix>());

                mi::base::Handle<MI::MDL::IType_vector const> e(m->get_element_type());
                mi::mdl::IType_vector const *e_tp =
                    mi::mdl::as<mi::mdl::IType_vector>(translate(e.get()));

                if (e_tp != nullptr) {
                    return m_type_fact.create_matrix(e_tp, int(m->get_size()));
                }
                set_error(EC_UNSUPPORTED);
                return m_type_fact.create_error();
            }
        case MI::MDL::IType::TK_COLOR:
            return m_type_fact.create_color();
        case MI::MDL::IType::TK_ARRAY:
            {
                mi::base::Handle<MI::MDL::IType_array const> a(
                    type->get_interface<MI::MDL::IType_array>());

                if (a->is_immediate_sized()) {
                    mi::base::Handle<MI::MDL::IType const> e(a->get_element_type());
                    mi::mdl::IType const *e_tp = translate(e.get());

                    if (!mi::mdl::is<mi::mdl::IType_error>(e_tp)) {
                        return m_type_fact.create_array(e_tp, a->get_size());
                    }
                    // already error
                    return e_tp;
                }
                set_error(EC_UNSUPPORTED);
                return m_type_fact.create_error();
            }
        case MI::MDL::IType::TK_STRUCT:
            {
                mi::base::Handle<MI::MDL::IType_struct const> s(
                    type->get_interface<MI::MDL::IType_struct>());

                switch (s->get_predefined_id()) {
                case MI::MDL::IType_struct::SID_USER:
                    {
                        char const *sym = s->get_symbol();
                        User_type_map::const_iterator it(m_user_types.find(sym));

                        if (it != m_user_types.end()) {
                            return it->second;
                        }
                        return create_struct(s.get());
                    }
                case MI::MDL::IType_struct::SID_MATERIAL_EMISSION:
                    return m_type_fact.get_predefined_struct(
                        mi::mdl::IType_struct::SID_MATERIAL_EMISSION);
                case MI::MDL::IType_struct::SID_MATERIAL_SURFACE:
                    return m_type_fact.get_predefined_struct(
                        mi::mdl::IType_struct::SID_MATERIAL_SURFACE);
                case MI::MDL::IType_struct::SID_MATERIAL_VOLUME:
                    return m_type_fact.get_predefined_struct(
                        mi::mdl::IType_struct::SID_MATERIAL_VOLUME);
                case MI::MDL::IType_struct::SID_MATERIAL_GEOMETRY:
                    return m_type_fact.get_predefined_struct(
                        mi::mdl::IType_struct::SID_MATERIAL_GEOMETRY);
                case MI::MDL::IType_struct::SID_MATERIAL:
                    return m_type_fact.get_predefined_struct(
                        mi::mdl::IType_struct::SID_MATERIAL);
                case MI::MDL::IType_struct::SID_FORCE_32_BIT:
                    break;
                }
                ASSERT(M_MDLC, !"Unsupported predefined struct");
                set_error(EC_UNSUPPORTED);
                return m_type_fact.create_error();
            }
        case MI::MDL::IType::TK_TEXTURE:
            {
                mi::base::Handle<MI::MDL::IType_texture const> t(
                    type->get_interface<MI::MDL::IType_texture>());

                switch (t->get_shape()) {
                case MI::MDL::IType_texture::TS_2D:
                    return m_type_fact.create_texture(mi::mdl::IType_texture::TS_2D);
                case MI::MDL::IType_texture::TS_3D:
                    return m_type_fact.create_texture(mi::mdl::IType_texture::TS_3D);
                case MI::MDL::IType_texture::TS_CUBE:
                    return m_type_fact.create_texture(mi::mdl::IType_texture::TS_CUBE);
                case MI::MDL::IType_texture::TS_PTEX:
                    return m_type_fact.create_texture(mi::mdl::IType_texture::TS_PTEX);
                case MI::MDL::IType_texture::TS_BSDF_DATA:
                    return m_type_fact.create_texture(mi::mdl::IType_texture::TS_BSDF_DATA);
                case MI::MDL::IType_texture::TS_FORCE_32_BIT:
                    break;
                }
                ASSERT(M_MDLC, !"Unsupported texture shape");
            }
            break;
        case MI::MDL::IType::TK_LIGHT_PROFILE:
            return m_type_fact.create_light_profile();
        case MI::MDL::IType::TK_BSDF_MEASUREMENT:
            return m_type_fact.create_bsdf_measurement();
        case MI::MDL::IType::TK_BSDF:
            return m_type_fact.create_bsdf();
        case MI::MDL::IType::TK_HAIR_BSDF:
            return m_type_fact.create_hair_bsdf();
        case MI::MDL::IType::TK_EDF:
            return m_type_fact.create_edf();
        case MI::MDL::IType::TK_VDF:
            return m_type_fact.create_vdf();
        case MI::MDL::IType::TK_FORCE_32_BIT:
            break;
        }
        ASSERT(M_MDLC, !"Unsupported type kind");
        set_error(EC_UNSUPPORTED);
        return m_type_fact.create_error();
    }

    /// Get the (first) occurred error.
    Err_codes get_error() const { return m_error; }

private:
    /// Set the error code.
    void set_error(Err_codes code) {
        if (m_error == EC_OK)
            m_error = code;
    }

private:
    /// The compiler
    mi::mdl::MDL *m_compiler;

    /// The transaction to be used.
    DB::Transaction *m_trans;

    /// If non-NULL, a callback to retrieve parameter arguments.
    IParameter_helper *m_param_cp;

    /// A memory arena.
    mi::mdl::Memory_arena m_arena;

    /// A Symbol table.
    mi::mdl::Symbol_table m_sym_tab;

    /// A type factory.
    mi::mdl::Type_factory m_type_fact;

    /// A value factory.
    mi::mdl::Value_factory m_value_fact;

    typedef std::map<std::string, mi::mdl::IType const *> User_type_map;

    /// The map for user types.
    User_type_map m_user_types;

    /// Maximum memory size allowed to used.
    size_t m_max_size;

    /// Maximum evaluation cycles allowed.
    size_t m_max_cycles;

    /// The error code if any.
    Err_codes m_error;
};


}  // anonymous


Mdl_evaluator_api_impl::Mdl_evaluator_api_impl(mi::neuraylib::INeuray *neuray)
: m_neuray(neuray)
, m_mdlc_module( true)
{
}

Mdl_evaluator_api_impl::~Mdl_evaluator_api_impl()
{
    m_neuray = nullptr;
}

// Evaluate the given (constant) expression.
/// Evaluates if a material instance parameter is enabled (i.e. the enable_if condition
/// computes true).
mi::neuraylib::IValue_bool const *Mdl_evaluator_api_impl::is_material_parameter_enabled(
    mi::neuraylib::ITransaction             *trans,
    mi::neuraylib::IValue_factory           *fact,
    mi::neuraylib::IMaterial_instance const *inst,
    mi::Size                                 index,
    mi::Sint32                               *error) const
{
    mi::Sint32 dummy_error;
    if (error == nullptr)
        error = &dummy_error;

    if (trans == nullptr || fact == nullptr || inst == nullptr) {
        *error = -1;
        return nullptr;
    }

    MI::MDL::Mdl_material_instance const *db_inst(
        static_cast<Material_instance_impl const *>(inst)->get_db_element());

    char const *name = db_inst->get_parameter_name(index);
    if (name == nullptr) {
        // wrong index
        *error = -2;
        return nullptr;
    }

    mi::base::Handle<MI::MDL::IExpression_list const> conds(db_inst->get_enable_if_conditions());

    mi::base::Handle<MI::MDL::IExpression const> cond(conds->get_expression(name));

    if (!cond) {
        // the parameter has no condition, always enabled
        *error = 0;
        return fact->create_bool(true);
    }

    mi::base::Handle<mi::mdl::IMDL> compiler(m_mdlc_module->get_mdl());

    Parameter_helper helper(db_inst);
    Evaluator eval(compiler.get(), trans, &helper);

    mi::mdl::IValue const *res = eval.evaluate(cond.get());

    if (!mi::mdl::is<mi::mdl::IValue_bool>(res)) {
        // could not be evaluated
        switch (eval.get_error()) {
        case Evaluator::EC_OK:
            *error = 0;
            break;
        case Evaluator::EC_TEMPORARY:
            *error = -3;
            break;
        case Evaluator::EC_NOT_CONSTANT:
        case Evaluator::EC_UNSUPPORTED:
        case Evaluator::EC_NON_FUNCTION_CALL:
        case Evaluator::EC_PARAMETER:
        case Evaluator::EC_INVALID_CALL:
            *error = -4;
            break;
        case Evaluator::EC_MEMORY_EXHAUSTED:
        case Evaluator::EC_CYCLES_EXHAUSTED:
            *error = -5;
            break;
        }
        return nullptr;
    }

    *error = 0;
    return fact->create_bool(mi::mdl::cast<mi::mdl::IValue_bool>(res)->get_value());
}

// Evaluates if a function call parameter is enabled (i.e. the enable_if condition
// computes true).
mi::neuraylib::IValue_bool const *Mdl_evaluator_api_impl::is_function_parameter_enabled(
    mi::neuraylib::ITransaction             *trans,
    mi::neuraylib::IValue_factory           *fact,
    mi::neuraylib::IFunction_call const     *call,
    mi::Size                                index,
    mi::Sint32                              *error) const
{
    mi::Sint32 dummy_error;
    if (error == nullptr)
        error = &dummy_error;

    if (trans == nullptr || fact == nullptr || call == nullptr) {
        *error = -1;
        return nullptr;
    }

    MI::MDL::Mdl_function_call const *db_call(
        static_cast<Function_call_impl const *>(call)->get_db_element());

    char const *name = db_call->get_parameter_name(index);
    if (name == nullptr) {
        // wrong index
        *error = -2;
        return nullptr;
    }

    mi::base::Handle<MI::MDL::IExpression_list const> conds(db_call->get_enable_if_conditions());

    mi::base::Handle<MI::MDL::IExpression const> cond(conds->get_expression(name));

    if (!cond) {
        // the parameter has no condition, always enabled
        *error = 0;
        return fact->create_bool(true);
    }

    mi::base::Handle<mi::mdl::IMDL> compiler(m_mdlc_module->get_mdl());

    Parameter_helper helper(db_call);
    Evaluator eval(compiler.get(), trans, &helper);

    mi::mdl::IValue const *res = eval.evaluate(cond.get());

    if (!mi::mdl::is<mi::mdl::IValue_bool>(res)) {
        // could not be evaluated
        switch (eval.get_error()) {
        case Evaluator::EC_OK:
            *error = 0;
            break;
        case Evaluator::EC_TEMPORARY:
            *error = -3;
            break;
        case Evaluator::EC_NOT_CONSTANT:
        case Evaluator::EC_UNSUPPORTED:
        case Evaluator::EC_NON_FUNCTION_CALL:
        case Evaluator::EC_PARAMETER:
        case Evaluator::EC_INVALID_CALL:
            *error = -4;
            break;
        case Evaluator::EC_MEMORY_EXHAUSTED:
        case Evaluator::EC_CYCLES_EXHAUSTED:
            *error = -5;
            break;
        }
        return nullptr;
    }

    *error = 0;
    return fact->create_bool(mi::mdl::cast<mi::mdl::IValue_bool>(res)->get_value());
}

mi::Sint32 Mdl_evaluator_api_impl::start()
{
    m_mdlc_module.set();
    return 0;
}

mi::Sint32 Mdl_evaluator_api_impl::shutdown()
{
    m_mdlc_module.reset();
    return 0;
}

} // namespace NEURAY
} // namespace MI
