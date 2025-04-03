/******************************************************************************
 * Copyright (c) 2012-2025, NVIDIA CORPORATION. All rights reserved.
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

/** \file
 ** \brief
 **/


#include "pch.h"

#define MI_TEST_AUTO_SUITE_NAME "Regression Test Suite for prod/lib/neuray"
#define MI_TEST_IMPLEMENT_TEST_MAIN_INSTEAD_OF_MAIN

#include <base/system/test/i_test_auto_driver.h>
#include <base/system/test/i_test_auto_case.h>

#include <mi/neuraylib/argument_editor.h>
#include <mi/neuraylib/definition_wrapper.h>
#include <mi/neuraylib/ibsdf_measurement.h>
#include <mi/neuraylib/icanvas.h>
#include <mi/neuraylib/icompiled_material.h>
#include <mi/neuraylib/idebug_configuration.h>
#include <mi/neuraylib/iimage.h>
#include <mi/neuraylib/iimage_api.h>
#include <mi/neuraylib/ilightprofile.h>
#include <mi/neuraylib/imdl_configuration.h>
#include <mi/neuraylib/iplugin_configuration.h>
#include <mi/neuraylib/ireader.h>

#include <mi/neuraylib/imdl_compiler.h>

#include <cctype>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "test_shared.h"

// To avoid race conditions, each unit test uses a separate subdirectory for all the files it
// creates (possibly with further subdirectories).
#define DIR_PREFIX "output_test_imodule_advanced"

#include "test_shared_mdl.h" // depends on DIR_PREFIX



mi::Sint32 result = 0;



// === Constants ===================================================================================

const char* native_module = "mdl 1.3; export color f(color c) [[ native() ]] { return c; }";

// === Tests =======================================================================================

void check_matrix( mi::neuraylib::ITransaction* transaction)
{
    // see test_mdl.mdl for rationale
    mi::base::Handle<const mi::neuraylib::IFunction_definition> c_fd(
        transaction->access<mi::neuraylib::IFunction_definition>(
            "mdl::" TEST_MDL "::fd_matrix(float3x2)"));
    MI_CHECK( c_fd);
    mi::base::Handle<const mi::neuraylib::IType_list> types( c_fd->get_parameter_types());

    mi::base::Handle<const mi::neuraylib::IType_matrix> matrix_type(
        types->get_type<mi::neuraylib::IType_matrix>( zero_size));
    MI_CHECK_EQUAL( 3, matrix_type->get_size());
    mi::base::Handle<const mi::neuraylib::IType_vector> vector_type(
        matrix_type->get_element_type());
    MI_CHECK_EQUAL( 2, vector_type->get_size());
    mi::base::Handle<const mi::neuraylib::IType_atomic> element_type(
        vector_type->get_element_type());
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_FLOAT, element_type->get_kind());

    mi::base::Handle<const mi::neuraylib::IValue_float> v;
    mi::base::Handle<const mi::neuraylib::IExpression_list> defaults( c_fd->get_defaults());

    mi::base::Handle<const mi::neuraylib::IExpression_constant> default_expr(
        defaults->get_expression<mi::neuraylib::IExpression_constant>( zero_size));
    mi::base::Handle<const mi::neuraylib::IValue_matrix> matrix_value(
        default_expr->get_value<mi::neuraylib::IValue_matrix>());

    mi::base::Handle<const mi::neuraylib::IValue_vector> column0_value(
        matrix_value->get_value( 0));
    v = column0_value->get_value<mi::neuraylib::IValue_float>( 0);
    MI_CHECK_EQUAL( 0.0f, v->get_value());
    v = column0_value->get_value<mi::neuraylib::IValue_float>( 1);
    MI_CHECK_EQUAL( 1.0f, v->get_value());

    mi::base::Handle<const mi::neuraylib::IValue_vector> column1_value(
        matrix_value->get_value( 1));
    v = column1_value->get_value<mi::neuraylib::IValue_float>( 0);
    MI_CHECK_EQUAL( 2.0f, v->get_value());
    v = column1_value->get_value<mi::neuraylib::IValue_float>( 1);
    MI_CHECK_EQUAL( 3.0f, v->get_value());

    mi::base::Handle<const mi::neuraylib::IValue_vector> column2_value(
        matrix_value->get_value( 2));
    v = column2_value->get_value<mi::neuraylib::IValue_float>( 0);
    MI_CHECK_EQUAL( 4.0f, v->get_value());
    v = column2_value->get_value<mi::neuraylib::IValue_float>( 1);
    MI_CHECK_EQUAL( 5.0f, v->get_value());

    mi::math::Matrix<mi::Float32,3,2> m;
    result = get_value( matrix_value.get(), m);
    MI_CHECK_EQUAL( 0, result);
    MI_CHECK_EQUAL( m[0][0], 0.0f);
    MI_CHECK_EQUAL( m[0][1], 1.0f);
    MI_CHECK_EQUAL( m[1][0], 2.0f);
    MI_CHECK_EQUAL( m[1][1], 3.0f);
    MI_CHECK_EQUAL( m[2][0], 4.0f);
    MI_CHECK_EQUAL( m[2][1], 5.0f);
}

void check_immediate_arrays(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    mi::base::Handle<const mi::neuraylib::IType> type;
    mi::base::Handle<const mi::neuraylib::IType_array> type_array;
    mi::base::Handle<const mi::neuraylib::IType_list> types;

    mi::base::Handle<const mi::neuraylib::IFunction_definition> c_fd(
        transaction->access<mi::neuraylib::IFunction_definition>(
            "mdl::" TEST_MDL "::fd_immediate(int[42])"));
    MI_CHECK( c_fd);
    MI_CHECK_EQUAL( 1, c_fd->get_parameter_count());
    types = c_fd->get_parameter_types();

    MI_CHECK_EQUAL_CSTR( "param0", c_fd->get_parameter_name( zero_size));
    type = types->get_type( zero_size);
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_ARRAY, type->get_kind());
    type_array = type->get_interface<mi::neuraylib::IType_array>();
    MI_CHECK( type_array->is_immediate_sized());
    MI_CHECK_EQUAL( 42, type_array->get_size());
    type = type_array->get_element_type();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, type->get_kind());

    type = c_fd->get_return_type();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, type->get_kind());

    // create array constructor instance
    {
        // prepare argument for even array slots
        mi::base::Handle<mi::neuraylib::IValue> slot0_value( vf->create_int( 1701));
        mi::base::Handle<mi::neuraylib::IExpression> slot0_expr(
            ef->create_constant( slot0_value.get()));

        // prepare argument for odd array slots
        mi::base::Handle<mi::neuraylib::IExpression> slot1_expr(
            ef->create_call( "mdl::" TEST_MDL "::fc_0"));

        // prepare arguments for array constructor
        mi::base::Handle<mi::neuraylib::IExpression_list> array_constructor_args(
            ef->create_expression_list());
        for( mi::Size i = 0; i < 42; ++i) {
            std::string name = "value" + std::to_string( i);
            array_constructor_args->add_expression(
                name.c_str(), (i%2 == 0) ? slot0_expr.get() : slot1_expr.get());
        }

        // instantiate array constructor definition with these arguments
        mi::base::Handle<const mi::neuraylib::IFunction_definition> c_fd_array_constructor(
            transaction->access<mi::neuraylib::IFunction_definition>( "mdl::T[](...)"));
        mi::base::Handle<mi::neuraylib::IFunction_call> m_fc_array_constructor(
            c_fd_array_constructor->create_function_call( array_constructor_args.get(), &result));
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK( m_fc_array_constructor);
        MI_CHECK_EQUAL( 0, transaction->store(
            m_fc_array_constructor.get(), "mdl::" TEST_MDL "::array_constructor_int_42"));
    }

    // instantiate fd_immediate with constant expression as argument
    {
        // prepare argument
        mi::base::Handle<mi::neuraylib::IValue> param0_value(
            vf->create_array( type_array.get()));
        mi::base::Handle<mi::neuraylib::IExpression> param0_expr(
            ef->create_constant( param0_value.get()));
        mi::base::Handle<mi::neuraylib::IExpression_list> args(
            ef->create_expression_list());
        args->add_expression( "param0", param0_expr.get());

        // instantiate the definition with that argument
        mi::base::Handle<mi::neuraylib::IFunction_call> m_fc(
            c_fd->create_function_call( args.get(), &result));
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK( m_fc);

        {
            // check the argument
            mi::base::Handle<const mi::neuraylib::IExpression_list> args(
               m_fc->get_arguments());
            mi::base::Handle<const mi::neuraylib::IExpression_constant> c_param0_expr(
                args->get_expression<mi::neuraylib::IExpression_constant>( "param0"));
            mi::base::Handle<const mi::neuraylib::IValue_array> c_param0_value(
                c_param0_expr->get_value<mi::neuraylib::IValue_array>());
            MI_CHECK_EQUAL( 42, c_param0_value->get_size());

            // set new argument (clone of old argument)
            mi::base::Handle<mi::neuraylib::IExpression_constant> m_param0_expr(
                ef->clone<mi::neuraylib::IExpression_constant>( c_param0_expr.get()));
            mi::base::Handle<mi::neuraylib::IValue_array> m_param0_value(
                m_param0_expr->get_value<mi::neuraylib::IValue_array>());
            MI_CHECK_EQUAL( -1, m_param0_value->set_size( 43));
            MI_CHECK_EQUAL( 0, m_fc->set_argument( "param0", m_param0_expr.get()));
        }

        // store the instance
        MI_CHECK_EQUAL( 0, transaction->store(
            m_fc.get(), "mdl::" TEST_MDL "::fc_immediate_constant"));
    }

    // instantiate fd_immediate with call expression as argument (array constructor)
    {
        // prepare argument
        mi::base::Handle<mi::neuraylib::IExpression> param0_expr(
            ef->create_call( "mdl::" TEST_MDL "::array_constructor_int_42"));
        mi::base::Handle<mi::neuraylib::IExpression_list> args(
            ef->create_expression_list());
        args->add_expression( "param0", param0_expr.get());

        // instantiate the definition with that argument
        mi::base::Handle<mi::neuraylib::IFunction_call> m_fc(
            c_fd->create_function_call( args.get(), &result));
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK( m_fc);

        {
            // check the argument
            mi::base::Handle<const mi::neuraylib::IExpression_list> args(
               m_fc->get_arguments());
            mi::base::Handle<const mi::neuraylib::IExpression_call> c_param0_expr(
                args->get_expression<mi::neuraylib::IExpression_call>( "param0"));
            MI_CHECK_EQUAL_CSTR(
                "mdl::" TEST_MDL "::array_constructor_int_42", c_param0_expr->get_call());

            // set new argument (clone of old argument)
            mi::base::Handle<mi::neuraylib::IExpression_call> m_param0_expr(
                ef->clone<mi::neuraylib::IExpression_call>( c_param0_expr.get()));
            MI_CHECK_EQUAL( 0, m_fc->set_argument( "param0", m_param0_expr.get()));
        }

        // store the instance
        MI_CHECK_EQUAL( 0, transaction->store( m_fc.get(), "mdl::" TEST_MDL "::fc_immediate_call"));
    }
}

void check_deferred_arrays(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    mi::base::Handle<const mi::neuraylib::IType> type;
    mi::base::Handle<const mi::neuraylib::IType_array> type_array;
    mi::base::Handle<const mi::neuraylib::IType_list> types;

    // check deferred arrays in a function definition (fd_deferred)
    {
        mi::base::Handle<const mi::neuraylib::IFunction_definition> c_fd(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::" TEST_MDL "::fd_deferred(int[N])"));
        MI_CHECK( c_fd);
        MI_CHECK_EQUAL( 1, c_fd->get_parameter_count());
        types = c_fd->get_parameter_types();

        MI_CHECK_EQUAL_CSTR( "param0", c_fd->get_parameter_name( zero_size));
        type = types->get_type( zero_size);
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_ARRAY, type->get_kind());
        type_array = type->get_interface<mi::neuraylib::IType_array>();
        MI_CHECK( !type_array->is_immediate_sized());
        MI_CHECK_EQUAL( minus_one_size, type_array->get_size());
        type = type_array->get_element_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, type->get_kind());

        type = c_fd->get_return_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_ARRAY, type->get_kind());
        type_array = type->get_interface<mi::neuraylib::IType_array>();
        MI_CHECK( !type_array->is_immediate_sized());
        MI_CHECK_EQUAL( minus_one_size, type_array->get_size());
        type = type_array->get_element_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, type->get_kind());

        // instantiate fd_deferred with constant expression as argument
        {
            // prepare argument
            mi::base::Handle<mi::neuraylib::IValue_array> param0_value(
                vf->create_array( type_array.get()));
            MI_CHECK_EQUAL( 0, param0_value->set_size( 1));
            mi::base::Handle<mi::neuraylib::IExpression> param0_expr(
                ef->create_constant( param0_value.get()));
            mi::base::Handle<mi::neuraylib::IExpression_list> args(
                ef->create_expression_list());
            args->add_expression( "param0", param0_expr.get());

            // instantiate the definition with that argument
            mi::base::Handle<mi::neuraylib::IFunction_call> m_fc(
                c_fd->create_function_call( args.get(), &result));
            MI_CHECK_EQUAL( 0, result);
            MI_CHECK( m_fc);

            {
                // check the argument
                mi::base::Handle<const mi::neuraylib::IExpression_list> args(
                   m_fc->get_arguments());
                mi::base::Handle<const mi::neuraylib::IExpression_constant> c_param0_expr(
                    args->get_expression<mi::neuraylib::IExpression_constant>( "param0"));
                mi::base::Handle<const mi::neuraylib::IValue_array> c_param0_value(
                    c_param0_expr->get_value<mi::neuraylib::IValue_array>());
                MI_CHECK_EQUAL( 1, c_param0_value->get_size());

                // set new argument
                mi::base::Handle<mi::neuraylib::IExpression_constant> m_param0_expr(
                    ef->clone<mi::neuraylib::IExpression_constant>( c_param0_expr.get()));
                mi::base::Handle<mi::neuraylib::IValue_array> m_param0_value(
                    m_param0_expr->get_value<mi::neuraylib::IValue_array>());
                MI_CHECK_EQUAL( 0, m_param0_value->set_size( 2));
                MI_CHECK_EQUAL( 0, m_fc->set_argument( "param0", m_param0_expr.get()));
            }

            // store the instance
            MI_CHECK_EQUAL( 0, transaction->store(
                m_fc.get(), "mdl::" TEST_MDL "::fc_deferred_constant"));
        }

        // instantiate fd_deferred with call expression as argument (array constructor)
        {
            // prepare argument
            mi::base::Handle<mi::neuraylib::IExpression> param0_expr(
                ef->create_call( "mdl::" TEST_MDL "::array_constructor_int_42"));
            mi::base::Handle<mi::neuraylib::IExpression_list> args(
                ef->create_expression_list());
            args->add_expression( "param0", param0_expr.get());

            // instantiate the definition with that argument
            mi::base::Handle<mi::neuraylib::IFunction_call> m_fc(
                c_fd->create_function_call( args.get(), &result));
            MI_CHECK_EQUAL( 0, result);
            MI_CHECK( m_fc);

            {
                // check the argument
                mi::base::Handle<const mi::neuraylib::IExpression_list> args(
                   m_fc->get_arguments());
                mi::base::Handle<const mi::neuraylib::IExpression_call> c_param0_expr(
                    args->get_expression<mi::neuraylib::IExpression_call>( "param0"));
                MI_CHECK_EQUAL_CSTR(
                   "mdl::" TEST_MDL "::array_constructor_int_42", c_param0_expr->get_call());

                // set new argument (clone of old argument)
                mi::base::Handle<mi::neuraylib::IExpression_call> m_param0_expr(
                    ef->clone<mi::neuraylib::IExpression_call>( c_param0_expr.get()));
                MI_CHECK_EQUAL( 0, m_fc->set_argument( "param0", m_param0_expr.get()));
            }

            // store the instance
            MI_CHECK_EQUAL( 0, transaction->store(
                m_fc.get(), "mdl::" TEST_MDL "::fc_deferred_call"));
        }
    }
}

void check_set_get_value(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    {
        // bool / IValue_bool
        mi::base::Handle<mi::neuraylib::IValue_bool> data( vf->create_bool( false));
        bool value = true;
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value, false);
        result = set_value( data.get(), true);
        MI_CHECK_EQUAL( result, 0);
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value, true);
    }
    {
        // mi::Sint32 / IValue_int
        mi::base::Handle<mi::neuraylib::IValue_int> data( vf->create_int( 1));
        mi::Sint32 value = 0;
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value, 1);
        result = set_value( data.get(), 2);
        MI_CHECK_EQUAL( result, 0);
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value, 2);
    }
    {
        // mi::Sint32 / IValue_enum
        mi::base::Handle<const mi::neuraylib::IType_enum> type(
            tf->create_enum( "::" TEST_MDL "::Enum"));
        mi::base::Handle<mi::neuraylib::IValue_enum> data( vf->create_enum( type.get(), 0));
        mi::Sint32 value = 0;
        const char* name = nullptr;
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value, 1); // index 0 => value 1
        result = get_value( data.get(), name);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL_CSTR( name, "one");

        result = set_value( data.get(), 2);
        MI_CHECK_EQUAL( result, 0);
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value, 2);
        result = get_value( data.get(), name);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL_CSTR( name, "two");

        result = set_value( data.get(), "one");
        MI_CHECK_EQUAL( result, 0);
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value, 1);
        result = get_value( data.get(), name);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL_CSTR( name, "one");
    }
    {
        // mi::Float32 / IValue_float
        mi::base::Handle<mi::neuraylib::IValue_float> data( vf->create_float( 1.0f));
        mi::Float32 value = 0.0f;
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value, 1.0f);
        result = set_value( data.get(), 2.0f);
        MI_CHECK_EQUAL( result, 0);
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value, 2.0f);
    }
    {
        // mi::Float64 / IValue_float
        mi::base::Handle<mi::neuraylib::IValue_float> data( vf->create_float( 1.0f));
        mi::Float64 value = 0.0;
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value, 1.0);
        result = set_value( data.get(), 2.0);
        MI_CHECK_EQUAL( result, 0);
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value, 2.0);
    }
    {
        // mi::Float32 / IValue_double
        mi::base::Handle<mi::neuraylib::IValue_double> data( vf->create_double( 1.0));
        mi::Float32 value = 0.0f;
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value, 1.0f);
        result = set_value( data.get(), 2.0f);
        MI_CHECK_EQUAL( result, 0);
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value, 2.0f);
    }
    {
        // mi::Float64 / IValue_double
        mi::base::Handle<mi::neuraylib::IValue_double> data( vf->create_double( 1.0));
        mi::Float64 value = 0.0;
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value, 1.0);
        result = set_value( data.get(), 2.0);
        MI_CHECK_EQUAL( result, 0);
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value, 2.0);
    }
    {
        // const char* / IValue_string
        mi::base::Handle<mi::neuraylib::IValue_string> data( vf->create_string( "foo"));
        const char* value = nullptr;
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL_CSTR( value, "foo");
        result = set_value( data.get(), "bar");
        MI_CHECK_EQUAL( result, 0);
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL_CSTR( value, "bar");
    }
    {
        // mi::math::Vector<mi::Sint32,3> / IValue_vector
        mi::base::Handle<const mi::neuraylib::IType_atomic> type_int( tf->create_int());
        mi::base::Handle<const mi::neuraylib::IType_vector> type_vector(
            tf->create_vector( type_int.get(), 3));
        mi::base::Handle<mi::neuraylib::IValue_vector> data( vf->create_vector( type_vector.get()));
        mi::math::Vector<mi::Sint32,3> value;
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK( value == (mi::math::Vector<mi::Sint32,3>( 0)));
        result = set_value( data.get(), mi::math::Vector<mi::Sint32,3>( 4, 5, 6));
        MI_CHECK_EQUAL( result, 0);
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK( value == (mi::math::Vector<mi::Sint32,3>( 4, 5, 6)));
    }
    {
        // mi::math::Matrix<mi::Float32,3,2> / IValue_matrix
        mi::base::Handle<const mi::neuraylib::IType_atomic> type_float( tf->create_float());
        mi::base::Handle<const mi::neuraylib::IType_vector> type_vector(
            tf->create_vector( type_float.get(), 2));
        mi::base::Handle<const mi::neuraylib::IType_matrix> type_matrix(
            tf->create_matrix( type_vector.get(), 3));
        mi::base::Handle<mi::neuraylib::IValue_matrix> data( vf->create_matrix( type_matrix.get()));
        mi::math::Matrix<mi::Float32,3,2> value;
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK( value == (mi::math::Matrix<mi::Float32,3,2>( 0)));
        mi::math::Matrix<mi::Float32,3,2> m( 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f);
        result = set_value( data.get(), m);
        MI_CHECK_EQUAL( result, 0);
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK( value == m);
    }
    {
        // mi::math::Color / IValue_color
        mi::base::Handle<mi::neuraylib::IValue_color> data( vf->create_color( 1.0f, 2.0f, 3.0f));
        mi::math::Color value;
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK( value == mi::math::Color( 1.0f, 2.0f, 3.0f));
        result = set_value( data.get(), mi::math::Color( 4.0f, 5.0f, 6.0f));
        MI_CHECK_EQUAL( result, 0);
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK( value == mi::math::Color( 4.0f, 5.0f, 6.0f));
    }
    {
        // mi::math::Spectrum / IValue_color
        mi::base::Handle<mi::neuraylib::IValue_color> data( vf->create_color( 1.0f, 2.0f, 3.0f));
        mi::math::Spectrum value;
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK( value == mi::math::Spectrum( 1.0f, 2.0f, 3.0f));
        result = set_value( data.get(), mi::math::Spectrum( 4.0f, 5.0f, 6.0f));
        MI_CHECK_EQUAL( result, 0);
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK( value == mi::math::Spectrum( 4.0f, 5.0f, 6.0f));
    }
    {
        // mi::Sint32[3] / IValue_array
        mi::base::Handle<const mi::neuraylib::IType> type_int( tf->create_int());
        mi::base::Handle<const mi::neuraylib::IType_array> type_array(
            tf->create_immediate_sized_array( type_int.get(), 3));
        mi::base::Handle<mi::neuraylib::IValue_array> data( vf->create_array( type_array.get()));
        mi::Sint32 value = -1;
        result = get_value( data.get(), zero_size, value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value, 0);
        result = set_value( data.get(), zero_size, 42);
        MI_CHECK_EQUAL( result, 0);
        result = get_value( data.get(), zero_size, value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value, 42);
        result = get_value( data.get(), 3, value);
        MI_CHECK_EQUAL( result, -3);
        result = set_value( data.get(), 3, 42);
        MI_CHECK_EQUAL( result, -3);

        int value2[3] = { 43, 44, 45 };
        result = set_value( data.get(), value2, 3);
        MI_CHECK_EQUAL( result, 0);
        int value3[3];
        result = get_value( data.get(), value3, 3);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value2[0], value3[0]);
        MI_CHECK_EQUAL( value2[1], value3[1]);
        MI_CHECK_EQUAL( value2[2], value3[2]);

    }
    {
        // (mi::Sint32,mi::Float32) / IValue_struct
        mi::base::Handle<const mi::neuraylib::IType_struct> type_struct(
            tf->create_struct( "::" TEST_MDL "::foo_struct"));
        mi::base::Handle<mi::neuraylib::IValue_struct> data( vf->create_struct( type_struct.get()));
        mi::Sint32 value_int = -1;
        mi::Float32 value_float = -1.0f;
        result = get_value( data.get(), zero_size, value_int);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value_int, 0);
        result = get_value( data.get(), 1, value_float);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value_float, 0.0f);
        result = set_value( data.get(), "param_int", 42);
        MI_CHECK_EQUAL( result, 0);
        result = set_value( data.get(), "param_float", -42.0f);
        MI_CHECK_EQUAL( result, 0);
        result = get_value( data.get(), "param_int", value_int);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value_int, 42);
        result = get_value( data.get(), "param_float", value_float);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value_float, -42.0f);
        result = set_value( data.get(), zero_size, 43);
        MI_CHECK_EQUAL( result, 0);
        result = set_value( data.get(), 1, -43.0f);
        MI_CHECK_EQUAL( result, 0);
        result = get_value( data.get(), zero_size, value_int);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value_int, 43);
        result = get_value( data.get(), 1, value_float);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value_float, -43.0f);
        result = get_value( data.get(), 2, value_int);
        MI_CHECK_EQUAL( result, -3);
        result = get_value( data.get(), zero_string, value_int);
        MI_CHECK_EQUAL( result, -3);
        result = get_value( data.get(), "invalid", value_int);
        MI_CHECK_EQUAL( result, -3);
        result = set_value( data.get(), 2, 42);
        MI_CHECK_EQUAL( result, -3);
        result = set_value( data.get(), zero_string, 42);
        MI_CHECK_EQUAL( result, -3);
        result = set_value( data.get(), "invalid", 42);
        MI_CHECK_EQUAL( result, -3);
    }
}

void check_wrappers(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    // check a function definition (array parameter/argument)
    {
        mi::neuraylib::Definition_wrapper dw_fail( transaction, "invalid", nullptr);
        MI_CHECK( !dw_fail.is_valid());

        mi::neuraylib::Definition_wrapper dw(
            transaction, "mdl::" TEST_MDL "::fd_deferred(int[N])", mdl_factory);
        MI_CHECK( dw.is_valid());

        MI_CHECK_EQUAL( dw.get_type(), mi::neuraylib::ELEMENT_TYPE_FUNCTION_DEFINITION);
        MI_CHECK( !dw.is_material());
        MI_CHECK_EQUAL_CSTR( dw.get_mdl_definition(), "::" TEST_MDL "::fd_deferred(int[N])");
        MI_CHECK_EQUAL_CSTR( dw.get_module(), "mdl::" TEST_MDL);

        MI_CHECK_EQUAL( 1, dw.get_parameter_count());
        MI_CHECK_EQUAL_CSTR( dw.get_parameter_name( 0), "param0");
        MI_CHECK( !dw.get_parameter_name( 1));
        mi::base::Handle<const mi::neuraylib::IType_list> parameter_types(
            dw.get_parameter_types());
        mi::base::Handle<const mi::neuraylib::IType> type( parameter_types->get_type( "param0"));
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_ARRAY, type->get_kind());
        MI_CHECK( !parameter_types->get_type( "invalid"));
        MI_CHECK( !parameter_types->get_type( zero_string));
        type = dw.get_return_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_ARRAY, type->get_kind());

        mi::base::Handle<const mi::neuraylib::IExpression_list> defaults( dw.get_defaults());
        mi::base::Handle<const mi::neuraylib::IExpression> default_(
            defaults->get_expression( "param0"));
        MI_CHECK( default_);

        mi::base::Handle<const mi::neuraylib::IAnnotation_block> av( dw.get_annotations());
        MI_CHECK( !av);
        mi::base::Handle<const mi::neuraylib::IAnnotation_list> al( dw.get_parameter_annotations());
        MI_CHECK_EQUAL( 0, al->get_size());
        MI_CHECK( !al->get_annotation_block( zero_size));
        MI_CHECK( !al->get_annotation_block( "param0"));
        MI_CHECK( !al->get_annotation_block( "invalid"));
        MI_CHECK( !al->get_annotation_block( zero_string));
        av = dw.get_return_annotations();
        MI_CHECK( !av);

        mi::base::Handle<mi::neuraylib::IFunction_call> fc(
            dw.create_instance<mi::neuraylib::IFunction_call>());
        MI_CHECK_EQUAL( 0, transaction->store(
            fc.get(), "mdl::" TEST_MDL "::fd_deferred(int[N])_wrapper"));
    }

    // check a function call (array parameter/argument)
    {
        mi::neuraylib::Argument_editor ae_fail( transaction, "invalid", nullptr);
        MI_CHECK( !ae_fail.is_valid());

        mi::neuraylib::Argument_editor ae(
            transaction, "mdl::" TEST_MDL "::fd_deferred(int[N])_wrapper", mdl_factory);
        MI_CHECK( ae.is_valid());

        MI_CHECK_EQUAL( ae.get_type(), mi::neuraylib::ELEMENT_TYPE_FUNCTION_CALL);
        MI_CHECK( !ae.is_material());
        MI_CHECK_EQUAL_CSTR( ae.get_definition(), "mdl::" TEST_MDL "::fd_deferred(int[N])");
        MI_CHECK_EQUAL_CSTR( ae.get_mdl_definition(), "::" TEST_MDL "::fd_deferred(int[N])");

        MI_CHECK_EQUAL( 1, ae.get_parameter_count());
        MI_CHECK_EQUAL_CSTR( ae.get_parameter_name( 0), "param0");
        MI_CHECK( !ae.get_parameter_name( 1));
        mi::base::Handle<const mi::neuraylib::IType> type( ae.get_return_type());
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_ARRAY, type->get_kind());

        MI_CHECK_EQUAL( mi::neuraylib::IExpression::EK_CONSTANT, ae.get_argument_kind( "param0"));
        MI_CHECK_EQUAL( mi::neuraylib::IExpression::EK_CONSTANT, ae.get_argument_kind( zero_size));

        mi::Size length = 0;
        MI_CHECK_EQUAL( 0, ae.get_array_length( "param0", length));
        MI_CHECK_EQUAL( 2, length);
        MI_CHECK_EQUAL( 0, ae.set_array_size( "param0", 3));
        MI_CHECK_EQUAL( 0, ae.get_array_length( "param0", length));
        MI_CHECK_EQUAL( 3, length);

        result = ae.reset_argument( "param0");
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK_EQUAL( 0, ae.get_array_length( "param0", length));
        MI_CHECK_EQUAL( 2, length);

        // set/get individual array elements
        mi::Sint32 s = 1;
        result = ae.get_value( "param0", zero_size, s);
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK_EQUAL( 0, s);
        result = ae.set_value( "param0", zero_size, 42);
        MI_CHECK_EQUAL( 0, result);
        result = ae.get_value( "param0", zero_size, s);
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK_EQUAL( 42, s);
        s = 1;
        result = ae.get_value( zero_size, zero_size, s);
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK_EQUAL( 42, s);
        result = ae.set_value( zero_size, zero_size, 43);
        MI_CHECK_EQUAL( 0, result);
        result = ae.get_value( zero_size, zero_size, s);
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK_EQUAL( 43, s);

        result = ae.get_value( "param0", 3, s);
        MI_CHECK_EQUAL( -3, result);
        result = ae.set_value( "param0", 3, 42);
        MI_CHECK_EQUAL( -3, result);

        // set/get entire array in one call
        mi::Sint32 a[2];
        result = ae.get_value( "param0", a, 2);
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK_EQUAL( a[0], 43);
        MI_CHECK_EQUAL( a[1], 0);
        mi::Sint32 b[3] {44, 45, 46};
        result = ae.set_value( "param0", b, 3);
        MI_CHECK_EQUAL( 0, result);
        mi::Sint32 c[3];
        result = ae.get_value( "param0", c, 3);
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK_EQUAL( c[0], 44);
        MI_CHECK_EQUAL( c[1], 45);
        MI_CHECK_EQUAL( c[2], 46);
        result = ae.get_value( "param0", c, 2);
        MI_CHECK_EQUAL( -5, result);

        mi::Sint32 e[3];
        result = ae.get_value( zero_size, e, 3);
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK_EQUAL( e[0], 44);
        MI_CHECK_EQUAL( e[1], 45);
        MI_CHECK_EQUAL( e[2], 46);
        mi::Sint32 f[2] {47, 48};
        result = ae.set_value( zero_size, f, 2);
        MI_CHECK_EQUAL( 0, result);
        mi::Sint32 g[2];
        result = ae.get_value( "param0", g, 2);
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK_EQUAL( g[0], 47);
        MI_CHECK_EQUAL( g[1], 48);
        result = ae.get_value( "param0", g, 3);
        MI_CHECK_EQUAL( -5, result);

        MI_CHECK_EQUAL( 0, ae.set_call( "param0", "mdl::" TEST_MDL "::fc_deferred_constant"));
        MI_CHECK_EQUAL( mi::neuraylib::IExpression::EK_CALL, ae.get_argument_kind( "param0"));
        MI_CHECK_EQUAL( mi::neuraylib::IExpression::EK_CALL, ae.get_argument_kind( zero_size));
        MI_CHECK_EQUAL_CSTR( ae.get_call( "param0"), "mdl::" TEST_MDL "::fc_deferred_constant");

        result = ae.set_value( "param0", zero_size, 44);
        MI_CHECK_EQUAL( -3, result);
        result = ae.set_array_size( "param0", 1);
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK_EQUAL( mi::neuraylib::IExpression::EK_CONSTANT, ae.get_argument_kind( "param0"));
        MI_CHECK_EQUAL( mi::neuraylib::IExpression::EK_CONSTANT, ae.get_argument_kind( zero_size));
        result = ae.set_value( "param0", zero_size, 45);
        MI_CHECK_EQUAL( 0, result);
        result = ae.get_value( "param0", zero_size, s);
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK_EQUAL( 45, s);
    }

    // check template functions (not supported without explicit arguments)
    {
        const char* names[] = {
            "mdl::T[](...)",
            "mdl::operator_len(%3C0%3E[])",
            "mdl::operator[](%3C0%3E[],int)",
            "mdl::operator%3F(bool,%3C0%3E,%3C0%3E)",
            "mdl::operator_cast(%3C0%3E)",
            "mdl::operator_decl_cast(%3C0%3E)",
        };

        for( const auto& name: names) {
            mi::neuraylib::Definition_wrapper dw(
                transaction,  name, mdl_factory);
            MI_CHECK( dw.is_valid());
            mi::base::Handle<mi::neuraylib::IFunction_call> fc(
                dw.create_instance<mi::neuraylib::IFunction_call>());
            MI_CHECK( !fc);
        }
    }

    // check instantiation/argument reset with argument taken from range annotation
    {
        mi::neuraylib::Definition_wrapper dw(
            transaction, "mdl::" TEST_MDL "::fd_with_annotations(color,float)", mdl_factory);
        MI_CHECK( dw.is_valid());

        mi::base::Handle<mi::neuraylib::IFunction_call> fc(
            dw.create_instance<mi::neuraylib::IFunction_call>());
        MI_CHECK( fc);
        mi::base::Handle<const mi::neuraylib::IExpression_list> args( fc->get_arguments());
        mi::base::Handle<const mi::neuraylib::IExpression_constant> param1_expr(
            args->get_expression<mi::neuraylib::IExpression_constant>( "param1"));
        mi::base::Handle<const mi::neuraylib::IValue_float> param1_value(
            param1_expr->get_value<mi::neuraylib::IValue_float>());
        MI_CHECK_EQUAL( param1_value->get_value(), 1.0f);

        // change argument
        mi::base::Handle<mi::neuraylib::IValue_float> value( vf->create_float( 2.0f));
        mi::base::Handle<mi::neuraylib::IExpression> expr( ef->create_constant( value.get()));
        MI_CHECK_EQUAL( 0, fc->set_argument( "param1", expr.get()));

        MI_CHECK_EQUAL( 0, transaction->store( fc.get(), "fc_with_annotations"));

        mi::neuraylib::Argument_editor ae(
            transaction, "fc_with_annotations", mdl_factory);
        MI_CHECK( ae.is_valid());

        mi::Float32 param1 = 0.0f;
        MI_CHECK_EQUAL( 0, ae.get_value( "param1", param1));
        MI_CHECK_EQUAL( param1, 2.0f);

        // reset argument
        MI_CHECK_EQUAL( 0, ae.reset_argument( "param1"));
        MI_CHECK_EQUAL( 0, ae.get_value( "param1", param1));
        MI_CHECK_EQUAL( param1, 1.0f);
    }
}

void check_resources(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    {
        // textures
        mi::base::Handle<mi::neuraylib::IValue_texture> texture;
        mi::neuraylib::IType_texture::Shape shape = mi::neuraylib::IType_texture::TS_2D;

        texture = mdl_factory->create_texture(
            transaction,
            "/mdl_elements/resources/test.png",
            shape,
            1.0f,
            "R",
            /*shared*/ false,
            context.get());
        MI_CHECK_CTX( context);
        MI_CHECK( texture);
        MI_CHECK_EQUAL_CSTR( texture->get_file_path(), "/mdl_elements/resources/test.png");
        MI_CHECK_EQUAL( texture->get_gamma(), 1.0f);
        MI_CHECK_EQUAL_CSTR( texture->get_selector(), "R");

        texture = mdl_factory->create_texture(
            transaction,
            "/test_archives/test_in_archive.png",
            shape,
            2.2f,
            "G",
            /*shared*/ false,
            context.get());
        MI_CHECK_CTX( context);
        MI_CHECK( texture);
        MI_CHECK_EQUAL_CSTR( texture->get_file_path(), "/test_archives/test_in_archive.png");
        MI_CHECK_EQUAL( texture->get_gamma(), 2.2f);
        MI_CHECK_EQUAL_CSTR( texture->get_selector(), "G");

        texture = mdl_factory->create_texture(
            /*transaction*/ nullptr,
            "/mdl_elements/resources/test.png",
            shape,
            0.0f,
            /*selector*/ nullptr,
            /*shared*/ false,
            context.get());
        MI_CHECK_CTX_CODE( context, -1);
        MI_CHECK( !texture);
        context->clear_messages();

        texture = mdl_factory->create_texture(
            transaction,
            /*file_path*/ nullptr,
            shape,
            0.0f,
            /*selector*/ nullptr,
            /*shared*/ false,
            context.get());
        MI_CHECK_CTX_CODE( context, -1);
        MI_CHECK( !texture);
        context->clear_messages();

        texture = mdl_factory->create_texture(
            transaction,
            "mdl_elements/resources/test.png",
            shape,
            0.0f,
            /*selector*/ nullptr,
            /*shared*/ false,
            context.get());
        MI_CHECK_CTX_CODE( context, -2);
        MI_CHECK( !texture);
        context->clear_messages();

        texture = mdl_factory->create_texture(
            transaction,
            "/mdl_elements/resources/test.ies",
            shape,
            0.0f,
            /*selector*/ nullptr,
            /*shared*/ false,
            context.get());
        MI_CHECK_CTX_CODE( context, -3);
        MI_CHECK( !texture);
        context->clear_messages();

        texture = mdl_factory->create_texture(
            transaction,
            "/mdl_elements/resources/non-existing.png",
            shape,
            0.0f,
            /*selector*/ nullptr,
            /*shared*/ false,
            context.get());
        MI_CHECK_CTX_CODE( context, -4);
        MI_CHECK( !texture);
        context->clear_messages();

        // -5 difficult to test

        texture = mdl_factory->create_texture(
            transaction,
            "/mdl_elements/resources/test.png",
            shape,
            0.0f,
            "X",
            /*shared*/ false,
            context.get());
        MI_CHECK_CTX_CODE( context, -7);
        MI_CHECK( !texture);
        context->clear_messages();

        texture = mdl_factory->create_texture(
            transaction,
            "/mdl_elements/resources/test.dds",
            shape,
            0.0f,
            "X",
            /*shared*/ false,
            context.get());
        MI_CHECK_CTX_CODE( context, -10);
        MI_CHECK( !texture);
        context->clear_messages();
    }
    {
        // light profiles
        mi::base::Handle<mi::neuraylib::IValue_light_profile> light_profile;

        light_profile = mdl_factory->create_light_profile(
            transaction, "/mdl_elements/resources/test.ies", /*shared*/ false, context.get());
        MI_CHECK_CTX( context);
        MI_CHECK( light_profile);
        MI_CHECK_EQUAL_CSTR( light_profile->get_file_path(), "/mdl_elements/resources/test.ies");
        context->clear_messages();

        light_profile = mdl_factory->create_light_profile(
            transaction, "/test_archives/test_in_archive.ies", /*shared*/ false, context.get());
        MI_CHECK_CTX( context);
        MI_CHECK( light_profile);
        MI_CHECK_EQUAL_CSTR( light_profile->get_file_path(), "/test_archives/test_in_archive.ies");
        context->clear_messages();

        light_profile = mdl_factory->create_light_profile(
            /*transaction*/ nullptr,
            "/mdl_elements/resources/test.ies",
            /*shared*/ false,
            context.get());
        MI_CHECK_CTX_CODE( context, -1);
        MI_CHECK( !light_profile);
        context->clear_messages();

        light_profile = mdl_factory->create_light_profile(
            transaction, /*file_path*/ nullptr, /*shared*/ false, context.get());
        MI_CHECK_CTX_CODE( context, -1);
        MI_CHECK( !light_profile);
        context->clear_messages();

        light_profile = mdl_factory->create_light_profile(
            transaction, "mdl_elements/resources/test.ies", /*shared*/ false, context.get());
        MI_CHECK_CTX_CODE( context, -2);
        MI_CHECK( !light_profile);
        context->clear_messages();

        light_profile = mdl_factory->create_light_profile(
            transaction, "/mdl_elements/resources/test.png", /*shared*/ false, context.get());
        MI_CHECK_CTX_CODE( context, -3);
        MI_CHECK( !light_profile);
        context->clear_messages();

        light_profile = mdl_factory->create_light_profile(
            transaction,
            "/mdl_elements/resources/non-existing.ies",
            /*shared*/ false,
            context.get());
        MI_CHECK_CTX_CODE( context, -4);
        MI_CHECK( !light_profile);
        context->clear_messages();

        // -5 difficult to test

        light_profile = mdl_factory->create_light_profile(
            transaction,
            "/mdl_elements/resources/test_file_format_error.ies",
            /*shared*/ false,
            context.get());
        MI_CHECK_CTX_CODE( context, -7);
        MI_CHECK( !light_profile);
        context->clear_messages();
    }
    {
        // BSDF measurements
        mi::base::Handle<mi::neuraylib::IValue_bsdf_measurement> bsdf_measurement;

        bsdf_measurement = mdl_factory->create_bsdf_measurement(
            transaction, "/mdl_elements/resources/test.mbsdf", /*shared*/ false, context.get());
        MI_CHECK_CTX( context);
        MI_CHECK( bsdf_measurement);
        MI_CHECK_EQUAL_CSTR(
            bsdf_measurement->get_file_path(), "/mdl_elements/resources/test.mbsdf");
        context->clear_messages();

        bsdf_measurement = mdl_factory->create_bsdf_measurement(
            transaction, "/test_archives/test_in_archive.mbsdf", /*shared*/ false, context.get());
        MI_CHECK_CTX( context);
        MI_CHECK( bsdf_measurement);
        MI_CHECK_EQUAL_CSTR(
            bsdf_measurement->get_file_path(), "/test_archives/test_in_archive.mbsdf");
        context->clear_messages();

        bsdf_measurement = mdl_factory->create_bsdf_measurement(
            /*transaction*/ nullptr,
            "/mdl_elements/resources/test.mbsdf",
            /*shared*/ false,
            context.get());
        MI_CHECK_CTX_CODE( context, -1);
        MI_CHECK( !bsdf_measurement);
        context->clear_messages();

        bsdf_measurement = mdl_factory->create_bsdf_measurement(
            transaction, /*file_path*/ nullptr, /*shared*/ false, context.get());
        MI_CHECK_CTX_CODE( context, -1);
        MI_CHECK( !bsdf_measurement);
        context->clear_messages();

        bsdf_measurement = mdl_factory->create_bsdf_measurement(
            transaction, "mdl_elements/resources/test.mbsdf", /*shared*/ false, context.get());
        MI_CHECK_CTX_CODE( context, -2);
        MI_CHECK( !bsdf_measurement);
        context->clear_messages();

        bsdf_measurement = mdl_factory->create_bsdf_measurement(
            transaction,
            "/mdl_elements/resources/non-existing.png",
            /*shared*/ false,
            context.get());
        MI_CHECK_CTX_CODE( context, -3);
        MI_CHECK( !bsdf_measurement);
        context->clear_messages();

        bsdf_measurement = mdl_factory->create_bsdf_measurement(
            transaction,
            "/mdl_elements/resources/non-existing.mbsdf",
            /*shared*/ false,
            context.get());
        MI_CHECK_CTX_CODE( context, -4);
        MI_CHECK( !bsdf_measurement);
        context->clear_messages();

        // -5 difficult to test

        bsdf_measurement = mdl_factory->create_bsdf_measurement(
            transaction,
            "/bsdf_measurement/test_file_format_error.mbsdf",
            /*shared*/ false,
            context.get());
        MI_CHECK_CTX_CODE( context, -7);
        MI_CHECK( !bsdf_measurement);
        context->clear_messages();
    }
}

void check_resources2(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::INeuray* neuray)
{
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
        neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());
    mi::base::Handle<mi::neuraylib::IImage_api> image_api(
        neuray->get_api_component<mi::neuraylib::IImage_api>());

    std::string path = MI::TEST::mi_src_path( "io/scene/mdl_elements/resources");
    {
        std::string file = path + dir_sep + "test.png";
        mi::base::Handle<mi::neuraylib::IReader> reader( mdl_impexp_api->create_reader(
            file.c_str()));
        MI_CHECK( reader);

        mi::base::Handle<mi::neuraylib::IImage> image(
           transaction->create<mi::neuraylib::IImage>( "Image"));
        result  = image->reset_reader( reader.get(), "png");
        MI_CHECK_EQUAL( result, 0);
    }

    {
        std::string file = path + dir_sep + "test.ies";
        mi::base::Handle<mi::neuraylib::IReader> reader( mdl_impexp_api->create_reader(
            file.c_str()));
        MI_CHECK( reader);

        mi::base::Handle<mi::neuraylib::ILightprofile> lp(
           transaction->create<mi::neuraylib::ILightprofile>( "Lightprofile"));
        result  = lp->reset_reader( reader.get());
        MI_CHECK_EQUAL( result, 0);
    }

    {
        std::string file = path + dir_sep + "test.mbsdf";
        mi::base::Handle<mi::neuraylib::IReader> reader( mdl_impexp_api->create_reader(
            file.c_str()));
        MI_CHECK( reader);

        mi::base::Handle<mi::neuraylib::IBsdf_measurement> bsdfm(
            transaction->create<mi::neuraylib::IBsdf_measurement>( "Bsdf_measurement"));
        result = bsdfm->reset_reader( reader.get());
        MI_CHECK_EQUAL( result, 0);
    }

    // selectors via intermediate canvas
    {
        std::string file = path + dir_sep + "test.png";
        mi::base::Handle<mi::neuraylib::IReader> reader( mdl_impexp_api->create_reader(
            file.c_str()));
        MI_CHECK( reader);

        mi::base::Handle<mi::neuraylib::ICanvas> canvas;
        canvas = image_api->create_canvas_from_reader( reader.get(), "png");
        MI_CHECK( canvas);
        MI_CHECK_EQUAL_CSTR( canvas->get_type(), "Rgb");
        canvas = image_api->create_canvas_from_reader( reader.get(), "png", "R");
        MI_CHECK( canvas);
        MI_CHECK_EQUAL_CSTR( canvas->get_type(), "Sint8");

        mi::base::Handle<mi::neuraylib::IImage> image(
           transaction->create<mi::neuraylib::IImage>( "Image"));
        bool ok = image->set_from_canvas( canvas.get(), /*selector*/ nullptr);
        MI_CHECK( ok);
        MI_CHECK( !image->get_selector());
        ok = image->set_from_canvas( canvas.get(), "whatever");
        MI_CHECK( ok);
        MI_CHECK_EQUAL_CSTR( image->get_selector(), "whatever");
    }
}

void check_cycles(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
        transaction->access<mi::neuraylib::IFunction_definition>(
            "mdl::" TEST_MDL "::fd_cycle(color)"));

    // Create two function calls
    mi::base::Handle<mi::neuraylib::IFunction_call> fc1( fd->create_function_call( nullptr));
    check_success( fc1);
    transaction->store( fc1.get(), "mdl::" TEST_MDL "::fc_cycle1");

    mi::base::Handle<mi::neuraylib::IFunction_call> fc2( fd->create_function_call( nullptr));
    check_success( fc2);
    transaction->store( fc2.get(), "mdl::" TEST_MDL "::fc_cycle2");

    // Create cycle
    mi::neuraylib::Argument_editor ae_fc1(
        transaction, "mdl::" TEST_MDL "::fc_cycle1", mdl_factory);
    check_success( ae_fc1.set_call( "tint", "mdl::" TEST_MDL "::fc_cycle2") == 0);

    mi::neuraylib::Argument_editor ae_fc2(
        transaction, "mdl::" TEST_MDL "::fc_cycle2", mdl_factory);
    check_success( ae_fc2.set_call( "tint", "mdl::" TEST_MDL "::fc_cycle1") == 0);

    // Create material instance
    mi::base::Handle<const mi::neuraylib::IFunction_definition> md(
        transaction->access<mi::neuraylib::IFunction_definition>(
            "mdl::" TEST_MDL "::md_1(color)"));
    mi::base::Handle<mi::neuraylib::IFunction_call> mi( md->create_function_call( nullptr));
    check_success( mi);
    transaction->store( mi.get(), "mdl::" TEST_MDL "::mi_cycle");

    // Call cycle
    mi::neuraylib::Argument_editor ae_mi(
        transaction, "mdl::" TEST_MDL "::mi_cycle", mdl_factory);
    check_success( ae_mi.set_call( "tint", "mdl::" TEST_MDL "::fc_cycle1") == -6);
}

void check_uniform_auto_varying(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    {
        mi::base::Handle<const mi::neuraylib::IType> type;
        mi::base::Handle<const mi::neuraylib::IType_list> types;
        mi::base::Handle<const mi::neuraylib::IExpression> arg;
        mi::base::Handle<const mi::neuraylib::IExpression_list> args;

        // check properties
        mi::base::Handle<const mi::neuraylib::IFunction_definition> c_md(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::" TEST_MDL "::md_parameters_uniform_auto_varying(int,int,int)"));
        types = c_md->get_parameter_types();

        type = types->get_type( zero_size);
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_UNIFORM, type->get_all_type_modifiers());
        type = types->get_type( "param0_uniform");
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_UNIFORM, type->get_all_type_modifiers());
        type = types->get_type( 1);
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_NONE, type->get_all_type_modifiers());
        type = types->get_type( "param1_auto");
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_NONE, type->get_all_type_modifiers());
        type = types->get_type( 2);
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_VARYING, type->get_all_type_modifiers());
        type = types->get_type( "param2_varying");
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_VARYING, type->get_all_type_modifiers());

        mi::base::Handle<const mi::neuraylib::IFunction_call> c_mi(
            transaction->access<mi::neuraylib::IFunction_call>(
                "mdl::" TEST_MDL "::mi_parameters_uniform_auto_varying"));
        args = c_mi->get_arguments();

        arg = args->get_expression( zero_size);
        type = arg->get_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_NONE, type->get_all_type_modifiers());
        arg = args->get_expression( "param0_uniform");
        type = arg->get_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_NONE, type->get_all_type_modifiers());
        arg = args->get_expression( 1);
        type = arg->get_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_NONE, type->get_all_type_modifiers());
        arg = args->get_expression( "param1_auto");
        type = arg->get_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_NONE, type->get_all_type_modifiers());
        arg = args->get_expression( 2);
        type = arg->get_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_NONE, type->get_all_type_modifiers());
        arg = args->get_expression( "param2_varying");
        type = arg->get_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_NONE, type->get_all_type_modifiers());

        mi::base::Handle<const mi::neuraylib::IFunction_definition> c_fd(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::" TEST_MDL "::fd_parameters_uniform_auto_varying(int,int,int)"));
        types = c_fd->get_parameter_types();

        type = types->get_type( zero_size);
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_UNIFORM, type->get_all_type_modifiers());
        type = types->get_type( "param0_uniform");
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_UNIFORM, type->get_all_type_modifiers());
        type = types->get_type( 1);
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_NONE, type->get_all_type_modifiers());
        type = types->get_type( "param1_auto");
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_NONE, type->get_all_type_modifiers());
        type = types->get_type( 2);
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_VARYING, type->get_all_type_modifiers());
        type = types->get_type( "param2_varying");
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_VARYING, type->get_all_type_modifiers());

        c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
            "mdl::" TEST_MDL "::fd_return_uniform()");
        type = c_fd->get_return_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_UNIFORM, type->get_all_type_modifiers());

        c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
            "mdl::" TEST_MDL "::fd_return_auto()");
        type = c_fd->get_return_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_NONE, type->get_all_type_modifiers());

        c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
            "mdl::" TEST_MDL "::fd_return_varying()");
        type = c_fd->get_return_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_VARYING, type->get_all_type_modifiers());

        mi::base::Handle<const mi::neuraylib::IFunction_call> c_fc(
            transaction->access<mi::neuraylib::IFunction_call>(
                "mdl::" TEST_MDL "::fc_parameters_uniform_auto_varying"));
        args = c_mi->get_arguments();

        arg = args->get_expression( zero_size);
        type = arg->get_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_NONE, type->get_all_type_modifiers());
        arg = args->get_expression( "param0_uniform");
        type = arg->get_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_NONE, type->get_all_type_modifiers());
        arg = args->get_expression( 1);
        type = arg->get_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_NONE, type->get_all_type_modifiers());
        arg = args->get_expression( "param1_auto");
        type = arg->get_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_NONE, type->get_all_type_modifiers());
        arg = args->get_expression( 2);
        type = arg->get_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_NONE, type->get_all_type_modifiers());
        arg = args->get_expression( "param2_varying");
        type = arg->get_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_NONE, type->get_all_type_modifiers());

        c_fc = transaction->access<mi::neuraylib::IFunction_call>(
            "mdl::" TEST_MDL "::fc_return_uniform");
        type = c_fc->get_return_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_UNIFORM, type->get_all_type_modifiers());

        c_fc = transaction->access<mi::neuraylib::IFunction_call>(
            "mdl::" TEST_MDL "::fc_return_auto");
        type = c_fc->get_return_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_NONE, type->get_all_type_modifiers());

        c_fc = transaction->access<mi::neuraylib::IFunction_call>(
            "mdl::" TEST_MDL "::fc_return_varying");
        type = c_fc->get_return_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_VARYING, type->get_all_type_modifiers());
    }

    {
        // check set_argument() -- reject varying arguments for uniform parameters
        mi::base::Handle<mi::neuraylib::IExpression_call> expr_uniform(
            ef->create_call( "mdl::" TEST_MDL "::fc_return_uniform"));
        mi::base::Handle<mi::neuraylib::IExpression_call> expr_auto(
            ef->create_call( "mdl::" TEST_MDL "::fc_return_auto"));
        mi::base::Handle<mi::neuraylib::IExpression_call> expr_varying(
            ef->create_call( "mdl::" TEST_MDL "::fc_return_varying"));

        mi::base::Handle<mi::neuraylib::IFunction_call> m_mi(
            transaction->edit<mi::neuraylib::IFunction_call>(
                "mdl::" TEST_MDL "::mi_parameters_uniform_auto_varying"));
        MI_CHECK_EQUAL( 0, m_mi->set_argument( zero_size, expr_uniform.get()));
        MI_CHECK_EQUAL( 0, m_mi->set_argument( "param0_uniform", expr_uniform.get()));
        MI_CHECK_EQUAL( 0, m_mi->set_argument( zero_size, expr_auto.get()));
        MI_CHECK_EQUAL( 0, m_mi->set_argument( "param0_uniform", expr_auto.get()));
        MI_CHECK_EQUAL( -5, m_mi->set_argument( zero_size, expr_varying.get()));
        MI_CHECK_EQUAL( -5, m_mi->set_argument( "param0_uniform", expr_varying.get()));

        MI_CHECK_EQUAL( 0, m_mi->set_argument( 1, expr_uniform.get()));
        MI_CHECK_EQUAL( 0, m_mi->set_argument( "param1_auto", expr_uniform.get()));
        MI_CHECK_EQUAL( 0, m_mi->set_argument( 1, expr_auto.get()));
        MI_CHECK_EQUAL( 0, m_mi->set_argument( "param1_auto", expr_auto.get()));
        MI_CHECK_EQUAL( 0, m_mi->set_argument( 1, expr_varying.get()));
        MI_CHECK_EQUAL( 0, m_mi->set_argument( "param1_auto", expr_varying.get()));

        MI_CHECK_EQUAL( 0, m_mi->set_argument( 2, expr_uniform.get()));
        MI_CHECK_EQUAL( 0, m_mi->set_argument( "param2_varying", expr_uniform.get()));
        MI_CHECK_EQUAL( 0, m_mi->set_argument( 2, expr_auto.get()));
        MI_CHECK_EQUAL( 0, m_mi->set_argument( "param2_varying", expr_auto.get()));
        MI_CHECK_EQUAL( 0, m_mi->set_argument( 2, expr_varying.get()));
        MI_CHECK_EQUAL( 0, m_mi->set_argument( "param2_varying", expr_varying.get()));

        mi::base::Handle<mi::neuraylib::IFunction_call> m_fc(
            transaction->edit<mi::neuraylib::IFunction_call>(
                "mdl::" TEST_MDL "::fc_parameters_uniform_auto_varying"));
        MI_CHECK_EQUAL( 0, m_fc->set_argument( zero_size, expr_uniform.get()));
        MI_CHECK_EQUAL( 0, m_fc->set_argument( "param0_uniform", expr_uniform.get()));
        MI_CHECK_EQUAL( 0, m_fc->set_argument( zero_size, expr_auto.get()));
        MI_CHECK_EQUAL( 0, m_fc->set_argument( "param0_uniform", expr_auto.get()));
        MI_CHECK_EQUAL( -5, m_fc->set_argument( zero_size, expr_varying.get()));
        MI_CHECK_EQUAL( -5, m_fc->set_argument( "param0_uniform", expr_varying.get()));

        MI_CHECK_EQUAL( 0, m_fc->set_argument( 1, expr_uniform.get()));
        MI_CHECK_EQUAL( 0, m_fc->set_argument( "param1_auto", expr_uniform.get()));
        MI_CHECK_EQUAL( 0, m_fc->set_argument( 1, expr_auto.get()));
        MI_CHECK_EQUAL( 0, m_fc->set_argument( "param1_auto", expr_auto.get()));
        MI_CHECK_EQUAL( 0, m_fc->set_argument( 1, expr_varying.get()));
        MI_CHECK_EQUAL( 0, m_fc->set_argument( "param1_auto", expr_varying.get()));

        MI_CHECK_EQUAL( 0, m_fc->set_argument( 2, expr_uniform.get()));
        MI_CHECK_EQUAL( 0, m_fc->set_argument( "param2_varying", expr_uniform.get()));
        MI_CHECK_EQUAL( 0, m_fc->set_argument( 2, expr_auto.get()));
        MI_CHECK_EQUAL( 0, m_fc->set_argument( "param2_varying", expr_auto.get()));
        MI_CHECK_EQUAL( 0, m_fc->set_argument( 2, expr_varying.get()));
        MI_CHECK_EQUAL( 0, m_fc->set_argument( "param2_varying", expr_varying.get()));

        // check set_call() -- reject all modifier variations
        MI_CHECK_EQUAL(  0, expr_uniform->set_call( "mdl::" TEST_MDL "::fc_return_uniform"));
        MI_CHECK_EQUAL( -4, expr_uniform->set_call( "mdl::" TEST_MDL "::fc_return_auto"));
        MI_CHECK_EQUAL( -4, expr_uniform->set_call( "mdl::" TEST_MDL "::fc_return_varying"));
        MI_CHECK_EQUAL( -4, expr_auto->set_call( "mdl::" TEST_MDL "::fc_return_uniform"));
        MI_CHECK_EQUAL(  0, expr_auto->set_call( "mdl::" TEST_MDL "::fc_return_auto"));
        MI_CHECK_EQUAL( -4, expr_auto->set_call( "mdl::" TEST_MDL "::fc_return_varying"));
        MI_CHECK_EQUAL( -4, expr_varying->set_call( "mdl::" TEST_MDL "::fc_return_uniform"));
        MI_CHECK_EQUAL( -4, expr_varying->set_call( "mdl::" TEST_MDL "::fc_return_auto"));
        MI_CHECK_EQUAL(  0, expr_varying->set_call( "mdl::" TEST_MDL "::fc_return_varying"));
    }

#if 0 // this can only be triggered by overwriting DB elements
    {
        // check compilation of material where a uniform parameter has a varying function with auto
        // return type (=> treated as varying return type according to spec) attached
        mi::base::Handle<const mi::neuraylib::IFunction_call> mi(
            transaction->access<mi::neuraylib::IFunction_call>(
                "mdl::" TEST_MDL "::mi_parameter_uniform"));
        mi::base::Handle<const mi::neuraylib::ICompiled_material> cm(
            mi_mi->create_compiled_material(
                mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS, 1.0f, 380.0f, 780.0f, &result));
        MI_CHECK_EQUAL( -3, result);
        MI_CHECK( !cm);
    }
#endif
}

void check_analyze_uniform(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    // create instances of "fd_parameters_uniform_auto_varying_color" with the uniform/varying
    // parameter indirectly connected to state::normal()

    {
        // instantiate ::state::normal() as "analyze_varying_normal"
        mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::state::normal()"));
        mi::base::Handle<mi::neuraylib::IFunction_call> fc(
            fd->create_function_call( nullptr, &result));
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK_EQUAL( 0, transaction->store( fc.get(), "analyze_varying_normal"));
    }
    {
        // instantiate ::color(float3) as "analyze_auto_color" with "color" connected to
        // "analyze_varying_normal"
        mi::base::Handle<mi::neuraylib::IExpression> color(
           ef->create_call( "analyze_varying_normal"));
        mi::base::Handle<mi::neuraylib::IExpression_list> args(
           ef->create_expression_list());
        args->add_expression( "rgb", color.get());

        mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
            transaction->access<mi::neuraylib::IFunction_definition>(
               "mdl::color(float3)"));
        mi::base::Handle<mi::neuraylib::IFunction_call> fc(
            fd->create_function_call( args.get(), &result));
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK_EQUAL( 0, transaction->store( fc.get(), "analyze_auto_color"));
    }
    {
        // instantiate "fd_parameters_uniform_auto_varying_color" as "analyze_uniform_root" with the
        // uniform "param0_uniform" parameter connected to "analyze_auto_color"
        mi::base::Handle<mi::neuraylib::IExpression> call(
           ef->create_call( "analyze_auto_color"));
        mi::base::Handle<mi::neuraylib::IExpression_list> args(
           ef->create_expression_list());
        args->add_expression( "param0_uniform", call.get());

        mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::" TEST_MDL "::fd_parameters_uniform_auto_varying_color(color,color,color)"));
        mi::base::Handle<mi::neuraylib::IFunction_call> fc(
            fd->create_function_call( args.get(), &result));
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK_EQUAL( 0, transaction->store( fc.get(), "analyze_uniform_root"));
    }
    {
        // instantiate "fd_parameters_uniform_auto_varying_color" as "analyze_varying_root" with the
        // uniform "param2_varying" parameter connected to "analyze_auto_color"
        mi::base::Handle<mi::neuraylib::IExpression> call(
           ef->create_call( "analyze_auto_color"));
        mi::base::Handle<mi::neuraylib::IExpression_list> args(
           ef->create_expression_list());
        args->add_expression( "param2_varying", call.get());

        mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::" TEST_MDL "::fd_parameters_uniform_auto_varying_color(color,color,color)"));
        mi::base::Handle<mi::neuraylib::IFunction_call> fc(
            fd->create_function_call( args.get(), &result));
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK_EQUAL( 0, transaction->store( fc.get(), "analyze_varying_root"));
    }
    {
        bool query_result = false;
        mi::base::Handle<mi::IString> error_string( transaction->create<mi::IString>());

        // The subgraph starting at "analyze_auto_color" is ok,
        query_result = false;
        mdl_factory->analyze_uniform(
            transaction, "analyze_auto_color", false, nullptr,
            query_result, error_string.get(), context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL_CSTR( error_string->get_c_str(), "");

        // Attached to the uniform "param0_uniform" parameter, it is broken.
        query_result = false;
        mdl_factory->analyze_uniform(
            transaction, "analyze_uniform_root", false, nullptr,
            query_result, error_string.get(), context.get());
        MI_CHECK_EQUAL_CSTR( error_string->get_c_str(), "param0_uniform.rgb");

        // Attached to the varying "param2_varying" parameter, it works.
        query_result = false;
        mdl_factory->analyze_uniform(
            transaction, "analyze_varying_root", false, nullptr,
            query_result, error_string.get(), context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL_CSTR( error_string->get_c_str(), "");

        // Access the "analyze_auto_color" node.
        mi::base::Handle<const mi::neuraylib::IFunction_call> query_fc(
            transaction->access<mi::neuraylib::IFunction_call>(
                "analyze_auto_color"));
        mi::base::Handle<const mi::neuraylib::IExpression_list> query_args(
            query_fc->get_arguments());
        mi::base::Handle<const mi::neuraylib::IExpression> query_expr(
            query_args->get_expression( "rgb"));

        // And query explicitly that node.
        query_result = false;
        mdl_factory->analyze_uniform(
            transaction, "analyze_uniform_root", false,
            query_expr.get(), query_result, nullptr, context.get());
        MI_CHECK_EQUAL( query_result, true);

        // Access the "param0_uniform" node (arguments of the root expression need a special
        // handling internally).
        query_fc = transaction->access<mi::neuraylib::IFunction_call>(
                "analyze_uniform_root");
        query_args = query_fc->get_arguments();
        query_expr = query_args->get_expression( "param0_uniform");

        // And query explicitly that node.
        query_result = false;
        mdl_factory->analyze_uniform(
            transaction, "analyze_uniform_root", false,
            query_expr.get(), query_result, nullptr, context.get());
        MI_CHECK_EQUAL( query_result, true);
    }
}

void check_deep_copy_of_defaults( mi::neuraylib::ITransaction* transaction)
{
    // Check that a deep copy is done when defaults are used due to the lack of arguments.

    // get DB name of the function call referenced in the default for param0 of
    // fd_default_call()
    mi::base::Handle<const mi::neuraylib::IFunction_definition> c_fd(
        transaction->access<mi::neuraylib::IFunction_definition>(
            "mdl::" TEST_MDL "::fd_default_call(int)"));
    MI_CHECK( c_fd);
    mi::base::Handle<const mi::neuraylib::IExpression_list> defaults( c_fd->get_defaults());
    mi::base::Handle<const mi::neuraylib::IExpression_call> param0_default(
        defaults->get_expression<mi::neuraylib::IExpression_call>( "param0"));
    const char* param0_default_call = param0_default->get_call();

    // instantiate fd_default_call() using the default,
    // get DB name of the function call referenced in the argument param0
    mi::base::Handle<mi::neuraylib::IFunction_call> m_fc(
        c_fd->create_function_call( nullptr, &result));
    MI_CHECK_EQUAL( 0, result);
    mi::base::Handle<const mi::neuraylib::IExpression_list> args( m_fc->get_arguments());
    mi::base::Handle<const mi::neuraylib::IExpression_call> param0_argument(
        args->get_expression<mi::neuraylib::IExpression_call>( "param0"));
    const char* param0_argument_call = param0_argument->get_call();

    // check that both are different (but have the same prefix)
    MI_CHECK_NOT_EQUAL_CSTR( param0_argument_call, param0_default_call);
    MI_CHECK( strncmp( param0_argument_call, "mdl::" TEST_MDL "::fd_1_", 20) == 0);
    MI_CHECK( strncmp( param0_default_call, "mdl::" TEST_MDL "::fd_1(int)_", 25) == 0);
}

void check_immutable_defaults(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory)
{
    // Check that function calls referenced by defaults are immutable.

    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    // Get DB name of the function call referenced in the default for param0 of
    // fd_default_call()
    mi::base::Handle<const mi::neuraylib::IFunction_definition> c_fd(
        transaction->access<mi::neuraylib::IFunction_definition>(
            "mdl::" TEST_MDL "::fd_default_call(int)"));
    MI_CHECK( c_fd);
    mi::base::Handle<const mi::neuraylib::IExpression_list> defaults( c_fd->get_defaults());
    mi::base::Handle<const mi::neuraylib::IExpression_call> param0_default(
        defaults->get_expression<mi::neuraylib::IExpression_call>( "param0"));
    const char* param0_default_call = param0_default->get_call();

    // Setting the param0 argument of the referenced function call should fail.
    mi::base::Handle<mi::neuraylib::IFunction_call> fc(
        transaction->edit<mi::neuraylib::IFunction_call>( param0_default_call));
    MI_CHECK( fc);
    mi::base::Handle<mi::neuraylib::IValue> value( vf->create_int( 42));
    mi::base::Handle<mi::neuraylib::IExpression> expression(
        ef->create_constant( value.get()));
    MI_CHECK_EQUAL( -4, fc->set_argument( "param0", expression.get()));
}

void check_annotations(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));

    {
        // check ::anno module

        mi::base::Handle<const mi::neuraylib::IModule> module(
            transaction->access<mi::neuraylib::IModule>( "mdl::anno"));
        mi::Size n = module->get_annotation_definition_count();
        MI_CHECK( n > 0);

        mi::base::Handle<const mi::neuraylib::IAnnotation_definition> ad(
            module->get_annotation_definition( nullptr));
        MI_CHECK( !ad);

        ad = module->get_annotation_definition( "::anno::display_name(string)");
        MI_CHECK( ad);

        MI_CHECK_EQUAL_CSTR( ad->get_module(), "mdl::anno");
        MI_CHECK_EQUAL_CSTR( ad->get_name(), "::anno::display_name(string)");
        MI_CHECK_EQUAL_CSTR( ad->get_mdl_module_name(), "::anno");
        MI_CHECK_EQUAL_CSTR( ad->get_mdl_simple_name(), "display_name");
        MI_CHECK_EQUAL_CSTR( ad->get_mdl_parameter_type_name( 0), "string");
        MI_CHECK_EQUAL( ad->get_parameter_count(), 1);
        MI_CHECK_EQUAL( ad->get_parameter_index( "name"), 0);
        MI_CHECK_EQUAL( ad->get_parameter_index( "does_not_exist"), mi::Size(-1));
        MI_CHECK_EQUAL_CSTR( ad->get_parameter_name( 0), "name");
        MI_CHECK_EQUAL( ad->get_parameter_name( 1), nullptr);
        MI_CHECK_EQUAL( ad->get_semantic(),
            mi::neuraylib::IAnnotation_definition::AS_DISPLAY_NAME_ANNOTATION);
        MI_CHECK_EQUAL( ad->is_exported(), true);

        mi::neuraylib::Mdl_version since, removed;
        ad->get_mdl_version( since, removed);
        MI_CHECK_EQUAL( since, mi::neuraylib::MDL_VERSION_1_0);
        MI_CHECK_EQUAL( removed, mi::neuraylib::MDL_VERSION_INVALID);

        mi::base::Handle<const mi::neuraylib::IType_list> types( ad->get_parameter_types());
        MI_CHECK_EQUAL( types->get_size(), 1);
        mi::base::Handle<const mi::neuraylib::IType> type_name( types->get_type( "name"));
        MI_CHECK_EQUAL( type_name->get_kind(), mi::neuraylib::IType::TK_STRING);

        mi::base::Handle<const mi::neuraylib::IExpression_list> anno_defaults( ad->get_defaults());
        MI_CHECK( anno_defaults);
        MI_CHECK_EQUAL( anno_defaults->get_size(), 0);

        mi::base::Handle<const mi::neuraylib::IAnnotation_block> block( ad->get_annotations());
        MI_CHECK( !block);

        // added in 1.3
        ad = module->get_annotation_definition( "::anno::deprecated()");
        ad->get_mdl_version( since, removed);
        MI_CHECK_EQUAL( since, mi::neuraylib::MDL_VERSION_1_3);
        MI_CHECK_EQUAL( removed, mi::neuraylib::MDL_VERSION_INVALID);

        // removed in MDL 1.3
        ad = module->get_annotation_definition( "::anno::version_number$1.2(int,int,int,int)");
        ad->get_mdl_version( since, removed);
        MI_CHECK_EQUAL( since, mi::neuraylib::MDL_VERSION_1_0);
        MI_CHECK_EQUAL( removed, mi::neuraylib::MDL_VERSION_1_3);

        // custom annotation
        module = transaction->access<mi::neuraylib::IModule>( "mdl::" TEST_MDL);
        ad = module->get_annotation_definition( "::" TEST_MDL "::anno_2_int(int,int)");
        ad->get_mdl_version( since, removed);
        MI_CHECK_EQUAL( since, mi::neuraylib::MDL_VERSION_1_9);
        MI_CHECK_EQUAL( removed, mi::neuraylib::MDL_VERSION_INVALID);
    }

    mi::base::Handle<const mi::neuraylib::IAnnotation_block> block;
    mi::base::Handle<const mi::neuraylib::IAnnotation> anno;
    mi::base::Handle<const mi::neuraylib::IAnnotation_definition> ad;
    mi::base::Handle<const mi::neuraylib::IAnnotation_list> annos;
    mi::base::Handle<const mi::neuraylib::IExpression_list> args;
    mi::base::Handle<const mi::neuraylib::IExpression_constant> arg;
    mi::base::Handle<const mi::neuraylib::IValue_string> value_string;
    mi::base::Handle<const mi::neuraylib::IValue_float> value_float;

    {
        // check annotations of a module

        mi::base::Handle<const mi::neuraylib::IModule> module(
            transaction->access<mi::neuraylib::IModule>( "mdl::" TEST_MDL));
        MI_CHECK( module);

        block = module->get_annotations();

        anno = block->get_annotation( zero_size);
        MI_CHECK_EQUAL_CSTR( "::anno::description(string)", anno->get_name());
        args = anno->get_arguments();
        MI_CHECK_EQUAL( 1, args->get_size());
        arg = args->get_expression<mi::neuraylib::IExpression_constant>( "description");
        value_string = arg->get_value<mi::neuraylib::IValue_string>();
        MI_CHECK_EQUAL_CSTR( "module description annotation", value_string->get_value());
        ad = anno->get_definition();
        MI_CHECK( ad);
        MI_CHECK_EQUAL_CSTR( ad->get_name(), "::anno::description(string)");
        MI_CHECK_EQUAL_CSTR( ad->get_mdl_module_name(), "::anno");
        MI_CHECK_EQUAL_CSTR( ad->get_mdl_simple_name(), "description");
        MI_CHECK_EQUAL_CSTR( ad->get_mdl_parameter_type_name( 0), "string");
        MI_CHECK_EQUAL( ad->get_parameter_count(), 1);
        MI_CHECK_EQUAL( ad->get_parameter_index( "description"), 0);
        MI_CHECK_EQUAL_CSTR( ad->get_parameter_name( 0), "description");
    }
    {
        // check block of a function definition

        mi::base::Handle<const mi::neuraylib::IFunction_definition> c_fd(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::" TEST_MDL "::fd_with_annotations(color,float)"));
        MI_CHECK( c_fd);

        block = c_fd->get_annotations();

        anno = block->get_annotation( zero_size);
        MI_CHECK_EQUAL_CSTR( "::anno::description(string)", anno->get_name());
        args = anno->get_arguments();
        MI_CHECK_EQUAL( 1, args->get_size());
        arg = args->get_expression<mi::neuraylib::IExpression_constant>( "description");
        value_string = arg->get_value<mi::neuraylib::IValue_string>();
        MI_CHECK_EQUAL_CSTR( "function description annotation", value_string->get_value());

        anno = block->get_annotation( 1);
        MI_CHECK_EQUAL_CSTR( "::anno::unused()", anno->get_name());
        args = anno->get_arguments();
        MI_CHECK_EQUAL( 0, args->get_size());

        ad = anno->get_definition();
        MI_CHECK( ad);
        MI_CHECK_EQUAL_CSTR( ad->get_name(), "::anno::unused()");
        MI_CHECK_EQUAL_CSTR( ad->get_mdl_module_name(), "::anno");
        MI_CHECK_EQUAL_CSTR( ad->get_mdl_simple_name(), "unused");
        MI_CHECK( !ad->get_mdl_parameter_type_name( 0));
        MI_CHECK_EQUAL( ad->get_parameter_count(), 0);

        annos = c_fd->get_parameter_annotations();
        block = annos->get_annotation_block( "param0");

        anno = block->get_annotation( zero_size);
        MI_CHECK_EQUAL_CSTR( "::anno::description(string)", anno->get_name());
        args = anno->get_arguments();
        MI_CHECK_EQUAL( 1, args->get_size());
        arg = args->get_expression<mi::neuraylib::IExpression_constant>( "description");
        value_string = arg->get_value<mi::neuraylib::IValue_string>();
        MI_CHECK_EQUAL_CSTR( "param0: one description annotation", value_string->get_value());

        block = annos->get_annotation_block( "param1");

        anno = block->get_annotation( zero_size);
        MI_CHECK_EQUAL_CSTR( "::anno::description(string)", anno->get_name());
        args = anno->get_arguments();
        MI_CHECK_EQUAL( 1, args->get_size());
        arg = args->get_expression<mi::neuraylib::IExpression_constant>( "description");
        value_string = arg->get_value<mi::neuraylib::IValue_string>();
        MI_CHECK_EQUAL_CSTR( "param1: two description annotations (1)", value_string->get_value());

        anno = block->get_annotation( 1);
        MI_CHECK_EQUAL_CSTR( "::anno::description(string)", anno->get_name());
        args = anno->get_arguments();
        MI_CHECK_EQUAL( 1, args->get_size());
        arg = args->get_expression<mi::neuraylib::IExpression_constant>( "description");
        value_string = arg->get_value<mi::neuraylib::IValue_string>();
        MI_CHECK_EQUAL_CSTR( "param1: two description annotations (2)", value_string->get_value());

        anno = block->get_annotation( 2);
        MI_CHECK_EQUAL_CSTR( "::anno::soft_range(float,float)", anno->get_name());
        args = anno->get_arguments();
        MI_CHECK_EQUAL( 2, args->get_size());
        arg = args->get_expression<mi::neuraylib::IExpression_constant>( "min");
        value_float = arg->get_value<mi::neuraylib::IValue_float>();
        MI_CHECK_EQUAL( 0.5f, value_float->get_value());
        arg = args->get_expression<mi::neuraylib::IExpression_constant>( "max");
        value_float = arg->get_value<mi::neuraylib::IValue_float>();
        MI_CHECK_EQUAL( 1.5f, value_float->get_value());

        ad = anno->get_definition();
        MI_CHECK( ad);
        MI_CHECK_EQUAL_CSTR( ad->get_name(), "::anno::soft_range(float,float)");
        MI_CHECK_EQUAL_CSTR( ad->get_mdl_module_name(), "::anno");
        MI_CHECK_EQUAL_CSTR( ad->get_mdl_simple_name(), "soft_range");
        MI_CHECK_EQUAL_CSTR( ad->get_mdl_parameter_type_name( 0), "float");
        MI_CHECK_EQUAL_CSTR( ad->get_mdl_parameter_type_name( 1), "float");
        MI_CHECK_EQUAL( ad->get_parameter_count(), 2);


        anno = block->get_annotation( 3);
        MI_CHECK_EQUAL_CSTR( "::anno::hard_range(float,float)", anno->get_name());
        args = anno->get_arguments();
        MI_CHECK_EQUAL( 2, args->get_size());
        arg = args->get_expression<mi::neuraylib::IExpression_constant>( "min");
        value_float = arg->get_value<mi::neuraylib::IValue_float>();
        MI_CHECK_EQUAL( 0.0f, value_float->get_value());
        arg = args->get_expression<mi::neuraylib::IExpression_constant>( "max");
        value_float = arg->get_value<mi::neuraylib::IValue_float>();
        MI_CHECK_EQUAL( 3.0f, value_float->get_value());

        anno = block->get_annotation( 4);
        MI_CHECK_EQUAL_CSTR( "::anno::display_name(string)", anno->get_name());
        args = anno->get_arguments();
        MI_CHECK_EQUAL( 1, args->get_size());
        arg = args->get_expression<mi::neuraylib::IExpression_constant>( "name");
        value_string = arg->get_value<mi::neuraylib::IValue_string>();
        MI_CHECK_EQUAL_CSTR( "fd_with_annotations_param1_display_name", value_string->get_value());

        block = c_fd->get_return_annotations();

        anno = block->get_annotation( zero_size);
        MI_CHECK_EQUAL_CSTR( "::anno::description(string)", anno->get_name());
        args = anno->get_arguments();
        MI_CHECK_EQUAL( 1, args->get_size());
        arg = args->get_expression<mi::neuraylib::IExpression_constant>( "description");
        value_string = arg->get_value<mi::neuraylib::IValue_string>();
        MI_CHECK_EQUAL_CSTR( "return type description annotation", value_string->get_value());
    }
    {
        // check annotation for enum types

        mi::base::Handle<const mi::neuraylib::IModule> module(
            transaction->access<mi::neuraylib::IModule>( "mdl::" TEST_MDL));
        mi::base::Handle<const mi::neuraylib::IType_list> types( module->get_types());
        MI_CHECK_EQUAL( 7, types->get_size());

        mi::base::Handle<const mi::neuraylib::IType_enum> type0(
            types->get_type<mi::neuraylib::IType_enum>( zero_size));
        MI_CHECK( type0);
        MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::Enum", types->get_name( 0));

        block = type0->get_annotations();
        anno = block->get_annotation( zero_size);
        MI_CHECK_EQUAL_CSTR( "::anno::description(string)", anno->get_name());
        args = anno->get_arguments();
        MI_CHECK_EQUAL( 1, args->get_size());
        arg = args->get_expression<mi::neuraylib::IExpression_constant>( "description");
        value_string = arg->get_value<mi::neuraylib::IValue_string>();
        MI_CHECK_EQUAL_CSTR( "enum annotation", value_string->get_value());

        ad = anno->get_definition();
        MI_CHECK( ad);
        MI_CHECK_EQUAL_CSTR( ad->get_name(), "::anno::description(string)");
        MI_CHECK_EQUAL_CSTR( ad->get_mdl_module_name(), "::anno");
        MI_CHECK_EQUAL_CSTR( ad->get_mdl_simple_name(), "description");
        MI_CHECK_EQUAL_CSTR( ad->get_mdl_parameter_type_name( 0), "string");
        MI_CHECK_EQUAL( ad->get_parameter_count(), 1);

        block = type0->get_value_annotations( zero_size);
        anno = block->get_annotation( zero_size);
        MI_CHECK_EQUAL_CSTR( "::anno::description(string)", anno->get_name());
        args = anno->get_arguments();
        MI_CHECK_EQUAL( 1, args->get_size());
        arg = args->get_expression<mi::neuraylib::IExpression_constant>( "description");
        value_string = arg->get_value<mi::neuraylib::IValue_string>();
        MI_CHECK_EQUAL_CSTR( "one annotation", value_string->get_value());

        ad = anno->get_definition();
        MI_CHECK( ad);
        MI_CHECK_EQUAL_CSTR( ad->get_name(), "::anno::description(string)");
        MI_CHECK_EQUAL_CSTR( ad->get_mdl_module_name(), "::anno");
        MI_CHECK_EQUAL_CSTR( ad->get_mdl_simple_name(), "description");
        MI_CHECK_EQUAL_CSTR( ad->get_mdl_parameter_type_name( 0), "string");
        MI_CHECK_EQUAL( ad->get_parameter_count(), 1);

        ad = anno->get_definition();
        MI_CHECK( ad);
        MI_CHECK_EQUAL_CSTR( ad->get_name(), "::anno::description(string)");
        MI_CHECK_EQUAL_CSTR( ad->get_mdl_module_name(), "::anno");
        MI_CHECK_EQUAL_CSTR( ad->get_mdl_simple_name(), "description");
        MI_CHECK_EQUAL_CSTR( ad->get_mdl_parameter_type_name( 0), "string");
        MI_CHECK_EQUAL( ad->get_parameter_count(), 1);

        block = type0->get_value_annotations( 1);
        MI_CHECK( !block);
    }
    {
        // check annotation for struct types

        mi::base::Handle<const mi::neuraylib::IModule> module(
            transaction->access<mi::neuraylib::IModule>(
                "mdl::" TEST_MDL));
        mi::base::Handle<const mi::neuraylib::IType_list> types( module->get_types());
        mi::base::Handle<const mi::neuraylib::IType_struct> type1(
            types->get_type<mi::neuraylib::IType_struct>( 1));
        MI_CHECK( type1);
        MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::foo_struct", types->get_name( 1));

        block = type1->get_annotations();
        anno = block->get_annotation( zero_size);
        MI_CHECK_EQUAL_CSTR( "::anno::description(string)", anno->get_name());
        args = anno->get_arguments();
        MI_CHECK_EQUAL( 1, args->get_size());
        arg = args->get_expression<mi::neuraylib::IExpression_constant>( "description");
        value_string = arg->get_value<mi::neuraylib::IValue_string>();
        MI_CHECK_EQUAL_CSTR( "struct annotation", value_string->get_value());

        ad = anno->get_definition();
        MI_CHECK( ad);
        MI_CHECK_EQUAL_CSTR( ad->get_name(), "::anno::description(string)");
        MI_CHECK_EQUAL_CSTR( ad->get_mdl_module_name(), "::anno");
        MI_CHECK_EQUAL_CSTR( ad->get_mdl_simple_name(), "description");
        MI_CHECK_EQUAL_CSTR( ad->get_mdl_parameter_type_name( 0), "string");
        MI_CHECK_EQUAL( ad->get_parameter_count(), 1);

        block = type1->get_field_annotations( zero_size);
        anno = block->get_annotation( zero_size);
        MI_CHECK_EQUAL_CSTR( "::anno::description(string)", anno->get_name());
        args = anno->get_arguments();
        MI_CHECK_EQUAL( 1, args->get_size());
        arg = args->get_expression<mi::neuraylib::IExpression_constant>( "description");
        value_string = arg->get_value<mi::neuraylib::IValue_string>();
        MI_CHECK_EQUAL_CSTR( "param_int annotation", value_string->get_value());

        ad = anno->get_definition();
        MI_CHECK( ad);
        MI_CHECK_EQUAL_CSTR( ad->get_name(), "::anno::description(string)");
        MI_CHECK_EQUAL_CSTR( ad->get_mdl_module_name(), "::anno");
        MI_CHECK_EQUAL_CSTR( ad->get_mdl_simple_name(), "description");
        MI_CHECK_EQUAL_CSTR( ad->get_mdl_parameter_type_name( 0), "string");
        MI_CHECK_EQUAL( ad->get_parameter_count(), 1);

        block = type1->get_field_annotations( 1);
        MI_CHECK( !block);
    }
    {
        // check annotation for struct categories

        mi::base::Handle<const mi::neuraylib::IModule> module(
            transaction->access<mi::neuraylib::IModule>( "mdl::" TEST_MDL));
        mi::base::Handle<const mi::neuraylib::IStruct_category_list> struct_categories(
            module->get_struct_categories());
        MI_CHECK_EQUAL( 1, struct_categories->get_size());

        mi::base::Handle<const mi::neuraylib::IStruct_category> sc0(
            struct_categories->get_struct_category( zero_size));
        MI_CHECK( sc0);
        MI_CHECK_EQUAL_CSTR(
            "::" TEST_MDL "::bar_struct_category", struct_categories->get_name( 0));

        block = sc0->get_annotations();
        anno = block->get_annotation( zero_size);
        MI_CHECK_EQUAL_CSTR( "::anno::description(string)", anno->get_name());
        args = anno->get_arguments();
        MI_CHECK_EQUAL( 1, args->get_size());
        arg = args->get_expression<mi::neuraylib::IExpression_constant>( "description");
        value_string = arg->get_value<mi::neuraylib::IValue_string>();
        MI_CHECK_EQUAL_CSTR( "struct category annotation", value_string->get_value());

        ad = anno->get_definition();
        MI_CHECK( ad);
        MI_CHECK_EQUAL_CSTR( ad->get_name(), "::anno::description(string)");
        MI_CHECK_EQUAL_CSTR( ad->get_mdl_module_name(), "::anno");
        MI_CHECK_EQUAL_CSTR( ad->get_mdl_simple_name(), "description");
        MI_CHECK_EQUAL_CSTR( ad->get_mdl_parameter_type_name( 0), "string");
        MI_CHECK_EQUAL( ad->get_parameter_count(), 1);

        anno = block->get_annotation( 1);
        MI_CHECK( !anno);
    }
    {
        // check annotation creation

        // prepare correct arguments
        mi::base::Handle<mi::neuraylib::IValue_string> value_name(
            vf->create_string( "The Name"));
        mi::base::Handle<mi::neuraylib::IExpression_constant> expr_name(
            ef->create_constant( value_name.get()));
        MI_CHECK( expr_name);
        mi::base::Handle<mi::neuraylib::IExpression_list> arguments(
            ef->create_expression_list());
        arguments->add_expression( "name", expr_name.get());

        // create with correct arguments
        mi::base::Handle<const mi::neuraylib::IModule> module(
            transaction->access<mi::neuraylib::IModule>( "mdl::anno"));
        mi::base::Handle<const mi::neuraylib::IAnnotation_definition> ad(
            module->get_annotation_definition( "::anno::display_name(string)"));
        mi::base::Handle<const mi::neuraylib::IAnnotation> anno(
            ad->create_annotation( arguments.get()));
        MI_CHECK( anno);

        ad = anno->get_definition();
        MI_CHECK( ad);

        // create with invalid number of arguments
        arguments->add_expression( "does_not_exist", expr_name.get());
        anno = ad->create_annotation( arguments.get());
        MI_CHECK( !anno);

        // prepare arguments with wrong type
        mi::base::Handle<mi::neuraylib::IValue_bool> value_name2( vf->create_bool( false));
        mi::base::Handle<mi::neuraylib::IExpression_constant> expr_name2(
            ef->create_constant( value_name2.get()));
        MI_CHECK( expr_name2);
        arguments = ef->create_expression_list();
        arguments->add_expression( "name", expr_name2.get());

        // create with wrong type
        anno = ad->create_annotation( arguments.get());
        MI_CHECK( !anno);

        // create without arguments
        anno = ad->create_annotation( nullptr);
        MI_CHECK( !anno);
    }
}

void check_parameter_references(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory)
{
    // Check default initializers that reference an earlier parameter.

    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    mi::base::Handle<const mi::neuraylib::IFunction_definition> c_fd;
    mi::base::Handle<const mi::neuraylib::IFunction_definition> c_md;
    mi::base::Handle<const mi::neuraylib::IFunction_call> c_fc;
    mi::base::Handle<mi::neuraylib::IFunction_call> m_fc;
    mi::base::Handle<const mi::neuraylib::IFunction_call> c_mi;
    mi::base::Handle<mi::neuraylib::IFunction_call> m_mi;

    mi::base::Handle<const mi::neuraylib::IType> type;
    mi::base::Handle<const mi::neuraylib::IType_list> types;
    mi::base::Handle<const mi::neuraylib::IValue_int> c_value_int;
    mi::base::Handle<const mi::neuraylib::IValue_float> c_value_float;
    mi::base::Handle<const mi::neuraylib::IValue_color> c_value_color;
    mi::base::Handle<const mi::neuraylib::IExpression> c_expr;
    mi::base::Handle<const mi::neuraylib::IExpression_constant> c_expr_constant;
    mi::base::Handle<const mi::neuraylib::IExpression_call> c_expr_call;
    mi::base::Handle<const mi::neuraylib::IExpression_parameter> c_expr_parameter;
    mi::base::Handle<const mi::neuraylib::IExpression_list> defaults, c_args;
    mi::base::Handle<mi::neuraylib::IExpression_list> args;

    // check parameter references of a function definition

    c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::" TEST_MDL "::fd_2_index(int,float)");
    MI_CHECK( c_fd);

    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL, c_fd->get_module());
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::fd_2_index(int,float)", c_fd->get_mdl_name());
    MI_CHECK_EQUAL_CSTR( "fd_2_index", c_fd->get_mdl_simple_name());

    types = c_fd->get_parameter_types();
    type = types->get_type( "param0");
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, type->get_kind());
    type = types->get_type( "param1");
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_FLOAT, type->get_kind());
    type = c_fd->get_return_type();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, type->get_kind());

    defaults = c_fd->get_defaults();
    c_expr = defaults->get_expression( "param0");
    MI_CHECK( !c_expr);

    c_expr_call = defaults->get_expression<mi::neuraylib::IExpression_call>( "param1");
    MI_CHECK( c_expr_call);
    type = c_expr_call->get_type();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_FLOAT, type->get_kind());
    const char* name = c_expr_call->get_call();
    c_fc = transaction->access<mi::neuraylib::IFunction_call>( name);
    MI_CHECK( c_fc);
    const char* definition_name = c_fc->get_mdl_function_definition();
    MI_CHECK_EQUAL_CSTR( definition_name, "float(int)");
    c_args = c_fc->get_arguments();
    c_expr_parameter = c_args->get_expression<mi::neuraylib::IExpression_parameter>( zero_size);
    MI_CHECK( c_expr_parameter);
    type = c_expr_parameter->get_type();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, type->get_kind());
    MI_CHECK_EQUAL( c_expr_parameter->get_index(), 0);

    // create a corresponding function call, supplying an argument only for "param0"
    {
        args = ef->create_expression_list();
        mi::base::Handle<mi::neuraylib::IValue_int> param0_value( vf->create_int( -42));
        mi::base::Handle<mi::neuraylib::IExpression> param0_expr(
            ef->create_constant( param0_value.get()));
        args->add_expression( "param0", param0_expr.get());
        m_fc = c_fd->create_function_call( args.get(), &result);
        MI_CHECK_EQUAL( 0, result);

        // check value of "param1"
        c_args = m_fc->get_arguments();
        c_expr_call = c_args->get_expression<mi::neuraylib::IExpression_call>( "param1");
        MI_CHECK( c_expr_call);
        type = c_expr_call->get_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_FLOAT, type->get_kind());
        const char* name = c_expr_call->get_call();
        c_fc = transaction->access<mi::neuraylib::IFunction_call>( name);
        MI_CHECK( c_fc);
        const char* definition_name = c_fc->get_mdl_function_definition();
        MI_CHECK_EQUAL_CSTR( definition_name, "float(int)");
        c_args = c_fc->get_arguments();
        c_expr_constant = c_args->get_expression<mi::neuraylib::IExpression_constant>( zero_size);
        MI_CHECK( c_expr_constant);
        c_value_int = c_expr_constant->get_value<mi::neuraylib::IValue_int>();
        MI_CHECK( c_value_int);
        MI_CHECK_EQUAL( -42, c_value_int->get_value());
        c_fc.reset();
        m_fc.reset();
    }

    // create a corresponding function call, supplying arguments for both parameters
    {
        args = ef->create_expression_list();
        mi::base::Handle<mi::neuraylib::IValue_int> param0_value( vf->create_int( -42));
        mi::base::Handle<mi::neuraylib::IExpression> param0_expr(
            ef->create_constant( param0_value.get()));
        args->add_expression( "param0", param0_expr.get());
        mi::base::Handle<mi::neuraylib::IValue_float> param1_value( vf->create_float( -43.0f));
        mi::base::Handle<mi::neuraylib::IExpression> param1_expr(
            ef->create_constant( param1_value.get()));
        args->add_expression( "param1", param1_expr.get());
        m_fc = c_fd->create_function_call( args.get(), &result);
        MI_CHECK_EQUAL( 0, result);

        // check value of "param0"
        c_args = m_fc->get_arguments();
        c_expr_constant = args->get_expression<mi::neuraylib::IExpression_constant>( "param0");
        MI_CHECK( c_expr_constant);
        c_value_int = c_expr_constant->get_value<mi::neuraylib::IValue_int>();
        MI_CHECK( c_value_int);
        MI_CHECK_EQUAL( -42, c_value_int->get_value());

        // check value of "param1"
        c_expr_constant = args->get_expression<mi::neuraylib::IExpression_constant>( "param1");
        MI_CHECK( c_expr_constant);
        c_value_float = c_expr_constant->get_value<mi::neuraylib::IValue_float>();
        MI_CHECK( c_value_float);
        MI_CHECK_EQUAL( -43.0f, c_value_float->get_value());
        m_fc.reset();
    }

    // check parameter reference via array constructor
    {
        c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
            "mdl::" TEST_MDL "::fd_2_index_nested(int,int[N])");
        MI_CHECK( c_fd);

        MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL, c_fd->get_module());
        MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::fd_2_index_nested(int,int[N])", c_fd->get_mdl_name());
        MI_CHECK_EQUAL_CSTR( "fd_2_index_nested", c_fd->get_mdl_simple_name());
        MI_CHECK_EQUAL( 2, c_fd->get_parameter_count());
        types = c_fd->get_parameter_types();

        type = types->get_type( "param0");
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, type->get_kind());
        type = types->get_type( "param1");
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_ARRAY, type->get_kind());
        type = c_fd->get_return_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, type->get_kind());

        defaults = c_fd->get_defaults();
        c_expr = defaults->get_expression( "param0");
        MI_CHECK( !c_expr);

        c_expr_call = defaults->get_expression<mi::neuraylib::IExpression_call>( "param1");
        MI_CHECK( c_expr_call);
        type = c_expr_call->get_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_ARRAY, type->get_kind());
        name = c_expr_call->get_call();
        c_fc = transaction->access<mi::neuraylib::IFunction_call>( name);
        MI_CHECK( c_fc);
        definition_name = c_fc->get_mdl_function_definition();
        MI_CHECK_EQUAL_CSTR( definition_name, "T[](...)");
        c_args = c_fc->get_arguments();
        c_expr_parameter = c_args->get_expression<mi::neuraylib::IExpression_parameter>( zero_size);
        MI_CHECK( c_expr_parameter);
        type = c_expr_parameter->get_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, type->get_kind());
        MI_CHECK_EQUAL( c_expr_parameter->get_index(), 0);
    }
}

void check_reexported_function(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    // prepare argument list with "normal" being a call expression referencing
    // mdl::" TEST_MDL "::fc_normal
    mi::base::Handle<mi::neuraylib::IExpression_call> arg(
        ef->create_call( "mdl::" TEST_MDL "::fc_normal"));
    mi::base::Handle<mi::neuraylib::IExpression_list> args(
        ef->create_expression_list());
    args->add_expression( "normal", arg.get());

    // instantiate mdl::" TEST_MDL "::md_reexport(float3) with a call to
    // mdl::" TEST_MDL "::fc_normal as argument for "normal"
    mi::base::Handle<const mi::neuraylib::IFunction_definition> c_md(
        transaction->access<mi::neuraylib::IFunction_definition>(
            "mdl::" TEST_MDL "::md_reexport(float3)"));
    MI_CHECK( c_md);
    mi::base::Handle<mi::neuraylib::IFunction_call> m_mi(
        c_md->create_function_call( args.get(), &result));
    MI_CHECK_EQUAL( 0, result);
    MI_CHECK_EQUAL( 0, transaction->store( m_mi.get(), "mdl::" TEST_MDL "::mi_reexport"));
}

void check_named_temporaries( mi::neuraylib::ITransaction* transaction)
{
    {
        mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::" TEST_MDL "::fd_named_temporaries(int)"));

        std::set<std::string> names;
        for( mi::Size i = 0, n = fd->get_temporary_count(); i < n; ++i) {
            const char* tn = fd->get_temporary_name( i);
            if( tn)
                names.insert( tn);
        }

        MI_CHECK( names.size() == 1);
        MI_CHECK( names.count( "a") == 1);
    }

    {
        mi::base::Handle<const mi::neuraylib::IFunction_definition> md(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::" TEST_MDL "::md_named_temporaries(float)"));

        std::set<std::string> names;
        for( mi::Size i = 0, n = md->get_temporary_count(); i < n; ++i) {
            const char* tn = md->get_temporary_name( i);
            if( tn)
                names.insert( tn);
        }

        MI_CHECK( names.size() == 11);
        MI_CHECK( names.count( "f0" ) == 1);
        MI_CHECK( names.count( "f1") == 1);
        MI_CHECK( names.count( "f2") == 1);
        MI_CHECK( names.count( "b") == 1);
        MI_CHECK( names.count( "c0") == 1);
        MI_CHECK( names.count( "c1") == 1);
        MI_CHECK( names.count( "s" ) == 1);
        MI_CHECK( names.count( "v0") == 1);
        MI_CHECK( names.count( "f3") == 1);
        MI_CHECK( names.count( "v1") == 1);
        MI_CHECK( names.count( "v2") == 1);

        // From test_mdl.mdl:
        // Keep the order of v0 and f3 to test that CSE does not identify the named expression f3
        // for param0 with a previous unnamed expression (as part of v0) for param0.
        // float3 v0 = float3(param0, 0.f, 0.f);
        // float f3 = param0;

        bool found_v0 = false;

        for( mi::Size i = 0, n = md->get_temporary_count(); i < n; ++i) {

            const char* tn = md->get_temporary_name( i);
            if( tn && strcmp( tn, "v0") == 0) {

                found_v0 = true;
                mi::base::Handle<const mi::neuraylib::IExpression_direct_call> v0(
                    md->get_temporary<mi::neuraylib::IExpression_direct_call>( i));
                mi::base::Handle<const mi::neuraylib::IExpression_list> args( v0->get_arguments());
                mi::base::Handle<const mi::neuraylib::IExpression> arg0(
                    args->get_expression( zero_size));
                // Not being a temporary implies not having a name. If this starts to fail, change
                // the test to retrieve the index, and check the temporary of that index does not
                // have a name. But temporaries for parameter references are unlikely.
                MI_CHECK_NOT_EQUAL( arg0->get_kind(), mi::neuraylib::IExpression::EK_TEMPORARY);
                MI_CHECK_EQUAL( arg0->get_kind(), mi::neuraylib::IExpression::EK_PARAMETER);
            }
        }

        MI_CHECK( found_v0);
    }
}


void load_secondary_modules_from_string(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::INeuray* neuray)
{
    mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
        neuray->get_api_component<mi::neuraylib::IMdl_factory>());
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
        neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());

    mi::Size always_imported_modules;
    {
        // import an MDL module from string
        const char* data = "mdl 1.0; export material some_material() = material();";
        MI_CHECK_EQUAL( 0,
            mdl_impexp_api->load_module_from_string( transaction, "::from_string", data));

        // check that the module exists now and has exactly one module definition
        mi::base::Handle<const mi::neuraylib::IModule> module;
        module = transaction->access<mi::neuraylib::IModule>( "mdl::from_string");
        MI_CHECK_EQUAL( 1, module->get_material_count());
        always_imported_modules = module->get_import_count();
    }
    {
        // import an MDL module (from strings) that imports an MDL module that was imported from
        // string

        // ::imports_from_string requires the module cache
        uninstall_external_resolver( neuray);

        const char* data = "mdl 1.0; import from_string::*;";
        MI_CHECK_EQUAL( 0,
            mdl_impexp_api->load_module_from_string( transaction, "::imports_from_string", data));

        // check that the module exists now and has exactly one imported module
        mi::base::Handle<const mi::neuraylib::IModule> module;
        module = transaction->access<mi::neuraylib::IModule>( "mdl::imports_from_string");
        MI_CHECK_EQUAL( always_imported_modules+1, module->get_import_count());

        install_external_resolver( neuray);
    }
    {
        // check that it is not possible to shadow modules found via the search path

        // check that the "mdl_elements::test_misc" module has not yet been loaded
        mi::base::Handle<const mi::neuraylib::IModule> module(
            transaction->access<mi::neuraylib::IModule>( "mdl::mdl_elements::test_misc"));
        MI_CHECK( !module);

        // check that creating a "mdl_elements::test_misc" module via import_elements_from_string()
        // fails
        const char* data = "mdl 1.0;";
        MI_CHECK_NOT_EQUAL( 0,
            mdl_impexp_api->load_module_from_string(
                transaction, "::mdl_elements::test_misc", data));

        // check that the "mdl_elements::test_misc" module still has not yet been loaded
        module = transaction->access<mi::neuraylib::IModule>( "mdl::mdl_elements::test_misc");
        MI_CHECK( !module);

        // check that it could be loaded, i.e., it is actually found in the search path
        MI_CHECK_EQUAL( 0, mdl_impexp_api->load_module( transaction, "::mdl_elements::test_misc"));

        // check that the "mdl_elements::test_misc" module has been loaded
        module = transaction->access<mi::neuraylib::IModule>( "mdl::mdl_elements::test_misc");
        MI_CHECK( module);
    }
    {
        // check that error messages from the MDL compiler are reported

        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
            mdl_factory->create_execution_context());

        const char* data = "mdl 0.42;";
        result = mdl_impexp_api->load_module_from_string(
            transaction, "::mdl_elements::test_messages", data, context.get());
        MI_CHECK_EQUAL( -2, result);
        MI_CHECK_EQUAL( 1, context.get()->get_error_messages_count());

        mi::base::Handle<const mi::neuraylib::IMessage> msg( context->get_error_message( 0));
        MI_CHECK_EQUAL( mi::neuraylib::IMessage::MSG_COMILER_CORE, msg->get_kind());
        MI_CHECK_EQUAL( mi::base::MESSAGE_SEVERITY_ERROR, msg->get_severity());
    }
}


void check_imodule_part2( mi::neuraylib::ITransaction* transaction, mi::neuraylib::INeuray* neuray)
{
    mi::base::Handle<mi::neuraylib::IFactory> factory(
        neuray->get_api_component<mi::neuraylib::IFactory>());
    mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
        neuray->get_api_component<mi::neuraylib::IMdl_factory>());
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
        neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());

    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    // check module in archive

    mi::base::Handle<const mi::neuraylib::IModule> a_module(
        transaction->access<mi::neuraylib::IModule>( "mdl::test_archives"));
    MI_CHECK( a_module);

    std::string filename = MI::TEST::mi_src_path( "prod/lib/neuray/") + "test_archives.mdr";
    MI_CHECK_EQUAL( filename, a_module->get_filename());
    MI_CHECK_EQUAL_CSTR( "::test_archives", a_module->get_mdl_name());
    MI_CHECK_EQUAL_CSTR( "test_archives", a_module->get_mdl_simple_name());
    MI_CHECK_EQUAL( 0, a_module->get_mdl_package_component_count());
    MI_CHECK( !a_module->is_standard_module());
    MI_CHECK( !a_module->is_mdle_module());

    // check MDLE module

    mi::base::Handle<const mi::IString> s( mdl_factory->get_db_module_name( mdle_path.c_str()));
    std::string mdle_module_db_name = s->get_c_str();

    mi::base::Handle<const mi::neuraylib::IModule> m_module(
        transaction->access<mi::neuraylib::IModule>( mdle_module_db_name.c_str()));
    MI_CHECK_EQUAL( 0, m_module->get_mdl_package_component_count());
    MI_CHECK( !m_module->is_standard_module());
    MI_CHECK( m_module->is_mdle_module());
    MI_CHECK_EQUAL_CSTR( m_module->get_mdl_name()+2, m_module->get_mdl_simple_name());

    {
        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
            mdl_factory->create_execution_context());
        std::string fd_name
            = mdle_module_db_name + "::main(texture_2d,light_profile,bsdf_measurement)";

        Mdle_serialization_callback serialization_cb( factory.get());
        mi::base::Handle<const mi::neuraylib::ISerialized_function_name> sfn(
            mdl_impexp_api->serialize_function_name(
                fd_name.c_str(),
                /*argument_types*/ nullptr,
                /*return_type*/ nullptr,
                &serialization_cb,
                context.get()));
        MI_CHECK_CTX( context);

        // "::FOO" should occur once for the module name
        std::string s = sfn->get_function_name();
        size_t first_foo  = s.find( "::FOO");
        MI_CHECK_NOT_EQUAL( first_foo, std::string::npos);
        size_t left_paren = s.find( "(", first_foo);
        MI_CHECK_NOT_EQUAL( left_paren, std::string::npos);

        Mdle_deserialization_callback deserialization_cb( factory.get());
        mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn(
            mdl_impexp_api->deserialize_function_name(
                transaction, sfn->get_function_name(), &deserialization_cb, context.get()));

        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL_CSTR( dfn->get_db_name(), fd_name.c_str());

        mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn2(
            mdl_impexp_api->deserialize_function_name(
                transaction,
                sfn->get_module_name(),
                sfn->get_function_name_without_module_name(),
                &deserialization_cb,
                context.get()));
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL_CSTR( dfn2->get_db_name(), fd_name.c_str());

        mi::base::Handle<const mi::neuraylib::IDeserialized_module_name> dmn(
            mdl_impexp_api->deserialize_module_name(
                sfn->get_module_name(), &deserialization_cb, context.get()));
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL_CSTR( dmn->get_db_name(), mdle_module_db_name.c_str());
        result = mdl_impexp_api->load_module(
            transaction, dmn->get_load_module_argument(), context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( result, 1);
    }
}

void check_mangled_names( mi::neuraylib::ITransaction* transaction)
{
    {
        mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::test_compatibility::structure_test_func(::test_compatibility::struct_one)"));
        MI_CHECK( fd);
        MI_CHECK_EQUAL_CSTR(
            fd->get_mangled_name(),
            "_ZN18test_compatibility19structure_test_funcEN18test_compatibility10struct_oneE");
    }
    {
        mi::base::Handle<const mi::neuraylib::IFunction_definition> md(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::" TEST_MDL "::md_0()"));
        MI_CHECK( md);
        MI_CHECK_EQUAL_CSTR( md->get_mangled_name(), "md_0");
    }
}

void check_implicit_casts(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::INeuray* neuray)
{
    mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
        neuray->get_api_component<mi::neuraylib::IMdl_factory>());

    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    // create struct constant struct_one_expr of type ...::struct_one
    mi::base::Handle<const mi::neuraylib::IType_struct> struct_one_type(
        tf->create_struct( "::test_compatibility::struct_one"));
    mi::base::Handle<mi::neuraylib::IValue_struct> struct_one_value(
        vf->create_struct( struct_one_type.get()));
    mi::base::Handle<const mi::neuraylib::IExpression_constant> struct_one_expr(
        ef->create_constant( struct_one_value.get()));

    // create struct constant struct_two_expr of type ...::struct_two
    mi::base::Handle<const mi::neuraylib::IType_struct> struct_two_type(
        tf->create_struct( "::test_compatibility::struct_two"));
    mi::base::Handle<mi::neuraylib::IValue_struct> struct_two_value(
        vf->create_struct( struct_two_type.get()));
    mi::base::Handle<mi::neuraylib::IExpression_constant> struct_two_expr(
        ef->create_constant( struct_two_value.get()));

    // create function call with defaults of ...::structure_test(...)
    mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
        transaction->access<mi::neuraylib::IFunction_definition>(
            "mdl::test_compatibility::structure_test(::test_compatibility::struct_one)"));
    MI_CHECK( fd);
    mi::base::Handle<mi::neuraylib::IFunction_call> fc( fd->create_function_call( nullptr));
    MI_CHECK( fc);

    // set argument to expression of correct type
    MI_CHECK_EQUAL( fc->set_argument( "v", struct_two_expr.get()), 0);

    {
        // create call of cast operator casting from struct_two_expr to struct_one type.
        mi::base::Handle<const mi::neuraylib::IFunction_definition> cast_def(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::operator_cast(%3C0%3E)"));
        MI_CHECK( cast_def);
        mi::base::Handle<mi::neuraylib::IExpression_list> args(
            ef->create_expression_list());
        MI_CHECK_EQUAL( args->add_expression( "cast", struct_two_expr.get()), 0);
        MI_CHECK_EQUAL( args->add_expression( "cast_return", struct_one_expr.get()), 0);

        mi::base::Handle<mi::neuraylib::IFunction_call> cast_inst(
            cast_def->create_function_call( args.get(), nullptr));
        MI_CHECK( cast_inst);
        MI_CHECK_EQUAL( transaction->store( cast_inst.get(), "my_cast"), 0);

        mi::base::Handle<const mi::neuraylib::IExpression> cast_call( ef->create_call( "my_cast"));
        MI_CHECK( cast_call);

        // set argument with explicit cast expression of correct type
        MI_CHECK_EQUAL( fc->set_argument( "v", cast_call.get()), 0);
    }

    // function call creation
    {
        // ... with argument that requires a cast
        mi::base::Handle<mi::neuraylib::IExpression_list> args( ef->create_expression_list());
        args->add_expression( "v", struct_two_expr.get());
        mi::Sint32 errors = 0;
        mi::base::Handle<mi::neuraylib::IFunction_call> call_inst(
            fd->create_function_call( args.get(), &errors));
        MI_CHECK( call_inst);
        MI_CHECK_EQUAL( errors, 0);

        mi::base::Handle<const mi::neuraylib::IExpression_list> call_args(
            call_inst->get_arguments());
        mi::base::Handle<const mi::neuraylib::IExpression_call> arg0(
            call_args->get_expression<mi::neuraylib::IExpression_call>( "v"));
        MI_CHECK( arg0);
        MI_CHECK_EQUAL( arg0->get_kind(), mi::neuraylib::IExpression::EK_CALL);
    }

    // function call creation, parameter type with modifier
    {
        // ... with argument that requires a cast
        mi::base::Handle<const mi::neuraylib::IFunction_definition> def_modifier(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::test_compatibility::structure_test_modifier("
                    "::test_compatibility::struct_one)"));
        MI_CHECK( def_modifier);

        mi::base::Handle<mi::neuraylib::IExpression_list> args( ef->create_expression_list());
        args->add_expression( "v", struct_two_expr.get());
        mi::Sint32 errors = 0;
        mi::base::Handle<mi::neuraylib::IFunction_call> call_inst(
            def_modifier->create_function_call( args.get(), &errors));
        MI_CHECK( call_inst);
        MI_CHECK_EQUAL( errors, 0);

        mi::base::Handle<const mi::neuraylib::IExpression_list> call_args(
            call_inst->get_arguments());
        mi::base::Handle<const mi::neuraylib::IExpression_call> arg0(
            call_args->get_expression<mi::neuraylib::IExpression_call>( "v"));
        MI_CHECK( arg0);
        MI_CHECK_EQUAL( arg0->get_kind(), mi::neuraylib::IExpression::EK_CALL);
    }

    // function call creation, argument type with modifier
    {
        // ... with argument that requires cast
        mi::base::Handle<const mi::neuraylib::IFunction_definition> def_no_modifier(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::test_compatibility::structure_test_func(::test_compatibility::struct_one)"));
        MI_CHECK( def_no_modifier);

        do_create_function_call(
            transaction,
            "mdl::test_compatibility::struct_return_modifier(bool)",
            "fc_struct_return_modifier");
        mi::base::Handle<mi::neuraylib::IExpression_call> call(
            ef->create_call( "fc_struct_return_modifier"));

        mi::base::Handle<mi::neuraylib::IExpression_list> args( ef->create_expression_list());
        args->add_expression( "v", call.get());
        mi::Sint32 errors = 0;
        mi::base::Handle<mi::neuraylib::IFunction_call> call_inst(
            def_no_modifier->create_function_call( args.get(), &errors));
        MI_CHECK( call_inst);
        MI_CHECK_EQUAL( errors, 0);
    }
    {
        // ... with argument that does not require cast
        mi::base::Handle<mi::neuraylib::IExpression_list> args( ef->create_expression_list());
        args->add_expression( "v", struct_one_expr.get());
        mi::Sint32 errors = 0;
        mi::base::Handle<mi::neuraylib::IFunction_call> call_inst(
            fd->create_function_call( args.get(), &errors));
        MI_CHECK( call_inst);
        MI_CHECK_EQUAL( errors, 0);

        mi::base::Handle<const mi::neuraylib::IExpression_list> call_args(
            call_inst->get_arguments());
        mi::base::Handle<const mi::neuraylib::IExpression_constant> arg0(
            call_args->get_expression<mi::neuraylib::IExpression_constant>( "v"));
        MI_CHECK( arg0);
        MI_CHECK_EQUAL( arg0->get_kind(), mi::neuraylib::IExpression::EK_CONSTANT);
    }
    {
        // ... with an incompatible argument
        mi::base::Handle<mi::neuraylib::IValue_int> int_value( vf->create_int( 42));
        mi::base::Handle<mi::neuraylib::IExpression_constant> int_expr(
            ef->create_constant( int_value.get()));

        mi::base::Handle<mi::neuraylib::IExpression_list> args( ef->create_expression_list());
        args->add_expression( "v", int_expr.get());
        mi::Sint32 errors = 0;
        mi::base::Handle<mi::neuraylib::IFunction_call> call_inst(
            fd->create_function_call( args.get(), &errors));
        MI_CHECK( !call_inst);
        MI_CHECK_EQUAL( errors, -2);
    }

    // test array constructor creation
    {
        mi::base::Handle<const mi::neuraylib::IFunction_definition> array_def(
            transaction->access<mi::neuraylib::IFunction_definition>( "mdl::T[](...)"));
        MI_CHECK( array_def);

        mi::base::Handle<mi::neuraylib::IExpression_list> args( ef->create_expression_list());
        args->add_expression( "value0", struct_one_expr.get());
        args->add_expression( "value1", struct_two_expr.get());
        args->add_expression( "value2", struct_one_expr.get());
        args->add_expression( "value3", struct_two_expr.get());

        mi::Sint32 errors = 0;
        mi::base::Handle<mi::neuraylib::IFunction_call> call_inst(
            array_def->create_function_call( args.get(), &errors));
        MI_CHECK( call_inst);
        MI_CHECK_EQUAL( errors, 0);

        mi::base::Handle<const mi::neuraylib::IExpression_list> call_args(
            call_inst->get_arguments());
        for( mi::Size i = 0; i < 4; ++i) {
            mi::base::Handle<const mi::neuraylib::IExpression> arg(
                call_args->get_expression<mi::neuraylib::IExpression>( i));
            MI_CHECK( arg);
            // The first argument determines the type.
            auto kind = i%2 == 0
                ? mi::neuraylib::IExpression::EK_CONSTANT : mi::neuraylib::IExpression::EK_CALL;
            MI_CHECK_EQUAL( arg->get_kind(), kind);
        }
    }
}

void check_mdl_reload( mi::neuraylib::ITransaction* transaction, mi::neuraylib::INeuray* neuray)
{
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
        neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());
    mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
        neuray->get_api_component<mi::neuraylib::IMdl_factory>());

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    {
        // create material instances and function calls
        const char* materials[] = {
            "mdl::test_mdl_reload::md_1(color)", "mdl::test_mdl_reload::mi_1",
            "mdl::test_mdl_reload::md_2(float)", "mdl::test_mdl_reload::mi_2",
            "mdl::test_mdl_reload::md_2(float)", "mdl::test_mdl_reload::mi_2_2",
            "mdl::test_mdl_reload::md_3()",      "mdl::test_mdl_reload::mi_3",
            "mdl::test_mdl_reload::md_4(float)", "mdl::test_mdl_reload::mi_4",
            "mdl::test_mdl_reload::fd_1(float)", "mdl::test_mdl_reload::fd_1_inst"
        };

        for( mi::Size i = 0; i < sizeof( materials) / sizeof( const char*); i += 2)
            do_create_function_call( transaction, materials[i], materials[i+1]);

        mi::neuraylib::Argument_editor ae(
            transaction, "mdl::test_mdl_reload::mi_2_2", mdl_factory.get());
        MI_CHECK( ae.is_valid());
        MI_CHECK_EQUAL( ae.set_call( "f", "mdl::test_mdl_reload::fd_1_inst"), 0);
    }
    {
        // modify file on disk and reload
        fs::copy_file(
            fs::u8path( MI::TEST::mi_src_path( "prod/lib/neuray") + "/test_mdl_reload_1.mdl"),
            fs::u8path( DIR_PREFIX "/test_mdl_reload.mdl"),
            fs::copy_options::overwrite_existing);
        mi::base::Handle<mi::neuraylib::IModule> module2(
            transaction->edit<mi::neuraylib::IModule>( "mdl::test_mdl_reload"));
        result = module2->reload( /*recursive*/ false, context.get());
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK_EQUAL( 3, module2->get_material_count());
    }
    {
        // access some definitions
        mi::base::Handle<const mi::neuraylib::IFunction_definition> md1(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::test_mdl_reload::md_1(color,color)"));
        mi::base::Handle<const mi::neuraylib::IFunction_definition> md2(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::test_mdl_reload::md_2(float)"));
        mi::base::Handle<const mi::neuraylib::IFunction_definition> md3(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::test_mdl_reload::md_3()"));
        mi::base::Handle<const mi::neuraylib::IFunction_definition> fd1(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::test_mdl_reload::fd_1(float)"));

        // check old definitions
        MI_CHECK( md1->is_valid( context.get()));
        MI_CHECK( md2->is_valid( context.get()));
        MI_CHECK( !md3->is_valid( context.get()));
        MI_CHECK_EQUAL( nullptr, md3->create_function_call( nullptr, &result));
        MI_CHECK_EQUAL( -9, result);
        MI_CHECK( fd1->is_valid( context.get()));

        // check old instances
        mi::base::Handle<const mi::neuraylib::IFunction_call> mi1(
            transaction->access<mi::neuraylib::IFunction_call>( "mdl::test_mdl_reload::mi_1"));
        MI_CHECK( !mi1->is_valid( context.get()));

        // ensure we cannot create a compiled material from this invalid instance.
        mi::base::Handle<const mi::neuraylib::IMaterial_instance> mi1_mi(
            mi1->get_interface<mi::neuraylib::IMaterial_instance>());
        mi::base::Handle<const mi::neuraylib::ICompiled_material> cm(
            mi1_mi->create_compiled_material(
                mi::neuraylib::IMaterial_instance::CLASS_COMPILATION, nullptr));
        MI_CHECK( !cm);

        // This instance is still valid.
        mi::base::Handle<const mi::neuraylib::IFunction_call> mi2(
            transaction->access<mi::neuraylib::IFunction_call>( "mdl::test_mdl_reload::mi_2"));
        MI_CHECK( mi2->is_valid( context.get()));

        mi::base::Handle<const mi::neuraylib::IFunction_call> mi3(
            transaction->access<mi::neuraylib::IFunction_call>( "mdl::test_mdl_reload::mi_3"));
        MI_CHECK( !mi3->is_valid( context.get()));

        // Interface has changed.
        mi::base::Handle<const mi::neuraylib::IFunction_call> fc1(
            transaction->access<mi::neuraylib::IFunction_call>( "mdl::test_mdl_reload::fd_1_inst"));
        MI_CHECK( !fc1->is_valid( context.get()));

        // This instance has an invalid call argument.
        mi::base::Handle<mi::neuraylib::IFunction_call> mi2a(
            transaction->edit<mi::neuraylib::IFunction_call>( "mdl::test_mdl_reload::mi_2_2"));
        MI_CHECK( !mi2a->is_valid( context.get()));

        // Impossible to repair since the return type of the attached function has changed.
        MI_CHECK_EQUAL( -1, mi2a->repair(
            mi::neuraylib::MDL_REPAIR_INVALID_ARGUMENTS, context.get()));

        // Repair by removing the invalid argument.
        MI_CHECK_EQUAL( 0, mi2a->repair(
            mi::neuraylib::MDL_REMOVE_INVALID_ARGUMENTS, context.get()));

        // The instance is valid again.
        MI_CHECK( mi2a->is_valid( context.get()));

        mi::base::Handle<mi::neuraylib::IFunction_call> mi4(
            transaction->edit<mi::neuraylib::IFunction_call>( "mdl::test_mdl_reload::mi_4"));
        MI_CHECK_EQUAL( 0, mi2a->repair(mi::neuraylib::MDL_REPAIR_DEFAULT, context.get()));
    }
    {
        // check module that imports mdl::test_mdl_reload
        mi::base::Handle<mi::neuraylib::IModule> module(
            transaction->edit<mi::neuraylib::IModule>( "mdl::test_mdl_reload_import"));

        // The module and its definitions should be invalid.
        MI_CHECK( !module->is_valid( context.get()));
        mi::base::Handle<const mi::neuraylib::IFunction_definition> md11(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::test_mdl_reload_import::md_1(color)"));
        MI_CHECK( !md11->is_valid( context.get()));
        mi::base::Handle<const mi::neuraylib::IFunction_definition> md21(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::test_mdl_reload_import::md_2(float)"));
        MI_CHECK( !md21->is_valid( context.get()));

        // reload
        MI_CHECK_EQUAL( 0, module->reload( /*recursive*/ true, context.get()));

        MI_CHECK( module->is_valid( context.get()));
        md11 = transaction->access<mi::neuraylib::IFunction_definition>(
            "mdl::test_mdl_reload_import::md_1(color)");
        MI_CHECK( md11->is_valid( context.get()));
        md11 = transaction->access<mi::neuraylib::IFunction_definition>(
            "mdl::test_mdl_reload_import::md_2(float)");
        MI_CHECK( md21->is_valid( context.get()));
    }
    {
        // modify file on disk again and try to reload
        fs::copy_file(
            fs::u8path( MI::TEST::mi_src_path( "prod/lib/neuray") + "/test_mdl_reload_2.mdl"),
            fs::u8path( DIR_PREFIX "/test_mdl_reload.mdl"),
            fs::copy_options::overwrite_existing);
        mi::base::Handle<mi::neuraylib::IModule> module(
            transaction->edit<mi::neuraylib::IModule>( "mdl::test_mdl_reload"));
        result = module->reload( /*recursive*/ false, context.get());
        MI_CHECK_EQUAL( -1, result);
        MI_CHECK_EQUAL( 1, context->get_error_messages_count());
    }
    {
        // check that reloading base.mdl is prohibited
        mi::base::Handle<mi::neuraylib::IModule> base_module(
            transaction->edit<mi::neuraylib::IModule>( "mdl::base"));
        if( base_module) {
            result = base_module->reload( /*recursive*/ false, context.get());
            MI_CHECK_EQUAL( -1, result);
            MI_CHECK_EQUAL( 1, context->get_error_messages_count());
        }
    }
    {
        // check that reloading distilling_support.mdl is prohibited
        mi::base::Handle<mi::neuraylib::IModule> ds_module(
            transaction->edit<mi::neuraylib::IModule>( "mdl::nvidia::distilling_support"));
        if( ds_module) {
            result = ds_module->reload( /*recursive*/ false, context.get());
            MI_CHECK_EQUAL( -1, result);
            MI_CHECK_EQUAL( 1, context->get_error_messages_count());
        }
    }

    // check reload from string
    {
        mi::base::Handle<mi::neuraylib::IModule> module(
            transaction->edit<mi::neuraylib::IModule>( "mdl::test_mdl_reload_from_string"));
        MI_CHECK( module);
        // null-source string should fail
        result = module->reload_from_string(
            /*module_source*/ nullptr, /*recursive*/ false, context.get());
        MI_CHECK_EQUAL( -1, result);
        MI_CHECK_EQUAL( 1,  context->get_error_messages_count());

        // empty-source string should fail, too.
        result = module->reload_from_string(
            /*module_source*/ "", /*recursive*/ false, context.get());
        MI_CHECK_EQUAL( -1, result);
        MI_CHECK_EQUAL( 1, context->get_error_messages_count());

        const char* module_source = "mdl 1.0; export float some_function() { return 1.0; }";
        result = module->reload_from_string( module_source, /*recursive*/ false, context.get());
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK_EQUAL( 0, context->get_error_messages_count());

        mi::Size material_count = module->get_material_count();
        MI_CHECK_EQUAL( 0, material_count);

        mi::Size function_count = module->get_function_count();
        MI_CHECK_EQUAL( 1, function_count);

        const char* filename = module->get_filename();
        MI_CHECK_EQUAL( nullptr, filename);
    }

    // check recursive reloads, test_mdl_reload_4/5.mdl import test_mdl_reload_3.mdl
    {
        {
            // create the three modules
            std::ofstream module_3( DIR_PREFIX "/test_mdl_reload_3.mdl");
            module_3 << "mdl 1.3;";
            std::ofstream module_4( DIR_PREFIX "/test_mdl_reload_4.mdl");
            module_4 << "mdl 1.3; \
                         import ::test_mdl_reload_3::*;";
            std::ofstream module_5( DIR_PREFIX "/test_mdl_reload_5.mdl");
            module_5 << "mdl 1.3; \
                         import ::test_mdl_reload_3::*;";
        }

        // and load them
        MI_CHECK_EQUAL( 0, mdl_impexp_api->load_module( transaction, "::test_mdl_reload_3"));
        MI_CHECK_EQUAL( 0, mdl_impexp_api->load_module( transaction, "::test_mdl_reload_4"));
        MI_CHECK_EQUAL( 0, mdl_impexp_api->load_module( transaction, "::test_mdl_reload_5"));

        mi::base::Handle<mi::neuraylib::IModule> module;
        mi::base::Handle<const mi::neuraylib::IModule> c_module;

        {
            // reload test_mdl_reload_4.mdl recursively, check that all modules remain valid
            module = transaction->edit<mi::neuraylib::IModule>( "mdl::test_mdl_reload_4");
            MI_CHECK_EQUAL( 0, module->reload( true, context.get()));
            MI_CHECK( module->is_valid( context.get()));
            c_module = transaction->access<mi::neuraylib::IModule>( "mdl::test_mdl_reload_3");
            MI_CHECK( c_module->is_valid( context.get()));
            c_module = transaction->access<mi::neuraylib::IModule>( "mdl::test_mdl_reload_5");
            MI_CHECK( c_module->is_valid( context.get()));
        }
        {
            // change test_mdl_reload_3.mdl in a compatible way
            std::ofstream module_3( DIR_PREFIX "/test_mdl_reload_3.mdl");
            module_3 << "mdl 1.3; \
                         export int f() { return 42; }";
        }
        {
            // reload test_mdl_reload_4.mdl recursively, check that all modules but
            // test_mdl_reload_5.mdl are valid
            module = transaction->edit<mi::neuraylib::IModule>( "mdl::test_mdl_reload_4");
            MI_CHECK_EQUAL( 0, module->reload( true, context.get()));
            MI_CHECK( module->is_valid( context.get()));
            c_module = transaction->access<mi::neuraylib::IModule>( "mdl::test_mdl_reload_3");
            MI_CHECK( c_module->is_valid( context.get()));
            c_module = transaction->access<mi::neuraylib::IModule>( "mdl::test_mdl_reload_5");
            MI_CHECK( !c_module->is_valid( context.get()));
        }
        {
            // reload also test_mdl_reload_5.mdl recursively, check that all modules are now valid
            module = transaction->edit<mi::neuraylib::IModule>( "mdl::test_mdl_reload_5");
            MI_CHECK_EQUAL( 0, module->reload( true, context.get()));
            MI_CHECK( module->is_valid( context.get()));
            c_module = transaction->access<mi::neuraylib::IModule>( "mdl::test_mdl_reload_3");
            MI_CHECK( c_module->is_valid( context.get()));
            c_module = transaction->access<mi::neuraylib::IModule>( "mdl::test_mdl_reload_4");
            MI_CHECK( c_module->is_valid( context.get()));
        }
    }
}

void check_mdl_export_reimport(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::INeuray* neuray)
{
    Exreimp_data modules[] = {
        { "mdl::non_existing", "::non_existing", 6002, 6002 },
        { "mdl::base", "::base", 6004, 6004 },
        { "mdl::mdl_elements::test_misc", "::test_misc", 0, 0 },

        { "mdl::test_archives", "::test_archives", 0, 0, false, false, false, {
            "/test_archives/test_in_archive.png",
            "/test_archives/test_in_archive.ies",
            "/test_archives/test_in_archive.mbsdf" } },

        { "mdl::test_archives", "::test_archives2", 0, 6015, true, false, false, {
            "./test_in_archive.png",
            "./test_in_archive.ies",
            "./test_in_archive.mbsdf" } },

        { "mdl::test_archives", "::test_archives3", 0, 6006, false, true, false, {
            "./test_in_archive_0.png",
            "./test_in_archive_0.ies",
            "./test_in_archive_0.mbsdf" } },

        { "mdl::test_archives", "::test_archives4", 0, 0, false, false, true, {
            "/test_archives/test_in_archive.png",
            "/test_archives/test_in_archive.ies",
            "/test_archives/test_in_archive.mbsdf" } },

        { "mdl::test_archives", "::test_archives5", 0, 6006, false, true, true, {
            "./test_archives5_export_test_in_archive.png",
            "./test_archives5_export_test_in_archive.ies",
            "./test_archives5_export_test_in_archive.mbsdf" } },

        { "mdl::from_string", "::from_string", 0, 0 },
        { "mdl::imports_from_string", "::imports_from_string", 0, 0 },
        { "mdl::mybuiltins", "::mybuiltins", 6004, 6004  },
    };

    for( const auto& module: modules)
        do_check_mdl_export_reimport( transaction, neuray, module);
}

void check_module_references( mi::neuraylib::IDatabase* database)
{
    // Check that a modules references its function definitions.

    mi::base::Handle<mi::neuraylib::IScope> scope( database->get_global_scope());

    // check that "mdl::" TEST_MDL "::fd_remove(int)" exists and schedule it for removal
    mi::base::Handle<mi::neuraylib::ITransaction> transaction( scope->create_transaction());
    {
        mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::" TEST_MDL "::fd_remove(int)"));
        MI_CHECK( fd);
    }
    MI_CHECK_EQUAL( 0, transaction->remove( "mdl::" TEST_MDL "::fd_remove(int)"));
    MI_CHECK_EQUAL( 0, transaction->commit());

    // the GC should not remove "mdl::" TEST_MDL "::fd_remove(int)" (it is supposed to be
    // referenced by its module)
    database->garbage_collection();

    // check that "mdl::" TEST_MDL "::fd_remove(int)" is not gone
    transaction = scope->create_transaction();
    {
        mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::" TEST_MDL "::fd_remove(int)"));
        MI_CHECK( fd);
    }

    // iterate over all functions and materials of "mdl::" TEST_MDL and check that
    // the DB elements still exists
    {
        mi::base::Handle<const mi::neuraylib::IModule> module(
            transaction->access<mi::neuraylib::IModule>( "mdl::" TEST_MDL));
        MI_CHECK( module);
        mi::Size function_count = module->get_function_count();
        for( mi::Size i = 0; i < function_count; ++i) {
            const char* name = module->get_function( i);
            mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
                transaction->access<mi::neuraylib::IFunction_definition>( name));
            MI_CHECK( fd);
        }
        mi::Size material_count = module->get_material_count();
        for( mi::Size i = 0; i < material_count; ++i) {
            const char* name = module->get_material( i);
            mi::base::Handle<const mi::neuraylib::IFunction_definition> md(
                transaction->access<mi::neuraylib::IFunction_definition>( name));
            MI_CHECK( md);
        }
    }

    MI_CHECK_EQUAL( 0, transaction->commit());
}

void run_tests( mi::neuraylib::INeuray* neuray)
{
    fs::remove_all( fs::u8path( DIR_PREFIX));
    fs::create_directory( fs::u8path( DIR_PREFIX));

    // No entity resolver available before mi::neuraylib::INeuray::start().
    mi::base::Handle<mi::neuraylib::IMdl_configuration> mdl_configuration(
        neuray->get_api_component<mi::neuraylib::IMdl_configuration>());
    mi::base::Handle<mi::neuraylib::IMdl_entity_resolver> resolver(
        mdl_configuration->get_entity_resolver());
    MI_CHECK( !resolver);

    MI_CHECK_EQUAL( 0, neuray->start());

    {
        mi::base::Handle<mi::neuraylib::IDatabase> database(
            neuray->get_api_component<mi::neuraylib::IDatabase>());
        mi::base::Handle<mi::neuraylib::IScope> global_scope( database->get_global_scope());
        mi::base::Handle<mi::neuraylib::ITransaction> transaction(
            global_scope->create_transaction());

        mi::base::Handle<mi::neuraylib::IMdl_configuration> mdl_configuration(
            neuray->get_api_component<mi::neuraylib::IMdl_configuration>());
        MI_CHECK_EQUAL( 0, mdl_configuration->add_mdl_path( DIR_PREFIX));
        MI_CHECK_EQUAL( 0, mdl_configuration->add_resource_path( DIR_PREFIX));

        mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
            neuray->get_api_component<mi::neuraylib::IMdl_factory>());

        // Required by check_mdl_export_reimport().
        mi::base::Handle<mi::neuraylib::IMdl_compiler> mdl_compiler(
            neuray->get_api_component<mi::neuraylib::IMdl_compiler>());
        result = mdl_compiler->add_builtin_module( "::mybuiltins", native_module);
        MI_CHECK_EQUAL( 0, result);
        mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
            neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());
        result = mdl_impexp_api->load_module( transaction.get(), "::mybuiltins");
        MI_CHECK_EQUAL( 0, result);

        install_external_resolver( neuray);

        // run the actual tests

        // Load the primary modules and check the IModule interface.
        load_primary_modules( transaction.get(), neuray);

        // Create function calls used by many later tests.
        create_function_calls( transaction.get(), mdl_factory.get());

        // Misc checks on the primary module.
        check_matrix( transaction.get());
        check_immediate_arrays( transaction.get(), mdl_factory.get());
        check_deferred_arrays( transaction.get(), mdl_factory.get());
        check_set_get_value( transaction.get(), mdl_factory.get());
        check_wrappers( transaction.get(), mdl_factory.get()); // after check_deferred_arrays()
        check_resources( transaction.get(), mdl_factory.get());
        check_resources2( transaction.get(), neuray);
        check_cycles( transaction.get(), mdl_factory.get());
        check_uniform_auto_varying( transaction.get(), mdl_factory.get());
        check_analyze_uniform( transaction.get(), mdl_factory.get());
        check_deep_copy_of_defaults( transaction.get());
        check_immutable_defaults( transaction.get(), mdl_factory.get());
        check_annotations( transaction.get(), mdl_factory.get());
        check_parameter_references( transaction.get(), mdl_factory.get());
        check_reexported_function( transaction.get(), mdl_factory.get());
        check_named_temporaries( transaction.get());

        // Misc checks that requires a few more modules to load.
        load_secondary_modules( transaction.get(), neuray);
        load_secondary_modules_from_string( transaction.get(), neuray);
        check_imodule_part2( transaction.get(), neuray);
        check_mangled_names( transaction.get());
        check_implicit_casts( transaction.get(), neuray);

        // Module reload
        check_mdl_reload( transaction.get(), neuray);

        // Export/re-import all modules created so far.
        check_mdl_export_reimport( transaction.get(), neuray);

        MI_CHECK_EQUAL( 0, transaction->commit());

        // Uses separate transactions (relying on the main transaction).
        check_module_references( database.get());

        uninstall_external_resolver( neuray);
    }

    MI_CHECK_EQUAL( 0, neuray->shutdown());
}

MI_TEST_AUTO_FUNCTION( test_imodule_advanced )
{
    mi::base::Handle<mi::neuraylib::INeuray> neuray( load_and_get_ineuray());
    MI_CHECK( neuray);

    {
        mi::base::Handle<mi::neuraylib::IDebug_configuration> debug_configuration(
            neuray->get_api_component<mi::neuraylib::IDebug_configuration>());
        MI_CHECK_EQUAL( 0, debug_configuration->set_option( "check_serializer_store=1"));
        MI_CHECK_EQUAL( 0, debug_configuration->set_option( "check_serializer_edit=1"));

        // set MDL paths
        std::string path1 = MI::TEST::mi_src_path( "prod/lib/neuray");
        std::string path2 = MI::TEST::mi_src_path( "io/scene");
        std::string path3 = MI::TEST::mi_src_path( "prod");
        std::string path4 = MI::TEST::mi_src_path( "prod/lib");
        set_mdl_paths( neuray.get(), {path1, path2, path3, path4});

        // load plugins
        mi::base::Handle<mi::neuraylib::IPlugin_configuration> plugin_configuration(
            neuray->get_api_component<mi::neuraylib::IPlugin_configuration>());
        MI_CHECK_EQUAL( 0, plugin_configuration->load_plugin_library( plugin_path_dds));
        MI_CHECK_EQUAL( 0, plugin_configuration->load_plugin_library( plugin_path_openimageio));

        // Required by check_module_builder_misc().
        mi::base::Handle<mi::neuraylib::IMdl_configuration> mdl_configuration(
            neuray->get_api_component<mi::neuraylib::IMdl_configuration>());
        MI_CHECK_EQUAL( 0, mdl_configuration->set_expose_names_of_let_expressions( true));

        run_tests( neuray.get());
        // MDL SDK must be able to run the test a second time, test that
        run_tests( neuray.get());
    }

    neuray = nullptr;
    MI_CHECK( unload());
}

MI_TEST_MAIN_CALLING_TEST_MAIN();

