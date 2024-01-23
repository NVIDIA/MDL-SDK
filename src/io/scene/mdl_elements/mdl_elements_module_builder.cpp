/***************************************************************************************************
 * Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
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

#include "pch.h"

#include "i_mdl_elements_module_builder.h"

#include <map>
#include <queue>

#include <mi/mdl/mdl_module_transformer.h>
#include <mi/mdl/mdl_thread_context.h>
#include <base/data/db/i_db_transaction.h>
#include <base/lib/config/config.h>
#include <base/util/registry/i_config_registry.h>
#include <base/util/string_utils/i_string_utils.h>
#include <base/system/main/access_module.h>
#include <mdl/compiler/compilercore/compilercore_analysis.h>
#include <mdl/compiler/compilercore/compilercore_modules.h>
#include <mdl/compiler/compilercore/compilercore_tools.h>
#include <mdl/compiler/compilercore/compilercore_checker.h>
#include <mdl/integration/mdlnr/i_mdlnr.h>

#include "mdl_elements_ast_builder.h"
#include "mdl_elements_utilities.h"
#include "i_mdl_elements_function_call.h"
#include "i_mdl_elements_function_definition.h"
#include "i_mdl_elements_module_transformer.h"

// Disable false positives (claiming expressions involving "class_id" always being true or false)
//-V:class_id:547 PVS

/// Usage of error codes
///  -1.. -8: constructor, is_valid(), version upgrade
/// -10..-18: shared functionality, analyze, removal
/// -20..-23: prototype-based functions/materials/variants
/// -30..-34: expression-based functions/materials
/// -40..-47: annotations
/// -50..-57: uniform analysis
/// -60..-66: enums/structs
///
/// In use by API wrapper: -1, -10

namespace MI {

namespace MDL {

namespace {

mi::mdl::IType_name* create_material_type_name( mi::mdl::IModule* module)
{
    mi::mdl::IName_factory& nf = *module->get_name_factory();
    const mi::mdl::ISymbol* return_type_symbol = nf.create_symbol( "material");
    const mi::mdl::ISimple_name* return_type_simple_name
        = nf.create_simple_name( return_type_symbol);
    mi::mdl::IQualified_name* return_type_qualified_name = nf.create_qualified_name();
    return_type_qualified_name->add_component( return_type_simple_name);
    return nf.create_type_name( return_type_qualified_name);
}

mi::mdl::IType_name* create_type_name( mi::mdl::IModule* module, const mi::mdl::IType* type)
{
    mi::mdl::IType_name* type_name = mi::mdl::create_type_name( type, module);
    if( type->get_type_modifiers() & mi::mdl::IType::MK_UNIFORM)
        type_name->set_qualifier( mi::mdl::FQ_UNIFORM);
    return type_name;
}

mi::mdl::IType_name* create_return_type_name(
    DB::Transaction* transaction,
    mi::mdl::IModule* module,
    const Mdl_function_definition* definition)
{
    if( definition->is_material())
        return create_material_type_name( module);

    const mi::mdl::IType* return_mdl_type = definition->get_mdl_return_type( transaction);
    return create_type_name( module, return_mdl_type);
}

} // anonymous

Mdl_module_builder::Mdl_module_builder(
    DB::Transaction* transaction,
    const char* db_module_name,
    mi::mdl::IMDL::MDL_version min_mdl_version,
    mi::mdl::IMDL::MDL_version max_mdl_version,
    bool export_to_db,
    Execution_context* context)
  : m_transaction( transaction),
    m_thread_context( nullptr),
    m_min_mdl_version( min_mdl_version),
    m_max_mdl_version( max_mdl_version),
    m_module( nullptr),
    m_export_to_db( export_to_db),
    m_symbol_importer( nullptr),
    m_name_mangler( nullptr),
    m_af( nullptr),
    m_df( nullptr),
    m_ef( nullptr),
    m_nf( nullptr),
    m_sf( nullptr),
    m_tf( nullptr),
    m_vf( nullptr),
    m_int_tf( get_type_factory()),
    m_int_ef( get_expression_factory())
{
    m_transaction->pin();

    m_empty_type_list        = m_int_tf->create_type_list( /*initial_capacity*/ 0);
    m_empty_expression_list  = m_int_ef->create_expression_list( /*initial_capacity*/ 0);
    m_empty_annotation_list  = m_int_ef->create_annotation_list( /*initial_capacity*/ 0);
    m_empty_annotation_block = m_int_ef->create_annotation_block( /*initial_capacity*/ 0);

    ASSERT( M_SCENE, context);

    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module( false);
    m_mdl = mdlc_module->get_mdl();
    m_implicit_cast_enabled = mdlc_module->get_implicit_cast_enabled();

    // Create thread context.
    m_thread_context = create_thread_context( m_mdl.get(), context);

    // Reject invalid module names.
    if( strncmp( db_module_name, "mdl::", 5) != 0) {
        add_error_message( context,
            STRING::formatted_string( "The module name \"%s\" is invalid.", db_module_name), -2);
        return;
    }
    std::string mdl_module_name = db_module_name + 3;
    if( !is_valid_module_name( mdl_module_name)) {
        add_error_message( context,
            STRING::formatted_string( "The module name \"%s\" is invalid.", db_module_name), -2);
        return;
    }

    m_db_module_name  = db_module_name;
    m_mdl_module_name = mdl_module_name;

    create_module( context);

    m_name_mangler.reset( new Name_mangler( m_mdl.get(), m_module.get()));
}

Mdl_module_builder::~Mdl_module_builder()
{
    m_transaction->unpin();
}

mi::Sint32 Mdl_module_builder::add_variant(
    const char* name,
    DB::Tag prototype_tag,
    const IExpression_list* defaults,
    const IAnnotation_block* annotations,
    const IAnnotation_block* return_annotations,
    bool is_exported,
    Execution_context* context)
{
    if( !check_valid( context))
        return -1;

    SERIAL::Class_id class_id = m_transaction->get_class_id( prototype_tag);
    if( class_id != ID_MDL_FUNCTION_DEFINITION) {
        add_error_message( context, "The prototype has an unsupported type.", -13);
        return -1;
    }

    // get prototype for obtaining defaults and annotations
    DB::Access<Mdl_function_definition> prototype( prototype_tag, m_transaction);

    // get defaults from prototype if NULL
    mi::base::Handle<const IExpression_list> prototype_defaults;
    if( !defaults) {
        prototype_defaults = prototype->get_defaults();
        defaults = prototype_defaults.get();
    }

    // check for unsupported functions
    mi::neuraylib::IFunction_definition::Semantics sema = prototype->get_semantic();
    if( !is_supported_prototype( sema, /*for_variant*/ true)) {
        add_error_message(
            context, "This kind of function is not supported as a prototype.", -12);
        return -1;
    }

    // get annotations from prototype if NULL
    mi::base::Handle<const IAnnotation_block> prototype_annotations;
    if( !annotations) {
        prototype_annotations = prototype->get_annotations();
        annotations = prototype_annotations.get();
    }

    if( prototype->is_material()) {
        // no return annotations for materials
        if( return_annotations && return_annotations->get_size() > 0) {
            add_error_message(
                context, "Return annotations are not feasible for materials.", -11);
            return -1;
        }
    } else {
        // get return annotations from prototype if NULL
        mi::base::Handle<const IAnnotation_block> prototype_return_annotations;
        if( !return_annotations) {
            prototype_return_annotations = prototype->get_return_annotations();
            return_annotations = prototype_return_annotations.get();
        }
    }

    // create the definition
    return add_prototype_based(
        name,
        prototype.get_ptr(),
        defaults,
        /*parameter_annotations*/ nullptr,
        annotations,
        return_annotations,
        /*is_variant*/ true,
        is_exported,
        /*for_mdle*/ false,
        context);
}

namespace {

bool is_material_type( const IType* type)
{
    mi::base::Handle<const IType> stripped_type( type->skip_all_type_aliases());
    mi::base::Handle<const IType_struct> struct_type(
        stripped_type->get_interface<IType_struct>());
    return struct_type && struct_type->get_predefined_id() == IType_struct::SID_MATERIAL;
}

} // namespace

mi::Sint32 Mdl_module_builder::add_function(
    const char* name,
    const IExpression* body,
    const IType_list* parameters,
    const IExpression_list* defaults,
    const IAnnotation_list* parameter_annotations,
    const IAnnotation_block* annotations,
    const IAnnotation_block* return_annotations,
    bool is_exported,
    IType::Modifier frequency_qualifier,
    Execution_context* context)
{
    // handle NULL arguments
    ASSERT( M_SCENE, name);
    ASSERT( M_SCENE, body);
    if( !parameters)
        parameters = m_empty_type_list.get();
    if( !defaults)
        defaults = m_empty_expression_list.get();
    if( !parameter_annotations)
        parameter_annotations = m_empty_annotation_list.get();
    if( !annotations)
        annotations = m_empty_annotation_block.get();
    if( !return_annotations)
        return_annotations = m_empty_annotation_block.get();
    ASSERT( M_SCENE, context);

    if( !check_valid( context))
        return -1;

    if( !validate_name( name, context))
        return -1;

    if( !validate_expression(
        m_transaction,
        body,
        /*allow_calls*/ false,
        /*alllow_direct_calls*/ true,
        /*allowed_parameter_count*/ parameters->get_size(),
        /*allowed_temporary_count*/ 0,
        context))
        return -1;

    if( !validate_expression_list(
        m_transaction,
        defaults,
        /*allow_calls*/ true,
        /*alllow_direct_calls*/ true,
        /*allowed_parameter_count*/ 0,
        /*allowed_temporary_count*/ 0,
        context))
        return -1;

    // upgrade MDL version if necessary (and possible)
    mi::neuraylib::Mdl_version version = get_min_required_mdl_version(
        m_transaction,
        body,
        defaults,
        parameter_annotations,
        annotations,
        return_annotations);
    upgrade_mdl_version( version, context);
    if( context->get_error_messages_count() > 0)
        return -1;

    // material or function
    mi::base::Handle<const IType> return_type( body->get_type());
    bool is_material = is_material_type( return_type.get());
    if( is_material && return_annotations && return_annotations->get_size() > 0) {
        add_error_message( context,
            "Return annotations are not feasible for materials.", -11);
        return -1;
    }

    // check parameter names and create symbols
    mi::Size n_parameters = parameters->get_size();
    std::vector<const mi::mdl::ISymbol*> parameter_symbols( n_parameters);
    for( mi::Size i = 0; i < n_parameters; ++i) {
        const char* parameter_name = parameters->get_name( i);
        if( !m_mdl->is_valid_mdl_identifier( parameter_name)) {
            add_error_message( context,
                STRING::formatted_string( "Invalid parameter name \"%s\".", parameter_name), -30);
            return -1;
        }
        parameter_symbols[i] = m_nf->create_symbol( parameter_name);
    }

    // check for surplus defaults
    mi::Size n_defaults = defaults->get_size();
    for( mi::Size i = 0; i < n_defaults; ++i) {
        const char* default_name = defaults->get_name( i);
        if( parameters->get_index( default_name) == mi::Size( -1)) {
            add_error_message( context,
                STRING::formatted_string( "Invalid default name \"%s\".", default_name), -31);
            return -1;
        }
    }

    // check for surplus parameter_anotations
    mi::Size n_parameter_annotations = parameter_annotations->get_size();
    for( mi::Size i = 0; i < n_parameter_annotations; ++i) {
        const char* parameter_annotation_name = parameter_annotations->get_name( i);
        if( parameters->get_index( parameter_annotation_name) == mi::Size( -1)) {
            add_error_message( context,
                STRING::formatted_string( "Invalid parameter annotation name \"%s\".",
                parameter_annotation_name), -32);
            return -1;
        }
    }

    // create type name for the return type
    const mi::mdl::IType_name* return_type_tn = nullptr;
    if( is_material) {
        return_type_tn = create_material_type_name( m_module.get());
    } else {
        const mi::mdl::IType* mdl_return_type(
            int_type_to_mdl_type( return_type.get(), *m_tf));
        return_type_tn = create_type_name( m_module.get(), mdl_return_type);
    }
    m_symbol_importer->collect_imports( return_type_tn);

    // setup AST builder
    Mdl_ast_builder ast_builder(
        m_module.get(),
        m_transaction,
        /*traverse_ek_parameter*/ false,
        /*args*/ nullptr,
        *m_name_mangler,
        /*avoid_resource_urls*/ false);

    // add parameters
    for( mi::Size i = 0; i < n_parameters; ++i) {
        const char* parameter_name = parameters->get_name( i);
        mi::base::Handle<const IExpression> default_(
            defaults->get_expression( parameter_name));
        ast_builder.declare_parameter( parameter_symbols[i], default_.get());
    }

    mi::mdl::IStatement* stmt_body = nullptr;

    IExpression::Kind kind = body->get_kind();
    if( kind == IExpression::EK_CONSTANT || kind == IExpression::EK_PARAMETER) {

        // handle constant and parameter references as body
        const mi::mdl::IExpression* mdl_body = ast_builder.transform_expr( body);
        if( is_material) {
            stmt_body = m_sf->create_expression( mdl_body);
        } else {
            mi::mdl::IStatement_return* stmt_ret = m_sf->create_return( mdl_body);
            mi::mdl::IStatement_compound* stmt_compound = m_sf->create_compound();
            stmt_compound->add_statement( stmt_ret);
            stmt_body = stmt_compound;
        }

   } else {

        // handle direct calls as body

        // access function definition
        mi::base::Handle<const IExpression_direct_call> body_direct_call(
            body->get_interface<IExpression_direct_call>());
        ASSERT( M_SCENE, body_direct_call);
        DB::Tag body_tag = body_direct_call->get_definition( m_transaction);
        SERIAL::Class_id class_id = m_transaction->get_class_id( body_tag);
        if( class_id != ID_MDL_FUNCTION_DEFINITION) {
            const char* name = m_transaction->tag_to_name( body_tag);
            add_error_message( context,
                STRING::formatted_string( "Invalid reference to DB element \"%s\" in call "
                    "expression.", name),
                -14);
            return -1;
        }

        DB::Access<Mdl_function_definition> body_definition( body_tag, m_transaction);

        // get semantic
        mi::mdl::IDefinition::Semantics sema = body_definition->get_mdl_semantic();
        bool is_operator = mi::mdl::semantic_is_operator( sema)
            || sema == mi::mdl::IDefinition::DS_INTRINSIC_DAG_FIELD_ACCESS;

        // get definition name (without signature)
        std::string definition_name = decode_name_without_signature(
            body_definition->get_mdl_name_without_parameter_types());
        definition_name = remove_qualifiers_if_from_module( definition_name, m_module->get_name());

        // start body creation
        mi::mdl::IExpression_call* call = nullptr;
        if( !is_operator) {
            const mi::mdl::IExpression_reference* ref = signature_to_reference(
                m_module.get(),
                definition_name.c_str(),
                m_name_mangler.get());
            call = m_ef->create_call( ref);
        }
        ASSERT( M_SCENE, is_operator ^ !!call);

        // create let expression for materials, compound statement for functions
        mi::mdl::IStatement_compound* function_body = nullptr;
        mi::mdl::IExpression_let* material_let = nullptr;
        if( is_material)
            material_let = m_ef->create_let( call);
        else
            function_body = m_sf->create_compound();
        ASSERT( M_SCENE, !is_material || material_let);
        ASSERT( M_SCENE,  is_material || function_body);

        class Operator_parameter
        {
        public:
            Operator_parameter(
                const mi::mdl::ISymbol* symbol, mi::base::Handle<const IExpression>& expression)
              : m_symbol( symbol), m_expression( expression) { }
            const mi::mdl::ISymbol* m_symbol;
            const mi::base::Handle<const IExpression> m_expression;
        };

        // setup variables for the body, temporaries are created with "auto" type
        mi::base::Handle<const IExpression_list> callee_args( body_direct_call->get_arguments());
        mi::Size n_callee_parameters = callee_args->get_size();
        std::vector<Operator_parameter> operator_parameters;
        for( mi::Size i = 0; i < n_callee_parameters; ++i) {

                mi::base::Handle<const IExpression> arg( callee_args->get_expression( i));
                mi::base::Handle<const IType> type( arg->get_type());

                mi::mdl::IType_name* tn = ast_builder.create_type_name( type);
                m_symbol_importer->collect_imports( tn);

                // create variable
                mi::mdl::IDeclaration_variable* variable
                    = m_df->create_variable( tn, /*exported*/ false);
                const mi::mdl::IExpression* mdl_arg = ast_builder.transform_expr( arg.get());
                m_symbol_importer->collect_imports( mdl_arg);
                const mi::mdl::ISymbol* tmp_sym = ast_builder.get_temporary_symbol();
                variable->add_variable( ast_builder.to_simple_name( tmp_sym), mdl_arg);

                // add variable to material let/function body
                if( is_material)
                    material_let->add_declaration( variable); //-V522 PVS
                else {
                    mi::mdl::IStatement_declaration* stmt = m_sf->create_declaration( variable);
                    function_body->add_statement( stmt); //-V522 PVS
                }

                // add variable as call argument (or add to vector for operators)
                const char* name = callee_args->get_name( i);
                const mi::mdl::ISymbol* parameter_symbol = m_nf->create_symbol( name);
                const mi::mdl::ISimple_name* sname = ast_builder.to_simple_name( parameter_symbol);
                const mi::mdl::IExpression_reference* ref = ast_builder.to_reference( tmp_sym);
                if( !is_operator)
                    call->add_argument( m_ef->create_named_argument( sname, ref)); //-V522 PVS
                else
                    operator_parameters.emplace_back( tmp_sym, arg);
        }
        ASSERT( M_SCENE, is_operator ^ operator_parameters.empty());

        // promote call expression
        const mi::mdl::IExpression* promoted_call = nullptr;
        if( !is_operator) {
            promoted_call = mi::mdl::promote_expressions_to_mdl_version( m_module.get(), call);
            if( !promoted_call) {
                add_error_message( context, "Failed to promote call expression to requested MDL "
                    "version.", -33);
                return -1;
            }
            m_symbol_importer->collect_imports( promoted_call);
            call = nullptr;

            if( is_material) {
                material_let->set_expression( promoted_call);
                promoted_call = nullptr;
            }
        }

        // finalize body
        if( is_material) {

            stmt_body = m_sf->create_expression( material_let);
            material_let = nullptr;

        } else {

            const mi::mdl::IExpression* return_expr = nullptr;
            if( !is_operator) {

                return_expr = promoted_call;
                promoted_call = nullptr;

            } else {

                ast_builder.remove_parameters();
                for( const auto& ov : operator_parameters)
                    ast_builder.declare_parameter( ov.m_symbol, ov.m_expression.get());

                mi::base::Handle<const IType> return_type( body->get_type());
                return_expr = ast_builder.transform_call(
                    return_type.get(),
                    sema,
                    definition_name,
                    n_callee_parameters,
                    callee_args.get(),
                    /*named_args*/ true);
            }

            mi::mdl::IStatement_return* stmt = m_sf->create_return( return_expr);
            function_body->add_statement( stmt);
            stmt_body = function_body;
            function_body = nullptr;

        }
    }

    // convert annotations and return annotations
    mi::mdl::IAnnotation_block* mdl_annotations
        = int_anno_block_to_mdl_anno_block( annotations, /*skip_anno_unused*/ false, context);
    if( context->get_error_messages_count() > 0)
        return -1;
    mi::mdl::IAnnotation_block* mdl_return_annotations
        = int_anno_block_to_mdl_anno_block( return_annotations, /*skip_anno_unused*/false, context);
    if( context->get_error_messages_count() > 0)
        return -1;

    // creation function declaration
    const mi::mdl::ISymbol* func_sym = m_nf->create_symbol( name);
    const mi::mdl::ISimple_name* func_sname = m_nf->create_simple_name( func_sym);
    mi::mdl::IDeclaration_function* func_decl = m_df->create_function(
        return_type_tn,
        mdl_return_annotations,
        func_sname,
        /*is_clone*/ false,
        stmt_body,
        mdl_annotations,
        is_exported);

    // set frequency qualifier
    if( (frequency_qualifier == IType::MK_UNIFORM) && !is_material)
        func_decl->set_qualifier( mi::mdl::FQ_UNIFORM);
    else if( (frequency_qualifier == IType::MK_VARYING) && !is_material)
        func_decl->set_qualifier( mi::mdl::FQ_VARYING);
    else if( frequency_qualifier == IType::MK_NONE)
        func_decl->set_qualifier( mi::mdl::FQ_NONE);
    else {
        add_error_message( context, STRING::formatted_string(
            "Invalid frequency qualifier for %s \"%s\".",
            is_material ? "material" : "function", name), -34);
        return -1;
    }

    // add parameters to function declaration
    ast_builder.remove_parameters();
    for( mi::Size i = 0; i < n_parameters; ++i) {

        const char* name = parameters->get_name( i);

        mi::base::Handle<const IType> parameter_type( parameters->get_type( i));
        mi::mdl::IType_name* tn = ast_builder.create_type_name( parameter_type);
        if( parameter_type->get_all_type_modifiers() & IType::MK_UNIFORM)
            tn->set_qualifier( mi::mdl::FQ_UNIFORM);
        m_symbol_importer->collect_imports( tn);

        const mi::mdl::ISimple_name* sname = m_nf->create_simple_name( parameter_symbols[i]);
        const mi::mdl::IExpression* init = nullptr;
        mi::base::Handle<const IExpression> default_( defaults->get_expression( name));
        if( default_) {
            init = ast_builder.transform_expr( default_.get());
            m_symbol_importer->collect_imports( init);
        }

        mi::base::Handle<const IAnnotation_block> annotations(
            parameter_annotations->get_annotation_block( name));
        mi::mdl::IAnnotation_block* mdl_annotations = int_anno_block_to_mdl_anno_block(
            annotations.get(), /*skip_unused*/ false, context);
        if( context->get_error_messages_count() > 0)
            return -1;

        const mi::mdl::IParameter* param
            = m_df->create_parameter( tn, sname, init, mdl_annotations);
        func_decl->add_parameter( param);
    }

    // add declaration to module
    m_module->add_declaration( func_decl);

    // add types/imports/aliases
    m_symbol_importer->add_names( ast_builder.get_used_user_types());
    m_symbol_importer->add_imports();
    m_name_mangler->add_namespace_aliases( m_module.get());

    analyze_module( context);
    if( context->get_error_messages_count() > 0)
        return -1;

    return 0;
}

mi::Sint32 Mdl_module_builder::add_annotation(
    const char* name,
    const IType_list* parameters,
    const IExpression_list* defaults,
    const IAnnotation_list* parameter_annotations,
    const IAnnotation_block* annotations,
    bool is_exported,
    Execution_context* context)
{
    // handle NULL arguments
    ASSERT( M_SCENE, name);
    if( !parameters)
        parameters = m_empty_type_list.get();
    if( !defaults)
        defaults = m_empty_expression_list.get();
    if( !parameter_annotations)
        parameter_annotations = m_empty_annotation_list.get();
    if( !annotations)
        annotations = m_empty_annotation_block.get();
    ASSERT( M_SCENE, context);

    if( !check_valid( context))
        return -1;

    if( !validate_name( name, context))
        return -1;

    if( !validate_expression_list(
        m_transaction,
        defaults,
        /*allow_calls*/ true,
        /*alllow_direct_calls*/ true,
        /*allowed_parameter_count*/ 0,
        /*allowed_temporary_count*/ 0,
        context))
        return -1;

    // upgrade MDL version if necessary (and possible)
    mi::neuraylib::Mdl_version version = get_min_required_mdl_version(
        m_transaction,
        defaults,
        parameter_annotations,
        annotations);
    upgrade_mdl_version( version, context);
    if( context->get_error_messages_count() > 0)
        return -1;

    // check parameter names and create symbols
    mi::Size n_parameters = parameters->get_size();
    std::vector<const mi::mdl::ISymbol*> parameter_symbols( n_parameters);
    for( mi::Size i = 0; i < n_parameters; ++i) {
        const char* parameter_name = parameters->get_name( i);
        if( !m_mdl->is_valid_mdl_identifier( parameter_name)) {
            add_error_message( context,
                STRING::formatted_string( "Invalid parameter name \"%s\".", parameter_name), -30);
            return -1;
        }
        parameter_symbols[i] = m_nf->create_symbol( parameter_name);
    }

    // check for surplus defaults
    mi::Size n_defaults = defaults->get_size();
    for( mi::Size i = 0; i < n_defaults; ++i) {
        const char* default_name = defaults->get_name( i);
        if( parameters->get_index( default_name) == mi::Size( -1)) {
            add_error_message( context,
                STRING::formatted_string( "Invalid default name \"%s\".", default_name), -31);
            return -1;
        }
    }

    // check for surplus parameter_anotations
    mi::Size n_parameter_annotations = parameter_annotations->get_size();
    for( mi::Size i = 0; i < n_parameter_annotations; ++i) {
        const char* parameter_annotation_name = parameter_annotations->get_name( i);
        if( parameters->get_index( parameter_annotation_name) == mi::Size( -1)) {
            add_error_message( context,
                STRING::formatted_string( "Invalid parameter annotation name \"%s\".",
                parameter_annotation_name), -32);
            return -1;
        }
    }

    // setup AST builder
    Mdl_ast_builder ast_builder(
        m_module.get(),
        m_transaction,
        /*traverse_ek_parameter*/ false,
        /*args*/ nullptr,
        *m_name_mangler,
        /*avoid_resource_urls*/ false);

    // add parameters
    for( mi::Size i = 0; i < n_parameters; ++i) {
        const char* parameter_name = parameters->get_name( i);
        mi::base::Handle<const IExpression> default_(
            defaults->get_expression( parameter_name));
        ast_builder.declare_parameter( parameter_symbols[i], default_.get());
    }

    // convert annotations
    mi::mdl::IAnnotation_block* mdl_annotations
        = int_anno_block_to_mdl_anno_block( annotations, /*skip_anno_unused*/ false, context);
    if( context->get_error_messages_count() > 0)
        return -1;

    // creation annotation declaration
    const mi::mdl::ISymbol* anno_sym = m_nf->create_symbol( name);
    const mi::mdl::ISimple_name* anno_sname = m_nf->create_simple_name( anno_sym);
    mi::mdl::IDeclaration_annotation* anno_decl = m_df->create_annotation(
        anno_sname,
        mdl_annotations,
        is_exported);

    // add parameters to annotation declaration
    ast_builder.remove_parameters();
    for( mi::Size i = 0; i < n_parameters; ++i) {

        const char* name = parameters->get_name( i);

        mi::base::Handle<const IType> parameter_type( parameters->get_type( i));
        mi::mdl::IType_name* tn = ast_builder.create_type_name( parameter_type);
        if( parameter_type->get_all_type_modifiers() & IType::MK_UNIFORM)
            tn->set_qualifier( mi::mdl::FQ_UNIFORM);
        m_symbol_importer->collect_imports( tn);

        const mi::mdl::ISimple_name* sname = m_nf->create_simple_name( parameter_symbols[i]);
        const mi::mdl::IExpression* init = nullptr;
        mi::base::Handle<const IExpression> default_( defaults->get_expression( name));
        if( default_) {
            init = ast_builder.transform_expr( default_.get());
            m_symbol_importer->collect_imports( init);
        }

        mi::base::Handle<const IAnnotation_block> annotations(
            parameter_annotations->get_annotation_block( name));
        mi::mdl::IAnnotation_block* mdl_annotations = int_anno_block_to_mdl_anno_block(
            annotations.get(), /*skip_unused*/ false, context);
        if( context->get_error_messages_count() > 0)
            return -1;

        const mi::mdl::IParameter* param
            = m_df->create_parameter( tn, sname, init, mdl_annotations);
        anno_decl->add_parameter( param);
    }

    // add declaration to module
    m_module->add_declaration( anno_decl);

    // add types/imports/aliases
    m_symbol_importer->add_names( ast_builder.get_used_user_types());
    m_symbol_importer->add_imports();
    m_name_mangler->add_namespace_aliases( m_module.get());

    analyze_module( context);
    if( context->get_error_messages_count() > 0)
        return -1;

    return 0;
}

mi::Sint32 Mdl_module_builder::add_enum_type(
    const char* name,
    const IExpression_list* enumerators,
    const IAnnotation_list* enumerator_annotations,
    const IAnnotation_block* annotations,
    bool is_exported,
    Execution_context* context)
{
    // handle NULL arguments
    ASSERT( M_SCENE, name);
    ASSERT( M_SCENE, enumerators);
    if( !enumerator_annotations)
        enumerator_annotations = m_empty_annotation_list.get();
    if( !annotations)
        annotations = m_empty_annotation_block.get();
    ASSERT( M_SCENE, context);

    if( !check_valid( context))
        return -1;

    if( !validate_name( name, context))
        return -1;

    if( !validate_expression_list(
        m_transaction,
        enumerators,
        /*allow_calls*/ false,
        /*alllow_direct_calls*/ true,
        /*allowed_parameter_count*/ 0,
        /*allowed_temporary_count*/ 0,
        context))
        return -1;

    // upgrade MDL version if necessary (and possible)
    mi::neuraylib::Mdl_version version = get_min_required_mdl_version(
        m_transaction,
        enumerators,
        enumerator_annotations,
        annotations);
    upgrade_mdl_version( version, context);
    if( context->get_error_messages_count() > 0)
        return -1;

    // check for non-empty enumerators list
    mi::Size n_enumerators = enumerators->get_size();
    if( n_enumerators == 0) {
        add_error_message( context,
            STRING::formatted_string( "Invalid enum type \"%s\" without any enumerators.", name),
            -60);
        return -1;
    }

    // check for surplus enumerator_anotations
    mi::Size n_enumerator_annotations = enumerator_annotations->get_size();
    for( mi::Size i = 0; i < n_enumerator_annotations; ++i) {
        const char* enumerator_annotation_name = enumerator_annotations->get_name( i);
        if( enumerators->get_index( enumerator_annotation_name) == mi::Size( -1)) {
            add_error_message( context,
                STRING::formatted_string( "Invalid enumerator annotation name \"%s\".",
                enumerator_annotation_name), -61);
            return -1;
        }
    }

    // convert annotations
    mi::mdl::IAnnotation_block* mdl_annotations
        = int_anno_block_to_mdl_anno_block( annotations, /*skip_anno_unused*/ false, context);
    if( context->get_error_messages_count() > 0)
        return -1;

    // create enum type declaration
    const mi::mdl::ISymbol* enum_sym = m_nf->create_symbol( name);
    const mi::mdl::ISimple_name* enum_sname = m_nf->create_simple_name( enum_sym);
    mi::mdl::IDeclaration_type_enum* enum_decl = m_df->create_enum(
        enum_sname,
        mdl_annotations,
        is_exported,
        /*is_enum_class*/ false);

    // setup AST builder
    Mdl_ast_builder ast_builder(
        m_module.get(),
        m_transaction,
        /*traverse_ek_parameter*/ false,
        /*args*/ nullptr,
        *m_name_mangler,
        /*avoid_resource_urls*/ false);

    // add enumerators (and check that they are unique)
    std::set<std::string> names;
    for( mi::Size i = 0; i < n_enumerators; ++i) {

        const char* enumerator_name = enumerators->get_name( i);
        if( names.find( enumerator_name) != names.end()) {
            add_error_message( context,
                STRING::formatted_string( "Non-unique enumerator name \"%s\".", enumerator_name),
                -62);
            return -1;
        }

        const mi::mdl::ISymbol* enumerator_sym = m_nf->create_symbol( enumerator_name);
        const mi::mdl::ISimple_name* enumerator_sname = m_nf->create_simple_name( enumerator_sym);

        mi::base::Handle<const IExpression> enumerator_expr( enumerators->get_expression( i));
        mi::base::Handle<const IType> enumerator_type( enumerator_expr->get_type());
        const mi::mdl::IExpression* mdl_enumerator_expr
            = ast_builder.transform_expr( enumerator_expr.get());
        m_symbol_importer->collect_imports( mdl_enumerator_expr);

        mi::base::Handle<const IAnnotation_block> enumerator_block(
            enumerator_annotations->get_annotation_block( enumerator_name));
        mi::mdl::IAnnotation_block* mdl_enumerator_block = int_anno_block_to_mdl_anno_block(
                enumerator_block.get(), /*skip_anno_unused*/ false, context);
        if( context->get_error_messages_count() > 0)
            return -1;

        enum_decl->add_value( enumerator_sname, mdl_enumerator_expr, mdl_enumerator_block);
    }

    // add declaration to module
    m_module->add_declaration( enum_decl);

    // add types/imports/aliases
    m_symbol_importer->add_names( ast_builder.get_used_user_types());
    m_symbol_importer->add_imports();
    m_name_mangler->add_namespace_aliases( m_module.get());

    analyze_module( context);
    if( context->get_error_messages_count() > 0)
        return -1;

    return 0;
}

mi::Sint32 Mdl_module_builder::add_struct_type(
    const char* name,
    const IType_list* fields,
    const IExpression_list* field_defaults,
    const IAnnotation_list* field_annotations,
    const IAnnotation_block* annotations,
    bool is_exported,
    Execution_context* context)
{
    // handle NULL arguments
    ASSERT( M_SCENE, name);
    ASSERT( M_SCENE, fields);
    if( !field_defaults)
        field_defaults = m_empty_expression_list.get();
    if( !field_annotations)
        field_annotations = m_empty_annotation_list.get();
    if( !annotations)
        annotations = m_empty_annotation_block.get();
    ASSERT( M_SCENE, context);

    if( !check_valid( context))
        return -1;

    if( !validate_name( name, context))
        return -1;

    if( !validate_expression_list(
        m_transaction,
        field_defaults,
        /*allow_calls*/ false,
        /*alllow_direct_calls*/ true,
        /*allowed_parameter_count*/ 0,
        /*allowed_temporary_count*/ 0,
        context))
        return -1;

    // upgrade MDL version if necessary (and possible)
    mi::neuraylib::Mdl_version version = get_min_required_mdl_version(
        m_transaction,
        field_defaults,
        field_annotations,
        annotations);
    upgrade_mdl_version( version, context);
    if( context->get_error_messages_count() > 0)
        return -1;

    // check for non-empty fields list
    mi::Size n_fields = fields->get_size();
    if( n_fields == 0) {
        add_error_message( context,
            STRING::formatted_string( "Invalid struct type \"%s\" without any fields.", name),
            -63);
        return -1;
    }

    // check for surplus field_defaults
    mi::Size n_field_defaults = field_defaults->get_size();
    for( mi::Size i = 0; i < n_field_defaults; ++i) {
        const char* field_default_name = field_defaults->get_name( i);
        if( fields->get_index( field_default_name) == mi::Size( -1)) {
            add_error_message( context,
                STRING::formatted_string( "Invalid field default name \"%s\".",
                field_default_name), -64);
            return -1;
        }
    }

    // check for surplus field_anotations
    mi::Size n_field_annotations = field_annotations->get_size();
    for( mi::Size i = 0; i < n_field_annotations; ++i) {
        const char* field_annotation_name = field_annotations->get_name( i);
        if( fields->get_index( field_annotation_name) == mi::Size( -1)) {
            add_error_message( context,
                STRING::formatted_string( "Invalid field annotation name \"%s\".",
                field_annotation_name), -65);
            return -1;
        }
    }

    // convert annotations
    mi::mdl::IAnnotation_block* mdl_annotations
        = int_anno_block_to_mdl_anno_block( annotations, /*skip_anno_unused*/ false, context);
    if( context->get_error_messages_count() > 0)
        return -1;

    // create struct type declaration
    const mi::mdl::ISymbol* struct_sym = m_nf->create_symbol( name);
    const mi::mdl::ISimple_name* struct_sname = m_nf->create_simple_name( struct_sym);
    mi::mdl::IDeclaration_type_struct* struct_decl = m_df->create_struct(
        struct_sname,
        mdl_annotations,
        is_exported);

    // setup AST builder
    Mdl_ast_builder ast_builder(
        m_module.get(),
        m_transaction,
        /*traverse_ek_parameter*/ false,
        /*args*/ nullptr,
        *m_name_mangler,
        /*avoid_resource_urls*/ false);

    // add fields (and check that they are unique)
    std::set<std::string> names;
    for( mi::Size i = 0; i < n_fields; ++i) {

        const char* field_name = fields->get_name( i);
        if( names.find( field_name) != names.end()) {
            add_error_message( context,
                STRING::formatted_string( "Non-unique field name \"%s\".", field_name),
                -66);
            return -1;
        }

        const mi::mdl::ISymbol* field_sym = m_nf->create_symbol( field_name);
        const mi::mdl::ISimple_name* field_sname = m_nf->create_simple_name( field_sym);

        mi::base::Handle<const IType> field_type( fields->get_type( i));
        const mi::mdl::IType* mdl_field_type
            = int_type_to_mdl_type( field_type.get(), *m_tf);
        mi::mdl::IType_name* field_tname = create_type_name( m_module.get(), mdl_field_type);
        m_symbol_importer->collect_imports( field_tname);

        mi::base::Handle<const IExpression> field_default(
            field_defaults->get_expression( field_name));
        const mi::mdl::IExpression* mdl_field_default
            = field_default ? ast_builder.transform_expr( field_default.get()) : nullptr;
        if( mdl_field_default)
            m_symbol_importer->collect_imports( mdl_field_default);

        mi::base::Handle<const IAnnotation_block> field_block(
            field_annotations->get_annotation_block( field_name));
        mi::mdl::IAnnotation_block* mdl_field_block = int_anno_block_to_mdl_anno_block(
                field_block.get(), /*skip_anno_unused*/ false, context);
        if( context->get_error_messages_count() > 0)
            return -1;

        struct_decl->add_field( field_tname, field_sname, mdl_field_default, mdl_field_block);
    }

    // add declaration to module
    m_module->add_declaration( struct_decl);

    // add types/imports/aliases
    m_symbol_importer->add_names( ast_builder.get_used_user_types());
    m_symbol_importer->add_imports();
    m_name_mangler->add_namespace_aliases( m_module.get());

    analyze_module( context);
    if( context->get_error_messages_count() > 0)
        return -1;

    return 0;
}

mi::Sint32 Mdl_module_builder::add_constant(
    const char* name,
    const IExpression* expr,
    const IAnnotation_block* annotations,
    bool is_exported,
    Execution_context* context)
{
    // handle NULL arguments
    ASSERT( M_SCENE, name);
    ASSERT( M_SCENE, expr);
    if( !annotations)
        annotations = m_empty_annotation_block.get();
    ASSERT( M_SCENE, context);

    if( !check_valid( context))
        return -1;

    if( !validate_name( name, context))
        return -1;

    if( !validate_expression(
        m_transaction,
        expr,
        /*allow_calls*/ false,
        /*alllow_direct_calls*/ true,
        /*allowed_parameter_count*/ 0,
        /*allowed_temporary_count*/ 0,
        context))
        return -1;

    // upgrade MDL version if necessary (and possible)
    mi::neuraylib::Mdl_version version = get_min_required_mdl_version(
        m_transaction,
        expr,
        annotations);
    upgrade_mdl_version( version, context);
    if( context->get_error_messages_count() > 0)
        return -1;

    // create constant declaration
    mi::base::Handle<const IType> type( expr->get_type());
    const mi::mdl::IType* mdl_type(
       int_type_to_mdl_type( type.get(), *m_tf));
    mi::mdl::IType_name* tn = create_type_name( m_module.get(), mdl_type);
    mi::mdl::IDeclaration_constant* constant_decl = m_df->create_constant(
        tn, is_exported);

    const mi::mdl::ISymbol* constant_sym = m_nf->create_symbol( name);
    const mi::mdl::ISimple_name* constant_sname = m_nf->create_simple_name( constant_sym);

    // convert expression
    Mdl_ast_builder ast_builder(
        m_module.get(),
        m_transaction,
        /*traverse_ek_parameter*/ false,
        /*args*/ nullptr,
        *m_name_mangler,
        /*avoid_resource_urls*/ false);
    const mi::mdl::IExpression* mdl_expr = ast_builder.transform_expr( expr);
    m_symbol_importer->collect_imports( mdl_expr);

    // convert annotations
    mi::mdl::IAnnotation_block* mdl_annotations = int_anno_block_to_mdl_anno_block(
        annotations, /*skip_anno_unused*/ false, context);
    if( context->get_error_messages_count() > 0)
        return -1;

    constant_decl->add_constant( constant_sname, mdl_expr, mdl_annotations);

    // add declaration to module
    m_module->add_declaration( constant_decl);

    // add types/imports/aliases
    m_symbol_importer->add_names( ast_builder.get_used_user_types());
    m_symbol_importer->add_imports();
    m_name_mangler->add_namespace_aliases( m_module.get());

    analyze_module( context);
    if( context->get_error_messages_count() > 0)
        return -1;

    return 0;
}

mi::Sint32 Mdl_module_builder::set_module_annotations(
    const IAnnotation_block* annotations,
    Execution_context* context)
{
    // convert annotations
    mi::mdl::IAnnotation_block* mdl_annotations = int_anno_block_to_mdl_anno_block(
        annotations, /*skip_anno_unused*/ false, context);
    if( context->get_error_messages_count() > 0)
        return -1;

    // copy declarations and remove existing DK_MODULE declaration (if existing)
    int n = m_module->get_declaration_count();
    std::vector<const mi::mdl::IDeclaration*> declarations;
    declarations.reserve( n);
    for( int i = 0; i < n; ++i) {
        const mi::mdl::IDeclaration* decl = m_module->get_declaration( i);
        if( decl->get_kind() != mi::mdl::IDeclaration::DK_MODULE)
            declarations.push_back( decl);
    }

    // insert new declaration if requested
    if( annotations) {
        mi::mdl::IDeclaration* decl = m_df->create_module( mdl_annotations);
        declarations.insert( declarations.begin(), decl);
    }

    static_cast<mi::mdl::Module*>( m_module.get())->replace_declarations(
        declarations.data(), declarations.size());

    analyze_module( context);
    if( context->get_error_messages_count() > 0)
        return -1;

    return 0;
}

namespace {

// Indicates whether the declaration defines an entity of the given name. Only DK_TYPE_ENUM,
// DK_TYPE_STRUCT, DK_FUNCTION, DK_ANNOTATION, and DK_CONSTANT are considered.
bool has_name( const mi::mdl::IDeclaration* decl, const char* name)
{
    switch( decl->get_kind()) {
        case mi::mdl::IDeclaration::DK_TYPE_ENUM: {
            const auto* decl_enum = mi::mdl::cast<mi::mdl::IDeclaration_type_enum>( decl);
            return strcmp( decl_enum->get_name()->get_symbol()->get_name(), name) == 0;
        }
        case mi::mdl::IDeclaration::DK_TYPE_STRUCT: {
            const auto* decl_struct = mi::mdl::cast<mi::mdl::IDeclaration_type_struct>( decl);
            return strcmp( decl_struct->get_name()->get_symbol()->get_name(), name) == 0;
        }
        case mi::mdl::IDeclaration::DK_FUNCTION: {
            const auto* decl_func = mi::mdl::cast<mi::mdl::IDeclaration_function>( decl);
            return strcmp( decl_func->get_name()->get_symbol()->get_name(), name) == 0;
        }
        case mi::mdl::IDeclaration::DK_ANNOTATION: {
            const auto* decl_anno = mi::mdl::cast<mi::mdl::IDeclaration_annotation>( decl);
            return strcmp( decl_anno->get_name()->get_symbol()->get_name(), name) == 0;
        }
        case mi::mdl::IDeclaration::DK_CONSTANT: {
            const auto* decl_constant = mi::mdl::cast<mi::mdl::IDeclaration_constant>( decl);
            for( int i = 0, n = decl_constant->get_constant_count(); i < n; ++i) {
                const char* s = decl_constant->get_constant_name( i)->get_symbol()->get_name();
                if( strcmp( s, name) == 0)
                    return true;
            }
            return false;
        }
        default:
            return false;
    }
}

// Clones constant declarations with \p name removed, returns NULL for other kinds (or if the
// constant declaration has exactly one element).
mi::mdl::IDeclaration* create_decl(
    mi::mdl::IModule* module, const mi::mdl::IDeclaration* decl, const char* name)
{
    if( decl->get_kind() != mi::mdl::IDeclaration::DK_CONSTANT)
        return nullptr;

    const auto* decl_constant = mi::mdl::cast<mi::mdl::IDeclaration_constant>( decl);
    int n = decl_constant->get_constant_count();
    ASSERT( M_SCENE, n >= 1); // there is at least one constant matching \p name
    if( n == 1)
        return nullptr;

    mi::mdl::IDeclaration_factory* df = module->get_declaration_factory();
    mi::mdl::IDeclaration_constant* result = df->create_constant(
        decl_constant->get_type_name(), decl_constant->is_exported());
    for( int i = 0; i < n; ++i) {
        const char* s = decl_constant->get_constant_name( i)->get_symbol()->get_name();
        if( strcmp( s, name) == 0)
            ; // skip this constant
        else
            result->add_constant(
                decl_constant->get_constant_name( i),
                decl_constant->get_constant_exp( i),
                decl_constant->get_annotations( i));
    }

    ASSERT( M_SCENE, result->get_constant_count() == n-1);
    return result;
}

} // namespace

mi::Sint32 Mdl_module_builder::remove_entity(
    const char* name,
    mi::Size index,
    Execution_context* context)
{
    // handle NULL arguments
    ASSERT( M_SCENE, name);
    ASSERT( M_SCENE, context);

    if( !check_valid( context))
        return -1;

    int n = m_module->get_declaration_count();
    std::vector<const mi::mdl::IDeclaration*> declarations;
    declarations.reserve( n);
    mi::Size to_skip = index;
    bool found = false;

    for( int i = 0; i < n; ++i) {
        const mi::mdl::IDeclaration* decl = m_module->get_declaration( i);
        if( has_name( decl, name)) {
            if( to_skip == 0) {
                found = true;
                mi::mdl::IDeclaration* new_decl = create_decl( m_module.get(), decl, name);
                if( new_decl) {
                    // add modified declarations
                    declarations.push_back( decl);
                } else {
                    // remove declaration
                    ;
                }
           }  else {
                // wrong overload, keep declarations
                declarations.push_back( decl);
                --to_skip; // underflows are fine
            }
        } else {
            // wrong name or irrelevant declaration kind, keep declaration
            declarations.push_back( decl);
        }
    }

    if( !found) {
        add_error_message( context,
            STRING::formatted_string( "No entity named \"%s\" with index %zu found.", name, index),
            -17);
        return -1;
    }

    static_cast<mi::mdl::Module*>( m_module.get())->replace_declarations(
        declarations.data(), declarations.size());

    analyze_module( context);
    if( context->get_error_messages_count() > 0)
        return -1;

    return 0;
}

mi::Sint32 Mdl_module_builder::clear_module(
    Execution_context* context)
{
    // handle NULL arguments
    ASSERT( M_SCENE, context);

    if( !check_valid( context))
        return -1;

    static_cast<mi::mdl::Module*>( m_module.get())->replace_declarations( nullptr, 0);

    analyze_module( context);
    if( context->get_error_messages_count() > 0)
        return -1;

    return 0;
}

std::vector<bool> Mdl_module_builder::analyze_uniform(
    const IExpression* root_expr,
    bool root_expr_uniform,
    Execution_context* context)
{
    std::vector<bool> result;

    if( !check_valid( context))
        return result;

    if( !root_expr)
        return result;

    bool dummy;
    std::string dummy2;
    analyze_uniform(
        m_transaction,
        root_expr,
        root_expr_uniform,
        /*quer_expr*/ nullptr,
        result,
        /*uniform_query_expr*/ dummy,
        /*error_path*/ dummy2,
        context);
    return result;
}

mi::Sint32 Mdl_module_builder::add_function(
    const char* name,
    DB::Tag prototype_tag,
    const IExpression_list* defaults,
    const IAnnotation_list* parameter_annotations,
    const IAnnotation_block* annotations,
    const IAnnotation_block* return_annotations,
    bool is_exported,
    IType::Modifier frequency_qualifier,
    Execution_context* context)
{
    if( !check_valid( context))
        return -1;

    SERIAL::Class_id class_id = m_transaction->get_class_id( prototype_tag);
    if( class_id != ID_MDL_FUNCTION_DEFINITION) {
        add_error_message( context, "The prototype has an unsupported type.", -13);
        return -1;
    }

    // get prototype for obtaining defaults and annotations
    DB::Access<Mdl_function_definition> prototype( prototype_tag, m_transaction);

    // check for unsupported functions
    mi::neuraylib::IFunction_definition::Semantics sema = prototype->get_semantic();
    if( !is_supported_prototype( sema, /*for_variant*/ false)) {
        add_error_message(
            context, "This kind of function is not supported as a prototype.", -12);
        return -1;
    }

    // get defaults from prototype if NULL
    mi::base::Handle<const IExpression_list> prototype_defaults;
    if( !defaults) {
        prototype_defaults = prototype->get_defaults();
        defaults = prototype_defaults.get();
    }

    // get parameter annotations from prototype if NULL
    mi::base::Handle<const IAnnotation_list> prototype_parameter_annotations;
    if( !parameter_annotations) {
        prototype_parameter_annotations = prototype->get_parameter_annotations();
        parameter_annotations = prototype_parameter_annotations.get();
    }

    // get annotations from prototype if NULL
    mi::base::Handle<const IAnnotation_block> prototype_annotations;
    if( !annotations) {
        prototype_annotations = prototype->get_annotations();
        annotations = prototype_annotations.get();
    }

    if( prototype->is_material()) {
        // no return annotations for materials
        if( return_annotations && return_annotations->get_size() > 0) {
            add_error_message(
                context, "Return annotations are not feasible for materials.", -11);
            return -1;
        }
    } else {
        // get return annotations from prototype if NULL
        mi::base::Handle<const IAnnotation_block> prototype_return_annotations;
        if( !return_annotations) {
            prototype_return_annotations = prototype->get_return_annotations();
            return_annotations = prototype_return_annotations.get();
        }
    }

    // create the definition
    return add_prototype_based(
        name,
        prototype.get_ptr(),
        defaults,
        parameter_annotations,
        annotations,
        return_annotations,
        /*is_variant*/ false,
        is_exported,
        /*for_mdle*/ true,
        context);
}

const mi::mdl::IModule* Mdl_module_builder::get_module() const
{
    if( !m_module || !m_module->is_valid())
        return nullptr;

    m_module->retain();
    return m_module.get();
}

bool Mdl_module_builder::check_valid( Execution_context* context)
{
    if( m_module && m_module->is_valid())
        return true;

    add_error_message(
        context, "Module builder is in an invalid state.", -7);
    return false;
}

mi::Sint32 Mdl_module_builder::add_prototype_based(
     const char* name,
     const Mdl_function_definition* prototype,
     const IExpression_list* defaults,
     const IAnnotation_list* parameter_annotations,
     const IAnnotation_block* annotations,
     const IAnnotation_block* return_annotations,
     bool is_variant,
     bool is_exported,
     bool for_mdle,
     Execution_context* context)
{
    ASSERT( M_SCENE, name);
    // variants have no parameter_annotations
    ASSERT( M_SCENE, !is_variant
                     || (!parameter_annotations || parameter_annotations->get_size() == 0));
    // materials have no return annotations
    ASSERT( M_SCENE, !prototype->is_material()
                     || (!return_annotations || return_annotations->get_size() == 0));

    if( !validate_name( name, context))
        return -1;

    if( !validate_expression_list(
        m_transaction,
        defaults,
        /*allow_calls*/ true,
        /*alllow_direct_calls*/ false,
        /*allowed_parameter_count*/ 0,
        /*allowed_temporary_count*/ 0,
        context))
        return -1;

    // upgrade MDL version if necessary (and possible)
    mi::neuraylib::Mdl_version version = get_min_required_mdl_version(
        m_transaction,
        prototype,
        defaults,
        parameter_annotations,
        annotations,
        return_annotations);
    bool is_material_variant = is_variant && prototype->is_material();
    if( is_material_variant && (version < mi::neuraylib::MDL_VERSION_1_4))
        version = mi::neuraylib::MDL_VERSION_1_4;
    upgrade_mdl_version( version, context);
    if( context->get_error_messages_count() > 0)
        return -1;

    // Check that the provided defaults are parameters of the prototype, and that their types match
    // the expected types, and if a cast is necessary (if allowed).
    mi::base::Handle<const IType_list> expected_types( prototype->get_parameter_types());
    std::map<std::string, bool> needs_cast;
    mi::Size n = defaults ? defaults->get_size() : 0;
    for( mi::Size i = 0; i < n; ++i) {

        const char* param_name = defaults->get_name( i);
        mi::base::Handle<const IType> expected_type( expected_types->get_type( param_name));
        if( !expected_type) {
            add_error_message( context,
                STRING::formatted_string( "Invalid default name \"%s\".", param_name), -20);
            return -1;
        }

        mi::base::Handle<const IExpression> default_( defaults->get_expression( i));
        mi::base::Handle<const IType> actual_type( default_->get_type());
        bool needs_cast_tmp = false;
        if( !argument_type_matches_parameter_type(
            m_int_tf.get(),
            actual_type.get(),
            expected_type.get(),
            m_implicit_cast_enabled,
            needs_cast_tmp)) {
            add_error_message( context,
                STRING::formatted_string( "The type of default for \"%s\" does not match the "
                    "expected type.", param_name), -21);
            return -1;
        }
        needs_cast[param_name] = needs_cast_tmp;
    }

    // Check that the provided parameter annotations have matching parameters on the prototype.
    for( mi::Size i = 0; parameter_annotations && i < parameter_annotations->get_size(); ++i) {

        const char* param_name = parameter_annotations->get_name( i);
        mi::base::Handle<const IType> expected_type( expected_types->get_type( param_name));
        if( !expected_type) {
            add_error_message( context,
                STRING::formatted_string( "Invalid parameter annotation name \"%s\".", param_name),
                -22);
            return -1;
        }
    }

    // operators (and field accesses) require special handling
    mi::mdl::IDefinition::Semantics sema = prototype->get_mdl_semantic();
    bool is_operator
        = mi::mdl::semantic_is_operator( sema)
          || sema == mi::mdl::IDefinition::DS_INTRINSIC_DAG_FIELD_ACCESS;

    // variants implies no operator
    ASSERT( M_SCENE, !is_variant || !is_operator);

    // create the call expression for non-operators
    mi::mdl::IExpression_call* call_to_prototype = nullptr;
    if( !is_operator) {
        std::string prototype_name = decode_name_without_signature(
            prototype->get_mdl_name_without_parameter_types());
        prototype_name = remove_qualifiers_if_from_module( prototype_name, m_module->get_name());
        const mi::mdl::IExpression_reference* prototype_ref
            = signature_to_reference( m_module.get(), prototype_name.c_str(), m_name_mangler.get());
        call_to_prototype = m_ef->create_call( prototype_ref);
    }

    // variants implies prototype
    ASSERT( M_SCENE, !is_variant || call_to_prototype);

    // setup the AST builder
    Mdl_ast_builder ast_builder(
       m_module.get(),
       m_transaction,
       /*traverse_ek_parameter*/ true,
       defaults,
       *m_name_mangler,
       /*avoid_resource_urls*/ for_mdle);

    // parameters that we will add to the new method later
    std::vector<const mi::mdl::IParameter*> forwarded_parameters;

    // create arguments for call/variant
    for( mi::Size i = 0, n = prototype->get_parameter_count(); defaults && (i < n); ++i) {

        // get the argument
        const char* param_name = prototype->get_parameter_name( i);
        mi::base::Handle<const IExpression> argument( defaults->get_expression(param_name));
        if( !argument)
            continue;

        // insert cast if necessary
        mi::base::Handle<const IType> expected_type( expected_types->get_type( param_name));
        if( needs_cast[param_name]) {
            mi::Sint32 errors = 0;
            mi::base::Handle<IExpression> cloned_argument(
                m_int_ef->clone( argument.get(), m_transaction, /*copy_immutable_calls*/ false));
            mi::base::Handle<IExpression> casted_argument(
                m_int_ef->create_cast(
                    m_transaction,
                    cloned_argument.get(),
                    expected_type.get(),
                    /*cast_db_name*/ nullptr,
                    /*force_cast*/ false,
                    /*create_direct_call*/ false,
                    &errors));
            argument = casted_argument;
        }

        // convert to mi::mdl and promote to module version number
        const mi::mdl::IType* param_type = prototype->get_mdl_parameter_type(
            m_transaction, static_cast<mi::Uint32>( i));
        const mi::mdl::IExpression* mdl_argument = ast_builder.transform_expr( argument.get());
        mdl_argument = mi::mdl::promote_expressions_to_mdl_version( m_module.get(), mdl_argument);
        if( !mdl_argument) {
            add_error_message( context, STRING::formatted_string( "Failed to promote default for "
                "parameter \"%s\" to requested MDL version.", param_name), -23);
            return -1;
        }
        m_symbol_importer->collect_imports( mdl_argument);

        // create type name
        mi::mdl::IType_name* tn = ast_builder.create_type_name( expected_type);
        m_symbol_importer->collect_imports( tn);

        // create simple name
        const mi::mdl::ISymbol* param_symbol = m_nf->create_symbol( param_name);
        const mi::mdl::ISimple_name* param_simple_name = m_nf->create_simple_name( param_symbol);

        if( is_variant) {

            // add argument to the call expression
            const mi::mdl::IArgument* call_argument
                = m_ef->create_named_argument( param_simple_name, mdl_argument);
            call_to_prototype->add_argument( call_argument); //-V522 PVS

        }  else {

            // create a new parameter that is added to the created function/material
            mi::base::Handle<const IType> actual_type( argument->get_type());
            mi::mdl::IType_name* type_name = ast_builder.create_type_name( actual_type);
            if( param_type->get_type_modifiers() & mi::mdl::IType::MK_UNIFORM)
                type_name->set_qualifier( mi::mdl::FQ_UNIFORM);
            type_name->set_type( param_type);

            // get parameter annotations
            mi::base::Handle<const IAnnotation_block> anno_block(
                parameter_annotations
                ? parameter_annotations->get_annotation_block( param_name) : nullptr);
            mi::mdl::IAnnotation_block* mdl_anno_block = int_anno_block_to_mdl_anno_block(
                anno_block.get(), /*skip_unused*/ true, context);
            if( context->get_error_messages_count() > 0)
                return -1;

            // keep the parameter for adding it later to the created function/material
            forwarded_parameters.push_back( m_df->create_parameter(
                type_name, param_simple_name, mdl_argument, mdl_anno_block));

            // add a reference to the call
            if( !is_operator) {
                const mi::mdl::IExpression_reference* ref = ast_builder.to_reference( param_symbol);
                const mi::mdl::IArgument* call_argument
                    = m_ef->create_named_argument( m_nf->create_simple_name( param_symbol), ref);
                call_to_prototype->add_argument( call_argument); //-V595 PVS
            } else {
                ast_builder.declare_parameter( param_symbol, argument.get());
            }
        }
    }

    // add imports required by arguments
    if( call_to_prototype)
        m_symbol_importer->collect_imports( call_to_prototype);

    // create return type for function/material
    mi::mdl::IType_name* return_type_type_name
        = create_return_type_name( m_transaction, m_module.get(), prototype);
    m_symbol_importer->collect_imports( return_type_type_name);

    // create body for new material/function
    mi::mdl::IStatement* body;
    if( !prototype->is_material() && !is_variant) {
        mi::mdl::IStatement_compound* comp = m_sf->create_compound();
        if( !is_operator) {
           comp->add_statement( m_sf->create_return( call_to_prototype));
        } else {
            mi::base::Handle<const IType> return_type( prototype->get_return_type());
            std::string definition_name = decode_name_without_signature(
                prototype->get_mdl_name_without_parameter_types());
            const mi::mdl::IExpression* return_expr = ast_builder.transform_call(
                return_type.get(),
                sema,
                definition_name,
                defaults ? defaults->get_size() : 0,
                defaults,
                /*named_args*/ true);
            comp->add_statement( m_sf->create_return( return_expr));
        }
        body = comp;
    } else {
        body = m_sf->create_expression( call_to_prototype);
    }

    mi::mdl::IAnnotation_block* mdl_annotations
        = int_anno_block_to_mdl_anno_block( annotations, /*skip_unused*/ true, context);
    if( context->get_error_messages_count() > 0)
        return -1;

    mi::mdl::IAnnotation_block* mdl_return_annotations
        = int_anno_block_to_mdl_anno_block( return_annotations, /*skip_unused*/ true, context);
    if( context->get_error_messages_count() > 0)
        return -1;

    // create new function
    const mi::mdl::ISymbol* symbol = m_nf->create_symbol( name);
    const mi::mdl::ISimple_name* simple_name = m_nf->create_simple_name( symbol);
    mi::mdl::IDeclaration_function* declaration = m_df->create_function(
        return_type_type_name,
        mdl_return_annotations,
        simple_name,
        /*is_clone*/ is_variant,
        body,
        mdl_annotations,
        is_exported);
    if( prototype->is_uniform())
        declaration->set_qualifier( mi::mdl::FQ_UNIFORM);

    // add parameters to the created function/material
    if( !is_variant)
        for( const auto& p : forwarded_parameters)
            declaration->add_parameter( p);

    // add declaration to module
    m_module->add_declaration( declaration);

    // add types/imports/aliases
    m_symbol_importer->add_names( ast_builder.get_used_user_types());
    m_symbol_importer->add_imports();
    m_name_mangler->add_namespace_aliases( m_module.get());

    analyze_module( context);
    if( context->get_error_messages_count() > 0)
        return -1;

    return 0;
}

namespace {

bool is_namespace_alias_legal( mi::mdl::IMDL::MDL_version version)
{
    return version == mi::mdl::IMDL::MDL_VERSION_1_6 || version == mi::mdl::IMDL::MDL_VERSION_1_7;
}

} // namespace

void Mdl_module_builder::upgrade_mdl_version(
    mi::neuraylib::Mdl_version version, Execution_context* context)
{
    mi::mdl::IMDL::MDL_version new_version = convert_mdl_version( version);
    mi::mdl::IMDL::MDL_version current_version
        = static_cast<mi::mdl::Module*>( m_module.get())->get_version();

    // Required version not higher than current version, nothing to do.
    if( new_version <= current_version)
        return;

    // Required version higher than allowed.
    if( new_version > m_max_mdl_version) {
        add_error_message( context,
            STRING::formatted_string(
                "Requires MDL version >= %s, but the maximum MDL version is set to %s.\n",
                stringify_mdl_version( new_version),
                stringify_mdl_version( m_max_mdl_version)),
            -8);
        return;
    }

    // Upgrade module to new version. Module transformer requires at least MDL 1.3.
    Mdl_module_transformer transformer( m_transaction, m_module.get());
    if( new_version < mi::mdl::IMDL::MDL_VERSION_1_3)
        new_version = mi::mdl::IMDL::MDL_VERSION_1_3;
    transformer.upgrade_mdl_version( convert_mdl_version( new_version), context);
    if( context->get_error_messages_count() > 0)
        return;

    // The module transformer is about to be destroyed, no need to serialize the module.
    m_module = const_cast<mi::mdl::IModule*>( transformer.get_module());
    update_module();

    // Re-create name mangler of namespace aliases legality changed.
    if( is_namespace_alias_legal( current_version) != is_namespace_alias_legal( new_version))
        m_name_mangler.reset( new Name_mangler( m_mdl.get(), m_module.get()));
}

mi::mdl::IAnnotation* Mdl_module_builder::int_anno_to_mdl_anno(
    const IAnnotation* annotation, Execution_context* context)
{
    const char* annotation_name = annotation->get_name();
    if( !is_absolute( annotation_name)) {
        add_error_message( context,
            STRING::formatted_string( "Invalid annotation name \"%s\".", annotation_name), -40);
        return nullptr;
    }

    mi::base::Handle<const IAnnotation_definition> definition(
        annotation->get_definition( m_transaction));
    if( !definition) {
        add_error_message( context,
            STRING::formatted_string( "Failed to find annotation \"%s\".", annotation_name), -41);
        return nullptr;
    }

    // get DB and MDL module of the annotation
    std::string module_name = definition->get_mdl_module_name();
    std::string db_module_name = get_db_name( module_name);
    DB::Tag module_tag = m_transaction->name_to_tag( db_module_name.c_str());
    if( !module_tag) {
        add_error_message( context,
            STRING::formatted_string( "Failed to find module of annotation \"%s\".",
                annotation_name), -42);
        return nullptr;
    }
    DB::Access<Mdl_module> db_module( module_tag, m_transaction);
    mi::base::Handle<const mi::mdl::IModule> mdl_module( db_module->get_mdl_module());

    // get parameter types
    mi::Size n = definition->get_parameter_count();
    std::vector<std::string> parameter_type_names;
    parameter_type_names.reserve( n);
    for( mi::Size i = 0; i < n; ++i)
        parameter_type_names.push_back(
            decode_name_without_signature( definition->get_mdl_parameter_type_name( i)));
    std::vector<const char*> parameter_type_names_c_str( n);
    for( mi::Size i = 0; i < n; ++i)
        parameter_type_names_c_str[i] = parameter_type_names[i].c_str();

    // check that there is such an annotation
    std::string annotation_name_without_parameter_types
        = decode_name_without_signature( definition->get_mdl_name_without_parameter_types());
    const mi::mdl::IDefinition* anno_def = mdl_module->find_annotation(
        annotation_name_without_parameter_types.c_str(),
        parameter_type_names_c_str.data(),
        n);
    if( !anno_def) {
        add_error_message( context,
            STRING::formatted_string( "Failed to find internal annotation \"%s\".",
                annotation_name), -43);
        return nullptr;
    }

    const mi::mdl::IQualified_name* anno_qualified_name = signature_to_qualified_name(
        m_nf, annotation_name_without_parameter_types.c_str(), m_name_mangler.get());

    // create annotation
    mi::mdl::IAnnotation* anno = m_af->create_annotation( anno_qualified_name);

    // store parameter types from annotation definition in a map by parameter name
    const mi::mdl::IType* type = anno_def->get_type();
    ASSERT( M_SCENE, type->get_kind() == mi::mdl::IType::TK_FUNCTION);
    const mi::mdl::IType_function* type_function = mi::mdl::as<mi::mdl::IType_function>( type);
    std::map<std::string, const mi::mdl::IType*> parameter_types;
    int parameter_count = type_function->get_parameter_count();
    for( int i = 0; i < parameter_count; ++i) {
        const mi::mdl::IType* parameter_type;
        const mi::mdl::ISymbol* parameter_name;
        type_function->get_parameter( i, parameter_type, parameter_name);
        parameter_types[parameter_name->get_name()] = parameter_type;
    }

    // setup AST builder
    Mdl_ast_builder ast_builder(
        m_module.get(),
        m_transaction,
        /*traverse_ek_parameter*/ false,
        /*args*/ nullptr,
        *m_name_mangler,
        /*avoid_resource_urls*/ false);

    // convert arguments
    mi::base::Handle<const IExpression_list> annotation_args( annotation->get_arguments());
    mi::Size argument_count = annotation_args->get_size();
    for( mi::Size i = 0; i < argument_count; ++i) {

        const char* arg_name = annotation_args->get_name( i);

        mi::base::Handle<const IExpression_constant> arg_expr(
            annotation_args->get_expression<IExpression_constant>( i));
        if( !arg_expr) {
            add_error_message( context,
                STRING::formatted_string( "Invalid expression for annotation argument "
                    "\"%s\".", arg_name), -44);
            return nullptr;
        }

        mi::base::Handle<const IValue> arg_value( arg_expr->get_value());
        mi::base::Handle<const IType> arg_type( arg_value->get_type());

        const mi::mdl::IType* mdl_parameter_type = parameter_types[arg_name];
        if( !mdl_parameter_type) {
            add_error_message( context,
                STRING::formatted_string( "Invalid annotation argument \"%s\".", arg_name), -45);
            return nullptr;
        }
        m_tf->import( mdl_parameter_type);

        const mi::mdl::IExpression* mdl_arg_expr = ast_builder.transform_value( arg_value.get());
        if( !mdl_arg_expr) {
            add_error_message( context,
                STRING::formatted_string( "Unsupported type for argument annotation argument "
                    "\"%s\".", arg_name), -46);
            return nullptr;
        }

        const mi::mdl::ISymbol* arg_symbol = m_nf->create_symbol( arg_name);
        const mi::mdl::ISimple_name* arg_simple_name = m_nf->create_simple_name( arg_symbol);
        const mi::mdl::IArgument* mdl_arg
            = m_ef->create_named_argument( arg_simple_name, mdl_arg_expr);
        anno->add_argument( mdl_arg);
    }

    return anno;
}

mi::mdl::IAnnotation_block* Mdl_module_builder::int_anno_block_to_mdl_anno_block(
    const IAnnotation_block* annotation_block,
    bool skip_anno_unused,
    Execution_context* context)
{
    if( !annotation_block)
        return nullptr;

    mi::mdl::IAnnotation_block* mdl_annotation_block = m_af->create_annotation_block();

    for( mi::Size i = 0; i < annotation_block->get_size(); ++i) {

        mi::base::Handle<const IAnnotation> anno( annotation_block->get_annotation( i));

        const char* anno_name = anno->get_name();
        if( skip_anno_unused && strcmp( anno_name, "::anno::unused()") == 0)
            continue;

        // skip deprecated annotations
        mi::base::Handle<const IAnnotation_definition> definition(
            anno->get_definition( m_transaction));
        const char* anno_simple_name = definition ? definition->get_mdl_simple_name() : nullptr;
        if( anno_simple_name && is_deprecated( anno_simple_name)) {
            add_warning_message( context,
                STRING::formatted_string( "Skipped deprecated annotation \"%s\".",
                decode_for_error_msg( anno_name).c_str()));
            continue;
        }

        mi::mdl::IAnnotation* mdl_anno = int_anno_to_mdl_anno( anno.get(), context);
        if( !mdl_anno)
            add_error_message( context,
                STRING::formatted_string( "Failed to add annotation \"%s\".", anno_name), -47);
        else
            mdl_annotation_block->add_annotation( mdl_anno);
    }

    m_symbol_importer->collect_imports( mdl_annotation_block);
    return mdl_annotation_block;
}

void Mdl_module_builder::create_module( Execution_context* context)
{
    m_module = nullptr;

    // Run sanity checks if module exists already.
    DB::Tag tag = m_transaction->name_to_tag( m_db_module_name.c_str());
    if( tag) {
        SERIAL::Class_id class_id = m_transaction->get_class_id( tag);
        if( class_id != ID_MDL_MODULE) {
            add_error_message(
                context,
                STRING::formatted_string(
                    "The module name \"%s\" is invalid.", m_db_module_name.c_str()),
                -3);
            return;
        }
        DB::Access<Mdl_module> db_module( tag, m_transaction);
        if( !db_module->supports_reload()) {
            add_error_message( context, "Cannot override standard and built-in modules.", -4);
            return;
        }
    }

    // Create module.
    bool ignore_existing = context->get_option<bool>( MDL_CTX_OPTION_DEPRECATED_REPLACE_EXISTING);
    if( tag && !ignore_existing) {

        // Start with existing module.
        DB::Access<Mdl_module> db_module( tag, m_transaction);
        mi::base::Handle<const mi::mdl::IModule> module( db_module->get_mdl_module());
        mi::mdl::Buffer_serializer serializer( m_mdl->get_mdl_allocator());
        m_mdl->serialize_module( module.get(), &serializer, /*include_dependencies*/ false);
        mi::mdl::Buffer_deserializer deserializer(
            m_mdl->get_mdl_allocator(), serializer.get_data(), serializer.get_size());
        m_module = mi::mdl::impl_cast<mi::mdl::Module>( const_cast<mi::mdl::IModule*>(
            m_mdl->deserialize_module( &deserializer)));

    } else {

        // Start with new empty module.
        m_module = m_mdl->create_module(
            m_thread_context.get(), m_mdl_module_name.c_str(), m_min_mdl_version);
        if( !m_module) {
            convert_messages( m_thread_context->access_messages(), context);
            add_error_message( context,
                STRING::formatted_string( "Failed to create the empty module \"%s\".",
                m_db_module_name.c_str()), -5);
            return;
        }
        m_module->analyze( /*module_cache*/ nullptr, m_thread_context.get());
        if( !m_module->is_valid()) {
            convert_messages( m_thread_context->access_messages(), context);
            add_error_message( context,
                STRING::formatted_string( "Failed to analyze the empty module \"%s\".",
                    m_db_module_name.c_str()), -6);
            return;
        }
    }

    update_module();
}

void Mdl_module_builder::analyze_module( Execution_context* context)
{
    // Note that the AST dump is not guaranteed to be valid MDL (even for valid modules), e.g., it
    // generates empty selector strings for MDL < 1.7.
    SYSTEM::Access_module<CONFIG::Config_module> config_module( false);
    const CONFIG::Config_registry& registry = config_module->get_configuration();
    bool dump_ast = false;
    registry.get_value( "mdl_dump_ast_in_module_builder", dump_ast);
    if( dump_ast)
        mi::mdl::dump_ast( m_module.get());

    ASSERT( M_SCENE, mi::mdl::Tree_checker::check( m_mdl.get(), m_module.get(), /*verbose*/ false));

    // replace all user defined constants
    User_constant_remover uc_remover(m_module.get());
    uc_remover.process();

    // analyze module
    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module( false);
    Module_cache module_cache( m_transaction, mdlc_module->get_module_wait_queue(), {});
    m_module->analyze( &module_cache, m_thread_context.get());
    if( !m_module->is_valid()) {
        convert_messages( m_module->access_messages(), context);
        add_error_message( context,
            STRING::formatted_string( "Failed to analyze created module \"%s\".",
                m_module->get_name()), -15);
        create_module( context);
        return;
    }

    // inline MDLEs
    if( m_symbol_importer->imports_mdle()) {
        mi::base::Handle<mi::mdl::IMDL_module_transformer> module_transformer(
            m_mdl->create_module_transformer());
        mi::base::Handle<mi::mdl::IModule> inlined_module(
            module_transformer->inline_mdle( m_module.get()));
        if( !inlined_module) {
            convert_messages( module_transformer->access_messages(), context);
            add_error_message( context,
                STRING::formatted_string( "Failed to inline MDLEs into \"%s\".",
                    m_module->get_name()), -16);
            create_module( context);
            return;
        }
        m_module = inlined_module;
    }

    if( !m_export_to_db) {
        update_module();
        return;
    }

    // export to DB
    std::string db_module_name = get_db_name( m_module->get_name());
    DB::Tag db_module_tag = m_transaction->name_to_tag( db_module_name.c_str());
    if( db_module_tag) {
        DB::Edit<Mdl_module> db_module_edit( db_module_tag, m_transaction);
        db_module_edit->reload_module_internal(
            m_transaction, m_mdl.get(), m_module.get(), context);
        if( context->get_error_messages_count() > 0) {
            create_module( context);
            return;
        }
    } else {
        mi::Sint32 result = Mdl_module::create_module_internal(
            m_transaction, m_mdl.get(), m_module.get(), context);
        if( result < 0) {
            create_module( context);
            return;
        }
    }

    // clone module and reinitialize dependent members
    mi::mdl::Buffer_serializer serializer( m_mdl->get_mdl_allocator());
    m_mdl->serialize_module( m_module.get(), &serializer, /*include_dependencies*/ false);
    mi::mdl::Buffer_deserializer deserializer(
        m_mdl->get_mdl_allocator(), serializer.get_data(), serializer.get_size());
    m_module = const_cast<mi::mdl::IModule*>( m_mdl->deserialize_module( &deserializer));

    update_module();
}

void Mdl_module_builder::update_module()
{
    if( !m_module)
        return;

    m_symbol_importer.reset( new Symbol_importer( m_module.get()));

    m_af = m_module->get_annotation_factory();
    m_df = m_module->get_declaration_factory();
    m_ef = m_module->get_expression_factory();
    m_nf = m_module->get_name_factory();
    m_sf = m_module->get_statement_factory();
    m_tf = m_module->get_type_factory();
    m_vf = m_module->get_value_factory();
}

bool Mdl_module_builder::validate_name( const char* name, Execution_context* context)
{
    if( !m_mdl->is_valid_mdl_identifier( name)) {
        add_error_message( context,
            STRING::formatted_string( "Invalid name \"%s\".", name), -18);
        return false;
    }

    return true;
}

bool Mdl_module_builder::validate_expression(
     DB::Transaction* transaction,
     const IExpression* expr,
     bool allow_calls,
     bool allow_direct_calls,
     mi::Size allowed_parameter_count,
     mi::Size allowed_temporary_count,
     Execution_context* context)
{
    ASSERT( M_SCENE, expr);

    switch( expr->get_kind()) {

        case IExpression::EK_CONSTANT:
            return true;

        case IExpression::EK_PARAMETER: {
            mi::base::Handle<const IExpression_parameter> parameter(
                expr->get_interface<IExpression_parameter>());
            mi::Size index = parameter->get_index();
            if( index >= allowed_parameter_count) {
                add_error_message( context,
                    STRING::formatted_string( "Infeasible parameter reference with index %zu.",
                        index), -14);
                return false;
            }
            return true;
        }

        case IExpression::EK_TEMPORARY: {
            mi::base::Handle<const IExpression_temporary> temporary(
                expr->get_interface<IExpression_temporary>());
            mi::Size index = temporary->get_index();
            if( index >= allowed_temporary_count) {
                add_error_message( context,
                    STRING::formatted_string( "Infeasible temporary reference with index %zu.",
                        index), -14);
                return false;
            }
            return true;
        }

        case IExpression::EK_CALL: {
            if( !allow_calls) {
                add_error_message( context,
                    STRING::formatted_string( "Infeasible call expression."),
                    -14);
                return false;
            }

            mi::base::Handle<const IExpression_call> call(
                expr->get_interface<IExpression_call>());
            DB::Tag tag = call->get_call();
            SERIAL::Class_id class_id = transaction->get_class_id( tag);
            if( class_id != ID_MDL_FUNCTION_CALL) {
                const char* name = transaction->tag_to_name( tag);
                add_error_message( context,
                    STRING::formatted_string( "Invalid reference to DB element \"%s\" in call "
                        "expression.", name),
                    -14);
               return false;
            }

            DB::Access<Mdl_function_call> function_call( tag, transaction);
            mi::base::Handle<const IExpression_list> args(
                function_call->get_arguments());

            return validate_expression_list(
                transaction,
                args.get(),
                allow_calls,
                allow_direct_calls,
                allowed_parameter_count,
                allowed_temporary_count,
                context);
        }

        case IExpression::EK_DIRECT_CALL: {
            if( !allow_direct_calls) {
                add_error_message( context,
                    STRING::formatted_string( "Invalid direct call expression."),
                    -14);
                return false;
            }

            mi::base::Handle<const IExpression_direct_call> direct_call(
                expr->get_interface<IExpression_direct_call>());
            DB::Tag tag = direct_call->get_definition( m_transaction);
            SERIAL::Class_id class_id = m_transaction->get_class_id( tag);
            if( class_id != ID_MDL_FUNCTION_DEFINITION) {
                const char* name = transaction->tag_to_name( tag);
                add_error_message( context,
                    STRING::formatted_string( "Invalid reference to DB element \"%s\" in direct "
                        "call expression.", name),
                    -14);
               return false;
            }

            mi::base::Handle<const IExpression_list> args( direct_call->get_arguments());
            return validate_expression_list(
                transaction,
                args.get(),
                allow_calls,
                allow_direct_calls,
                allowed_parameter_count,
                allowed_temporary_count,
                context);
         }

         case IExpression::EK_FORCE_32_BIT: {
            ASSERT( M_SCENE, false);
            return false;
        }
    }

    ASSERT( M_SCENE, false);
    return false;
}

bool Mdl_module_builder::validate_expression_list(
     DB::Transaction* transaction,
     const IExpression_list* expr_list,
     bool allow_calls,
     bool allow_direct_calls,
     mi::Size allowed_parameter_count,
     mi::Size allowed_temporary_count,
     Execution_context* context)
{
    ASSERT( M_SCENE, expr_list);

    for( mi::Size i = 0, n = expr_list->get_size(); i < n; ++i) {
        mi::base::Handle<const IExpression> expr( expr_list->get_expression( i));
        if( !validate_expression(
            transaction,
            expr.get(),
            allow_calls,
            allow_direct_calls,
            allowed_parameter_count,
            allowed_temporary_count,
            context))
            return false;
    }

    return true;
}

namespace {

// Type of the queue entries.
struct Entry
{
    Entry(
        const mi::base::Handle<const IExpression>& expr,
        bool is_uniform,
        const std::string& path)
      : m_expr( expr), m_is_uniform( is_uniform), m_path( path) { }

    mi::base::Handle<const IExpression> m_expr;
    bool m_is_uniform;
    std::string m_path;
};

std::string get_path(
    const std::string& prefix, const std::string& element, bool is_array_constructor)
{
    std::string result = prefix;
    if( is_array_constructor)
        result += '[';
    else if( !result.empty())
        result += '.';
    result += element;
    if( is_array_constructor)
        result += ']';
    return result;
}

// Adds arguments to the queue for analyze_uniform().
void analyze_uniform_args(
    std::queue<Entry>& queue,
    bool ternary_condition_is_uniform,
    const IType_list* parameter_types,
    const IExpression_list* arguments,
    mi::mdl::IDefinition::Semantics sema,
    bool auto_must_be_uniform,
    const std::string& path)
{
    bool is_array_constructor
        = sema == mi::mdl::IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR;
    bool is_ternary_operator
        = sema == mi::mdl::operator_to_semantic( mi::mdl::IExpression::OK_TERNARY);
    bool is_cast_operator
        = sema == mi::mdl::operator_to_semantic( mi::mdl::IExpression::OK_CAST);

    for( mi::Size i = 0, n = arguments->get_size(); i < n; ++i) {

        mi::base::Handle<const IExpression> arg( arguments->get_expression( i));
        const char* name = arguments->get_name( i);
        std::string new_path = get_path( path, name, is_array_constructor);

        if( is_array_constructor || is_cast_operator)
            queue.emplace( arg, auto_must_be_uniform, new_path);
        else {

            mi::base::Handle<const IType> parameter_type( parameter_types->get_type( i));
            mi::Uint32 modifiers = parameter_type->get_all_type_modifiers();
            bool p_is_uniform = (modifiers & IType::MK_UNIFORM) != 0;
            bool p_is_varying = (modifiers & IType::MK_VARYING) != 0;

            if( ternary_condition_is_uniform && is_ternary_operator && i == 0) {
                p_is_uniform = true;
                p_is_varying = false;
            }

            queue.emplace(
                arg, p_is_uniform || (!p_is_varying && auto_must_be_uniform), new_path);
        }
    }
}

} // namespace

void Mdl_module_builder::analyze_uniform(
    DB::Transaction* transaction,
    const IExpression* root_expr,
    bool root_expr_uniform,
    const IExpression* query_expr,
    std::vector<bool>& uniform_parameters,
    bool& uniform_query_expr,
    std::string& error_path,
    Execution_context* context)
{
    uniform_parameters.clear();
    uniform_query_expr = false;
    error_path.clear();

    // Determine whether the condition of the ternary operator needs to be uniform.
    mi::base::Handle<const IType> root_type( root_expr->get_type());
    root_type = root_type->skip_all_type_aliases();
    mi::base::Handle<const IType_df> root_type_df( root_type->get_interface<IType_df>());
    mi::base::Handle<const IType_struct> root_type_struct(
        root_type->get_interface<IType_struct>());
    IType_struct::Predefined_id root_type_struct_id
        = root_type_struct ? root_type_struct->get_predefined_id() :  IType_struct::SID_USER;
    bool ternary_condition_is_uniform
        = root_type_df
       || (root_type_struct_id == IType_struct::SID_MATERIAL_EMISSION)
       || (root_type_struct_id == IType_struct::SID_MATERIAL_SURFACE)
       || (root_type_struct_id == IType_struct::SID_MATERIAL_VOLUME)
       || (root_type_struct_id == IType_struct::SID_MATERIAL_GEOMETRY);

    std::queue<Entry> queue;

    queue.emplace( make_handle_dup( root_expr), root_expr_uniform, std::string());

    // Note that nodes might be visited several times. uniform_parameters and uniform_query_expr
    // never change their value(s) from true to false.
    while( !queue.empty()) {

        const Entry& e = queue.front();
        mi::base::Handle<const IExpression> expr = e.m_expr;
        bool expr_is_uniform = e.m_is_uniform;
        std::string path = e.m_path;
        queue.pop();

        mi::base::Handle<const IType> type( expr->get_type());
        mi::base::Handle<const IType_resource> resource_type(
            type->get_interface<IType_resource>());
        bool expr_has_resource_type = !!resource_type;

        // Record analysis result for query expression.
        if( expr.get() == query_expr && (expr_is_uniform || expr_has_resource_type))
            uniform_query_expr = true;

        switch( expr->get_kind()) {

            case IExpression::EK_CONSTANT:
                // Constants are always uniform.
                break;

            case IExpression::EK_PARAMETER: {
                // Record analysis result for parameter references.
                mi::base::Handle<const IExpression_parameter> parameter(
                    expr->get_interface<IExpression_parameter>());
                mi::Size index = parameter->get_index();
                mi::Size new_size = std::max(
                    index+1, static_cast<mi::Size>( uniform_parameters.size()));
                uniform_parameters.resize( new_size, false);
                if( expr_is_uniform || expr_has_resource_type)
                    uniform_parameters[index] = true;
                break;
            }

            case IExpression::EK_TEMPORARY: {
                add_error_message( context, "Infeasible temporary reference.", -50);
                return;
            }

            case IExpression::EK_CALL: {
                mi::base::Handle<const IExpression_call> call(
                    expr->get_interface<IExpression_call>());
                DB::Tag tag = call->get_call();
                SERIAL::Class_id class_id = transaction->get_class_id( tag);
                const char* name = transaction->tag_to_name( tag);

                if( class_id != Mdl_function_call::id) {
                     add_error_message( context, STRING::formatted_string(
                         "Invalid reference to DB element \"%s\" in call expression.", name), -54);
                    return;
                }

                DB::Access<Mdl_function_call> function_call( tag, transaction);
                DB::Tag def_tag = function_call->get_function_definition( transaction);
                if( !def_tag) {
                    add_error_message( context, STRING::formatted_string(
                        "Invalid function call \"%s\".", name), -52);
                    return;
                }
                DB::Access<Mdl_function_definition> def( def_tag, transaction);

                // Determine whether auto parameters must be uniform.
                bool auto_must_be_uniform = false;
                if( expr_is_uniform) {
                    mi::base::Handle<const IType> return_type( def->get_return_type());
                    if( return_type->get_all_type_modifiers() & IType::MK_UNIFORM) {
                        // The return type is uniform, no need to enforce auto parameters as
                        // uniform.
                        auto_must_be_uniform = false;
                    } else if( !def->is_uniform()) {
                        // The called function is not uniform.
                        add_error_message( context, STRING::formatted_string(
                            "Function call \"%s\" at node \"%s\" needs to be uniform, but is "
                            "not.", name, path.c_str()), -53);
                        error_path = path;
                        return;
                    } else {
                        // The function is uniform and the result must be uniform, enforce auto
                        // parameters as uniform.
                        auto_must_be_uniform = true;
                    }
                }

                // Add all arguments to the queue.
                mi::base::Handle<const IType_list> parameter_types(
                    def->get_parameter_types());
                mi::base::Handle<const IExpression_list> args( function_call->get_arguments());
                mi::mdl::IDefinition::Semantics sema = def->get_mdl_semantic();
                analyze_uniform_args(
                    queue,
                    ternary_condition_is_uniform,
                    parameter_types.get(),
                    args.get(),
                    sema,
                    auto_must_be_uniform,
                    path);

                break;
            }

            case IExpression::EK_DIRECT_CALL: {
                mi::base::Handle<const IExpression_direct_call> direct_call(
                    expr->get_interface<IExpression_direct_call>());
                DB::Tag def_tag = direct_call->get_definition( transaction);
                SERIAL::Class_id class_id = transaction->get_class_id( def_tag);
                const char* name = transaction->tag_to_name( def_tag);

                if( class_id != Mdl_function_definition::id) {
                     add_error_message( context, STRING::formatted_string(
                         "Invalid reference to DB element \"%s\" in direct call expression.", name),
                         -57);
                    return;
                }

                DB::Access<Mdl_function_definition> def( def_tag, transaction);

                // Determine whether auto parameters must be uniform.
                bool auto_must_be_uniform = false;
                if( expr_is_uniform) {
                    mi::base::Handle<const IType> return_type( def->get_return_type());
                    if( return_type->get_all_type_modifiers() & IType::MK_UNIFORM) {
                        // The return type is uniform, no need to enforce auto parameters as
                        // uniform.
                        auto_must_be_uniform = false;
                    } else if( !def->is_uniform()) {
                        // The called function is not uniform.
                        add_error_message( context, STRING::formatted_string(
                            "Direct call to function definition \"%s\" at node \"%s\" needs to "
                            "be uniform, but is not.", name, path.c_str()), -56);
                        error_path = path;
                        return;
                    } else {
                        // The function is uniform and the result must be uniform, enforce auto
                        // parameters as uniform.
                        auto_must_be_uniform = true;
                    }
                }

                // Add all arguments to the queue.
                mi::base::Handle<const IType_list> parameter_types(
                    def->get_parameter_types());
                mi::base::Handle<const IExpression_list> args( direct_call->get_arguments());
                mi::mdl::IDefinition::Semantics sema = def->get_mdl_semantic();
                analyze_uniform_args(
                    queue,
                    ternary_condition_is_uniform,
                    parameter_types.get(),
                    args.get(),
                    sema,
                    auto_must_be_uniform,
                    path);

                break;
            }

            case IExpression::EK_FORCE_32_BIT:
                ASSERT( M_SCENE, false);
                return;
        }
    }
}

} // namespace MDL

} // namespace MI
