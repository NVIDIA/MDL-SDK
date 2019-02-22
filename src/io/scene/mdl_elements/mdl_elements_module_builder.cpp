/***************************************************************************************************
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "i_mdl_elements_module.h"
#include "i_mdl_elements_utilities.h"
#include "i_mdl_elements_function_call.h"
#include "i_mdl_elements_function_definition.h"
#include "i_mdl_elements_material_definition.h"
#include "i_mdl_elements_material_instance.h"
#include "mdl_elements_utilities.h"
#include <base/data/db/i_db_access.h>
#include <base/data/db/i_db_transaction.h>
#include <base/system/main/access_module.h>
#include <base/util/string_utils/i_string_utils.h>
#include <io/scene/mdl_elements/mdl_elements_ast_builder.h>
#include <mdl/integration/mdlnr/i_mdlnr.h>

#include <mdl/compiler/compilercore/compilercore_modules.h>
#include <mdl/compiler/compilercore/compilercore_tools.h>

namespace MI {

namespace MDL {

Mdl_module_builder::Mdl_module_builder(
    mi::mdl::IMDL* imdl,
    MI::DB::Transaction* transaction,
    const char* module_name,
    mi::mdl::IMDL::MDL_version version,
    MI::MDL::Execution_context* context)
: m_mdl(mi::base::make_handle_dup(imdl))
, m_transaction(transaction)
, m_thread_context(nullptr)
, m_module(nullptr)
, m_symbol_importer(nullptr)
, m_added_functions(imdl->get_mdl_allocator())
, m_added_function_annotations(imdl->get_mdl_allocator())
, m_added_function_symbols(imdl->get_mdl_allocator())
, m_annotation_factory(nullptr)
, m_declaration_factory(nullptr)
, m_expression_factory(nullptr)
, m_name_factory(nullptr)
, m_statement_factory(nullptr)
, m_type_factory(nullptr)
, m_value_factory(nullptr)
{
    ASSERT(M_SCENE, context);

    context->clear_messages();

    // Reject invalid module names (in particular, names containing slashes and backslashes).
    if (!MI::MDL::Mdl_module::is_valid_module_name(module_name, m_mdl.get()))
    {
        const MI::MDL::Message message(
            MI::MDL::Message(
            mi::base::MESSAGE_SEVERITY_ERROR,
            "The module name module_name is invalid.",
            -1, MI::MDL::Message::MSG_INTEGRATION));
        
        context->set_result(-1);
        context->add_message(message);
        context->add_error_message(message); 
        return;
    }

    // Check whether the module exists already in the DB.
    std::string db_module_name = MI::MDL::add_mdl_db_prefix(module_name);
    DB::Tag db_module_tag = transaction->name_to_tag(db_module_name.c_str());
    if (db_module_tag)
    {
        if (transaction->get_class_id(db_module_tag) != Mdl_module::id)
        {
            const MI::MDL::Message message(
                MI::MDL::Message(mi::base::MESSAGE_SEVERITY_ERROR,
                "The DB name for an imported module is already in use but is "
                "not an MDL module, or the DB name for a definition in this "
                "module is already in use.",
                -3, MI::MDL::Message::MSG_INTEGRATION));
            
            context->set_result(-3);
            context->add_message(message);
            context->add_error_message(message);
            return;
        }

        // TODO assuming that this is an error here. check back.
        context->set_result(-1);
        context->add_message(MI::MDL::Message(mi::base::MESSAGE_SEVERITY_WARNING,
                             "Module exists already.",
                             -1, MI::MDL::Message::MSG_INTEGRATION));
        return; // 1
    }

    // Create module
    m_thread_context = m_mdl->create_thread_context();

    const char* enable_experimental = context->get_option<bool>(MDL_CTX_OPTION_EXPERIMENTAL)
        ? "true" : "false";
    m_thread_context->access_options().set_option(
        MDL_OPTION_EXPERIMENTAL_FEATURES, enable_experimental);

    m_module = m_mdl->create_module(m_thread_context.get(), module_name, version);
    if (!m_module)
    {
        report_messages(m_thread_context->access_messages(), context);

        // TODO is this the correct error code here? check back.
        const MI::MDL::Message message(
            MI::MDL::Message(mi::base::MESSAGE_SEVERITY_ERROR,
            " Failed to compile the module module_name",
            -2, MI::MDL::Message::MSG_INTEGRATION));

        context->set_result(-2);
        context->add_message(message);
        context->add_error_message(message);
        return;
    }

    m_symbol_importer = new Symbol_importer(m_module.get());


    // keep all required factories as members
    m_annotation_factory = m_module->get_annotation_factory();
    m_declaration_factory = m_module->get_declaration_factory();
    m_expression_factory = m_module->get_expression_factory();
    m_name_factory = m_module->get_name_factory();
    m_statement_factory = m_module->get_statement_factory();
    m_type_factory = m_module->get_type_factory();
    m_value_factory = m_module->get_value_factory();
}

Mdl_module_builder::~Mdl_module_builder()
{
    if (m_symbol_importer)
        delete m_symbol_importer;
}

namespace
{
    // create return type for variant
    template <class T>
    static mi::mdl::IType_name* create_return_type_name(
        DB::Transaction* transaction,
        mi::mdl::IModule* module,
        DB::Access<T> prototype);

    template <>
    mi::mdl::IType_name* create_return_type_name(
        DB::Transaction* transaction,
        mi::mdl::IModule* module,
        DB::Access<Mdl_material_definition> prototype)
    {
        mi::mdl::IName_factory &nf = *module->get_name_factory();
        const mi::mdl::ISymbol* return_type_symbol = nf.create_symbol("material");
        const mi::mdl::ISimple_name* return_type_simple_name
            = nf.create_simple_name(return_type_symbol);
        mi::mdl::IQualified_name* return_type_qualified_name = nf.create_qualified_name();
        return_type_qualified_name->add_component(return_type_simple_name);
        return nf.create_type_name(return_type_qualified_name);
    }

    template <>
     mi::mdl::IType_name* create_return_type_name(
        DB::Transaction* transaction,
        mi::mdl::IModule* module,
        DB::Access<Mdl_function_definition> prototype)
    {
        const mi::mdl::IType* ret_type = prototype->get_mdl_return_type(transaction);
        return mi::mdl::create_type_name(ret_type, module);
    }
}   // anonymous


template <typename T>
bool has_uniform_qualifier(T const *) { return false; }

template <>
bool has_uniform_qualifier(Mdl_function_definition const* func)
{
    return func->is_uniform();
}

template <typename T>
bool has_uniform_return_qualifier(T const *) { return false; }

template <>
bool has_uniform_return_qualifier(Mdl_function_definition const* func)
{
    return (func->get_return_type()->get_all_type_modifiers() & MI::MDL::IType::MK_UNIFORM) != 0;
}


template <typename T>
mi::Sint32 Mdl_module_builder::add_intern(
    MI::DB::Tag prototype_tag,
    const char* simple_name,
    const MI::MDL::IExpression_list* defaults,
    bool is_variant,
    bool is_exported,
    MI::MDL::Execution_context* context)
{
    MI::SERIAL::Class_id id = m_transaction->get_class_id(prototype_tag);
    ASSERT(M_SCENE, id == T::id);
    bool treat_as_preset = is_variant || (id == ID_MDL_FUNCTION_DEFINITION);

    DB::Access<T> prototype(prototype_tag, m_transaction);

    // check that the provided arguments are parameters of the material definition
    // and that their types match the expected types
    mi::base::Handle<const IType_list> expected_types(prototype->get_parameter_types());
    for (mi::Size i = 0; defaults && i < defaults->get_size(); ++i)
    {
        const char* param_name = defaults->get_name(i);
        mi::base::Handle<const IType> expected_type(expected_types->get_type(param_name));
        if (!expected_type)
        {
            const MI::MDL::Message message(
                MI::MDL::Message(mi::base::MESSAGE_SEVERITY_ERROR,
                "A default for a non-existing parameter was provided.",
                -6, MI::MDL::Message::MSG_INTEGRATION));

            context->set_result(-6);
            context->add_message(message);
            context->add_error_message(message);
            return -1;
        }
        mi::base::Handle<const IExpression> argument(defaults->get_expression(i));
        mi::base::Handle<const IType> actual_type(argument->get_type());
        mi::base::Handle<IType_factory> tf(get_type_factory());
        if (!argument_type_matches_parameter_type(tf.get(), actual_type.get(), expected_type.get()))
        {
            const MI::MDL::Message message(MI::MDL::Message(mi::base::MESSAGE_SEVERITY_ERROR,
                 "The type of a default does not match the expected/correct type.",
                 -7, MI::MDL::Message::MSG_INTEGRATION));

            context->set_result(-7);
            context->add_message(message);
            context->add_error_message(message);
            return -1;
        }
    }

    // create call expression
    const char* prototype_name = prototype->get_mdl_name();
    const mi::mdl::IExpression_reference* prototype_ref
        = signature_to_reference(m_module.get(), prototype_name);
    mi::mdl::IExpression_call* call_to_prototype = m_expression_factory->create_call(prototype_ref);

    // setup the builder
    mi::base::Handle<MI::MDL::IExpression_list const> arguments_list_handle =
        mi::base::make_handle_dup(defaults);
    MI::MDL::Mdl_ast_builder ast_builder(m_module.get(), m_transaction, arguments_list_handle);

    // parameters that we will add to the new method later
    std::vector<const mi::mdl::IParameter*> forwarded_parameters;

    // create arguments for call/variant
    if (defaults != NULL)
    {
        mi::base::Handle<const MI::MDL::IAnnotation_list> param_annotations(
            prototype->get_parameter_annotations());

        for (mi::Size i = 0, n = prototype->get_parameter_count(); i < n; ++i)
        {
            // get the MI::MDL argument
            const char* param_name = prototype->get_parameter_name(i);
            mi::base::Handle<const IExpression> argument(defaults->get_expression(param_name));
            if (!argument)
                continue;

            // convert to mi::mdl and promote to module version number
            const mi::mdl::IType* param_type = prototype->get_mdl_parameter_type(
                m_transaction, static_cast<mi::Uint32>(i));
            const mi::mdl::IExpression* arg_expr = int_expr_to_mdl_ast_expr(
                m_transaction, m_module.get(), param_type, argument.get());
            arg_expr = promote_expressions_to_mdl_version(m_module.get(), arg_expr);
            if (!arg_expr)
            {
                const MI::MDL::Message message(
                    MI::MDL::Message(mi::base::MESSAGE_SEVERITY_ERROR,
                    "Unspecified error. Failed to promote expression.",
                    -8, MI::MDL::Message::MSG_INTEGRATION));

                context->set_result(-8);
                context->add_message(message);
                context->add_error_message(message);
                return -1;
            }

            // collect imports for that argument
            m_symbol_importer->collect_imports(arg_expr);

            const mi::mdl::ISymbol* param_symbol = m_name_factory->create_symbol(param_name);
            const mi::mdl::ISimple_name* param_simple_name = 
                m_name_factory->create_simple_name(param_symbol);

            if (treat_as_preset)
            {
                const mi::mdl::IArgument* argument =
                    m_expression_factory->create_named_argument(param_simple_name, arg_expr);
                call_to_prototype->add_argument(argument);
            }
            else
            {
                // create a new parameter that is added to the created function/material
                ast_builder.declare_parameter(param_symbol, argument);
                mi::base::Handle<const IType> argument_type(argument->get_type());
                mi::mdl::IType_name* type_name = ast_builder.create_type_name(argument_type);
                if (param_type->get_type_modifiers() & mi::mdl::IType::MK_UNIFORM)
                    type_name->set_qualifier(mi::mdl::FQ_UNIFORM);
                type_name->set_type(param_type);

                // get parameter annotations
                mi::base::Handle<const IAnnotation_block> annotations(
                    param_annotations->get_annotation_block(param_name));
                mi::mdl::IAnnotation_block* mdl_annotations = 0;
                if (!create_annotations(annotations.get(), mdl_annotations, context))
                    return -1;

                // keep the parameter for adding it later to the created function/material
                forwarded_parameters.push_back(m_declaration_factory->create_parameter(
                    type_name, param_simple_name, arg_expr, mdl_annotations));

                // add a reference to the call
                const mi::mdl::IExpression_reference* ref = ast_builder.to_reference(param_symbol);
                const mi::mdl::IArgument* call_argument =
                    m_expression_factory->create_named_argument(param_simple_name, ref);
                call_to_prototype->add_argument(call_argument);
            }
        }
    }

    // add imports required by arguments
    m_symbol_importer->collect_imports(call_to_prototype);

    // create return type for variant
    mi::mdl::IType_name* return_type_type_name
        = create_return_type_name(m_transaction, m_module.get(), prototype);

    // return type modifier
    if (has_uniform_return_qualifier(prototype.get_ptr()))
        return_type_type_name->set_qualifier(mi::mdl::FQ_UNIFORM);

    // create body for variant
    const mi::mdl::IStatement_expression* variant_body
        = m_statement_factory->create_expression(call_to_prototype);

    // create new function to add
    const mi::mdl::ISymbol* symbol_to_add = m_name_factory->create_symbol(simple_name);
    const mi::mdl::ISimple_name* simple_name_to_add = 
        m_name_factory->create_simple_name(symbol_to_add);

    mi::mdl::IDeclaration_function* declaration_to_add = m_declaration_factory->create_function(
        return_type_type_name, /*ret_annotations*/ 0, simple_name_to_add, 
        /*is_clone*/ treat_as_preset, variant_body, /*annotation*/ nullptr, is_exported);

    // function modifier
    if (has_uniform_qualifier(prototype.get_ptr()))
        declaration_to_add->set_qualifier(mi::mdl::FQ_UNIFORM);

    // add parameters to the created function/material
    if (!treat_as_preset)
        for (auto&& p : forwarded_parameters)
            declaration_to_add->add_parameter(p);

    // defer the actual adding to the module until the module is build
    m_added_functions.push_back(declaration_to_add);
    m_added_function_annotations.push_back(m_annotation_factory->create_annotation_block());
    m_added_function_symbols.push_back(symbol_to_add);
    return m_added_functions.size() - 1; // get index
}

mi::Sint32 Mdl_module_builder::add_variant_intern(
    mi::Sint32 index,
    const char* simple_name,
    const MI::MDL::IExpression_list* defaults,
    bool is_exported,
    MI::MDL::Execution_context* context)
{
    if (index >= static_cast<mi::Sint32>(m_added_functions.size()))
        return false;

    // get prototype
    mi::mdl::IDeclaration_function* prototype_declaration = m_added_functions[index];
    const mi::mdl::ISymbol* protoype_symbol = m_added_function_symbols[index];

    // lookup table for parameters by name
    std::unordered_map<std::string, const mi::mdl::IParameter*> prototype_params;
    for (size_t i = 0, n = prototype_declaration->get_parameter_count(); i < n; ++i) {
        const mi::mdl::IParameter* param = prototype_declaration->get_parameter(i);
        const char* name =  param->get_name()->get_symbol()->get_name();
        prototype_params[name] = param;
    }

    // check that the provided arguments are parameters of the material definition
    // and that their types match the expected types
    for (mi::Size i = 0; defaults && i < defaults->get_size(); ++i) {
        const char* param_name = defaults->get_name(i);
        auto param = prototype_params.find(param_name);
        if (param == prototype_params.end()) {
            const MI::MDL::Message message(
                MI::MDL::Message(mi::base::MESSAGE_SEVERITY_ERROR,
                "A default for a non-existing parameter was provided.",
                -6, MI::MDL::Message::MSG_INTEGRATION));

            context->set_result(-6);                
            context->add_message(message);
            context->add_error_message(message);
            return -1;
        }
        mi::base::Handle<const IExpression> argument(defaults->get_expression(i));
        mi::base::Handle<const IType> actual_type(argument->get_type());
        mi::base::Handle<IType_factory> tf(get_type_factory());

        mi::base::Handle<const MI::MDL::IType> expected_type(
            mdl_type_to_int_type(tf.get(), param->second->get_type_name()->get_type()));

        if (!argument_type_matches_parameter_type(tf.get(), 
            actual_type.get(), expected_type.get())) {
            const MI::MDL::Message message(
                MI::MDL::Message(mi::base::MESSAGE_SEVERITY_ERROR,
                "The type of a default does not have the correct type.",
                -7, MI::MDL::Message::MSG_INTEGRATION));
            
            context->set_result(-7);
            context->add_message(message);
            context->add_error_message(message);
            return -1;
        }
    }


    // create a call to the prototype
    mi::mdl::ISimple_name const *sname = m_name_factory->create_simple_name(protoype_symbol);
    mi::mdl::IQualified_name *qname = m_name_factory->create_qualified_name();
    qname->add_component(sname);
    mi::mdl::IType_name *tn = m_name_factory->create_type_name(qname);
    mi::mdl::IExpression_reference *prototype_ref = m_expression_factory->create_reference(tn);
    mi::mdl::IExpression_call* call_to_prototype = m_expression_factory->create_call(prototype_ref);

    // create arguments for call/variant
    if (defaults != NULL) {
        for (auto&& param : prototype_params) {

            // get the MI::MDL argument
            mi::base::Handle<const IExpression> argument(
                defaults->get_expression(param.first.c_str()));
            if (!argument)
                continue;

            // convert to mi::mdl and promote to module version number
            const mi::mdl::IType* param_type = param.second->get_type_name()->get_type();
            const mi::mdl::IExpression* arg_expr = int_expr_to_mdl_ast_expr(
                m_transaction, m_module.get(), param_type, argument.get());
            arg_expr = promote_expressions_to_mdl_version(m_module.get(), arg_expr);
            if (!arg_expr)
            {
                const MI::MDL::Message message(
                    MI::MDL::Message(mi::base::MESSAGE_SEVERITY_ERROR,
                    "Unspecified error. Failed to promote expression.",
                    -8, MI::MDL::Message::MSG_INTEGRATION));

                context->set_result(-8);
                context->add_message(message);
                context->add_error_message(message);
                return -1;
            }

            // collect imports for that argument
            m_symbol_importer->collect_imports(arg_expr);

            const mi::mdl::ISymbol* param_symbol = 
                m_name_factory->create_symbol(param.first.c_str());
            const mi::mdl::ISimple_name* param_simple_name =
                m_name_factory->create_simple_name(param_symbol);

            const mi::mdl::IArgument* call_argument =
                m_expression_factory->create_named_argument(param_simple_name, arg_expr);
            call_to_prototype->add_argument(call_argument);
        }
    }

    // add imports required by arguments
    m_symbol_importer->collect_imports(call_to_prototype);

    // create body for variant
    const mi::mdl::IStatement_expression* variant_body
        = m_statement_factory->create_expression(call_to_prototype);

    // create new function to add
    const mi::mdl::ISymbol* symbol_to_add = m_name_factory->create_symbol(simple_name);
    const mi::mdl::ISimple_name* simple_name_to_add =
        m_name_factory->create_simple_name(symbol_to_add);
    mi::mdl::IDeclaration_function* declaration_to_add = m_declaration_factory->create_function(
        prototype_declaration->get_return_type_name(), 
        /*ret_annotations*/ 0, 
        simple_name_to_add,
        /*is_clone*/ true,
        variant_body, 
        /*annotation*/ nullptr, 
        is_exported);

    // defer the actual adding to the module until the module is build
    m_added_functions.push_back(declaration_to_add);
    m_added_function_annotations.push_back(m_annotation_factory->create_annotation_block());
    m_added_function_symbols.push_back(symbol_to_add);
    return m_added_functions.size() - 1; // get index
}

// Adds a prototype (material or function definition) to the module to create.
mi::Sint32 Mdl_module_builder::add_prototype(
    MI::DB::Tag prototype_tag,
    const char* name,
    const MI::MDL::IExpression_list* defaults,
    const MI::MDL::IAnnotation_block* annotations,
    bool is_exported,
    MI::MDL::Execution_context* context)
{
    if (!clear_and_check_valid(context)) return -1;

    // wraps the function/material to add into a call that simply forwards all parameters
    // along with annotations
    SERIAL::Class_id class_id = m_transaction->get_class_id(prototype_tag);
    std::string org_name;

    if (class_id == ID_MDL_MATERIAL_DEFINITION)
    {
        // get definition for obtaining default arguments and annotations
        DB::Access<Mdl_material_definition> prototype(prototype_tag, m_transaction);
        const MI::MDL::Mdl_material_definition* prototype_def = prototype.get_ptr();

        // get the original name of the material
        if (!name)
        {
            org_name = prototype_def->get_mdl_name();
            size_t pos = org_name.find_last_of(':');
            if (pos != std::string::npos)
                org_name = org_name.substr(pos + 1);
        }

        // create the definition
        mi::base::Handle<const MI::MDL::IExpression_list> d(prototype_def->get_defaults());
        mi::Sint32 index = add_intern<Mdl_material_definition>(
            prototype_tag,
            (name != nullptr) ? name : org_name.c_str(),
            (defaults != nullptr) ? defaults : d.get(),
            false,
            is_exported,
            context);

        // only if no error occurred
        if (index >= 0)
        {
            // copy annotations from prototype
            if (annotations == NULL)
            {
                // get definition for obtaining default arguments and annotations
                DB::Access<Mdl_material_definition> prototype(prototype_tag, m_transaction);
                const MI::MDL::Mdl_material_definition* prototype_def = prototype.get_ptr();

                mi::base::Handle<const MI::MDL::IAnnotation_block> a(
                    prototype_def->get_annotations());
                set_annotations(index, a.get(), context);
            }
            // or set specified annotations if there are some
            else if (annotations->get_size() > 0)
                set_annotations(index, annotations, context);
        }
        return index;
    }
    else if (class_id == ID_MDL_FUNCTION_DEFINITION)
    {
        // get definition for obtaining default arguments and annotations
        DB::Access<Mdl_function_definition> prototype(prototype_tag, m_transaction);
        const MI::MDL::Mdl_function_definition* prototype_def = prototype.get_ptr();

        // get the original name of the material
        if (!name)
        {
            org_name = prototype_def->get_mdl_name();
            size_t pos = org_name.find_first_of('(');
            if (pos == std::string::npos)
                pos = 0;
            pos += org_name.find_last_of(':', pos);
            if (pos != std::string::npos)
                org_name = org_name.substr(pos + 1);
        }

        // create the definition
        mi::base::Handle<const MI::MDL::IExpression_list> d(prototype_def->get_defaults());
        mi::Sint32 index = add_intern<Mdl_function_definition>(
            prototype_tag,
            (name != nullptr) ? name : org_name.c_str(),
            (defaults != nullptr) ? defaults : d.get(),
            false,
            is_exported,
            context);

        // only if no error occurred
        if (index >= 0)
        {
            // set annotations
            mi::base::Handle<const MI::MDL::IAnnotation_block> a(prototype_def->get_annotations());
            set_annotations(index, a.get(), context);

            // set return value annotations
            a = prototype_def->get_return_annotations();
            set_return_annotations(index, a.get(), context);
        }
        return index;
    }

    ASSERT(M_SCENE, false);
    const MI::MDL::Message message(
        MI::MDL::Message(mi::base::MESSAGE_SEVERITY_ERROR,
        "The DB element of one of the prototypes has the wrong type.",
        -5, MI::MDL::Message::MSG_INTEGRATION));

    context->set_result(-5);
    context->add_message(message);
    context->add_error_message(message);
    return -1;
}

/// Adds a variant of an prototype (material or function definition) to the module to create.
mi::Sint32 Mdl_module_builder::add_variant(
    mi::Sint32 index,
    const char* name,
    const MI::MDL::IExpression_list* defaults,
    const MI::MDL::IAnnotation_block* annotations,
    const MI::MDL::IAnnotation_block* return_annotations,
    bool is_exported,
    MI::MDL::Execution_context* context)
{
    if (index >= static_cast<mi::Sint32>(m_added_functions.size()))
    {
        report_messages(m_module->access_messages(), context);
        const MI::MDL::Message message(
            MI::MDL::Message(mi::base::MESSAGE_SEVERITY_ERROR,
            "Unspecified error. Provided material/function index is invalid.",
            -8, MI::MDL::Message::MSG_INTEGRATION));

        context->set_result(-8);
        context->add_message(message);
        context->add_error_message(message);
        return -1;
    }

    // create the actual variant
    mi::Sint32 res_index = add_variant_intern(
        index,
        name,
        defaults,
        is_exported,
        context);

    // only if no error occurred
    if (res_index >= 0)
    {
        mi::mdl::IDeclaration_function* prototype_declaration = m_added_functions[index];
        mi::mdl::Module* module_impl = mi::mdl::impl_cast<mi::mdl::Module>(m_module.get());

        // copy annotations from prototype
        if (annotations == NULL) {
            const mi::mdl::IAnnotation_block* to_copy = prototype_declaration->get_annotations();
            mi::mdl::IAnnotation_block* to_add = module_impl->clone_annotation_block(to_copy, NULL);
            m_added_functions[res_index]->set_annotation(to_add);
        } 
        // copy user specified annotations (or none, if the block is empty)
        else if(annotations->get_size() > 0) {
            set_annotations(res_index, annotations, context);
        }

        // same for return annotations of functions
        if (return_annotations == NULL)
        {
            const mi::mdl::IAnnotation_block* to_copy =
                prototype_declaration->get_return_annotations();
            mi::mdl::IAnnotation_block* to_add = 
                module_impl->clone_annotation_block(to_copy, NULL);
            m_added_functions[res_index]->set_return_annotation(to_add);
        }
        else if (return_annotations->get_size() > 0)
        {
            set_return_annotations(res_index, return_annotations, context);
        }
    }

    return res_index;
}

/// Adds a variant of an prototype (material or function definition) to the module to create.
mi::Sint32 Mdl_module_builder::add_variant(
    const MI::DB::Tag prototype_tag,
    const char* name,
    const MI::MDL::IExpression_list* defaults,
    const MI::MDL::IAnnotation_block* annotations,
    const MI::MDL::IAnnotation_block* return_annotations,
    bool is_exported,
    MI::MDL::Execution_context* context)
{
    if (!clear_and_check_valid(context)) return -1;

    // wraps the function/material to add into a call that simply forwards all parameters
    // along with annotations
    SERIAL::Class_id class_id = m_transaction->get_class_id(prototype_tag);


    if (class_id == ID_MDL_MATERIAL_DEFINITION)
    {
        // create the definition
        mi::Sint32 index = add_intern<Mdl_material_definition>(
            prototype_tag,
            name,
            defaults,
            true,
            is_exported,
            context);

        // only if no error occurred
        if (index >= 0)
        {
            // copy annotations from prototype
            if (annotations == NULL)
            {
                // get definition for obtaining default arguments and annotations
                DB::Access<Mdl_material_definition> prototype(prototype_tag, m_transaction);
                const MI::MDL::Mdl_material_definition* prototype_def = prototype.get_ptr();

                mi::base::Handle<const MI::MDL::IAnnotation_block> a(
                    prototype_def->get_annotations());
                set_annotations(index, a.get(), context);
            }
            // or set specified annotations if there are some
            else if(annotations->get_size() > 0)
                set_annotations(index, annotations, context);
        }
        return index;
    }
    else if (class_id == ID_MDL_FUNCTION_DEFINITION)
    {
        // create the definition
        mi::Sint32 index = add_intern<Mdl_function_definition>(
            prototype_tag,
            name,
            defaults,
            true,
            is_exported,
            context);

        // only if no error occurred
        if (index >= 0)
        {
            // get definition for obtaining default arguments and annotations
            DB::Access<Mdl_function_definition> prototype(prototype_tag, m_transaction);
            const MI::MDL::Mdl_function_definition* prototype_def = prototype.get_ptr();

            // copy annotations from prototype
            if (annotations == NULL)
            {
                mi::base::Handle<const MI::MDL::IAnnotation_block> a(
                    prototype_def->get_annotations());
                set_annotations(index, a.get(), context);
            }
            // or set specified annotations if there are some
            else if (annotations->get_size() > 0)
                set_annotations(index, annotations, context);


            // copy return annotations from prototype
            if (return_annotations == NULL)
            {
                mi::base::Handle<const MI::MDL::IAnnotation_block> a(
                    prototype_def->get_return_annotations());
                set_return_annotations(index, a.get(), context);
            }
            // or set specified return annotations if there are some
            else if (return_annotations->get_size() > 0)
                set_return_annotations(index, return_annotations, context);
        }
        return index;
    }

    ASSERT(M_SCENE, false);
    const MI::MDL::Message message(
        MI::MDL::Message(mi::base::MESSAGE_SEVERITY_ERROR,
        "The DB element of one of the prototypes has the wrong type.",
        -5, MI::MDL::Message::MSG_INTEGRATION));

    context->set_result(-5);
    context->add_message(message);
    context->add_error_message(message);
    return -1;
}

// Adds a variant (of a material or function definition) to the module to create.
mi::Sint32 Mdl_module_builder::add_variant(
    const MI::MDL::Variant_data* variant_data,
    bool is_exported,
    MI::MDL::Execution_context* context)
{
    if (!clear_and_check_valid(context)) return -1;

    return add_variant(
        variant_data->m_prototype_tag,
        variant_data->m_variant_name.c_str(),
        variant_data->m_defaults.get(),
        variant_data->m_annotations.get(),
        nullptr,
        is_exported,
        context);
}

// Set/change the annotations of an added material, function or variant.
bool Mdl_module_builder::set_annotations(
    mi::Sint32 index,
    const MI::MDL::IAnnotation_block* annotations,
    MI::MDL::Execution_context* context)
{
    if (index >= static_cast<mi::Sint32>(m_added_functions.size()))
        return false;

    // get the annotations block
    mi::mdl::IAnnotation_block* mdl_annotation_block = m_added_function_annotations[index];

    // delete all current annotations
    for (int a = 0; a < mdl_annotation_block->get_annotation_count(); ++a)
        mdl_annotation_block->delete_annotation(a);

    // add the new ones
    for (mi::Size a = 0; annotations && a < annotations->get_size(); ++a)
        if (!add_annotation(index, annotations->get_annotation(a), context))
            return false;

    return true;
}


// Set/change the annotations of an added material, function or variant.
bool Mdl_module_builder::add_annotation(
    mi::Sint32 index,
    const MI::MDL::IAnnotation* annotation,
    MI::MDL::Execution_context* context)
{
    if (index >= static_cast<mi::Sint32>(m_added_functions.size()))
        return false;

    // get the annotations block
    mi::mdl::IAnnotation_block* mdl_annotation_block = m_added_function_annotations[index];

    // convert the annotation to add
    const char* anno_name = annotation->get_name();

    // skip deprecated annotations
    if (strchr(anno_name, '$') != nullptr) {
        context->add_message(
            MI::MDL::Message(mi::base::MESSAGE_SEVERITY_WARNING,
            MI::STRING::formatted_string("Skipped deprecated annotation: %s.", anno_name),
            0, MI::MDL::Message::MSG_INTEGRATION));
        return true;
    }

    mi::base::Handle<const IExpression_list> anno_args(annotation->get_arguments());
    mi::Sint32 result = add_annotation(mdl_annotation_block, anno_name, anno_args.get());
    return result == 0;
}

// Set/change the annotations of an added material, function or variant.
bool Mdl_module_builder::set_return_annotations(
    mi::Sint32 index,
    const MI::MDL::IAnnotation_block* annotations,
    MI::MDL::Execution_context* context)
{
    if (index >= static_cast<mi::Sint32>(m_added_functions.size()))
        return false;

    // convert the annotations block
    mi::mdl::IAnnotation_block* mdl_annotation_block = 0;
    if (!create_annotations(annotations, mdl_annotation_block, context))
        return false;

    // attach the block to the material
    mi::mdl::IDeclaration_function* declaration = m_added_functions[index];
    declaration->set_return_annotation(mdl_annotation_block);
    return true;
}


// Completes the module building process and returns the resulting module or NULL in case of
mi::mdl::IModule const* Mdl_module_builder::build(
    MI::MDL::Execution_context* context)
{
    if (!clear_and_check_valid(context)) return nullptr;

    // add annotations
    for(size_t a = 0; a < m_added_functions.size(); ++a) {
        mi::mdl::IAnnotation_block* mdl_annotation_block = m_added_function_annotations[a];
        m_symbol_importer->collect_imports(mdl_annotation_block);
        m_added_functions[a]->set_annotation(mdl_annotation_block);
    }

    // add declarations to module
    for(auto decl : m_added_functions)
        m_module->add_declaration(decl);

    // add all collected imports
    m_symbol_importer->add_imports();

    //Module_cache module_cache(m_transaction);
    m_module->analyze(/*cache=*/NULL, m_thread_context.get());
    if (!m_module->is_valid())
    {
        report_messages(m_module->access_messages(), context);
        const MI::MDL::Message message(
            MI::MDL::Message(mi::base::MESSAGE_SEVERITY_ERROR,
            "Unspecified error. Failed to build a valid module from selected elements.",
            -8, MI::MDL::Message::MSG_INTEGRATION));

        context->set_result(-8);
        context->add_message(message);
        context->add_error_message(message);
        return nullptr;
    }

    m_module->retain();
    return m_module.get();
}

bool Mdl_module_builder::clear_and_check_valid(MI::MDL::Execution_context* context)
{
    context->clear_messages();

    if (m_module)
        return true;

    const MI::MDL::Message message(
        MI::MDL::Message(mi::base::MESSAGE_SEVERITY_ERROR,
        "Module builder is in invalid state. "
        "build() is already completed or a previous step failed.",
        -8, MI::MDL::Message::MSG_INTEGRATION));

    context->set_result(-8);
    context->add_message(message);
    context->add_error_message(message);
    return false;
}

// wraps the function below
bool Mdl_module_builder::create_annotations(
    const MI::MDL::IAnnotation_block* annotation_block,
    mi::mdl::IAnnotation_block* &mdl_annotation_block,
    MI::MDL::Execution_context* context)
{
    mi::Sint32 result = create_annotations(annotation_block, mdl_annotation_block);

    switch (result)
    {
        case 0:
            return true;

        case -9:
        {
            const MI::MDL::Message message(
                MI::MDL::Message(mi::base::MESSAGE_SEVERITY_ERROR,
                "One of the annotation arguments is wrong "
                "(wrong argument name, not a constant expression, or the "
                "argument type does not match the parameter type)",
                result, MI::MDL::Message::MSG_INTEGRATION));

            context->set_result(result);
            context->add_message(message);
            context->add_error_message(message);
            return false;
        }
        case -10:
        {
            const MI::MDL::Message message(
                MI::MDL::Message(mi::base::MESSAGE_SEVERITY_ERROR,
                "One of the annotations does not exist or it has a currently "
                "unsupported parameter type like deferred-sized arrays.",
                result, MI::MDL::Message::MSG_INTEGRATION));

            context->set_result(result);
            context->add_message(message);
            context->add_error_message(message);
            return false;
        }
        default:
            ASSERT(M_SCENE, false);
            return false;
    }
    return true;
}


mi::Sint32 Mdl_module_builder::create_annotations(
    const MI::MDL::IAnnotation_block* annotation_block,
    mi::mdl::IAnnotation_block* &mdl_annotation_block)
{
    if (!annotation_block)
    {
        mdl_annotation_block = 0;
        return 0;
    }

    mdl_annotation_block = m_annotation_factory->create_annotation_block();
    for (mi::Size i = 0; i < annotation_block->get_size(); ++i)
    {
        mi::base::Handle<const IAnnotation> anno(annotation_block->get_annotation(i));
        const char* anno_name = anno->get_name();
        mi::base::Handle<const IExpression_list> anno_args(anno->get_arguments());
        mi::Sint32 result = add_annotation(mdl_annotation_block, anno_name, anno_args.get());
        if (result != 0)
            return result;
    }

    m_symbol_importer->collect_imports(mdl_annotation_block);
    return 0;
}

mi::Sint32 Mdl_module_builder::add_annotation(
    mi::mdl::IAnnotation_block* mdl_annotation_block,
    const char* annotation_name,
    const MI::MDL::IExpression_list* annotation_args)
{
    if (strncmp(annotation_name, "::", 2) != 0)
        return -10;

    std::string annotation_name_str = add_mdl_db_prefix(annotation_name);
    size_t mdle_offset = annotation_name_str.find(".mdle::");
    bool is_mdle_annotation = mdle_offset != std::string::npos; // true only for MDLE
    size_t prefix_length = is_mdle_annotation ? 4 : 3;          // mdle or mdl 
    mdle_offset = is_mdle_annotation ? (mdle_offset + 7) : 0;   // offset to get after the file ext.
                                                                // avoid problems with '(' in path

    // compute DB name of module containing the annotation
    std::string anno_db_module_name = annotation_name_str;
    size_t left_paren = anno_db_module_name.find('(', mdle_offset);
    if (left_paren == std::string::npos)
        return -10;
    anno_db_module_name = anno_db_module_name.substr(0, left_paren);
    size_t last_double_colon = anno_db_module_name.rfind("::");
    if (last_double_colon == std::string::npos)
        return -10;
    anno_db_module_name = anno_db_module_name.substr(0, last_double_colon);

    // get definition of the annotation
    DB::Tag anno_db_module_tag = m_transaction->name_to_tag(anno_db_module_name.c_str());
    if (!anno_db_module_tag)
        return -10;
    DB::Access<Mdl_module> anno_db_module(anno_db_module_tag, m_transaction);
    mi::base::Handle<const mi::mdl::IModule> anno_mdl_module(anno_db_module->get_mdl_module());
    std::string annotation_name_wo_signature = 
        annotation_name_str.substr(prefix_length, left_paren - prefix_length);
    std::string signature = annotation_name_str.substr(
        left_paren + 1, annotation_name_str.size() - left_paren - 2);
    const mi::mdl::IDefinition* definition = anno_mdl_module->find_annotation(
        annotation_name_wo_signature.c_str(), signature.c_str());
    if (!definition)
        return -10;

    // compute IQualified_name for annotation name
    mi::mdl::IQualified_name* anno_qualified_name = m_name_factory->create_qualified_name();
    anno_qualified_name->set_absolute();
    size_t start = prefix_length + 2; // skip leading "mdle::" or "mdl::"
    while (true)
    {
        size_t end = annotation_name_str.find("::", start);
        if (end == std::string::npos || end >= left_paren)
            end = left_paren;
        std::string chunk = annotation_name_str.substr(start, end - start);
        const mi::mdl::ISymbol* anno_symbol
            = m_name_factory->create_symbol(chunk.c_str());
        const mi::mdl::ISimple_name* anno_simple_name = m_name_factory->create_simple_name(anno_symbol);
        anno_qualified_name->add_component(anno_simple_name);
        if (end == left_paren)
            break;
        start = end + 2;
    }

    // create annotation
    mi::mdl::IAnnotation* anno = m_annotation_factory->create_annotation(anno_qualified_name);

    // store parameter types from annotation definition in a map by parameter name
    const mi::mdl::IType* type = definition->get_type();
    ASSERT(M_SCENE, type->get_kind() == mi::mdl::IType::TK_FUNCTION);
    const mi::mdl::IType_function* type_function = mi::mdl::as<mi::mdl::IType_function>(type);
    std::map<std::string, const mi::mdl::IType*> parameter_types;
    int parameter_count = type_function->get_parameter_count();
    for (int i = 0; i < parameter_count; ++i)
    {
        const mi::mdl::IType* parameter_type;
        const mi::mdl::ISymbol* parameter_name;
        type_function->get_parameter(i, parameter_type, parameter_name);
        parameter_types[parameter_name->get_name()] = parameter_type;
    }

    // convert arguments

    mi::Size argument_count = annotation_args->get_size();
    for (mi::Size i = 0; i < argument_count; ++i)
    {
        const char* arg_name = annotation_args->get_name(i);

        mi::base::Handle<const IExpression_constant> arg_expr(
            annotation_args->get_expression<IExpression_constant>(i));
        if (!arg_expr)
            return -9;
        mi::base::Handle<const IValue> arg_value(arg_expr->get_value());
        mi::base::Handle<const IType> arg_type(arg_value->get_type());

        // The legacy API always provides "argument" as argument name. Since it supports only single
        // string arguments we map that argument name to the correct one if all these conditions are
        // met -- even for the non-legacy API.
        if (i == 0
            && parameter_count == 1
            && argument_count == 1
            && strcmp(arg_name, "argument") == 0
            && arg_type->get_kind() == IType::TK_STRING)
        {
            arg_name = parameter_types.begin()->first.c_str();
        }

        const mi::mdl::IType* mdl_parameter_type = parameter_types[arg_name];
        if (!mdl_parameter_type)
            return -9;
        mdl_parameter_type = m_type_factory->import(mdl_parameter_type);
        const mi::mdl::IValue* mdl_arg_value = int_value_to_mdl_value(
            m_transaction, m_value_factory, mdl_parameter_type, arg_value.get());
        if (!mdl_arg_value)
            return -9;

        const mi::mdl::IExpression* mdl_arg_expr
            = m_expression_factory->create_literal(mdl_arg_value);
        const mi::mdl::ISymbol* arg_symbol = m_name_factory->create_symbol(arg_name);
        const mi::mdl::ISimple_name* arg_simple_name = m_name_factory->create_simple_name(arg_symbol);
        const mi::mdl::IArgument* mdl_arg
            = m_expression_factory->create_named_argument(arg_simple_name, mdl_arg_expr);
        anno->add_argument(mdl_arg);
    }

    mdl_annotation_block->add_annotation(anno);
    return 0;
}


} // namespace MDL
} // namespace MI
