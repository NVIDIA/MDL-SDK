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
#include <base/lib/log/i_log_logger.h>
#include <base/system/main/access_module.h>
#include <base/util/string_utils/i_string_utils.h>
#include <io/scene/mdl_elements/mdl_elements_ast_builder.h>
#include <mdl/integration/mdlnr/i_mdlnr.h>

#include <mdl/compiler/compilercore/compilercore_modules.h>
#include <mdl/compiler/compilercore/compilercore_tools.h>

#include <mi/mdl/mdl_module_transformer.h>

namespace MI {

namespace MDL {

namespace {

/// Add message to context
mi::Sint32 add_error_message(
    MDL::Execution_context* context,
    const std::string& message,
    mi::Sint32 result)
{
    if (!context)
        return -1;

    const MDL::Message err(
        mi::base::MESSAGE_SEVERITY_ERROR, message, -1, MI::MDL::Message::MSG_INTEGRATION);

    context->add_error_message(err);
    context->add_message(err);
    context->set_result(result);

    return -1;
}

} // namespace

Mdl_module_builder::New_parameter::New_parameter(
    const mi::mdl::ISymbol* sym,
    const mi::base::Handle<const MI::MDL::IExpression>& init,
    const mi::base::Handle<const MI::MDL::IAnnotation_block> annos,
    bool is_uniform)
    : m_sym(sym)
    , m_init(init)
    , m_annos(annos)
    , m_is_uniform(is_uniform)
{
}

Mdl_module_builder::Mdl_module_builder(
    mi::mdl::IMDL* imdl,
    MI::DB::Transaction* transaction,
    const char* module_name,
    mi::mdl::IMDL::MDL_version version,
    bool allow_compatible_types,
    bool inline_mdle,
    MI::MDL::Execution_context* context)
: m_mdl(mi::base::make_handle_dup(imdl))
, m_transaction(transaction)
, m_thread_context(nullptr)
, m_module(nullptr)
, m_symbol_importer(nullptr)
, m_name_mangler(nullptr)
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
, m_allow_compatible_types(allow_compatible_types)
, m_inline_mdle(inline_mdle)
{
    ASSERT(M_SCENE, context);

    context->clear_messages();

    // Reject invalid module names (in particular, names containing slashes and backslashes).
    if (!MI::MDL::Mdl_module::is_valid_module_name(module_name, m_mdl.get()))
    {
        add_error_message(
            context, "The module name " + std::string(module_name) + " is invalid.", -1);
        return;
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

        add_error_message(context, " Failed to create the module " + std::string(module_name), -2);
        return;
    }

    m_symbol_importer = new Symbol_importer(m_module.get());
    m_name_mangler = new Name_mangler(imdl);

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
    delete m_symbol_importer;
    delete m_name_mangler;
}

namespace
{

mi::mdl::IType_name* create_material_type_name(
    mi::mdl::IName_factory &nf)
{
    const mi::mdl::ISymbol* return_type_symbol = nf.create_symbol("material");
    const mi::mdl::ISimple_name* return_type_simple_name
        = nf.create_simple_name(return_type_symbol);
    mi::mdl::IQualified_name* return_type_qualified_name = nf.create_qualified_name();
    return_type_qualified_name->add_component(return_type_simple_name);
    return nf.create_type_name(return_type_qualified_name);
}

// create return type for variant
template <class T>
mi::mdl::IType_name* create_return_type_name(
    DB::Transaction* transaction,
    mi::mdl::IModule* module,
    DB::Access<T> prototype);

template <>
mi::mdl::IType_name* create_return_type_name(
    DB::Transaction*,
    mi::mdl::IModule* module,
    DB::Access<Mdl_material_definition>)
{
    return create_material_type_name(*module->get_name_factory());
}

template <>
mi::mdl::IType_name* create_return_type_name(
    DB::Transaction*,
    mi::mdl::IModule* module,
    DB::Access<Mdl_material_instance>)
{
    return create_material_type_name(*module->get_name_factory());
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

template <>
mi::mdl::IType_name* create_return_type_name(
    DB::Transaction* transaction,
    mi::mdl::IModule* module,
    DB::Access<Mdl_function_call> prototype)
{
    const mi::mdl::IType* ret_type = prototype->get_mdl_return_type(transaction);
    return mi::mdl::create_type_name(ret_type, module);
}

template <class T>
const char* get_mdl_definition(
    DB::Access<T> prototype);

template <>
const char* get_mdl_definition(
    DB::Access<Mdl_material_instance> prototype)
{
    return prototype->get_mdl_material_definition();
}

template <>
const char* get_mdl_definition(
    DB::Access<Mdl_function_call> prototype)
{
    return prototype->get_mdl_function_definition();
}

template <class T>
const IType* get_return_type(
    DB::Access<T> prototype);

template <>
const IType* get_return_type(
    DB::Access<Mdl_material_instance> prototype)
{
    return nullptr;
}

template <>
const IType* get_return_type(
    DB::Access<Mdl_material_definition> prototype)
{
    return nullptr;
}

template <>
const IType* get_return_type(
    DB::Access<Mdl_function_call> prototype)
{
    return prototype->get_return_type();
}

template <>
const IType* get_return_type(
    DB::Access<Mdl_function_definition> prototype)
{
    return prototype->get_return_type();
}

template <class T>
mi::mdl::IDefinition::Semantics get_mdl_semantic(
    DB::Access<T> prototype);

template <>
mi::mdl::IDefinition::Semantics get_mdl_semantic(
    DB::Access<Mdl_material_instance> prototype)
{
    return mi::mdl::IDefinition::DS_UNKNOWN;
}

template <>
mi::mdl::IDefinition::Semantics get_mdl_semantic(
    DB::Access<Mdl_material_definition> prototype)
{
    return mi::mdl::IDefinition::DS_UNKNOWN;
}

template <>
mi::mdl::IDefinition::Semantics get_mdl_semantic(
    DB::Access<Mdl_function_call> prototype)
{
    return prototype->get_mdl_semantic();
}

template <>
mi::mdl::IDefinition::Semantics get_mdl_semantic(
    DB::Access<Mdl_function_definition> prototype)
{
    return prototype->get_mdl_semantic();
}

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

}   // anonymous

template <typename T>
mi::Sint32 Mdl_module_builder::add_intern(
    MI::DB::Tag prototype_tag,
    const char* simple_name,
    const MI::MDL::IExpression_list* defaults,
    const MI::MDL::IAnnotation_block* annotations,
    const MI::MDL::IAnnotation_block* return_annotations,
    bool is_variant,
    bool is_exported,
    MI::MDL::Execution_context* context)
{
    MI::SERIAL::Class_id id = m_transaction->get_class_id(prototype_tag);
    ASSERT(M_SCENE, id == T::id);

    DB::Access<T> prototype(prototype_tag, m_transaction);

    // check that the provided arguments are parameters of the material definition
    // and that their types match the expected types
    mi::base::Handle<const IType_list> expected_types(prototype->get_parameter_types());
    std::vector<bool> needs_cast(expected_types->get_size(), false);
    for (mi::Size i = 0; defaults && i < defaults->get_size(); ++i)
    {
        const char* param_name = defaults->get_name(i);
        mi::base::Handle<const IType> expected_type(expected_types->get_type(param_name));
        if (!expected_type)
        {
            return add_error_message(
                context, "A default for a non-existing parameter was provided.", -6);
        }
        mi::base::Handle<const IExpression> argument(defaults->get_expression(i));
        mi::base::Handle<const IType> actual_type(argument->get_type());
        mi::base::Handle<IType_factory> tf(get_type_factory());

        bool needs_cast_tmp = false;
        if (!argument_type_matches_parameter_type(
            tf.get(), actual_type.get(), expected_type.get(), m_allow_compatible_types, needs_cast_tmp))
        {
            return add_error_message(
                context, "The type of a default does not match the expected/correct type.", -7);
        }
        needs_cast[i] = needs_cast_tmp;
    }

    mi::mdl::IDefinition::Semantics sema = get_mdl_semantic(prototype);
    // some semantics require special handling
    bool maps_to_mdl_function = true;
    if (semantic_is_operator(sema) ||
        sema == mi::mdl::IDefinition::DS_INTRINSIC_DAG_FIELD_ACCESS) {

        if (is_variant) {
            return add_error_message(
                context, "Prototype definition unsuitable as base of a variant.", -7);
        }
        maps_to_mdl_function = false;
    }

    mi::mdl::IExpression_call* call_to_prototype = nullptr;
    if (maps_to_mdl_function) {
        // create call expression
        const char* prototype_name = prototype->get_mdl_name();
        const mi::mdl::IExpression_reference* prototype_ref
            = signature_to_reference(m_module.get(), prototype_name, m_name_mangler);
        call_to_prototype = m_expression_factory->create_call(prototype_ref);
    }

    // setup the builder
    MI::MDL::Mdl_ast_builder ast_builder(
       m_module.get(), m_transaction, mi::base::make_handle_dup(defaults), *m_name_mangler);

    // parameters that we will add to the new method later
    std::vector<const mi::mdl::IParameter*> forwarded_parameters;

    // create arguments for call/variant
    if (defaults != nullptr)
    {
        mi::base::Handle<const MI::MDL::IAnnotation_list> param_annotations(
            prototype->get_parameter_annotations());

        mi::base::Handle<IExpression_factory> int_ef(MDL::get_expression_factory());
        for (mi::Size i = 0, n = prototype->get_parameter_count(); i < n; ++i)
        {
            // get the MI::MDL argument
            const char* param_name = prototype->get_parameter_name(i);
            mi::base::Handle<const IExpression> argument(defaults->get_expression(param_name));
            if (!argument)
                continue;

            mi::base::Handle<const IType> expected_type(expected_types->get_type(param_name));
            if (needs_cast[i]) { // argument needs cast
                mi::Sint32 errors = 0;
                mi::base::Handle<IExpression> cloned_argument(
                    int_ef->clone(argument.get(), m_transaction, /*copy_immutable_calls=*/false));
                mi::base::Handle<IExpression> casted_argument(
                    int_ef->create_cast(m_transaction, cloned_argument.get(), expected_type.get(),
                        /*cast_db_name=*/nullptr,
                        /*force_cast=*/false,
                        /*create_direct_call=*/false, &errors));

                argument = casted_argument;
            }

            // convert to mi::mdl and promote to module version number
            const mi::mdl::IType* param_type = prototype->get_mdl_parameter_type(
                m_transaction, static_cast<mi::Uint32>(i));

            const mi::mdl::IExpression* arg_expr = int_expr_to_mdl_ast_expr(
                m_transaction, m_module.get(), param_type, argument.get(), m_name_mangler);
            arg_expr = promote_expressions_to_mdl_version(m_module.get(), arg_expr);
            if (!arg_expr)
            {
                return add_error_message(
                    context, "Unspecified error. Failed to promote expression.",-8);
            }

            // collect imports for that argument
            mi::mdl::IType_name *tn = ast_builder.create_type_name(expected_type);
            m_symbol_importer->collect_imports(tn);
            m_symbol_importer->collect_imports(arg_expr);

            const mi::mdl::ISymbol* param_symbol = m_name_factory->create_symbol(param_name);
            const mi::mdl::ISimple_name* param_simple_name = 
                m_name_factory->create_simple_name(param_symbol);

            if (is_variant)
            {
                const mi::mdl::IArgument* argument =
                    m_expression_factory->create_named_argument(param_simple_name, arg_expr);
                call_to_prototype->add_argument(argument);
            }
            else
            {
                // create a new parameter that is added to the created function/material
                mi::base::Handle<const IType> argument_type(argument->get_type());
                mi::mdl::IType_name* type_name = ast_builder.create_type_name(argument_type);
                if (param_type->get_type_modifiers() & mi::mdl::IType::MK_UNIFORM)
                    type_name->set_qualifier(mi::mdl::FQ_UNIFORM);
                type_name->set_type(param_type);

                // get parameter annotations
                mi::base::Handle<const IAnnotation_block> annotations(
                    param_annotations->get_annotation_block(param_name));
                mi::mdl::IAnnotation_block* mdl_annotations = 0;
                if (!create_annotations(annotations.get(), mdl_annotations, context, /*skip_unused=*/true))
                    return -1;

                // keep the parameter for adding it later to the created function/material
                forwarded_parameters.push_back(m_declaration_factory->create_parameter(
                    type_name, param_simple_name, arg_expr, mdl_annotations));

                // add a reference to the call
                if (maps_to_mdl_function) {
                    const mi::mdl::IExpression_reference* ref = ast_builder.to_reference(param_symbol);
                    const mi::mdl::IArgument* call_argument =
                        m_expression_factory->create_named_argument(param_simple_name, ref);
                    call_to_prototype->add_argument(call_argument);
                }
                else {
                    ast_builder.declare_parameter(param_symbol, argument);
                }
            }
        }
    }

    // add imports required by arguments
    if (call_to_prototype)
        m_symbol_importer->collect_imports(call_to_prototype);
   
    // create return type for function/material
    mi::mdl::IType_name* return_type_type_name
        = create_return_type_name(m_transaction, m_module.get(), prototype);

    // add imports required by return type
    m_symbol_importer->collect_imports(return_type_type_name);

    // return type modifier
    if (has_uniform_return_qualifier(prototype.get_ptr()))
        return_type_type_name->set_qualifier(mi::mdl::FQ_UNIFORM);

    // create body for new material/function
    mi::mdl::IStatement* body;
    if (id == ID_MDL_FUNCTION_DEFINITION && !is_variant) {
        mi::mdl::IStatement_compound* comp = m_statement_factory->create_compound();
        if (maps_to_mdl_function) {
            comp->add_statement(
                m_statement_factory->create_return(call_to_prototype));
        }
        else {
            mi::base::Handle<const IType> ret_type(get_return_type(prototype));
            std::string def = ast_builder.dag_unmangle(prototype->get_mdl_name());
            const mi::mdl::IExpression *ret_expr = ast_builder.transform_call(
                ret_type,
                sema,
                def,
                defaults->get_size(),
                mi::base::make_handle_dup(defaults),
                true);

            comp->add_statement(
                m_statement_factory->create_return(ret_expr));
        }
        body = comp;
    }
    else
        body = m_statement_factory->create_expression(call_to_prototype);

    // create new function
    const mi::mdl::ISymbol* symbol_to_add = m_name_factory->create_symbol(simple_name);
    const mi::mdl::ISimple_name* simple_name_to_add = 
        m_name_factory->create_simple_name(symbol_to_add);

    mi::mdl::IDeclaration_function* declaration_to_add = m_declaration_factory->create_function(
        return_type_type_name,
        /*ret_annotations=*/0,
        simple_name_to_add, 
        /*is_clone=*/is_variant,
        body,
        /*annotation=*/nullptr,
        is_exported);

    // function modifier
    if (has_uniform_qualifier(prototype.get_ptr()))
        declaration_to_add->set_qualifier(mi::mdl::FQ_UNIFORM);

    // add parameters to the created function/material
    if (!is_variant)
        for (auto&& p : forwarded_parameters)
            declaration_to_add->add_parameter(p);

    // defer the actual adding to the module until the module is built
    m_added_functions.push_back(declaration_to_add);
    m_added_function_annotations.push_back(m_annotation_factory->create_annotation_block());
    m_added_function_symbols.push_back(symbol_to_add);

    mi::Sint32 res_index = m_added_functions.size() - 1;
    if (annotations != nullptr)
        set_annotations(res_index, annotations, context);
    if (return_annotations != nullptr)
        set_return_annotations(res_index, return_annotations, context);

    return res_index;
}

mi::Sint32 Mdl_module_builder::add_material_or_function(
    MI::DB::Tag prototype_tag,
    const char* name,
    const MI::MDL::IExpression_list* defaults,
    const MI::MDL::IAnnotation_block* annotations,
    const MI::MDL::IAnnotation_block* ret_annotations,
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

        mi::base::Handle<const MI::MDL::IExpression_list> prototype_defaults;
        mi::base::Handle<const MI::MDL::IAnnotation_block> prototype_annotations;

        // get defaults from prototype if requested
        if( defaults == nullptr) {
            prototype_defaults = prototype_def->get_defaults();
            defaults = prototype_defaults.get();
        }

        // get annotations from prototype if requested
        if( annotations == nullptr) {
            prototype_annotations = prototype->get_annotations();
            annotations = prototype_annotations.get();
        }
        
        ASSERT(M_SCENE, ret_annotations == nullptr || ret_annotations->get_size() == 0);

        // create the definition
        mi::Sint32 index = add_intern<Mdl_material_definition>(
            prototype_tag,
            (name != nullptr) ? name : org_name.c_str(),
            defaults,
            annotations,
            /*return_annotations*/ nullptr,
            /*is_variant*/ false,
            is_exported,
            context);

        return index;
    }
    else if (class_id == ID_MDL_FUNCTION_DEFINITION)
    {
        // get definition for obtaining default arguments and annotations
        DB::Access<Mdl_function_definition> prototype(prototype_tag, m_transaction);
        const MI::MDL::Mdl_function_definition* prototype_def = prototype.get_ptr();

        // check for unsupported functions
        if (!is_supported_prototype(prototype_def, /*for_variant=*/false))
            return add_error_message(
                context, "This kind of function is not supported as a prototype.", -5);

        // get the original name of the function
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

        mi::base::Handle<const MI::MDL::IExpression_list> prototype_defaults;
        mi::base::Handle<const MI::MDL::IAnnotation_block> prototype_annotations;
        mi::base::Handle<const MI::MDL::IAnnotation_block> prototype_return_annotations;

        // get defaults from prototype if requested
        if( defaults == nullptr) {
            prototype_defaults = prototype_def->get_defaults();
            defaults = prototype_defaults.get();
        }

        // get annotations from prototype if requested
        if( annotations == nullptr) {
            prototype_annotations = prototype->get_annotations();
            annotations = prototype_annotations.get();
        }

        // get return annotations from prototype if requested
        if( ret_annotations == nullptr) {
            prototype_return_annotations = prototype->get_return_annotations();
            ret_annotations = prototype_return_annotations.get();
        }

        // create the definition
        mi::Sint32 index = add_intern<Mdl_function_definition>(
            prototype_tag,
            (name != nullptr) ? name : org_name.c_str(),
            defaults,
            annotations,
            ret_annotations,
            /*is_variant*/ false,
            is_exported,
            context);

        return index;
    }

    ASSERT(M_SCENE, false);
    return add_error_message(
        context, "The DB element of one of the prototypes has the wrong type.", -5);
}

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

    SERIAL::Class_id class_id = m_transaction->get_class_id(prototype_tag);

    if (class_id == ID_MDL_MATERIAL_DEFINITION)
    {
        DB::Access<Mdl_material_definition> prototype;
        mi::base::Handle<const MI::MDL::IAnnotation_block> prototype_annotations;

        // get annotations from prototype if requested
        if( annotations == nullptr) {
            prototype.set(prototype_tag, m_transaction);
            prototype_annotations = prototype->get_annotations();
            annotations = prototype_annotations.get();
        }

        // create the definition
        mi::Sint32 index = add_intern<Mdl_material_definition>(
            prototype_tag,
            name,
            defaults,
            annotations,
            /*return_annotations*/ nullptr,
            /*is_variant*/ true,
            is_exported,
            context);

        return index;
    }
    else if (class_id == ID_MDL_FUNCTION_DEFINITION)
    {
        DB::Access<Mdl_function_definition> prototype;
        mi::base::Handle<const MI::MDL::IAnnotation_block> prototype_annotations;
        mi::base::Handle<const MI::MDL::IAnnotation_block> prototype_return_annotations;

        // get annotations from prototype if requested
        if( annotations == nullptr || return_annotations == nullptr) {
            prototype.set(prototype_tag, m_transaction);
            prototype_annotations = prototype->get_annotations();
            prototype_return_annotations = prototype->get_return_annotations();
            if (annotations == nullptr)
                annotations = prototype_annotations.get();
            if (return_annotations == nullptr)
                return_annotations = prototype_return_annotations.get();
        }

        // create the definition
        mi::Sint32 index = add_intern<Mdl_function_definition>(
            prototype_tag,
            name,
            defaults,
            annotations,
            return_annotations,
            /*is_variant*/ true,
            is_exported,
            context);

        return index;
    }

    ASSERT(M_SCENE, false);
    return add_error_message(
        context, "The DB element of one of the prototypes has the wrong type.", -5);
}

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

mi::Sint32 Mdl_module_builder::prepare_parameters(
    const Parameter_data* in_parameters,
    mi::Size param_count,
    std::vector<New_parameter>& out_parameters,
    const mi::base::Handle<const IExpression_list>& args,
    const mi::base::Handle<const IType_list>& param_types,
    Execution_context* context) const
{
    std::set<std::string> param_names;
    for (mi::Size i = 0; i < param_count; ++i) {

        const Parameter_data& param = in_parameters[i];
        if (!m_mdl->is_valid_mdl_identifier(param.m_name.c_str()))
            return add_error_message(
                context,
                "Parameter name '" + param.m_name + "' is not a valid MDL identifier.", -11);
        if (!param_names.insert(param.m_name).second) {
            return add_error_message(
                context,
                "Parameter name '" + param.m_name + "' is not unique.", -11);
        }

        mi::base::Handle<const IExpression> expr(find_path(m_transaction, param.m_path, args));
        if (!expr.is_valid_interface()) {
            return add_error_message(
                context, "Path '" + param.m_path + "' does not point to an expression", -13);
        }

        bool must_be_uniform = false;
        if (!can_enforce_uniform(
            m_transaction, args, param_types, param.m_path, expr, must_be_uniform)) {
            return add_error_message(
                context, "Parameter '" + param.m_name + "' cannot be enforced uniform.", -15);
        }

        out_parameters.push_back(New_parameter(
                m_name_factory->create_symbol(param.m_name.c_str()),
                expr,
                param.m_annotations,
                must_be_uniform || param.m_enforce_uniform));
    }
    return 0;
}

mi::Sint32 Mdl_module_builder::add_function(
    const Mdl_data* mdl_data,
    bool is_exported,
    MI::MDL::Execution_context* context)
{
    return add_function_intern<Mdl_function_call>(
        mdl_data, is_exported, context);
}

mi::Sint32 Mdl_module_builder::add_material(
    const Mdl_data* mdl_data,
    bool is_exported,
    MI::MDL::Execution_context* context)
{
    SERIAL::Class_id class_id = m_transaction->get_class_id(mdl_data->m_prototype_tag);
    if (class_id == MDL::Mdl_material_instance::id)
        return add_function_intern<Mdl_material_instance>(
            mdl_data, is_exported, context);
    else if (class_id == MDL::Mdl_function_call::id)
        return add_function_intern<Mdl_function_call>(
            mdl_data, is_exported, context);

    return -1;
}

template <typename T>
mi::Sint32 Mdl_module_builder::add_function_intern(
    const Mdl_data* md,
    bool is_exported,
    Execution_context* context)
{
    if (!clear_and_check_valid(context))
        return -1;

    // access prototype
    DB::Access<T> prototype(md->m_prototype_tag, m_transaction);
    ASSERT(M_SCENE, prototype);

    mi::base::Handle<const IExpression_list> args(prototype->get_arguments());
    if (!args.is_valid_interface() && !md->m_parameters.empty()) {
        return add_error_message(
            context, "Prototype does not have any parameters.", -6);
    }

    // check and setup parameters
    mi::Size param_size = md->m_parameters.size();
    std::vector<New_parameter> new_params;
    mi::base::Handle<const IType_list> param_types(prototype->get_parameter_types());

    if (prepare_parameters(
        md->m_parameters.data(), param_size,
        new_params,
        args, param_types, context) == -1) {
        return -1;
    }

    // convert annotations to MDL AST
    mi::mdl::IAnnotation_block* mdl_annotation_block = nullptr;
    mi::Sint32 result = create_annotations(
        md->m_annotations.get(), mdl_annotation_block, /*skip_unused=*/false);
    if (result != 0)
        return result;
    mi::mdl::IAnnotation_block* mdl_ret_annotation_block = nullptr;
    result = create_annotations(
        md->m_return_annotations.get(), mdl_ret_annotation_block, /*skip_unused=*/false);
    if (result != 0)
        return result;

    // create return type name
    const mi::mdl::IType_name *ret_type_tn;
    if (md->m_is_material) {
        const mi::mdl::ISymbol* ret_tp_sym = m_name_factory->create_symbol("material");
        const mi::mdl::ISimple_name* ret_tp_sname = m_name_factory->create_simple_name(ret_tp_sym);
        mi::mdl::IQualified_name* ret_tp_qname = m_name_factory->create_qualified_name();
        ret_tp_qname->add_component(ret_tp_sname);
        ret_type_tn = m_name_factory->create_type_name(ret_tp_qname);
    }
    else {
        ret_type_tn = create_return_type_name(m_transaction, m_module.get(), prototype);
    }
    m_symbol_importer->collect_imports(ret_type_tn);

    mi::mdl::IDefinition::Semantics sema = get_mdl_semantic(prototype);
    // some semantics require special handling
    bool maps_to_mdl_function = true;
    if (semantic_is_operator(sema) || 
        sema == mi::mdl::IDefinition::DS_INTRINSIC_DAG_FIELD_ACCESS) {
        maps_to_mdl_function = false;
    }

    // create name
    const mi::mdl::ISymbol* fct_sym = m_name_factory->create_symbol(md->m_definition_name.c_str());
    const mi::mdl::ISimple_name* fct_sname = m_name_factory->create_simple_name(fct_sym);

    // setup mdl AST builder
    Mdl_ast_builder ast_builder(m_module.get(), m_transaction, args, *m_name_mangler);
    // add parameters
    for (const auto& pd : new_params) {
        ast_builder.declare_parameter(pd.get_sym(), pd.get_init());
    }

    // create the body

    mi::mdl::IExpression_call *call = nullptr;
    const char *definition_name = get_mdl_definition(prototype);
    if (maps_to_mdl_function) {

        // call to prototype
        const mi::mdl::IExpression_reference *ref = signature_to_reference(
            m_module.get(),
            definition_name,
            m_name_mangler);
        call = m_expression_factory->create_call(ref);
    }

    // type specific setup
    mi::mdl::IStatement_compound *fbody = nullptr;
    mi::mdl::IExpression_let *mlet = nullptr;
    if (md->m_is_material) {
        ASSERT(M_SCENE, maps_to_mdl_function);
        mlet = m_expression_factory->create_let(call);
    }
    else {
        fbody = m_statement_factory->create_compound();
    }

    // setup variables
    mi::Size n_callee_params = args->get_size();
    std::vector<New_parameter> new_variables;

    if (n_callee_params > 0) {

        // Note: the temporaries are create with "auto" type
        for (mi::Size i = 0; i < n_callee_params; ++i) {
            mi::base::Handle<const IExpression> arg(args->get_expression(i));
            mi::base::Handle<const IType> arg_tp(arg->get_type());

            mi::mdl::IType_name *tn = ast_builder.create_type_name(arg_tp);
            m_symbol_importer->collect_imports(tn);

            mi::mdl::IDeclaration_variable *vdecl = m_declaration_factory->create_variable(
                tn, /*exported=*/false);
            const mi::mdl::IExpression* init = ast_builder.transform_expr(arg);
            const mi::mdl::ISymbol* tmp_sym = ast_builder.get_temporary_symbol();

            vdecl->add_variable(ast_builder.to_simple_name(tmp_sym), init);

            if (md->m_is_material)
                mlet->add_declaration(vdecl);
            else {
                mi::mdl::IStatement_declaration *sdecl = m_statement_factory->create_declaration(vdecl);
                fbody->add_statement(sdecl);
            }

            const char* pname = args->get_name(i);
            const mi::mdl::ISymbol* psym = m_name_factory->create_symbol(pname);
            const mi::mdl::ISimple_name *psname = ast_builder.to_simple_name(psym);
            const mi::mdl::IExpression_reference *ref = ast_builder.to_reference(tmp_sym);

            if (maps_to_mdl_function)
                call->add_argument(m_expression_factory->create_named_argument(psname, ref));
            else {
                new_variables.push_back(New_parameter(tmp_sym, arg, mi::base::Handle<IAnnotation_block>(), false));
            }

            if (init)
                m_symbol_importer->collect_imports(init);
        }
    }
    const mi::mdl::IExpression* res_call = nullptr;
    if (call) {
        res_call = mi::mdl::promote_expressions_to_mdl_version(m_module.get(), call);
        if (!res_call)
        {
            return add_error_message(
                context, "Failed to promote expression.", -8);
        }
        if (mlet)
            mlet->set_expression(res_call);
        m_symbol_importer->collect_imports(res_call);
    }
    mi::mdl::IStatement *body;
    if (md->m_is_material) {
        body = m_statement_factory->create_expression(mlet);
    }
    else {

        const mi::mdl::IExpression *ret_expr = nullptr;
        if (maps_to_mdl_function) {
            ret_expr = res_call;
        }
        else {
            ast_builder.remove_parameters();
            for (const auto& vd : new_variables)
                ast_builder.declare_parameter(vd.get_sym(), vd.get_init());

            mi::base::Handle<const IType> ret_type(get_return_type(prototype));
            std::string def = ast_builder.dag_unmangle(get_mdl_definition(prototype));
            ret_expr = ast_builder.transform_call(
                ret_type, sema, def, n_callee_params, args, true);
        }

        // 3) return statement for functions
        mi::mdl::IStatement_return *sret = m_statement_factory->create_return(ret_expr);
        fbody->add_statement(sret);

        body = fbody;
    }
   
    mi::mdl::IDeclaration_function *fdecl = m_declaration_factory->create_function(
        ret_type_tn,
        mdl_ret_annotation_block,
        fct_sname,
        /*is_clone*/false,
        body,
        mdl_annotation_block,
        /*is_exported=*/true);

    // add parameters
    ast_builder.remove_parameters();
    for (const auto& pd : new_params) {

        mi::base::Handle<const IType> ptype(pd.get_init()->get_type());
        mi::mdl::IType_name* tn = ast_builder.create_type_name(ptype);

        if (pd.is_uniform()) {
            tn->set_qualifier(mi::mdl::FQ_UNIFORM);
        }

        // work-around until the expression are correctly typed: resource parameters
        // must be uniform
        IType::Kind kind = ptype->get_kind();
        if (kind == IType::TK_TEXTURE || kind == IType::TK_BSDF_MEASUREMENT ||
            kind == IType::TK_LIGHT_PROFILE)
        {
            tn->set_qualifier(mi::mdl::FQ_UNIFORM);
        }

        const mi::mdl::ISimple_name* sname = m_name_factory->create_simple_name(pd.get_sym());
        const mi::mdl::IExpression* init = ast_builder.transform_expr(pd.get_init());

        mi::mdl::IAnnotation_block* p_annos = 0;
        mi::Sint32 result = create_annotations(
            pd.get_annos().get(), p_annos, /*skip_unused=*/false);
        if (result != 0)
            return result;

        const mi::mdl::IParameter* param = m_declaration_factory->create_parameter(tn, sname, init, p_annos);
        fdecl->add_parameter(param);
        if (init)
            m_symbol_importer->collect_imports(init);
    }
    // add types seen by the ast builder
    m_symbol_importer->add_names(ast_builder.get_used_user_types());

    // defer the actual adding to the module until the module is built
    m_added_functions.push_back(fdecl);
    m_added_function_annotations.push_back(m_annotation_factory->create_annotation_block());
    m_added_function_symbols.push_back(fct_sym);
    return m_added_functions.size() - 1; // get index
}

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
    for (int i = 0; i < mdl_annotation_block->get_annotation_count(); ++i)
        mdl_annotation_block->delete_annotation(i);

    // add the new ones
    for (mi::Size i = 0; annotations && i < annotations->get_size(); ++i) {
        mi::base::Handle<const IAnnotation> annotation(annotations->get_annotation(i));
        if (!add_annotation(index, annotation.get(), context))
            return false;
    }

    return true;
}

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
    if (result != 0)
    {
        context->add_error_message(
            MI::MDL::Message(mi::base::MESSAGE_SEVERITY_ERROR,
            MI::STRING::formatted_string("Failed to add annotation: %s.", anno_name),
            result, MI::MDL::Message::MSG_INTEGRATION));
        return false;
    }

    return true;
}

bool Mdl_module_builder::set_return_annotations(
    mi::Sint32 index,
    const MI::MDL::IAnnotation_block* annotations,
    MI::MDL::Execution_context* context)
{
    if (index >= static_cast<mi::Sint32>(m_added_functions.size()))
        return false;

    // convert the annotations block
    mi::mdl::IAnnotation_block* mdl_annotation_block = 0;
    if (!create_annotations(annotations, mdl_annotation_block, context, /*skip_unused=*/false))
        return false;

    // attach the block to the material
    mi::mdl::IDeclaration_function* declaration = m_added_functions[index];
    declaration->set_return_annotation(mdl_annotation_block);
    return true;
}

const mi::mdl::IModule* Mdl_module_builder::build(
    MI::MDL::Execution_context* context)
{
    if (!clear_and_check_valid(context)) return nullptr;

    // add alias declarations
    const Name_mangler::Name_map& mangled_names = m_name_mangler->get_names();
    if (mangled_names.size() > 0) {
        mi::mdl::IDeclaration_factory *df = m_module->get_declaration_factory();
        mi::mdl::IName_factory *nf = m_module->get_name_factory();
        for (const auto& entry : mangled_names) {

            const std::string& alias_str = entry.second;
            const std::string& aliased_str = entry.first;

            const mi::mdl::ISimple_name* alias_name = nf->create_simple_name(nf->create_symbol(alias_str.c_str()));

            mi::mdl::IQualified_name* aliased_name = nf->create_qualified_name();
            const mi::mdl::ISimple_name* sn = nf->create_simple_name(nf->create_symbol(aliased_str.c_str()));
            aliased_name->add_component(sn);

            mi::mdl::IDeclaration_namespace_alias* alias_decl = df->create_namespace_alias(alias_name, aliased_name);
            m_module->add_declaration(alias_decl);
        }
    }

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

    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module(false);
    Module_cache module_cache(m_transaction, mdlc_module->get_module_wait_queue(), {});
    m_module->analyze(&module_cache, m_thread_context.get());
    if (!m_module->is_valid())
    {
        report_messages(m_module->access_messages(), context);
        add_error_message(
            context,
            "Unspecified error. Failed to build a valid module from selected elements.", -8);
        return nullptr;
    }

    if (m_inline_mdle && m_symbol_importer->imports_mdle()) {
        mi::base::Handle<mi::mdl::IMDL_module_transformer> module_transformer(
            m_mdl->create_module_transformer());
        mi::base::Handle<const mi::mdl::IModule> inlined_module(
            module_transformer->inline_mdle(m_module.get()));
        if (!inlined_module) {
            report_messages(module_transformer->access_messages(), context);
            add_error_message(
                context,
                "Unspecified error. Failed to create inlined module.", -8);
            return nullptr;
        }
        inlined_module->retain();
        return inlined_module.get();
    }

    m_module->retain();
    return m_module.get();
}

bool Mdl_module_builder::clear_and_check_valid(MI::MDL::Execution_context* context)
{
    context->clear_messages();

    if (m_module)
        return true;

    add_error_message(
        context,
        "Module builder is in invalid state. "
        "build() is already completed or a previous step failed.",
        -8);

    return false;
}

bool Mdl_module_builder::create_annotations(
    const MI::MDL::IAnnotation_block* annotation_block,
    mi::mdl::IAnnotation_block* &mdl_annotation_block,
    MI::MDL::Execution_context* context,
    bool skip_unused)
{
    mi::Sint32 result = create_annotations(annotation_block, mdl_annotation_block, skip_unused);

    switch (result)
    {
        case 0:
            return true;

        case -9:
        {
            add_error_message(
                context,
                "One of the annotation arguments is wrong "
                "(wrong argument name, not a constant expression, or the "
                "argument type does not match the parameter type)",
                result);
            return false;
        }
        case -10:
        {
            add_error_message(
                context,
                "One of the annotations does not exist or it has a currently "
                "unsupported parameter type like deferred-sized arrays.",
                result);

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
    mi::mdl::IAnnotation_block* &mdl_annotation_block,
    bool skip_unused)
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
        if (skip_unused && strcmp(anno_name, "::anno::unused()") == 0)
            continue;
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

    // handle leading slash on windows
    size_t mdle_prefix_slash_offset = 0;
    if (is_mdle_annotation && 
        annotation_name_wo_signature[0] == ':' &&
        annotation_name_wo_signature[1] == ':' &&
        annotation_name_wo_signature[2] == '/' &&
        isalpha(annotation_name_wo_signature[3]) &&
        annotation_name_wo_signature[4] == ':')
    {
        // the module name does not have this extra slash, only the db_name
        annotation_name_wo_signature[2] = ':';
        annotation_name_wo_signature = annotation_name_wo_signature.substr(1);
        mdle_prefix_slash_offset = 1;
    }

    std::string signature = annotation_name_str.substr(
        left_paren + 1, annotation_name_str.size() - left_paren - 2);
    const mi::mdl::IDefinition* definition = anno_mdl_module->find_annotation(
        annotation_name_wo_signature.c_str(), signature.c_str());
    if (!definition)
        return -10;

    // compute IQualified_name for annotation name
    mi::mdl::IQualified_name* anno_qualified_name = m_name_factory->create_qualified_name();
    anno_qualified_name->set_absolute();
    size_t start = prefix_length + 2 + // skip leading "mdle::" or "mdl::"
                   mdle_prefix_slash_offset; // additional '/' on windows
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

        // The legacy API always provided "argument" as argument name. Since it supported only
        // single string arguments we mapped that argument name to the correct one if all these
        // conditions are met. This was also done for the current API since we were not able to
        // tell them apart here.
        if (i == 0
            && parameter_count == 1
            && argument_count == 1
            && strcmp(arg_name, "argument") == 0
            && arg_type->get_kind() == IType::TK_STRING)
        {
            const char* correct_arg_name = parameter_types.begin()->first.c_str();
            if (strcmp( arg_name, correct_arg_name) != 0)
            {
                LOG::mod_log->warning( M_SCENE, LOG::Mod_log::C_DATABASE,
                    "Use of dummy annotation argument name \"argument\" is deprecated.");
                arg_name = correct_arg_name;
            }
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
