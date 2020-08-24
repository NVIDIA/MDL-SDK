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

// examples/mdl_sdk/traversal/compiled_material_traverser_print.cpp

#include "compiled_material_traverser_print.h"

#include <iostream>
#include <algorithm> 
#include <utility>

// forward declaration of local helper functions
static std::string replace_all(const std::string& input,
                               const std::string& old, const std::string& with);

// check semantics of function calls
static bool is_constructor(mi::neuraylib::IFunction_definition::Semantics semantic);
static bool is_type_converter(mi::neuraylib::IFunction_definition::Semantics semantic);
static bool is_array_constructor(mi::neuraylib::IFunction_definition::Semantics semantic);
static bool is_array_index_operator(mi::neuraylib::IFunction_definition::Semantics semantic);
static bool is_unary_operator(mi::neuraylib::IFunction_definition::Semantics semantic);
static bool is_binary_operator(mi::neuraylib::IFunction_definition::Semantics semantic);
static bool is_ternary_operator(mi::neuraylib::IFunction_definition::Semantics semantic);
static bool is_selector_operator(mi::neuraylib::IFunction_definition::Semantics semantic);
static bool is_call_like_operator(mi::neuraylib::IFunction_definition::Semantics semantic);

Compiled_material_traverser_print::Context::Context(
    mi::neuraylib::ITransaction* transaction, 
    bool keep_structure)
    : m_transaction(transaction)
    , m_keep_compiled_material_structure(keep_structure)
{
    reset();
}


void Compiled_material_traverser_print::Context::reset()
{
    m_print.str("");
    m_print.clear();
    m_indent = 0;
    m_imports.clear();

    m_used_modules.clear();
    m_used_resources.clear();

    m_parameters_to_inline.clear();
    m_print_inline_swap.str("");
    m_print_inline_swap.clear();
    m_indent_inline_swap = 0;

    m_is_valid_mdl = true;
    m_stage = Compiled_material_traverser_base::ES_NOT_STARTED;
}


std::string Compiled_material_traverser_print::print_mdl(
    const mi::neuraylib::ICompiled_material* material,
    Context& context,
    const std::string& original_module_name,
    const std::string& output_material_name)
{
    // reset in case of reuse
    context.reset();

    // run the traversal and print MDL to the string stream of the context
    Compiled_material_traverser_base::traverse(material, &context);

    // version string
    std::stringstream output;
    output << "mdl 1.6;\n\n";

    // add required includes
    size_t last_sep_pos = std::string::npos;
    std::string last_module("");
    for (std::set<std::string>::iterator it = context.m_imports.begin();
         it != context.m_imports.end(); ++it)
    {
        const size_t current_sep_pos = it->rfind("::");
        if (current_sep_pos != last_sep_pos
            || it->find(last_module) != 0)
        {
            last_module = it->substr(0, current_sep_pos);
            last_sep_pos = current_sep_pos;
            output << "// import " << last_module << "::*;\n";
            context.m_used_modules.insert(last_module);
        }
        if (current_sep_pos == 0) // show imports of the base namespace (just to list them up)
            output << "//* ";

        output << "import " << it->c_str() << ";\n";
    }

    // ... and other information not directly part of the compiled material.
    // Here, we assume that all used functions defined in the original module are exported, too.
    output << "// import original package\n";
    output << "import " << original_module_name << "::*;\n\n";
    output << "export material " << output_material_name << "";

    // append the result of the traversal
    output << context.m_print.str();

    return output.str();
}

//--------------------------------------------------------------------------------------------------
// Stage functions
//--------------------------------------------------------------------------------------------------

void Compiled_material_traverser_print::stage_begin(
    const mi::neuraylib::ICompiled_material* material, Traveral_stage stage, void* context)
{
    Context* ctx = static_cast<Compiled_material_traverser_print::Context*>(context);
    ctx->m_stage = stage;

    switch (stage)
    {
        case ES_PARAMETERS:
        {
            ctx->m_print << "(\n";
            break;
        }

        case ES_TEMPORARIES:
        {
            ctx->m_print << (material->get_temporary_count() == 0 ? " = " : " = let{\n");
            break;
        }

        case ES_BODY:
        default:
            break;
    }
}

void Compiled_material_traverser_print::stage_end(const mi::neuraylib::ICompiled_material* material,
                                                  Traveral_stage stage, void* context)
{
    Context* ctx = static_cast<Compiled_material_traverser_print::Context*>(context);

    switch (stage)
    {
        case ES_PARAMETERS:
        {
            ctx->m_print << ")\n";
            // at this point one could add annotations here,
            // if they are made available to the context
            break;
        }

        case ES_TEMPORARIES:
        {
            ctx->m_print << (material->get_temporary_count() == 0 ? "" : "} in ");
            break;
        }

        case ES_BODY:
        {
            ctx->m_print << ";\n";
            break;
        }

        default:
            break;
    }
}

//--------------------------------------------------------------------------------------------------
// Traversal functions
//--------------------------------------------------------------------------------------------------

void Compiled_material_traverser_print::visit_begin(
    const mi::neuraylib::ICompiled_material* material,
    const Traversal_element& element, void* context)
{
    Context* ctx = static_cast<Compiled_material_traverser_print::Context*>(context);
    ctx->m_indent++;

    // major cases: parameter, temporary, expression or value
    if (element.expression)
    {
        switch (element.expression->get_kind())
        {
            case mi::neuraylib::IExpression::EK_CONSTANT:
                return; // nothing to print here

            case mi::neuraylib::IExpression::EK_CALL:
                return; // for compiled materials, this will not happen

            case mi::neuraylib::IExpression::EK_PARAMETER:
            {
                mi::base::Handle<const mi::neuraylib::IExpression_parameter> expr_param(
                    element.expression->get_interface<const mi::neuraylib::IExpression_parameter
                    >());

                // get the parameter name 
                bool generated;
                std::string name = get_parameter_name(material, expr_param->get_index(), 
                                                      &generated);

                // if we choose to inline generated parameters, we do it here
                if (!ctx->m_keep_compiled_material_structure && generated)
                {
                    // to get the right indentation, we need to add the indent for each line
                    std::string to_inline = ctx->m_parameters_to_inline[name];
                    ctx->m_print << replace_all(to_inline, "\n", "\n" + indent(ctx));
                    ctx->m_print << "/*inlined generated param*/";
                }
                // otherwise handle generated parameters like any other
                else
                {
                    ctx->m_print << name << "/*param*/";
                }
                
                return;
            }

            case mi::neuraylib::IExpression::EK_DIRECT_CALL:
            {
                const mi::base::Handle<const mi::neuraylib::IExpression_direct_call> expr_dcall(
                    element.expression->get_interface<const mi::neuraylib::IExpression_direct_call
                    >());
                const mi::base::Handle<const mi::neuraylib::IExpression_list> args(
                    expr_dcall->get_arguments());
                const mi::base::Handle<const mi::neuraylib::IFunction_definition> func_def(
                    ctx->m_transaction->access<mi::neuraylib::IFunction_definition>(
                        expr_dcall->get_definition()));
                const mi::neuraylib::IFunction_definition::Semantics semantic = func_def->
                    get_semantic();

                std::string module_name = func_def->get_mdl_module_name();
                bool is_builtins = module_name.substr(0, 12) == "::<builtins>";

                std::string function_name;
                if (is_builtins)
                    function_name = func_def->get_mdl_simple_name();
                else
                    function_name = module_name + "::" +  func_def->get_mdl_simple_name();

                // keep track of used modules and/or imports
                const std::string m = func_def->get_module();
                if (!is_builtins)
                {
                    // type conversion using constructors can lead to invalid mdl code
                    // as these conversion constructors are created in the local module space
                    if (is_type_converter(semantic))
                    {
                        // Keep imports for structure definitions and arrays of structures as
                        // well as enums and arrays of enums as they can be user defined.
                        // For ther types, we can drop the qualification part
                        bool drop_qualification = true;

                        const mi::base::Handle<const mi::neuraylib::IType> return_type(
                            func_def->get_return_type());

                        const mi::base::Handle<const mi::neuraylib::IType_struct> 
                            return_type_struct(
                                return_type->get_interface<const mi::neuraylib::IType_struct>());

                        const mi::base::Handle<const mi::neuraylib::IType_enum>
                            return_type_enum(
                                return_type->get_interface<const mi::neuraylib::IType_enum>());
                        
                        const mi::base::Handle<const mi::neuraylib::IType_array> return_type_array(
                            return_type->get_interface<const mi::neuraylib::IType_array>());

                        if (return_type_struct || return_type_enum)
                        {
                            drop_qualification = false;
                        }
                        else if (return_type_array)
                        {
                            const mi::base::Handle<const mi::neuraylib::IType> return_type_array_e(
                                return_type_array->get_element_type());

                            const mi::base::Handle<const mi::neuraylib::IType_struct>
                                return_type_e_struct(
                                    return_type_array_e->get_interface<
                                    const mi::neuraylib::IType_struct>());

                            const mi::base::Handle<const mi::neuraylib::IType_enum>
                                return_type_e_enum(
                                    return_type_array_e->get_interface<
                                    const mi::neuraylib::IType_enum>());

                            if (return_type_e_struct || return_type_e_enum)
                                drop_qualification = false;
                        }

                        if (drop_qualification)
                        {
                            // strip qualification part of the name
                            function_name = func_def->get_mdl_simple_name();
                        }
                        else
                        {
                            // add the struct or enum definition to the imports 
                            ctx->m_imports.insert(function_name);
                        }
                    }
                    // everything before the dot needs to be imported
                    else if (is_selector_operator(semantic))
                    {
                        // if this is a qualified name (and not the name of variable)
                        if (function_name.rfind("::") != std::string::npos)
                        {
                            const size_t pos_dot = function_name.find('.');
                            ctx->m_imports.insert(function_name.substr(0, pos_dot));
                        }
                    }
                    // - ternary operator can have non build-in types in the signature and we don't
                    //   need an import for them
                    // - the index operator can introduce no new type as the array type is known
                    else if (is_ternary_operator(semantic) || is_array_index_operator(semantic))
                    {
                        // nothing to do
                    }
                    // general case, import the function name
                    else
                    {
                        ctx->m_imports.insert(function_name);
                    }
                }

                // array constructors are one special case of constructor
                if (is_array_constructor(semantic))
                {
                    // array type is the type of the first parameter
                    const mi::base::Handle<const mi::neuraylib::IExpression> arg0(
                        args->get_expression(mi::Size(0)));
                    const mi::base::Handle<const mi::neuraylib::IType> arg0_type(arg0->get_type());
                    const std::string typeString = type_to_string(arg0_type.get(), ctx);
                    ctx->m_print << typeString << "[](/*array constructor*/";
                    return;
                }

                // check for special cases based on the semantic
                if (   is_selector_operator(semantic)
                    || is_unary_operator(semantic)
                    || is_array_index_operator(semantic))
                {
                    return;
                }

                if (   is_binary_operator(semantic)
                    || is_ternary_operator(semantic))
                {
                    ctx->m_print << "(";
                    return;
                }

                if (is_type_converter(semantic))
                {
                    ctx->m_print << function_name << "(";
                    return;
                }

                mi::Size arg_count = func_def->get_parameter_count();
                if (is_constructor(semantic))
                {
                    if(arg_count > 0)
                        ctx->m_print << function_name << "(/*constructor*/";
                    else
                        ctx->m_print << function_name << "()/*constructor*/";
                    return;
                }

                if (is_call_like_operator(semantic))
                {
                    if (arg_count > 0)
                        ctx->m_print << function_name << "(/*call*/";
                    else
                        ctx->m_print << function_name << "()/*call*/";
                    return;
                }

                // error case (should not happen)
                std::cerr << "[Compiled_material_traverser_print] ran into unhandled semantic: '"
                    << function_name << "' Semantic:" << semantic << "\n";
                return;
            }

            case mi::neuraylib::IExpression::EK_TEMPORARY:
            {
                const mi::base::Handle<const mi::neuraylib::IExpression_temporary> expr_temp(
                    element.expression->get_interface<const mi::neuraylib::IExpression_temporary
                    >());
                ctx->m_print << get_temporary_name(material, expr_temp->get_index());
                return;
            }

            case mi::neuraylib::IExpression::EK_FORCE_32_BIT:
                return; // not a valid case
        }
        return;
    }

    // major cases: parameter, temporary, expression or value
    else if (element.value)
    {
        switch (element.value->get_kind())
        {
            case mi::neuraylib::IValue::VK_BOOL:
            {
                const mi::base::Handle<const mi::neuraylib::IValue_bool> value_bool(
                    element.value->get_interface<const mi::neuraylib::IValue_bool>());
                ctx->m_print << (value_bool->get_value() ? "true" : "false");
                return;
            }

            case mi::neuraylib::IValue::VK_INT:
            {
                const mi::base::Handle<const mi::neuraylib::IValue_int> value_int(
                    element.value->get_interface<const mi::neuraylib::IValue_int>());
                ctx->m_print << value_int->get_value();
                return;
            }

            case mi::neuraylib::IValue::VK_ENUM:
            {
                const mi::base::Handle<const mi::neuraylib::IValue_enum> value_enum(
                    element.value->get_interface<const mi::neuraylib::IValue_enum>());

                const mi::base::Handle<const mi::neuraylib::IType_enum> type_enum(
                    value_enum->get_type());

                std::string enum_name = enum_type_to_string(type_enum.get(), ctx);
                std::string enum_namespace = enum_name.substr(0, enum_name.rfind("::"));

                ctx->m_print << enum_namespace << "::" << value_enum->get_name()
                             << "/*enum of type: '" << enum_name << "'*/";
                return;
            }

            case mi::neuraylib::IValue::VK_FLOAT:
            {
                const mi::base::Handle<const mi::neuraylib::IValue_float> value_float(
                    element.value->get_interface<const mi::neuraylib::IValue_float>());
                ctx->m_print << std::showpoint << value_float->get_value() << "f";
                return;
            }

            case mi::neuraylib::IValue::VK_DOUBLE:
            {
                const mi::base::Handle<const mi::neuraylib::IValue_double> value_double(
                    element.value->get_interface<const mi::neuraylib::IValue_double>());
                ctx->m_print << std::showpoint << value_double->get_value();
                return;
            }

            case mi::neuraylib::IValue::VK_STRING:
            {
                const mi::base::Handle<const mi::neuraylib::IValue_string> value_string(
                    element.value->get_interface<const mi::neuraylib::IValue_string>());
                ctx->m_print << "\"" << value_string->get_value() << "\"";
                return;
            }

            case mi::neuraylib::IValue::VK_VECTOR:
            {
                const mi::base::Handle<const mi::neuraylib::IValue_vector> value_vector(
                    element.value->get_interface<const mi::neuraylib::IValue_vector>());
                const mi::base::Handle<const mi::neuraylib::IType_vector> vector_type(
                    value_vector->get_type());
                ctx->m_print << vector_type_to_string(vector_type.get(), ctx) << "(";
                return;
            }

            case mi::neuraylib::IValue::VK_MATRIX:
            {
                const mi::base::Handle<const mi::neuraylib::IValue_matrix> value_matrix(
                    element.value->get_interface<const mi::neuraylib::IValue_matrix>());
                const mi::base::Handle<const mi::neuraylib::IType_matrix> matrix_type(
                    value_matrix->get_type());
                ctx->m_print << matrix_type_to_string(matrix_type.get(), ctx) << "(";
                return;
            }

            case mi::neuraylib::IValue::VK_COLOR:
            {
                ctx->m_print << "color(";
                return;
            }

            case mi::neuraylib::IValue::VK_ARRAY:
            {
                const mi::base::Handle<const mi::neuraylib::IType> type(element.value->get_type());
                const mi::base::Handle<const mi::neuraylib::IType_array> array_type(
                    type.get_interface<const mi::neuraylib::IType_array>());
                ctx->m_print << array_type_to_string(array_type.get(), ctx) << "(";
                return;
            }

            case mi::neuraylib::IValue::VK_STRUCT:
            {
                const mi::base::Handle<const mi::neuraylib::IValue_struct> value_struct(
                    element.value->get_interface<const mi::neuraylib::IValue_struct>());
                const mi::base::Handle<const mi::neuraylib::IType_struct> struct_type(
                    value_struct->get_type());

                // there are special mdl keywords that are handled as struct internally
                bool is_keyword;
                std::string type_name = struct_type_to_string(struct_type.get(), ctx, &is_keyword);
                ctx->m_print << type_name << "(";

                if (!is_keyword) 
                    ctx->m_print << "/*struct*/";
                return;
            }

            case mi::neuraylib::IValue::VK_INVALID_DF:
            {
                const mi::base::Handle<const mi::neuraylib::IValue_invalid_df> invalid_df(
                    element.value->get_interface<const mi::neuraylib::IValue_invalid_df>());
                const mi::base::Handle<const mi::neuraylib::IType_reference> ref_type(
                    invalid_df->get_type());

                // use the default constructor for distribution functions
                ctx->m_print << type_to_string(ref_type.get(), ctx) << "()";
                return;
            }

            case mi::neuraylib::IValue::VK_TEXTURE:
            {
                const mi::base::Handle<const mi::neuraylib::IValue_texture> texture_value(
                    element.value->get_interface<const mi::neuraylib::IValue_texture>());
                const mi::base::Handle<const mi::neuraylib::IType_texture> texture_type(
                    texture_value->get_type());
                ctx->m_print << type_to_string(texture_type.get(), ctx) << "(";

                // path
                const char* mdl_path = texture_value->get_file_path();
                if (mdl_path)
                {
                    ctx->m_print << '\"' << mdl_path << '\"';
                    ctx->m_used_resources.insert(mdl_path); // keep track of this information
                
                    // get the texture for gamma value
                    const mi::base::Handle<const mi::neuraylib::ITexture> texture(
                        ctx->m_transaction->access<mi::neuraylib::ITexture>(
                        texture_value->get_value()));

                    if (texture)
                    {
                        // gamma
                        const float gamma = texture->get_gamma();
                        if (gamma == 0.0)
                        {
                            ctx->m_print << ", gamma: ::tex::gamma_default";
                        }
                        else if (gamma == 1.0)
                        {
                            ctx->m_print << ", gamma: ::tex::gamma_linear";
                        }
                        else if (gamma == 2.2)
                        {
                            ctx->m_print << ", gamma: ::tex::gamma_srgb";
                        }
                        ctx->m_imports.insert("::tex::gamma_mode");
                    }
                    else
                    {
                        // when the compiler removes unresolved resources, this is never reached
                        std::cerr << "[Compiled_material_traverser_print] unsolved texture: '"
                            << mdl_path << "'. The default gamma value might be incorrect.\n";
                    }

                }
                ctx->m_print << ")";
                return;
            }

            case mi::neuraylib::IValue::VK_LIGHT_PROFILE:
            {
                const mi::base::Handle<const mi::neuraylib::IValue_light_profile> val_light_profile(
                    element.value->get_interface<const mi::neuraylib::IValue_light_profile>());

                const mi::base::Handle<const mi::neuraylib::IType_light_profile> type_light_profile(
                    val_light_profile->get_type());

                ctx->m_print << type_to_string(type_light_profile.get(), ctx) << "(";

                const char* mdl_path = val_light_profile->get_file_path();
                if (mdl_path)
                {
                    ctx->m_used_resources.insert(mdl_path); // keep track of this information
                    ctx->m_print << '\"' << mdl_path << '\"';
                }

                ctx->m_print << ")";
                return;
            }

            case mi::neuraylib::IValue::VK_BSDF_MEASUREMENT:
            {
                const mi::base::Handle<const mi::neuraylib::IValue_bsdf_measurement>
                    val_bsdf_measurement(
                        element.value->get_interface<const mi::neuraylib::IValue_bsdf_measurement
                        >());
                const mi::base::Handle<const mi::neuraylib::IType_bsdf_measurement>
                    type_bsdf_measurement(val_bsdf_measurement->get_type());

                ctx->m_print << type_to_string(type_bsdf_measurement.get(), ctx) << "(";

                const char* mdl_path = val_bsdf_measurement->get_file_path();
                if (mdl_path)
                {
                    ctx->m_used_resources.insert(mdl_path); // keep track of this information
                    ctx->m_print << '\"' << mdl_path << '\"';
                }

                ctx->m_print << ")";
                return;
            }

            default:
                return;
        }
    }

    // major cases: parameter, temporary, expression or value
    else if (element.parameter)
    {
        bool generated;
        std::string name = get_parameter_name(material, element.sibling_index, &generated);

        // in case we want to inline the compiler generated parameters
        // we temporarily swap the the stream for the time we process the parameter
        // we will swap back in the 'visit_end' method
        if (!ctx->m_keep_compiled_material_structure && generated)
        {
            {
                // c++11
                //std::swap(ctx->m_print, ctx->m_print_inline_swap);

                // c++0x (knowing that ctx->m_print will be discarded afterwards)
                ctx->m_print_inline_swap.str("");
                ctx->m_print_inline_swap.clear();
                ctx->m_print_inline_swap << ctx->m_print.str();
            }
            ctx->m_print.str("");
            ctx->m_print.clear();

            std::swap(ctx->m_indent, ctx->m_indent_inline_swap);
            ctx->m_indent = 0;
        }
        // if we don't line, we need to define a parameter
        else
        {
            mi::base::Handle<const mi::neuraylib::IType> type(element.parameter->value->get_type());
            // assuming all parameters are uniforms at this point
            ctx->m_print << indent(ctx) << "uniform " 
                         << type_to_string(type.get(), ctx) << " " <<
                name << " = ";
        }
        return;
    }

    // major cases: parameter, temporary, expression or value
    else if (element.temporary)
    {
        mi::base::Handle<const mi::neuraylib::IType> temporary_return_type(
            element.temporary->expression->get_type());

        // include the return type if required
        const std::string return_type = type_to_string(temporary_return_type.get(), ctx);
        if(return_type.rfind("::") != std::string::npos) // no '::' has to be a build-in type
            ctx->m_imports.insert(return_type);             

        ctx->m_print << indent(ctx) << return_type << " " << get_temporary_name(
            material, element.sibling_index) << " = ";
        return;
    }
}

void Compiled_material_traverser_print::visit_child(
    const mi::neuraylib::ICompiled_material* /*material*/,
    const Traversal_element& element, mi::Size /*children_count*/,
    mi::Size child_index, void* context)
{
    Context* ctx = static_cast<Compiled_material_traverser_print::Context*>(context);

    // major cases: parameter, temporary, expression or value
    if (element.expression)
    {
        switch (element.expression->get_kind())
        {
            case mi::neuraylib::IExpression::EK_DIRECT_CALL:
            {
                const mi::base::Handle<const mi::neuraylib::IExpression_direct_call> expr_dcall(
                    element.expression->get_interface<const mi::neuraylib::IExpression_direct_call
                    >());
                const mi::base::Handle<const mi::neuraylib::IExpression_list> arguments(
                    expr_dcall->get_arguments());
                const mi::base::Handle<const mi::neuraylib::IFunction_definition> func_def(
                    ctx->m_transaction->access<mi::neuraylib::IFunction_definition>(
                        expr_dcall->get_definition()));
                const mi::neuraylib::IFunction_definition::Semantics semantic = func_def->
                    get_semantic();
                std::string function_name = func_def->get_mdl_simple_name();

                // check for special cases based on the semantic

                if (is_selector_operator(semantic))
                    return;

                if (is_unary_operator(semantic))
                {
                    std::string op = function_name.substr(8);
                    ctx->m_print << op;
                    return;
                }

                if (is_array_index_operator(semantic))
                {
                    if (child_index == 1)
                        ctx->m_print << "[";
                    return;
                }

                if (is_binary_operator(semantic))
                {
                    if (child_index == 1)
                    {
                        std::string op = function_name.substr(8);
                        ctx->m_print << " " << op << " ";
                    }
                    return;
                }

                if (is_ternary_operator(semantic))
                {
                    if (child_index == 1)
                        ctx->m_print << "\n" << indent(ctx) << "? ";
                    else if (child_index == 2)
                        ctx->m_print << "\n" << indent(ctx) << ": ";
                    return;
                }


                // argument lists without line breaks
                if (is_type_converter(semantic))
                {
                    ctx->m_print << (child_index == 0 ? "" : ",");
                    return;
                }

                // default case: argument lists with line breaks
                // syntax of function call with parameters
                if (   is_call_like_operator(semantic)
                    || is_constructor(semantic))
                {
                    ctx->m_print << (child_index == 0 ? "\n" : ",\n") << indent(ctx) 
                                 << arguments->get_name(child_index) << ": "; // named parameters
                    return;
                }

                if (is_array_constructor(semantic))
                {
                    ctx->m_print << (child_index == 0 ? "\n" : ",\n") << indent(ctx); // no names
                    return;
                }
                
                // error case (should not happen):
                std::cerr << "[Compiled_material_traverser_print] ran into unhandled semantic: '"
                          << func_def->get_mdl_name() << "' Semantic:" << semantic << "\n";
                return;
            }

            default:
                return;
        }
    }

    // major cases: parameter, temporary, expression or value
    else if (element.value)
    {
        switch (element.value->get_kind())
        {
            case mi::neuraylib::IValue::VK_VECTOR:
            case mi::neuraylib::IValue::VK_COLOR:
            case mi::neuraylib::IValue::VK_ARRAY:
            {
                if (child_index != 0)
                    ctx->m_print << ", ";
                return;
            }

            case mi::neuraylib::IValue::VK_MATRIX:
            case mi::neuraylib::IValue::VK_STRUCT:
            {
                ctx->m_print << (child_index == 0 ? "\n" : ",\n") << indent(ctx);
                return;
            }

            default:
                return;
        }
    }
}

void Compiled_material_traverser_print::visit_end(
    const mi::neuraylib::ICompiled_material* material,
    const Traversal_element& element,
    void* context)
{
    Context* ctx = static_cast<Compiled_material_traverser_print::Context*>(context);

    // major cases: parameter, temporary, expression or value
    if (element.expression)
    {
        switch (element.expression->get_kind())
        {
            case mi::neuraylib::IExpression::EK_DIRECT_CALL:
            {
                const mi::base::Handle<const mi::neuraylib::IExpression_direct_call> expr_dcall(
                    element.expression->get_interface<const mi::neuraylib::IExpression_direct_call
                    >());
                const mi::base::Handle<const mi::neuraylib::IExpression_list> args(
                    expr_dcall->get_arguments());

                //* ctx->format.pop();
                const mi::base::Handle<const mi::neuraylib::IFunction_definition> func_def(
                    ctx->m_transaction->access<mi::neuraylib::IFunction_definition>(
                        expr_dcall->get_definition()));
                const mi::neuraylib::IFunction_definition::Semantics semantic = func_def->
                    get_semantic();


                if (is_unary_operator(semantic))
                    break;

                if (is_array_index_operator(semantic))
                {
                    ctx->m_print << "]";
                    break;
                }

                if (is_selector_operator(semantic))
                {
                    std::string selector = func_def->get_mdl_simple_name();
                    selector = selector.substr(selector.rfind('.'));
                    ctx->m_print << selector;
                    break;
                }

                // function call syntax without line break
                if (   is_binary_operator(semantic)
                    || is_ternary_operator(semantic))
                {
                    ctx->m_print << ")";
                    break;
                }

                if(is_type_converter(semantic))
                {
                    ctx->m_print << ")/*type conversion*/";
                    break;
                }

                // function call syntax with line break
                // contains standard functions
                if (   is_call_like_operator(semantic)
                    || is_constructor(semantic))
                {
                    mi::Size arg_count = func_def->get_parameter_count();
                    if (arg_count == 0)
                        break;

                    ctx->m_print << "\n" << indent(ctx, -1) << ")";
                    break;
                }

                if (is_array_constructor(semantic))
                {
                    // a bit special because the parameter count is zero 
                    // for the array constructor semantic
                    ctx->m_print << "\n" << indent(ctx, -1) << ")";
                    break;
                }

                // error case (should not happen):
                std::string name = func_def->get_mdl_name();
                std::cerr << "[Compiled_material_traverser_print] ran into unhandled semantic: '"
                    << name << "' Semantic:" << semantic << "\n";
                break;
            }
            default:
                break;;
        }
    }

    // major cases: parameter, temporary, expression or value
    else if (element.value)
    {
        switch (element.value->get_kind())
        {
            case mi::neuraylib::IValue::VK_VECTOR:
            case mi::neuraylib::IValue::VK_COLOR:
            case mi::neuraylib::IValue::VK_ARRAY:
            {
                ctx->m_print << ")";
                break;;
            }

            case mi::neuraylib::IValue::VK_STRUCT:
            case mi::neuraylib::IValue::VK_MATRIX:
            {
                ctx->m_print << "\n" << indent(ctx, -1) << ")";
                break;
            }

            default:
                break;
        }
    }

    // major cases: parameter, temporary, expression or value
    else if (element.parameter)
    {
        // we need to know if this is a generated parameter
        bool generated;
        std::string name = get_parameter_name(material, element.sibling_index, &generated);

        // optionally add annotations here (in the generated == false case)
        
        // in case we want to inline the compiler generated parameters
        // we temporarily swapped the the stream for the time we process the parameter
        // this happened in the 'visit_begin' method
        if (!ctx->m_keep_compiled_material_structure && generated)
        {
            // keep the printed code and swap back
            ctx->m_parameters_to_inline[name] = ctx->m_print.str();

            {
                // c++11
                //std::swap(ctx->m_print, ctx->m_print_inline_swap);
                
                // c++0x (knowing that ctx->m_print_inline_swap will be discarded)
                ctx->m_print.str("");
                ctx->m_print.clear();
                ctx->m_print << ctx->m_print_inline_swap.str();
            }

            std::swap(ctx->m_indent, ctx->m_indent_inline_swap);

            // there is a special case to keep the mdl output right we need to remove the
            // comma if the last parameter was a generated one.
            // In case there is no 'normal' parameter, we must not do that.
            if (element.sibling_index == element.sibling_count - 1)
            {
                // therefore we simply check if the last two characters are ",\n"
                std::string current = ctx->m_print.str();
                current = current.substr(std::max(size_t(0), current.length() - 2));

                // and if so, we replace them by "\n "
                if (current == ",\n")
                {
                    size_t pos = ctx->m_print.tellp();
                    ctx->m_print.seekp(pos - 2);
                    ctx->m_print << "\n ";
                }
            }
        }
        // if we don't inline, we finish the parameter definition
        else
        {
            ctx->m_print << ((element.sibling_index < element.sibling_count - 1) ? ",\n" : "\n");
        }
    }

    // major cases: parameter, temporary, expression or value
    else if (element.temporary)
    {
        ctx->m_print << ";\n";
    }

    ctx->m_indent--;
}


//--------------------------------------------------------------------------------------------------
// Helper functions
//--------------------------------------------------------------------------------------------------

const std::string Compiled_material_traverser_print::indent(
    const Context* context, mi::Sint32 offset) const
{
    size_t n = std::max(0, mi::Sint32(context->m_indent) + offset);
    return std::string(2 * n, ' ');
}

const std::string Compiled_material_traverser_print::type_to_string(
    const mi::neuraylib::IType* type,
    Compiled_material_traverser_print::Context* context)
{
    mi::base::Handle<const mi::neuraylib::IType_atomic> atomic_type(
        type->get_interface<const mi::neuraylib::IType_atomic>());
    if (atomic_type)
        return atomic_type_to_string(atomic_type.get(), context);

    switch (type->get_kind())
    {
        case mi::neuraylib::IType::TK_COLOR:
            return "color";
        case mi::neuraylib::IType::TK_STRUCT:
        {
            mi::base::Handle<const mi::neuraylib::IType_struct> struct_type(
                type->get_interface<const mi::neuraylib::IType_struct>());
            return struct_type_to_string(struct_type.get(), context);
        }
        case mi::neuraylib::IType::TK_ARRAY:
        {
            mi::base::Handle<const mi::neuraylib::IType_array> array_type(
                type->get_interface<const mi::neuraylib::IType_array>());
            return array_type_to_string(array_type.get(), context);
        }
        case mi::neuraylib::IType::TK_VECTOR:
        {
            mi::base::Handle<const mi::neuraylib::IType_vector> vector_type(
                type->get_interface<const mi::neuraylib::IType_vector>());
            return vector_type_to_string(vector_type.get(), context);
        }

        case mi::neuraylib::IType::TK_TEXTURE:
        {
            mi::base::Handle<const mi::neuraylib::IType_texture> texture_type(
                type->get_interface<const mi::neuraylib::IType_texture>());
            switch (texture_type->get_shape())
            {
                case mi::neuraylib::IType_texture::TS_2D:
                    return "texture_2d";

                case mi::neuraylib::IType_texture::TS_3D:
                    return "texture_3d";

                case mi::neuraylib::IType_texture::TS_CUBE:
                    return "texture_cube";

                case mi::neuraylib::IType_texture::TS_PTEX:
                    return "texture_ptex";

                default:
                    break;
            }
        }
            break;

        case mi::neuraylib::IType::TK_LIGHT_PROFILE:
            return "light_profile";

        case mi::neuraylib::IType::TK_BSDF_MEASUREMENT:
            return "bsdf_measurement";

        case mi::neuraylib::IType::TK_BSDF:
            return "bsdf";

        case mi::neuraylib::IType::TK_EDF:
            return "edf";

        case mi::neuraylib::IType::TK_VDF:
            return "vdf";

        case mi::neuraylib::IType::TK_HAIR_BSDF:
            return "hair_bsdf";

        default:
            break;
    }
    return "UNKNOWN_TYPE";
}

// Returns the type of an enum as string
std::string Compiled_material_traverser_print::enum_type_to_string(
    const mi::neuraylib::IType_enum* enum_type,
    Compiled_material_traverser_print::Context* context)
{
    std::string s = enum_type->get_symbol();

    // make sure the type is included properly
    context->m_imports.insert(s);
    return s;
}

// Returns the type of a struct as string
std::string Compiled_material_traverser_print::struct_type_to_string(
    const mi::neuraylib::IType_struct* struct_type,
    Compiled_material_traverser_print::Context* context,
    bool* out_is_material_keyword)
{
    std::string s = struct_type->get_symbol();

    // deal with keywords defined in MDL spec section 13 on 'Materials'
    // the compiler handles these constructs as structures
    if (s == "::material" ||
        s == "::bsdf" ||
        s == "::edf" ||
        s == "::vdf" ||
        s == "::material_surface" ||
        s == "::material_emission" ||
        s == "::material_volume" ||
        s == "::material_geometry")
    {
        if (context->m_keep_compiled_material_structure &&
            context->m_stage == Compiled_material_traverser_base::ES_PARAMETERS)
        {
            std::cerr << "error:\tThe compiled material defines a parameter of type '"
                      << s.substr(2) << "', which results in the printing of invalid mdl code.\n";
            context->m_is_valid_mdl = false;
        }

        if (out_is_material_keyword) *out_is_material_keyword = true;
        return s.substr(2);
    }
    if (out_is_material_keyword) *out_is_material_keyword = false;

    // if this is not the case, we need to make sure the type is included properly
    context->m_imports.insert(s);
    return s;
}

// Returns the type of an atomic type as string
std::string Compiled_material_traverser_print::atomic_type_to_string(
    const mi::neuraylib::IType_atomic* atomic_type,
    Compiled_material_traverser_print::Context* context)
{
    switch (atomic_type->get_kind())
    {
        case mi::neuraylib::IType::TK_BOOL:
            return "bool";
        case mi::neuraylib::IType::TK_INT:
            return "int";
        case mi::neuraylib::IType::TK_FLOAT:
            return "float";
        case mi::neuraylib::IType::TK_DOUBLE:
            return "double";
        case mi::neuraylib::IType::TK_STRING: 
            return "string";
        case mi::neuraylib::IType::TK_ENUM:
        {
            mi::base::Handle<const mi::neuraylib::IType_enum> enum_type(
                atomic_type->get_interface<const mi::neuraylib::IType_enum>());
            return enum_type_to_string(enum_type.get(), context);
        }
        default:
            break;
    }
    return "UNKNOWN_TYPE";
}

// Returns a vector type as string.
std::string Compiled_material_traverser_print::vector_type_to_string(
    const mi::neuraylib::IType_vector* vector_type,
    Compiled_material_traverser_print::Context* context)
{
    const mi::base::Handle<const mi::neuraylib::IType_atomic> elem_type(
        vector_type->get_element_type());

    std::stringstream s;
    s << atomic_type_to_string(elem_type.get(), context) << vector_type->get_size();
    return s.str();;
}

// Returns a matrix type as string.
std::string Compiled_material_traverser_print::matrix_type_to_string(
    const mi::neuraylib::IType_matrix* matrix_type,
    Compiled_material_traverser_print::Context* context)
{
    const mi::base::Handle<const mi::neuraylib::IType_vector>
        column_type(matrix_type->get_element_type());
    const mi::base::Handle<const mi::neuraylib::IType_atomic> elem_type(
        column_type->get_element_type());

    std::stringstream s;
    s << atomic_type_to_string(elem_type.get(), context) 
      << column_type->get_size() << "x" << matrix_type->get_size();
    return s.str();;
}

// Returns an array type as string.
std::string Compiled_material_traverser_print::array_type_to_string(
    const mi::neuraylib::IType_array* array_type,
    Compiled_material_traverser_print::Context* context)
{
    const mi::base::Handle<const mi::neuraylib::IType> elem_type(array_type->get_element_type());
    const mi::base::Handle<const mi::neuraylib::IType_atomic> atomic_type(
        elem_type->get_interface<const mi::neuraylib::IType_atomic>());

    std::stringstream s; 
    s << type_to_string(elem_type.get(), context);

    if (array_type->is_immediate_sized())
        s  << "[" << array_type->get_size() << "]";
    else
        s  << "[]";

    return s.str();

}

//--------------------------------------------------------------------------------------------------
// local helper functions
//--------------------------------------------------------------------------------------------------

inline std::string replace_all(const std::string& input, 
                               const std::string& old, const std::string& with)
{
    if (input.empty()) return input;

    std::string output(input);
    size_t offset(0);
    size_t pos(0);
    while (pos != std::string::npos)
    {
        pos = output.find(old, offset);
        if (pos == std::string::npos)
            break;

        output.replace(pos, old.length(), with);
        offset = pos + with.length();
    }
    return output;
}

inline bool is_constructor(mi::neuraylib::IFunction_definition::Semantics semantic)
{
    return semantic >= mi::neuraylib::IFunction_definition::DS_CONV_CONSTRUCTOR
        && semantic <= mi::neuraylib::IFunction_definition::DS_TEXTURE_CONSTRUCTOR;
}

inline bool is_type_converter(mi::neuraylib::IFunction_definition::Semantics semantic)
{
    return semantic == mi::neuraylib::IFunction_definition::DS_CONV_CONSTRUCTOR // also constructor
        || semantic == mi::neuraylib::IFunction_definition::DS_CONV_OPERATOR;
}


inline bool is_array_constructor(mi::neuraylib::IFunction_definition::Semantics semantic)
{
    return semantic == mi::neuraylib::IFunction_definition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR;
}

inline bool is_array_index_operator(mi::neuraylib::IFunction_definition::Semantics semantic)
{
    return semantic == mi::neuraylib::IFunction_definition::DS_ARRAY_INDEX;
}

inline bool is_unary_operator(mi::neuraylib::IFunction_definition::Semantics semantic)
{
    return semantic >= mi::neuraylib::IFunction_definition::DS_UNARY_FIRST
        && semantic <= mi::neuraylib::IFunction_definition::DS_UNARY_LAST;
}

inline bool is_binary_operator(mi::neuraylib::IFunction_definition::Semantics semantic)
{
    return semantic >= mi::neuraylib::IFunction_definition::DS_BINARY_FIRST
        && semantic <= mi::neuraylib::IFunction_definition::DS_BINARY_LAST;
}

inline bool is_ternary_operator(mi::neuraylib::IFunction_definition::Semantics semantic)
{
    return semantic == mi::neuraylib::IFunction_definition::DS_TERNARY;
}

bool is_selector_operator(mi::neuraylib::IFunction_definition::Semantics semantic)
{
    return semantic == mi::neuraylib::IFunction_definition::DS_SELECT
        || semantic == mi::neuraylib::IFunction_definition::DS_INTRINSIC_DAG_FIELD_ACCESS;
}

bool is_call_like_operator(mi::neuraylib::IFunction_definition::Semantics semantic)
{
    return
        // intrinsic functions
        (semantic >= mi::neuraylib::IFunction_definition::DS_INTRINSIC_MATH_FIRST
         && semantic <= mi::neuraylib::IFunction_definition::DS_INTRINSIC_DEBUG_LAST)

        // this includes standard function calls
        || semantic == mi::neuraylib::IFunction_definition::DS_UNKNOWN;
}

// unhandled semantics:
// DS_INTRINSIC_DAG_ARRAY_LENGTH which is not relevant for compiled materials

