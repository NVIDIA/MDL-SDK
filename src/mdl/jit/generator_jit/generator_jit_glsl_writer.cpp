/******************************************************************************
 * Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
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

#include "pch.h"

#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/IntrinsicInst.h>

#include <mdl/compiler/compilercore/compilercore_allocator.h>
#include <mdl/compiler/compilercore/compilercore_errors.h>
#include <mdl/compiler/compilercore/compilercore_messages.h>

#include <mdl/compiler/compiler_glsl/compiler_glsl_analysis.h>
#include <mdl/compiler/compiler_glsl/compiler_glsl_builtins.h>
#include <mdl/compiler/compiler_glsl/compiler_glsl_compiler.h>
#include <mdl/compiler/compiler_glsl/compiler_glsl_tools.h>

#include "generator_jit_glsl_writer.h"
#include "generator_jit_streams.h"

#define DEBUG_TYPE "glsl_writer"

namespace mi {
namespace mdl {
namespace glsl {

// explicit instantiation necessary
template void GLSLWriterBasePass::add_array_specifiers<glsl::Init_declarator>(
    glsl::Init_declarator *, glsl::Type *);
template void GLSLWriterBasePass::add_array_specifiers<glsl::Declaration_param>(
    glsl::Declaration_param *, glsl::Type *);
template void GLSLWriterBasePass::add_array_specifiers<glsl::Type_name>(
    glsl::Type_name *, glsl::Type *);

// The zero location.
Location const GLSLWriterBasePass::zero_loc(0, 0, 0);

/// The prototype language.
IGenerated_code_executable::Prototype_language const GLSLWriterBasePass::proto_lang =
    IGenerated_code_executable::PL_GLSL;

// Constructor.
GLSLWriterBasePass::GLSLWriterBasePass(
    char                        &pid,
    mi::mdl::IAllocator         *alloc,
    Type_mapper const           &type_mapper,
    mi::mdl::Options_impl const &options,
    mi::mdl::Messages_impl      &messages,
    bool                        enable_debug,
    bool                        enable_opt_remarks)
: Base(pid)
, m_alloc(alloc)
, m_type_mapper(type_mapper)
, m_compiler(impl_cast<glsl::Compiler>(glsl::initialize(m_alloc)))
, m_unit(create_compilation_unit(m_compiler.get(), "generated"))
, m_decl_factory(m_unit->get_declaration_factory())
, m_expr_factory(m_unit->get_expression_factory())
, m_stmt_factory(m_unit->get_statement_factory())
, m_tc(m_unit->get_type_factory())
, m_value_factory(m_unit->get_value_factory())
, m_symbol_table(m_unit->get_symbol_table())
, m_def_tab(m_unit->get_definition_table())
, m_ctx(m_unit->get_glslang_context())
, m_type_cache(0, Type2type_map::hasher(), Type2type_map::key_equal(), alloc)
, m_debug_types(alloc)
, m_ref_fnames(0, Ref_fname_id_map::hasher(), Ref_fname_id_map::key_equal(), alloc)
, m_messages(messages)
, m_api_decls(0, Decl_set::hasher(), Decl_set::key_equal(), alloc)
, m_uniform_inits(alloc)
, m_max_const_size(1024u)
, m_glslang_version(glsl::GLSL_VERSION_4_50)
, m_glslang_profile(glsl::GLSL_PROFILE_CORE)
, m_glsl_enabled_extentions(nullptr)
, m_glsl_required_extentions(nullptr)
, m_ssbo_decl(nullptr)
, m_glsl_uniform_ssbo_name(nullptr)
, m_glsl_uniform_ssbo_binding(~0u)
, m_glsl_uniform_ssbo_set(~0u)
, m_next_unique_name_id(0u)
, m_place_uniform_inits_into_ssbo(false)
, m_use_dbg(enable_debug)
{
    unsigned major, minor;
    if (options.get_version_option(MDL_JIT_OPTION_GLSL_VERSION, major, minor)) {
        set_target_version(major, minor);
    }

    if (char const *v = options.get_string_option(MDL_JIT_OPTION_GLSL_PROFILE)) {
        if (strcmp(v, "core") == 0) {
            m_glslang_profile = glsl::GLSL_PROFILE_CORE;
        } else if (strcmp(v, "compatibility") == 0) {
            m_glslang_profile = glsl::GLSL_PROFILE_COMPATIBILITY;
        } else if (strcmp(v, "es") == 0) {
            m_glslang_profile = glsl::GLSL_PROFILE_ES;
        }
    }

    m_glsl_enabled_extentions =
        options.get_string_option(MDL_JIT_OPTION_GLSL_ENABLED_EXTENSIONS);
    m_glsl_required_extentions =
        options.get_string_option(MDL_JIT_OPTION_GLSL_REQUIRED_EXTENSIONS);

    m_max_const_size =
        options.get_unsigned_option(MDL_JIT_OPTION_GLSL_MAX_CONST_DATA);
#if 0
    m_lambda_enforce_vec3_return =
        options.get_bool_option(MDL_JIT_OPTION_GLSL_LAMBDA_ENFORCE_VEC3_RETURN);
    m_glsl_function_remaps =
        options.get_string_option(MDL_JIT_OPTION_GLSL_REMAP_FUNCTIONS);
#endif
    m_place_uniform_inits_into_ssbo =
        options.get_bool_option(MDL_JIT_OPTION_GLSL_PLACE_UNIFORMS_INTO_SSBO);
    m_glsl_uniform_ssbo_name =
        options.get_string_option(MDL_JIT_OPTION_GLSL_UNIFORM_SSBO_NAME);
    {
        char const *ssbo_binding =
            options.get_string_option(MDL_JIT_OPTION_GLSL_UNIFORM_SSBO_BINDING);
        if (ssbo_binding != nullptr && ssbo_binding[0] != '\0') {
            m_glsl_uniform_ssbo_binding =
                options.get_unsigned_option(MDL_JIT_OPTION_GLSL_UNIFORM_SSBO_BINDING);
        }
    }
    {
        char const *ssbo_set =
            options.get_string_option(MDL_JIT_OPTION_GLSL_UNIFORM_SSBO_SET);
        if (ssbo_set != nullptr && ssbo_set[0] != '\0') {
            m_glsl_uniform_ssbo_set =
                options.get_unsigned_option(MDL_JIT_OPTION_GLSL_UNIFORM_SSBO_SET);
        }
    }
#if 0
    m_state_includes_uniform_state =
        options.get_bool_option(MDL_JIT_OPTION_GLSL_INCLUDE_UNIFORM_STATE);
#endif

    set_glsl_target_context(m_unit.get());

    if (m_place_uniform_inits_into_ssbo) {
        if (!m_unit->get_glslang_context().has_SSBO()) {
            // this option cannot be used and must be disabled
            m_place_uniform_inits_into_ssbo = false;

            warning(GLSL_SSBO_UNAVAILABLE);
        }
    }

    fill_predefined_entities();
}

// Return the name for this pass.
llvm::StringRef GLSLWriterBasePass::getPassName() const
{
    return "GLSL writer";
}

// Create a new (empty) compilation unit with the given name.
glsl::Compilation_unit *GLSLWriterBasePass::create_compilation_unit(
    glsl::ICompiler *compiler,
    char const      *name)
{
    // assume we create code for fragment shaders
    return impl_cast<glsl::Compilation_unit>(
        compiler->create_unit(glsl::GLSL_LANG_FRAGMENT, name));
}

// Set the GLSL target language version.
bool GLSLWriterBasePass::set_target_version(unsigned major, unsigned minor)
{
    switch (major) {
    case 1:
        switch (minor) {
        case 0:
            m_glslang_version = glsl::GLSL_VERSION_ES_1_00;  // Version 100 ES
            m_glslang_profile = glsl::GLSL_PROFILE_ES;
            break;
        case 10:
            m_glslang_version = glsl::GLSL_VERSION_1_10;  // Version 1.10
            break;
        case 20:
            m_glslang_version = glsl::GLSL_VERSION_1_20;  // Version 1.20
            break;
        case 30:
            m_glslang_version = glsl::GLSL_VERSION_1_30;  // Version 1.30
            break;
        case 40:
            m_glslang_version = glsl::GLSL_VERSION_1_40;  // Version 1.40
            break;
        case 50:
            m_glslang_version = glsl::GLSL_VERSION_1_50;  // Version 1.50
            break;
        default:
            // unsupported
            return false;
        }
        break;
    case 3:
        switch (minor) {
        case 0:
            m_glslang_version = glsl::GLSL_VERSION_ES_3_00;  // Version 300 ES
            m_glslang_profile = glsl::GLSL_PROFILE_ES;
            break;
        case 10:
            m_glslang_version = glsl::GLSL_VERSION_ES_3_10;  // Version 310 ES
            m_glslang_profile = glsl::GLSL_PROFILE_ES;
            break;
        case 30:
            m_glslang_version = glsl::GLSL_VERSION_3_30;  // Version 3.30
            break;
        default:
            // unsupported
            return false;
        }
        break;
    case 4:
        switch (minor) {
        case 0:
            m_glslang_version = glsl::GLSL_VERSION_4_00;  // Version 4.00
            break;
        case 10:
            m_glslang_version = glsl::GLSL_VERSION_4_10;  // Version 4.10
            break;
        case 20:
            m_glslang_version = glsl::GLSL_VERSION_4_20;  // Version 4.20
            break;
        case 30:
            m_glslang_version = glsl::GLSL_VERSION_4_30;  // Version 4.30
            break;
        case 40:
            m_glslang_version = glsl::GLSL_VERSION_4_40;  // Version 4.40
            break;
        case 50:
            m_glslang_version = glsl::GLSL_VERSION_4_50;  // Version 4.50
            break;
        case 60:
            m_glslang_version = glsl::GLSL_VERSION_4_60;  // Version 4.60
            break;
        default:
            // unsupported
            return false;
        }
        break;
    default:
        // unsupported
        return false;
    }
    return true;
}

// Parse extensions and set them.
bool GLSLWriterBasePass::set_extensions(
    glsl::GLSLang_context                     &ctx,
    char const                                *ext,
    glsl::GLSLang_context::Extension_behavior eb)
{
    bool res = true;
    string buf(m_alloc);

    char const *s = ext;
    char const *e = s;

    for (;;) {
        while (*e != '\0' && *e != ',') {
            ++e;
        }
        while (*s == ' ') {
            ++s;
        }

        buf = string(s, e, m_alloc);

        if (ctx.update_extension(buf.c_str(), eb) != glsl::GLSLang_context::EC_OK) {
            res = false;
        }
        if (*e == '\0') {
            break;
        }
        s = e + 1;
    }
    return res;
}

// Set the GLSL target context to the compilation unit.
void GLSLWriterBasePass::set_glsl_target_context(glsl::Compilation_unit *unit)
{
    glsl::GLSLang_context &ctx = unit->get_glslang_context();

    ctx.set_version(m_glslang_version, m_glslang_profile);

    if (m_glsl_enabled_extentions) {
        set_extensions(ctx, m_glsl_enabled_extentions, glsl::GLSLang_context::EB_ENABLE);
    }
    if (m_glsl_required_extentions) {
        set_extensions(ctx, m_glsl_required_extentions, glsl::GLSLang_context::EB_REQUIRE);
    }
}

// Fill the predefined entities into the (at this point empty) compilation unit;
void GLSLWriterBasePass::fill_predefined_entities()
{
    m_def_tab.transition_to_scope(m_def_tab.get_predef_scope());

    // enter all predefined entities
    glsl::enter_predefined_entities(*m_unit, m_tc);

    m_def_tab.transition_to_scope(m_def_tab.get_global_scope());
}

// Find the API debug type info for a given struct type.
sl::DebugTypeHelper::API_type_info const *GLSLWriterBasePass::find_api_type_info(
    llvm::StructType *s_type) const
{
    llvm::StringRef name = s_type->getName();
    sl::DebugTypeHelper::API_type_info const *api_info = m_debug_types.find_api_type_info(name);
    if (api_info != nullptr) {
        // for C++ compiled types, an extra padding field might be added
        if (api_info->n_fields == s_type->getNumContainedTypes() ||
            api_info->n_fields == s_type->getNumContainedTypes() - 1) {
            return api_info;
        } else {
            MDL_ASSERT(!"unexpected number of fields in an API type");
        }
    }
    return nullptr;
}

// Return the "inner" element type of array types.
glsl::Type *GLSLWriterBasePass::inner_element_type(
    glsl::Type *type)
{
    while (glsl::Type_array *a_type = as<glsl::Type_array>(type)) {
        type = a_type->get_element_type();
    }
    return type;
}

// Get an GLSL symbol for an LLVM string.
glsl::Symbol *GLSLWriterBasePass::get_sym(llvm::StringRef const &str)
{
    return m_symbol_table.get_symbol(str.str().c_str());
}

// Get an GLSL symbol for a string.
glsl::Symbol *GLSLWriterBasePass::get_sym(char const *str)
{
    return m_symbol_table.get_symbol(str);
}

// Get an unique GLSL symbol for an LLVM string and a template.
glsl::Symbol *GLSLWriterBasePass::get_unique_sym(
    llvm::StringRef const &str,
    char const *templ)
{
    return get_unique_sym(str.str().c_str(), templ);
}

// Get an unique GLSL symbol for an LLVM string and a template.
glsl::Symbol *GLSLWriterBasePass::get_unique_sym(
    char const *str,
    char const *templ)
{
    bool valid = true;
    char const *name = str;

    char tbuf[60];

    if (!isalpha(str[0]) && str[0] != '_') {
        valid = false;
    } else {
        char const *p = &str[1];
        char const *dot = NULL;

        for (; *p != '\0'; ++p) {
            if (*p == '.') {
                dot = p;
                break;
            }
            if (!isalnum(*p) && *p != '_') {
                valid = false;
                break;
            }
        }

        if (dot != NULL) {
            for (p = dot + 1; *p != '\0'; ++p) {
                if (!isdigit(*p)) {
                    valid = false;
                    break;
                }
            }

            if (valid && p != dot + 1) {
                // be smarter with LLVM renamed symbols that ends with .[digit]+
                size_t l = dot - str;
                if (l >= sizeof(tbuf)) {
                    l = sizeof(tbuf) - 1;
                }
                strncpy(tbuf, str, l);
                tbuf[l] = '\0';
                str = name = tbuf;
            } else {
                valid = false;
            }
        }
    }

    if (!valid) {
        str = templ;
        name = nullptr;  // skip lookup and append id before trying
    }

    // check scope
    glsl::Symbol *sym = nullptr;

    char buffer[65];
    while (true) {
        if (name != nullptr) {
            if (!m_unit->get_glsl_keyword_map().keyword_or_reserved(name, strlen(name))) {
                sym = m_symbol_table.lookup_symbol(name);
                if (sym == nullptr) {
                    // this is the first occurrence of this symbol
                    sym = m_symbol_table.get_symbol(name);
                    break;
                }
                size_t id = sym->get_id();
                if (id >= Symbol::SYM_USER && m_def_tab.get_definition(sym) == nullptr) {
                    // symbol exists, but is user defined and no definition in this scope, good
                    break;
                }
            }
        }

        // rename it and try again
        strncpy(buffer, str, 58);
        buffer[58] = '\0';
        snprintf(buffer + strlen(buffer), 6, "%u", m_next_unique_name_id);
        buffer[64] = '\0';
        name = buffer;
        ++m_next_unique_name_id;
    }
    return sym;
}

// Get an GLSL name for the given location and symbol.
glsl::Name *GLSLWriterBasePass::get_name(Location loc, Symbol *sym)
{
    return m_decl_factory.create_name(loc, sym);
}

// Get an GLSL name for the given location and a C-string.
glsl::Name *GLSLWriterBasePass::get_name(Location loc, const char *str)
{
    glsl::Symbol *sym = m_symbol_table.get_symbol(str);
    return get_name(loc, sym);
}

// Get an GLSL name for the given location and LLVM string reference.
glsl::Name *GLSLWriterBasePass::get_name(Location loc, llvm::StringRef const &str)
{
    return get_name(loc, str.str().c_str());
}

// Get an GLSL type name for an LLVM type.
glsl::Type_name *GLSLWriterBasePass::get_type_name(
    llvm::Type *type)
{
    glsl::Type *glsl_type = convert_type(type);
    return get_type_name(glsl_type);
}

// Get an GLSL type name for an GLSL type.
glsl::Type_name *GLSLWriterBasePass::get_type_name(
    glsl::Type *type)
{
    glsl::Type::Modifiers mod = type->get_type_modifiers();

    glsl::Type      *e_type    = inner_element_type(type);
    glsl::Name      *name      = m_decl_factory.create_name(zero_loc, e_type->get_sym());
    glsl::Type_name *type_name = m_decl_factory.create_type_name(zero_loc);
    type_name->set_name(name);
    add_array_specifiers(type_name, type);

    type_name->set_type(type);

    // handle type modifier
    if (mod & glsl::Type::MK_CONST) {
        type_name->get_qualifier().set_storage_qualifier(glsl::SQ_CONST);
    }

    return type_name;
}

// Get an GLSL type name for an GLSL name.
glsl::Type_name *GLSLWriterBasePass::get_type_name(
    glsl::Symbol *sym)
{
    glsl::Type_name *type_name = m_decl_factory.create_type_name(zero_loc);
    glsl::Name *name = get_name(zero_loc, sym);
    type_name->set_name(name);

    return type_name;
}

// Add array specifier to an init declarator if necessary.
template<typename Decl_type>
void GLSLWriterBasePass::add_array_specifiers(
    Decl_type  *decl,
    glsl::Type *type)
{
    while (glsl::Type_array *a_type = glsl::as<glsl::Type_array>(type)) {
        type = a_type->get_element_type();

        glsl::Expr *size = nullptr;
        if (!a_type->is_unsized()) {
            size = this->m_expr_factory.create_literal(
                zero_loc,
                this->m_value_factory.get_int32(int(a_type->get_size())));
        }
        glsl::Array_specifier *as = this->m_decl_factory.create_array_specifier(zero_loc, size);
        decl->add_array_specifier(as);
    }
}

// Add parameter qualifier from a function type parameter at index.
void GLSLWriterBasePass::add_param_qualifier(
    glsl::Type_name     *param_type_name,
    glsl::Type_function *func_type,
    size_t              index)
{
    glsl::Type_function::Parameter *param = func_type->get_parameter(index);

    glsl::Parameter_qualifier param_qualifier = convert_type_modifier_to_param_qualifier(param);
    param_type_name->get_qualifier().set_parameter_qualifier(param_qualifier);
}

// Add a field to a struct declaration.
glsl::Type_struct::Field GLSLWriterBasePass::add_struct_field(
    glsl::Declaration_struct *decl_struct,
    glsl::Type               *type,
    glsl::Symbol             *sym)
{
    glsl::Declaration_field *decl_field =
        m_decl_factory.create_field_declaration(get_type_name(inner_element_type(type)));
    glsl::Field_declarator *field_declarator = m_decl_factory.create_field(zero_loc);
    field_declarator->set_name(get_name(zero_loc, sym));
    add_array_specifiers(field_declarator, type);

    decl_field->add_field(field_declarator);
    decl_struct->add(decl_field);

    return glsl::Type_struct::Field(type, sym);
}

// Add a field to a struct declaration.
glsl::Type_struct::Field GLSLWriterBasePass::add_struct_field(
    glsl::Declaration_struct *decl_struct,
    glsl::Type               *type,
    char const               *name)
{
    glsl::Symbol *sym = m_symbol_table.get_symbol(name);
    return add_struct_field(decl_struct, type, sym);
}

// Create the GLSL resource data struct for the corresponding LLVM struct type.
glsl::Type_struct *GLSLWriterBasePass::create_res_data_struct_type(
    llvm::StructType *type)
{
    // The res_data struct type is opaque in the generated code, but we don't support
    // this in the glsl compiler, so we create a dummy type, but do not add it to the
    // compilation unit.

    glsl::Declaration_struct *decl_struct = m_decl_factory.create_struct(zero_loc);
    glsl::Symbol             *struct_sym  = m_symbol_table.get_symbol("Res_data");

    decl_struct->set_name(get_name(zero_loc, struct_sym));

    glsl::Type_struct::Field dummy_field[1] = {
        add_struct_field(decl_struct, m_tc.int_type, "dummy")
    };

    glsl::Type_struct *res = m_tc.get_struct(dummy_field, struct_sym);

    // do not add to the unit to avoid printing it

    m_type_cache[type] = res;
    return res;
}

// Convert an LLVM struct type to an GLSL struct type.
glsl::Type *GLSLWriterBasePass::convert_struct_type(
    llvm::StructType *s_type)
{
    auto it = m_type_cache.find(s_type);
    if (it != m_type_cache.end()) {
        return it->second;
    }

    string struct_name(m_alloc);

    llvm::DICompositeType                    *di_type  = nullptr;
    sl::DebugTypeHelper::API_type_info const *api_info = nullptr;

    if (s_type->hasName()) {
        llvm::StringRef name = s_type->getName();

        // must be handled first as this is treated as a opaque type
        if (name == "Res_data") {
            return create_res_data_struct_type(s_type);
        }

        // check if it is an API type
        api_info = find_api_type_info(s_type);

        if (name.startswith("struct.")) {
            name = name.substr(7);

            // map libbsdf float3/float4 types
            if (name == "float3") {
                m_type_cache[s_type] = m_tc.vec3_type;
                return m_tc.vec3_type;
            }
            if (name == "float4") {
                m_type_cache[s_type] = m_tc.vec4_type;
                return m_tc.vec4_type;
            }
        }

        // retrieve debug info if there is any for non-API types
        di_type = api_info == nullptr ? find_composite_type_info(name) : nullptr;

        bool mangle_name = true;

        // strip the leading "::" from the fully qualified MDL type name
        if (name.startswith("::")) {
            name = name.substr(2);
        } else {
            // not an MDL name
            mangle_name = false;
        }

        if (!mangle_name) {
            // these are types from the interface, do not mangle them, just replace "some" bad
            struct_name = string(name.begin(), name.end(), m_alloc);

            // replace all ':' and '.' by '_'
            for (char &ch : struct_name) {
                if (ch == ':' || ch == '.') {
                    ch = '_';
                }
            }
        } else {
            // use default MDL mangler
            mi::mdl::MDL_name_mangler mangler(m_alloc, struct_name);

            size_t pos = name.rfind("::");

            if (pos != llvm::StringRef::npos) {
                string prefix(name.begin(), name.begin() + pos, m_alloc);
                string n(name.begin() + pos + 2, name.end(), m_alloc);

                mangler.mangle_name(prefix.c_str(), n.c_str());
            } else {
                mangler.mangle_name(nullptr, name.str().c_str());
            }
        }
    }


    unsigned n_fields = s_type->getNumElements();

    // check if we have at least ONE non-void field
    bool has_non_void_field = false;
    for (unsigned i = 0; i < n_fields; ++i) {
        glsl::Type *field_type = convert_type(s_type->getElementType(i));
        if (glsl::is<glsl::Type_void>(field_type)) {
            continue;
        }

        has_non_void_field = true;
        break;
    }

    if (!has_non_void_field) {
        // an empty struct, map to void
        return m_type_cache[s_type] = m_tc.void_type;
    }

    bool is_deriv = m_type_mapper.is_deriv_type(s_type);

    // Some derivative types are part of the API and need to have fixed names.
    Symbol *struct_sym;
    bool is_api_type = api_info != nullptr;

    if (is_deriv) {
        // handle special derived types
        if (s_type == m_type_mapper.get_deriv_float_type()) {
            struct_sym = get_sym("Derived_float");
            is_api_type = true;
        } else if (s_type == m_type_mapper.get_deriv_float2_type()) {
            struct_sym = get_sym("Derived_float2");
            is_api_type = true;
        } else if (s_type == m_type_mapper.get_deriv_float3_type()) {
            struct_sym = get_sym("Derived_float3");
            is_api_type = true;
        } else if (s_type == m_type_mapper.get_deriv_float4_type()) {
            struct_sym = get_sym("Derived_float4");
            is_api_type = true;
        } else {
            if (api_info != nullptr) {
                // we must use the API name
                struct_sym = get_sym(api_info->api_name);
            } else {
                struct_sym = get_unique_sym(
                    struct_name.c_str(),
                    "deriv_type");
            }
        }
    } else {
        // non-derived types
        if (api_info != nullptr) {
            // we must use the API name
            struct_sym = get_sym(api_info->api_name);
        } else {
            struct_sym = get_unique_sym(
                struct_name.c_str(),
                "structtype");
        }
    }

    glsl::Declaration_struct *decl_struct = m_decl_factory.create_struct(zero_loc);
    decl_struct->set_name(get_name(zero_loc, struct_sym));

    Small_VLA<glsl::Type_struct::Field, 8> fields(m_alloc, n_fields);

    static char const * const deriv_names[] = { "val", "dx", "dy" };

    unsigned n = 0;
    for (unsigned i = 0; i < n_fields; ++i) {
        glsl::Type *field_type = convert_type(s_type->getElementType(i));

        if (api_info == nullptr && glsl::is<glsl::Type_void>(field_type)) {
            // do NOT skip void fields for API types: they might be accessed during
            // field index which would break then
            continue;
        }

        glsl::Symbol *sym = nullptr;

        if (is_deriv) {
            // use predefined field names for derived types
            MDL_ASSERT(i < llvm::array_lengthof(deriv_names));
            sym = get_sym(deriv_names[i]);
        } else if (api_info != nullptr) {
            // use API description
            if (i < api_info->n_fields) {
                sym = get_sym(api_info->fields[i]);
            } else {
                MDL_ASSERT(api_info->n_fields + 1 == n_fields);
                // reached padding field
                break;
            }
        } else if (di_type != nullptr) {
            // get original struct member name from the debug information
            if (llvm::DIType *e_tp = find_subelement_type_info(di_type, i)) {
                llvm::StringRef s = e_tp->getName();

                if (!s.empty()) {
                    string field_name = string(s.begin(), s.end(), m_alloc);

                    sym = get_sym(field_name.c_str());
                    if (sym->get_id() < glsl::Symbol::SYM_USER) {
                        // cannot be used
                        sym = get_unique_sym(field_name.c_str(), field_name.c_str());
                    }
                }
            }
        }

        if (sym == nullptr) {
            // if we still name no name, create a generic field name
            // Note: we assume here, that either ALL field names are retrieved, or none, so
            // these generic names do not clash
            char name_buf[16];
            snprintf(name_buf, sizeof(name_buf), "m_%u", i);
            sym = get_sym(name_buf);
        }
        fields[n++] = add_struct_field(decl_struct, field_type, sym);
    }

    glsl::Type_struct *res = m_tc.get_struct(
        Array_ref<glsl::Type_struct::Field>(fields.data(), n), struct_sym);

    // create the definition for the struct type and its fields
    {
        glsl::Definition_table::Scope_transition scope_trans(
            m_def_tab, m_def_tab.get_global_scope());

        Def_type *type_def = m_def_tab.enter_type_definition(
            struct_sym,
            res,
            &zero_loc);
        decl_struct->set_definition(type_def);

        glsl::Definition_table::Scope_enter scope(m_def_tab, res, type_def);

        for (size_t i = 0; i < n; ++i) {
            (void)m_def_tab.enter_member_definition(
                fields[i].get_symbol(),
                fields[i].get_type(),
                /*field_index=*/i,
                &zero_loc);
        }
    }

    m_type_cache[s_type] = res;
    m_unit->add_decl(decl_struct);

    if (is_api_type) {
        // this is an API type
        m_api_decls.insert(decl_struct);
    }

    return res;
}

// Convert an LLVM type to an GLSL type.
glsl::Type *GLSLWriterBasePass::convert_type(
    llvm::Type *type)
{
    switch (type->getTypeID()) {
    case llvm::Type::VoidTyID:
        return m_tc.void_type;
    case llvm::Type::HalfTyID:
        return m_tc.half_type;
    case llvm::Type::FloatTyID:
        return m_tc.float_type;
    case llvm::Type::DoubleTyID:
        if (m_ctx.has_double_type()) {
            return m_tc.double_type;
        }
        // no double in this GLSL version, map the float
        return m_tc.float_type;
    case llvm::Type::IntegerTyID:
        {
            llvm::IntegerType *int_type = llvm::cast<llvm::IntegerType>(type);
            unsigned int bit_width = int_type->getBitWidth();

            // Support such constructs
            // %X = trunc i32 %Y to i2
            // %Z = icmp i2 %X, 1
            if (bit_width > 1 && bit_width <= 32) {
                return m_tc.int_type;
            }

            switch (int_type->getBitWidth()) {
            case 1:
                return m_tc.bool_type;
            case 16:
            case 64:  // TODO: maybe not a good idea
                return m_tc.int_type;
            default:
                MDL_ASSERT(!"unexpected LLVM integer type");
                error(mi::mdl::INTERNAL_JIT_UNSUPPORTED_INTEGER_TYPE);
                return m_tc.int_type;
            }
        }
    case llvm::Type::StructTyID:
        return convert_struct_type(llvm::cast<llvm::StructType>(type));

    case llvm::Type::ArrayTyID:
        {
            llvm::ArrayType *array_type = llvm::cast<llvm::ArrayType>(type);
            size_t          n_elem      = array_type->getNumElements();

            if (n_elem == 0) {
                // map zero length array to void, we cannot handle that in HLSL
                return m_tc.void_type;
            }

            // map some floating point arrays to matrices:
            //   float2[2] -> float2x2
            //   float2[3] -> float3x2
            //   float2[4] -> float4x2
            //   float3[2] -> float2x3
            //   float3[3] -> float3x3
            //   float3[4] -> float4x3
            //   float4[2] -> float2x4
            //   float4[3] -> float3x4
            //   float4[4] -> float4x4

            llvm::Type *array_elem_type = array_type->getElementType();
            if (llvm::FixedVectorType *vt =
                    llvm::dyn_cast<llvm::FixedVectorType>(array_elem_type))
            {
                llvm::Type         *vt_elem_type = vt->getElementType();
                llvm::Type::TypeID type_id       = vt_elem_type->getTypeID();

                if (type_id == llvm::Type::FloatTyID || type_id == llvm::Type::DoubleTyID) {
                    size_t cols = n_elem;
                    size_t rows = vt->getNumElements();

                    if (2 <= cols && cols <= 4 && 2 <= rows && rows <= 4) {
                        // map to a matrix type
                        glsl::Type_scalar *res_elem_type = m_tc.float_type;
                        if (type_id == llvm::Type::DoubleTyID && m_ctx.has_double_type()) {
                            res_elem_type = m_tc.double_type;
                        }
                        // else no double in this GLSL version, map the float
                        glsl::Type_vector *res_vt_type =
                            m_tc.get_vector(res_elem_type, rows);
                        return m_tc.get_matrix(res_vt_type, cols);
                    }
                }
            }

            glsl::Type *res_elem_type = convert_type(array_elem_type);
            return m_tc.get_array(res_elem_type, n_elem);
        }

    case llvm::Type::FixedVectorTyID:
        {
            llvm::FixedVectorType *vector_type = cast<llvm::FixedVectorType>(type);
            glsl::Type            *elem_type   = convert_type(vector_type->getElementType());
            if (glsl::Type_scalar *scalar_type = glsl::as<glsl::Type_scalar>(elem_type)) {
                glsl::Type *res = m_tc.get_vector(
                    scalar_type, vector_type->getNumElements());
                if (res != nullptr) {
                    return res;
                }
            }
            MDL_ASSERT(!"invalid vector type");
            error(mi::mdl::INTERNAL_JIT_UNSUPPORTED_VECTOR_TYPE);
            return m_tc.error_type;
        }

    case llvm::Type::X86_FP80TyID:
    case llvm::Type::FP128TyID:
    case llvm::Type::PPC_FP128TyID:
        MDL_ASSERT(!"unexpected LLVM type");
        error(mi::mdl::INTERNAL_JIT_UNSUPPORTED_FP_TYPE);
        if (m_ctx.has_double_type()) {
            return m_tc.double_type;
        }
        // no double in this GLSL version, map the float
        return m_tc.float_type;

    case llvm::Type::PointerTyID:
        if (llvm::ArrayType *array_type =
            llvm::dyn_cast<llvm::ArrayType>(type->getPointerElementType()))
        {
            uint64_t size = array_type->getNumElements();
            if (size == 0) {
                // map zero length array to void, we cannot handle that in HLSL
                return m_tc.void_type;
            }

            glsl::Type *base_type = convert_type(array_type->getElementType());
            return m_tc.get_array(base_type, size_t(size));
        }
        if (llvm::StructType *struct_type =
            llvm::dyn_cast<llvm::StructType>(type->getPointerElementType()))
        {
            // for pointers to structs, just skip the pointer
            return convert_struct_type(struct_type);
        }
        MDL_ASSERT(!"pointer types not supported, yet");
        error(mi::mdl::INTERNAL_JIT_UNSUPPORTED_PTR_TYPE);
        return m_tc.error_type;

    case llvm::Type::BFloatTyID:
    case llvm::Type::LabelTyID:
    case llvm::Type::MetadataTyID:
    case llvm::Type::X86_MMXTyID:
    case llvm::Type::X86_AMXTyID:
    case llvm::Type::TokenTyID:
    case llvm::Type::FunctionTyID:
    case llvm::Type::ScalableVectorTyID:
        MDL_ASSERT(!"unexpected LLVM type");
        error(mi::mdl::INTERNAL_JIT_UNSUPPORTED_TYPE);
        return m_tc.error_type;
    }

    MDL_ASSERT(!"unknown LLVM type");
    error(mi::mdl::INTERNAL_JIT_UNSUPPORTED_TYPE);
    return m_tc.error_type;
}

// Create the GLSL definition for a user defined LLVM function.
glsl::Def_function *GLSLWriterBasePass::create_definition(
    llvm::Function *func)
{
    llvm::FunctionType *llvm_func_type = func->getFunctionType();
    glsl::Type         *ret_type = convert_type(llvm_func_type->getReturnType());
    glsl::Type         *out_type = nullptr;

    llvm::SmallVector<glsl::Type_function::Parameter, 8> params;

    if (glsl::is<glsl::Type_array>(ret_type)) {
        // GLSL does not support returning arrays, turn into an out parameter
        out_type = ret_type;
        ret_type = m_tc.void_type;

        params.push_back(glsl::Type_function::Parameter(
            out_type,
            glsl::Type_function::Parameter::PM_OUT));
    }

    // collect parameters for the function definition
    for (llvm::Argument &arg_it : func->args()) {
        llvm::Type *arg_llvm_type = arg_it.getType();
        glsl::Type *param_type    = convert_type(arg_llvm_type);

        // skip void typed parameters
        if (glsl::is<glsl::Type_void>(param_type)) {
            continue;
        }

        glsl::Type_function::Parameter::Modifier param_mod = glsl::Type_function::Parameter::PM_IN;

        if (llvm::isa<llvm::PointerType>(arg_llvm_type)) {
            if (arg_it.hasStructRetAttr()) {
                // the sret attribute marks "return" values, so OUT is enough
                param_mod = glsl::Type_function::Parameter::PM_OUT;
            } else if (arg_it.onlyReadsMemory()) {
                // can be safely passed as an IN attribute IF noalias
                param_mod = glsl::Type_function::Parameter::PM_IN;
            } else {
                // can be safely passed as INOUT IF noalias
                param_mod = glsl::Type_function::Parameter::PM_INOUT;
            }
        }

        params.push_back(glsl::Type_function::Parameter(param_type, param_mod));
    }

    // create the function definition
    glsl::Symbol        *func_sym = get_unique_sym(func->getName(), "func");
    glsl::Type_function *func_type = m_tc.get_function(
        ret_type, Array_ref<glsl::Type_function::Parameter>(params.data(), params.size()));
    glsl::Def_function  *func_def = m_def_tab.enter_function_definition(
        func_sym,
        func_type,
        glsl::Def_function::DS_UNKNOWN,
        &zero_loc);

    return func_def;
}

// Check if a given LLVM array type is the representation of the GLSL matrix type.
bool GLSLWriterBasePass::is_matrix_type(
    llvm::ArrayType *array_type) const
{
    // map some floating point arrays to matrices:
    //   float2[2] -> float2x2
    //   float2[3] -> float3x2
    //   float2[4] -> float4x2
    //   float3[2] -> float2x3
    //   float3[3] -> float3x3
    //   float3[4] -> float4x3
    //   float4[2] -> float2x4
    //   float4[3] -> float3x4
    //   float4[4] -> float4x4

    llvm::Type *array_elem_type = array_type->getElementType();
    if (llvm::FixedVectorType *vt = llvm::dyn_cast<llvm::FixedVectorType>(array_elem_type)) {
        llvm::Type         *vt_elem_type = vt->getElementType();
        llvm::Type::TypeID type_id = vt_elem_type->getTypeID();

        if (type_id == llvm::Type::FloatTyID || type_id == llvm::Type::DoubleTyID) {
            size_t cols = array_type->getNumElements();
            size_t rows = vt->getNumElements();

            if (2 <= cols && cols <= 4 && 2 <= rows && rows <= 4) {
                // map to a matrix type
                return true;
            }
        }
    }
    return false;
}

// Get the name for a given vector index, if possible.
// Translates constants 0 - 3 to x, y, z, w. Otherwise returns nullptr.
char const *GLSLWriterBasePass::get_vector_index_str(
    uint64_t index) const
{
    switch (index) {
    case 0: return "x";
    case 1: return "y";
    case 2: return "z";
    case 3: return "w";
    }
    return nullptr;
}

// Create a reference to an entity of the given name and type.
glsl::Expr_ref *GLSLWriterBasePass::create_reference(
    glsl::Type_name *type_name,
    glsl::Type *type)
{
    glsl::Expr_ref *ref = m_expr_factory.create_reference(type_name);
    ref->set_type(type);
    return ref;
}

// Create a reference to an entity of the given symbol and GLSL type.
glsl::Expr_ref *GLSLWriterBasePass::create_reference(
    glsl::Symbol *sym,
    glsl::Type *type)
{
    return create_reference(get_type_name(sym), type);
}

// Create a reference to a variable of the given type.
glsl::Expr_ref *GLSLWriterBasePass::create_reference(
    glsl::Symbol *sym,
    llvm::Type   *type)
{
    return create_reference(sym, convert_type(type));
}

// Create a reference to an entity.
glsl::Expr_ref *GLSLWriterBasePass::create_reference(
    glsl::Definition *def)
{
    glsl::Expr_ref *ref = create_reference(def->get_symbol(), def->get_type());
    ref->set_definition(def);
    return ref;
}

// Create a new unary expression.
glsl::Expr *GLSLWriterBasePass::create_unary(
    glsl::Location const       &loc,
    glsl::Expr_unary::Operator op,
    glsl::Expr                *arg)
{
    if (op == glsl::Expr_unary::OK_LOGICAL_NOT) {
        glsl::Type *type = arg->get_type();
        if (glsl::is<glsl::Type_vector>(type->skip_type_alias())) {
            // GLSL defined the logical not operator only on bool, use the
            // runtime function not() on boolean vectors
            return create_runtime_call(loc, "not", arg);
        }
    }
    return m_expr_factory.create_unary(loc, op, arg);
}

/// Check if the given GLSL type is either bool or a bool based vector.
static bool is_bool_or_bool_vector(glsl::Type *type)
{
    if (glsl::Type_vector *vt = glsl::as<glsl::Type_vector>(type)) {
        type = vt->get_element_type();
    }

    return glsl::is<glsl::Type_bool>(type->skip_type_alias());
}

// Create a new binary expression.
glsl::Expr *GLSLWriterBasePass::create_binary(
    glsl::Expr_binary::Operator op,
    glsl::Expr                  *left,
    glsl::Expr                  *right)
{
    if (op == Expr_binary::OK_BITWISE_XOR) {
        if (is_bool_or_bool_vector(left->get_type())) {
            // map XOR on boolean values to NOT-EQUAL
            op = Expr_binary::OK_NOT_EQUAL;
        }
    }

    if (op == glsl::Expr_binary::OK_EQUAL || op == glsl::Expr_binary::OK_NOT_EQUAL) {
        Value *left_val = nullptr;
        Value *right_val = nullptr;

        if (is_bool_or_bool_vector(left->get_type())) {
            // if we compare bool entities we can remove some ugly cases
            if (Expr_literal *left_lit = as<Expr_literal>(left)) {
                left_val = left_lit->get_value();
            }

            if (Expr_literal *right_lit = as<Expr_literal>(right)) {
                right_val = right_lit->get_value();
            }
        }

        if (left_val == nullptr) {
            // compute left expression
            if (right_val == nullptr) {
                // compute right expression and the compare result below
                ;
            } else {
                // right expression is a constant
                if (op ==  glsl::Expr_binary::OK_NOT_EQUAL) {
                    // left != FALSE ==> left
                    // left != TRUE == !left
                    return right_val->is_zero() ?
                        left :
                        create_unary(
                            zero_loc,
                            glsl::Expr_unary::OK_LOGICAL_NOT,
                            left);
                } else {
                    // left == FALSE ==> !left
                    // left == TRUE ==> left
                    return right_val->is_zero() ?
                        create_unary(
                            zero_loc,
                            glsl::Expr_unary::OK_LOGICAL_NOT,
                            left) :
                        left;
                }
            }
        } else if (right_val == nullptr) {
            // left expression is a constant
            if (op ==  glsl::Expr_binary::OK_NOT_EQUAL) {
                // FALSE != right ==> right
                // TRUE != right == !right
                return left_val->is_zero() ?
                    right :
                    create_unary(
                        zero_loc,
                        glsl::Expr_unary::OK_LOGICAL_NOT,
                        right);
            } else {
                // FALSE == right ==> !right
                // TRUE == right ==> right
                return left_val->is_zero() ?
                    create_unary(
                        zero_loc,
                        glsl::Expr_unary::OK_LOGICAL_NOT,
                        right) :
                    right;
            }
        }
    }

#define CASE(ok, name) case glsl::Expr_binary::ok: func = name; break;

    char const *func = nullptr;
    switch (op) {
    CASE(OK_LESS,             "lessThan")
    CASE(OK_LESS_OR_EQUAL,    "lessThanEqual")
    CASE(OK_GREATER_OR_EQUAL, "greaterThanEqual")
    CASE(OK_GREATER,          "greaterThan")
    CASE(OK_EQUAL,            "equal")
    CASE(OK_NOT_EQUAL,        "notEqual")
    default: break;
    }
#undef CASE

    // check, if we need a vector op
    if (func != nullptr) {
        glsl::Type_vector *left_tp  = glsl::as<glsl::Type_vector>(left->get_type());
        glsl::Type_vector *right_tp = glsl::as<glsl::Type_vector>(right->get_type());

        if (left_tp != nullptr) {
            if (right_tp == nullptr) {
                right = create_type_cast(left_tp, right);
            }
        } else if (right_tp != nullptr) {
            left = create_type_cast(right_tp, left);
        } else {
            func = nullptr;
        }
    }

    if (func != nullptr) {
        glsl::Expr *args[] = { left, right };
        return create_runtime_call(zero_loc, func, args);
    }

    return m_expr_factory.create_binary(op, left, right);
}

// Modify a double based argument type to a float based.
glsl::Type *GLSLWriterBasePass::to_genType(glsl::Type *arg_type) const
{
    switch (arg_type->get_kind()) {
    case glsl::Type::TK_DOUBLE:
        return m_tc.float_type;
    case glsl::Type::TK_VECTOR:
        {
            glsl::Type_vector *v_type = glsl::cast<glsl::Type_vector>(arg_type);
            glsl::Type        *e_type = v_type->get_element_type();

            if (glsl::is<glsl::Type_double>(e_type)) {
                switch (v_type->get_size()) {
                case 2: return m_tc.vec2_type;
                case 3: return m_tc.vec3_type;
                case 4: return m_tc.vec4_type;
                }
            }
        }
        return arg_type;
    case glsl::Type::TK_MATRIX:
        {
            glsl::Type_matrix *m_type = glsl::cast<glsl::Type_matrix>(arg_type);
            glsl::Type_vector *v_type = m_type->get_element_type();
            glsl::Type_vector *n_type = glsl::cast<glsl::Type_vector>(to_genType(v_type));

            if (n_type != v_type) {
                switch (m_type->get_columns()) {
                case 2: return m_tc.get_matrix(n_type, 2);
                case 3: return m_tc.get_matrix(n_type, 3);
                case 4: return m_tc.get_matrix(n_type, 4);
                }
            }
        }
        return arg_type;
    default:
        return arg_type;
    }
}

// Modify a float based argument type to a double based.
glsl::Type *GLSLWriterBasePass::to_genDType(glsl::Type *arg_type) const
{
    switch (arg_type->get_kind()) {
    case glsl::Type::TK_FLOAT:
        return m_tc.double_type;
    case glsl::Type::TK_VECTOR:
        {
            glsl::Type_vector *v_type = glsl::cast<glsl::Type_vector>(arg_type);
            glsl::Type        *e_type = v_type->get_element_type();

            if (glsl::is<glsl::Type_float>(e_type)) {
                switch (v_type->get_size()) {
                case 2: return m_tc.dvec2_type;
                case 3: return m_tc.dvec3_type;
                case 4: return m_tc.dvec4_type;
                }
            }
        }
        return arg_type;
    case glsl::Type::TK_MATRIX:
        {
            glsl::Type_matrix *m_type = glsl::cast<glsl::Type_matrix>(arg_type);
            glsl::Type_vector *v_type = m_type->get_element_type();
            glsl::Type_vector *n_type = glsl::cast<glsl::Type_vector>(to_genDType(v_type));

            if (n_type != v_type) {
                switch (m_type->get_columns()) {
                case 2: return m_tc.get_matrix(n_type, 2);
                case 3: return m_tc.get_matrix(n_type, 3);
                case 4: return m_tc.get_matrix(n_type, 4);
                }
            }
        }
        return arg_type;
    default:
        return arg_type;
    }
}

// Find an (fully) matching overload.
glsl::Def_function *GLSLWriterBasePass::find_overload(
    glsl::Def_function            *f_def,
    Array_ref<glsl::Expr *> const &args,
    Runtime_flags                 &flags) const
{
    Runtime_flags possible_flags = RTF_NONE;

    flags = RTF_NONE;

    switch (f_def->get_semantics()) {
    case glsl::Def_function::DS_RT_acos:
    case glsl::Def_function::DS_RT_asin:
    case glsl::Def_function::DS_RT_atan:
    case glsl::Def_function::DS_RT_cos:
    case glsl::Def_function::DS_RT_degrees:
    case glsl::Def_function::DS_RT_exp:
    case glsl::Def_function::DS_RT_exp2:
    case glsl::Def_function::DS_RT_log:
    case glsl::Def_function::DS_RT_log2:
    case glsl::Def_function::DS_RT_pow:
    case glsl::Def_function::DS_RT_radians:
    case glsl::Def_function::DS_RT_sin:
    case glsl::Def_function::DS_RT_tan:
    case glsl::Def_function::DS_RT_cosh:
    case glsl::Def_function::DS_RT_sinh:
    case glsl::Def_function::DS_RT_tanh:
        // always on genType, but NO genDType
        possible_flags |= RTF_CONV_DOUBLE;
        break;
    default:
        break;
    }

    size_t n_args = args.size();

    Small_VLA<glsl::Type *, 8> arg_types(m_alloc, n_args);
    for (size_t i = 0; i < n_args; ++i) {
        glsl::Type *arg_type = args[i]->get_type()->skip_type_alias();

        if (possible_flags & RTF_CONV_DOUBLE) {
            glsl::Type *n_type = to_genType(arg_type);
            if (n_type != arg_type) {
                flags |= RTF_CONV_DOUBLE;
                arg_type = n_type;
            }
        }
        arg_types[i] = arg_type;
    }

    for (; f_def != nullptr; f_def = as_or_null<Def_function>(f_def->get_prev_def())) {
        glsl::Type_function *func_type = f_def->get_type();

        if (func_type->get_parameter_count() != n_args) {
            continue;
        }

        bool match = true;
        for (size_t i = 0; match && i < n_args; ++i) {
            glsl::Type_function::Parameter *param = func_type->get_parameter(i);
            glsl::Type *param_type = param->get_type()->skip_type_alias();
            glsl::Type *arg_type   = arg_types[i];

            if (param_type != arg_type) {
                match = false;
            }
        }
        if (match) {
            // found it
            return f_def;
        }
    }
    return nullptr;
}

// Get the constructor for the given GLSL type.
glsl::Def_function *GLSLWriterBasePass::lookup_constructor(
    glsl::Type                    *type,
    Array_ref<glsl::Expr *> const &args) const
{
    glsl::Scope  *scope = m_def_tab.get_type_scope(type);
    if (scope == nullptr) {
        // this type has no scope, bad
        MDL_ASSERT(!"no type scope in GLSLWriter::lookup_constructor()");
        return nullptr;
    }

    glsl::Symbol     *sym = type->get_sym();
    glsl::Definition *def = scope->find_definition_in_scope(sym);
    if (def == nullptr) {
        MDL_ASSERT(!"type scope has no constructor in GLSLWriter::lookup_constructor()");
        return nullptr;
    }

    glsl::Def_function *f_def = as<glsl::Def_function>(def);
    if (f_def != nullptr) {
        Runtime_flags flags;
        f_def = find_overload(f_def, args, flags);
        MDL_ASSERT(flags == RTF_NONE);
    }
    // parameter types did not match
    MDL_ASSERT(
        f_def != nullptr &&
        "unexpected parameter type mismatch in GLSLWriter::lookup_constructor()");
    return f_def;
}

// Get the runtime function for the given GLSL type.
glsl::Def_function *GLSLWriterBasePass::lookup_runtime(
    glsl::Symbol                  *sym,
    Array_ref<glsl::Expr *> const &args,
    Runtime_flags                 &flags) const
{
    flags = RTF_NONE;
    glsl::Scope *scope = m_def_tab.get_predef_scope();

    glsl::Definition *def = scope->find_definition_in_scope(sym);
    if (def == nullptr) {
        // does not exits
        return nullptr;
    }
    glsl::Def_function *f_def = as<glsl::Def_function>(def);
    if (f_def != nullptr) {
        f_def = find_overload(f_def, args, flags);
    }
    // parameter types did not match
    MDL_ASSERT(
        f_def != nullptr &&
        "unexpected parameter type mismatch in GLSLWriter::lookup_runtime()");
    return f_def;
}

// Get the runtime function for the given GLSL type.
glsl::Def_function *GLSLWriterBasePass::lookup_runtime(
    char const                    *name,
    Array_ref<glsl::Expr *> const &args,
    Runtime_flags                 &flags) const
{
    glsl::Symbol *sym = m_symbol_table.lookup_symbol(name);
    if (sym == nullptr) {
        // does not exists
        return nullptr;
    }
    return lookup_runtime(sym, args, flags);
}

// Create a type cast expression.
glsl::Expr *GLSLWriterBasePass::create_type_cast(
    glsl::Type *dst,
    glsl::Expr *arg)
{
    if (glsl::Expr_literal *lit = glsl::as<glsl::Expr_literal>(arg)) {
        // try to convert the literal first
        glsl::Value *v = lit->get_value();
        glsl::Value *n = v->convert(m_value_factory, dst);

        if (!glsl::is<glsl::Value_bad>(n)) {
            return m_expr_factory.create_literal(arg->get_location(), n);
        }
    }

    glsl::Def_function *def = lookup_constructor(dst, arg);
    if (def == nullptr) {
        error(mi::mdl::INTERNAL_JIT_UNSUPPORTED_EXPR);
        return m_expr_factory.create_invalid(zero_loc);
    }
    MDL_ASSERT(dst->skip_type_alias() == def->get_type()->get_return_type()->skip_type_alias());

    glsl::Expr *callee = create_reference(def);
    glsl::Expr *call = m_expr_factory.create_call(callee, arg);
    call->set_type(def->get_type()->get_return_type());
    return call;
}

// Creates a bitcast expression from float to int.
glsl::Expr *GLSLWriterBasePass::create_float2int_bitcast(
    glsl::Type *dst,
    glsl::Expr *arg)
{
    Runtime_flags flags;
    glsl::Def_function *def = lookup_runtime("floatBitsToInt", arg, flags);
    if (def == nullptr) {
        error(mi::mdl::INTERNAL_JIT_UNSUPPORTED_EXPR);
        return m_expr_factory.create_invalid(zero_loc);
    }
    MDL_ASSERT(dst->skip_type_alias() == def->get_type()->get_return_type()->skip_type_alias());
    MDL_ASSERT(flags == RTF_NONE);

    glsl::Expr *callee = create_reference(def);
    glsl::Expr *call = m_expr_factory.create_call(callee, arg);
    call->set_type(def->get_type()->get_return_type());
    return call;
}

// Creates a bitcast expression from int to float.
glsl::Expr *GLSLWriterBasePass::create_int2float_bitcast(
    glsl::Type *dst,
    glsl::Expr *arg)
{
    Runtime_flags flags;
    glsl::Def_function *def = lookup_runtime("intBitsToFloat", arg, flags);
    if (def == nullptr) {
        error(mi::mdl::INTERNAL_JIT_UNSUPPORTED_EXPR);
        return m_expr_factory.create_invalid(zero_loc);
    }
    MDL_ASSERT(dst->skip_type_alias() == def->get_type()->get_return_type()->skip_type_alias());
    MDL_ASSERT(flags == RTF_NONE);

    glsl::Expr *callee = create_reference(def);
    glsl::Expr *call = m_expr_factory.create_call(callee, arg);
    call->set_type(def->get_type()->get_return_type());
    return call;
}

// Create a call to a GLSL runtime function.
glsl::Expr *GLSLWriterBasePass::create_runtime_call(
    glsl::Location const          &loc,
    llvm::Function                *func,
    Array_ref<glsl::Expr *> const &args)
{
    // call to an unknown entity
    llvm::StringRef name = func->getName();

    bool is_llvm_intrinsic = name.startswith("llvm.");
    if (is_llvm_intrinsic || name.startswith("glsl.")) {
        // handle GLSL or LLVM intrinsics
        name = name.drop_front(5);
        size_t pos = name.find('.');
        name = name.slice(0, pos);

        if (is_llvm_intrinsic) {
            // need some mapping between LLVM intrinsics and HLSL/GLSL
            if (name == "fabs") {
                name = "abs";
            } else if (name == "minnum") {
                name = "min";
            } else if (name == "maxnum") {
                name = "max";
            }
        }
    }

    return create_runtime_call(loc, name.str().c_str(), args);
}

// Create a call to a GLSL runtime function.
glsl::Expr *GLSLWriterBasePass::create_runtime_call(
    glsl::Location const          &loc,
    char const                    *func,
    Array_ref<glsl::Expr *> const &args)
{
    Runtime_flags flags;
    glsl::Def_function *def = lookup_runtime(func, args, flags);
    if (def == nullptr) {
        error(mi::mdl::INTERNAL_JIT_UNSUPPORTED_EXPR);
        return m_expr_factory.create_invalid(loc);
    }

    glsl::Expr_ref *callee = create_reference(def);

    size_t n_args = args.size();
    Small_VLA<glsl::Expr *, 8> mod_args(m_alloc, n_args);

    for (size_t i = 0; i < n_args; ++i) {
        glsl::Expr *arg = args[i];

        if (flags & RTF_CONV_DOUBLE) {
            // adapt arguments
            glsl::Type *arg_type = arg->get_type();
            glsl::Type *n_type   = to_genType(arg_type);

            if (arg_type != n_type) {
                // add a down-cast
                arg = create_type_cast(n_type, arg);
            }
        }
        mod_args[i] = arg;
    }

    glsl::Expr *call = m_expr_factory.create_call(callee, mod_args);

    glsl::Type *ret_type = def->get_type()->get_return_type();
    call->set_type(ret_type);
    call->set_location(loc);

    if (flags & RTF_CONV_DOUBLE) {
        // adapt result
        glsl::Type *n_type = to_genDType(ret_type);

        if (ret_type != n_type) {
            call = create_type_cast(n_type, call);
            call->set_location(loc);
        }
    }

    return call;
}

// Returns true if a compound expression is allowed in the given context.
bool GLSLWriterBasePass::compound_allowed(
    bool is_global) const
{
    // FIXME: context is ti simple here
    if (is_global) {
        return m_unit->get_glslang_context().has_c_style_initializer();
    }
    return false;
}

// Creates an initializer.
glsl::Expr *GLSLWriterBasePass::create_initializer(
    glsl::Location const          &loc,
    glsl::Type                   *type,
    Array_ref<glsl::Expr *> const &args)
{
    if (m_unit->get_glslang_context().has_c_style_initializer()) {
        glsl::Expr *res = m_expr_factory.create_compound(loc, args);
        res->set_type(type);
        return res;
    }

    glsl::Type_name *tn      = get_type_name(type);
    glsl::Expr      *callee  = create_reference(tn, type);
    glsl::Expr      *call    = m_expr_factory.create_call(callee, args);

    call->set_type(type);
    return call;
}

// Set the type qualifier for a global constant in GLSL.
void GLSLWriterBasePass::set_global_constant_qualifier(
    glsl::Type_qualifier &tq)
{
    tq.set_storage_qualifier(glsl::SQ_CONST);
}

// Convert a function type parameter qualifier to a AST parameter qualifier.
glsl::Parameter_qualifier GLSLWriterBasePass::convert_type_modifier_to_param_qualifier(
    glsl::Type_function::Parameter *param)
{
    glsl::Parameter_qualifier param_qualifier = glsl::PQ_IN;
    switch (param->get_modifier()) {
    case Type_function::Parameter::PM_IN:
        param_qualifier = glsl::PQ_IN;
        break;
    case Type_function::Parameter::PM_OUT:
        param_qualifier = glsl::PQ_OUT;
        break;
    case Type_function::Parameter::PM_INOUT:
        param_qualifier = glsl::PQ_INOUT;
        break;
    }
    return param_qualifier;
}

// Set the out parameter qualifier.
void GLSLWriterBasePass::make_out_parameter(
    glsl::Type_name *param_type_name)
{
    param_type_name->get_qualifier().set_storage_qualifier(glsl::SQ_OUT);
}

// Convert the LLVM debug location (if any is attached to the given instruction)
// to an GLSL location.
glsl::Location GLSLWriterBasePass::convert_location(
    llvm::Instruction *I)
{
    if (m_use_dbg) {
        if (llvm::MDNode *md_node = I->getMetadata(llvm::LLVMContext::MD_dbg)) {
            if (llvm::isa<llvm::DILocation>(md_node)) {
                llvm::DebugLoc Loc(md_node);
                unsigned        Line   = Loc->getLine();
                unsigned        Column = Loc->getColumn();
                llvm::StringRef fname  = Loc->getFilename();

                string s(fname.data(), fname.data() + fname.size(), m_alloc);
                Ref_fname_id_map::const_iterator it = m_ref_fnames.find(s);
                unsigned file_id;
                if (it == m_ref_fnames.end()) {
                    file_id = m_unit->register_filename(s.c_str());
                    m_ref_fnames.insert(Ref_fname_id_map::value_type(s, file_id));
                } else {
                    file_id = it->second;
                }

                return glsl::Location(Line, Column, file_id);
            }
        }
    }
    return zero_loc;
}

// Add a JIT backend warning message to the messages.
void GLSLWriterBasePass::warning(int code)
{
    // FIXME: get from JIT be
    char MESSAGE_CLASS = 'J';

    mi::mdl::Error_params params(m_alloc);

    string msg(m_messages.format_msg(code, MESSAGE_CLASS, params));
    m_messages.add_warning_message(code, MESSAGE_CLASS, 0, NULL, msg.c_str());
}

// Add a JIT backend error message to the messages.
void GLSLWriterBasePass::error(int code)
{
    // FIXME: get from JIT be
    char MESSAGE_CLASS = 'J';

    mi::mdl::Error_params params(m_alloc);

    string msg(m_messages.format_msg(code, MESSAGE_CLASS, params));
    m_messages.add_warning_message(code, MESSAGE_CLASS, 0, NULL, msg.c_str());
}

/// Check if this declaration needs an extra newline when dumped.
static int last_decl_kind(
    glsl::Declaration const *decl)
{
    int kind = decl->get_kind();
    if (kind == glsl::Declaration::DK_FUNCTION) {
        glsl::Declaration_function const *f_decl = cast<glsl::Declaration_function>(decl);
        if (!f_decl->is_prototype()) {
            kind = -2;
        }
    } else if (kind == glsl::Declaration::DK_VARIABLE) {
        glsl::Declaration_variable const *vdecl = cast<glsl::Declaration_variable>(decl);
        if (vdecl->empty()) {
            kind = -2;
        }
    }
    return kind;
}

// Called for every function that is just a prototype in the original LLVM module.
glsl::Def_function *GLSLWriterBasePass::create_prototype(
    llvm::Function &func)
{
    glsl::Def_function  *func_def  = create_definition(&func);
    glsl::Type_function *func_type = func_def->get_type();
    glsl::Type          *ret_type  = func_type->get_return_type();
    glsl::Type          *out_type  = nullptr;

    // reset the name IDs
    m_next_unique_name_id = 0;

    if (is<glsl::Type_void>(ret_type) &&
        !func.getFunctionType()->getReturnType()->isVoidTy())
    {
        // return type was converted into out parameter
        typename glsl::Type_function::Parameter *param = func_type->get_parameter(0);
        if (param->get_modifier() == glsl::Type_function::Parameter::PM_OUT) {
            out_type = param->get_type();
        }
    }

    // create the declaration for the function
    Type_name *ret_type_name = get_type_name(ret_type);

    glsl::Symbol *func_sym   = func_def->get_symbol();
    glsl::Name   *func_name  = get_name(zero_loc, func_sym);
    Declaration_function *decl_func = m_decl_factory.create_function(
        ret_type_name, func_name);

    // create the function body
    {
        glsl::Definition_table::Scope_enter enter(m_def_tab, func_def);

        // now create the declarations
        unsigned first_param_ofs = 0;
        if (out_type != nullptr) {
            glsl::Type_name         *param_type_name = get_type_name(out_type);
            glsl::Declaration_param *decl_param = m_decl_factory.create_param(param_type_name);
            add_array_specifiers(decl_param, out_type);
            make_out_parameter(param_type_name);


            glsl::Symbol *param_sym  = get_unique_sym("p_result", "p_result");
            glsl::Name   *param_name = get_name(zero_loc, param_sym);
            decl_param->set_name(param_name);

            glsl::Def_param *m_out_def = m_def_tab.enter_parameter_definition(
                param_sym, out_type, &param_name->get_location());
            m_out_def->set_declaration(decl_param);
            param_name->set_definition(m_out_def);

            decl_func->add_param(decl_param);

            ++first_param_ofs;
        }

        for (llvm::Argument &arg_it : func.args()) {
            llvm::Type *arg_llvm_type = arg_it.getType();
            glsl::Type *param_type    = convert_type(arg_llvm_type);

            if (is<glsl::Type_void>(param_type)) {
                // skip void typed parameters
                continue;
            }

            unsigned        i                = arg_it.getArgNo();
            glsl::Type_name *param_type_name = get_type_name(param_type);

            glsl::Declaration_param *decl_param = m_decl_factory.create_param(param_type_name);
            add_array_specifiers(decl_param, param_type);

            glsl::Type_function::Parameter *param = func_type->get_parameter(i + first_param_ofs);

            glsl::Parameter_qualifier param_qualifier =
                convert_type_modifier_to_param_qualifier(param);
            param_type_name->get_qualifier().set_parameter_qualifier(param_qualifier);

            char templ[16];
            snprintf(templ, sizeof(templ), "p_%u", i);

            glsl::Symbol *param_sym  = get_unique_sym(arg_it.getName(), templ);
            glsl::Name   *param_name = get_name(zero_loc, param_sym);
            decl_param->set_name(param_name);

            glsl::Def_param *param_def = m_def_tab.enter_parameter_definition(
                param_sym, param_type, &param_name->get_location());
            param_def->set_declaration(decl_param);
            param_name->set_definition(param_def);

            decl_func->add_param(decl_param);
        }
    }

    func_def->set_declaration(decl_func);
    decl_func->set_definition(func_def);
    func_name->set_definition(func_def);

    // do not add the prototype to the compilation unit: this will be done automatically
    // by the declaration inserter IFF t is called

    return func_def;
}

// Return true if the user of an instruction requires its materialization
bool  GLSLWriterBasePass::must_be_materialized(
    llvm::Instruction *user)
{
    unsigned user_op_code = user->getOpcode();
    switch (user_op_code) {
    case llvm::Instruction::Call:
        {
            llvm::CallInst *call = llvm::cast<llvm::CallInst>(user);
            llvm::Function *called_func = call->getCalledFunction();

            // don't let memset, memcpy or llvm.lifetime_end enforce materialization
            // of instructions
            llvm::Intrinsic::ID intrinsic_id =
                called_func ? called_func->getIntrinsicID()
                : llvm::Intrinsic::not_intrinsic;
            if (intrinsic_id != llvm::Intrinsic::memset &&
                intrinsic_id != llvm::Intrinsic::memcpy &&
                intrinsic_id != llvm::Intrinsic::lifetime_end &&
                !call->onlyReadsMemory())
            {
                return true;
            }
            break;
        }
    case llvm::Instruction::ShuffleVector:
        return true;
        break;
    default:
        break;
    }
    return false;
}

template <typename Value_type, typename Basic_type>
static void store(unsigned char *data, glsl::Value *value, size_t size)
{
    Value_type *v = cast<Value_type>(value);
    Basic_type d = v->get_value();

    MDL_ASSERT(sizeof(d) == size);
    memcpy(data, &d, size);
}

// Convert a GLSL value into binary data.
size_t GLSLWriterBasePass::fill_binary_data(
    unsigned char *data,
    glsl::Value *value,
    size_t        offset)
{
    glsl::Type *type = value->get_type();
    size_t     align = m_tc.get_type_alignment(type);

    offset = (offset + align - 1) & ~(align - 1);

    size_t size = m_tc.get_type_size(type);

    switch (value->get_kind()) {
    case glsl::Value::VK_BAD:
        // should not happen
        MDL_ASSERT(!"value_bad should not occur here");
        break;

    case glsl::Value::VK_VOID:
        // strange, but ok
        MDL_ASSERT(!"value_void should not occur here");
        break;

    case glsl::Value::VK_BOOL:
        store<glsl::Value_bool, bool>(&data[offset], value, size);
        offset += size;
        break;

    case glsl::Value::VK_INT8:
        store<glsl::Value_int_8, int8_t>(&data[offset], value, size);
        offset += size;
        break;

    case glsl::Value::VK_UINT8:
        store<glsl::Value_uint_8, uint8_t>(&data[offset], value, size);
        offset += size;
        break;

    case glsl::Value::VK_INT16:
        store<glsl::Value_int_16, int16_t>(&data[offset], value, size);
        offset += size;
        break;

    case glsl::Value::VK_UINT16:
        store<glsl::Value_uint_16, uint16_t>(&data[offset], value, size);
        offset += size;
        break;

    case glsl::Value::VK_INT:
        store<glsl::Value_int_32, int32_t>(&data[offset], value, size);
        offset += size;
        break;

    case glsl::Value::VK_UINT:
        store<glsl::Value_uint_32, uint32_t>(&data[offset], value, size);
        offset += size;
        break;

    case glsl::Value::VK_INT64:
        store<glsl::Value_int_64, int64_t>(&data[offset], value, size);
        offset += size;
        break;

    case glsl::Value::VK_UINT64:
        store<glsl::Value_uint_64, uint64_t>(&data[offset], value, size);
        offset += size;
        break;

    case glsl::Value::VK_HALF:
        store<glsl::Value_half, float>(&data[offset], value, size);
        offset += size;
        break;

    case glsl::Value::VK_FLOAT:
        store<glsl::Value_float, float>(&data[offset], value, size);
        offset += size;
        break;

    case glsl::Value::VK_DOUBLE:
        store<glsl::Value_double, double>(&data[offset], value, size);
        offset += size;
        break;

    case glsl::Value::VK_VECTOR:
    case glsl::Value::VK_MATRIX:
    case glsl::Value::VK_ARRAY:
    case glsl::Value::VK_STRUCT:
        {
            glsl::Value_compound *v = cast<glsl::Value_compound>(value);

            for (size_t i = 0, n = v->get_component_count(); i < n; ++i) {
                glsl::Value *elem = v->get_value(i);

                offset = fill_binary_data(data, elem, offset);
            }
        }
        break;
    }
    return offset;
}

// Finalize the compilation unit and write it to the given output stream.
void GLSLWriterBasePass::finalize(
    llvm::Module                                               &M,
    Generated_code_source                                      *code,
    list<std::pair<char const *, glsl::Symbol *> >::Type const &remaps)
{
    String_stream_writer out(code->access_src_code());
    mi::base::Handle<glsl::Printer> printer(m_compiler->create_printer(&out));

    // analyze and optimize it
    m_unit->analyze(*m_compiler.get());

    printer->enable_locations(m_use_dbg);

    // helper data
    typedef ptr_hash_map<glsl::Symbol, char const *>::Type Mapped_type;
    Mapped_type mapped(m_alloc);
    if (!remaps.empty()) {
        typedef list<std::pair<char const *, Symbol *> >::Type Symbol_list;

        for (Symbol_list::const_iterator it(remaps.begin()), end(remaps.end()); it != end; ++it) {
            mapped[it->second] = it->first;
        }
    }

    // generate the version fragment
    {
        // ES even does not allow comments before version
        if (m_glslang_profile != glsl::GLSL_PROFILE_ES) {
            printer->print_comment("GLSL version and extensions");
        }
        printer->print_version(m_ctx);
        printer->print_extensions(m_ctx);
        printer->nl();

        if (m_glslang_profile == glsl::GLSL_PROFILE_ES) {
            // set the precision for the used types, is might be not defined by default
            printer->print("precision highp float;"); printer->nl();
            printer->print("precision highp int;"); printer->nl();
        }
    }

    // generate the defines fragment
    if (!remaps.empty()) {
        printer->print_comment("defines");

        typedef list<std::pair<char const *, Symbol *> >::Type Symbol_list;

        for (Symbol_list::const_iterator it(remaps.begin()), end(remaps.end()); it != end; ++it) {
            printer->print("#define MAPPED_");
            printer->print((*it).first);
            printer->print(" 1");
            printer->nl();
        }
        printer->nl();
    }

    // generate the API type fragment
    {
        printer->print_comment("API types");

        int last_kind = -1;
        for (glsl::Compilation_unit::const_iterator
            it(m_unit->decl_begin()), end(m_unit->decl_end());
            it != end;
            ++it)
        {
            glsl::Declaration const *decl = it;

            if (m_api_decls.find(decl) != m_api_decls.end()) {
                glsl::Declaration::Kind kind = decl->get_kind();
                if (last_kind != -1 && last_kind != kind) {
                    printer->nl();
                }

                printer->print(decl);
                printer->nl();

                last_kind = last_decl_kind(decl);
            }
        }
    }

    // generate the user type segment
    {
        printer->print_comment("user defined structs");
        for (glsl::Compilation_unit::const_iterator
            it(m_unit->decl_begin()), end(m_unit->decl_end());
            it != end;
            ++it)
        {
            glsl::Declaration const *decl = it;

            if (!is<glsl::Declaration_struct>(decl)) {
                continue;
            }

            // ignore API type declarations
            if (m_api_decls.find(decl) != m_api_decls.end()) {
                // already dumped in state fragment
                continue;
            }

            printer->print(decl);
            printer->nl();
        }
    }

    // generate the globals fragment
    {
        printer->print_comment("globals");

        int last_kind = -1;
        for (glsl::Compilation_unit::const_iterator
            it(m_unit->decl_begin()), end(m_unit->decl_end());
            it != end;
            ++it)
        {
            glsl::Declaration const *decl = it;

            // ignore structs and functions
            if (is<glsl::Declaration_struct>(decl) || is<glsl::Declaration_function>(decl)) {
                continue;
            }

            // ignore API declarations
            if (m_api_decls.find(decl) != m_api_decls.end()) {
                // already dumped in API fragment
                continue;
            }

            glsl::Declaration::Kind kind = decl->get_kind();
            if (last_kind != -1 && last_kind != kind) {
                printer->nl();
            }

            printer->print(decl);
            printer->nl();

            last_kind = last_decl_kind(decl);
        }
    }

    // generate the functions prototype fragment
    {
        printer->print_comment("prototypes");
        for (glsl::Compilation_unit::const_iterator
            it(m_unit->decl_begin()), end(m_unit->decl_end());
            it != end;
            ++it)
        {
            glsl::Declaration const *decl = it;

            if (!is<glsl::Declaration_function>(decl)) {
                continue;
            }

            glsl::Declaration_function const *f_decl = cast<glsl::Declaration_function>(decl);

            if (!f_decl->is_prototype()) {
                continue;
            }

            glsl::Symbol *f_sym = f_decl->get_definition()->get_symbol();
            Mapped_type::const_iterator mit(mapped.find(f_sym));
            if (mit != mapped.end()) {
                mi::mdl::string s(m_alloc);
                mi::mdl::MDL_name_mangler mangler(m_alloc, s);

                if (!mangler.demangle(mit->second, strlen(mit->second))) {
                    // not a mangled name
                    s = mit->second;
                }

                printer->print_comment(("replaces " + s).c_str());
            }
            printer->print(decl);
            printer->nl();
        }
    }

    // generate the functions fragment
    {
        printer->print_comment("functions");

        int last_kind = -1;
        for (glsl::Compilation_unit::const_iterator
            it(m_unit->decl_begin()), end(m_unit->decl_end());
            it != end;
            ++it)
        {
            glsl::Declaration const *decl = it;

            if (glsl::Declaration_function const *f_decl = as<glsl::Declaration_function>(decl)) {
                if (f_decl->is_prototype()) {
                    // ignore prototypes
                    continue;
                }
            } else {
                // ignore non-functions
                continue;
            }

            // ignore API type declarations
            if (/*m_state_decls.find(decl) != m_state_decls.end()*/false) {
                // already dumped in API type fragment
                continue;
            }

            glsl::Declaration::Kind kind = decl->get_kind();
            if (last_kind != -1 && last_kind != kind) {
                printer->nl();
            }

            printer->print(decl);
            printer->nl();

            last_kind = last_decl_kind(decl);
        }
    }

    // generate uniform initializers
    // use the allocator of the GLSL code object, so the ownership is transfered easily
    IAllocator *code_alloc = code->get_allocator();

    if (m_place_uniform_inits_into_ssbo && m_ssbo_decl != NULL) {
        // combine all initializers into one
        glsl::Symbol *sym = m_ssbo_decl->get_name()->get_symbol();

        // collect types first
        typedef vector<glsl::Type_struct::Field>::Type Field_vec;
        typedef vector<glsl::Value *>::Type            Value_vec;

        Field_vec fields(m_alloc);
        Value_vec inits(m_alloc);

        for (Uniform_initializers::const_iterator it(m_uniform_inits.begin()),
            end(m_uniform_inits.end());
            it != end; ++it)
        {
            Uniform_initializer const &ui = *it;
            glsl::Definition *def  = ui.get_def();
            glsl::Value      *init = ui.get_init();

            fields.push_back(glsl::Type_struct::Field(init->get_type(), def->get_symbol()));
            inits.push_back(init);
        }

        glsl::Type_struct *ssbo_type = m_tc.get_struct(fields, sym);
        glsl::Value *ssbo_init = m_value_factory.get_struct(ssbo_type, inits);

        size_t len = m_tc.get_type_size(ssbo_type);
        unsigned char *data = static_cast<unsigned char *>(code_alloc->malloc(len));

        fill_binary_data(data, ssbo_init, 0);

        code->add_data_segment(sym->get_name(), data, len);

        code_alloc->free(data);
    } else {
        // generate every initializer alone
        for (Uniform_initializers::const_iterator it(m_uniform_inits.begin()),
            end(m_uniform_inits.end());
            it != end; ++it)
        {
            Uniform_initializer const &ui = *it;

            glsl::Definition *def  = ui.get_def();
            glsl::Value      *init = ui.get_init();

            size_t len = m_tc.get_type_size(init->get_type());
            unsigned char *data = static_cast<unsigned char *>(code_alloc->malloc(len));

            fill_binary_data(data, init, 0);

            code->add_data_segment(def->get_symbol()->get_name(), data, len);

            code_alloc->free(data);
        }
    }
}

// Check if the constant is too big for the code itself and should be moved to the SSBO.
bool GLSLWriterBasePass::move_to_uniform(glsl::Value *v)
{
    size_t ts = m_tc.get_type_size(v->get_type());

    return ts > m_max_const_size;
}

// Add an uniform initializer.
void GLSLWriterBasePass::add_uniform_initializer(
    glsl::Definition *def,
    glsl::Value      *v)
{
    m_uniform_inits.push_back(Uniform_initializer(def, v));
}

// Generates a new global static const variable to hold an LLVM value.
glsl::Definition *GLSLWriterBasePass::create_global_const(
    llvm::StringRef name, glsl::Expr *c_expr)
{
    glsl::Value *v = nullptr;
    if (glsl::Expr_literal *lit_expr = as<glsl::Expr_literal>(c_expr)) {
        v = lit_expr->get_value();
    }

    bool to_uniform = v != nullptr && move_to_uniform(v);

    if (to_uniform && m_place_uniform_inits_into_ssbo) {
        if (m_ssbo_decl == NULL) {
            // create the uniform buffer interface

            char const *ssbo_name = m_glsl_uniform_ssbo_name;
            if (ssbo_name == NULL || ssbo_name[0] == '\0') {
                ssbo_name = "mdl_buffer";
            }
            glsl::Symbol *if_sym = get_unique_sym(ssbo_name, "mdl_buffer");

            glsl::Name *if_name = m_decl_factory.create_name(zero_loc, if_sym);
            m_ssbo_decl = m_decl_factory.create_interface(zero_loc, if_name);

            glsl::Type_qualifier   &tq = m_ssbo_decl->get_qualifier();
            glsl::Layout_qualifier &lq = tq.get_layout_qualifier();

            // std430
            {
                glsl::Symbol *sym   = m_symbol_table.get_symbol("std430");
                glsl::Name   *ident = m_decl_factory.create_name(zero_loc, sym);

                glsl::Layout_qualifier_id *id =
                    m_decl_factory.create_layout_qualifier_id(zero_loc, ident, /*expr=*/nullptr);

                lq.push(id);
            }
            // binding = uConst
            if (m_glsl_uniform_ssbo_binding != ~0u) {
                glsl::Symbol *sym       = m_symbol_table.get_symbol("binding");
                glsl::Name   *ident     = m_decl_factory.create_name(zero_loc, sym);
                glsl::Value  *v_binding = m_value_factory.get_uint32(m_glsl_uniform_ssbo_binding);
                glsl::Expr   *e_binding = m_expr_factory.create_literal(zero_loc, v_binding);

                glsl::Layout_qualifier_id *id =
                    m_decl_factory.create_layout_qualifier_id(zero_loc, ident, e_binding);

                lq.push(id);
            }
            // set = uConst
            if (m_glsl_uniform_ssbo_set != ~0u) {
                glsl::Symbol *sym   = m_symbol_table.get_symbol("set");
                glsl::Name   *ident = m_decl_factory.create_name(zero_loc, sym);
                glsl::Value  *v_set = m_value_factory.get_uint32(m_glsl_uniform_ssbo_set);
                glsl::Expr   *e_set = m_expr_factory.create_literal(zero_loc, v_set);

                glsl::Layout_qualifier_id *id =
                    m_decl_factory.create_layout_qualifier_id(zero_loc, ident, e_set);

                lq.push(id);
            }

            tq.set_storage_qualifier(glsl::SQ_BUFFER);

            // set the readonly memory qualifier if possible
            if (m_unit->get_glslang_context().has_memory_qualifier()) {
                tq.set_storage_qualifier(glsl::SQ_READONLY);
            }

            m_unit->add_decl(m_ssbo_decl);
        }

        // Note: currently, we do not generate an instance for the SSBO buffer, hence we place
        // all buffer fields into the global scope
        glsl::Definition_table::Scope_transition scope(
            m_def_tab, m_def_tab.get_global_scope());

        // generate a new interface field
        glsl::Symbol *sym = get_unique_sym("mdl_field", "mdl_field");

        glsl::Type_name         *tn     = get_type_name(v->get_type());
        glsl::Declaration_field *f_decl = m_decl_factory.create_field_declaration(tn);
        glsl::Field_declarator  *field  = m_decl_factory.create_field(zero_loc);

        // give name
        field->set_name(get_name(zero_loc, sym));
        f_decl->add_field(field);

        glsl::Def_variable *var_def =
            m_def_tab.enter_variable_definition(sym, v->get_type(), &zero_loc);
        field->get_name()->set_definition(var_def);

        m_ssbo_decl->add_field(f_decl);

        // and add the initializer to the list
        add_uniform_initializer(var_def, v);

        return var_def;
    }

    // create a constant in the global scope
    glsl::Definition_table::Scope_transition scope(
        m_def_tab, m_def_tab.get_global_scope());

    glsl::Symbol *cnst_sym  = get_unique_sym(name, "glob_cnst");
    glsl::Type   *cnst_type = c_expr->get_type();

    cnst_type = m_tc.get_alias(cnst_type, Type::MK_CONST);

    // Note: GLSL support array specifiers on type AND variable
    glsl::Type_name      *cnst_type_name = get_type_name(inner_element_type(cnst_type));
    glsl::Type_qualifier &tq             = cnst_type_name->get_qualifier();

    set_global_constant_qualifier(tq);

    glsl::Declaration_variable *decl_cnst = m_decl_factory.create_variable(cnst_type_name);

    glsl::Init_declarator *init_decl = m_decl_factory.create_init_declarator(zero_loc);
    glsl::Name *var_name = get_name(zero_loc, cnst_sym);
    init_decl->set_name(var_name);
    add_array_specifiers(init_decl, cnst_type);
    decl_cnst->add_init(init_decl);

    init_decl->set_initializer(c_expr);

    Def_variable *cnst_def = m_def_tab.enter_variable_definition(
        cnst_sym, cnst_type, &var_name->get_location());
    cnst_def->set_declaration(decl_cnst);
    var_name->set_definition(cnst_def);

    // so far, add all declarations to the function scope
    m_unit->add_decl(decl_cnst);

    return cnst_def;
}

// Dump the current AST.
void GLSLWriterBasePass::dump_ast()
{
    glsl::dump_ast(m_unit.get());
}

}  // glsl
}  // mdl
}  // mi
