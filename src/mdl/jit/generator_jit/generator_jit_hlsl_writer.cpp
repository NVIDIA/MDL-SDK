/******************************************************************************
 * Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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
#include <llvm/Analysis/OptimizationRemarkEmitter.h>

#include <mdl/compiler/compilercore/compilercore_allocator.h>
#include <mdl/compiler/compilercore/compilercore_errors.h>
#include <mdl/compiler/compilercore/compilercore_messages.h>

#include <mdl/compiler/compiler_hlsl/compiler_hlsl_compiler.h>
#include <mdl/compiler/compiler_hlsl/compiler_hlsl_tools.h>
#include <mdl/compiler/compiler_hlsl/compiler_hlsl_visitor.h>

#include "generator_jit_hlsl_writer.h"
#include "generator_jit_streams.h"

#define DEBUG_TYPE "hlsl_writer"

namespace mi {
namespace mdl {
namespace hlsl {

// explicit instantiation for the HLSL case
template void HLSLWriterBasePass::add_array_specifiers<hlsl::Init_declarator>(
    hlsl::Init_declarator *, hlsl::Type *);
template void HLSLWriterBasePass::add_array_specifiers<hlsl::Declaration_param>(
    hlsl::Declaration_param *, hlsl::Type *);

// The zero location.
Location const HLSLWriterBasePass::zero_loc(0, 0);

// The prototype language.
IGenerated_code_executable::Prototype_language const HLSLWriterBasePass::proto_lang =
    IGenerated_code_executable::PL_HLSL;

// Constructor.
HLSLWriterBasePass::HLSLWriterBasePass(
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
, m_compiler(impl_cast<hlsl::Compiler>(initialize(m_alloc)))
, m_unit(create_compilation_unit(m_compiler.get(), "generated"))
, m_decl_factory(m_unit->get_declaration_factory())
, m_expr_factory(m_unit->get_expression_factory())
, m_stmt_factory(m_unit->get_statement_factory())
, m_tc(m_unit->get_type_factory())
, m_value_factory(m_unit->get_value_factory())
, m_symbol_table(m_unit->get_symbol_table())
, m_def_tab(m_unit->get_definition_table())
, m_type_cache(0, Type2type_map::hasher(), Type2type_map::key_equal(), alloc)
, m_debug_types(alloc)
, m_ref_fnames(0, Ref_fname_id_map::hasher(), Ref_fname_id_map::key_equal(), alloc)
, m_messages(messages)
, m_api_decls(0, Decl_set::hasher(), Decl_set::key_equal(), alloc)
, m_next_unique_name_id(0u)
, m_noinline_mode(hlsl::IPrinter::ATTR_NOINLINE_WRAP)
, m_use_dbg(enable_debug)
, m_opt_remarks(enable_opt_remarks)
{
}

// Return the name for this pass.
llvm::StringRef HLSLWriterBasePass::getPassName() const
{
    return "HLSL writer";
}

// Create a new compilation unit with the given name.
hlsl::Compilation_unit *HLSLWriterBasePass::create_compilation_unit(
    hlsl::ICompiler *compiler,
    char const *name)
{
    hlsl::Compilation_unit *unit    =
        impl_cast<hlsl::Compilation_unit>(compiler->create_unit(name));
    hlsl::Definition_table &def_tab = unit->get_definition_table();

    def_tab.transition_to_scope(def_tab.get_predef_scope());

    // create all HLSL predefined entities first
    fillPredefinedEntities(unit);

    return unit;
}

// Generate HLSL predefined entities into the definition table.
void HLSLWriterBasePass::fillPredefinedEntities(
    hlsl::Compilation_unit *unit)
{
    // This is a work-around function, so far it adds only the float3 and float4 constructors
    // and the intrinsics.
    hlsl::Definition_table &def_tab = unit->get_definition_table();
    hlsl::Type_factory     &tf      = unit->get_type_factory();
    hlsl::Symbol_table     &st      = unit->get_symbol_table();

    MDL_ASSERT(def_tab.get_curr_scope() == def_tab.get_predef_scope());

    hlsl::Type_scalar *float_type = tf.get_float();

    hlsl::Type_vector *float3_type = tf.get_vector(float_type, 3);

    {
        hlsl::Type_function::Parameter p(float_type, hlsl::Type_function::Parameter::PM_IN);
        hlsl::Type_function::Parameter params[] = { p, p, p };
        hlsl::Type_function *func_type = tf.get_function(float3_type, params);

        def_tab.enter_function_definition(
            st.get_symbol("float3"),
            func_type,
            hlsl::Def_function::DS_ELEM_CONSTRUCTOR,
            NULL);
    }

    {
        hlsl::Type_vector *float4_type = tf.get_vector(float_type, 4);

        hlsl::Type_function::Parameter p(float_type, hlsl::Type_function::Parameter::PM_IN);
        hlsl::Type_function::Parameter params[] = { p, p, p, p };
        hlsl::Type_function *func_type = tf.get_function(float4_type, params);

        def_tab.enter_function_definition(
            st.get_symbol("float4"),
            func_type,
            hlsl::Def_function::DS_ELEM_CONSTRUCTOR,
            NULL);
    }
}

// Find the API debug type info for a given struct type.
sl::DebugTypeHelper::API_type_info const *HLSLWriterBasePass::find_api_type_info(
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
hlsl::Type *HLSLWriterBasePass::inner_element_type(
    hlsl::Type *type)
{
    while (hlsl::Type_array *a_type = as<hlsl::Type_array>(type)) {
        type = a_type->get_element_type();
    }
    return type;
}

// Get an HLSL symbol for an LLVM string.
hlsl::Symbol *HLSLWriterBasePass::get_sym(llvm::StringRef const &str)
{
    return m_symbol_table.get_symbol(str.str().c_str());
}

// Get an HLSL symbol for a string.
hlsl::Symbol *HLSLWriterBasePass::get_sym(char const *str)
{
    return m_symbol_table.get_symbol(str);
}

// Get an unique HLSL symbol for an LLVM string and a template.
hlsl::Symbol *HLSLWriterBasePass::get_unique_sym(
    llvm::StringRef const &str,
    char const *templ)
{
    return get_unique_sym(str.str().c_str(), templ);
}

// Get an unique HLSL symbol for an LLVM string and a template.
hlsl::Symbol *HLSLWriterBasePass::get_unique_sym(
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
    hlsl::Symbol *sym = nullptr;

    char buffer[65];
    while (true) {
        if (name != nullptr) {
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

// Get an HLSL name for the given location and symbol.
hlsl::Name *HLSLWriterBasePass::get_name(Location loc, Symbol *sym)
{
    return m_decl_factory.create_name(loc, sym);
}

// Get an HLSL name for the given location and a C-string.
hlsl::Name *HLSLWriterBasePass::get_name(Location loc, const char *str)
{
    hlsl::Symbol *sym = m_symbol_table.get_symbol(str);
    return get_name(loc, sym);
}

// Get an HLSL name for the given location and LLVM string reference.
hlsl::Name *HLSLWriterBasePass::get_name(Location loc, llvm::StringRef const &str)
{
    return get_name(loc, str.str().c_str());
}

// Get an HLSL type name for an LLVM type.
hlsl::Type_name *HLSLWriterBasePass::get_type_name(
    llvm::Type *type)
{
    hlsl::Type *hlsl_type = convert_type(type);
    return get_type_name(hlsl_type);
}

// Get an HLSL type name for an HLSL type.
hlsl::Type_name *HLSLWriterBasePass::get_type_name(
    hlsl::Type *type)
{
    hlsl::Type::Modifiers mod = type->get_type_modifiers();

    // in HLSL type names have no array specifiers, so check that type is NOT an array
    MDL_ASSERT(!is<hlsl::Type_array>(type) && "Cannot create type name for array types");

    hlsl::Name *name = m_decl_factory.create_name(zero_loc, type->get_sym());
    hlsl::Type_name *type_name = m_decl_factory.create_type_name(zero_loc);
    type_name->set_name(name);
    type_name->set_type(type);

    // handle type modifier
    if (mod & hlsl::Type::MK_CONST) {
        type_name->get_qualifier().set_type_modifier(TM_CONST);
    }
    if (mod & hlsl::Type::MK_COL_MAJOR) {
        type_name->get_qualifier().set_type_modifier(TM_COLUMN_MAJOR);
    }
    if (mod & hlsl::Type::MK_ROW_MAJOR) {
        type_name->get_qualifier().set_type_modifier(TM_ROW_MAJOR);
    }

    return type_name;
}

// Get an HLSL type name for an HLSL name.
hlsl::Type_name *HLSLWriterBasePass::get_type_name(
    hlsl::Symbol *sym)
{
    hlsl::Type_name *type_name = m_decl_factory.create_type_name(zero_loc);
    hlsl::Name *name = get_name(zero_loc, sym);
    type_name->set_name(name);

    return type_name;
}

// Add array specifier to an init declarator if necessary.
template<typename Decl_type>
void HLSLWriterBasePass::add_array_specifiers(
    Decl_type  *decl,
    hlsl::Type *type)
{
    while (hlsl::Type_array *a_type = hlsl::as<hlsl::Type_array>(type)) {
        type = a_type->get_element_type();

        hlsl::Expr *size = nullptr;
        if (!a_type->is_unsized()) {
            size = this->m_expr_factory.create_literal(
                zero_loc,
                this->m_value_factory.get_int32(int(a_type->get_size())));
        }
        hlsl::Array_specifier *as = this->m_decl_factory.create_array_specifier(zero_loc, size);
        decl->add_array_specifier(as);
    }
}

// Add parameter qualifier from a function type parameter at index.
void HLSLWriterBasePass::add_param_qualifier(
    hlsl::Type_name     *param_type_name,
    hlsl::Type_function *func_type,
    size_t              index)
{
    hlsl::Type_function::Parameter *param = func_type->get_parameter(index);

    hlsl::Parameter_qualifier param_qualifier = convert_type_modifier_to_param_qualifier(param);
    param_type_name->get_qualifier().set_parameter_qualifier(param_qualifier);
}

// Add a field to a struct declaration.
hlsl::Type_struct::Field HLSLWriterBasePass::add_struct_field(
    hlsl::Declaration_struct *decl_struct,
    hlsl::Type               *type,
    hlsl::Symbol             *sym)
{
    hlsl::Declaration_field *decl_field =
        m_decl_factory.create_field_declaration(get_type_name(inner_element_type(type)));
    hlsl::Field_declarator *field_declarator = m_decl_factory.create_field(zero_loc);
    field_declarator->set_name(get_name(zero_loc, sym));
    add_array_specifiers(field_declarator, type);

    decl_field->add_field(field_declarator);
    decl_struct->add(decl_field);

    return hlsl::Type_struct::Field(type, sym);
}

// Add a field to a struct declaration.
hlsl::Type_struct::Field HLSLWriterBasePass::add_struct_field(
    hlsl::Declaration_struct *decl_struct,
    hlsl::Type               *type,
    char const               *name)
{
    hlsl::Symbol *sym = m_symbol_table.get_symbol(name);
    return add_struct_field(decl_struct, type, sym);
}

// Create the HLSL resource data struct for the corresponding LLVM struct type.
hlsl::Type_struct *HLSLWriterBasePass::create_res_data_struct_type(
    llvm::StructType *type)
{
    // The res_data struct type is opaque in the generated code, but we don't support
    // this in the HLSL compiler, so we create a dummy type, but do not add it to the
    // compilation unit.

    hlsl::Declaration_struct *decl_struct = m_decl_factory.create_struct(zero_loc);
    hlsl::Symbol             *struct_sym  = m_symbol_table.get_symbol("Res_data");

    decl_struct->set_name(get_name(zero_loc, struct_sym));

    hlsl::Type_struct::Field dummy_field[1] = {
        add_struct_field(decl_struct, m_tc.int_type, "dummy")
    };

    hlsl::Type_struct *res = m_tc.get_struct(dummy_field, struct_sym);

    // do not add to the unit to avoid printing it

    m_type_cache[type] = res;
    return res;
}

// Convert an LLVM struct type to an HLSL struct type.
hlsl::Type *HLSLWriterBasePass::convert_struct_type(
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
                m_type_cache[s_type] = m_tc.float3_type;
                return m_tc.float3_type;
            }
            if (name == "float4") {
                m_type_cache[s_type] = m_tc.float4_type;
                return m_tc.float4_type;
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
        hlsl::Type *field_type = convert_type(s_type->getElementType(i));
        if (hlsl::is<hlsl::Type_void>(field_type)) {
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
    // While HLSL supports implicit conversion of different struct types with matching types,
    // Slang does not
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

    hlsl::Declaration_struct *decl_struct = m_decl_factory.create_struct(zero_loc);
    decl_struct->set_name(get_name(zero_loc, struct_sym));

    Small_VLA<hlsl::Type_struct::Field, 8> fields(m_alloc, n_fields);

    static char const * const deriv_names[] = { "val", "dx", "dy" };

    unsigned n = 0;
    for (unsigned i = 0; i < n_fields; ++i) {
        hlsl::Type *field_type = convert_type(s_type->getElementType(i));

        if (api_info == nullptr && hlsl::is<hlsl::Type_void>(field_type)) {
            // do NOT skip void fields for API types: they might be accessed during
            // field index which would break then
            continue;
        }

        hlsl::Symbol *sym = nullptr;

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
                    if (sym->get_id() < hlsl::Symbol::SYM_USER) {
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

    hlsl::Type_struct *res = m_tc.get_struct(
        Array_ref<hlsl::Type_struct::Field>(fields.data(), n), struct_sym);

    // create the definition for the struct type and its fields
    {
        hlsl::Definition_table::Scope_transition scope_trans(
            m_def_tab, m_def_tab.get_global_scope());

        Def_type *type_def = m_def_tab.enter_type_definition(
            struct_sym,
            res,
            &zero_loc);
        decl_struct->set_definition(type_def);

        hlsl::Definition_table::Scope_enter scope(m_def_tab, res, type_def);

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

// Convert an LLVM type to an HLSL type.
hlsl::Type *HLSLWriterBasePass::convert_type(
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
        return m_tc.double_type;
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
                        hlsl::Type_scalar *res_elem_type =
                            type_id == llvm::Type::FloatTyID ?
                            (hlsl::Type_scalar *)m_tc.float_type :
                            (hlsl::Type_scalar *)m_tc.double_type;
                        hlsl::Type_vector *res_vt_type =
                            m_tc.get_vector(res_elem_type, rows);
                        return m_tc.get_matrix(res_vt_type, cols);
                    }
                }
            }

            hlsl::Type *res_elem_type = convert_type(array_elem_type);
            return m_tc.get_array(res_elem_type, n_elem);
        }

    case llvm::Type::FixedVectorTyID:
        {
            llvm::FixedVectorType *vector_type = cast<llvm::FixedVectorType>(type);
            hlsl::Type            *elem_type   = convert_type(vector_type->getElementType());
            if (hlsl::Type_scalar *scalar_type = hlsl::as<hlsl::Type_scalar>(elem_type)) {
                hlsl::Type *res = m_tc.get_vector(
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
        return m_tc.double_type;

    case llvm::Type::PointerTyID:
        if (llvm::ArrayType *array_type =
            llvm::dyn_cast<llvm::ArrayType>(type->getPointerElementType()))
        {
            uint64_t size = array_type->getNumElements();
            if (size == 0) {
                // map zero length array to void, we cannot handle that in HLSL
                return m_tc.void_type;
            }

            hlsl::Type *base_type = convert_type(array_type->getElementType());
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

// Create the HLSL definition for a user defined LLVM function.
hlsl::Def_function *HLSLWriterBasePass::create_definition(
    llvm::Function *func)
{
    llvm::FunctionType *llvm_func_type = func->getFunctionType();
    hlsl::Type         *ret_type = convert_type(llvm_func_type->getReturnType());
    hlsl::Type         *out_type = nullptr;

    llvm::SmallVector<hlsl::Type_function::Parameter, 8> params;

    if (hlsl::is<hlsl::Type_array>(ret_type)) {
        // HLSL does not support returning arrays, turn into an out parameter
        out_type = ret_type;
        ret_type = m_tc.void_type;

        params.push_back(hlsl::Type_function::Parameter(
            out_type,
            hlsl::Type_function::Parameter::PM_OUT));
    }

    // collect parameters for the function definition
    for (llvm::Argument &arg_it : func->args()) {
        llvm::Type *arg_llvm_type = arg_it.getType();
        hlsl::Type *param_type    = convert_type(arg_llvm_type);

        // skip void typed parameters
        if (hlsl::is<hlsl::Type_void>(param_type)) {
            continue;
        }

        hlsl::Type_function::Parameter::Modifier param_mod = hlsl::Type_function::Parameter::PM_IN;

        if (llvm::isa<llvm::PointerType>(arg_llvm_type)) {
            if (arg_it.hasStructRetAttr()) {
                // the sret attribute marks "return" values, so OUT is enough
                param_mod = hlsl::Type_function::Parameter::PM_OUT;
            } else if (arg_it.onlyReadsMemory()) {
                // can be safely passed as an IN attribute IF noalias
                param_mod = hlsl::Type_function::Parameter::PM_IN;
            } else {
                // can be safely passed as INOUT IF noalias
                param_mod = hlsl::Type_function::Parameter::PM_INOUT;
            }
        }

        params.push_back(hlsl::Type_function::Parameter(param_type, param_mod));
    }

    // create the function definition
    hlsl::Symbol        *func_sym = get_unique_sym(func->getName(), "func");
    hlsl::Type_function *func_type = m_tc.get_function(
        ret_type, Array_ref<hlsl::Type_function::Parameter>(params.data(), params.size()));
    hlsl::Def_function  *func_def = m_def_tab.enter_function_definition(
        func_sym, func_type, hlsl::Def_function::DS_UNKNOWN, &zero_loc);

    return func_def;
}

// Check if a given LLVM array type is the representation of the AST matrix type.
bool HLSLWriterBasePass::is_matrix_type(
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
char const *HLSLWriterBasePass::get_vector_index_str(
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
hlsl::Expr_ref *HLSLWriterBasePass::create_reference(
    hlsl::Type_name *type_name,
    hlsl::Type *type)
{
    hlsl::Expr_ref *ref = m_expr_factory.create_reference(type_name);
    ref->set_type(type);
    return ref;
}

// Create a reference to an entity of the given symbol and HLSL type.
hlsl::Expr_ref *HLSLWriterBasePass::create_reference(
    hlsl::Symbol *sym,
    hlsl::Type *type)
{
    return create_reference(get_type_name(sym), type);
}

// Create a reference to a variable of the given type.
hlsl::Expr_ref *HLSLWriterBasePass::create_reference(
    hlsl::Symbol *sym,
    llvm::Type   *type)
{
    return create_reference(sym, convert_type(type));
}

// Create a reference to an entity.
hlsl::Expr_ref *HLSLWriterBasePass::create_reference(
    hlsl::Definition *def)
{
    hlsl::Expr_ref *ref = create_reference(def->get_symbol(), def->get_type());
    ref->set_definition(def);
    return ref;
}

// Create a new unary expression.
hlsl::Expr *HLSLWriterBasePass::create_unary(
    hlsl::Location const       &loc,
    hlsl::Expr_unary::Operator op,
    hlsl::Expr                *arg)
{
    return m_expr_factory.create_unary(loc, op, arg);
}

// Create a new binary expression.
hlsl::Expr *HLSLWriterBasePass::create_binary(
    hlsl::Expr_binary::Operator op,
    hlsl::Expr                  *left,
    hlsl::Expr                  *right)
{
    if (op == hlsl::Expr_binary::OK_BITWISE_XOR) {
        if (is<Type_bool>(left->get_type()->skip_type_alias()) &&
            is<Type_bool>(right->get_type()->skip_type_alias()))
        {
            // map XOR on boolean values to NOT-EQUAL to be compatible to SLANG
            op = hlsl::Expr_binary::OK_NOT_EQUAL;
        }
    }
    return m_expr_factory.create_binary(op, left, right);
}

// Create a call to a HLSL runtime function.
hlsl::Expr *HLSLWriterBasePass::create_runtime_call(
    hlsl::Location const          &loc,
    llvm::Function                *func,
    Array_ref<hlsl::Expr *> const &args)
{
    // call to an unknown entity
    llvm::StringRef name = func->getName();

    bool is_llvm_intrinsic = name.startswith("llvm.");
    if (is_llvm_intrinsic || name.startswith("hlsl.")) {
        // handle HLSL or LLVM intrinsics
        name = name.drop_front(5);
        size_t pos = name.find('.');
        name = name.slice(0, pos);

        if (is_llvm_intrinsic) {
            // need some mapping between LLVM intrinsics and HLSL
            if (name == "fabs") {
                name = "abs";
            } else if (name == "minnum") {
                name = "min";
            } else if (name == "maxnum") {
                name = "max";
            }
        }
    }
    hlsl::Symbol *func_sym = get_sym(name);
    hlsl::Type   *ret_type = convert_type(func->getReturnType());

    // FIXME: wrong type
    hlsl::Expr_ref *callee = create_reference(func_sym, ret_type);
    hlsl::Expr     *call   = m_expr_factory.create_call(callee, args);

    call->set_type(ret_type);
    call->set_location(loc);

    return call;
}

// Creates a bitcast expression from float to int.
hlsl::Expr *HLSLWriterBasePass::create_float2int_bitcast(
    hlsl::Type *dst,
    hlsl::Expr *arg)
{
    // FIXME: the type of the reference is wrong
    hlsl::Expr *callee = create_reference(get_sym("asint"), dst);
    hlsl::Expr *call   = m_expr_factory.create_call(callee, arg);
    call->set_type(dst);
    return call;
}

// Creates a bitcast expression from int to float.
hlsl::Expr *HLSLWriterBasePass::create_int2float_bitcast(
    hlsl::Type *dst,
    hlsl::Expr *arg)
{
    // FIXME: the type of the reference is wrong
    hlsl::Expr *callee = create_reference(get_sym("asfloat"), dst);
    hlsl::Expr *call   = m_expr_factory.create_call(callee, arg);
    call->set_type(dst);
    return call;
}

// Create a type cast expression.
hlsl::Expr *HLSLWriterBasePass::create_type_cast(
    hlsl::Type *dst,
    hlsl::Expr *arg)
{
    if (hlsl::Expr_literal *lit = hlsl::as<hlsl::Expr_literal>(arg)) {
        // try to convert the literal first
        hlsl::Value *v = lit->get_value();
        hlsl::Value *n = v->convert(m_value_factory, dst);

        if (!hlsl::is<hlsl::Value_bad>(n)) {
            return m_expr_factory.create_literal(arg->get_location(), n);
        }
    }

    hlsl::Expr *callee = nullptr;

    // FIXME: lookup_constructor is heavily broken on HLSL, do not depend on it
    hlsl::Def_function *def      = lookup_constructor(dst, arg);
    hlsl::Type         *ret_type = dst;
    if (def != nullptr) {
        MDL_ASSERT(dst->skip_type_alias() == def->get_type()->get_return_type()->skip_type_alias());
        ret_type = def->get_type()->get_return_type();
        callee = create_reference(def);
    } else {
        // just create a type reference
        callee = create_reference(dst->get_sym(), dst);
    }

    hlsl::Expr *call = nullptr;
    if (!hlsl::is<hlsl::Type_scalar>(dst)) {
        // use C-style cast if possible
        call = m_expr_factory.create_typecast(callee, arg);
    } else {
        call = m_expr_factory.create_call(callee, arg);
    }

    call->set_type(ret_type);
    return call;
}

// Get the constructor for the given HLSL type.
hlsl::Def_function *HLSLWriterBasePass::lookup_constructor(
    hlsl::Type                    *type,
    Array_ref<hlsl::Expr *> const &args) const
{
    // FIXME: this implementation is wrong, it works only for the fake fillPredefinedEntities()
    if (hlsl::Def_function *def = hlsl::as_or_null<hlsl::Def_function>(
        m_def_tab.get_predef_scope()->find_definition_in_scope(type->get_sym())))
    {
        if (def->get_semantics() == hlsl::Def_function::DS_ELEM_CONSTRUCTOR) {
            return def;
        }
    }
    return NULL;
}

// Set the type qualifier for a global constant in HLSL.
void HLSLWriterBasePass::set_global_constant_qualifier(
    hlsl::Type_qualifier &tq)
{
    tq.set_storage_qualifier(hlsl::SQ_STATIC);
}

// Convert a function type parameter qualifier to a AST parameter qualifier.
hlsl::Parameter_qualifier HLSLWriterBasePass::convert_type_modifier_to_param_qualifier(
    hlsl::Type_function::Parameter *param)
{
    hlsl::Parameter_qualifier param_qualifier = hlsl::PQ_NONE;
    switch (param->get_modifier()) {
    case Type_function::Parameter::PM_IN:
        param_qualifier = hlsl::PQ_IN;
        break;
    case Type_function::Parameter::PM_OUT:
        param_qualifier = hlsl::PQ_OUT;
        break;
    case Type_function::Parameter::PM_INOUT:
        param_qualifier = hlsl::PQ_INOUT;
        break;
    }
    return param_qualifier;
}

// Creates an initializer.
hlsl::Expr *HLSLWriterBasePass::create_initializer(
    hlsl::Location const          &loc,
    hlsl::Type                    *type,
    Array_ref<hlsl::Expr *> const &args)
{
    hlsl::Expr *res = m_expr_factory.create_compound(loc, args);
    res->set_type(type);
    return res;
}

// Set the out parameter qualifier.
void HLSLWriterBasePass::make_out_parameter(
    hlsl::Type_name *param_type_name)
{
    param_type_name->get_qualifier().set_parameter_qualifier(hlsl::PQ_OUT);
}

// Convert the LLVM debug location (if any is attached to the given instruction)
// to an HLSL location.
hlsl::Location HLSLWriterBasePass::convert_location(
    llvm::Instruction *I)
{
    if (m_use_dbg) {
        if (llvm::MDNode *md_node = I->getMetadata(llvm::LLVMContext::MD_dbg)) {
            if (llvm::isa<llvm::DILocation>(md_node)) {
                llvm::DebugLoc Loc(md_node);
                unsigned        Line = Loc->getLine();
                unsigned        Column = Loc->getColumn();
                llvm::StringRef fname = Loc->getFilename();

                string s(fname.data(), fname.size(), m_alloc);
                Ref_fname_id_map::const_iterator it = m_ref_fnames.find(s);
                unsigned file_id;
                if (it == m_ref_fnames.end()) {
                    file_id = m_unit->register_filename(s.c_str());
                    m_ref_fnames.insert(Ref_fname_id_map::value_type(s, file_id));
                } else {
                    file_id = it->second;
                }

                return hlsl::Location(Line, Column, file_id);
            }
        }
    }
    return zero_loc;
}

// Called for every function that is just a prototype in the original LLVM module.
hlsl::Def_function *HLSLWriterBasePass::create_prototype(
    llvm::Function &func)
{
    hlsl::Def_function  *func_def  = create_definition(&func);
    hlsl::Type_function *func_type = func_def->get_type();
    hlsl::Type *ret_type = func_type->get_return_type();
    hlsl::Type *out_type = nullptr;

    // reset the name IDs
    m_next_unique_name_id = 0;

    if (is<hlsl::Type_void>(ret_type) &&
        !func.getFunctionType()->getReturnType()->isVoidTy())
    {
        // return type was converted into out parameter
        typename hlsl::Type_function::Parameter *param = func_type->get_parameter(0);
        if (param->get_modifier() == hlsl::Type_function::Parameter::PM_OUT) {
            out_type = param->get_type();
        }
    }

    // create the declaration for the function
    Type_name *ret_type_name = get_type_name(ret_type);

    hlsl::Symbol         *func_sym  = func_def->get_symbol();
    hlsl::Name           *func_name = get_name(zero_loc, func_sym);
    Declaration_function *decl_func = m_decl_factory.create_function(
        ret_type_name, func_name);

    // create the function body
    {
        hlsl::Definition_table::Scope_enter enter(m_def_tab, func_def);

        // now create the declarations
        unsigned first_param_ofs = 0;
        if (out_type != nullptr) {
            hlsl::Type_name *param_type_name = get_type_name(out_type);
            hlsl::Declaration_param *decl_param = m_decl_factory.create_param(param_type_name);
            add_array_specifiers(decl_param, out_type);
            make_out_parameter(param_type_name);


            hlsl::Symbol *param_sym = get_unique_sym("p_result", "p_result");
            hlsl::Name *param_name = get_name(zero_loc, param_sym);
            decl_param->set_name(param_name);

            hlsl::Def_param *m_out_def = m_def_tab.enter_parameter_definition(
                param_sym, out_type, &param_name->get_location());
            m_out_def->set_declaration(decl_param);
            param_name->set_definition(m_out_def);

            decl_func->add_param(decl_param);

            ++first_param_ofs;
        }

        for (llvm::Argument &arg_it : func.args()) {
            llvm::Type *arg_llvm_type = arg_it.getType();
            hlsl::Type *param_type = convert_type(arg_llvm_type);

            if (is<hlsl::Type_void>(param_type)) {
                // skip void typed parameters
                continue;
            }

            unsigned        i = arg_it.getArgNo();
            hlsl::Type_name *param_type_name = get_type_name(param_type);

            hlsl::Declaration_param *decl_param = m_decl_factory.create_param(param_type_name);
            add_array_specifiers(decl_param, param_type);

            hlsl::Type_function::Parameter *param = func_type->get_parameter(i + first_param_ofs);

            hlsl::Parameter_qualifier param_qualifier =
                convert_type_modifier_to_param_qualifier(param);
            param_type_name->get_qualifier().set_parameter_qualifier(param_qualifier);

            char templ[16];
            snprintf(templ, sizeof(templ), "p_%u", i);

            hlsl::Symbol *param_sym = get_unique_sym(arg_it.getName(), templ);
            hlsl::Name *param_name = get_name(zero_loc, param_sym);
            decl_param->set_name(param_name);

            hlsl::Def_param *param_def = m_def_tab.enter_parameter_definition(
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
bool  HLSLWriterBasePass::must_be_materialized(
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

/// Count number of functions inside a compilation unit.
static size_t get_function_count(
    hlsl::Compilation_unit const *unit)
{
    size_t cnt = 0;

    for (hlsl::Compilation_unit::const_iterator it(unit->decl_begin()), end(unit->decl_end());
        it != end;
        ++it)
    {
        hlsl::Declaration const *decl = it;

        if (hlsl::Declaration_function const *fdecl = as<hlsl::Declaration_function>(decl)) {
            if (fdecl->is_prototype()) {
                // ignore prototypes
                continue;
            }
            ++cnt;
        }
    }
    return cnt;
}

/// Count number of expressions (ignored references).
static size_t get_expr_count(
    hlsl::Compilation_unit const *unit)
{
    class Expr_visitor : public hlsl::CUnit_visitor {
    public:
        /// Constructor.
        Expr_visitor()
        : m_expr_cnt(0u)
        {}

        size_t get_count() const { return m_expr_cnt; }

        void post_visit(Expr *expr) HLSL_FINAL {
            ++m_expr_cnt;
        }

        void post_visit(Expr_ref *expr) HLSL_FINAL {
            // do not count references
        }

    private:
        size_t m_expr_cnt;
    };

    Expr_visitor visitor;

    visitor.visit(const_cast<hlsl::Compilation_unit *>(unit));
    return visitor.get_count();
}

/// Check if this declaration needs an extra newline when dumped.
static int last_decl_kind(
    hlsl::Declaration const *decl)
{
    int kind = decl->get_kind();
    if (kind == hlsl::Declaration::DK_FUNCTION) {
        hlsl::Declaration_function const *f_decl = cast<hlsl::Declaration_function>(decl);
        if (!f_decl->is_prototype()) {
            kind = -2;
        }
    } else if (kind == hlsl::Declaration::DK_VARIABLE) {
        hlsl::Declaration_variable const *vdecl = cast<hlsl::Declaration_variable>(decl);
        if (vdecl->empty()) {
            kind = -2;
        }
    }
    return kind;
}

// Finalize the compilation unit and write it to the given output stream.
void HLSLWriterBasePass::finalize(
    llvm::Module                                               &M,
    Generated_code_source                                      *code,
    list<std::pair<char const *, hlsl::Symbol *> >::Type const &remaps)
{
    String_stream_writer            out(code->access_src_code());
    mi::base::Handle<hlsl::Printer> printer(m_compiler->create_printer(&out));

    // analyze and optimize it
    m_unit->analyze(*m_compiler.get());

    printer->enable_locations(m_use_dbg);

    // use defined to wrap [noinline]
    printer->set_attr_noinline_mode(m_noinline_mode);

    // helper data
    typedef ptr_hash_map<hlsl::Symbol, char const *>::Type Mapped_type;
    Mapped_type mapped(m_alloc);
    if (!remaps.empty()) {
        typedef list<std::pair<char const *, Symbol *> >::Type Symbol_list;

        for (Symbol_list::const_iterator it(remaps.begin()), end(remaps.end()); it != end; ++it) {
            mapped[it->second] = it->first;
        }
    }

    // generate the version fragment: EMPTY yet

    // generate the defines fragment
    if (m_noinline_mode == hlsl::IPrinter::ATTR_NOINLINE_WRAP || !remaps.empty()) {
        printer->print_comment("defines");

        if (m_noinline_mode == hlsl::IPrinter::ATTR_NOINLINE_WRAP) {
            printer->print("#if COMPUTE_SHADER_BUILD\n");
            printer->print("#define ATTR_NOINLINE\n");
            printer->print("#else\n");
            printer->print("#define ATTR_NOINLINE [noinline]\n");
            printer->print("#endif\n");
            printer->nl();
        }

        if (!remaps.empty()) {
            typedef list<std::pair<char const *, Symbol *> >::Type Symbol_list;

            for (Symbol_list::const_iterator it(remaps.begin()), end(remaps.end());
                it != end;
                ++it)
            {
                printer->print("#define MAPPED_");
                printer->print(it->first);
                printer->print(" 1");
                printer->nl();
            }
            printer->nl();
        }
    }

    // generate the API type fragment
    if (false) {
        bool first = true;
        int last_kind = -1;
        for (hlsl::Compilation_unit::const_iterator
            it(m_unit->decl_begin()), end(m_unit->decl_end());
            it != end;
            ++it)
        {
            hlsl::Declaration const *decl = it;

            if (m_api_decls.find(decl) != m_api_decls.end()) {
                if (first) {
                    printer->print_comment("API types");
                    first = false;
                }

                hlsl::Declaration::Kind kind = decl->get_kind();
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
        bool first = true;
        for (hlsl::Compilation_unit::const_iterator
            it(m_unit->decl_begin()), end(m_unit->decl_end());
            it != end;
            ++it)
        {
            hlsl::Declaration const *decl = it;

            if (!is<hlsl::Declaration_struct>(decl)) {
                continue;
            }

            // ignore API type declarations
            if (m_api_decls.find(decl) != m_api_decls.end()) {
                // already dumped in state fragment
                continue;
            }

            if (first) {
                printer->print_comment("user defined structs");
                first = false;
            }

            printer->print(decl);
            printer->nl();
        }
    }

    // generate the globals fragment
    {
        bool first = true;
        int last_kind = -1;
        for (hlsl::Compilation_unit::const_iterator
            it(m_unit->decl_begin()), end(m_unit->decl_end());
            it != end;
            ++it)
        {
            hlsl::Declaration const *decl = it;

            // ignore structs and functions
            if (is<hlsl::Declaration_struct>(decl) || is<hlsl::Declaration_function>(decl)) {
                continue;
            }

            // ignore API declarations
            if (m_api_decls.find(decl) != m_api_decls.end()) {
                // already dumped in API fragment
                continue;
            }

            if (first) {
                printer->print_comment("globals");
                first = false;
            }

            hlsl::Declaration::Kind kind = decl->get_kind();
            if (last_kind != -1 && last_kind != kind) {
                printer->nl();
            }

            printer->print(decl);
            printer->nl();

            last_kind = last_decl_kind(decl);
        }
    }

    // generate the functions prototype fragment
    if (false) {
        bool first = true;
        for (hlsl::Compilation_unit::const_iterator
            it(m_unit->decl_begin()), end(m_unit->decl_end());
            it != end;
            ++it)
        {
            hlsl::Declaration const *decl = it;

            if (!is<hlsl::Declaration_function>(decl)) {
                continue;
            }

            hlsl::Declaration_function const *f_decl = cast<hlsl::Declaration_function>(decl);

            if (!f_decl->is_prototype()) {
                continue;
            }

            if (first) {
                printer->print_comment("prototypes");
                first = false;
            }

            hlsl::Symbol *f_sym = f_decl->get_definition()->get_symbol();
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
        bool first = true;
        int last_kind = -1;
        for (hlsl::Compilation_unit::const_iterator
            it(m_unit->decl_begin()), end(m_unit->decl_end());
            it != end;
            ++it)
        {
            hlsl::Declaration const *decl = it;

            if (hlsl::Declaration_function const *f_decl = as<hlsl::Declaration_function>(decl)) {
                if (f_decl->is_prototype()) {
                    // ignore prototypes
                    continue;
                }
            } else {
                // ignore non-functions
                continue;
            }

            if (first) {
                printer->print_comment("functions");
                first = false;
            }

            hlsl::Declaration::Kind kind = decl->get_kind();
            if (last_kind != -1 && last_kind != kind) {
                printer->nl();
            }

            printer->print(decl);
            printer->nl();

            last_kind = last_decl_kind(decl);
        }
    }

    if (m_opt_remarks) {
        // just use the first function
        llvm::Function const &F = *M.begin();

        llvm::OptimizationRemarkEmitter ORE(&F);

        ORE.emit([&]() {
            return llvm::OptimizationRemark(DEBUG_TYPE, "HLSL", &F)
                << "#HLSL Functions "
                << llvm::ore::NV("fcount", get_function_count(m_unit.get()));
        });

        ORE.emit([&]() {
            return llvm::OptimizationRemark(DEBUG_TYPE, "HLSL", &F)
                << "#HLSL Expressions "
                << llvm::ore::NV("ecount", get_expr_count(m_unit.get()));
        });
    }
}

// Generates a new global static const variable to hold an LLVM value.
hlsl::Definition *HLSLWriterBasePass::create_global_const(
    llvm::StringRef name, hlsl::Expr *c_expr)
{
    hlsl::Definition_table::Scope_transition scope(
        m_def_tab, m_def_tab.get_global_scope());

    hlsl::Symbol *cnst_sym  = get_unique_sym(name, "glob_cnst");
    hlsl::Type   *cnst_type = c_expr->get_type();

    cnst_type = m_tc.get_alias(cnst_type, Type::MK_CONST);

    // Note: HLSL does not support array specifiers
    hlsl::Type_name *cnst_type_name = get_type_name(inner_element_type(cnst_type));
    hlsl::Type_qualifier &tq = cnst_type_name->get_qualifier();

    set_global_constant_qualifier(tq);

    hlsl::Declaration_variable *decl_cnst = m_decl_factory.create_variable(cnst_type_name);

    hlsl::Init_declarator *init_decl = m_decl_factory.create_init_declarator(zero_loc);
    hlsl::Name *var_name = get_name(zero_loc, cnst_sym);
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

// Add a JIT backend error message to the messages.
void HLSLWriterBasePass::error(int code)
{
    // FIXME: get from JIT be
    char MESSAGE_CLASS = 'J';

    mi::mdl::Error_params params(m_alloc);

    string msg(m_messages.format_msg(code, MESSAGE_CLASS, params));
    m_messages.add_warning_message(code, MESSAGE_CLASS, 0, NULL, msg.c_str());
}

// Dump the current AST.
void HLSLWriterBasePass::dump_ast()
{
    // no dump_ast for HLSL yet
}

}  // hlsl
}  // mdl
}  // mi
