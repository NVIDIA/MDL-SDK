/******************************************************************************
 * Copyright (c) 2012-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <cstdarg>
#include <cctype>
#include <cstdio>
#include <cfloat>
#include <cstring>

#include <algorithm>
#include <list>
#include <utility>
#include <base/system/main/types.h>

#include <mi/mdl/mdl_translator_plugin.h>

#include "compilercore_allocator.h"
#include "compilercore_malloc_allocator.h"
#include "compilercore_analysis.h"
#include "compilercore_modules.h"
#include "compilercore_printers.h"
#include "compilercore_streams.h"
#include "compilercore_def_table.h"
#include "compilercore_errors.h"
#include "compilercore_builder.h"
#include "compilercore_mdl.h"
#include "compilercore_tools.h"
#include "compilercore_bitset.h"
#include "compilercore_assert.h"
#include "compilercore_positions.h"
#include "compilercore_file_resolution.h"
#include "compilercore_file_utils.h"
#include "compilercore_wchar_support.h"

namespace mi {
namespace mdl {

typedef Store<IType const *>             IType_store;
typedef Store<Definition *>              Definition_store;
typedef Store<bool>                      Flag_store;
typedef Store<IExpression_call const *>  Expr_call_store;
typedef Store<Messages_impl *>           Msgs_store;

#define POS(pos) \
    (pos).get_start_line(), (pos).get_start_column(), \
    (pos).get_end_line(), (pos).get_end_column()

/// For debugging. If true, dump_ast() prints positions.
static bool g_position_enabled = false;

/// Debug helper: Enables the dumping of positions.
///
/// \param enable  if true, following dump_ast() calls
///                will include positions
void dump_enable_positions(bool enable)
{
    g_position_enabled = enable;
}

static unsigned min(unsigned a, unsigned b) { return a < b ? a : b; }
static unsigned max(unsigned a, unsigned b) { return a > b ? a : b; }

namespace {
    /// A local helper class that creates a printer for the debug output stream.
    class Alloc {
    public:
        Alloc(IAllocator *a) : m_builder(a) {}

        Debug_Output_stream *dbg() {
            return m_builder.create<Debug_Output_stream>(m_builder.get_allocator());
        }

        Printer *prt(IOutput_stream *s) {
            return m_builder.create<Printer>(m_builder.get_allocator(), s);
        }

    private:
        Allocator_builder m_builder;
    };
}

// Returns true for error names.
bool is_error(IQualified_name const *name)
{
    for (int i = 0, n = name->get_component_count(); i < n; ++i) {
        if (is_error(name->get_component(i)))
            return true;
    }
    return false;
}

// Debug helper: Dump the AST of a compilation unit.
void dump_ast(IModule const *module)
{
    if (module == NULL)
        return;

    Module const *mod = static_cast<Module const *>(module);
    Alloc alloc(mod->get_allocator());

    mi::base::Handle<Debug_Output_stream> dbg(alloc.dbg());
    mi::base::Handle<Printer>             printer(alloc.prt(dbg.get()));

    printer->show_positions(g_position_enabled);
    printer->print(module);
}

// Debug helper: Dump the AST of a declaration.
void dump_ast(IDeclaration const *decl)
{
    if (decl == NULL)
        return;

    mi::base::Handle<IAllocator> mallocator(MallocAllocator::create_instance());
    Alloc alloc(mallocator.get());

    mi::base::Handle<Debug_Output_stream> dbg(alloc.dbg());
    mi::base::Handle<Printer>             printer(alloc.prt(dbg.get()));

    printer->show_positions(g_position_enabled);
    printer->print(decl, /*is_toplevel=*/ true);
}

// Debug helper: Dump the definition table of a compilation unit.
void dump_def_tab(IModule const *module)
{
    if (module == NULL)
        return;

    Module const *mod = static_cast<Module const *>(module);
    Alloc alloc(mod->get_allocator());

    mi::base::Handle<Debug_Output_stream> dbg(alloc.dbg());
    mi::base::Handle<Printer>             printer(alloc.prt(dbg.get()));

    printer->show_positions(g_position_enabled);
    Definition_table const &def_tab = mod->get_definition_table();
    def_tab.dump(printer.get(), mod->get_name());
}

// Debug helper: Dump the AST of a compilation unit.
void dump_expr(IExpression const *expr)
{
    if (! expr)
        return;

    mi::base::Handle<IAllocator> mallocator(MallocAllocator::create_instance());
    Alloc alloc(mallocator.get());

    mi::base::Handle<Debug_Output_stream> dbg(alloc.dbg());
    mi::base::Handle<Printer>             printer(alloc.prt(dbg.get()));

    printer->show_positions(g_position_enabled);
    printer->print(expr);
    printer->print("\n");
}

// Debug helper: Dump a definition.
void dump_def(IDefinition const *def)
{
    mi::base::Handle<IAllocator> mallocator(MallocAllocator::create_instance());
    Alloc alloc(mallocator.get());

    mi::base::Handle<Debug_Output_stream> dbg(alloc.dbg());
    mi::base::Handle<Printer>             printer(alloc.prt(dbg.get()));

    switch (def->get_kind()) {
    case IDefinition::DK_ERROR:
        printer->print("<ERROR");
        break;
    case IDefinition::DK_CONSTANT:
        printer->print(def->get_constant_value());
        break;
    case IDefinition::DK_ENUM_VALUE:
        printer->print(def->get_type());
        printer->print("(");
        printer->print(def->get_constant_value());
        printer->print(")");
        break;
    case IDefinition::DK_ANNOTATION:
        printer->print("annotation ");
        printer->print(def->get_type());
        break;
    case IDefinition::DK_TYPE:
        printer->print(def->get_type());
        break;
    case IDefinition::DK_FUNCTION:
    case IDefinition::DK_OPERATOR:
        {
            IType_function const *ftype = cast<IType_function>(def->get_type());
            printer->print(ftype->get_return_type());
            printer->print(" ");
            printer->print(def->get_symbol());
            printer->print("(");
            for (int i = 0, n = ftype->get_parameter_count(); i < n; ++i) {
                IType const   *ptype;
                ISymbol const *psym;
                ftype->get_parameter(i, ptype, psym);

                if (i > 0)
                    printer->print(", ");
                printer->print(ptype);
                printer->print(" ");
                printer->print(psym);
            }
            printer->print(")");
            break;
        }
    case IDefinition::DK_CONSTRUCTOR:
        {
            IType_function const *ftype = cast<IType_function>(def->get_type());
            printer->print(def->get_symbol());
            printer->print("(");
            for (int i = 0, n = ftype->get_parameter_count(); i < n; ++i) {
                IType const   *ptype;
                ISymbol const *psym;
                ftype->get_parameter(i, ptype, psym);

                if (i > 0)
                    printer->print(", ");
                printer->print(ptype);
                printer->print(" ");
                printer->print(psym);
            }
            printer->print(")");
            break;
        }
    case IDefinition::DK_VARIABLE:
    case IDefinition::DK_PARAMETER:
    case IDefinition::DK_MEMBER:
        printer->print(def->get_symbol());
        break;
    case IDefinition::DK_ARRAY_SIZE:
        printer->print("<");
        printer->print(def->get_symbol());
        printer->print(">");
        break;
    case IDefinition::DK_NAMESPACE:
        printer->print("namespace ");
        printer->print(def->get_symbol());
        break;
    }
}

/// Given a relative resource url, make it absolute ab appending its owner module package name.
///
/// \param alloc        an allocator
/// \param url          the relative url
/// \param module_name  the module name of the owner module
///
/// \return  the absolute url
///
/// \note because the resolver is not used here, the absolute url is not guaranteed to be correct,
///       especially in the weak relative case where two possibilities to resolve it exists
static string make_absolute_package(
    IAllocator *alloc,
    char const *url,
    char const *module_name)
{
    // Resolver failed. This is bad, because letting the name unchanged
    // might lead to "wrong" fixes later. One possible solution would be to
    // return an invalid resource here, but then the user will not see ANY
    // error.
    // Hence we do a "stupid" transformation here, i.e. compute an absolute
    // path that will point into PA. Note that the resource still will fail,
    // otherwise the resolver had found it there.
    string pa(module_name, alloc);
    size_t p = pa.rfind(':');
    if (p != string::npos)
        --p;
    pa = pa.substr(0, p);

    string abs_url(alloc);
    for (size_t i = 0, n = pa.size(); i < n; ++i) {
        if (pa[i] == ':') {
            abs_url.append('/');
            ++i;
        } else {
            abs_url.append(pa[i]);
        }
    }
    if (url[0] == '.' && url[1] == '/')
        url += 2;
    // strip "../" prefixes
    while (url[0] == '.' && url[1] == '.' && url[2] == '/') {
        url += 3;

        size_t pos = abs_url.rfind('/');
        if (pos != string::npos)
            abs_url = abs_url.substr(0, pos);
    }
    abs_url.append('/');
    abs_url.append(url);

    return abs_url;
}

// Retarget a (relative path) resource url from one package to be accessible from another
// package.
IValue_resource const *retarget_resource_url(
    IValue_resource const *r,
    Position const        &pos,
    IAllocator            *alloc,
    Type_factory          &tf,
    Value_factory         &vf,
    IModule const         *src,
    File_resolver         &resolver)
{
    // check if the URL is relative. If yes, update it
    char const *url = r->get_string_value();
    if (url != NULL && url[0] != '/') {
        string abs_url(alloc);

        mi::base::Handle<IMDL_resource_set> res(resolver.resolve_resource(
            pos,
            url,
            src->get_name(),
            src->get_filename()));

        if (res.is_valid_interface()) {
            abs_url = res->get_mdl_url_mask();
        } else {
            // Resolver failed. This is bad, because letting the name unchanged
            // might lead to "wrong" fixes later. One possible solution would be to
            // return an invalid resource here, but then the user will not see ANY
            // error.
            // Hence we do a "stupid" transformation here, i.e. compute an absolute
            // path that will point into PA. Note that the resource still will fail,
            // otherwise the resolver had found it there.
            abs_url = make_absolute_package(alloc, url, src->get_name());
        }

        // for now, just make it absolute
        switch (r->get_kind()) {
        case IValue::VK_TEXTURE:
            {
                IValue_texture const *tex = cast<IValue_texture>(r);
                IType_texture const *t = cast<IType_texture>(tf.import(tex->get_type()));
                MDL_ASSERT(t->get_shape() != IType_texture::TS_BSDF_DATA);
                return vf.create_texture(
                    t,
                    abs_url.c_str(),
                    tex->get_gamma_mode(),
                    tex->get_tag_value(),
                    tex->get_tag_version());
            }
        case IValue::VK_LIGHT_PROFILE:
            {
                IValue_light_profile const *lp = cast<IValue_light_profile>(r);
                IType const *t = tf.import(lp->get_type());
                return vf.create_light_profile(
                    cast<IType_light_profile>(t),
                    abs_url.c_str(),
                    lp->get_tag_value(),
                    lp->get_tag_version());
            }
        case IValue::VK_BSDF_MEASUREMENT:
            {
                IValue_bsdf_measurement const *bm = cast<IValue_bsdf_measurement>(r);
                IType const *t = tf.import(bm->get_type());
                return vf.create_bsdf_measurement(
                    cast<IType_bsdf_measurement>(t),
                    abs_url.c_str(),
                    bm->get_tag_value(),
                    bm->get_tag_version());
            }
        default:
            MDL_ASSERT(!"Unsuported resource kind");
        }
    }
    // unmodified
    return r;
}

// Retarget a (relative path) resource url from one package to be accessible from another
// package.
IValue_resource const *retarget_resource_url(
    IValue_resource const *r,
    Position const        &pos,
    Module                *dst,
    Module const          *src,
    File_resolver         &resolver)
{
    return retarget_resource_url(
        r,
        pos,
        dst->get_allocator(),
        *dst->get_type_factory(),
        *dst->get_value_factory(),
        src,
        resolver);
}

/* ------------------------------ Helper shims and functions ------------------------------- */

/// Returns true if the given type is the integer type (or an alias).
///
/// \param type  the type to check
static bool is_integer_type(IType const *type)
{
    return is<IType_int>(type->skip_type_alias());
}

/// Returns true if the given type is the integer or an enum type (or an alias).
///
/// \param type  the type to check
static bool is_integer_or_enum_type(IType const *type)
{
restart:
    switch (type->get_kind()) {
    case IType::TK_ALIAS:
        {
            type = cast<IType_alias>(type)->get_aliased_type();
            goto restart;
        }
    case IType::TK_INT:
    case IType::TK_ENUM:
        return true;
    default:
        return false;
    }
}

/// Returns true if the given type is the given type is the integer, an enum, or a
/// integer vector type  (or an alias).
///
/// \param type  the type to check
static bool is_integer_or_integer_vector_type(IType const *type)
{
restart:
    switch (type->get_kind()) {
    case IType::TK_ALIAS:
        {
            type = cast<IType_alias>(type)->get_aliased_type();
            goto restart;
        }
    case IType::TK_INT:
    case IType::TK_ENUM:
        return true;
    case IType::TK_VECTOR:
        {
            IType_vector const *v_type = cast<IType_vector>(type);
            type = v_type->get_element_type();
            goto restart;
        }
    default:
        return false;
    }
}

/// Returns true if the given type is the bool type (or an alias).
///
/// \param type  the type to check
static bool is_bool_type(IType const *type)
{
    return is<IType_bool>(type->skip_type_alias());
}

/// Returns true if an operator is a comparison operator.
static bool is_comparison(IExpression::Operator op)
{
    switch (op) {
    case IExpression::OK_LESS:
    case IExpression::OK_LESS_OR_EQUAL:
    case IExpression::OK_GREATER_OR_EQUAL:
    case IExpression::OK_GREATER:
    case IExpression::OK_EQUAL:
    case IExpression::OK_NOT_EQUAL:
        return true;
    default:
        return false;
    }
}

/// Returns true if the given symbol result of a syntax error.
static bool is_syntax_error(ISymbol const *sym)
{
    return sym->get_id() == ISymbol::SYM_ERROR;
}

/// Returns true if the given simple name is a syntax error.
static bool is_syntax_error(ISimple_name const *sname)
{
    ISymbol const *sym = sname->get_symbol();
    return is_syntax_error(sym);
}

/// Returns true if the given qualified name is a syntax error.
static bool is_syntax_error(IQualified_name const *qname)
{
    int n = qname->get_component_count();
    if (n <= 0)
        return true;
    ISimple_name const *sname = qname->get_component(n - 1);
    return is_syntax_error(sname);
}

/// Check if types are equal
static bool equal_types(IType const *a, IType const *b)
{
    if (a == b)
        return true;
    if (a->get_type_modifiers() != b->get_type_modifiers())
        return false;
    if (a->skip_type_alias() == b->skip_type_alias())
        return true;

    // for arrays, we need to check aliases on element types
    if (IType_array const *a_arr = as<IType_array>(a)) {
        if (IType_array const *b_arr = as<IType_array>(b)) {
            if (a_arr->get_element_type()->skip_type_alias()
                    != b_arr->get_element_type()->skip_type_alias())
                return false;
            if (a_arr->is_immediate_sized()) {
                if (!b_arr->is_immediate_sized()) return false;
                if (a_arr->get_size() != b_arr->get_size()) return false;
            } else {
                if (b_arr->is_immediate_sized()) return false;
                if (a_arr->get_deferred_size() != b_arr->get_deferred_size()) return false;
            }
            return true;
        }
    }
    return false;
}

/// Check if types need a conversion.
///
/// \param from  the source type
/// \param to    the destination type
///
/// This function must be called for types that are already known to be assignable.
static bool need_type_conversion(IType const *from, IType const *to)
{
    from = from->skip_type_alias();
    to   = to->skip_type_alias();

    // no type conversion is needed if base types are equal
    if (from == to)
        return false;
    if (is<IType_error>(to))
        return false;
    if (is<IType_array>(from)) {
        if (is<IType_array>(to)) {
            // Note: we did not check further here but assume that
            // it is already clear that from CAN be assigned to to
            return false;
        }
    }
    return true;
}

/// Retrieve the Position of the argument at given position.
///
/// \param call   a call, binary, or unary expression
/// \param index  the argument's index
static Position const *get_argument_position(IExpression const *call, int index) {
    switch (call->get_kind()) {
    case IExpression::EK_CALL:
        return &cast<IExpression_call>(call)->get_argument(index)->access_position();
    case IExpression::EK_BINARY:
        if (index == 0) {
            return &cast<IExpression_binary>(call)->get_left_argument()->access_position();
        } else {
            return &cast<IExpression_binary>(call)->get_right_argument()->access_position();
        }
    case IExpression::EK_UNARY:
        return &cast<IExpression_unary>(call)->get_argument()->access_position();
    default:
        MDL_ASSERT(!"Unsupported expression type in get_argument_position()");
        return NULL;
    }
}

/// Check if an expression represents an lvalue.
///
/// \param expr  the expression to check
static bool is_lvalue(IExpression const *expr)
{
    IType const *type = expr->get_type();

    if (is<IType_error>(type)) {
        // do not force more errors, this catches invalid expression also
        // Note here the difference to Analysis::get_lvalue_base()
        return true;
    }

    IType::Modifiers mod = type->get_type_modifiers();

    if (mod & IType::MK_CONST)
        return false;

    for (;;) {
        // in MDL, lvalues are variables
        switch (expr->get_kind()) {
        case IExpression::EK_REFERENCE:
            {
                IExpression_reference const *ref = cast<IExpression_reference>(expr);
                Definition const            *def = impl_cast<Definition>(ref->get_definition());

                if (def->has_flag(Definition::DEF_IS_LET_TEMPORARY)) {
                    // let temporaries are rvalues.
                    // Note here the difference to Analysis::get_lvalue_base(), we want
                    // get_lvalue_base() return the Definition here.
                    return false;
                }

                Definition::Kind kind = def->get_kind();

                // only variables and parameters are
                return kind == Definition::DK_VARIABLE || kind == Definition::DK_PARAMETER;
            }
        case IExpression::EK_BINARY:
            {
                IExpression_binary const     *bin_expr = cast<IExpression_binary>(expr);
                IExpression_binary::Operator op        = bin_expr->get_operator();

                // array indexes and select expression can be lvalues
                if (op == IExpression_binary::OK_ARRAY_INDEX ||
                    op == IExpression_binary::OK_SELECT)
                {
                    expr = bin_expr->get_left_argument();
                    continue;
                }
                return false;
            }
        default:
            return false;
        }
    }
}

/// Get the function qualifier from a (function) definition.
///
/// \param def  the definition
static Qualifier get_function_qualifier(Definition const *def)
{
    if (def->has_flag(Definition::DEF_IS_VARYING))
        return FQ_VARYING;
    if (def->has_flag(Definition::DEF_IS_UNIFORM))
        return FQ_UNIFORM;
    return FQ_NONE;
}

/// Check whether the given simple name is the start import ('*').
///
/// \param sname  the simple name to check
static bool is_star_import(ISimple_name const *sname)
{
    ISymbol const *sym = sname->get_symbol();
    return sym->get_id() == ISymbol::SYM_STAR;
}

/// Check whether the given qualified name is the star import ('*').
///
/// \param qname  the qualified name to check
static bool is_star_import(IQualified_name const *qname)
{
    if (qname->get_component_count() != 1)
        return false;
    ISimple_name const *sname = qname->get_component(0);
    return is_star_import(sname);
}

/// Find an annotation of the given semantics in a annotation block.
///
/// \param block  the annotation block
/// \param sema   the semantics
///
/// \return the first annotation of the given semantics or NULL if none could be found
static IAnnotation const *find_annotation_by_semantics(
    IAnnotation_block const *block,
    IDefinition::Semantics  sema)
{
    for (int i = 0, n = block->get_annotation_count(); i < n; ++i) {
        IAnnotation const     *anno  = block->get_annotation(i);
        IQualified_name const *qname = anno->get_name();
        IDefinition const     *adef  = qname->get_definition();

        if (adef->get_semantics() == sema)
            return anno;
    }
    return NULL;
}

/// Replace the since version of a definition.
///
//  \param def    the annotation whose since flags are replaced
/// \param major  the major version
/// \param minor  the minor version
static void replace_since_version(
    Definition *def,
    int        major,
    int        minor)
{
    unsigned flags = def->get_version_flags() & ~0xFF;
    if (major == 1) {
        switch (minor) {
        case 0:
            flags |= unsigned(IMDL::MDL_VERSION_1_0);
            break;
        case 1:
            flags |= unsigned(IMDL::MDL_VERSION_1_1);
            break;
        case 2:
            flags |= unsigned(IMDL::MDL_VERSION_1_2);
            break;
        case 3:
            flags |= unsigned(IMDL::MDL_VERSION_1_3);
            break;
        case 4:
            flags |= unsigned(IMDL::MDL_VERSION_1_4);
            break;
        case 5:
            flags |= unsigned(IMDL::MDL_VERSION_1_5);
            break;
        case 6:
            flags |= unsigned(IMDL::MDL_VERSION_1_6);
            break;
        case 7:
            flags |= unsigned(IMDL::MDL_VERSION_1_7);
            break;
        default:
            MDL_ASSERT(!"Unsupported version");
            break;
        }
    }
    def->set_version_flags(flags);
}

/// Replace the removed version of a definition.
///
//  \param def    the annotation whose since flags are replaced
/// \param major  the major version
/// \param minor  the minor version
static void replace_removed_version(
    Definition *def,
    int        major,
    int        minor)
{
    unsigned flags = def->get_version_flags() & ~0xFF00;
    if (major == 1) {
        switch (minor) {
        case 0:
            flags |= (unsigned(IMDL::MDL_VERSION_1_0) << 8);
            break;
        case 1:
            flags |= (unsigned(IMDL::MDL_VERSION_1_1) << 8);
            break;
        case 2:
            flags |= (unsigned(IMDL::MDL_VERSION_1_2) << 8);
            break;
        case 3:
            flags |= (unsigned(IMDL::MDL_VERSION_1_3) << 8);
            break;
        case 4:
            flags |= (unsigned(IMDL::MDL_VERSION_1_4) << 8);
            break;
        case 5:
            flags |= (unsigned(IMDL::MDL_VERSION_1_5) << 8);
            break;
        case 6:
            flags |= (unsigned(IMDL::MDL_VERSION_1_6) << 8);
            break;
        case 7:
            flags |= (unsigned(IMDL::MDL_VERSION_1_7) << 8);
            break;
        default:
            MDL_ASSERT(!"Unsupported version");
            break;
        }
    }
    def->set_version_flags(flags);
}

/// Check if the given type is supported by range annotations.
///
/// \param type     the type to check
/// \param version  current MDL version
static bool is_supported_by_range_anno(
    IType const       *type,
    IMDL::MDL_version version)
{
    if (version >= IMDL::MDL_VERSION_1_2) {
        // MDL 1.2 supports the color ...
        if (is<IType_color>(type))
            return true;
        // ... and vector types
        if (IType_vector const *v_type = as<IType_vector>(type)) {
            type = v_type->get_element_type();
        }
    }

    IType::Kind tk = type->get_kind();
    if (tk == IType::TK_INT ||
        tk == IType::TK_FLOAT ||
        tk == IType::TK_DOUBLE)
        return true;
    return false;
}

/// Check if the deprecated annotation is supported.
///
/// \param version  current MDL version
static bool has_deprecated_anno(
    IMDL::MDL_version version)
{
    // available from 1.3
    return version >= IMDL::MDL_VERSION_1_3;
}

/// Compare two values element-wise.
///
/// \param l  first value
/// \param r  second value
///
/// if both values are atomic, just compare them, if both are color
/// or vector typed, compare them element-wise, otherwise return unordered.
static IValue::Compare_results element_wise_compare(
    IValue const *l,
    IValue const *r)
{
    IType const *type = l->get_type();
    if (r->get_type() != type)
        return IValue::CR_UO;
    if (is<IType_atomic>(type))
        return l->compare(r);
    if (is<IType_color>(type) || is<IType_vector>(type)) {
        IValue_compound const *lc = cast<IValue_compound>(l);
        IValue_compound const *rc = cast<IValue_compound>(r);

        IValue::Compare_results res = lc->get_value(0)->compare(rc->get_value(0));
        for (int i = 1, n = lc->get_component_count(); i < n; ++i) {
            IValue::Compare_results lres = lc->get_value(i)->compare(rc->get_value(i));
            if (lres == IValue::CR_EQ) {
                // ignore component eq
                continue;
            }
            if (res == IValue::CR_EQ) {
                // promote
                res = lres;
                continue;
            }
            if (res != lres) {
                // cannot compare
                return IValue::CR_UO;
            }
        }
        return res;
    }
    return IValue::CR_UO;
}

/// Check if it's an explicit or implicit assign operator.
static bool is_assign_operator(IExpression::Operator op)
{
    switch (op) {
    case mi::mdl::IExpression::OK_BITWISE_COMPLEMENT:
    case mi::mdl::IExpression::OK_LOGICAL_NOT:
    case mi::mdl::IExpression::OK_POSITIVE:
    case mi::mdl::IExpression::OK_NEGATIVE:
    case mi::mdl::IExpression::OK_CAST:
    case mi::mdl::IExpression::OK_SEQUENCE:
    case mi::mdl::IExpression::OK_TERNARY:
    case mi::mdl::IExpression::OK_CALL:
    case mi::mdl::IExpression::OK_SELECT:
    case mi::mdl::IExpression::OK_ARRAY_INDEX:
    case mi::mdl::IExpression::OK_MULTIPLY:
    case mi::mdl::IExpression::OK_DIVIDE:
    case mi::mdl::IExpression::OK_MODULO:
    case mi::mdl::IExpression::OK_PLUS:
    case mi::mdl::IExpression::OK_MINUS:
    case mi::mdl::IExpression::OK_SHIFT_LEFT:
    case mi::mdl::IExpression::OK_SHIFT_RIGHT:
    case mi::mdl::IExpression::OK_UNSIGNED_SHIFT_RIGHT:
    case mi::mdl::IExpression::OK_LESS:
    case mi::mdl::IExpression::OK_LESS_OR_EQUAL:
    case mi::mdl::IExpression::OK_GREATER_OR_EQUAL:
    case mi::mdl::IExpression::OK_GREATER:
    case mi::mdl::IExpression::OK_EQUAL:
    case mi::mdl::IExpression::OK_NOT_EQUAL:
    case mi::mdl::IExpression::OK_BITWISE_AND:
    case mi::mdl::IExpression::OK_BITWISE_XOR:
    case mi::mdl::IExpression::OK_BITWISE_OR:
    case mi::mdl::IExpression::OK_LOGICAL_AND:
    case mi::mdl::IExpression::OK_LOGICAL_OR:
        return false;

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

    case mi::mdl::IExpression::OK_PRE_INCREMENT:
    case mi::mdl::IExpression::OK_PRE_DECREMENT:
    case mi::mdl::IExpression::OK_POST_INCREMENT:
    case mi::mdl::IExpression::OK_POST_DECREMENT:
        return true;
    }
    MDL_ASSERT(!"unsupported operator");
    return false;
}

/* ------------------------------ Helper classes ------------------------------- */

/// Helper class to modify a default initializer
class Default_initializer_modifier : public IClone_modifier
{
    typedef IClone_modifier Base;
public:

    /// Constructor.
    ///
    /// \param ana           the name and type analysis object
    /// \param num_args      number of arguments of the processed call
    /// \param origin        the original module from which the initializer is cloned
    explicit Default_initializer_modifier(
        NT_analysis  &ana,
        size_t       num_args,
        Module const *origin)
    : Base()
    , m_ana(ana)
    , m_dst(ana.m_module)
    , m_src(origin)
    , m_param_expr(ana.get_allocator(), num_args)
    , m_auto_imports(ana.m_auto_imports)
    {
        std::fill_n(m_param_expr.data(), m_param_expr.size(), (IExpression const *)0);
    }

    /// Set a parameter value.
    ///
    /// \param index   the index of a parameter
    /// \param expr    its argument
    void set_parameter_value(size_t index, IExpression const *expr)
    {
        m_param_expr[index] = expr;
    }

    /// Clone a reference expression.
    ///
    /// \param ref   the expression to clone
    IExpression *clone_expr_reference(IExpression_reference const *ref) MDL_FINAL
    {
        if (!ref->is_array_constructor()) {
            IDefinition const *idef = ref->get_definition();
            Definition const  *def  = impl_cast<Definition>(idef);

            if (def->get_owner_module_id() != m_dst.get_unique_id()) {
                // an entity from another module
                m_auto_imports.insert(def);
            }

            switch (def->get_kind()) {
            case IDefinition::DK_PARAMETER:
                {
                    int index = def->get_parameter_index();

                    MDL_ASSERT(0 <= index && size_t(index) < m_param_expr.size());
                    if (IExpression const *expr = m_param_expr[index])
                        return m_dst.clone_expr(expr, NULL);
                }
                break;
            case IDefinition::DK_FUNCTION:
            case IDefinition::DK_CONSTRUCTOR:
            case IDefinition::DK_OPERATOR:
                // update the call graph here
                m_ana.update_call_graph(def);
                break;
            default:
                break;
            }
        }
        // just clone it
        return m_dst.clone_expr(ref, NULL);
    }

    /// Clone a literal.
    ///
    /// \param lit  the literal to clone
    IExpression *clone_literal(IExpression_literal const *lit) MDL_FINAL
    {
        IValue const *value = lit->get_value();
        if (IValue_enum const *v = as<IValue_enum>(value)) {
            // cloning an enum, might need an auto-import
            IValue_enum const *nv = cast<IValue_enum>(m_dst.import_value(v));

            IType_enum const *e_tp = nv->get_type();

            Definition_table &dt = m_dst.get_definition_table();

            Scope const *type_scope = dt.get_type_scope(e_tp);
            if (type_scope == NULL) {
                // We have no scope for this type. This can only happen, if it was not imported.
                // Must be auto-imported then.
                if (e_tp->get_predefined_id() == IType_enum::EID_USER) {
                    Definition const *tp_def = m_dst.find_imported_type_definition(v->get_type());
                    if (tp_def != NULL) {
                        m_auto_imports.insert(tp_def);
                    } else {
                        // this should only happen, if some error has occurred, specifically
                        // in an import
                        MDL_ASSERT(m_dst.access_messages().get_error_message_count()> 0);
                    }

                    Expression_factory *fact = m_dst.get_expression_factory();
                    return fact->create_literal(nv);
                } else {
                    // for predefined enums, the default code handles all cases
                }
            }
        } else if (m_src != NULL && m_src != &m_dst) {
            if (IValue_resource const *r = as<IValue_resource>(value)) {
                // check if the URL is relative. If yes, update it
                char const *url = r->get_string_value();
                if (url != NULL && url[0] != '/') {
                    Messages_impl ignore_msg(m_dst.get_allocator(), m_src->get_filename());

                    File_resolver resolver(
                        *m_ana.m_compiler,
                        /*module_cache=*/NULL,
                        m_ana.m_compiler->get_external_resolver(),
                        m_ana.m_compiler->get_search_path(),
                        m_ana.m_compiler->get_search_path_lock(),
                        ignore_msg,
                        m_ana.m_ctx.get_front_path());

                    IValue_resource const *n = retarget_resource_url(
                        r,
                        lit->access_position(),
                        &m_dst,
                        m_src,
                        resolver);

                    if (n != r) {
                        Expression_factory *fact = m_dst.get_expression_factory();
                        return fact->create_literal(n);
                    }
                }
            }
        }
        // just clone it
        return m_dst.clone_expr(lit, NULL);
    }

    /// Clone a call.
    ///
    /// \param call the expression to clone
    IExpression *clone_expr_call(IExpression_call const *c_expr) MDL_FINAL
    {
        IExpression const *ref = m_dst.clone_expr(c_expr->get_reference(), this);
        IExpression_call *call = m_dst.get_expression_factory()->create_call(ref);

        for (int i = 0, n = c_expr->get_argument_count(); i < n; ++i) {
            IArgument const *arg = m_dst.clone_arg(c_expr->get_argument(i), this);
            call->add_argument(arg);
        }
        return call;
    }

    /// Clone a qualified name.
    ///
    /// \param name  the name to clone
    IQualified_name *clone_name(IQualified_name const *qname) MDL_FINAL
    {
        return m_dst.clone_name(qname, NULL);
    }

private:
    /// The name and typer analysis for building the call graph.
    NT_analysis &m_ana;

    /// The destination module.
    Module &m_dst;

    /// The source module if any.
    Module const *m_src;

    /// The map of other parameters.
    VLA<IExpression const *> m_param_expr;

    /// The auto-import map.
    Auto_imports &m_auto_imports;
};

namespace {

/// Helper class to modify a default initializer that references let temporaries.
class Let_temporary_modifier : public IClone_modifier
{
    typedef IClone_modifier Base;
public:
    /// Constructor.
    ///
    /// \param mod  the owner module
    explicit Let_temporary_modifier(
        Module &mod)
    : Base()
    , m_mod(mod)
    {
    }

    /// Clone a reference expression.
    ///
    /// \param ref   the expression to clone
    IExpression *clone_expr_reference(IExpression_reference const *ref) MDL_FINAL
    {
        if (!ref->is_array_constructor()) {
            IDefinition const *idef = ref->get_definition();
            Definition const  *def  = impl_cast<Definition>(idef);

            if (def->has_flag(Definition::DEF_IS_LET_TEMPORARY)) {
                IDeclaration_variable const *vdecl =
                    cast<IDeclaration_variable>(def->get_declaration());

                IExpression const *init = NULL;
                for (int i = 0, n = vdecl->get_variable_count(); i < n; ++i) {
                    ISimple_name const *sname = vdecl->get_variable_name(i);

                    if (sname->get_definition() == def) {
                        // found it
                        init = vdecl->get_variable_init(i);
                        break;
                    }
                }

                if (init != NULL)
                    return m_mod.clone_expr(init, this);
            }
        }
        // just clone it
        return m_mod.clone_expr(ref, NULL);
    }

    /// Clone a call expression.
    ///
    /// \param call  the expression to clone
    IExpression *clone_expr_call(IExpression_call const *c_expr) MDL_FINAL
    {
        // just clone it
        IExpression const *ref = m_mod.clone_expr(c_expr->get_reference(), this);
        IExpression_call *call = m_mod.get_expression_factory()->create_call(ref);

        for (int i = 0, n = c_expr->get_argument_count(); i < n; ++i) {
            IArgument const *arg = m_mod.clone_arg(c_expr->get_argument(i), this);
            call->add_argument(arg);
        }
        return call;
    }

    /// Clone a literal.
    ///
    /// \param lit  the literal to clone
    IExpression *clone_literal(IExpression_literal const *lit) MDL_FINAL
    {
        return m_mod.clone_expr(lit, NULL);
    }

    /// Clone a qualified name.
    ///
    /// \param name  the name to clone
    IQualified_name *clone_name(IQualified_name const *qname) MDL_FINAL
    {
        return m_mod.clone_name(qname, NULL);
    }

private:
    /// The module.
    Module &m_mod;
};

/// A simple module cache using the import list of a module.
class Imported_module_cache : public IModule_cache
{
public:
    /// Constructor.
    ///
    /// \param mod    the module whose import list should be looked up
    /// \param cache  a higher level module cache or NULL
    Imported_module_cache(Module const &mod, IModule_cache *cache)
    : m_mod(mod), m_cache(cache)
    {
    }

    /// Create an \c IModule_cache_lookup_handle for this \c IModule_cache implementation.
    /// Has to be freed using \c free_lookup_handle.
    IModule_cache_lookup_handle *create_lookup_handle() const MDL_FINAL
    {
        return (m_cache == NULL) ? NULL : m_cache->create_lookup_handle();
    }

    /// Free a handle created by \c create_lookup_handle.
    void free_lookup_handle(IModule_cache_lookup_handle *handle) const MDL_FINAL
    {
        if (m_cache != NULL)
            m_cache->free_lookup_handle(handle);
    }

    /// Lookup a module.
    IModule const *lookup(
        char const *absname,
        IModule_cache_lookup_handle *cache_lookup_handle) const MDL_FINAL
    {
        bool direct;
        Module const *imp_mod = m_mod.find_imported_module(absname, direct);

        if (imp_mod != NULL) {
            // reference count is not increased by find_imported_mode(), do it here
            imp_mod->retain();
            return imp_mod;
        }

        if (strcmp(absname, m_mod.get_name()) == 0) {
            // bad: cyclic import
            m_mod.retain();
            return &m_mod;
        }

        if (m_cache != NULL)
            return m_cache->lookup(absname, cache_lookup_handle);

        return NULL;
    }

    /// Get the module loading callback.
    IModule_loaded_callback *get_module_loading_callback() const MDL_FINAL {
        if (m_cache != NULL)
            return m_cache->get_module_loading_callback();

        return NULL;
    }

private:
    /// The module whose import list should be lookup'ed.
    Module const &m_mod;

    /// A higher level module cache or NULL.
    IModule_cache *m_cache;
};

}  // anon namespace

/* --------------------------------- Analysis ---------------------------------- */

// Enter an imported definition.
Definition *Analysis::import_definition(
    Definition const *imported,
    size_t           owner_import_idx,
    Position const   *loc)
{
    Definition *imp = m_def_tab->import_definition(imported, owner_import_idx);
    if (loc != NULL)
        m_import_locations[imp] = loc;
    return imp;
}

// Get the import location of an imported definition.
Position const *Analysis::get_import_location(Definition const *def) const
{
    MDL_ASSERT(def->has_flag(Definition::DEF_IS_IMPORTED));
    Import_locations::const_iterator it = m_import_locations.find(def);
    if (it != m_import_locations.end())
        return it->second;
    return NULL;
}

// Add a compiler note for a previous definition.
void Analysis::add_prev_definition_note(Definition const *prev_def)
{
    if (prev_def->get_position() != NULL) {
        add_note(
            PREVIOUS_DEFINITION,
            prev_def,
            Error_params(*this)
                .add_signature(prev_def));
    }
}

// Add a compiler note if the given expression references an let temporary.
void Analysis::add_let_temporary_note(IExpression const *lhs)
{
    if (IDefinition const *idef = get_lvalue_base(lhs)) {
        Definition const *def = impl_cast<Definition>(idef);
        if (def->has_flag(Definition::DEF_IS_LET_TEMPORARY)) {
            add_note(
                LET_TEMPORARIES_ARE_RVALUES,
                def,
                Error_params(*this).add_signature(def));
        }
    }
}

// Returns true if this is the first message from the given module.
size_t Analysis::get_file_id(
    Messages_impl  &msgs,
    size_t         imp_idx)
{
    Module_2_file_id_map::iterator it = m_modid_2_fileid.find(imp_idx);

    if (it == m_modid_2_fileid.end()) {
        Module::Import_entry const *entry = m_module.get_import_entry(imp_idx);
        MDL_ASSERT(entry != NULL && "message origin not imported in current module");
        char const *fname = entry->get_file_name();
        it = m_modid_2_fileid.insert(
            Module_2_file_id_map::value_type(imp_idx, msgs.register_fname(fname))).first;
    }
    return it->second;
}

// Helper, get the file id from an error param.
size_t Analysis::get_file_id(
    Messages_impl      &msgs,
    Err_location const &loc)
{
    size_t fname_id = 0;

    size_t mod_imp_idx = loc.get_module_import_idx();
    if (mod_imp_idx != 0) {
        fname_id = get_file_id(msgs, mod_imp_idx);
    }
    return fname_id;
}

// Return a fully qualified name for the current scope.
string Analysis::get_current_scope_name() const
{
    Scope *scope = m_def_tab->get_curr_scope();

    // the predef scope is "outside" of a module
    if (scope == m_def_tab->get_predef_scope())
        return string("", m_builder.get_allocator());

    string res(m_module.get_name(), m_builder.get_allocator());

    if (scope != m_def_tab->get_global_scope()) {
        // we are in a local scope, generate a scope unique suffix
        size_t id = scope->get_unique_id();
        char buf[32];

        snprintf(buf, sizeof(buf), "::L%" FMT_SIZE_T, id);
        buf[sizeof(buf) - 1] = '\0';

        res += buf;
    }

    return res;
}

// Return the full name of an entity.
string Analysis::full_name(IQualified_name const *name, bool scope_only)
{
    string res(m_builder.get_allocator());

    for (int i = 0, n = name->get_component_count() - (scope_only ? 1 : 0); i < n; ++i) {
        if (i > 0)
            res += "::";
        res += name->get_component(i)->get_symbol()->get_name();
    }
    return res;
}

// Return the full name of a definition.
string Analysis::full_name(Definition const *def) const
{
    m_string_buf->clear();

    Scope const *def_scope = def->get_def_scope();
    IType const             *def_type  = def_scope->get_scope_type();

    if (def_type) {
        m_printer->print(def_type);
        m_printer->print("::");
    }
    m_printer->print(def->get_sym());
    return string(m_string_buf->get_data(), m_builder.get_allocator());
}

// Create a reference expression for a given definition and put it into given position.
IExpression_reference *Analysis::create_reference(Definition const *def, Position const &pos)
{
    Name_factory       &nf    = *m_module.get_name_factory();
    IQualified_name    *qname = nf.create_qualified_name(POS(pos));
    ISimple_name const *sname = nf.create_simple_name(def->get_sym(), POS(pos));
    qname->add_component(sname);
    qname->set_definition(def);

    IType_name            *tn = nf.create_type_name(qname, POS(pos));
    Expression_factory    *ef = m_module.get_expression_factory();
    IExpression_reference *ref = ef->create_reference(tn, POS(pos));

    ref->set_definition(def);
    ref->set_type(def->get_type());

    return ref;
}

// Get the default expression of a parameter of a function, constructor or annotation.
IExpression const *Analysis::get_default_param_initializer(
    Definition const *def,
    int              param_idx) const
{
    Definition const *orig = m_module.get_original_definition(def);
    return orig->get_default_param_initializer(param_idx);
}

// Format a message.
string Analysis::format_msg(int code, Error_params const &params)
{
    m_string_buf->clear();

    print_error_message(code, MESSAGE_CLASS, params, m_printer.get());
    return string(m_string_buf->get_data(), m_builder.get_allocator());
}

// Parse warning options.
void Analysis::parse_warning_options()
{
    char const *opt = m_compiler->get_compiler_option(&m_ctx, MDL::option_warn);

    if (opt == NULL)
        return;

    for (;;) {
        if (opt[0] == 'e' && opt[1] == 'r' && opt[2] == 'r') {
            m_all_warnings_are_errors = true;
            opt += 3;
        }
        else {
            size_t code = 0;

            while (isdigit(opt[0])) {
                code = code * 10 + opt[0] - '0';
                ++opt;
            }
            if (opt[0] == '=') {
                ++opt;
            }
            if (opt[0] == 'o') {
                if (opt[1] == 'f' && opt[2] == 'f') {
                    if (code < m_disabled_warnings.get_size())
                        m_disabled_warnings.set_bit(code);
                    opt += 3;
                } else if (opt[1] == 'n') {
                    if (code < m_disabled_warnings.get_size())
                        m_disabled_warnings.clear_bit(code);
                    opt += 2;
                }
            }
            if (opt[0] == 'e' && opt[1] == 'r' && opt[2] == 'r') {
                if (code < m_warnings_are_errors.get_size())
                    m_warnings_are_errors.set_bit(code);
                opt += 3;
            }
        }

        if (opt[0] != ',')
            break;
        ++opt;
    }
}

// Creates a new error.
void Analysis::error(int code, Err_location const &loc, Error_params const &params)
{
    Messages_impl &msgs =
        m_compiler_msgs != NULL ? *m_compiler_msgs : m_module.access_messages_impl();

    string msg(format_msg(code, params));
    size_t fname_id = get_file_id(msgs, loc);
    m_last_msg_idx = msgs.add_error_message(
        code, MESSAGE_CLASS, fname_id, loc.get_position(), msg.c_str());
}

// Creates a new error in MDL 1.1+ and strict mode, a warning in MDL <1.1 relaxed mode.
void Analysis::error_mdl_11(
    int                code,
    Err_location const &loc,
    Error_params const &params)
{
    if (!m_strict_mode && m_module.get_mdl_version() < IMDL::MDL_VERSION_1_1) {
        warning(code, loc, params);
    } else {
        error(code, loc, params);
    }
}

// Creates a new error in MDL 1.3+ and strict mode, a warning in MDL <1.3 relaxed mode.
void Analysis::error_mdl_13(
    int                code,
    Err_location const &loc,
    Error_params const &params)
{
    if (!m_strict_mode && m_module.get_mdl_version() < IMDL::MDL_VERSION_1_3) {
        warning(code, loc, params);
    } else {
        error(code, loc, params);
    }
}

// Creates a new error in strict mode, a warning in relaxed mode.
void Analysis::error_strict(
    int                code,
    Err_location const &loc,
    Error_params const &params)
{
    if (!m_strict_mode) {
        warning(code, loc, params);
    } else {
        error(code, loc, params);
    }
}

// Creates a new warning.
void Analysis::warning(int code, Err_location const &loc, Error_params const &params)
{
    bool marked_as_error = m_warnings_are_errors.test_bit(code);

    if (!marked_as_error && (m_all_warnings_are_off || m_disabled_warnings.test_bit(code))) {
        // suppress this warning
        m_last_msg_idx = ~size_t(0);
        return;
    }

    bool is_error = marked_as_error || m_all_warnings_are_errors;

    Messages_impl &msgs =
        m_compiler_msgs != NULL ? *m_compiler_msgs : m_module.access_messages_impl();

    string msg(format_msg(code, params));
    size_t fname_id = get_file_id(msgs, loc);
    m_last_msg_idx = is_error ?
        msgs.add_error_message(code, MESSAGE_CLASS, fname_id, loc.get_position(), msg.c_str()) :
        msgs.add_warning_message(code, MESSAGE_CLASS, fname_id, loc.get_position(), msg.c_str());
}

// Add a note to the last error.
void Analysis::add_note(int code, Err_location const &loc, Error_params const &params)
{
    if (m_last_msg_idx == ~size_t(0)) {
        // associated warning was suppressed
        return;
    }

    Messages_impl &msgs =
        m_compiler_msgs != NULL ? *m_compiler_msgs : m_module.access_messages_impl();

    string msg(format_msg(code, params));
    size_t fname_id = get_file_id(msgs, loc);
    msgs.add_note(
        m_last_msg_idx,
        IMessage::MS_INFO,
        code,
        MESSAGE_CLASS,
        fname_id,
        loc.get_position(),
        msg.c_str());
}

// Add an imported message to the last error
void Analysis::add_imported_message(size_t fname_id, IMessage const *msg)
{
    Messages_impl &msgs =
        m_compiler_msgs != NULL ? *m_compiler_msgs : m_module.access_messages_impl();

    msgs.add_imported(m_last_msg_idx, fname_id, msg);
}

// Get the definition of the base entity in a lvalue.
IDefinition const *Analysis::get_lvalue_base(IExpression const *expr)
{
    IType const *type = expr->get_type();

    if (is<IType_error>(type)) {
        // do not force more errors, this catches invalid expression also
        return NULL;
    }

    IType::Modifiers mod = type->get_type_modifiers();

    if (mod & IType::MK_CONST)
        return NULL;

    for (;;) {
        // in MDL, lvalues are variables
        switch (expr->get_kind()) {
        case IExpression::EK_REFERENCE:
            {
                IExpression_reference const *ref = cast<IExpression_reference>(expr);
                IDefinition const           *def = ref->get_definition();
                Definition::Kind            kind = def->get_kind();

                if (kind == Definition::DK_VARIABLE || kind == Definition::DK_PARAMETER)
                    return def;
                return NULL;
            }
        case IExpression::EK_BINARY:
            {
                IExpression_binary const     *bin_expr = cast<IExpression_binary>(expr);
                IExpression_binary::Operator op        = bin_expr->get_operator();

                // array indexes and select expression can be lvalues
                if (op == IExpression_binary::OK_ARRAY_INDEX ||
                    op == IExpression_binary::OK_SELECT)
                {
                    expr = bin_expr->get_left_argument();
                    continue;
                }
                return NULL;
            }
        default:
            return NULL;
        }
    }
}

// Returns the return type of a function definition.
IType const *Analysis::get_result_type(IDefinition const *def)
{
    IType const *type = def->get_type();

    switch (type->get_kind()) {
    case IType::TK_ERROR:
        // can be the error type, so handle this gracefully
        return type;
    case IType::TK_FUNCTION:
        {
            IType_function const *func_type = cast<IType_function>(type);
            return func_type->get_return_type();
        }
    default:
        break;
    }

    MDL_ASSERT(!"get_result_type() called for non-function definition");
    return type;
}

// Handle constant folding exceptions.
void Analysis::Const_fold_expression::exception(
    Reason            r,
    IExpression const *expr,
    int               index,
    int               length)
{
    m_error_state = true;

    switch (r) {
    case IConst_fold_handler::ER_INT_DIVISION_BY_ZERO:
        m_ana.error(
            DIVISION_BY_ZERO_IN_CONSTANT_EXPR,
            expr->access_position(),
            Error_params(m_ana));
        return;
    case IConst_fold_handler::ER_INDEX_OUT_OF_BOUND:
        if (index < 0) {
            m_ana.error(
                ARRAY_INDEX_OUT_OF_RANGE,
                expr->access_position(),
                Error_params(m_ana).add(index).add("<").add("0"));
        } else {
            m_ana.error(
                ARRAY_INDEX_OUT_OF_RANGE,
                expr->access_position(),
                Error_params(m_ana).add(index).add(">=").add(length));
        }
        return;
    }
    MDL_ASSERT(!"Unsupported exception reason");
}

// Handle variable lookup.
IValue const *Analysis::Const_fold_expression::lookup(
    IDefinition const *var)
{
    // not supported
    return m_ana.get_module().get_value_factory()->create_bad();
}

// Handle intrinsic call evaluation.
IValue const *Analysis::Const_fold_expression::evaluate_intrinsic_function(
    IDefinition::Semantics semantic,
    const IValue *const arguments[],
    size_t n_arguments)
{
    // not supported
    return m_ana.get_module().get_value_factory()->create_bad();
}

// Issue an error for some previously defined entity.
void Analysis::err_redeclaration(
    Definition::Kind  kind,
    Definition const  *def,
    Position const    &pos,
    Compilation_error err,
    bool              as_mat)
{
    if (is_error(def)) {
        // already error
        return;
    }

    if (kind != def->get_kind())
        err = REDECLARATION_OF_DIFFERENT_KIND;
    else if (as_mat && kind == IDefinition::DK_FUNCTION) {
        // Argh, materials are functions returning "material"
        IType const *ret_type = as<IType_function>(def->get_type())->get_return_type();

        if (ret_type != m_tc.material_type)
            err = REDECLARATION_OF_DIFFERENT_KIND;
    }

    error(
        err,
        pos,
        Error_params(*this)
            .add_signature(def));
    add_prev_definition_note(def);
}

// Issue a warning if some previously defined entity is shadowed.
void Analysis::warn_shadow(
    Definition const  *def,
    Position const    &pos,
    Compilation_error warn)
{
    warning(
        warn,
        pos,
        Error_params(*this)
            .add_signature(def));
    add_prev_definition_note(def);
}

// Resolve a scope operator and enter the scope.
Scope *Analysis::resolve_scope(
    IQualified_name const *scope_node,
    bool &had_error,
    bool silent)
{
    bool error_reported = had_error;
    Scope *scope        = NULL;

    size_t n = scope_node->get_component_count();

    if (n > 1) {
        // we have a scope operator, search ALWAYS from global,
        // because only there imports can happen.
        scope = m_def_tab->get_global_scope();

        for (size_t i = 0; i < n - 1; ++i) {
            ISimple_name const *identifier = scope_node->get_component(i);
            ISymbol const *scope_sym = identifier->get_symbol();

            Scope *named_scope = scope->find_named_subscope(scope_sym);
            if (named_scope == NULL) {
                if (! error_reported) {
                    // only one error ...
                    if (! silent) {
                        ISymbol const *bsym = find_best_named_subscope_match(scope_sym, scope);
                        string tname = full_name(scope_node, /*only_scopy=*/true);
                        error(
                            UNKNOWN_PACKAGE_NAME,
                            scope_node->access_position(),
                            Error_params(*this)
                                .add(tname.c_str())
                                .add_possible_match(bsym));
                    }
                    error_reported = true;
                }
                // reset the scope here ??
                break;
            } else {
                scope = named_scope;
            }
        }
    }

    had_error = error_reported;
    return scope;
}

// Returns true if the given type is the base material.
bool Analysis::is_base_material(IType const *type) const
{
    IType const *base_mat = m_tc.material_type;
    return base_mat == type;
}

namespace {

/// Helper class to evaluate a set of candidates for finding one with the minimum Levenshtein
/// distance to the searched one.
class Levenshtein_matcher {
public:
    /// Constructor.
    Levenshtein_matcher(
        IAllocator    *alloc,
        ISymbol const *src)
    : m_src(src)
    , m_best(NULL)
    , m_dist(size_t(-1))
    , m_s_len(strlen(src->get_name()))
    , m_buffer(alloc)
    {
    }

    /// Get the best found match so far if any.
    ISymbol const *get_best_match() const
    {
        if (m_best != NULL) {
            char const *t    = m_best->get_name();
            size_t     t_len = strlen(t);

            // limit the possible best match: we do not what that "abc" is a substitute for "def".
            // If more than half of the letters were misspelled, ignore the suggestion.
            size_t threshold = t_len > m_s_len ? t_len >> 1 : m_s_len >> 1;
            if (m_dist > threshold)
                return NULL;
        }

        return m_best;
    }

    /// Evaluate a candidate.
    void eval_candidate(ISymbol const *dst)
    {
        if (m_src == dst) {
            // this should not happen: we found an exact match. Ignore it, because
            // it was obviously not taken into account because of other factors.
            return;
        }

        char const *t    = dst->get_name();
        size_t     t_len = strlen(t);

        // at best we must add/remove this number of characters
        size_t min_dist = t_len > m_s_len ? t_len - m_s_len : m_s_len - t_len;

        if (min_dist >= m_dist)
            return;

        // limit the possible best match: we do not want that "abc" is a substitute for "def".
        // If more than half of the letters were misspelled, ignore the suggestion.
        size_t threshold = t_len > m_s_len ? t_len >> 1 : m_s_len >> 1;

        if (min_dist > threshold)
            return;

        size_t dist = edit_distance(t, t_len);
        if (dist > threshold)
            return;

        if (dist < m_dist) {
            // found a better match
            m_dist  = dist;
            m_best  = dst;
        }
    }

private:
    /// Implementation of the Wagner-Fischer algorithm.
    size_t edit_distance(
        char const *t, size_t t_len)
    {
        size_t s_len = m_s_len;

        // should not happen ...
        if (s_len == 0)
            return t_len;
        if (t_len == 0)
            return s_len;

        char const *s = m_src->get_name();

        m_buffer.resize(2 * (s_len + 1));

        // The original alrorithm computes a matrix (t_len + 1) * (s_len + 1)
        // where every (i, j) contains the Levenshtein distance between the
        // prefixes s[0:j] and t[0:i].
        // However, for the computation itself only the previous row and the current
        // row are needed.
        size_t *prev_row = &m_buffer[0];
        size_t *curr_row = &m_buffer[s_len + 1];

        // the distance of any first string (s) to an empty second string (t)
        // (transforming the string of the first i characters of s into
        // the empty string requires i deletions)
        for (size_t i = 0; i < s_len + 1; ++i)
            prev_row[i] = i;

        // compute other rows
        for (size_t j = 1; j < t_len + 1; ++j) {
            // the distance of any second string to an empty first string
            curr_row[0] = j;

            for (size_t i = 1; i < s_len + 1; ++i) {
                if (s[i - 1] == t[j - 1]) {
                    curr_row[i] = prev_row[i - 1]; // no operation required
                } else {
                    curr_row[i] = min3(
                        curr_row[i - 1], // a deletion
                        prev_row[i    ], // an insertion
                        prev_row[i - 1]  // a substitution
                    ) + 1;
                }
            }

            // switch rows
            size_t *tmp = curr_row; curr_row = prev_row; prev_row = tmp;
        }

        // the result is in M(t_len, s_len) now, but we have already switched the rows
        return prev_row[s_len];
    }

    static size_t min3(size_t a, size_t b, size_t c) {
        if (a < b)
            return a < c ? a : c;
        else
            return b < c ? b : c;
    }

private:
    ISymbol const *m_src;
    ISymbol const *m_best;
    size_t        m_dist;
    size_t        m_s_len;

    vector<size_t>::Type m_buffer;

};

}  // anonymous

// Check if a given definition kind is allowed under the given match restriction.
bool Analysis::allow_definition(
    IDefinition::Kind kind,
    Match_restriction restriction)
{
    switch (restriction) {
    case MR_ANY:
        return true;
    case MR_MEMBER:
        return kind == IDefinition::DK_MEMBER;
    case MR_CALLABLE:
        switch (kind) {
        case IDefinition::DK_FUNCTION:
        case IDefinition::DK_CONSTRUCTOR:
        case IDefinition::DK_OPERATOR:
            return true;
        default:
            return false;
        }
    case MR_NON_CALLABLE:
        return !allow_definition(kind, MR_CALLABLE);
    case MR_TYPE:
        return kind == IDefinition::DK_TYPE;
    case MR_ANNO:
        return kind == IDefinition::DK_ANNOTATION;
    }
    MDL_ASSERT(!"Unknwon match restriction");
    return true;
}

// Try to find the best candidate for a search symbol using an edit distance metric.
ISymbol const *Analysis::find_best_match(
    ISymbol const      *sym,
    Scope              *scope,
    Match_restriction  restriction) const
{
    typedef list<Definition const *>::Type Candidate_list;

    Candidate_list candidates(get_allocator());

    if (scope != NULL) {
        for (Definition const *def = scope->get_first_definition_in_scope();
            def != NULL;
            def = def->get_next_def_in_scope())
        {
            candidates.push_back(def);
        }
    } else {
        size_t index = 0;
        for (Definition const *def = m_def_tab->get_visible_definition(index);
            def != NULL;
            def = m_def_tab->get_visible_definition(index))
        {
            candidates.push_back(def);
        }
    }

    Levenshtein_matcher matcher(get_allocator(), sym);

    for (Candidate_list::const_iterator it(candidates.begin()), end(candidates.end());
        it != end;
        ++it)
    {
        Definition const *def = *it;
        if (allow_definition(def->get_kind(), restriction))
            matcher.eval_candidate(def->get_symbol());
    }
    return matcher.get_best_match();
}

// Try to find the best named subscope candidate for a search symbol using an
// edit distance metric.
ISymbol const *Analysis::find_best_named_subscope_match(
    ISymbol const *sym,
    Scope         *scope) const
{
    Levenshtein_matcher matcher(get_allocator(), sym);

    for (Scope *s = scope->get_first_named_subscope();
        s != NULL;
        s = s->get_next_named_subscope())
    {
        matcher.eval_candidate(s->get_scope_name());
    }
    return matcher.get_best_match();
}

// Find the definition for a qualified name.
Definition *Analysis::find_definition_for_qualified_name(
    IQualified_name const *qual_name,
    bool                  ignore_error)
{
    // check for scopes first
    ISimple_name const *identifier = qual_name->get_component(
        qual_name->get_component_count() - 1);
    bool error_reported = ignore_error;
    Scope *bound_scope  = resolve_scope(qual_name, error_reported);

    ISymbol const *sym = identifier->get_symbol();
    Definition *def = NULL;

    if (m_in_select != NULL) {
        // if we are inside an select, we can only look up this select scope; there should
        // also be no additional scope here
        if (bound_scope == NULL) {
            def = m_def_tab->get_curr_scope()->find_definition_in_scope(sym);
        }
    } else {
        def = bound_scope
                ? bound_scope->find_definition_in_scope(sym)
                : m_def_tab->get_definition(sym);
    }
    if (def == NULL) {
        if (is_syntax_error(sym)) {
            // parse errors on names are expressed by error symbols
            error_reported = true;
        }
        if (! error_reported) {
            string sc_name = full_name(qual_name, /*only_scope=*/false);

            if (m_in_select != NULL) {
                if (!is<IType_error>(m_in_select)) {
                    Scope *type_scope = m_def_tab->get_type_scope(m_in_select);
                    ISymbol const *bsym = find_best_match(sym, type_scope, MR_MEMBER);

                    error(UNKNOWN_MEMBER,
                        qual_name->access_position(),
                        Error_params(*this)
                            .add(m_in_select)
                            .add(sc_name.c_str())
                            .add_possible_match(bsym));
                }
            } else {
                ISymbol const *bsym = find_best_match(sym, bound_scope, m_curr_restriction);

                error(
                    UNKNOWN_IDENTIFIER,
                    qual_name->access_position(),
                    Error_params(*this)
                        .add(sc_name.c_str())
                        .add_possible_match(bsym));
            }
        }

        def = get_error_definition();
    }
    return def;
}

// Find the definition for an annotation name.
Definition *Analysis::find_definition_for_annotation_name(IQualified_name const *qual_name)
{
    // check for scopes first
    ISimple_name const *identifier = qual_name->get_component(
        qual_name->get_component_count() - 1);
    bool has_error = false;
    Scope *bound_scope  = resolve_scope(qual_name, has_error, /*silent=*/true);

    if (has_error) {
        string tname = full_name(qual_name, /*only_scopy=*/true);
        warning(
            UNKNOWN_PACKAGE_NAME,
            qual_name->access_position(),
            Error_params(*this).add(tname.c_str()));
        return get_error_definition();
    }

    ISymbol const *sym = identifier->get_symbol();
    Definition *def = NULL;

    def = bound_scope
        ? bound_scope->find_definition_in_scope(sym)
        : m_def_tab->get_definition(sym);

    if (def == NULL) {
        // do not issue errors for parse errors on names
        if (!is_syntax_error(sym)) {
            ISymbol const *bsym = find_best_match(sym, bound_scope, MR_ANNO);
            warning(
                UNKNOWN_IDENTIFIER,
                qual_name->access_position(),
                Error_params(*this)
                    .add(sym)
                    .add_possible_match(bsym));
        }
        def = get_error_definition();
    }
    return def;
}

// Check if a given qualified name name the material (type).
bool Analysis::is_material_qname(IQualified_name const *qual_name)
{
    // check for scopes first
    ISimple_name const *identifier = qual_name->get_component(
        qual_name->get_component_count() - 1);
    bool had_error = false;
    Scope *bound_scope  = resolve_scope(qual_name, had_error, /*silent=*/true);

    if (had_error)
        return false;

    ISymbol const *sym = identifier->get_symbol();
    if (is_syntax_error(sym)) {
        // parse errors on names are expressed by error symbols
        return false;
    }

    Definition *def = bound_scope
        ? bound_scope->find_definition_in_scope(sym)
        : m_def_tab->get_definition(sym);
    if (def != NULL) {
        if (IType_struct const *s_type = as<IType_struct>(def->get_type())) {
            return s_type->get_predefined_id() == IType_struct::SID_MATERIAL;
        }
    }
    return false;
}

// Get the "error definition".
Definition *Analysis::get_error_definition() const
{
    ISymbol const *err_sym      = m_st->get_error_symbol();
    Scope         *predef_scope = m_def_tab->get_predef_scope();
    Definition    *def          = predef_scope->find_definition_in_scope(err_sym);

    MDL_ASSERT(def != NULL && "Cannot find the error definition");
    return def;
}

// Checks if the give expression is a constant one.
bool Analysis::is_const_expression(IExpression const *expr, bool &is_invalid)
{
    switch (expr->get_kind()) {
    case IExpression::EK_INVALID:
        is_invalid = true;
        return false;
    case IExpression::EK_LITERAL:
        return true;
    case IExpression::EK_REFERENCE:
    {
        IExpression_reference const *r_expr = cast<IExpression_reference>(expr);
        if (r_expr->is_array_constructor())
            return false;

        IDefinition const *def = r_expr->get_definition();

        Definition::Kind kind = def->get_kind();
        if (is_error(def)) {
            // an undefined entity
            is_invalid = true;
            return false;
        }

        if (kind == Definition::DK_CONSTANT || kind == Definition::DK_ENUM_VALUE) {
            IValue const *val = def->get_constant_value();
            if (is<IValue_bad>(val)) {
                // an error constant
                is_invalid = true;
                return false;
            }
            return true;
        }
        return false;
    }
    case IExpression::EK_UNARY:
    {
        IExpression_unary const *u_expr = cast<IExpression_unary>(expr);
        return is_const_expression(u_expr->get_argument(), is_invalid);
    }
    case IExpression::EK_BINARY:
    {
        IExpression_binary const *b_expr = cast<IExpression_binary>(expr);
        switch (b_expr->get_operator()) {
        case IExpression_binary::OK_SELECT:
            return is_const_expression(b_expr->get_left_argument(), is_invalid);

        // simple arithmetic is ok
        case IExpression_binary::OK_ARRAY_INDEX:
        case IExpression_binary::OK_MULTIPLY:
        case IExpression_binary::OK_DIVIDE:
        case IExpression_binary::OK_MODULO:
        case IExpression_binary::OK_PLUS:
        case IExpression_binary::OK_MINUS:
        case IExpression_binary::OK_SHIFT_LEFT:
        case IExpression_binary::OK_SHIFT_RIGHT:
        case IExpression_binary::OK_UNSIGNED_SHIFT_RIGHT:
        case IExpression_binary::OK_LESS:
        case IExpression_binary::OK_LESS_OR_EQUAL:
        case IExpression_binary::OK_GREATER_OR_EQUAL:
        case IExpression_binary::OK_GREATER:
        case IExpression_binary::OK_EQUAL:
        case IExpression_binary::OK_NOT_EQUAL:
        case IExpression_binary::OK_BITWISE_AND:
        case IExpression_binary::OK_BITWISE_XOR:
        case IExpression_binary::OK_BITWISE_OR:
        case IExpression_binary::OK_LOGICAL_AND:
        case IExpression_binary::OK_LOGICAL_OR:
            return is_const_expression(b_expr->get_left_argument(), is_invalid) &&
                is_const_expression(b_expr->get_right_argument(), is_invalid);

        // no assignments
        case IExpression_binary::OK_ASSIGN:
        case IExpression_binary::OK_MULTIPLY_ASSIGN:
        case IExpression_binary::OK_DIVIDE_ASSIGN:
        case IExpression_binary::OK_MODULO_ASSIGN:
        case IExpression_binary::OK_PLUS_ASSIGN:
        case IExpression_binary::OK_MINUS_ASSIGN:
        case IExpression_binary::OK_SHIFT_LEFT_ASSIGN:
        case IExpression_binary::OK_SHIFT_RIGHT_ASSIGN:
        case IExpression_binary::OK_UNSIGNED_SHIFT_RIGHT_ASSIGN:
        case IExpression_binary::OK_BITWISE_AND_ASSIGN:
        case IExpression_binary::OK_BITWISE_XOR_ASSIGN:
        case IExpression_binary::OK_BITWISE_OR_ASSIGN:
            return false;
        case IExpression_binary::OK_SEQUENCE:
            return false;
        }
    }
    case IExpression::EK_CONDITIONAL:
        // not allowed
        return false;
    case IExpression::EK_CALL:
    {
        IExpression_call const *c_expr = cast<IExpression_call>(expr);

        IExpression const *ref = c_expr->get_reference();
        if (is<IExpression_invalid>(ref)) {
            // error already reported, not create another one
            return true;
        }
        IExpression_reference const *callee = as<IExpression_reference>(ref);
        if (callee == NULL) {
            // not a reference: should not happen yet, but if it does
            // it is not constant
            return false;
        }

        if (callee->is_array_constructor()) {
            // an array constructor is a const expression if all of its arguments are
            for (int i = 0, n = c_expr->get_argument_count(); i < n; ++i) {
                IArgument const   *arg  = c_expr->get_argument(i);
                IExpression const *expr = arg->get_argument_expr();

                if (!is_const_expression(expr, is_invalid))
                    return false;
            }
            return true;
        }

        IDefinition const *c_def = callee->get_definition();
        Definition const  *def   = impl_cast<Definition>(c_def);

        if (def->get_semantics() == IDefinition::DS_COLOR_SPECTRUM_CONSTRUCTOR) {
            // FIXME: until we have a working representation for the spectrum color
            // we cannot fold it
            return false;
        }

        // we have const_expr in MDL 1.3+
        if (m_module.get_mdl_version() >= IMDL::MDL_VERSION_1_3 &&
            def->has_flag(Definition::DEF_IS_CONST_EXPR))
        {
            // check if all arguments are const
            for (int i = 0, n = c_expr->get_argument_count(); i < n; ++i) {
                IArgument const   *arg  = c_expr->get_argument(i);
                IExpression const *expr = arg->get_argument_expr();

                if (!is_const_expression(expr, is_invalid))
                    return false;
            }
            return true;
        }

        // check if this is a constructor call
        if (def->get_kind() != Definition::DK_CONSTRUCTOR) {
            if (def->get_kind() == Definition::DK_ERROR)
                is_invalid = true;
            return false;
        }

        if (def->get_semantics() == IDefinition::DS_DEFAULT_STRUCT_CONSTRUCTOR) {
            // check if the default constructor creates const values
            return def->has_flag(Definition::DEF_IS_CONST_CONSTRUCTOR);
        }
        for (size_t i = 0, n = c_expr->get_argument_count(); i < n; ++i) {
            IArgument const   *arg  = c_expr->get_argument(i);
            IExpression const *expr = arg->get_argument_expr();

            if (!is_const_expression(expr, is_invalid))
                return false;
        }
        return true;
    }
    default:
        return false;
    }
}

// Checks if the give expression is a constant array size.
bool Analysis::is_const_array_size(
    IExpression const *expr,
    IDefinition const *&def,
    bool              &is_invalid)
{
    def = NULL;
    if (IExpression_reference const *ref = as<IExpression_reference>(expr)) {
        IDefinition const *arr_size_def = ref->get_definition();

        if (arr_size_def->get_kind() == Definition::DK_ARRAY_SIZE) {
            def = arr_size_def;
            return true;
        }
    }
    return is_const_expression(expr, is_invalid);
}

// Apply a type qualifier to a type.
IType const *Analysis::qualify_type(Qualifier qual, IType const *type)
{
    switch (qual) {
    case FQ_NONE:
        // do nothing, auto-typed is the default
        break;
    case FQ_VARYING:
        type = m_tc.decorate_type(type, IType::MK_VARYING);
        break;
    case FQ_UNIFORM:
        type = m_tc.decorate_type(type, IType::MK_UNIFORM);
        break;
    }
    return type;
}

Analysis::Analysis(
    MDL            *compiler,
    Module         &module,
    Thread_context &ctx)
: m_builder(module.get_allocator())
, m_compiler(compiler)
, m_module(module)
, m_ctx(ctx)
, m_mdl_version(module.get_mdl_version())
, m_st(&module.get_symbol_table())
, m_tc(*module.get_type_factory())
, m_def_tab(&module.get_definition_table())
, m_in_select(NULL)
, m_last_msg_idx(~size_t(0))
, m_curr_restriction(MR_ANY)
, m_string_buf(m_builder.create<Buffer_output_stream>(module.get_allocator()))
, m_printer(m_builder.create<Printer>(module.get_allocator(), m_string_buf.get()))
, m_exc_handler(*this)
, m_disabled_warnings()
, m_warnings_are_errors()
, m_compiler_msgs(NULL)
, m_all_warnings_are_errors(false)
, m_all_warnings_are_off(ctx.all_warnings_are_off())
, m_strict_mode(
    compiler->get_compiler_bool_option(&ctx, MDL::option_strict, true))
, m_enable_experimental_features(
    compiler->get_compiler_bool_option(&ctx, MDL::option_experimental_features, false))
, m_resolve_resources(
    compiler->get_compiler_bool_option(&ctx, MDL::option_resolve_resources, true))
, m_modid_2_fileid(Module_2_file_id_map::key_compare(), module.get_allocator())
, m_import_locations(
    0, Import_locations::hasher(), Import_locations::key_equal(), module.get_allocator())
{
    // disable "operator is strict in material" warning by default
    m_disabled_warnings.set_bit(OPERATOR_IS_STRICT_IN_MATERIAL);

    // disable "double type used" warning by default
    m_disabled_warnings.set_bit(DOUBLE_TYPE_USED);

    // spec forbids import MODULE without '::*' suffix, make it an error
    m_warnings_are_errors.set_bit(MODULE_NAME_GIVEN_NOT_ENTITY);

    parse_warning_options();
}

Analysis::~Analysis()
{
}

/* --------------------------- Operator cache ---------------------- */

/// Constructor.
Operator_lookup_cache::Operator_lookup_cache(IAllocator *alloc)
: m_alloc(alloc, mi::base::DUP_INTERFACE)
, m_cache(0, Op_cache::hasher(), Op_cache::key_equal(), alloc)
{
}

// Lookup an operator.
Definition const *Operator_lookup_cache::lookup(
     IExpression::Operator op,
     IType const           *left_tp,
     IType const           *right_tp)
{
    Op_cache::const_iterator it = m_cache.find(Operator_signature(op, left_tp, right_tp));
    if (it != m_cache.end())
        return it->second;
    return NULL;
}

// Store an operator definition into the cache
void Operator_lookup_cache::insert(
    IExpression::Operator op,
    IType const           *left_tp,
    IType const           *right_tp,
    Definition const      *def)
{
    m_cache[Operator_signature(op, left_tp, right_tp)] = def;
}

/* --------------------------- Auto imports ----------------------- */

Auto_imports::Auto_imports(IAllocator *alloc)
: m_index_map(0, Index_map::hasher(), Index_map::key_equal(), alloc)
, m_imports(alloc)
{
}

// Find the entry for a given definition.
Auto_imports::Entry const *Auto_imports::find(Definition const *def) const
{
    Index_map::const_iterator it = m_index_map.find(def);
    if (it != m_index_map.end())
        return &m_imports[it->second];
    return NULL;
}

// Insert a new foreign definition.
bool Auto_imports::insert(Definition const *def)
{
    std::pair<Index_map::iterator, bool> res =
        m_index_map.insert(Index_map::value_type(def, 0u));
    if (res.second) {
        // new entry
        size_t index = m_imports.size();
        m_imports.push_back(Entry(def, NULL));
        res.first->second = index;
    }
    return res.second;
}

/* --------------------------- Name analysis ---------------------- */

// Constructor.
NT_analysis::NT_analysis(
    MDL              *compiler,
    Module           &module,
    Thread_context   &ctx,
    IModule_cache    *cache)
: Analysis(compiler, module, ctx)
, m_preset_overload(NULL)
, m_is_stdlib(module.is_stdlib())
, m_in_expected_const_expr(false)
, m_ignore_deferred_size(false)
, m_in_array_size(false)
, m_params_are_used(false)
, m_inside_material_constr(false)
, m_allow_let_expression(false)
, m_inside_let_decls(false)
, m_inside_material_defaults(false)
, m_inside_param_initializer(false)
, m_can_throw_bounds(false)
, m_can_throw_divzero(false)
, m_opt_dump_cg(
    compiler->get_compiler_bool_option(&ctx, MDL::option_dump_call_graph, /*def_value=*/false))
, m_opt_keep_original_resource_file_paths(
    compiler->get_compiler_bool_option(&ctx, MDL::option_keep_original_resource_file_paths, /*def_value*/false))
, m_is_module_annotation(false)
, m_is_return_annotation(false)
, m_has_array_assignment(module.get_mdl_version() >= IMDL::MDL_VERSION_1_3)
, m_module_cache(cache)
, m_annotated_def(NULL)
, m_next_param_idx(0)
, m_func_stack_depth(0)
, m_func_stack_pos(0)
, m_func_stack(module.get_allocator())
, m_op_cache(module.get_allocator())
, m_cg(module.get_allocator(), module.get_name(), module.is_stdlib())
, m_type_bindings(0, Bind_type_map::hasher(), Bind_type_map::key_equal(), module.get_allocator())
, m_size_bindings(0, Bind_size_map::hasher(), Bind_size_map::key_equal(), module.get_allocator())
, m_sym_bindings(0, Bind_symbol_map::hasher(), Bind_symbol_map::key_equal(), module.get_allocator())
, m_array_size_map(0, Array_size_map::hasher(), Array_size_map::key_equal(), module.get_allocator())
, m_exported_decls_only(module.get_allocator())
, m_auto_imports(module.get_allocator())
, m_initializers_must_be_fixed(module.get_allocator())
, m_sema_version_pos(NULL)
, m_resource_entries(Resource_table::key_compare(), module.get_allocator())
{
}

// Returns the definition of a symbol at the at a given scope.
Definition *NT_analysis::get_definition_at_scope(ISymbol const *sym, Scope *scope) const
{
    if (Definition *def = m_def_tab->get_definition(sym))
        if (def->get_def_scope() == scope)
            return def;
    return NULL;
}

// Returns the definition of a symbol at the current scope only.
Definition *NT_analysis::get_definition_at_scope(ISymbol const *sym) const
{
    return get_definition_at_scope(sym, m_def_tab->get_curr_scope());
}

// If the given definition is auto-imported, return its imported definition.
Definition const *NT_analysis::get_auto_imported_definition(Definition const *def) const
{
    Auto_imports::Entry const *entry = m_auto_imports.find(def);
    if (entry != NULL)
        return entry->imported;
    return NULL;
}

// Push a Definition on the function stack.
void NT_analysis::push_function(Definition *def)
{
    if (m_func_stack_pos >= m_func_stack_depth) {
        m_func_stack.push_back(def);
        ++m_func_stack_depth;
    } else {
        m_func_stack[m_func_stack_pos] = def;
    }
    ++m_func_stack_pos;
}

// Pop the function stack and return the TOS.
Definition *NT_analysis::pop_function() {
    MDL_ASSERT(m_func_stack_pos > 0);
    return m_func_stack[--m_func_stack_pos];
}

// Return the current function on the stack.
Definition *NT_analysis::tos_function() const
{
    if (m_func_stack_pos > 0)
        return m_func_stack[m_func_stack_pos - 1];
    // stack can be empty for calls inside function initializers
    return NULL;
}

// Enter one predefined constant into the current scope.
Definition *NT_analysis::enter_builtin_constant(
    ISymbol const *sym,
    IType const   *type,
    IValue const  *val)
{
    Definition *def = m_def_tab->enter_definition(Definition::DK_CONSTANT, sym, type, /*pos=*/NULL);
    // enable this if the constants should not be redefined
    // def->set_flag(Definition::DEF_NO_SHADOW);

    def->set_flag(Definition::DEF_IS_PREDEFINED);
    def->set_constant_value(val);

    return def;
}

// Enter all predefined constants into the current scope.
void NT_analysis::enter_builtin_constants()
{
    Value_factory *val_fac  = m_module.get_value_factory();
    IType const *const_bool = m_tc.decorate_type(m_tc.bool_type, IType::MK_CONST);

    enter_builtin_constant(
        m_st->get_predefined_symbol(ISymbol::SYM_CNST_FALSE),
        const_bool, val_fac->create_bool(false));
    enter_builtin_constant(
        m_st->get_predefined_symbol(ISymbol::SYM_CNST_TRUE),
        const_bool, val_fac->create_bool(true));
}

// Enter builtin annotations for stdlib modules.
void NT_analysis::enter_builtin_annotations()
{
    // the intrinsic() annotation: assigns a semantic to a declaration
    {
        ISymbol const        *sym  = m_st->get_symbol("intrinsic");
        IType_function const *type = m_tc.create_function(NULL, Type_cache::Function_parameters());

        Definition *def = m_def_tab->enter_definition(Definition::DK_ANNOTATION, sym, type, NULL);
        def->set_semantic(Definition::DS_INTRINSIC_ANNOTATION);
    }

    // the throws() annotation: marks a function that can thrown an exception
    {
        ISymbol const        *sym  = m_st->get_symbol("throws");
        IType_function const *type = m_tc.create_function(NULL, Type_cache::Function_parameters());

        Definition *def = m_def_tab->enter_definition(Definition::DK_ANNOTATION, sym, type, NULL);
        def->set_semantic(Definition::DS_THROWS_ANNOTATION);
    }

    // the since(int major, int minor) annotation: assigns a MDL version to a declaration
    {
        ISymbol const *sym_major = m_st->get_symbol("major");
        ISymbol const *sym_minor = m_st->get_symbol("minor");

        IType_factory::Function_parameter params[2] = {
            { m_tc.int_type, sym_major },
            { m_tc.int_type, sym_minor },
        };

        ISymbol const        *sym  = m_st->get_symbol("since");
        IType_function const *type = m_tc.create_function(NULL, params);

        Definition *def = m_def_tab->enter_definition(Definition::DK_ANNOTATION, sym, type, NULL);
        def->set_semantic(Definition::DS_SINCE_ANNOTATION);
    }

    // the removed(int major, int minor) annotation: assigns a MDL version to a declaration
    {
        ISymbol const *sym_major = m_st->get_symbol("major");
        ISymbol const *sym_minor = m_st->get_symbol("minor");

        IType_factory::Function_parameter params[2] = {
            { m_tc.int_type, sym_major },
            { m_tc.int_type, sym_minor },
        };

        ISymbol const        *sym  = m_st->get_symbol("removed");
        IType_function const *type = m_tc.create_function(NULL, params);

        Definition *def = m_def_tab->enter_definition(Definition::DK_ANNOTATION, sym, type, NULL);
        def->set_semantic(Definition::DS_REMOVED_ANNOTATION);
    }

    // the const_expr() annotation: marks a const_expr function
    {
        ISymbol const        *sym  = m_st->get_symbol("const_expr");
        IType_function const *type = m_tc.create_function(NULL, Type_cache::Function_parameters());

        Definition *def = m_def_tab->enter_definition(Definition::DK_ANNOTATION, sym, type, NULL);
        def->set_semantic(Definition::DS_CONST_EXPR_ANNOTATION);
    }

    // the derivable() annotation: marks a parameter or a return type as derivable
    {
        ISymbol const        *sym  = m_st->get_symbol("derivable");
        IType_function const *type = m_tc.create_function(NULL, Type_cache::Function_parameters());

        Definition *def = m_def_tab->enter_definition(Definition::DK_ANNOTATION, sym, type, NULL);
        def->set_semantic(Definition::DS_DERIVABLE_ANNOTATION);
    }

    // the experimental() annotation: assigns a experimental MDL version to a declaration
    {
        ISymbol const        *sym = m_st->get_symbol("experimental");
        IType_function const *type = m_tc.create_function(
            NULL, Array_ref<IType_factory::Function_parameter>());

        Definition *def = m_def_tab->enter_definition(Definition::DK_ANNOTATION, sym, type, NULL);
        def->set_semantic(Definition::DS_EXPERIMENTAL_ANNOTATION);
    }

    // the literal_param() annotation: marks teh first parameter of a function accepting literals
    // only (const_expr)
    {
        ISymbol const        *sym = m_st->get_symbol("literal_param");
        IType_function const *type = m_tc.create_function(
            NULL, Array_ref<IType_factory::Function_parameter>());

        Definition *def = m_def_tab->enter_definition(Definition::DK_ANNOTATION, sym, type, NULL);
        def->set_semantic(Definition::DS_LITERAL_PARAM_ANNOTATION);
    }
}

// Enter builtin annotations for native modules.
void NT_analysis::enter_native_annotations()
{
    // the native() annotation: marks a native function
    {
        ISymbol const        *sym  = m_st->get_symbol("native");
        IType_function const *type = m_tc.create_function(NULL, Type_cache::Function_parameters());

        Definition *def = m_def_tab->enter_definition(Definition::DK_ANNOTATION, sym, type, NULL);
        def->set_semantic(Definition::DS_NATIVE_ANNOTATION);
    }
}

// Create an exported constant declaration for a given value and add it to the current module.
void NT_analysis::create_exported_decl(ISymbol const *sym, IValue const *val, int line)
{
    Name_factory &name_fact = *m_module.get_name_factory();

    // add it one line AFTER the last declaration
    if (line == 0) {
        int n_decls = m_module.get_declaration_count();
        if (n_decls > 0) {
            IDeclaration const *last_decl = m_module.get_declaration(n_decls - 1);
            line = last_decl->access_position().get_end_line();
        }
        ++line;
    }

#define DECL_POS line, 1, line, 1

    IType const *type = val->get_type();

    Scope const   *scope  = m_def_tab->get_type_scope(type);
    ISymbol const *tp_sym = scope->get_scope_name();

    ISimple_name const *tname   = name_fact.create_simple_name(tp_sym, DECL_POS);
    IQualified_name    *qname   = name_fact.create_qualified_name(DECL_POS);
    IType_name         *tp_name = name_fact.create_type_name(qname, DECL_POS);
    qname->add_component(tname);

    Expression_factory &expr_fact = *m_module.get_expression_factory();
    IExpression_literal const *expr = expr_fact.create_literal(val, DECL_POS);

    Declaration_factory &decl_fact = *m_module.get_declaration_factory();
    IDeclaration_constant *decl = decl_fact.create_constant(tp_name, /*exported=*/true, DECL_POS);

    ISimple_name const *c_name = name_fact.create_simple_name(sym, DECL_POS);
    decl->add_constant(c_name, expr);

    m_module.insert_constant_declaration(decl);

#undef DECL_POS
}

// Enter builtin constants for stdlib modules.
void NT_analysis::enter_stdlib_constants()
{
    char const *mod_name = m_module.get_name();

    if (strcmp(mod_name, "::limits") == 0) {
        Value_factory *val_fac = m_module.get_value_factory();
        float         f;
        double        d;

        f = m_compiler->get_compiler_float_option(&m_ctx, MDL::option_limits_float_min, FLT_MIN);
        create_exported_decl(m_st->get_symbol("FLOAT_MIN"), val_fac->create_float(f), 0);

        f = m_compiler->get_compiler_float_option(&m_ctx, MDL::option_limits_float_max, FLT_MAX);
        create_exported_decl(m_st->get_symbol("FLOAT_MAX"), val_fac->create_float(f), 0);

        d = m_compiler->get_compiler_double_option(&m_ctx, MDL::option_limits_double_min, DBL_MIN);
        create_exported_decl(m_st->get_symbol("DOUBLE_MIN"), val_fac->create_double(d), 0);

        d = m_compiler->get_compiler_double_option(&m_ctx, MDL::option_limits_double_max, DBL_MAX);
        create_exported_decl(m_st->get_symbol("DOUBLE_MAX"), val_fac->create_double(d), 0);
    } else if (strcmp(mod_name, "::state") == 0) {
        Value_factory *val_fac = m_module.get_value_factory();
        int           i;

        i = m_compiler->get_compiler_int_option(&m_ctx, MDL::option_state_wavelength_base_max, 1);
        create_exported_decl(m_st->get_symbol("WAVELENGTH_BASE_MAX"), val_fac->create_int(i), 1);
    }
}

// Check if the given type has a const constructor.
bool NT_analysis::has_const_default_constructor(IType const *type) const
{
restart:
    switch (type->get_kind()) {
    case IType::TK_ALIAS:
        type = type->skip_type_alias();
        goto restart;
    case IType::TK_BOOL:
    case IType::TK_INT:
    case IType::TK_ENUM:
    case IType::TK_FLOAT:
    case IType::TK_DOUBLE:
    case IType::TK_STRING:
    case IType::TK_VECTOR:
    case IType::TK_MATRIX:
    case IType::TK_COLOR:
        // all atomic, enum, vector, and matrix types ones have
        return true;
    case IType::TK_LIGHT_PROFILE:
    case IType::TK_TEXTURE:
    case IType::TK_BSDF_MEASUREMENT:
    case IType::TK_BSDF:
    case IType::TK_HAIR_BSDF:
    case IType::TK_EDF:
    case IType::TK_VDF:
        // although the reference types HAVE a default constructor,
        // the spec said these are not allowed in const expressions
        return false;
    case IType::TK_ARRAY:
        {
            // an array type has a const constructor IF its element type has
            IType_array const *a_type = cast<IType_array>(type);
            type = a_type->get_element_type();
            goto restart;
        }
    case IType::TK_FUNCTION:
        return false;
    case IType::TK_STRUCT:
        // check those
        break;
    case IType::TK_INCOMPLETE:
        MDL_ASSERT(!"incomplete type occured unexpected");
        return false;
    case IType::TK_ERROR:
        // to suppress further errors say true here
        return true;
    }
    for (Definition const *c_def = m_module.get_first_constructor(type);
         c_def != NULL;
         c_def = m_module.get_next_constructor(c_def))
    {
        if (c_def->has_flag(Definition::DEF_IS_CONST_CONSTRUCTOR))
            return true;
    }
    return false;
}

namespace {

/// Helper class to handle the renaming of struct fields to constructor arguments.
class Parameter_modifier : public IClone_modifier {
    typedef ptr_hash_map<IDefinition const, Definition *>::Type Param_map;
public:
    IExpression_reference *clone_expr_reference(
        IExpression_reference const *ref) MDL_FINAL
    {
        IType_name const *name = m_module.clone_name(ref->get_name(), NULL);
        IExpression_reference *res = m_fact->create_reference(name);
        if (ref->is_array_constructor()) {
            res->set_array_constructor();
            return res;
        }

        IDefinition const *def = ref->get_definition();
        Param_map::const_iterator it = m_param_map.find(def);
        if (it != m_param_map.end())
            def = it->second;
        res->set_definition(def);
        return res;
    }

    /// Clone a call expression.
    ///
    /// \param call   the expression to clone
    IExpression *clone_expr_call(IExpression_call const *c_expr) MDL_FINAL
    {
        // just clone it
        IExpression const *ref = m_module.clone_expr(c_expr->get_reference(), this);
        IExpression_call *call = m_fact->create_call(ref);

        for (int i = 0, n = c_expr->get_argument_count(); i < n; ++i) {
            IArgument const *arg = m_module.clone_arg(c_expr->get_argument(i), this);
            call->add_argument(arg);
        }
        return call;
    }

    /// Clone a literal.
    ///
    /// \param lit  the literal to clone
    IExpression *clone_literal(IExpression_literal const *lit) MDL_FINAL
    {
        // just clone it
        return m_module.clone_expr(lit, NULL);
    }

    /// Clone a qualified name.
    ///
    /// \param name  the name to clone
    IQualified_name *clone_name(IQualified_name const *qname) MDL_FINAL
    {
        return m_module.clone_name(qname, NULL);
    }

    /// Add a new mapping from a field definition to a parameter definition.
    ///
    /// \param field_def  the field definition
    /// \param param_def  the parameter definition
    void add_mapping(IDefinition const *field_def, Definition *param_def)
    {
        m_param_map[field_def] = param_def;
    }

    explicit Parameter_modifier(Module &mod)
    : m_module(mod)
    , m_fact(mod.get_expression_factory())
    , m_param_map(0, Param_map::hasher(), Param_map::key_equal(), mod.get_allocator())
    {
    }

private:
    /// The module.
    Module &m_module;

    /// The expression factory
    Expression_factory *m_fact;

    /// The parameter map.
    Param_map m_param_map;
};

}  // anon namespace

// Create the default constructors and operators for a given struct type.
void NT_analysis::create_default_members(
    IType_struct const             *s_type,
    ISymbol const                  *sym,
    Definition const               *def,
    IDeclaration_type_struct const *struct_decl)
{
    // enter the type scope for constructors
    Scope *type_scope = m_def_tab->get_type_scope(s_type);
    Definition_table::Scope_transition transition(*m_def_tab, type_scope);

    // create the explicit element wise constructor
    {
        int n = s_type->get_field_count();

        VLA<IType_factory::Function_parameter> params(get_allocator(), n);
        VLA<IExpression const *>               inits(get_allocator(), n);

        Definition *c_def = get_error_definition();

        if (n > 0)
            m_module.allocate_initializers(c_def, n);

        bool need_default_constructor = false;
        bool default_is_const         = true;
        bool is_const_constructor     = true;
        bool need_default_argument    = false;

        Scope *constructor_scope;
        {
            Definition_table::Scope_enter constr_scope(*m_def_tab, c_def);
            constructor_scope = m_def_tab->get_curr_scope();

            Parameter_modifier param_modifier(m_module);

            // add a function parameter for every field
            for (int i = 0; i < n; ++i) {
                IType const   *f_type;
                ISymbol const *f_sym;

                s_type->get_field(i, f_type, f_sym);

                params[i].p_type = f_type;
                params[i].p_sym  = f_sym;

                ISimple_name const *f_name = struct_decl->get_field_name(i);

                Definition *p_def = m_def_tab->enter_definition(
                    Definition::DK_PARAMETER, f_sym, f_type, &f_name->access_position());
                p_def->set_parameter_index(i);

                // compiler generated, so they are used ...
                p_def->set_flag(Definition::DEF_IS_USED);

                IDefinition const *f_def = f_name->get_definition();

                param_modifier.add_mapping(f_def, p_def);

                inits[i] = NULL;
                if (IExpression const *init = struct_decl->get_field_init(i)) {
                    need_default_argument = true;

                    init = m_module.clone_expr(init, &param_modifier);
                    inits[i] = init;
                    bool is_invalid = false;
                    if (is_const_constructor && !is_const_expression(init, is_invalid)) {
                        is_const_constructor = false;
                    }
                } else {
                    need_default_constructor = true;

                    if (!has_const_default_constructor(f_type)) {
                        // this field has no constant default constructor,
                        // so the default constructor will be NOT constant
                        default_is_const = false;
                    }

                    if (need_default_argument) {
                        ISimple_name const *f_name = struct_decl->get_field_name(i);
                        error(
                            MISSING_DEFAULT_FIELD_INITIALIZER,
                            f_name->access_position(),
                            Error_params(*this).add(f_name->get_symbol()).add(sym));

                        if (is_const_constructor) {
                            // constant constructors cannot require an argument
                            is_const_constructor = false;
                        }
                    }
                }
            }
        }
        IType_function const *func = m_tc.create_function(s_type, params);
        c_def = m_def_tab->enter_definition(
            Definition::DK_CONSTRUCTOR, sym, func, def->get_position());
        c_def->set_own_scope(constructor_scope);

        // set default initializers if any
        if (need_default_argument) {
            m_module.allocate_initializers(c_def, n);
            for (int i = 0; i < n; ++i) {
                if (IExpression const *init = inits[i])
                    c_def->set_default_param_initializer(i, init);
            }
        }

        if (is_const_constructor)
            c_def->set_flag(Definition::DEF_IS_CONST_CONSTRUCTOR);
        c_def->set_flag(Definition::DEF_IS_EXPLICIT);
        c_def->set_flag(Definition::DEF_IS_COMPILER_GEN);
        c_def->set_semantic(IDefinition::DS_ELEM_CONSTRUCTOR);
        c_def->set_declaration(struct_decl);

        if (def->has_flag(Definition::DEF_IS_EXPORTED)) {
            // copy the exported flag but do not export this definition, constructors
            // are handled using a different mechanism
            c_def->set_flag(Definition::DEF_IS_EXPORTED);
        }

        if (need_default_constructor) {
            // create the (implicit) default constructor
            IType_function const *func = m_tc.create_function(
                s_type, Type_cache::Function_parameters());

            Definition *c_def = m_def_tab->enter_definition(
            Definition::DK_CONSTRUCTOR, sym, func, def->get_position());

            if (default_is_const)
                c_def->set_flag(Definition::DEF_IS_CONST_CONSTRUCTOR);
            c_def->set_flag(Definition::DEF_IS_COMPILER_GEN);
            c_def->set_semantic(IDefinition::DS_DEFAULT_STRUCT_CONSTRUCTOR);
            c_def->set_declaration(struct_decl);

            if (def->has_flag(Definition::DEF_IS_EXPORTED)) {
                // copy the exported flag but do not export this definition, constructors
                // are handled using a different mechanism
                c_def->set_flag(Definition::DEF_IS_EXPORTED);
            }
        }
    }
    // create the (implicit) copy constructor
    {
        IType_factory::Function_parameter param[1];

        param[0].p_type = s_type;
        param[0].p_sym  = m_st->get_symbol("other");

        IType_function const *func = m_tc.create_function(s_type, param);
        Definition *c_def = m_def_tab->enter_definition(
            Definition::DK_CONSTRUCTOR, sym, func, def->get_position());

        c_def->set_flag(Definition::DEF_IS_COMPILER_GEN);
        c_def->set_semantic(IDefinition::DS_COPY_CONSTRUCTOR);
        c_def->set_declaration(struct_decl);

        if (def->has_flag(Definition::DEF_IS_EXPORTED)) {
            // copy the exported flag but do not export this definition, constructors
            // are handled using a different mechanism
            c_def->set_flag(Definition::DEF_IS_EXPORTED);
        }
    }
    // create the assignment operator=(struct x, struct y)
    {
        IType_factory::Function_parameter params[2];

        params[0].p_type = s_type;
        params[0].p_sym  = m_st->get_symbol("x");

        params[1].p_type = s_type;
        params[1].p_sym  = m_st->get_symbol("y");

        ISymbol const        *op_sym = m_st->get_operator_symbol(IExpression::OK_ASSIGN);
        IType_function const *func   = m_tc.create_function(s_type, params);
        Definition           *op_def = m_def_tab->enter_definition(
            Definition::DK_OPERATOR, op_sym, func, def->get_position());

        op_def->set_flag(Definition::DEF_IS_COMPILER_GEN);
        op_def->set_flag(Definition::DEF_OP_LVALUE);
        op_def->set_semantic(operator_to_semantic(IExpression::OK_ASSIGN));

        if (def->has_flag(Definition::DEF_IS_EXPORTED)) {
            // copy the exported flag but do not export this definition, member-operators
            // are handled using a different mechanism
            op_def->set_flag(Definition::DEF_IS_EXPORTED);
        }
    }
}

// Create the default operators for a given enum type.
void NT_analysis::create_default_operators(
    IType_enum const       *e_type,
    ISymbol const          *sym,
    Definition const       *def,
    IDeclaration_type_enum *enum_decl)
{
    // enter the type scope for constructors
    Scope *type_scope = m_def_tab->get_type_scope(e_type);
    Definition_table::Scope_transition transition(*m_def_tab, type_scope);

    // create the assignment operator=(enum x, enum y)
    {
        IType_factory::Function_parameter params[2];

        params[0].p_type = e_type;
        params[0].p_sym  = m_st->get_symbol("x");

        params[1].p_type = e_type;
        params[1].p_sym  = m_st->get_symbol("y");

        ISymbol const        *op_sym = m_st->get_operator_symbol(IExpression::OK_ASSIGN);
        IType_function const *func   = m_tc.create_function(e_type, params);
        Definition           *op_def = m_def_tab->enter_definition(
            Definition::DK_OPERATOR, op_sym, func, def->get_position());

        op_def->set_flag(Definition::DEF_IS_COMPILER_GEN);
        op_def->set_flag(Definition::DEF_OP_LVALUE);
        op_def->set_semantic(operator_to_semantic(IExpression::OK_ASSIGN));

        if (def->has_flag(Definition::DEF_IS_EXPORTED)) {
            // copy the exported flag but do not export this definition, member-operators
            // are handled using a different mechanism
            op_def->set_flag(Definition::DEF_IS_EXPORTED);
        }
    }
    // create the operator int(enum x)
    {
        IType_factory::Function_parameter param;

        param.p_type = e_type;
        param.p_sym  = m_st->get_symbol("x");

        IType_function const *func = m_tc.create_function(m_tc.int_type, param);

        // mark it as constructor for now, only constructors are const folded by default
        Definition *c_def = m_def_tab->enter_definition(
            Definition::DK_CONSTRUCTOR,
            m_st->get_predefined_symbol(ISymbol::SYM_TYPE_INT),
            func,
            def->get_position());
        c_def->set_flag(Definition::DEF_IS_COMPILER_GEN);

        c_def->set_semantic(IDefinition::DS_CONV_OPERATOR);
        c_def->set_declaration(enum_decl);

        if (def->has_flag(Definition::DEF_IS_EXPORTED)) {
            // copy the exported flag but do not export this definition, constructors
            // are handled using a different mechanism
            c_def->set_flag(Definition::DEF_IS_EXPORTED);
        }
    }
}

// Create the default constructors and operators for a given enum type.
void NT_analysis::create_default_constructors(
    IType_enum const       *e_type,
    ISymbol const          *sym,
    Definition const       *def,
    IDeclaration_type_enum *enum_decl)
{
    // enter the type scope for constructors
    Scope *type_scope = m_def_tab->get_type_scope(e_type);
    Definition_table::Scope_transition transition(*m_def_tab, type_scope);

    // create the (implicit) default/copy constructor
    {
        IType_factory::Function_parameter param;

        param.p_type = e_type;
        param.p_sym = m_st->get_symbol("v");

        IType_function const *func = m_tc.create_function(e_type, param);

        Definition *c_def = m_def_tab->enter_definition(
            Definition::DK_CONSTRUCTOR, sym, func, def->get_position());
        c_def->set_flag(Definition::DEF_IS_COMPILER_GEN);

        // add the default argument: enums are initialized with their first value
        IValue const *init_v = m_module.get_value_factory()->create_enum(e_type, /*index=*/0);
        IExpression *init = m_module.create_literal(init_v, NULL);

        m_module.allocate_initializers(c_def, 1);
        c_def->set_default_param_initializer(0, init);
        c_def->set_semantic(IDefinition::DS_COPY_CONSTRUCTOR);
        c_def->set_declaration(enum_decl);

        if (def->has_flag(Definition::DEF_IS_EXPORTED)) {
            // copy the exported flag but do not export this definition, constructors
            // are handled using a different mechanism
            c_def->set_flag(Definition::DEF_IS_EXPORTED);
        }
    }
}

// Check archive dependencies on a newly imported module.
void NT_analysis::check_imported_module_dependencies(
    Module const   *imp_mod,
    Position const &pos)
{
    Module::Archive_version const *arc_ver = m_module.get_owner_archive_version();

    if (arc_ver == NULL)
        return;

    if (Module::Archive_version const *a_ver = imp_mod->get_owner_archive_version()) {
        char const *imp_archive = a_ver->get_name();
        bool       found        = false;

        if (strcmp(imp_archive, arc_ver->get_name()) == 0) {
            // from out archive, no need to check
            found = true;
        } else {
            for (size_t i = 0, n = m_module.get_archive_dependencies_count(); i < n; ++i) {
                Module::Archive_version const *dep = m_module.get_archive_dependency(i);

                if (strcmp(imp_archive, dep->get_name()) == 0) {
                    Semantic_version const &imp_ver = a_ver->get_version();
                    Semantic_version const &dep_ver = dep->get_version();

                    if (!(dep_ver <= imp_ver)) {
                        // import from an old archive
                        warning(
                            IMPORT_FROM_OLD_ARCHIVE_VERSION,
                            pos,
                            Error_params(*this)
                                .add(&dep_ver)
                                .add(&imp_ver));
                    }
                    found = true;
                }
            }
        }
        if (!found) {
            // dependency is missing inside this archive
            char const *arc_name = arc_ver->get_name();
            warning(
                ARCHIVE_DEPENDENCY_MISSING,
                pos,
                Error_params(*this)
                    .add(imp_mod->get_name())
                    .add(imp_archive)
                    .add(arc_name));
        }
    }
}

// Find and load a module to import.
Module const *NT_analysis::load_module_to_import(
    IQualified_name const *rel_name,
    bool                  ignore_last)
{
    bool is_absolute = rel_name->is_absolute();

    string import_name(is_absolute ? "::" : "", m_builder.get_allocator());

    size_t n = rel_name->get_component_count() - (ignore_last ? 1 : 0);
    for (size_t i = 0; i < n; ++i) {
        // add a package separator, if the name doesn't already end with a separator
        // (due to an absolute alias name)
        if (i > 0 && !(!import_name.empty() && import_name[import_name.size() - 1] == ':'))
            import_name += "::";

        ISimple_name const *sname = rel_name->get_component(i);
        ISymbol const      *sym   = sname->get_symbol();

        // handle alias names
        if (Definition const *def = m_def_tab->get_namespace_alias(sym)) {
            ISymbol const *ns_sym = sym;
            sym = def->get_namespace();

            // is this an alias referring to an absolute path?
            if (sym->get_name()[0] == ':') {
                if ((is_absolute || i > 0)) {
                    error(ABSOLUTE_ALIAS_NOT_AT_BEGINNING,
                        rel_name->access_position(),
                        Error_params(*this)
                            .add(ns_sym));
                    import_name.clear();
                }
                is_absolute = true;
            }
        }

        import_name += sym->get_name();
    }

    bool is_weak_16 = false;
    if (m_module.get_mdl_version() >= IMDL::MDL_VERSION_1_6) {
        // from MDL 1.6 weak imports do not exists
        if (!is_absolute && import_name[0] != '.') {
            is_weak_16 = true;
            // previous weak imports are relative now
            import_name = ".::" + import_name;
        }
    }

    IMDL_foreign_module_translator *translator = is_absolute ?
        m_compiler->is_foreign_module(import_name.c_str()) : NULL;

    if (translator == NULL) {
        Messages_impl messages(get_allocator(), m_module.get_filename());
        File_resolver resolver(
            *m_compiler,
            m_module_cache,
            m_compiler->get_external_resolver(),
            m_compiler->get_search_path(),
            m_compiler->get_search_path_lock(),
            messages,
            m_ctx.get_front_path());

        // let the resolver find the absolute name
        mi::base::Handle<IMDL_import_result> result(m_compiler->resolve_import(
            resolver,
            import_name.c_str(),
            &m_module,
            &rel_name->access_position()));

        // copy messages
        copy_resolver_messages_to_module(messages, /*is_resource=*/ false);

        if (is_weak_16 &&
            !result.is_valid_interface() &&
            messages.get_error_message_count() == 1)
        {
            // resolving a formally weak reference failed, check if it is absolute
            mi::base::Handle<IMDL_import_result> result(m_compiler->resolve_import(
                resolver,
                import_name.c_str() + 1,
                &m_module,
                &rel_name->access_position()));
            if (result.is_valid_interface()) {
                // .. and add a note
                add_note(
                    POSSIBLE_ABSOLUTE_IMPORT,
                    rel_name->access_position(),
                    Error_params(*this).add(import_name.c_str() + 1));
            }
        }

        if (!result.is_valid_interface()) {
            // name could not be resolved
            return NULL;
        }
        import_name = result->get_absolute_name();
    }

    char const *abs_name = import_name.c_str();

    // check if we have this module already in other import table
    bool direct = false;
    Module const *imp_mod = m_module.find_imported_module(abs_name, direct);
    if (imp_mod != NULL) {
        // reference count is not increased by find_imported_mode(), do it here
        imp_mod->retain();
        if (!direct) {
            // increase the import entry count here, it will be dropped at and of compilation
            imp_mod->restore_import_entries(NULL);

            bool first_import = false;
            m_module.register_import(imp_mod, &first_import);

            if (first_import) {
                check_imported_module_dependencies(imp_mod, rel_name->access_position());
            }
        }
        return imp_mod;
    }

    Imported_module_cache cache(m_module, m_module_cache);

    {
        // Create a new context here: we don't want the compilation errors to be appended
        // to the current context.
        mi::base::Handle<Thread_context> ctx(
            m_compiler->create_thread_context(*this, m_ctx.get_front_path()));

        if (translator != NULL) {
            // compile this foreign module
            imp_mod = m_compiler->compile_foreign_module(*translator, *ctx.get(), abs_name, &cache);
        } else {
            // compile this module
            imp_mod = m_compiler->compile_module(*ctx.get(), abs_name, &cache);
        }
    }

    if (imp_mod == NULL) {
        mi::base::Handle<IModule const> import(cache.lookup(abs_name, NULL));
        if (import.is_valid_interface()) {
            if (!import->is_analyzed()) {
                // Found a non-analyzed module, this could only happen
                // in a loop import
                error(
                    IMPORT_LOOP,
                    rel_name->access_position(),
                    Error_params(*this).add(rel_name, ignore_last));
            }
        }
        return NULL;
    } else {
        bool restored = imp_mod->restore_import_entries(&cache);

        if (!restored) {
            // really bad, we could not restored the imports, this must be
            // an error at higher level.
           imp_mod->drop_import_entries();
           imp_mod->release();
           return NULL;
        }
    }

    // Register this module.
    size_t imp_mod_idx = m_module.register_import(imp_mod);

    // might contain errors so copy the error messages if any
    Messages const &imp_msg = imp_mod->access_messages();
    size_t cnt = imp_msg.get_error_message_count();
    if (cnt > 0) {
        error(
            ERRONEOUS_IMPORT,
            rel_name->access_position(),
            Error_params(*this)
                .add(imp_mod->get_name()));

        size_t fname_id = get_file_id(m_module.access_messages_impl(), imp_mod_idx);
        for (size_t i = 0; i < cnt; ++i) {
            IMessage const *msg = imp_msg.get_error_message(i);
            add_imported_message(fname_id, msg);
        }
    }
    check_imported_module_dependencies(imp_mod, rel_name->access_position());

    return imp_mod;
}

// Check if the given imported definition is a re-export, if true, add its module
// to the current import table and return the import index of the owner module.
size_t NT_analysis::handle_reexported_entity(
    Definition const *imported,
    Module const     *from,
    size_t           from_idx)
{
    size_t import_idx = imported->get_original_import_idx();
    if (import_idx == 0) {
        // definition originated in the from module
        return from_idx;
    } else {
        // definition "imported" was imported itself
        Module::Import_entry const *from_entry = from->get_import_entry(import_idx);
        return m_module.register_import(from_entry->get_module());
    }
}

// Enter the relative scope starting at the current scope
// of an imported module given by the module name and a prefix skip.
Scope *NT_analysis::enter_import_scope(
    IQualified_name const *name_space,
    int                   prefix_skip,
    bool                  ignore_last)
{
    int ns_len = name_space->get_component_count() - (ignore_last ? 1 : 0);

    MDL_ASSERT(prefix_skip <= ns_len && "prefix skip to long in import");

    Scope *scope = m_def_tab->get_global_scope();

    for (int i = prefix_skip; i < ns_len; ++i) {
        ISimple_name const *name    = name_space->get_component(i);
        ISymbol const      *imp_sym = m_module.import_symbol(name->get_symbol());

        Scope *n = scope->find_named_subscope(imp_sym);
        if (n == NULL)
            n = m_def_tab->enter_named_scope(imp_sym);
        else
            m_def_tab->transition_to_scope(n);
        scope = n;
    }
    return scope;
}

// Import a qualified entity or a module.
void NT_analysis::import_qualified(
    IQualified_name const *rel_name,
    Position const        &err_pos)
{
    int count = rel_name->get_component_count();
    ISimple_name const *sname = rel_name->get_component(count - 1);

    int skip_dots = 0;
    for (; skip_dots < count; ++skip_dots) {
        ISymbol const *sym = rel_name->get_component(skip_dots)->get_symbol();
        size_t        id   = sym->get_id();
        if (id != ISymbol::SYM_DOT && id != ISymbol::SYM_DOTDOT)
            break;
    }
    count -= skip_dots;

    if (is_star_import(sname)) {
        if (count == 1) {
            // import * is forbidden
            error(
                FORBIDDED_GLOBAL_STAR,
                err_pos,
                Error_params(*this));
            return;
        }
        // import a module
        mi::base::Handle<const Module> imp_mod(
            load_module_to_import(rel_name, /*ignore_last=*/true));
        if (!imp_mod) {
            // errors already reported
            return;
        }
        // import it under the relative name
        import_all_definitions(imp_mod.get(), rel_name, skip_dots, err_pos);
    } else {
        // import a qualified entity
        mi::base::Handle<const Module> imp_mod;

        imp_mod = mi::base::make_handle(
            load_module_to_import(rel_name, /*ignore_last=*/true));
        if (!imp_mod) {
            // errors already reported
            return;
        }

        // import only one entity
        import_qualified_entity(rel_name, imp_mod.get(), skip_dots);
    }
}

// Import a definition from a module.
Definition const *NT_analysis::import_definition(
    Module const     *from,
    size_t           from_idx,
    Definition const *old_def,
    bool             is_exported,
    Position const   &err_pos)
{
    Definition const *imported = old_def;

    // start with the first one to preserve the order:
    // this is not strictly needed, just for cosmetics
    for (;;) {
        if (Definition const *prev = imported->get_prev_def())
            imported = prev;
        else
            break;
    }

    // check for double import
    ISymbol const *imp_sym = m_module.import_symbol(imported->get_sym());
    Scope         *curr    = m_def_tab->get_curr_scope();

    if (Definition const *old_def = curr->find_definition_in_scope(imp_sym)) {
        // there IS already a definition for this symbol, check if it's from
        // the same module
        if (m_module.get_original_owner_id(old_def) == from->get_original_owner_id(imported)) {
            // because we import always a whole set, we do not check for matching
            // id here
            return NULL;
        }
        // otherwise we cannot import it
        return old_def;
    }

    for (; imported != NULL; imported = imported->get_next_def()) {
        Definition const *def_def = imported->get_definite_definition();
        if (def_def != NULL && def_def != imported) {
            // ignore this, it is only a prototype
            continue;
        }
        if (!is_available_in_mdl(m_mdl_version, imported->get_version_flags())) {
            // this entity is not available in the current MDL language level
            continue;
        }
        if (! imported->has_flag(Definition::DEF_IS_EXPORTED))
            continue;

        // the imported definition might be an re-exported one, check this
        size_t import_idx = handle_reexported_entity(imported, from, from_idx);
        Definition *new_def = Analysis::import_definition(imported, import_idx, &err_pos);
        if (is_exported) {
            export_definition(new_def);
        }

        // copy deprecated messages if any
        if (imported->has_flag(Definition::DEF_IS_DEPRECATED)) {
            IValue_string const *msg = from->get_deprecated_message(imported);
            if (msg != NULL) {
                msg = cast<IValue_string>(m_module.get_value_factory()->import(msg));
                m_module.set_deprecated_message(new_def, msg);
            }
        }

        Definition::Kind kind = imported->get_kind();
        if (kind == Definition::DK_TYPE) {
            // we import a type/material, import the sub scope
            import_type_scope(
                imported, from, from_idx, new_def, /*is_exported=*/false, err_pos);
        }
    }
    // all went fine
    return NULL;
}

// Import a complete module.
void NT_analysis::import_all_definitions(
    Module const          *from,
    IQualified_name const *name_space,
    int                   prefix_skip,
    Position const        &err_pos)
{
    Definition_table const &def_tab   = from->get_definition_table();
    size_t                 from_idx   = m_module.register_import(from);

    MDL_ASSERT(prefix_skip < name_space->get_component_count() && "prefix skip to long in import");
    {
        Definition_table::Scope_transition transition(*m_def_tab, m_def_tab->get_global_scope());

        enter_import_scope(name_space, prefix_skip, /*ignore_last=*/true);

        // Note: only entities at global scope can be imported, so check for them only there
        Scope const *global_scope = def_tab.get_global_scope();

        import_scope_entities(
            global_scope, from, from_idx, /*is_exported=*/false, /*forced=*/false, err_pos);
    }
}

// Import a type scope.
void NT_analysis::import_type_scope(
    Definition const *imported,
    Module const     *from,
    size_t           from_idx,
    Definition       *new_def,
    bool             is_exported,
    Position const   &err_pos)
{
    // we import a type/material, import the sub scope
    IType const *orig_type = imported->get_type();
    if (is<IType_alias>(orig_type)) {
        // do nothing, as the alias type has no type scope
        return;
    }

    Scope const *orig_scope = from->get_definition_table().get_type_scope(orig_type);
    IType const *imp_type   = new_def->get_type();
    {
        Definition_table::Scope_enter enter(*m_def_tab, imp_type, new_def);
        import_scope_entities(
            orig_scope, from, from_idx, is_exported, /*forced=*/true, err_pos);
    }
}

// Import all entities of a scope (and possible sub-scopes) into the current scope.
void NT_analysis::import_scope_entities(
    Scope const    *imported_scope,
    Module const   *from,
    size_t         from_idx,
    bool           is_exported,
    bool           forced,
    Position const &err_pos)
{
    Definition const *imported;

    IAllocator *alloc = m_module.get_allocator();

    // restore entity definitions
    // First turn the list around, we must ensure that they are imported in the same
    // order as they are defined (needed for synonyms)
    Definition_vec defs(alloc);

    for (imported = imported_scope->get_first_definition_in_scope();
         imported != NULL;
         imported = imported->get_next_def_in_scope())
    {
        // we do not allow to export only parts of an overload set, so it it safe to check for
        // the exported flag just on the lat member of a set
        if (forced || imported->has_flag(Definition::DEF_IS_EXPORTED)) {
            // check if we can import def
            ISymbol const *imp_sym = m_module.import_symbol(imported->get_sym());
            Scope         *curr    = m_def_tab->get_curr_scope();

            if (Definition const *clash_def = curr->find_definition_in_scope(imp_sym)) {
                // there IS already a definition for this symbol, check if it's from
                // the same module
                if (m_module.get_original_owner_id(clash_def) ==
                    from->get_original_owner_id(imported)) {
                    // because we import always a whole set, we do not check for matching
                    // id here
                    continue;
                }
                // otherwise we cannot import it
                mi::base::Handle<IModule const> mod(m_module.get_owner_module(clash_def));
                char const *clash_name = mod->get_name();

                error(
                    IMPORT_SYMBOL_CLASH,
                    err_pos,
                    Error_params(*this)
                        .add(imp_sym)
                        .add_signature(clash_def)
                        .add(clash_name));
                continue;
            }
            defs.push_back(imported);
        }
    }

    for (size_t i = defs.size(); i > 0;) {
        for (imported = defs[--i]; imported != NULL; imported = imported->get_next_def()) {

            Definition const *def_def = imported->get_definite_definition();
            if (def_def != NULL && def_def != imported) {
                // ignore this, it is only a prototype
                continue;
            }
            if (!is_available_in_mdl(m_mdl_version, imported->get_version_flags())) {
                // this entity is not available in the current MDL language level
                continue;
            }

            // the imported definition might be an re-exported one (i.e. not originated
            // from the "from" module), handle this
            size_t import_idx = handle_reexported_entity(imported, from, from_idx);
            Definition *new_def = Analysis::import_definition(imported, import_idx, &err_pos);
            if (is_exported)
                export_definition(new_def);

            // copy deprecated messages if any
            if (imported->has_flag(Definition::DEF_IS_DEPRECATED)) {
                IValue_string const *msg = from->get_deprecated_message(imported);
                if (msg != NULL) {
                    msg = cast<IValue_string>(m_module.get_value_factory()->import(msg));
                    m_module.set_deprecated_message(new_def, msg);
                }
            }

            Definition::Kind kind = imported->get_kind();
            if (kind == Definition::DK_TYPE) {
                import_type_scope(
                    imported, from, from_idx, new_def, /*is_exported=*/false, err_pos);
            }
        }
    }
}

/// Check if at least one definition of an overload set is available
/// at the given MDL language level.
///
/// \param mdl_version  the current MDL language level
/// \param def          the overload set to check, maybe NULL
static bool some_available(unsigned mdl_version, Definition const *def)
{
    if (def == NULL)
        return false;
    for (Definition const *d = def; d != NULL; d = d->get_prev_def()) {
        if (is_available_in_mdl(mdl_version, d->get_version_flags())) {
            return true;
        }
    }
    return false;
}

// Import an entity from a module.
bool NT_analysis::import_entity(
    IQualified_name const *rel_name,
    Module const          *imp_mod,
    IQualified_name const *entity_name,
    bool                  is_exported)
{
    Definition_table const &imp_deftab = imp_mod->get_definition_table();
    size_t                 imp_idx     = m_module.get_import_index(imp_mod);

    Scope *imp_scope = imp_deftab.get_global_scope();

    size_t n = entity_name->get_component_count();
    if (n > 1) {
        for (size_t i = 0; i < n - 1; ++i) {
            ISimple_name const *identifier = entity_name->get_component(i);
            ISymbol const *imp_scope_sym = imp_mod->lookup_symbol(identifier->get_symbol());

            if (imp_scope_sym == NULL) {
                // this scope is unknown in the imported module
                error(
                    UNKNOWN_EXPORT,
                    entity_name->access_position(),
                    Error_params(*this)
                        .add(imp_mod->get_name())
                        .add(identifier->get_symbol()));
            }

            Scope *named_scope = imp_scope->find_named_subscope(imp_scope_sym);
            if (named_scope == NULL) {
                string tname = full_name(entity_name, /*only_scopy=*/true);
                error(
                    UNKNOWN_PACKAGE_NAME,
                    entity_name->access_position(),
                    Error_params(*this)
                        .add(imp_mod->get_name())
                        .add(tname.c_str()));
                return false;
            } else {
                imp_scope = named_scope;
            }
        }
    }
    ISimple_name const *identifier = entity_name->get_component(n - 1);
    ISymbol const *imp_sym = imp_mod->lookup_symbol(identifier->get_symbol());

    Definition const *def = imp_scope->find_definition_in_scope(imp_sym);
    if (!some_available(m_mdl_version, def)) {
        // the name is unknown in the imported module or filtered out
        error(
            UNKNOWN_EXPORT,
            entity_name->access_position(),
            Error_params(*this)
                .add(imp_mod->get_name())
                .add(identifier->get_symbol()));
        if (def != NULL) {
            add_note(
                USES_FUTURE_FEATURE,
                entity_name->access_position(),
                Error_params(*this)
                    .add(identifier->get_symbol()));
        }
        return false;
    }

    // ... importing enum values alone is forbidden
    if (def->get_kind() == IDefinition::DK_ENUM_VALUE) {
        string tname = full_name(entity_name, /*only_scopy=*/false);

        // We could have a more specific error here, but probably
        // this is a "logic" one: There is no "stand-alone" thing.
        error(
            ENT_NOT_EXPORTED,
            entity_name->access_position(),
            Error_params(*this)
                .add(tname.c_str())
                .add(imp_mod->get_name()));

        if (def->has_flag(Definition::DEF_IS_EXPORTED)) {
            IType_enum const *e_type = cast<IType_enum>(def->get_type());
            ISymbol const    *e_sym  = e_type->get_symbol();
            char const       *e_name = e_sym->get_name();

            char const *p = strrchr(e_name, ':');

            add_note(
                IMPORT_ENUM_TYPE,
                entity_name->access_position(),
                Error_params(*this)
                    .add(tname.c_str())
                    .add(p + 1));
        }
        return false;
    }

    if (!def->has_flag(Definition::DEF_IS_EXPORTED)) {
        // this entity is not exported from the imported module
        string tname = full_name(entity_name, /*only_scopy=*/false);
        error(
            ENT_NOT_EXPORTED,
            entity_name->access_position(),
            Error_params(*this)
                .add(tname.c_str())
                .add(imp_mod->get_name()));
        return false;
    }

    // if it is a enum type, its enum values are at the same scope ... must check for them
    Scope::Definition_list values(get_allocator());
    if (def->get_kind() == IDefinition::DK_TYPE) {
        if (IType_enum const *e_type = as<IType_enum>(def->get_type())) {
            imp_scope->collect_enum_values(e_type, values);
        }
    }

    // import into global scope
    {
        Definition_table::Scope_transition transition(*m_def_tab, m_def_tab->get_global_scope());
        Definition const *clash_def = import_definition(
            imp_mod, imp_idx, def, is_exported, entity_name->access_position());
        if (clash_def != NULL) {
            string tname = full_name(entity_name, /*only_scopy=*/false);

            mi::base::Handle<IModule const> mod(m_module.get_owner_module(clash_def));
            char const *clash_name = mod->get_name();

            error(
                IMPORT_SYMBOL_CLASH,
                entity_name->access_position(),
                Error_params(*this)
                    .add(tname.c_str())
                    .add_signature(clash_def)
                    .add(clash_name));
            return false;
        }

        bool ev_error = false;
        for (Scope::Definition_list::const_iterator it(values.begin()), end(values.end());
            it != end;
            ++it)
        {
            Definition const *ev_def = *it;

            Definition const *clash_def = import_definition(
                imp_mod, imp_idx, ev_def, is_exported, entity_name->access_position());
            if (clash_def != NULL) {
                string tname = full_name(entity_name, /*only_scopy=*/false);

                mi::base::Handle<IModule const> mod(m_module.get_owner_module(clash_def));
                char const *clash_name = mod->get_name();

                error(
                    IMPORT_ENUM_VALUE_SYMBOL_CLASH,
                    entity_name->access_position(),
                    Error_params(*this)
                        .add(tname.c_str())
                        .add_signature(ev_def)
                        .add_signature(clash_def)
                        .add(clash_name));
                ev_error = true;
            }
        }
        if (ev_error)
            return false;
    }

    // import into named scope
    {
        Definition_table::Scope_transition transition(*m_def_tab, m_def_tab->get_global_scope());

        // skip "." and ".."
        int rel_name_len = rel_name->get_component_count();
        int first = 0;
        for (; first < rel_name_len; ++first) {
            ISymbol const *sym = rel_name->get_component(first)->get_symbol();
            size_t        id   = sym->get_id();

            if (id != ISymbol::SYM_DOT && id != ISymbol::SYM_DOTDOT)
                break;
        }
        enter_import_scope(rel_name, first, /*ignore_last=*/false);

        // qualified imports are never exported
        Definition const *clash_def = import_definition(
            imp_mod, imp_idx, def, /*is_exported=*/false, entity_name->access_position());
        if (clash_def != NULL) {
            string tname = full_name(entity_name, /*only_scopy=*/false);

            mi::base::Handle<IModule const> mod(m_module.get_owner_module(clash_def));
            char const *clash_name = mod->get_name();

            error(
                IMPORT_SYMBOL_CLASH,
                entity_name->access_position(),
                Error_params(*this)
                .add(tname.c_str())
                .add_signature(clash_def)
                .add(clash_name));
            return false;
        }

        bool ev_error = false;
        for (Scope::Definition_list::const_iterator it(values.begin()), end(values.end());
            it != end;
            ++it)
        {
            Definition const *ev_def = *it;

            Definition const *clash_def = import_definition(
                imp_mod, imp_idx, ev_def, is_exported, entity_name->access_position());
            if (clash_def != NULL) {
                string tname = full_name(entity_name, /*only_scopy=*/false);

                mi::base::Handle<IModule const> mod(m_module.get_owner_module(clash_def));
                char const *clash_name = mod->get_name();

                error(
                    IMPORT_ENUM_VALUE_SYMBOL_CLASH,
                    entity_name->access_position(),
                    Error_params(*this)
                        .add(tname.c_str())
                        .add_signature(ev_def)
                        .add_signature(clash_def)
                        .add(clash_name));
                ev_error = true;
            }
        }
        if (ev_error)
            return false;
    }
    return true;
}

// Import a qualified entity from a module.
bool NT_analysis::import_qualified_entity(
    IQualified_name const *entity_name,
    Module const          *imp_mod,
    int                   prefix_skip)
{
    Definition_table const &imp_deftab = imp_mod->get_definition_table();
    size_t                 imp_idx     = m_module.get_import_index(imp_mod);

    Scope *imp_scope = imp_deftab.get_global_scope();

    size_t n = entity_name->get_component_count();
    MDL_ASSERT(n > 0 && "entity name is not qualified");

    ISimple_name const *identifier = entity_name->get_component(n - 1);
    ISymbol const      *imp_sym = imp_mod->lookup_symbol(identifier->get_symbol());

    // check, if the imported entity exists in the imported modules
    Definition const *def = imp_sym != NULL ? imp_scope->find_definition_in_scope(imp_sym) : NULL;
    if (!some_available(m_mdl_version, def)) {
        // the name is unknown in the imported module
        error(
            UNKNOWN_EXPORT,
            entity_name->access_position(),
            Error_params(*this)
                .add(imp_mod->get_name())
                .add(identifier->get_symbol()));
        if (def != NULL) {
            add_note(
                USES_FUTURE_FEATURE,
                entity_name->access_position(),
                Error_params(*this)
                    .add(identifier->get_symbol()));
        }
        return false;
    }

    // ... importing enum values alone is forbidden
    if (def->get_kind() == IDefinition::DK_ENUM_VALUE) {
        string tname = full_name(entity_name, /*only_scopy=*/false);

        // We could have a more specific error here, but probably
        // this is a "logic" one: There is no "stand-alone" thing.
        error(
            ENT_NOT_EXPORTED,
            entity_name->access_position(),
            Error_params(*this)
                .add(tname.c_str())
                .add(imp_mod->get_name()));

        if (def->has_flag(Definition::DEF_IS_EXPORTED)) {
            IType_enum const *e_type = cast<IType_enum>(def->get_type());
            ISymbol const    *e_sym  = e_type->get_symbol();
            char const       *e_name = e_sym->get_name();

            size_t pos  = tname.rfind("::");
            string name = tname.substr(0, pos + 2);

            char const *p = strrchr(e_name, ':');

            name += p + 1;

            add_note(
                IMPORT_ENUM_TYPE,
                entity_name->access_position(),
                Error_params(*this)
                    .add(tname.c_str())
                    .add(name.c_str()));
        }
        return false;
    }

    // ... and is exported there
    if (!def->has_flag(Definition::DEF_IS_EXPORTED)) {
        // this entity is not exported from the imported module
        string tname = full_name(entity_name, /*only_scopy=*/false);
        error(
            ENT_NOT_EXPORTED,
            entity_name->access_position(),
            Error_params(*this)
                .add(tname.c_str())
                .add(imp_mod->get_name()));
        return false;
    }

    // import into named scope
    {
        Definition_table::Scope_transition transition(*m_def_tab, m_def_tab->get_global_scope());

        enter_import_scope(entity_name, prefix_skip, /*ignore_last=*/true);

        // qualified imports are never exported
        Definition const *clash_def = import_definition(
            imp_mod, imp_idx, def, /*is_exported=*/false, entity_name->access_position());
        if (clash_def != NULL) {
            string tname = full_name(entity_name, /*only_scopy=*/false);

            mi::base::Handle<IModule const> mod(m_module.get_owner_module(clash_def));
            char const *clash_name = mod->get_name();

            error(
                IMPORT_SYMBOL_CLASH,
                entity_name->access_position(),
                Error_params(*this)
                .add(tname.c_str())
                .add_signature(clash_def)
                .add(clash_name));
            return false;
        }

        // if it is a enum type, its enum values are at the same scope ... must check for them
        if (def->get_kind() == IDefinition::DK_TYPE) {
            if (IType_enum const *e_type = as<IType_enum>(def->get_type())) {

                Scope::Definition_list values(get_allocator());
                imp_scope->collect_enum_values(e_type, values);

                bool ev_error = false;
                for (Scope::Definition_list::const_iterator it(values.begin()), end(values.end());
                    it != end;
                    ++it)
                {
                    Definition const *ev_def = *it;

                    // no need to check the export flag, the export of the type is enough
                    Definition const *clash_def = import_definition(
                        imp_mod, imp_idx, ev_def, /*is_exported=*/false,
                        entity_name->access_position());
                    if (clash_def != NULL) {
                        string tname = full_name(entity_name, /*only_scopy=*/false);

                        mi::base::Handle<IModule const> mod(m_module.get_owner_module(clash_def));
                        char const *clash_name = mod->get_name();

                        error(
                            IMPORT_ENUM_VALUE_SYMBOL_CLASH,
                            entity_name->access_position(),
                            Error_params(*this)
                                .add(tname.c_str())
                                .add_signature(ev_def)
                                .add_signature(clash_def)
                                .add(clash_name));
                        ev_error = true;
                    }
                }
                if (ev_error)
                    return false;
            }
        }
    }
    return true;
}

// Import all exported entities from a module.
void NT_analysis::import_all_entities(
    IQualified_name const *rel_name,
    Module const          *imp_mod,
    bool                  is_exported,
    Position const        &err_pos)
{
    Definition_table const &imp_deftab = imp_mod->get_definition_table();
    size_t                 imp_idx     = m_module.get_import_index(imp_mod);

    Scope *imp_scope = imp_deftab.get_global_scope();

    // import into global scope
    {
        Definition_table::Scope_transition transition(*m_def_tab, m_def_tab->get_global_scope());
        import_scope_entities(
            imp_scope, imp_mod, imp_idx, is_exported, /*forced=*/false, err_pos);
    }

    // import into named scope (never exported)
    {
        Definition_table::Scope_transition transition(*m_def_tab, m_def_tab->get_global_scope());

        // skip "." and ".."
        int rel_name_len = rel_name->get_component_count();
        int first = 0;
        for (; first < rel_name_len; ++first) {
            ISymbol const *sym = rel_name->get_component(first)->get_symbol();
            size_t        id   = sym->get_id();

            if (id != ISymbol::SYM_DOT && id != ISymbol::SYM_DOTDOT)
                break;
        }

        enter_import_scope(rel_name, first, /*ignore_last=*/false);

        import_scope_entities(
            imp_scope, imp_mod, imp_idx, /*is_exported=*/false, /*forced=*/false, err_pos);
    }
}

/// Check if the given type is allowed for a enable_if sub-expression.
///
/// \param type        the type to check
static bool is_allowed_enable_if_type(IType const *type)
{
restart:
    switch (type->get_kind()) {
    case IType::TK_ALIAS:
        {
            IType_alias const *a_type = cast<IType_alias>(type);
            type = a_type->get_aliased_type();
        }
        goto restart;
    case IType::TK_BOOL:
    case IType::TK_INT:
    case IType::TK_ENUM:
    case IType::TK_FLOAT:
    case IType::TK_DOUBLE:
    case IType::TK_STRING:
    case IType::TK_COLOR:
    case IType::TK_VECTOR:
    case IType::TK_MATRIX:
        // allowed
        return true;

    case IType::TK_ARRAY:
    case IType::TK_STRUCT:
        // any compound type of allowed element types is allowed
        {
            IType_compound const *c_tp = cast<IType_compound>(type);

            for (int i = 0, n = c_tp->get_compound_size(); i < n; ++i) {
                IType const *e_tp = c_tp->get_compound_type(i);

                if (!is_allowed_enable_if_type(e_tp))
                    return false;
            }
            return true;
        }

    case IType::TK_LIGHT_PROFILE:
    case IType::TK_TEXTURE:
    case IType::TK_BSDF_MEASUREMENT:
    case IType::TK_BSDF:
    case IType::TK_HAIR_BSDF:
    case IType::TK_EDF:
    case IType::TK_VDF:
    case IType::TK_FUNCTION:
        // forbidden
        return false;

    case IType::TK_INCOMPLETE:
        MDL_ASSERT(!"incomplete type occured unexpected");
        return true;

    case IType::TK_ERROR:
        // error already reported
        return true;
    }

    MDL_ASSERT(!"Unsupported type kind");
    return false;
}

/// Checks if the given type is allowed for function return types.
///
/// \param type        the type to check
/// \param is_std_mod  true if the current module is a standard module
///
/// \returns  the forbidden type or NULL if the type is ok
static IType const *has_forbidden_function_return_type(
    IType const *type,
    bool        is_std_mod)
{
restart:
    switch (type->get_kind()) {
    case IType::TK_ALIAS:
        {
            IType_alias const *a_type = cast<IType_alias>(type);
            type = a_type->get_aliased_type();
        }
        goto restart;
    case IType::TK_BOOL:
    case IType::TK_INT:
    case IType::TK_ENUM:
    case IType::TK_FLOAT:
    case IType::TK_DOUBLE:
    case IType::TK_STRING:
    case IType::TK_VECTOR:
    case IType::TK_MATRIX:
    case IType::TK_COLOR:
        return NULL;

    case IType::TK_LIGHT_PROFILE:
    case IType::TK_TEXTURE:
    case IType::TK_BSDF_MEASUREMENT:
        // texture, light profiles, and bsdf measurements are only allowed as
        // parameters of functions and materials
        return type;

    case IType::TK_INCOMPLETE:
        MDL_ASSERT(!"incomplete type occured unexpected");
        return NULL;

    case IType::TK_ERROR:
        // error already reported
        return NULL;

    case IType::TK_BSDF:
    case IType::TK_HAIR_BSDF:
    case IType::TK_EDF:
    case IType::TK_VDF:
        // reference types are allowed in std module functions, otherwise forbidden
        return is_std_mod ? NULL : type;

    case IType::TK_ARRAY:
        {
            IType_array const *a_type = cast<IType_array>(type);
            IType const       *e_type = a_type->get_element_type();
            // do a deep check
            return has_forbidden_function_return_type(e_type, is_std_mod);
        }
    case IType::TK_FUNCTION:
        // functions are forbidden (and impossible by syntax)
        return type;

    case IType::TK_STRUCT:
        {
            IType_struct const *s_type = cast<IType_struct>(type);
            if (s_type->get_predefined_id() != IType_struct::SID_USER) {
                // materials and its parts are forbidden
                return s_type;
            }
            // do a deep check
            for (int i = 0, n = s_type->get_field_count(); i < n; ++i) {
                ISymbol const *f_name;
                IType const   *f_type;

                s_type->get_field(i, f_type, f_name);
                IType const *bad_type = has_forbidden_function_return_type(f_type, is_std_mod);
                if (bad_type != NULL)
                    return bad_type;
            }
        }
        return NULL;
    }
    MDL_ASSERT(!"Unsupported type kind");
    return type;
}

/// Checks if the given type is allowed for variable types.
///
/// \param type  the type to check
///
/// \returns  the forbidden type or NULL if the type is ok
static IType const *has_forbidden_variable_type(IType const *type)
{
    // same restriction as return types for non-standard modules
    return has_forbidden_function_return_type(type, /*is_std_mod=*/false);
}

/// Checks if the given type is allowed for structure field types.
///
/// \param type  the type to check
static bool is_allowed_field_type(IType const *type)
{
    // same restriction as return types for non-standard modules
    return has_forbidden_function_return_type(type, /*is_std_mod=*/false) == NULL;
}

/// Checks if the given type is allowed for array element types.
///
/// \param type  the type to check
static bool is_allowed_array_type(IType const *type)
{
    for (;;) {
        switch (type->get_kind()) {
        case IType::TK_ALIAS:
            type = cast<IType_alias>(type)->get_aliased_type();
            continue;

        case IType::TK_BOOL:
        case IType::TK_INT:
        case IType::TK_ENUM:
        case IType::TK_FLOAT:
        case IType::TK_DOUBLE:
        case IType::TK_STRING:
            return true;

        case IType::TK_LIGHT_PROFILE:
            // arrays of light profiles are forbidden
            return false;

        case IType::TK_BSDF:
        case IType::TK_HAIR_BSDF:
        case IType::TK_EDF:
        case IType::TK_VDF:
        case IType::TK_VECTOR:
        case IType::TK_MATRIX:
            return true;

        case IType::TK_ARRAY:
            // nested arrays are forbidden
            return false;

        case IType::TK_COLOR:
            return true;

        case IType::TK_FUNCTION:
            // functions are forbidden (and impossible by syntax)
            return false;

        case IType::TK_STRUCT:
            {
                IType_struct const          *s_type = cast<IType_struct>(type);
                IType_struct::Predefined_id sid     = s_type->get_predefined_id();
                if (sid != IType_struct::SID_USER && sid != IType_struct::SID_MATERIAL) {
                    // material parts are forbidden, material itself is allowed
                    return false;
                }
                // deep check is not necessary here, because we do not allow the creation
                // of bad structs
            }
            return true;

        case IType::TK_TEXTURE:
            // arrays of textures are forbidden
            return false;

        case IType::TK_BSDF_MEASUREMENT:
            // arrays of bsdf measurements are forbidden
            return false;

        case IType::TK_INCOMPLETE:
            MDL_ASSERT(!"incomplete type occured unexpected");
            return true;

        case IType::TK_ERROR:
            // error was already reported
            return true;
        }
        MDL_ASSERT(!"Unsupported type kind");
        return false;
    }
}

/// Checks if the given type is allowed for function parameter types.
///
/// \param type            the type to check
/// \param allow_df        true *df types are allowed
/// \param allow_resource  true if resource types are allowed
///
/// \returns  the forbidden type or NULL if the type is ok
static IType const *has_forbidden_parameter_type(
    IType const *type,
    bool        allow_df,
    bool        allow_resource)
{
    for (;;) {
        switch (type->get_kind()) {
        case IType::TK_ALIAS:
            type = cast<IType_alias>(type)->get_aliased_type();
            continue;

        case IType::TK_BOOL:
        case IType::TK_INT:
        case IType::TK_ENUM:
        case IType::TK_FLOAT:
        case IType::TK_DOUBLE:
        case IType::TK_STRING:
            return NULL;

        case IType::TK_BSDF:
        case IType::TK_HAIR_BSDF:
        case IType::TK_EDF:
        case IType::TK_VDF:
            // only allowed in std library functions
            return allow_df ? NULL : type;

        case IType::TK_VECTOR:
        case IType::TK_MATRIX:
            return NULL;

        case IType::TK_ARRAY:
            {
                IType_array const *a_type = cast<IType_array>(type);
                IType const       *e_type = a_type->get_element_type();
                return has_forbidden_parameter_type(e_type, allow_df, allow_resource);
            }

        case IType::TK_COLOR:
            return NULL;

        case IType::TK_FUNCTION:
            // functions are forbidden (and impossible by syntax)
            return type;

        case IType::TK_STRUCT:
            {
                IType_struct const *s_type = cast<IType_struct>(type);
                if (s_type->get_predefined_id() != IType_struct::SID_USER) {
                    // materials and its parts are forbidden
                    return s_type;
                }
                // do a deep check
                for (int i = 0, n = s_type->get_field_count(); i < n; ++i) {
                    ISymbol const *f_name;
                    IType const   *f_type;

                    s_type->get_field(i, f_type, f_name);
                    IType const *bad_type =
                        has_forbidden_parameter_type(f_type, allow_df, allow_resource);
                    if (bad_type != NULL) {
                        return bad_type;
                    }
                }
            }
            return NULL;

        case IType::TK_TEXTURE:
        case IType::TK_BSDF_MEASUREMENT:
        case IType::TK_LIGHT_PROFILE:
            return allow_resource ? NULL : type;

        case IType::TK_INCOMPLETE:
            MDL_ASSERT(!"incomplete type occured unexpected");
            return NULL;

        case IType::TK_ERROR:
            // error was already reported
            return NULL;
        }
        MDL_ASSERT(!"Unsupported type kind");
        return type;
    }
}

/// Checks if the given type is allowed for function parameter types.
///
/// \param type         the type to check
/// \param is_std_mod   true if the current module is a standard module
///
/// \returns  the forbidden type or NULL if the type is ok
static IType const *has_forbidden_function_parameter_type(
    IType const *type,
    bool        is_std_mod)
{
    return has_forbidden_parameter_type(type, /*allow_df=*/is_std_mod, /*allow_resource=*/true);
}

/// Checks if the given type is allowed for annotation parameter types.
///
/// \param type         the type to check
///
/// \returns  the forbidden type or NULL if the type is ok
static IType const *has_forbidden_annotation_parameter_type(
    IType const *type)
{
    return has_forbidden_parameter_type(type, /*allow_df=*/false, /*allow_resource=*/false);
}

// Checks if the given type is allowed for material parameter types.
IType const *NT_analysis::has_forbidden_material_parameter_type(
    IType const *type)
{
    IType::Modifiers modifiers = type->get_type_modifiers();

    for (;;) {
        switch (type->get_kind()) {
        case IType::TK_ALIAS:
            type = cast<IType_alias>(type)->get_aliased_type();
            continue;

        case IType::TK_BOOL:
        case IType::TK_INT:
        case IType::TK_ENUM:
        case IType::TK_FLOAT:
        case IType::TK_DOUBLE:
        case IType::TK_STRING:
        case IType::TK_LIGHT_PROFILE:
            return NULL;

        case IType::TK_BSDF:
        case IType::TK_HAIR_BSDF:
        case IType::TK_EDF:
        case IType::TK_VDF:
            return type;

        case IType::TK_VECTOR:
        case IType::TK_MATRIX:
            return NULL;

        case IType::TK_ARRAY:
            {
                IType_array const *a_type = cast<IType_array>(type);
                IType const       *e_type = a_type->get_element_type();
                return has_forbidden_material_parameter_type(e_type);
            }

        case IType::TK_COLOR:
            return NULL;

        case IType::TK_FUNCTION:
            // functions are forbidden (and impossible by syntax)
            return type;

        case IType::TK_STRUCT:
            {
                IType_struct const          *s_type = cast<IType_struct>(type);
                IType_struct::Predefined_id sid     = s_type->get_predefined_id();
                if (sid != IType_struct::SID_USER) {
                    if (sid == IType_struct::SID_MATERIAL) {
                        if (modifiers & IType::MK_UNIFORM) {
                            // uniform material is not allowed, material type is inherent varying
                            return m_tc.decorate_type(type, IType::MK_UNIFORM);
                        }
                        return NULL;
                    } else {
                        // material parts are forbidden
                        return s_type;
                    }
                }
                // do a deep check
                for (int i = 0, n = s_type->get_field_count(); i < n; ++i) {
                    ISymbol const *f_name;
                    IType const   *f_type;

                    s_type->get_field(i, f_type, f_name);
                    IType const *bad_type = has_forbidden_material_parameter_type(f_type);
                    if (bad_type != NULL) {
                        return bad_type;
                    }
                }
            }
            return NULL;

        case IType::TK_TEXTURE:
        case IType::TK_BSDF_MEASUREMENT:
            return NULL;

        case IType::TK_INCOMPLETE:
            MDL_ASSERT(!"incomplete type occured unexpected");
            return NULL;

        case IType::TK_ERROR:
            // error was already reported
            return NULL;
        }
        MDL_ASSERT(!"Unsupported type kind");
        return type;
    }
}

// Checks if the given type is allowed for function return types.
bool NT_analysis::check_function_return_type(
    IType const    *type,
    Position const &pos,
    ISymbol const  *func_name,
    bool           is_std_mod)
{
    IType const *bad_type = has_forbidden_function_return_type(type, is_std_mod);
    if (bad_type != NULL) {
        error(
            FORBIDDEN_RETURN_TYPE,
            pos,
            Error_params(*this).add(type).add(func_name));
        if (bad_type->skip_type_alias() != type->skip_type_alias()) {
            add_note(
                TYPE_CONTAINS_FORBIDDEN_SUBTYPE,
                pos,
                Error_params(*this)
                    .add(type)
                    .add(bad_type));
        }
        return false;
    }
    return true;
}

// Check (and copy) default parameters from prototype.
void NT_analysis::check_prototype_default_parameter(
    Definition const *proto_def,
    Definition       *curr_def)
{
    IType_function const *curr_func = cast<IType_function>(curr_def->get_type());

    // check the default parameters of this definition
    for (int i = 0, n_params = curr_func->get_parameter_count(); i < n_params; ++i) {
        if (curr_def->get_default_param_initializer(i) != NULL) {
            IType const   *unused;
            ISymbol const *curr_name;

            curr_func->get_parameter(i, unused, curr_name);

            error(
                DEFAULT_PARAM_REDEFINITION,
                curr_def,
                Error_params(*this)
                    .add_numword(i + 1)
                    .add(curr_name)
                    .add_signature(curr_def)
                );
            if (proto_def->get_position() != NULL) {
                add_note(
                    DECLARED_AT,
                    proto_def,
                    Error_params(*this)
                        .add_signature(curr_def));
            }
        }
    }
    // copy the initializers, so they are available for overload resolution
    curr_def->copy_initializers(&m_module, proto_def);
}

// Check if a function is redeclared or redefined.
Definition const *NT_analysis::check_function_redeclaration(Definition *curr_def)
{
    IType const *type = curr_def->get_type();

    // errors are never redeclarations
    if (is<IType_error>(type))
        return NULL;

    IType_function const *curr_func = cast<IType_function>(type);
    int n_params = curr_func->get_parameter_count();

    bool remove_it = false;
    bool is_set_exported = false;

    Definition *first_default_param_def = NULL;
    Definition *proto = NULL;
    for (Definition *same_def = curr_def->get_prev_def();
        same_def != NULL;
        same_def = same_def->get_prev_def())
    {
        if (same_def->has_flag(Definition::DEF_IGNORE_OVERLOAD)) {
            // we will find a better one, ignore this
            continue;
        }

        type = same_def->get_type();
        if (is<IType_error>(type)) {
            // ignore previous errors
            continue;
        }

        IType_function const *func = as<IType_function>(type);
        if (func == NULL) {
            // the previous declaration is NOT a function, redeclaring with different type
            // no prototype found
            error(
                CONFLICTING_REDECLARATION,
                curr_def,
                Error_params(*this)
                    .add(curr_def->get_sym()));
            add_prev_definition_note(same_def);
            // remove it
            remove_it = true;
            break;
        }

        if (same_def->has_flag(Definition::DEF_IS_EXPORTED))
            is_set_exported = true;

        // two overloaded functions MUST differ
        // a) number of parameters
        if (n_params != func->get_parameter_count())
            continue;

        // b) types of parameters
        int i;
        bool overload_on_modifier = false;
        for (i = 0; i < n_params; ++i) {
            IType const   *ptype_curr, *ptype;
            ISymbol const *unused;

            curr_func->get_parameter(i, ptype_curr, unused);
            func->get_parameter(i, ptype, unused);

            if (!equal_types(ptype_curr->skip_type_alias(), ptype->skip_type_alias()))
                break;

            // we do not allow overloads on type modifiers
            if (!equal_types(ptype_curr, ptype))
                overload_on_modifier = true;
        }
        if (i < n_params)
            continue;

        // remember the prototype found
        proto = same_def;

        if (overload_on_modifier) {
            error(
                FUNCTION_OVERLOADING_ON_MODIFIER,
                curr_def,
                Error_params(*this)
                    .add_signature(curr_def));
            add_prev_definition_note(same_def);
            // mark this definition or errors might follow due to ambiguous versions.
            curr_def->set_flag(Definition::DEF_IGNORE_OVERLOAD);

            if (curr_def->has_flag(Definition::DEF_IS_DECL_ONLY)) {
                remove_it = true;
            }
        }

        // found same signature, check return types
        IType const *curr_ret_type = curr_func->get_return_type();
        IType const *ret_type      = func->get_return_type();

        // return type must match
        if (! equal_types(curr_ret_type, ret_type)) {
            error(
                FUNCTION_REDECLARED_DIFFERENT_RET_TYPE,
                curr_def,
                Error_params(*this)
                    .add(curr_def->get_sym())
                    .add(curr_ret_type)
                    .add(ret_type)
            );
            add_prev_definition_note(same_def);

            // mark this definition or errors might follow due to ambiguous versions.
            curr_def->set_flag(Definition::DEF_IGNORE_OVERLOAD);

            if (curr_def->has_flag(Definition::DEF_IS_DECL_ONLY)) {
                remove_it = true;
            }
        }

        bool same_exp = same_def->has_flag(Definition::DEF_IS_EXPORTED);
        bool curr_exp = curr_def->has_flag(Definition::DEF_IS_EXPORTED);

        if (same_exp != curr_exp) {
            if (same_exp) {
                warning(
                    FUNCTION_PREVIOUSLY_EXPORTED,
                    curr_def,
                    Error_params(*this)
                        .add_signature(curr_def));
                add_prev_definition_note(same_def);

                export_definition(curr_def);
                if (IDeclaration *decl = const_cast<IDeclaration *>(curr_def->get_declaration()))
                    decl->set_export(true);
            } else {
                warning(
                    FUNCTION_NOW_EXPORTED,
                    curr_def,
                    Error_params(*this)
                        .add_signature(curr_def));
                add_prev_definition_note(same_def);

                export_definition(same_def);
                if (IDeclaration *decl = const_cast<IDeclaration *>(same_def->get_declaration()))
                    decl->set_export(true);
            }
        }

        Qualifier prev_qual = get_function_qualifier(same_def);
        Qualifier curr_qual = get_function_qualifier(curr_def);

        if (prev_qual != curr_qual) {
            error(
                FUNCTION_REDECLARED_DIFFERENT_QUALIFIER,
                curr_def,
                Error_params(*this)
                    .add_signature(curr_def)
                    .add(curr_qual)
                    .add(prev_qual));
            add_prev_definition_note(same_def);

            // mark this definition or errors might follow due to ambiguous versions.
            curr_def->set_flag(Definition::DEF_IGNORE_OVERLOAD);

            if (curr_def->has_flag(Definition::DEF_IS_DECL_ONLY)) {
                remove_it = true;
            }
        }

        if (!curr_def->has_flag(Definition::DEF_IS_DECL_ONLY)) {
            Definition *prev_def = same_def->get_definite_definition();

            if (prev_def != NULL) {
                error(
                    FUNCTION_REDEFINITION,
                    curr_def,
                    Error_params(*this)
                        .add_signature(curr_def));
                add_prev_definition_note(prev_def);

                // mark this definition or errors might follow due to ambiguous versions.
                curr_def->set_flag(Definition::DEF_IGNORE_OVERLOAD);
                return proto;
            }
        } else {
            if (!remove_it) {
                // a declaration followed another declaration/definition is useless
                warning(
                    FUNCTION_PROTOTYPE_USELESS,
                    curr_def,
                    Error_params(*this)
                        .add_signature(curr_def));
                add_prev_definition_note(same_def);
            }
        }

        // names must match
        for (int i = 0; i < n_params; ++i) {
            IType const   *unused;
            ISymbol const *curr_name, *name;

            curr_func->get_parameter(i, unused, curr_name);
            func->get_parameter(i, unused, name);

            if (curr_name != name) {
                error(
                    REDECLARED_DIFFERENT_PARAM_NAME,
                    curr_def,
                    Error_params(*this)
                        .add_signature(curr_def)
                        .add_numword(i + 1)
                        .add(curr_name)
                        .add(name)
                );
                add_prev_definition_note(same_def);

                // mark this definition or errors might follow due to ambiguous versions.
                curr_def->set_flag(Definition::DEF_IGNORE_OVERLOAD);
            }

            if (get_default_param_initializer(same_def, i) != NULL)
                first_default_param_def = same_def;
        }
        if (! curr_def->has_flag(Definition::DEF_IGNORE_OVERLOAD)) {
            // fine, found the same declaration again, update the definitions
            if (!curr_def->has_flag(Definition::DEF_IS_DECL_ONLY)) {
                same_def->set_definite_definition(curr_def);
                same_def->set_flag(Definition::DEF_IGNORE_OVERLOAD);
            } else {
                curr_def->set_definite_definition(same_def->get_definite_definition());
                curr_def->set_flag(Definition::DEF_IGNORE_OVERLOAD);
            }
        }
    }

    if (first_default_param_def != NULL) {
        check_prototype_default_parameter(first_default_param_def, curr_def);
    }

    if (is_set_exported) {
        if (!curr_def->has_flag(Definition::DEF_IS_EXPORTED)) {
            error(
                OVERLOADS_MUST_BE_EXPORTED,
                curr_def,
                Error_params(*this)
                    .add_signature(curr_def));
            curr_def->set_flag(Definition::DEF_IS_EXPORTED);
        }
    } else {
        if (curr_def->has_flag(Definition::DEF_IS_EXPORTED)) {
            for (Definition *same_def = curr_def->get_prev_def();
                 same_def != NULL;
                 same_def = same_def->get_prev_def())
            {
                if (same_def->has_flag(Definition::DEF_IGNORE_OVERLOAD)) {
                    // we will find a better one, ignore this
                    continue;
                }

                type = same_def->get_type();
                if (is<IType_error>(type)) {
                    // ignore previous errors
                    continue;
                }

                IType_function const *func = as<IType_function>(type);
                if (func == NULL) {
                    // already reported
                    break;
                }

                if (!same_def->has_flag(Definition::DEF_IS_EXPORTED)) {
                    error(
                        OVERLOADS_MUST_BE_EXPORTED,
                        same_def,
                        Error_params(*this)
                            .add_signature(same_def));
                    same_def->set_flag(Definition::DEF_IS_EXPORTED);
                    if (same_def->has_flag(Definition::DEF_IS_IMPORTED)) {
                        Position_impl null_pos(0, 0, 0, 0);

                        Position const *import_pos = get_import_location(same_def);
                        if (import_pos == NULL)
                            import_pos = &null_pos;

                        add_note(
                            IMPORTED_WITHOUT_EXPORT,
                            *import_pos,
                            Error_params(*this)
                                .add_signature(same_def));
                    }
                }
            }
        }
    }
    return proto;
}

// Check if function annotations are only present at the first declaration.
void NT_analysis::check_function_annotations(
    Definition const *curr_def,
    Definition const *proto_def)
{
    if (proto_def == NULL || is_error(curr_def))
        return;

    IDeclaration_function const *decl = cast<IDeclaration_function>(curr_def->get_declaration());

    IAnnotation_block const *block = decl->get_return_annotations();

    if (block == NULL) {
        for (size_t i = 0, n = decl->get_parameter_count(); i < n; ++i) {
            IParameter const *p = decl->get_parameter(i);

            block = p->get_annotations();
            if (block != NULL)
                break;
        }
    }
    if (block == NULL)
        block = decl->get_annotations();

    if (block != NULL) {
        error(
            FUNCTION_ANNOTATIONS_NOT_AT_PROTOTYPE,
            block->access_position(),
            Error_params(*this));
        if (proto_def->get_position() != NULL) {
            add_note(
                DECLARED_AT,
                proto_def,
                Error_params(*this)
                    .add_signature(proto_def));
        }
    }
}

// Check an expression against a range.
void NT_analysis::check_expression_range(
    IAnnotation const *anno,
    IExpression const *expr,
    Compilation_error code,
    ISymbol const     *extra)
{
    // now check if the initializer is a constant expression
    bool is_invalid = false;
    if (!is_const_expression(expr, is_invalid))
        return;
    if (is_invalid)
        return;

    IValue const *v = expr->fold(&m_module, m_module.get_value_factory(), NULL);
    if (is<IValue_bad>(v))
        return;

    IExpression_literal const *min_expr =
        cast<IExpression_literal>(anno->get_argument(0)->get_argument_expr());
    IExpression_literal const *max_expr =
        cast<IExpression_literal>(anno->get_argument(1)->get_argument_expr());
    IValue const *min_v = min_expr->get_value();
    IValue const *max_v = max_expr->get_value();

    MDL_ASSERT(min_v->get_type() == expr->get_type()->skip_type_alias());
    if ((element_wise_compare(min_v, v) & ~(IValue::CR_LE)) != 0 ||
        (element_wise_compare(v, max_v) & ~(IValue::CR_LE)) != 0) {
            warning(
                code,
                expr->access_position(),
                Error_params(*this)
                    .add(expr)
                    .add(min_v)
                    .add(max_v)
                    .add(extra));
    }
}

// Check the parameter range for the given parameter if a hard_range was specified.
void NT_analysis::check_parameter_range(
    IParameter const  *param,
    IExpression const *init_expr)
{
    // first check if we have a hard_range annotation
    IAnnotation_block const *block = param->get_annotations();
    if (block == NULL)
        return;

    IAnnotation const *hard_range =
        find_annotation_by_semantics(block, IDefinition::DS_HARD_RANGE_ANNOTATION);
    if (hard_range == NULL)
        return;

    check_expression_range(hard_range, init_expr, DEFAULT_INITIALIZER_OUTSIDE_HARD_RANGE, NULL);
}

// Check the field range for the given structure field if a hard_range was specified.
void NT_analysis::check_field_range(
    IAnnotation_block const *block,
    IExpression const       *init_expr)
{
    IAnnotation const *hard_range =
        find_annotation_by_semantics(block, IDefinition::DS_HARD_RANGE_ANNOTATION);
    if (hard_range == NULL)
        return;

    check_expression_range(hard_range, init_expr, DEFAULT_INITIALIZER_OUTSIDE_HARD_RANGE, NULL);
}

// Check the field range for the given structure field if a hard_range was specified.
void NT_analysis::check_field_assignment_range(
    IDefinition const *f_def,
    IExpression const *expr)
{
    Definition const *def       = impl_cast<Definition>(f_def);
    Scope const      *def_scope = def->get_def_scope();
    Definition const *s_def     = def_scope->get_owner_definition();

    if (Definition const *orig = m_module.get_original_definition(s_def))
        s_def = orig;
    IDeclaration_type_struct const *s_decl =
        as<IDeclaration_type_struct>(s_def->get_declaration());

    if (s_decl == NULL)
        return;
    IAnnotation_block const *block = s_decl->get_annotations(f_def->get_field_index());
    if (block == NULL)
        return;

    IAnnotation const *hard_range =
        find_annotation_by_semantics(block, IDefinition::DS_HARD_RANGE_ANNOTATION);
    if (hard_range == NULL)
        return;

    check_expression_range(hard_range, expr, FIELD_VALUE_OUTSIDE_HARD_RANGE, f_def->get_symbol());
}

// Check a parameter initializer.
IExpression const *NT_analysis::handle_parameter_initializer(
    IExpression const *init_expr,
    IType const       *p_type,
    IType_name const  *pt_name,
    IParameter const  *param)
{
    IType const *init_type = init_expr->get_type()->skip_type_alias();
    if (IType_array const *a_type = as<IType_array>(p_type)) {
        // we do not allow array conversions

        if (is<IType_error>(init_type)) {
            // error already reported
        } else if (!equal_types(a_type, init_type)) {
            IType_array const *a_init_type = as<IType_array>(init_type);
            bool              has_error = false;
            if (a_init_type != NULL && a_init_type->is_immediate_sized()) {
                if (!a_type->is_immediate_sized()) {
                    // special case here: we ALLOW that a default parameter for a
                    // deferred size array is of concrete size
                    has_error = !equal_types(
                        a_type->get_element_type()->skip_type_alias(),
                        a_init_type->get_element_type()->skip_type_alias());
                } else if (a_type->is_immediate_sized() &&
                    a_type->get_size() == a_init_type->get_size()) {
                        // check aliasing of element types
                        has_error = !equal_types(
                            a_type->get_element_type()->skip_type_alias(),
                            a_init_type->get_element_type()->skip_type_alias());
                } else {
                    has_error = true;
                }
            } else {
                // cannot convert to array type
                has_error = true;
            }
            if (has_error) {
                error(
                    NO_ARRAY_CONVERSION,
                    init_expr->access_position(),
                    Error_params(*this).add(init_type).add(p_type));
            }
        }
    } else if (is<IType_error>(init_type)) {
        // error already reported
    } else if (!equal_types(init_type, p_type->skip_type_alias())) {
        // need a conversion
        if (is<IType_array>(init_type)) {
            if (!is<IType_error>(p_type)) {
                error(
                    NO_ARRAY_CONVERSION,
                    init_expr->access_position(),
                    Error_params(*this).add(init_type).add(p_type));
            }
        } else {
            // find the constructor
            Position const &pos = init_expr->access_position();

            init_expr = find_init_constructor(
                p_type->skip_type_alias(),
                m_module.clone_name(pt_name, /*modifier=*/NULL),
                init_expr,
                pos);
        }
    }
    check_parameter_range(param, init_expr);
    return init_expr;
}


// Declare a new function.
void NT_analysis::declare_function(IDeclaration_function *fkt_decl)
{
    ISimple_name const *f_name = fkt_decl->get_name();
    ISymbol const      *f_sym  = f_name->get_symbol();

    bool       has_error = false;
    Definition *f_def    = get_definition_at_scope(f_sym);
    if (f_def != NULL && f_def->get_kind() != Definition::DK_FUNCTION) {
        err_redeclaration(
            Definition::DK_FUNCTION,
            f_def,
            fkt_decl->access_position(),
            ENT_REDECLARATION);
        has_error = true;
    }

    // for now, set the error definition, update it later
    f_def = get_error_definition();

    const_cast<ISimple_name *>(f_name)->set_definition(f_def);
    fkt_decl->set_definition(f_def);

    // handle parameters first, BEFORE the return type
    size_t n_params = fkt_decl->get_parameter_count();

    VLA<IType_factory::Function_parameter> params(m_module.get_allocator(), n_params);
    VLA<IExpression const *>               inits(m_module.get_allocator(), n_params);

    bool has_initializers = false;

    Scope                *f_scope = NULL;
    IType_function const *f_type  = NULL;

    IStatement const *body = fkt_decl->get_body();

    // create new scope for the parameters and the body
    m_next_param_idx = 0;
    unsigned deriv_mask = 0;
    {
        Flag_store params_are_used(m_params_are_used, body == NULL);

        Definition_table::Scope_enter scope(*m_def_tab, f_def);
        f_scope = m_def_tab->get_curr_scope();

        for (size_t i = 0; i < n_params; ++i) {
            IParameter const *param = fkt_decl->get_parameter(i);

            visit(param);

            IType_name const *pt_name = param->get_type_name();
            IType const      *p_type  = pt_name->get_type();

            ISimple_name const *p_name = param->get_name();
            ISymbol const      *p_sym  = p_name->get_symbol();

            Definition const *p_def = impl_cast<Definition>(p_name->get_definition());
            if (p_def->has_flag(Definition::DEF_IS_DERIVABLE))
                deriv_mask |= unsigned(1 << i);

            IType const *bad_type = has_forbidden_function_parameter_type(p_type, m_is_stdlib);
            if (bad_type != NULL) {
                error(
                    FORBIDDEN_FUNCTION_PARAMETER_TYPE,
                    pt_name->access_position(),
                    Error_params(*this).add(p_type));
                if (bad_type->skip_type_alias() != p_type->skip_type_alias()) {
                    add_note(
                        TYPE_CONTAINS_FORBIDDEN_SUBTYPE,
                        pt_name->access_position(),
                        Error_params(*this)
                            .add(p_type)
                            .add(bad_type));
                }
                p_type = m_tc.error_type;
            }

            params[i].p_type = p_type;
            params[i].p_sym  = p_sym;
            inits[i]         = NULL;

            if (is<IType_error>(p_type)) {
                has_error = true;
            }

            if (IExpression const *init_expr = param->get_init_expr()) {
                has_initializers = true;
                inits[i] = handle_parameter_initializer(init_expr, p_type, pt_name, param);
            }
        }

        // now visit the return type
        // We do this INSIDE the parameter scope, so all array size definitions
        // are visible.
        IType_name const *ret_name = fkt_decl->get_return_type_name();
        {
            Store<Match_restriction> restrict_type(m_curr_restriction, MR_TYPE);
            visit(ret_name);
        }

        if (ret_name->is_incomplete_array()) {
            error(
                INCOMPLETE_ARRAY_SPECIFICATION,
                ret_name->access_position(),
                Error_params(*this));
            const_cast<IType_name*>(ret_name)->set_type(m_tc.error_type);
        }

        // Handle enable_if annotations here after all parameters have been processed because
        // use-before-def is allowed here
        for (int i = 0; i < n_params; ++i) {
            IParameter const *param = fkt_decl->get_parameter(i);
            handle_enable_ifs(param);
        }

        IType const *ret_type = as_type(ret_name);
        if (is<IType_error>(ret_type)) {
            has_error = true;
        }

        if (IType_array const *a_type = as<IType_array>(ret_type)) {
            if (!a_type->is_immediate_sized()) {
                // an abstract type size cannot be defined inside an return type, check if the same
                // type size is used INSIDE the parameters, if yes, it is not defined here
                bool is_defined = true;
                IType_array_size const *size = a_type->get_deferred_size();
                for (size_t i = 0; i < n_params; ++i) {
                    if (IType_array const *p_type = as<IType_array>(params[i].p_type)) {
                        if (!p_type->is_immediate_sized() && size == p_type->get_deferred_size()) {
                            is_defined = false;
                            break;
                        }
                    }
                }
                if (is_defined) {
                    error(
                        ABSTRACT_ARRAY_RETURN_TYPE,
                        ret_name->access_position(),
                        Error_params(*this));
                    const_cast<IType_name*>(ret_name)->set_type(m_tc.error_type);
                }
            }
        }

        if (!check_function_return_type(ret_type, ret_name->access_position(), f_sym, m_is_stdlib))
            ret_type = m_tc.error_type;

        // create the function type
        f_type = m_tc.create_function(ret_type, params);
    }

    if (!has_error) {
        // create a new definition for this function
        f_def = m_def_tab->enter_definition(
            Definition::DK_FUNCTION, f_sym, f_type, &fkt_decl->access_position());
        f_def->set_own_scope(f_scope);
        if (m_is_stdlib) {
            f_def->set_flag(Definition::DEF_IS_STDLIB);

            char const *module_name = m_module.get_name();
            if (strcmp(module_name, "::state") == 0) {
                // every function from the state module uses the state
                f_def->set_flag(Definition::DEF_USES_STATE);
                if (fkt_decl->get_qualifier() == FQ_VARYING) {
                    // uses the varying state subset
                    f_def->set_flag(Definition::DEF_USES_VARYING_STATE);
                }
            } else if (strcmp(module_name, "::debug") == 0) {
                // mark every function from the debug module
                f_def->set_flag(Definition::DEF_USES_DEBUG_CALLS);
            } else if (strcmp(module_name, "::scene") == 0) {
                // every function from the scene module uses the state
                f_def->set_flag(Definition::DEF_USES_STATE);
                // uses the varying state subset
                f_def->set_flag(Definition::DEF_USES_VARYING_STATE);
            }
        }
        f_def->set_parameter_derivable_mask(deriv_mask);
        f_def->set_declaration(fkt_decl);

        if (has_initializers)
            m_module.allocate_initializers(f_def, n_params);

        // check default arguments
        bool need_default_arg = false;
        for (size_t i = 0; i < n_params; ++i) {
            if (IExpression const *init = inits[i]) {
                f_def->set_default_param_initializer(i, init);
                need_default_arg = true;
            } else if (need_default_arg) {
                IParameter const   *param  = fkt_decl->get_parameter(i);
                ISimple_name const *p_name = param->get_name();

                error(
                    MISSING_DEFAULT_ARGUMENT,
                    p_name->access_position(),
                    Error_params(*this).add(i + 1).add_signature(f_def));
            }
        }

        const_cast<ISimple_name *>(f_name)->set_definition(f_def);
        fkt_decl->set_definition(f_def);
    }

    // now handle the body
    {
        Definition_table::Scope_enter scope(*m_def_tab, f_scope);

        if (body != NULL) {
            Enter_function enter(*this, f_def);
            Flag_store     can_thow_bounds(m_can_throw_bounds, false);
            Flag_store     can_thow_divzero(m_can_throw_divzero, false);

            // this is a function definition
            if (IStatement_compound const *block = as<IStatement_compound>(body)) {
                for (size_t i = 0, n = block->get_statement_count(); i < n; ++i) {
                    IStatement const *stmt = block->get_statement(i);

                    visit(stmt);
                }
            } else if (is<IStatement_invalid>(body)) {
                // an invalid statement, we are ready
                visit(body);
            } else {
                bool single_expr_func = false;
                if (m_module.get_mdl_version() >= IMDL::MDL_VERSION_1_6) {
                   if (is<IStatement_expression>(body)) {
                       single_expr_func = true;
                   }
                }

                Flag_store allow_let(m_allow_let_expression, single_expr_func);
                visit(body);

                if (!single_expr_func) {
                    // not a compound statement
                    error(
                        FUNCTION_BODY_NOT_BLOCK,
                        body->access_position(),
                        Error_params(*this));
                }
            }

            if (m_can_throw_bounds)
                f_def->set_flag(Definition::DEF_CAN_THROW_BOUNDS);
            if (m_can_throw_divzero)
                f_def->set_flag(Definition::DEF_CAN_THROW_DIVZERO);
        } else {
            // declaration only
            if (!has_error)
                f_def->set_flag(Definition::DEF_IS_DECL_ONLY);
        }
    }

    if (IAnnotation_block const *anno = fkt_decl->get_annotations()) {
        Definition_store store(m_annotated_def, f_def);
        visit(anno);
    }

    if (IAnnotation_block const *ret_anno = fkt_decl->get_return_annotations()) {
        Definition_store store(m_annotated_def, f_def);
        Flag_store is_return_annotation(m_is_return_annotation, true);
        visit(ret_anno);
    }


    // add it to the call graph
    if (!has_error) {
        if (!m_is_stdlib)
            calc_mdl_versions(f_def);
        if (fkt_decl->is_exported())
            export_definition(f_def);

        switch (fkt_decl->get_qualifier()) {
        case FQ_NONE:
            break;
        case FQ_VARYING:
            f_def->set_flag(Definition::DEF_IS_VARYING);
            break;
        case FQ_UNIFORM:
            f_def->set_flag(Definition::DEF_IS_UNIFORM);
            break;
        }

        // add it to the call graph
        m_cg.add_node(f_def);
    }

    Definition const *proto_def = check_function_redeclaration(f_def);
    if (proto_def != NULL) {
        IDeclaration const *proto_decl = proto_def->get_declaration();
        f_def->set_prototype_declaration(proto_decl);

        check_function_annotations(f_def, proto_def);
    }
}

/// Get the name of a parameter of a function or material definition.
///
/// \param def  the definition of the entity
/// \param pos  position of the parameter
static ISymbol const *get_param_sym(Definition const *def, int pos)
{
    MDL_ASSERT(def->get_kind() == Definition::DK_FUNCTION);

    // There might be no declaration, hence use the type

    IType_function const *f_tp = cast<IType_function>(def->get_type());

    ISymbol const *p_sym;
    IType const *p_type;

    MDL_ASSERT(pos < f_tp->get_parameter_count());

    f_tp->get_parameter(pos, p_type, p_sym);

    return p_sym;
}

// Declare a new function preset.
void NT_analysis::declare_function_preset(
    IDeclaration_function *fkt_decl)
{
    if (IAnnotation_block const *ret_anno = fkt_decl->get_return_annotations()) {
        visit(ret_anno);
    }

    bool has_error = false;

    bool                   has_let = false;
    IExpression_call const *call   = NULL;
    IStatement const       *body   = fkt_decl->get_body();

    // let expressions are allowed inside presets
    Flag_store allow_let(m_allow_let_expression, true);

    if (IStatement_expression const *stmt = as<IStatement_expression>(body)) {
        IExpression const *expr = stmt->get_expression();

        while (IExpression_let const *c_let = as<IExpression_let>(expr)) {
            has_let = true;
            expr = c_let->get_expression();
        }

        if (IExpression_call const *c_expr = as<IExpression_call>(expr)) {
            // the material instance
            call = c_expr;
        } else {
            // not a function instantiation
            error(
                FUNCTION_PRESET_BODY_NOT_CALL,
                body->access_position(),
                Error_params(*this));
            has_error = true;
        }

        Expr_call_store store(m_preset_overload, call);
        visit(stmt);
    } else if (is<IStatement_invalid>(body)) {
        // an invalid statement, we are ready
        visit(body);
        has_error = true;
    } else {
        visit(body);

        // not an expression
        error(FUNCTION_PRESET_BODY_NOT_CALL, body->access_position(), Error_params(*this));
        has_error = true;
    }

    Definition const *instance_def = NULL;
    mi::base::Handle<Module const>   origin;

    IType const *call_ret_type = m_tc.error_type;

    if (!has_error) {
        IExpression_reference const *callee = as<IExpression_reference>(call->get_reference());

        if (callee == NULL) {
            const_cast<IExpression_call *>(call)->set_type(m_tc.error_type);
            has_error = true;
        } else {
            IType_name const      *tn    = callee->get_name();
            IQualified_name const *qname = tn->get_qualified_name();
            IDefinition const     *def   = qname->get_definition();

            instance_def = impl_cast<Definition>(def);
            if (instance_def->has_flag(Definition::DEF_IS_IMPORTED)) {
                origin = mi::base::make_handle(m_module.get_owner_module(instance_def));
            }

            // The overload resolution returns the base type at this point, only the auto-type
            // analysis will enter the right one as the call result.
            // Hence gen the return type form the called function's type.
            if (!is<IType_error>(call->get_type())) {
                IType_function const *f_tp = cast<IType_function>(def->get_type());

                call_ret_type = f_tp->get_return_type();
            }
        }
    }

    IType_name const *ret_name = fkt_decl->get_return_type_name();

    bool ignore_size_errors = false;
    if (IType_array const *a_call_ret_type = as<IType_array>(call_ret_type)) {
        if (!a_call_ret_type->is_immediate_sized()) {
            error(
                FUNC_VARIANT_WITH_DEFERRED_ARRAY_RET_TYPE,
                call->access_position(),
                Error_params(*this));
            const_cast<IType_name*>(ret_name)->set_type(m_tc.error_type);
            ignore_size_errors = true;
        }
    }

    {
        Store<Match_restriction> restrict_type(m_curr_restriction, MR_TYPE);
        Flag_store               ignore_deferred_size(m_ignore_deferred_size, ignore_size_errors);
        visit(ret_name);
    }

    IType const *ret_type = as_type(ret_name);
    if (is<IType_error>(ret_type)) {
        has_error = true;
    }

    if (ret_name->is_incomplete_array()) {
        error(
            INCOMPLETE_ARRAY_SPECIFICATION,
            ret_name->access_position(),
            Error_params(*this));
        const_cast<IType_name*>(ret_name)->set_type(m_tc.error_type);
    }

    if (is<IType_error>(call_ret_type)) {
        // overload was not resolved
        has_error = true;
    } else {
        if (ret_type != call_ret_type && !is<IType_error>(ret_type)) {
            error(
                FUNCTION_PRESET_WRONG_RET_TYPE,
                ret_name->access_position(),
                Error_params(*this).add(ret_type).add(call_ret_type));
            // this is a minor error, do not set has_error here
        }
    }

    ISimple_name const *m_name = fkt_decl->get_name();
    Definition         *con_def;
    if (has_error) {
error_found:
        con_def = get_error_definition();
    } else {
        IType_function const *ftype   = cast<IType_function>(instance_def->get_type());
        int                  n_params = ftype->get_parameter_count();

        VLA<IType_factory::Function_parameter> params(m_module.get_allocator(), n_params);

        for (int i = 0; i < n_params; ++i) {
            IType const   *p_type;
            ISymbol const *p_sym;

            ftype->get_parameter(i, p_type, p_sym);

            params[i].p_type = p_type;
            params[i].p_sym  = p_sym;
        }

        VLA<IExpression const *> new_defs(m_module.get_allocator(), n_params);
        memset(new_defs.data(), 0, sizeof(new_defs[0]) * n_params);

        has_error = !collect_preset_defaults(instance_def, call, new_defs);
        if (has_error)
            goto error_found;

        ISymbol const *m_sym = m_name->get_symbol();

        // create a definition for the creator function of this material instance
        IType_function const *m_con_type = m_tc.create_function(ret_type, params);

        con_def = get_definition_at_scope(m_sym);
        if (con_def != NULL) {
            err_redeclaration(
                Definition::DK_FUNCTION,
                con_def,
                fkt_decl->access_position(),
                ENT_REDECLARATION);

            con_def = get_error_definition();
            has_error = true;
        } else {
            con_def = m_def_tab->enter_definition(
                Definition::DK_FUNCTION, m_sym, m_con_type, &fkt_decl->access_position());
            if (fkt_decl->is_exported()) {
                if (!instance_def->has_flag(Definition::DEF_IS_EXPORTED) &&
                    !instance_def->has_flag(Definition::DEF_IS_IMPORTED)) {
                    error(
                        EXPORTED_PRESET_OF_UNEXPORTED_ORIGIN,
                        *con_def->get_position(),
                        Error_params(*this)
                            .add(con_def->get_sym())
                            .add_signature(instance_def)
                    );
                }
                export_definition(con_def);
            }
            if (m_is_stdlib)
                con_def->set_flag(Definition::DEF_IS_STDLIB);

            con_def->set_declaration(fkt_decl);

            // add it to the call graph
            m_cg.add_node(con_def);
        }

        // create new scope for the parameters and the body of the constructor
        {
            Definition_table::Scope_enter scope(*m_def_tab, con_def);

            // finish constructor
            Default_initializer_modifier def_modifier(*this, n_params, origin.get());

            bool has_initializers = false;
            bool need_default_arg = false;
            for (int i = 0; i < n_params; ++i) {
                IExpression const *init_expr;

                if (new_defs[i] != NULL) {
                    init_expr = new_defs[i];

                    // clone the expression here: if might reference let temporaries, so
                    // these must be resolved
                    if (has_let) {
                        Let_temporary_modifier let_modifier(m_module);
                        init_expr = m_module.clone_expr(init_expr, &let_modifier);
                    }
                } else {
                    IExpression const *old_init = get_default_param_initializer(instance_def, i);

                    init_expr = old_init != NULL ?
                        m_module.clone_expr(old_init, &def_modifier) : NULL;
                }

                def_modifier.set_parameter_value(i, init_expr);

                if (init_expr != NULL) {
                    need_default_arg = true;
                    if (!has_error) {
                        if (!has_initializers) {
                            m_module.allocate_initializers(con_def, n_params);
                            has_initializers = true;
                        }
                        // Beware: the initializers set here might still use not-yet imported
                        // definitions, hence we must register them for auto-import
                        con_def->set_default_param_initializer(i, init_expr);
                        need_default_arg = true;
                    }
                } else if (need_default_arg) {
                    ISymbol const *psym = get_param_sym(instance_def, i);
                    error(
                        MISSING_DEFAULT_ARGUMENT,
                        call->access_position(),
                        Error_params(*this).add(psym).add_signature(instance_def));
                }
            }
            // register for waiting fix, see above
            if (!is_error(con_def))
                m_initializers_must_be_fixed.push_back(con_def);
        }
    }

    const_cast<ISimple_name *>(m_name)->set_definition(con_def);
    fkt_decl->set_definition(con_def);

    if (IAnnotation_block const *anno = fkt_decl->get_annotations()) {
        Definition_store store(m_annotated_def, con_def);
        visit(anno);
    }

    if (!m_is_stdlib && !is_error(con_def))
        calc_mdl_versions(con_def);
}

// Check restrictions of material parameter type.
bool NT_analysis::check_material_parameter_type(
    IType const    *p_type,
    Position const &pos)
{
    if (IMDL::MDL_VERSION_1_0 == m_module.get_mdl_version()) {
        // for MDL 1.0 we allow bsdfs parameters, but issue a warning
        if (as<IType_bsdf>(p_type) != NULL) {
            warning(
                FORBIDDEN_MATERIAL_PARAMETER_TYPE,
                pos,
                Error_params(*this).add(p_type));
            return true;
        }
    }

    IType const *bad_type = has_forbidden_material_parameter_type(p_type);
    if (bad_type != NULL) {
        error(
            FORBIDDEN_MATERIAL_PARAMETER_TYPE,
            pos,
            Error_params(*this).add(p_type));
        if (bad_type->skip_type_alias() != p_type->skip_type_alias()) {
            add_note(
                TYPE_CONTAINS_FORBIDDEN_SUBTYPE,
                pos,
                Error_params(*this)
                .add(p_type)
                .add(bad_type));
        }
        return false;
    }
    return true;
}

// Handle all enable_if annotations on the given parameter.
void NT_analysis::handle_enable_ifs(IParameter const *param)
{
    if (IAnnotation_block const *anno_block = param->get_annotations()) {
        for (int j = 0, n = anno_block->get_annotation_count(); j < n; ++j) {
            IAnnotation_enable_if const *enable_if =
                as<IAnnotation_enable_if>(anno_block->get_annotation(j));
            if (enable_if == NULL)
                continue;

            // at this point we have already checked that the parameter is a string literal
            IExpression_literal const *lit =
                cast<IExpression_literal>(enable_if->get_argument(0)->get_argument_expr());
            IValue_string const *sval = cast<IValue_string>(lit->get_value());

            char const *str = sval->get_value();
            Messages_impl msgs(get_allocator(), m_module.get_filename());
            IExpression const *cond = m_compiler->parse_expression(
                str,
                lit->access_position().get_start_line(),
                lit->access_position().get_start_column() + 1,  // skip '"'
                &m_module,
                m_enable_experimental_features,
                msgs);

            // visit the condition in special "enable_if" mode
            {
                Msgs_store enable_if_msgs(m_compiler_msgs, &msgs);
                visit(cond);

                bool has_error = false;
                IExpression *conv = check_bool_condition(
                    cond, has_error, /*inside_enable_if=*/true);
                if (conv != NULL) {
                    cond = conv;
                }
                const_cast<IAnnotation_enable_if *>(enable_if)->set_expression(cond);

                ISymbol const *param_sym = param->get_name()->get_symbol();
                check_enable_if_condition(cond, param_sym);
            }

            size_t n_msgs = msgs.get_message_count();
            if (msgs.get_error_message_count() > 0) {
                // we found errors
                warning(
                    INVALID_ENABLE_IF_CONDITION,
                    enable_if->access_position(),
                    Error_params(*this));

                for (size_t i = 0; i < n_msgs; ++i) {
                    IMessage const *msg = msgs.get_message(i);
                    add_imported_message(/*filename_id=*/0, msg);
                }
                // KILL the annotation
                IQualified_name const *qn = enable_if->get_name();
                const_cast<IQualified_name *>(qn)->set_definition(get_error_definition());

                // delete erroneous annotation
                const_cast<IAnnotation_block *>(anno_block)->delete_annotation(j);
                --j;
                --n;
            } else if (n_msgs > 0) {
                // found warnings
                warning(
                    ENABLE_IF_CONDITION_HAS_WARNINGS,
                    enable_if->access_position(),
                    Error_params(*this));

                for (int i = 0; i < n_msgs; ++i) {
                    IMessage const *msg = msgs.get_error_message(i);
                    add_imported_message(/*filename_id=*/0, msg);
                }
            }
        }
    }
}

// Declare a new material.
void NT_analysis::declare_material(IDeclaration_function *mat_decl)
{
    IType_name const *ret_name = mat_decl->get_return_type_name();

    // we known it is the material type here, but visit it to set type and definition
    visit(ret_name);

    if (ret_name->is_incomplete_array()) {
        error(
            INCOMPLETE_ARRAY_SPECIFICATION,
            ret_name->access_position(),
            Error_params(*this));
        const_cast<IType_name*>(ret_name)->set_type(m_tc.error_type);
    }

    {
        Qualifier qual = ret_name->get_qualifier();
        if (qual != FQ_NONE) {
            char const *q = "";
            switch (qual) {
            case FQ_NONE:
                break;
            case FQ_VARYING:
                q = "varying";
                break;
            case FQ_UNIFORM:
                q = "uniform";
                break;
            }
            error(
                MATERIAL_QUALIFIER_NOT_ALLOW,
                ret_name->access_position(),
                Error_params(*this).add(q));
        }
    }

    {
        Qualifier qual = mat_decl->get_qualifier();
        if (qual != FQ_NONE) {
            char const *q = "";
            switch (qual) {
            case FQ_NONE:
                break;
            case FQ_VARYING:
                q = "varying";
                break;
            case FQ_UNIFORM:
                q = "uniform";
                break;
            }
            error(
                MATERIAL_QUALIFIER_NOT_ALLOW,
                mat_decl->access_position(),
                Error_params(*this).add(q));
        }
    }

    ISimple_name const *m_name = mat_decl->get_name();
    ISymbol const      *m_sym  = m_name->get_symbol();

    bool has_error           = false;
    bool has_prototype_error = false;

    Definition           *proto_def  = NULL;
    IType_function const *proto_type = NULL;

    Definition *con_def = get_definition_at_scope(m_sym);

    // check for prototype
    if (con_def != NULL) {
        if (con_def->has_flag(Definition::DEF_IS_DECL_ONLY) &&
            con_def->get_kind() == Definition::DK_FUNCTION) {
            IType_function const *ftype = cast<IType_function>(con_def->get_type());
            IType const          *rtype = ftype->get_return_type();

            if (rtype == m_tc.material_type) {
                // is a material prototype
                proto_def  = con_def;
                proto_type = ftype;
                con_def    = NULL;
            }
        }
    }

    // create a definition for the creator function of this material instance
    if (con_def != NULL) {
        err_redeclaration(
            Definition::DK_FUNCTION,
            con_def,
            mat_decl->access_position(),
            ENT_REDECLARATION,
            /*is_material=*/true);
        has_error = true;
    }

    // for now, set the error definition, update it later
    con_def = get_error_definition();

    const_cast<ISimple_name *>(m_name)->set_definition(con_def);
    mat_decl->set_definition(con_def);

    int n_params = mat_decl->get_parameter_count();

    VLA<IType_factory::Function_parameter> params(m_module.get_allocator(), n_params);
    VLA<IExpression const *>               inits(m_module.get_allocator(), n_params);

    Scope                *con_scope = NULL;
    IType_function const *con_type  = NULL;

    IStatement const *body = mat_decl->get_body();

    // create new scope for the parameters and the body of the constructor
    bool has_initializers = false;
    bool has_param_error  = false;

    m_next_param_idx = 0;
    {
        Definition_table::Scope_enter scope(*m_def_tab, con_def);
        con_scope = m_def_tab->get_curr_scope();

        if (proto_type != NULL) {
            if (n_params != proto_type->get_parameter_count()) {
                has_error           = true;
                has_prototype_error = true;
                error(
                    CONFLICTING_PROTOTYPE,
                    mat_decl->access_position(),
                    Error_params(*this).add_signature(proto_def));
                add_note(
                    DECLARED_AT,
                    proto_def,
                    Error_params(*this)
                        .add_signature(proto_def));
            }
        }

        Flag_store params_are_used(m_params_are_used, body == NULL);

        // finish types
        for (int i = 0; i < n_params; ++i) {
            IParameter const *param = mat_decl->get_parameter(i);

            visit(param);

            IType_name const *pt_name = param->get_type_name();
            IType const      *p_type  = pt_name->get_type();

            ISimple_name const *p_name = param->get_name();
            ISymbol const      *p_sym  = p_name->get_symbol();

            if (!check_material_parameter_type(p_type, p_name->access_position())) {
                p_type = m_tc.error_type;
            }

            params[i].p_type = p_type;
            params[i].p_sym  = p_sym;
            inits[i]         = NULL;

            if (is<IType_error>(p_type)) {
                has_param_error = true;
            } else if (!has_prototype_error && proto_type != NULL) {
                IType const   *pp_type;
                ISymbol const *pp_sym;

                proto_type->get_parameter(i, pp_type, pp_sym);

                if (!equal_types(p_type, pp_type)) {
                    has_error           = true;
                    has_prototype_error = true;
                    error(
                        CONFLICTING_PROTOTYPE,
                        mat_decl->access_position(),
                        Error_params(*this).add_signature(proto_def));
                    add_note(
                        DECLARED_AT,
                        proto_def,
                        Error_params(*this)
                            .add_signature(proto_def));
                }
                if (p_sym != pp_sym) {
                    error(
                        REDECLARED_DIFFERENT_PARAM_NAME,
                        mat_decl->access_position(),
                        Error_params(*this)
                            .add_signature(proto_def)
                            .add_numword(i + 1)
                            .add(p_sym)
                            .add(pp_sym)
                    );
                    add_note(
                        DECLARED_AT,
                        proto_def,
                        Error_params(*this)
                            .add_signature(proto_def));
                    has_error = true;
                    // do not set has_prototype_error here, we can report more ...
                }
            }

            if (IExpression const *init_expr = param->get_init_expr()) {
                has_initializers = true;
                inits[i] = handle_parameter_initializer(init_expr, p_type, pt_name, param);
            }
        }

        // Handle enable_if annotations here after all parameters have been processed because
        // use-before-def is allowed here
        for (int i = 0; i < n_params; ++i) {
            IParameter const *param = mat_decl->get_parameter(i);
            handle_enable_ifs(param);
        }

        // create the function type
        con_type = m_tc.create_function(m_tc.material_type, params);
    }

    if (!has_error) {
        // create a new definition for this material
        con_def = m_def_tab->enter_definition(
            Definition::DK_FUNCTION, m_sym, con_type, &mat_decl->access_position());
        con_def->set_own_scope(con_scope);
        if (m_is_stdlib)
            con_def->set_flag(Definition::DEF_IS_STDLIB);

        con_def->set_declaration(mat_decl);

        if (has_initializers)
            m_module.allocate_initializers(con_def, n_params);

        // check default arguments
        bool need_default_arg = false;
        for (size_t i = 0; i < n_params; ++i) {
            if (IExpression const *init = inits[i]) {
                con_def->set_default_param_initializer(i, init);
                need_default_arg = true;
            } else if (need_default_arg) {
                IParameter const   *param  = mat_decl->get_parameter(i);
                ISimple_name const *p_name = param->get_name();

                error(
                    MISSING_DEFAULT_ARGUMENT,
                    p_name->access_position(),
                    Error_params(*this).add(i + 1).add_signature(con_def));
            }
        }

        const_cast<ISimple_name *>(m_name)->set_definition(con_def);
        mat_decl->set_definition(con_def);

        if (!has_param_error) {
            if (proto_def != NULL)
                check_prototype_default_parameter(proto_def, con_def);
        }
    }

    // now handle the body
    IExpression const *expr = NULL;
    if (body != NULL) {
        if (proto_def != NULL && !has_error) {
            proto_def->set_definite_definition(con_def);

            IDeclaration const *proto_decl = proto_def->get_declaration();
            con_def->set_prototype_declaration(proto_decl);
        }

        Definition_table::Scope_enter scope(*m_def_tab, con_scope);
        Enter_function enter(*this, con_def);
        Flag_store     inside_mat(m_inside_material_constr, true);
        Flag_store     can_thow_bounds(m_can_throw_bounds, false);
        Flag_store     can_thow_divzero(m_can_throw_divzero, false);

        // this is a function definition
        if (IStatement_expression const *stmt = as<IStatement_expression>(body)) {
            // the material expression
            visit(stmt);

            expr = stmt->get_expression();

            IType const *body_type = expr->get_type();
            if (is<IType_error>(body_type)) {
                // error already reported
                has_error = true;
            } else if (!has_error) {
                IType_struct const *s_type = as<IType_struct>(body_type);

                if (s_type == NULL || s_type->get_predefined_id() != IType_struct::SID_MATERIAL) {
                    error(
                        MATERIAL_BODY_IS_NOT_A_MATERIAL,
                        expr->access_position(),
                        Error_params(*this).add(body_type));
                    has_error = true;
                }
            }

            if (!has_error) {
                if (m_can_throw_bounds) {
                    con_def->set_flag(Definition::DEF_CAN_THROW_BOUNDS);
                }
                if (m_can_throw_divzero) {
                    con_def->set_flag(Definition::DEF_CAN_THROW_DIVZERO);
                }
            }
        } else if (is<IStatement_invalid>(body)) {
            // an invalid statement, we are ready
            visit(body);
        } else {
            visit(body);

            // not an expression
            error(MATERIAL_BODY_NOT_EXPR, body->access_position(), Error_params(*this));
        }
        if (expr == NULL) {
            // set an invalid expression into the material definition
            expr = m_module.get_expression_factory()->create_invalid();
        }
    } else {
        // declaration only
        if (!has_error) {
            con_def->set_flag(Definition::DEF_IS_DECL_ONLY);
        }
    }

    if (IAnnotation_block const *anno = mat_decl->get_annotations()) {
        Definition_store store(m_annotated_def, con_def);
        visit(anno);
    }

    if (IAnnotation_block const *ret_anno = mat_decl->get_return_annotations()) {
        Definition_store stor(m_annotated_def, con_def);
        Flag_store is_return_annotation(m_is_return_annotation, true);
        visit(ret_anno);
    }


    if (!is_error(con_def)) {
        // Don't check for is_error here: even enter it in that case to suppress further
        // errors if entries are missing.
        if (!m_is_stdlib)
            calc_mdl_versions(con_def);

        if (mat_decl->is_exported())
            export_definition(con_def);

        // add it to the call graph
        m_cg.add_node(con_def);
    }
}

// Collect the new default values for a preset.
bool NT_analysis::collect_preset_defaults(
    Definition const         *def,
    IExpression_call const   *call,
    VLA<IExpression const *> &new_defaults)
{
    IType_function const *ftype   = cast<IType_function>(def->get_type());
    int                  n_params = ftype->get_parameter_count();
    bool                 res = true;

    Name_index_map nim(
        0, Name_index_map::hasher(), Name_index_map::key_equal(), m_module.get_allocator());

    // fill the name index map
    for (int i = 0; i < n_params; ++i) {
        IType const   *p_type;
        ISymbol const *p_sym;

        ftype->get_parameter(i, p_type, p_sym);

        nim[p_sym] = i;
    }

    for (int i = 0, n = call->get_argument_count(); i < n; ++i) {
        IArgument const *arg = call->get_argument(i);

        int n_pos = -1;

        IExpression const *expr = arg->get_argument_expr();
        ISymbol const     *sym   = NULL;

        if (arg->get_kind() == IArgument::AK_NAMED) {
            IArgument_named const *narg = cast<IArgument_named>(arg);
            ISimple_name const *sname = narg->get_parameter_name();

            sym = sname->get_symbol();
            Name_index_map::const_iterator it = nim.find(sym);

            if (it != nim.end()) {
                n_pos = it->second;
            } else {
                string func_name = get_short_signature(def);

                res = false;
                error(
                    UNKNOWN_NAMED_PARAMETER,
                    arg->access_position(),
                    Error_params(*this)
                    .add(func_name.c_str())
                    .add(sym));
            }
        } else {
            n_pos = i;
        }

        if (res)
            new_defaults[n_pos] = expr;
    }
    return res;
}

// Declare a new material preset.
void NT_analysis::declare_material_preset(IDeclaration_function *mat_decl)
{
    IType_name const *ret_name = mat_decl->get_return_type_name();

    visit(ret_name);

    if (ret_name->is_incomplete_array()) {
        error(
            INCOMPLETE_ARRAY_SPECIFICATION,
            ret_name->access_position(),
            Error_params(*this));
        const_cast<IType_name*>(ret_name)->set_type(m_tc.error_type);
    }

    if (IAnnotation_block const *ret_anno = mat_decl->get_return_annotations()) {
        visit(ret_anno);
    }

    bool has_error = false;

    IType const *ret_type = as_type(ret_name);
    if (is<IType_error>(ret_type)) {
        has_error = true;
    } else {
        if (IType_struct const *s_type = as<IType_struct>(ret_type)) {
            if (s_type->get_predefined_id() != IType_struct::SID_MATERIAL) {
                has_error = true;
            }
        } else {
            has_error = true;
        }

        if (has_error) {
            ret_type = m_tc.error_type;

            error(
                WRONG_MATERIAL_PRESET,
                mat_decl->access_position(),
                Error_params(*this));
        }
    }


    Qualifier qual = ret_name->get_qualifier();
    if (qual != FQ_NONE) {
        char const *q = "";
        switch (qual) {
        case FQ_NONE:
            break;
        case FQ_VARYING:
            q = "varying";
            break;
        case FQ_UNIFORM:
            q = "uniform";
            break;
        }
        error(
            MATERIAL_QUALIFIER_NOT_ALLOW,
            mat_decl->access_position(),
            Error_params(*this).add(q));
    }

    bool                   has_let = false;
    IExpression_call const *call   = NULL;
    IStatement const       *body   = mat_decl->get_body();

    Flag_store inside_mat(m_inside_material_constr, true);

    if (IStatement_expression const *stmt = as<IStatement_expression>(body)) {
        IExpression const *expr = stmt->get_expression();

        if (m_module.get_mdl_version() >= IMDL::MDL_VERSION_1_3) {
            // MDL 1.3+ supports let expressions inside presets
            while (IExpression_let const *c_let = as<IExpression_let>(expr)) {
                has_let = true;
                expr = c_let->get_expression();
            }
        }

        if (IExpression_call const *c_expr = as<IExpression_call>(expr)) {
            // the material instance
            call = c_expr;
        } else {
            // not a material instantiation
            error(
                MATERIAL_PRESET_BODY_NOT_CALL,
                body->access_position(),
                Error_params(*this));
            has_error = true;
        }

        Expr_call_store store(m_preset_overload, call);
        visit(stmt);
    } else if (is<IStatement_invalid>(body)) {
        // an invalid statement, we are ready
        visit(body);
        has_error = true;
    } else {
        visit(body);

        // not an expression
        error(MATERIAL_BODY_NOT_EXPR, body->access_position(), Error_params(*this));
        has_error = true;
    }

    Definition const *instance_def = NULL;
    mi::base::Handle<Module const> origin;

    if (!has_error) {
        IExpression_reference const *callee = as<IExpression_reference>(call->get_reference());

        if (callee == NULL) {
            const_cast<IExpression_call *>(call)->set_type(m_tc.error_type);
            has_error = true;
        } else {
            IType_name const      *tn    = callee->get_name();
            IQualified_name const *qname = tn->get_qualified_name();
            IDefinition const     *def   = qname->get_definition();

            if (is_error(def)) {
                // previous error
                has_error = true;
            } else if (def->get_kind() == IDefinition::DK_FUNCTION) {
                if (IType_function const *ftype = as<IType_function>(def->get_type())) {
                    IType const *ret_type = ftype->get_return_type();

                    if (ret_type == m_tc.material_type) {
                        instance_def = impl_cast<Definition>(def);
                        if (instance_def->has_flag(Definition::DEF_IS_IMPORTED)) {
                            origin = mi::base::make_handle(m_module.get_owner_module(instance_def));
                        }
                    }
                }
            }

            if (!has_error && instance_def == NULL) {
                error(
                    NOT_A_MATERIAL_NAME,
                    qname->access_position(),
                    Error_params(*this).add(qname));
                has_error = true;
            }
        }
    }

    ISimple_name const *m_name = mat_decl->get_name();
    Definition         *con_def;
    if (has_error) {
error_found:
        con_def = get_error_definition();
    } else {
        IType_function const *ftype   = cast<IType_function>(instance_def->get_type());
        int                  n_params = ftype->get_parameter_count();

        VLA<IType_factory::Function_parameter> params(m_module.get_allocator(), n_params);

        for (int i = 0; i < n_params; ++i) {
            IType const   *p_type;
            ISymbol const *p_sym;

            ftype->get_parameter(i, p_type, p_sym);

            params[i].p_type = p_type;
            params[i].p_sym  = p_sym;
        }

        VLA<IExpression const *> new_defs(m_module.get_allocator(), n_params);
        memset(new_defs.data(), 0, sizeof(new_defs[0]) * n_params);

        has_error = !collect_preset_defaults(instance_def, call, new_defs);
        if (has_error)
            goto error_found;
        const_cast<IExpression_call *>(call)->set_type(m_tc.material_type);

        ISymbol const *m_sym = m_name->get_symbol();

        // create a definition for the creator function of this material instance
        IType_function const *m_con_type =
            m_tc.create_function(m_tc.material_type, params);

        con_def = get_definition_at_scope(m_sym);
        if (con_def != NULL) {
            err_redeclaration(
                Definition::DK_FUNCTION,
                con_def,
                mat_decl->access_position(),
                ENT_REDECLARATION,
                /*is_material=*/true);

            con_def = get_error_definition();
            has_error = true;
        } else {
            con_def = m_def_tab->enter_definition(
                Definition::DK_FUNCTION, m_sym, m_con_type, &mat_decl->access_position());
            if (mat_decl->is_exported()) {
                if (!instance_def->has_flag(Definition::DEF_IS_EXPORTED) &&
                    !instance_def->has_flag(Definition::DEF_IS_IMPORTED)) {
                    error(
                        EXPORTED_PRESET_OF_UNEXPORTED_ORIGIN,
                        *con_def->get_position(),
                        Error_params(*this)
                            .add(con_def->get_sym())
                            .add_signature(instance_def)
                    );
                }
                export_definition(con_def);
            }
            if (m_is_stdlib)
                con_def->set_flag(Definition::DEF_IS_STDLIB);

            con_def->set_declaration(mat_decl);

            // add it to the call graph
            m_cg.add_node(con_def);
        }

        // create new scope for the parameters and the body of the constructor
        {
            Definition_table::Scope_enter scope(*m_def_tab, con_def);

            // finish constructor
            Default_initializer_modifier def_modifier(*this, n_params, origin.get());

            bool has_initializers = false;
            bool need_default_arg = false;
            for (int i = 0; i < n_params; ++i) {
                IExpression const *init_expr;

                if (new_defs[i] != NULL) {
                    init_expr = new_defs[i];

                    // clone the expression here: if might reference let temporaries, so
                    // these must be resolved
                    if (has_let) {
                        Let_temporary_modifier let_modifier(m_module);
                        init_expr = m_module.clone_expr(init_expr, &let_modifier);
                    }
                } else {
                    IExpression const *old_init = get_default_param_initializer(instance_def, i);

                    init_expr = old_init != NULL ?
                        m_module.clone_expr(old_init, &def_modifier) : NULL;
                }

                def_modifier.set_parameter_value(i, init_expr);

                if (init_expr != NULL) {
                    need_default_arg = true;
                    if (!has_error) {
                        if (!has_initializers) {
                            m_module.allocate_initializers(con_def, n_params);
                            has_initializers = true;
                        }
                        // Beware: the initializers set here might still use not-yet imported
                        // definitions, hence we must register them for auto-import
                        con_def->set_default_param_initializer(i, init_expr);
                    }
                } else if (need_default_arg) {
                    ISymbol const *psym = get_param_sym(instance_def, i);
                    error(
                        MISSING_DEFAULT_ARGUMENT,
                        call->access_position(),
                        Error_params(*this).add(psym).add_signature(instance_def));
                }
            }
            // register for waiting fix, see above
            if (!is_error(con_def))
                m_initializers_must_be_fixed.push_back(con_def);
        }
    }

    const_cast<ISimple_name *>(m_name)->set_definition(con_def);
    mat_decl->set_definition(con_def);

    if (IAnnotation_block const *anno = mat_decl->get_annotations()) {
        Definition_store store(m_annotated_def, con_def);
        visit(anno);
    }

    if (!m_is_stdlib && !is_error(con_def)) {
        calc_mdl_versions(con_def);
    }
}

// Handle scope transitions for a select expression name lookup.
void NT_analysis::handle_select_scopes(IExpression_binary *sel_expr)
{
    IExpression const *lhs   = sel_expr->get_left_argument();
    IExpression const *n_lhs = visit(lhs);
    if (n_lhs != lhs) {
        sel_expr->set_left_argument(n_lhs);
        lhs = n_lhs;
    }

    IType const *lhs_type = lhs->get_type();

    IExpression const *rhs = sel_expr->get_right_argument();

    if (is<IType_error>(lhs_type)) {
        // already error condition
        IType_store condition(m_in_select, lhs_type);

        IExpression const *n_rhs = visit(rhs);
        if (n_rhs != rhs) {
            sel_expr->set_right_argument(n_rhs);
            rhs = n_rhs;
        }

        // kill the right type, because lhs is invalid, but later the type of the whole
        // expression is taken from this
        const_cast<IExpression *>(rhs)->set_type(m_tc.error_type);
    } else if (Scope *scope = m_def_tab->get_type_scope(lhs_type->skip_type_alias())) {
        IType_store                        condition(m_in_select, lhs_type);
        Definition_table::Scope_transition transition(*m_def_tab, scope);

        IExpression const *n_rhs = visit(rhs);
        if (n_rhs != rhs) {
            sel_expr->set_right_argument(n_rhs);
        }
    } else {
        // cannot select from this
        IExpression const *n_rhs = visit(rhs);
        if (n_rhs != rhs) {
            sel_expr->set_right_argument(n_rhs);
        }

        // kill the right type, because lhs is invalid, but later the type of the whole
        // expression is taken from this
        const_cast<IExpression *>(n_rhs)->set_type(m_tc.error_type);
    }
}

/// Get the argument of index idx from a binary or unary expression.
///
/// \param expr  a binary or unary expression
/// \param idx   the index
static IExpression const *get_argument_exp(IExpression const *expr, size_t idx)
{
    if (expr->get_kind() == IExpression::EK_BINARY) {
        IExpression_binary const *bin_expr = cast<IExpression_binary>(expr);

        return idx == 0 ? bin_expr->get_left_argument() : bin_expr->get_right_argument();
    } else {
        IExpression_unary const *un_expr = cast<IExpression_unary>(expr);

        return un_expr->get_argument();
    }
}

/// Get the argument of index idx of a binary or unary expression.
///
/// \param expr  a binary or unary expression
/// \param idx   the index
/// \param arg   the new argument
static void set_argument_expr(
    IExpression       *expr,
    size_t            idx,
    IExpression const *arg)
{
    if (expr->get_kind() == IExpression::EK_BINARY) {
        IExpression_binary *bin_expr = cast<IExpression_binary>(expr);

        idx == 0 ? bin_expr->set_left_argument(arg) : bin_expr->set_right_argument(arg);
    } else {
        IExpression_unary *un_expr = cast<IExpression_unary>(expr);

        un_expr->set_argument(arg);
    }
}

// Resolve an overload.
NT_analysis::Definition_list NT_analysis::resolve_operator_overload(
    IExpression const      *call_expr,
    Definition_list        &def_list,
    IType const            *arg_types[],
    size_t                 num_args,
    unsigned               &arg_mask)
{
    arg_mask = 0;
    Memory_arena arena(m_module.get_allocator());

    Signature_list best_matches(&arena);

    int best_value = 0;

    VLA<IType const *> signature(m_module.get_allocator(), num_args);

    unsigned matching_types = 0;
    for (Definition_list::iterator it(def_list.begin()), end(def_list.end());
        it != end;
        ++it)
    {
        Definition const     *def  = *it;
        IType const          *type = def->get_type();

        // TODO: there should be no errors at this place
        if (is<IType_error>(type))
            continue;

        IType_function const *func_type = cast<IType_function>(type);

        int this_value = 0;
        bool type_must_match = def->has_flag(Definition::DEF_OP_LVALUE);
        for (size_t k = 0; k < num_args; ++k, type_must_match = false) {
            IType const   *param_type;
            ISymbol const *name;

            IType const *arg_type = arg_types[k];
            func_type->get_parameter(k, param_type, name);

            param_type = param_type->skip_type_alias();

            bool new_bound = false;
            bool need_explicit_conv = false;
            if (equal_types(param_type, arg_type)) {
                // types match
                this_value += 3;
                matching_types |= (1U << k);
            } else if (!type_must_match &&
                       can_assign_param(param_type, arg_type, new_bound, need_explicit_conv)) {
                // can be implicitly converted (or explicit with warning)
                this_value += need_explicit_conv ? 1 : 2;
                matching_types |= (1U << k);
            } else {
                // this operator failed
                goto failed_operator;
            }
            signature[k] = param_type;
        }

        if (this_value > best_value) {
            // higher values mean "more specific"
            best_matches.clear();
            Signature_entry entry(
                (IType const *const *)Arena_memdup(
                    arena, signature.data(), signature.size() * sizeof(signature[0])),
                /*bounds=*/NULL,
                signature.size(),
                def);
            best_matches.push_back(entry);
            best_value = this_value;
        } else if (this_value == best_value) {
            Signature_entry entry(
                (IType const *const *)Arena_memdup(
                    arena, signature.data(), signature.size() * sizeof(signature[0])),
                /*bounds=*/NULL,
                signature.size(),
                def);
            if (kill_less_specific(best_matches, entry)) {
                best_matches.push_back(entry);
            }
        }
failed_operator:;
    }

    arg_mask = matching_types;

    Definition_list res(m_module.get_allocator());

    for (Signature_list::iterator it(best_matches.begin()), end(best_matches.end());
        it != end;
        ++it)
    {
        Signature_entry const &entry = *it;
        res.push_back(entry.def);
    }
    return res;
}

// Find the definition of a binary or unary operator.
Definition const *NT_analysis::find_operator(
    IExpression           *op_call,
    IExpression::Operator op,
    IType const           *arg_types[],
    size_t                arg_count)
{
    IType const *mod_arg_types[2] = { 0, 0 };

    for (size_t i = 0; i < arg_count; ++i) {
        if (is<IType_error>(arg_types[i])) {
            // error already reported
            return get_error_definition();
        }
        mod_arg_types[i] = arg_types[i]->skip_type_alias();
    }

    if (is_comparison(op) &&
        mod_arg_types[0] != mod_arg_types[1] &&
        (is<IType_enum>(mod_arg_types[0]) || is<IType_enum>(mod_arg_types[1]))) {
        error(
            WRONG_COMPARISON_BETWEEN_ENUMS,
            op_call->access_position(),
            Error_params(*this).add(mod_arg_types[0]).add(mod_arg_types[1]));
        return get_error_definition();
    }

    // try fast lookup first
    Definition const *def = m_op_cache.lookup(
        op, mod_arg_types[0], arg_count > 1 ? mod_arg_types[1] : (IType const *)0);

    if (def == NULL) {
        // lookup failed, search it

        IAllocator *alloc = m_module.get_allocator();

        // collect the possible set
        Definition_list function_list(alloc);

        size_t cnt = 0;
        bool oo_like = false;

        if (op == IExpression::OK_ASSIGN) {
            // assignment operators might be defined OO-like inside the type scope
            IType const *this_type = mod_arg_types[0];

            if (Scope const *scope = m_def_tab->get_type_scope(this_type)) {
                ISymbol const *op_sym = m_st->get_operator_symbol(op);
                Definition    *op_def = scope->find_definition_in_scope(op_sym);

                if (op_def != NULL && !op_def->has_flag(Definition::DEF_IGNORE_OVERLOAD)) {
                    function_list.push_back(op_def);
                    ++cnt;
                    oo_like = true;
                }
            }
        }

        if (!oo_like) {
            // not found in the type scope, lookup global operators
            for (def = m_def_tab->get_operator_definition(op);
                 def != NULL;
                 def = def->get_prev_def())
            {
                if (!def->has_flag(Definition::DEF_IGNORE_OVERLOAD)) {
                    function_list.push_back(def);
                    ++cnt;
                }
            }
        }

        MDL_ASSERT(cnt > 0 && "No definition for a binary/unary builtin operator");

        // resolve it
        unsigned arg_mask = 0;
        Definition_list overloads = resolve_operator_overload(
            op_call,
            function_list,
            mod_arg_types,
            arg_count,
            arg_mask);

        bool has_error = false;
        if ((arg_mask & 1) == 0) {
            // could not find an operator for the left (or single) operand type
            if (is<IType_function>(mod_arg_types[0])) {
                error(
                    FUNCTION_PTR_USED,
                    *get_argument_position(op_call, 0),
                    Error_params(*this));
                if (arg_count == 2) {
                    // suppress error for right argument if any
                    arg_mask |= 2;
                }
            } else if (is<IType_array>(mod_arg_types[0]) && is_assign_operator(op)) {
                error(
                    ARRAY_ASSIGNMENT_FORBIDDEN,
                    *get_argument_position(op_call, 0),
                    Error_params(*this)
                        .add(op)
                );
                // suppress error for right argument
                arg_mask |= 2;
            } else {
                error(
                    arg_count == 1 &&
                    (op != IExpression::OK_POST_INCREMENT &&
                     op != IExpression::OK_POST_DECREMENT) ?
                        OPERATOR_RIGHT_ARGUMENT_ILLEGAL :
                        OPERATOR_LEFT_ARGUMENT_ILLEGAL,
                    *get_argument_position(op_call, 0),
                    Error_params(*this)
                        .add(op)
                        .add(arg_types[0])
                );
            }
            has_error = true;
        }
        if (arg_count == 2 && ((arg_mask & 2) == 0)) {
            // could not find an operator for the right operand type
            if (is<IType_function>(mod_arg_types[1])) {
                error(
                    FUNCTION_PTR_USED,
                    *get_argument_position(op_call, 1),
                    Error_params(*this));
            } else {
                error(
                    OPERATOR_RIGHT_ARGUMENT_ILLEGAL,
                    *get_argument_position(op_call, 1),
                    Error_params(*this)
                        .add(op)
                        .add(arg_types[1])
                );
            }
            has_error = true;
        }
        if (has_error)
            return get_error_definition();

        if (overloads.empty()) {
            // we have an empty overload set
            if (arg_count == 2) {
                error(
                    AMBIGUOUS_BINARY_OPERATOR,
                    op_call->access_position(),
                    Error_params(*this).add(op).add(arg_types[0]).add(arg_types[1]));
            } else {
                error(
                    AMBIGUOUS_UNARY_OPERATOR,
                    op_call->access_position(),
                    Error_params(*this).add(op).add(arg_types[0]));
            }
            return get_error_definition();
        }
        Definition_list::iterator it = overloads.begin();
        def = *it;
        if (++it != overloads.end()) {
            // ambiguous operators found
            if (arg_count == 2) {
                error(
                    AMBIGUOUS_BINARY_OPERATOR,
                    op_call->access_position(),
                    Error_params(*this).add(op).add(arg_types[0]).add(arg_types[1]));
            } else {
                error(
                    AMBIGUOUS_UNARY_OPERATOR,
                    op_call->access_position(),
                    Error_params(*this).add(op).add(arg_types[0]));
            }
            return get_error_definition();
        }

        // All fine, found only ONE possible overload, cache it.
        if (arg_count <= 2) {
            // cache the result
            m_op_cache.insert(
                op, arg_types[0], arg_count > 1 ? arg_types[1] : (IType const *)0, def);
        }
    }

    // Finally add argument conversions if needed.
    IType_function const *func_type = cast<IType_function>(def->get_type());

    for (size_t k = 0; k < arg_count; ++k) {
        IType const   *param_type;
        ISymbol const *param_sym;
        func_type->get_parameter(k, param_type, param_sym);

        param_type = param_type->skip_type_alias();

        IType const *arg_type = arg_types[k];
        if (need_type_conversion(arg_type, param_type)) {
            IExpression const *expr = get_argument_exp(op_call, k);
            IExpression const *conv = convert_to_type_implicit(expr, param_type);

            MDL_ASSERT(conv != NULL && "Failed to find implicit conversion constructor");
            set_argument_expr(op_call, k, conv);
        }
    }
    return def;
}

// Check if there is an array assignment operator.
Definition const *NT_analysis::find_array_assignment(
    IExpression_binary *op_call,
    IType const        *left_tp,
    IType const        *right_tp)
{
    left_tp  = left_tp->skip_type_alias();
    right_tp = right_tp->skip_type_alias();

    if (!is<IType_array>(left_tp) || !is<IType_array>(right_tp))
        return NULL;

    IType_array const *a_l_tp = cast<IType_array>(left_tp);
    IType_array const *a_r_tp = cast<IType_array>(right_tp);

    IType const *e_tp = a_l_tp->get_element_type()->skip_type_alias();

    if (a_l_tp != a_r_tp) {
        bool is_sized = a_l_tp->is_immediate_sized();
        if (is_sized != a_r_tp->is_immediate_sized()) {
            // cannot convert from immediate <==> deferred size
            error(
                INVALID_CONVERSION,
                op_call->access_position(),
                Error_params(*this).add(a_r_tp).add(a_l_tp));
            return get_error_definition();
        }
        if (is_sized) {
            if (a_l_tp->get_size() != a_r_tp->get_size()) {
                // cannot assign different length
                error(
                    INVALID_CONVERSION,
                    op_call->access_position(),
                    Error_params(*this).add(a_r_tp).add(a_l_tp));
                return get_error_definition();
            }
        } else {
            if (a_l_tp->get_deferred_size() != a_r_tp->get_deferred_size()) {
                // cannot assign different length
                error(
                    INVALID_CONVERSION,
                    op_call->access_position(),
                    Error_params(*this).add(a_r_tp).add(a_l_tp));
                return get_error_definition();
            }
        }
        IType const *e_r_tp = a_r_tp->get_element_type()->skip_type_alias();

        if (e_tp != e_r_tp) {
            // cannot assign different element types
            error(
                INVALID_CONVERSION,
                op_call->access_position(),
                Error_params(*this).add(a_r_tp).add(a_l_tp));
            return get_error_definition();
        }
    }

    // get the assignment operator

    IExpression::Operator op = IExpression::OK_ASSIGN;

    IType const *op_type;
    if (a_l_tp->is_immediate_sized()) {
        op_type = m_tc.create_array(e_tp, a_l_tp->get_size());
    } else {
        op_type = m_tc.create_array(e_tp, a_l_tp->get_deferred_size());
    }

    // try fast lookup first
    Definition const *def = m_op_cache.lookup(op, op_type, op_type);

    if (def == NULL) {
        // lookup failed, search it

        for (def = m_def_tab->get_operator_definition(op);
            def != NULL;
            def = def->get_prev_def())
        {
            if (!def->has_flag(Definition::DEF_IGNORE_OVERLOAD)) {
                IType_function const *f_tp = cast<IType_function>(def->get_type());

                IType const   *p_tp;
                ISymbol const *p_sym;

                f_tp->get_parameter(0, p_tp, p_sym);
                if (p_tp != op_type)
                    continue;

                f_tp->get_parameter(1, p_tp, p_sym);
                if (p_tp != op_type)
                    continue;

                // found it
                break;
            }
        }

        if (def == NULL) {
            // The operator was not found, create it on demand.
            // there are too many array assignment operator to create them in advance.
            ISymbol const *op_sym = m_st->get_operator_symbol(op);

            IType_factory::Function_parameter params[] = {
                { op_type, m_st->get_predefined_symbol(ISymbol::SYM_PARAM_X) },
                { op_type, m_st->get_predefined_symbol(ISymbol::SYM_PARAM_Y) }
            };

            IType_function const *f_tp = m_tc.create_function(op_type, params);

            Scope *predef_scope = m_def_tab->get_predef_scope();
            Scope *curr_scope   = m_def_tab->get_curr_scope();

            // into the predef scope
            m_def_tab->transition_to_scope(predef_scope);
            Definition *ndef = m_def_tab->enter_operator_definition(op, op_sym, f_tp);
            m_def_tab->transition_to_scope(curr_scope);

            ndef->set_flag(Definition::DEF_OP_LVALUE);

            def = ndef;
        }

        // cache the result
        m_op_cache.insert(op, op_type, op_type, def);
    }
    return def;
}

// Find the Definition of a conversion constructor or an conversion operator.
Definition *NT_analysis::do_find_conversion(
    IType const *from_tp,
    IType const *to_tp,
    bool        implicit_only)
{
    MDL_ASSERT(!is<IType_error>(from_tp) && !is<IType_error>(to_tp));

    IType const *dt = to_tp->skip_type_alias();
    if (is<IType_array>(dt) || is<IType_function>(dt)) {
        // cannot convert to array or function
        return NULL;
    }

    IType const *st = from_tp->skip_type_alias();

    // in the MDL compiler all implicit conversions are implemented "the OO-way", i.e.
    // to convert from st to dt, there must either exists an implicit dt(st) constructor
    // or an operator st(dt).
    Scope const *src_scope = m_def_tab->get_type_scope(st);
    if (src_scope == NULL)
        return NULL;

    Scope const *dst_scope = m_def_tab->get_type_scope(dt);
    if (dst_scope == NULL) {
        // If this happens, something really bad goes from, for instance the type of an
        // argument is a function type
        MDL_ASSERT(!"do_find_conversion(): destination type has no scope");
        return NULL;
    }

    ISymbol const *construct_sym = dst_scope->get_scope_name();

    Definition *def = NULL;

    // search first for operator TYPE()
    Definition *set = src_scope->find_definition_in_scope(construct_sym);
    for (def = set; def != NULL; def = def->get_prev_def()) {
        if (def->get_semantics() != Definition::DS_CONV_OPERATOR) {
            // search conversion operator
            continue;
        }
        IType_function const *f_type = cast<IType_function>(def->get_type());

        MDL_ASSERT(f_type->get_parameter_count() == 1);

        IType const *ret_type = f_type->get_return_type();

        if (ret_type->skip_type_alias() == dt) {
            // found it
            return def;
        }
    }

    // not found, search for an implicit constructor TYPE()
    set = dst_scope->find_definition_in_scope(construct_sym);
    for (def = set; def != NULL; def = def->get_prev_def()) {
        if (implicit_only && def->has_flag(Definition::DEF_IS_EXPLICIT)) {
            // we are searching for implicit constructors only
            continue;
        }
        if (def->get_kind() != Definition::DK_CONSTRUCTOR) {
            // ignore operators/members ...
            continue;
        }
        IType_function const *f_type = cast<IType_function>(def->get_type());

        if (f_type->get_parameter_count() != 1) {
            // not exactly one argument: not a conversion
            continue;
        }
        IType const   *p_type;
        ISymbol const *p_sym;
        f_type->get_parameter(0, p_type, p_sym);

        if (p_type->skip_type_alias() == st) {
            // found it
            return def;
        }
    }
    // not found
    return NULL;
}

// Find the Definition of an implicit conversion constructor or an conversion
// operator.
Definition *NT_analysis::find_implicit_conversion(
    IType const *from_tp,
    IType const *to_tp)
{
    return do_find_conversion(from_tp, to_tp, /*implicit_only=*/true);
}

// Find the Definition of an implicit or explicit conversion constructor or an conversion
// operator.
Definition *NT_analysis::find_type_conversion(
    IType const *from_tp,
    IType const *to_tp)
{
    return do_find_conversion(from_tp, to_tp, /*implicit_only=*/false);
}

// Convert a given expression to the destination type using a given conversion constructor.
IExpression_call *NT_analysis::create_type_conversion_call(
    IExpression const *expr,
    Definition const  *def,
    bool              warn_if_explicit)
{
    Position const     &pos  = expr->access_position();
    Expression_factory *fact = m_module.get_expression_factory();

    IExpression_reference *ref    = create_reference(def, pos);
    IExpression_call      *call   = fact->create_call(ref, POS(pos));
    IType const           *res_tp = get_result_type(def);
    call->set_type(check_performance_restriction(res_tp, pos));

    IArgument_positional const *arg = fact->create_positional_argument(expr, POS(pos));
    call->add_argument(arg);

    if (warn_if_explicit && def->has_flag(Definition::DEF_IS_EXPLICIT_WARN)) {
        error_strict(
            TYPE_CONVERSION_MUST_BE_EXPLICIT,
            pos,
            Error_params(*this).add(expr->get_type()->skip_type_alias()).add(res_tp));
    }

    // check if default arguments must be added
    IType_function const *f_type = cast<IType_function>(def->get_type());

    int n_params = f_type->get_parameter_count();
    if (n_params > 1) {
        // FIXME: need origin
        Default_initializer_modifier def_modifier(*this, n_params, NULL);

        for (int i = 1; i < n_params; ++i) {
            IExpression const *expr = get_default_param_initializer(def, i);
            MDL_ASSERT(expr != NULL);

            expr = m_module.clone_expr(expr, &def_modifier);
            IArgument_positional const *arg = fact->create_positional_argument(expr, POS(pos));
            call->add_argument(arg);
        }
    }
    return call;
}

// Convert a given expression to the destination type if an implicit
// constructor is available.
IExpression *NT_analysis::convert_to_type_implicit(
    IExpression const *expr,
    IType const *dst_type)
{
    IType const *src_type = expr->get_type()->skip_type_alias();

    if (is<IType_error>(src_type)) {
        // no conversion from error type
        return NULL;
    }

    if (is<IType_enum>(src_type) && !is<IType_int>(dst_type->skip_type_alias())) {
        // there are no conversions from enum type, first convert to int
        Definition *def = find_implicit_conversion(src_type, m_tc.int_type);
        if (def == NULL) {
            // no conversion found
            return NULL;
        }
        expr = create_type_conversion_call(expr, def, /*warn_if_explicit=*/true);
        src_type = m_tc.int_type;
    }

    Definition *def = find_implicit_conversion(src_type, dst_type);
    if (def == NULL) {
        // no conversion found
        return NULL;
    }
    return create_type_conversion_call(expr, def, /*warn_if_explicit=*/true);
}

// Convert a given expression to the destination type if an implicit or explicit
// constructor is available.
IExpression *NT_analysis::convert_to_type_explicit(
    IExpression const *expr,
    IType const       *dst_type)
{
    IType const *src_type = expr->get_type()->skip_type_alias();

    if (is<IType_error>(src_type)) {
        // no conversion from error type
        return NULL;
    }

    if (is<IType_enum>(src_type) && !is<IType_int>(dst_type->skip_type_alias())) {
        // there are no conversions from enum type, first convert to int
        Definition *def = find_type_conversion(src_type, m_tc.int_type);
        if (def == NULL) {
            // no conversion found
            return NULL;
        }
        expr = create_type_conversion_call(expr, def, /*warn_if_explicit=*/false);
        src_type = m_tc.int_type;
    }

    Definition *def = find_type_conversion(src_type, dst_type);
    if (def == NULL) {
        // no conversion found
        return NULL;
    }
    return create_type_conversion_call(expr, def, /*warn_if_explicit=*/false);
}

// Convert a given const expression to the destination type if an implicit
// constructor is available.
IExpression *NT_analysis::convert_to_type_and_fold(
    IExpression const *expr,
    IType const       *dst_type)
{
    if (IExpression const *call = convert_to_type_implicit(expr, dst_type)) {
        IValue const *v = call->fold(&m_module, m_module.get_value_factory(), NULL);
        if (!is<IValue_bad>(v)) {
            Position const *pos = &expr->access_position();
            return m_module.create_literal(v, pos);
        }
    }
    return NULL;
}

// Check if the given expression represents a boolean condition.
IExpression *NT_analysis::check_bool_condition(
    IExpression const *cond,
    bool              &has_error,
    bool              inside_enable_if)
{
    IType const *cond_type = cond->get_type();
    if (is<IType_error>(cond_type)) {
        has_error = true;
        return NULL;
    }
    if (!is_bool_type(cond_type)) {
        if (IExpression *conv = convert_to_type_implicit(cond, m_tc.bool_type)) {
            // insert an explicit conversion
            return conv;
        } else {
            error(
                CONDITION_BOOL_TYPE_REQUIRED, cond->access_position(),
                Error_params(*this).add(cond_type));
            has_error = true;
        }
    } else if (!inside_enable_if) {
        // warn if a '=' is used, which might be the indication of an error
        if (IExpression_binary const *b_expr = as<IExpression_binary>(cond)) {
            if (!b_expr->in_parenthesis() &&
                b_expr->get_operator() == IExpression_binary::OK_ASSIGN) {

                warning(
                    SUGGEST_PARENTHESES_AROUND_ASSIGNMENT,
                    b_expr->access_position(),
                    Error_params(*this));
            }
        }
    }
    return NULL;
}

/// Update version flags from a type.
///
/// \param def_tab      the current definition table
/// \param type         the type to check
/// \param since_ver    the since version
/// \param removed_ver  the removed version
static void update_flags(
    Definition_table *def_tab,
    IType const      *type,
    unsigned         &since_ver,
    unsigned         &removed_ver)
{
    type = type->skip_type_alias();
    if (is_material_type_or_sub_type(type)) {
        // material and subtypes can only be used in materials itself where they alwys are
        // promoted to the lastest version, hence there is no restriction
        return;
    }
    if (Scope *scope = def_tab->get_type_scope(type)) {
        if (Definition const *type_def = scope->get_owner_definition()) {
            unsigned flags = type_def->get_version_flags();
            since_ver   = max(since_ver,   mdl_since_version(flags));
            removed_ver = min(removed_ver, mdl_removed_version(flags));
        }
    }
}

/// Update version flags from a symbol.
///
/// \param sym          the symbol to check
/// \param since_ver    the since version
/// \param removed_ver  the removed version
static void update_flags(
     ISymbol const *sym,
     unsigned      &since_ver,
     unsigned      &removed_ver)
{
    // Version flags of predefined entities.
    enum Version_flags {
        SINCE_1_1   = IMDL::MDL_VERSION_1_1,
        REMOVED_1_1 = (IMDL::MDL_VERSION_1_1 << 8),
        SINCE_1_2   = IMDL::MDL_VERSION_1_2,
        REMOVED_1_2 = (IMDL::MDL_VERSION_1_2 << 8),
        SINCE_1_3   = IMDL::MDL_VERSION_1_3,
        REMOVED_1_3 = (IMDL::MDL_VERSION_1_3 << 8),
        SINCE_1_4   = IMDL::MDL_VERSION_1_4,
        REMOVED_1_4 = (IMDL::MDL_VERSION_1_4 << 8),
        SINCE_1_5   = IMDL::MDL_VERSION_1_5,
        REMOVED_1_5 = (IMDL::MDL_VERSION_1_5 << 8),
        SINCE_1_6   = IMDL::MDL_VERSION_1_6,
        REMOVED_1_6 = (IMDL::MDL_VERSION_1_6 << 8),
        SINCE_1_7   = IMDL::MDL_VERSION_1_7,
        REMOVED_1_7 = (IMDL::MDL_VERSION_1_7 << 8),
    };

    size_t id = sym->get_id();
    if (id < ISymbol::SYM_FREE) {
        switch (ISymbol::Predefined_id(id)) {
        case ISymbol::SYM_TYPE_BSDF_MEASUREMENT:
        case ISymbol::SYM_TYPE_INTENSITY_MODE:
        case ISymbol::SYM_ENUM_INTENSITY_RADIANT_EXITANCE:
        case ISymbol::SYM_ENUM_INTENSITY_POWER:
            // keyword in MDL 1.1, so it cannot be used as identifier
            since_ver = max(since_ver, REMOVED_1_1);
            break;
        case ISymbol::SYM_TYPE_HAIR_BSDF:
            // keyword in MDL 1.5, so it cannot be used as identifier
            since_ver = max(since_ver, REMOVED_1_5);
            break;
        default:
            break;
        }
    }
}

/// Update version flags from an expression.
///
/// \param def_tab      the current definition table
/// \param expr         the expression to check
/// \param since_ver    the since version
/// \param removed_ver  the removed version
static void update_flags(
     Definition_table  *def_tab,
     IExpression const *expr,
     unsigned          &since_ver,
     unsigned          &removed_ver)
{
    /// Helper class to visit expressions.
    class Expr_visitor : public Module_visitor {
        typedef Module_visitor Base;
    public:
        /// Constructor.
        ///
        /// \param def_tab      the current definition table
        /// \param since_ver    the since version
        /// \param removed_ver  the removed version
        Expr_visitor(Definition_table *def_tab, unsigned &since_ver, unsigned &removed_ver)
        : Base()
        , m_def_tab(def_tab)
        , m_since_ver(since_ver)
        , m_removed_ver(removed_ver)
        {
        }

        /// Default post visitor for expressions.
        ///
        /// \param expr  the expression
        ///
        /// Overwrite this method if some general processing for every
        /// not explicitly overwritten expression is needed.
        IExpression *post_visit(IExpression *expr) MDL_FINAL
        {
            IType const *type = expr->get_type();

            update_flags(m_def_tab, type, m_since_ver, m_removed_ver);
            return expr;
        }

        /// Handle reference expressions.
        IExpression *post_visit(IExpression_reference *expr) MDL_FINAL
        {
            if (Definition const *def = impl_cast<Definition>(expr->get_definition())) {
                if (def->get_kind() == IDefinition::DK_CONSTRUCTOR) {
                    IType_function const *ftype = cast<IType_function>(def->get_type());

                    if (is_material_type_or_sub_type(ftype->get_return_type())) {
                        // material and subtypes can only be used inside materials,
                        // where always a promotion exists, hence no restriction
                        // on the usage of those constructors
                        return expr;
                    }
                }
                unsigned flags = def->get_version_flags();

                m_since_ver   = max(m_since_ver,   mdl_since_version(flags));
                m_removed_ver = min(m_removed_ver, mdl_removed_version(flags));
            }
            return expr;
        }

    private:
        /// The current definition table.
        Definition_table *m_def_tab;

        /// The since version.
        unsigned         &m_since_ver;

        /// The removed version.
        unsigned         &m_removed_ver;
    };

    Expr_visitor visitor(def_tab, since_ver, removed_ver);
    (void)visitor.visit(expr);
}

// Calculate the since and removed version for a given definition.
void NT_analysis::calc_mdl_versions(Definition *def)
{
    unsigned since_ver   = IMDL::MDL_VERSION_1_0;
    unsigned removed_ver = 0xFFFFFFFF;

    IType const *type = def->get_type();
    if (IType_function const *func_type = as<IType_function>(type)) {
        // check the signature types

        // check the name of the function itself
        update_flags(def->get_sym(), since_ver, removed_ver);

        if (IType const *ret_type = func_type->get_return_type()) {
            update_flags(m_def_tab, ret_type, since_ver, removed_ver);
        }

        for (int i = 0, n = func_type->get_parameter_count(); i < n; ++i) {
            IType const   *p_type;
            ISymbol const *p_sym;

            func_type->get_parameter(i, p_type, p_sym);

            update_flags(m_def_tab, p_type, since_ver, removed_ver);
            update_flags(p_sym, since_ver, removed_ver);

            if (IExpression const *def_value = def->get_default_param_initializer(i)) {
                update_flags(m_def_tab, def_value, since_ver, removed_ver);
            }
        }
    } else {
        // check only the type
        update_flags(m_def_tab, type, since_ver, removed_ver);
    }
    if (removed_ver == 0xFFFFFFFF) {
        // unchanged
        removed_ver = 0;
    }
    def->set_version_flags(since_ver | (removed_ver << 8));
}

// Export the given definition.
void NT_analysis::export_definition(Definition *def)
{
    def->set_flag(Definition::DEF_IS_EXPORTED);

    if (def->has_flag(Definition::DEF_IS_DECL_ONLY)) {
        if (!def->has_flag(Definition::DEF_IS_STDLIB)) {
            // do not put declarations into the export list, but remember
            // for later check
            m_exported_decls_only.push_back(def);
            return;
        }
    }

    // do not register enum values, it is not expected that they are visible
    // in the exported list, although they are automatically exported together
    // with there type (because they live outside of their type scope
    if (def->get_kind() != Definition::DK_ENUM_VALUE) {
        if (is_available_in_mdl(m_mdl_version, def->get_version_flags())) {
            // do not export things that are not available yet
            m_module.register_exported(def);
        } else if (m_module.is_stdlib() && def->get_kind() == IDefinition::DK_FUNCTION) {
            IType_function const *f_tp = cast<IType_function>(def->get_type());
            if (!is_material_type(f_tp->get_return_type())) {
                // for the standard library export ALL functions even if not available in the
                // current MDL version (because removed)
                m_module.register_exported(def);
            }
        }
    }
}

// Compare two signature entries representing functions for "specific-ness".
bool NT_analysis::is_more_specific(Signature_entry const &a, Signature_entry const &b)
{
    size_t n_params = a.sig_length;

    MDL_ASSERT(n_params == b.sig_length);

    for (size_t i = 0; i < n_params; ++i) {
        IType const *param_a = a.signature[i];
        IType const *param_b = b.signature[i];

        param_a = param_a->skip_type_alias();
        param_b = param_b->skip_type_alias();

        if (equal_types(param_a, param_b)) {
            if (is<IType_array>(param_a) && is<IType_array>(param_b)) {
                if (a.bounds != NULL && b.bounds != NULL) {
                    if (a.bounds[i] && !b.bounds[i])
                        return false;
                }
            }
            continue;
        }

        if (is<IType_enum>(param_b)) {
            switch (param_a->get_kind()) {
            case IType::TK_ENUM:
                // no enum is more specific
            case IType::TK_INT:
                // enum -> int
            case IType::TK_FLOAT:
                // enum -> float
            case IType::TK_DOUBLE:
                // enum -> double
                return false;
            default:
                break;
            }
        }

        if (find_implicit_conversion(param_b, param_a) != NULL) {
            return false;
        }
    }
    return true;
}

// Given a list and a (call) signature, kill any less specific definition from the list.
bool NT_analysis::kill_less_specific(Signature_list &list, Signature_entry const &new_sig)
{
    for (Signature_list::iterator it(list.begin()), end(list.end()); it != end; ++it) {
        Signature_entry const &curr_sig = *it;

        bool curr_is_more = is_more_specific(curr_sig, new_sig);
        bool new_is_more  = is_more_specific(new_sig, curr_sig);

        if (curr_is_more && !new_is_more) {
            // current def is more specific than new def, no need for new def
            return false;
        }
        if (!curr_is_more && new_is_more) {
            // current def is less specific the new def, kill it
            list.erase(it);
            return true;
        }

        // FIXME: is the following still true?
        // Note: due to the policy that import "redefinitions" are only reported
        // at use, it CAN happen that we find two definitions that are really the same,
        // in which case (curr_is_more && new_is_more) == true
        // Return true in THAT case, so both are added and an overload error is reported.
        if (curr_is_more && new_is_more) {
            return true;
        }
    }
    return true;
}

// Check if a parameter type is already bound.
bool NT_analysis::is_bound_type(IType_array const *abs_type) const
{
    Bind_type_map::const_iterator it = m_type_bindings.find(abs_type);
    return it != m_type_bindings.end();
}

/// Checks if a array type needs type binding.
static bool needs_binding(IType_array const *dst, IType_array const *src)
{
    if (dst->is_immediate_sized()) {
        // the destination type IS already immediate, no binding possible
        MDL_ASSERT(
            src->is_immediate_sized() &&
            dst->get_size() == src->get_size() &&
            src->get_element_type()->skip_type_alias() ==
            dst->get_element_type()->skip_type_alias());
        return false;
    } else {
        // can be bound
        MDL_ASSERT(
            src->get_element_type()->skip_type_alias() ==
            dst->get_element_type()->skip_type_alias());
        return true;
    }
}

// Bind the given deferred type to a type.
void NT_analysis::bind_array_type(
    IType_array const *deferred_type,
    IType_array const *type)
{
    MDL_ASSERT(!deferred_type->is_immediate_sized() && "Wrong type binding");

    IType_array_size const *abs_size = deferred_type->get_deferred_size();
    if (type->is_immediate_sized()) {
        m_size_bindings[abs_size] = type->get_size();
    } else {
        m_sym_bindings[abs_size] = type->get_deferred_size();
    }

    // bind the size NOT the element type
    IType const *e_type = deferred_type->get_element_type();
    IType const *n_type;
    if (type->is_immediate_sized()) {
        n_type = m_tc.create_array(e_type, type->get_size());
    } else {
        n_type = m_tc.create_array(e_type, type->get_deferred_size());
    }
    m_type_bindings[deferred_type] = n_type;
}

// Return the bound type for a deferred type.
IType const *NT_analysis::get_bound_type(IType const *type)
{
    Bind_type_map::const_iterator it = m_type_bindings.find(type);
    if (it != m_type_bindings.end())
        return it->second;

    if (IType_array const *a_type = as<IType_array>(type)) {
        // check if the size is bound
        if (!a_type->is_immediate_sized()) {
            IType_array_size const *abs_size = a_type->get_deferred_size();

            Bind_size_map::const_iterator sit = m_size_bindings.find(abs_size);
            if (sit != m_size_bindings.end()) {
                int size = sit->second;

                IType const *e_type = a_type->get_element_type();

                IType const *r_type = m_tc.create_array(e_type, size);

                m_type_bindings[type] = r_type;
                return r_type;
            }

            Bind_symbol_map::const_iterator ait = m_sym_bindings.find(abs_size);
            if (ait != m_sym_bindings.end()) {
                IType_array_size const *size = ait->second;

                IType const *e_type = a_type->get_element_type();

                IType const *r_type = m_tc.create_array(e_type, size);

                m_type_bindings[type] = r_type;
                return r_type;
            }
        }
    }
    return type;
}

// Clear all bindings of deferred sized array types.
void NT_analysis::clear_type_bindings()
{
    m_type_bindings.clear();
    m_size_bindings.clear();
    m_sym_bindings.clear();
}

// Check if it is possible to assign an argument type to the parameter
// type of a call.
bool NT_analysis::can_assign_param(
    IType const *param_type,
    IType const *arg_type,
    bool        &new_bound,
    bool        &need_explicit_conv)
{
    new_bound          = false;
    need_explicit_conv = false;
    if (param_type == arg_type)
        return true;

    if (is<IType_error>(param_type)) {
        // this should only happen in materials, where overload is forbidden, so
        // we can create definitions with error parameters to improve error checking
        return true;
    }

    if (as<IType_enum>(arg_type) != NULL) {
        // special case for enums: an enum can be assigned to int, float, double
        IType const *base = param_type->skip_type_alias();
        IType::Kind kind = base->get_kind();

        bool res = (kind == IType::TK_INT || kind == IType::TK_FLOAT || kind == IType::TK_DOUBLE);
        if (res) {
            return true;
        }
        return false;
    }

    if (IType_array const *a_param_type = as<IType_array>(param_type)) {
        if (IType_array const *a_arg_type = as<IType_array>(arg_type)) {
            if (!a_param_type->is_immediate_sized()) {
                // the parameter type is abstract, check for bindings
                a_param_type = cast<IType_array>(get_bound_type(a_param_type));
            }

            if (a_param_type->is_immediate_sized()) {
                // concrete parameter type, size must match
                if (!a_arg_type->is_immediate_sized()) {
                    return false;
                }
                if (a_param_type->get_size() != a_arg_type->get_size()) {
                    return false;
                }
                return equal_types(
                    a_param_type->get_element_type()->skip_type_alias(),
                    a_arg_type->get_element_type()->skip_type_alias());
            } else {
                // param type is an deferred size array
                if (a_arg_type->is_immediate_sized()) {
                    // can pass a immediate size array to a deferred size parameter, but this will
                    // bind the parameter type
                    bool res = equal_types(
                        a_param_type->get_element_type()->skip_type_alias(),
                        a_arg_type->get_element_type()->skip_type_alias());
                    if (res) {
                        new_bound = true;
                        bind_array_type(a_param_type, a_arg_type);
                    }
                    return res;
                } else {
                    if (is_bound_type(a_param_type)) {
                        // must use the same deferred size
                        if (a_param_type->get_deferred_size() != a_arg_type->get_deferred_size())
                            return false;
                        return equal_types(
                            a_param_type->get_element_type()->skip_type_alias(),
                            a_arg_type->get_element_type()->skip_type_alias());
                    } else {
                        // can pass a deferred size array to a deferred size parameter, but
                        // this will bind the parameter type
                        bool res = equal_types(
                            a_param_type->get_element_type()->skip_type_alias(),
                            a_arg_type->get_element_type()->skip_type_alias());
                        if (res) {
                            new_bound = true;
                            bind_array_type(a_param_type, a_arg_type);
                        }
                        return res;
                    }
                }
            }
        }
    }

    Definition *def = find_implicit_conversion(arg_type, param_type);
    if (def != NULL) {
        if (def->has_flag(Definition::DEF_IS_EXPLICIT_WARN)) {
            need_explicit_conv = true;
        }
        return true;
    }

    return false;
}

// Returns a short signature from a (function) definition.
string NT_analysis::get_short_signature(Definition const *def) const
{
    char const *s = "";
    switch (def->get_kind()) {
    case Definition::DK_CONSTRUCTOR:
        s = "constructor ";
        break;
    case Definition::DK_ANNOTATION:
        s = "annotation ";
        break;
    default:
        break;
    }
    string func_name(s, m_module.get_allocator());
    func_name += def->get_sym()->get_name();
    return func_name;
}

// Check if a parameter name exists in the parameters of a candidate.
bool NT_analysis::is_known_parameter(
    Definition const     *candidate,
    IType_function const *type,
    ISymbol const        *name) const
{
    size_t num_params = type->get_parameter_count();

    for (size_t i = 0; i < num_params; ++i) {
        IType const   *p_type;
        ISymbol const *p_sym;

        type->get_parameter(i, p_type, p_sym);

        if (p_sym == name) {
            return true;
        }
    }
    return false;
}

enum Arg_mode {
    PASSED_ARG,
    DEFAULT_ARG,
    IGNORED_ARG
};

// Resolve an overload.
NT_analysis::Definition_list NT_analysis::resolve_overload(
    bool                   &error_reported,
    IExpression_call const *call_expr,
    Definition_list        &def_list,
    bool                   is_overloaded,
    IType const            *arg_types[],
    size_t                 num_pos_args,
    size_t                 num_named_args,
    Name_index_map const   *name_arg_indexes,
    int                    arg_index)
{
    bool is_preset_match = call_expr == m_preset_overload;

    error_reported = false;

    Memory_arena arena(m_module.get_allocator());

    Signature_list best_matches(&arena);

    Bitset used_names(m_module.get_allocator(), num_named_args);
    int best_value = 0;

    VLA<IType const *> signature(m_module.get_allocator(), num_pos_args + num_named_args);
    VLA<bool>          bounds(m_module.get_allocator(), num_pos_args + num_named_args);

    // because it is not possible to pass an argument by position AND by name
    // we must have at least this amount of arguments
    size_t min_params = num_pos_args > num_named_args ? num_pos_args : num_named_args;

    for (Definition_list::iterator it(def_list.begin()), end(def_list.end());
         it != end;
         ++it)
    {
        Definition const     *candidate = *it;
        IType const          *type = candidate->get_type();

        // TODO: there should be no errors at this place
        if (is<IType_error>(type)) {
            continue;
        }

        IType_function const *func_type = cast<IType_function>(type);
        size_t num_params = func_type->get_parameter_count();

        if (is_overloaded && num_params < min_params) {
            // we have more parameters that this overload, cannot match
            continue;
        }

        int this_value = 0;
        bool type_must_match = candidate->has_flag(Definition::DEF_OP_LVALUE);
        for (size_t k = 0; k < num_pos_args && k < num_params; ++k, type_must_match = false) {
            IType const   *param_type;
            ISymbol const *name;
            int           arg_idx = arg_index >= 0 ? arg_index : int(k);
            size_t        num     = arg_index >= 0 ? 0 : k;

            IType const *arg_type = arg_types[k];
            func_type->get_parameter(k, param_type, name);

            param_type = param_type->skip_type_alias();

            bool new_bound = false;
            bool need_explicit_conv = false;
            if (equal_types(param_type, arg_type)) {
                // types match
                this_value += 3;
            } else if (!type_must_match &&
                       can_assign_param(param_type, arg_type, new_bound, need_explicit_conv)) {
                // can be implicitly converted (or explicit with warning)
                this_value += need_explicit_conv ? 1 : 2;
            } else {
                if (! is_overloaded) {
                    string func_name = get_short_signature(candidate);

                    if (is<IType_function>(arg_type->skip_type_alias())) {
                        error(
                            CANNOT_CONVERT_POS_ARG_TYPE_FUNCTION,
                            *get_argument_position(call_expr, arg_idx),
                            Error_params(*this)
                            .add(func_name.c_str())
                            .add_numword(num + 1));  // count from 1
                        error_reported = true;
                    } else {
                        IType const *bound_type = get_bound_type(param_type);
                        error(
                            CANNOT_CONVERT_POS_ARG_TYPE,
                            *get_argument_position(call_expr, arg_idx),
                            Error_params(*this)
                                .add(func_name.c_str())
                                .add_numword(num + 1)  // count from 1
                                .add(arg_type)
                                .add(bound_type));
                        error_reported = true;
                        if (bound_type != param_type) {
                            add_note(
                                BOUND_IN_CONTEXT,
                                *get_argument_position(call_expr, arg_idx),
                                Error_params(*this).add(param_type).add(bound_type));
                        }
                    }
                }
                this_value = -1;
                break;
            }
            signature[k] = get_bound_type(param_type);
            bounds[k]    = new_bound;
        }
        if (!is_overloaded && num_params < num_pos_args) {
            Definition const *candidate = *(def_list.begin());

            error(
                TOO_MANY_ARGUMENTS,
                *get_argument_position(call_expr, num_params),
                Error_params(*this).add_signature(candidate));
            this_value = -1;
            error_reported = true;
        }

        if (this_value >= 0) {
            // check named parameters
            for (size_t k = num_pos_args; k < num_params; ++k) {
                IType const   *param_type;
                ISymbol const *name;

                func_type->get_parameter(k, param_type, name);

                IType const    *arg_type = NULL;
                size_t         idx       = k;
                Arg_mode       arg_mode  = PASSED_ARG;
                Position const *def_pos  = NULL;

                // find this parameter
                bool found = false;
                if (name_arg_indexes != NULL) {
                    Name_index_map::const_iterator it = name_arg_indexes->find(name);
                    if (it != name_arg_indexes->end()) {
                        idx = it->second;
                        found = true;
                    }
                }
                if (!found) {
                    // not found, must have a default value
                    IExpression const *def_expr = get_default_param_initializer(candidate, k);
                    if (def_expr == NULL && !is_preset_match) {
                        this_value = -1;
                        if (!is_overloaded) {
                            string func_name = get_short_signature(candidate);

                            error(
                                MISSING_PARAMETER,
                                call_expr->access_position(),
                                Error_params(*this)
                                    .add(func_name.c_str())
                                    .add(name));
                            error_reported = true;
                        }
                        break;
                    } else if (is_preset_match) {
                        arg_mode = IGNORED_ARG;
                        arg_type = param_type->skip_type_alias();
                    } else {
                        // remember this default parameter
                        arg_mode = DEFAULT_ARG;

                        // beware: the default parameter might be owned by another module,
                        // import its type
                        arg_type = def_expr->get_type();
                        arg_type = m_module.import_type(arg_type->skip_type_alias());
                        def_pos  = &def_expr->access_position();
                    }
                } else {
                    // found the position of a named parameter
                    arg_type = arg_types[idx];

                    if (used_names.test_bit(idx - num_pos_args)) {
                        if (! is_overloaded) {
                            string func_name = get_short_signature(candidate);

                            error(
                                PARAMETER_PASSED_TWICE,
                                *get_argument_position(call_expr, idx),
                                Error_params(*this)
                                    .add(func_name.c_str())
                                    .add(name));
                            error_reported = true;
                        }
                        this_value = -1;
                        break;
                    }

                    used_names.set_bit(idx - num_pos_args);
                }

                // check types and calculate type binding
                param_type = param_type->skip_type_alias();

                bool new_bound          = false;
                bool need_explicit_conv = false;
                if (is<IType_error>(arg_type)) {
                    // some error occurred on the (default) argument expression
                    this_value = -1;
                    break;
                } else if (equal_types(param_type, arg_type)) {
                    // types match
                    if (arg_mode == PASSED_ARG)
                        this_value += 3;
                } else if (can_assign_param(param_type, arg_type, new_bound, need_explicit_conv)) {
                    // can be implicitly converted (or explicit with warning)
                    if (arg_mode == PASSED_ARG)
                        this_value += need_explicit_conv ? 1 : 2;
                } else {
                    MDL_ASSERT(arg_mode != IGNORED_ARG && "type error on ignored argument");
                    if (! is_overloaded) {
                        string func_name = get_short_signature(candidate);

                        if (is<IType_function>(arg_type->skip_type_alias())) {
                            if (arg_mode == DEFAULT_ARG) {
                                error(
                                    CANNOT_CONVERT_DEFAULT_ARG_TYPE_FUNCTION,
                                    call_expr->access_position(),
                                    Error_params(*this)
                                        .add(func_name.c_str())
                                        .add(name));
                                add_note(
                                    DEFAULT_ARG_DECLARED_AT,
                                    *def_pos,
                                    Error_params(*this).add(name));
                            } else {
                                error(
                                    CANNOT_CONVERT_ARG_TYPE_FUNCTION,
                                    *get_argument_position(call_expr, idx),
                                    Error_params(*this)
                                        .add(func_name.c_str())
                                        .add(name));
                            }
                            error_reported = true;
                        } else {
                            IType const *bound_type = get_bound_type(param_type);
                            if (arg_mode == DEFAULT_ARG) {
                                error(
                                    CANNOT_CONVERT_DEFAULT_ARG_TYPE,
                                    call_expr->access_position(),
                                    Error_params(*this)
                                        .add(func_name.c_str())
                                        .add(name)
                                        .add(arg_type)
                                        .add(bound_type));
                                // this should only happen with bounding
                                MDL_ASSERT(bound_type != param_type);
                                add_note(
                                    BOUND_IN_CONTEXT,
                                    call_expr->access_position(),
                                    Error_params(*this).add(param_type).add(bound_type));
                                add_note(
                                    DEFAULT_ARG_DECLARED_AT,
                                    *def_pos,
                                    Error_params(*this).add(name));
                            } else {
                                error(
                                    CANNOT_CONVERT_ARG_TYPE,
                                    *get_argument_position(call_expr, idx),
                                    Error_params(*this)
                                        .add(func_name.c_str())
                                        .add(name)
                                        .add(arg_type)
                                        .add(bound_type));
                                if (bound_type != param_type) {
                                    add_note(
                                        BOUND_IN_CONTEXT,
                                        *get_argument_position(call_expr, idx),
                                        Error_params(*this).add(param_type).add(bound_type));
                                }
                            }
                            error_reported = true;
                        }
                    }
                    this_value = -1;
                    break;
                }
                if (arg_mode == PASSED_ARG) {
                    signature[idx] = get_bound_type(param_type);
                    bounds[idx]    = new_bound;
                }
            }
        }
        if (this_value >= 0) {
            // check if all args where used
            for (size_t k = 0; k < num_named_args; ++k) {
                if (! used_names.test_bit(k)) {
                    // unused arg
                    this_value = -1;
                    IArgument_named const *arg =
                        cast<IArgument_named>(call_expr->get_argument(num_pos_args + k));

                    if (! is_overloaded) {
                        string func_name = get_short_signature(candidate);
                        ISymbol const *p_name = arg->get_parameter_name()->get_symbol();

                        // this parameter is either already given by pos/name OR
                        // completely unknown
                        if (is_known_parameter(candidate, func_type, p_name)) {
                            error(
                                PARAMETER_PASSED_TWICE,
                                arg->access_position(),
                                Error_params(*this)
                                    .add(func_name.c_str())
                                    .add(p_name));
                            error_reported = true;
                        } else {
                            error(
                                UNKNOWN_NAMED_PARAMETER,
                                arg->access_position(),
                                Error_params(*this)
                                    .add(func_name.c_str())
                                    .add(p_name));
                            error_reported = true;
                        }
                    }
                }
            }
        }
        used_names.clear_bits();
        clear_type_bindings();

        if (this_value > best_value) {
            // higher values mean "more specific"
            best_matches.clear();
            Signature_entry entry(
                (IType const *const *)Arena_memdup(
                    arena, signature.data(), signature.size() * sizeof(signature[0])),
                (bool const *const)Arena_memdup(
                    arena, bounds.data(), bounds.size() * sizeof(bounds[0])),
                signature.size(),
                candidate);
            best_matches.push_back(entry);
            best_value = this_value;
        } else if (this_value == best_value) {
            Signature_entry entry(
                (IType const *const *)Arena_memdup(
                    arena, signature.data(), signature.size() * sizeof(signature[0])),
                (bool const *const)Arena_memdup(
                    arena, bounds.data(), bounds.size() * sizeof(bounds[0])),
                signature.size(),
                candidate);
            if (kill_less_specific(best_matches, entry)) {
                best_matches.push_back(entry);
            }
        }
    }

    Definition_list res(m_module.get_allocator());

    for (Signature_list::iterator it(best_matches.begin()), end(best_matches.end());
         it != end;
         ++it)
    {
        Signature_entry const &entry = *it;
        res.push_back(entry.def);
    }

    return res;
}

// Add a "candidates are" notes to the current error message.
void NT_analysis::add_candidates(
    Definition_list const &overloads)
{
    Position_impl zero_pos(0, 0, 0, 0);

    bool first = true;

    // Note: we "known" that the candidate set is in the order of the definition table, with is
    // reverse. Hence iterating in reverse order will add definition in "definition order", i.e.
    // first defined first. This is necessary or the CANDIDATES_ARE message will be after
    // CANDIDATES_ARE_NEXT ...
    for (Definition_list::const_reverse_iterator it(overloads.rbegin()), end(overloads.rend());
         it != end;
         ++it)
    {
        Definition const  *def = *it;

        Position const *def_pos = def->get_position();
        if (def_pos == NULL) {
            def_pos = &zero_pos;
        }

        add_note(
            first ? CANDIDATES_ARE : CANDIDATES_ARE_NEXT,
            *def_pos,
            Error_params(*this).add_signature(def));
        first = false;
    }
}

// Resolve overloads for a call
Definition const *NT_analysis::find_overload(
    Definition const *def,
    IExpression_call *call,
    bool             bound_to_scope)
{
    clear_type_bindings();

    if (is_error(def)) {
        // name resolving failed, nothing to do here
        return def;
    }

    // check if it's a function
    if (!is<IType_function>(def->get_type()) || def->get_kind() == IDefinition::DK_ANNOTATION) {
        error(
            CALLED_OBJECT_NOT_A_FUNCTION,
            call->get_reference()->access_position(),
            Error_params(*this).add(def->get_sym()));
        if (Position const *def_pos = def->get_position()) {
            add_note(
                DEFINED_AT,
                *def_pos,
                Error_params(*this)
                     .add("'")
                    .add_signature(def)
                    .add("' "));
        }
        return get_error_definition();
    }

    IAllocator *alloc = m_module.get_allocator();

    m_string_buf->clear();
    m_printer->print(call->get_reference());
    m_printer->print("(");

    size_t arg_count = call->get_argument_count();

    VLA<IType const *> arg_types(alloc, arg_count);
    VLA<IType const *> mod_arg_types(alloc, arg_count);
    VLA<IArgument const *> arguments(alloc, arg_count);

    Name_index_map name_arg_indexes(
        0, Name_index_map::hasher(), Name_index_map::key_equal(), alloc);

    size_t pos_arg_count = 0, named_arg_count = 0;

    // do NOT issue an error if one of the types is the error type, silently
    // return the error definition

    ISymbol const *last_named_arg = NULL;
    bool argument_error           = false;

    for (size_t i = 0; i < arg_count; ++i) {
        IArgument const   *arg    = call->get_argument(i);
        IExpression const *expr   = arg->get_argument_expr();
        IType const       *arg_tp = expr->get_type();

        // save arguments for later
        arguments[i] = arg;

        if (i > 0) {
            m_printer->print(", ");
        }

        if (is<IType_error>(arg_tp)) {
            // don't stop here, process all errors
            argument_error = true;
        }
        if (arg->get_kind() == IArgument::AK_POSITIONAL) {
            // positional argument
            if (last_named_arg != NULL) {
                // we switch from named to positional mode, this is not allowed
                argument_error = true;
                error(
                    POS_ARGUMENT_AFTER_NAMED,
                    arg->access_position(),
                    Error_params(*this).add(i + 1).add(last_named_arg));
            }
            ++pos_arg_count;
        } else {
            // named argument
            IArgument_named const *named_arg = cast<IArgument_named>(arg);
            ISimple_name const    *sname     = named_arg->get_parameter_name();
            ISymbol const         *arg_sym   = sname->get_symbol();

            last_named_arg = arg_sym;

            if (name_arg_indexes.find(arg_sym) != name_arg_indexes.end()) {
                error(
                    PARAMETER_PASSED_TWICE,
                    arg->access_position(),
                    Error_params(*this)
                        .add(call->get_reference())
                        .add(arg_sym));
                argument_error = true;
            } else {
                ++named_arg_count;

                name_arg_indexes[arg_sym] = i;

                m_printer->print(arg_sym);
                m_printer->print(" : ");
            }
        }
        arg_types[i]     = arg_tp;
        mod_arg_types[i] = arg_tp->skip_type_alias();
        m_printer->print(arg_tp);
    }
    m_printer->print(")");

    if (argument_error) {
        // could not resolve due to argument errors
        return get_error_definition();
    }

    string signature(m_string_buf->get_data(), alloc);

    for (;;) {
        // collect the possible set
        Definition_list function_list(alloc);

        size_t cnt = 0;
        for (Definition const *curr_def = def;
             curr_def != NULL;
             curr_def = curr_def->get_prev_def())
        {
            if (! curr_def->has_flag(Definition::DEF_IGNORE_OVERLOAD)) {
                function_list.push_back(curr_def);
                ++cnt;
            }
        }

        if (arg_types.size() == 1 && def->get_kind() == IDefinition::DK_CONSTRUCTOR) {
            // a constructor call, could also be a operator TYPE() type
            IType const *arg_tp = arg_types[0]->skip_type_alias();

            if (is<IType_enum>(arg_tp)) {
                // Currently, only enum types have operator TYPE, but the mechanism
                // works generally
                if (Scope *s = m_def_tab->get_type_scope(arg_tp)) {
                    for (Definition const *curr_def = s->find_definition_in_scope(def->get_sym());
                        curr_def != NULL;
                        curr_def = curr_def->get_prev_def())
                    {
                        if (!curr_def->has_flag(Definition::DEF_IGNORE_OVERLOAD) &&
                            curr_def->get_semantics() == IDefinition::DS_CONV_OPERATOR) {
                            function_list.push_back(curr_def);
                            ++cnt;
                        }
                    }
                }
            }
        }

        // resolve it
        bool errors_reported = false;
        Definition_list overloads = resolve_overload(
            errors_reported,
            call,
            function_list,
            cnt > 1,
            mod_arg_types.data(),
            pos_arg_count,
            named_arg_count,
            &name_arg_indexes);

        if (overloads.empty()) {
            // no matching overload found
            if (!bound_to_scope) {
                def = def->get_outer_def();
                if (def) {
                    // try outer if we are not bound
                    continue;
                }
            }
            // else finished
            if (!errors_reported) {
                // we have an overload set
                error(
                    NO_MATCHING_OVERLOAD,
                    call->access_position(),
                    Error_params(*this).add(signature.c_str()));
            }
            return get_error_definition();
        }
        Definition_list::iterator it = overloads.begin();
        def = *it;
        if (++it != overloads.end()) {
            // ambiguous overload found
            error(
                AMBIGUOUS_OVERLOAD,
                call->access_position(),
                Error_params(*this).add(signature.c_str()));
            add_candidates(overloads);
            return get_error_definition();
        } else {
            clear_type_bindings();

            // All fine, found only ONE possible overload.
            if (def->get_semantics() == IDefinition::DS_DEFAULT_STRUCT_CONSTRUCTOR) {
                // reformat to elemental struct constructor
                def = reformat_default_struct_constructor(def, call);
            } else {
                // Otherwise add argument conversions if needed and reorder the arguments.
                reformat_arguments(
                    def, call, mod_arg_types.data(), pos_arg_count,
                    name_arg_indexes, arguments.data(), named_arg_count);
            }
            return def;
        }
    }
}

// Resolve an annotation overload.
//
// This is basicyally a copy of resolve_overload() which these changes:
// - handles IAnnotation instead of IExpression_call
// - issues warnings instead of errors
// - name_arg_indexes cannot be NULL
NT_analysis::Definition_list NT_analysis::resolve_annotation_overload(
    bool                   &error_reported,
    IAnnotation const      *anno,
    Definition_list        &def_list,
    bool                   is_overloaded,
    IType const            *arg_types[],
    size_t                 num_pos_args,
    size_t                 num_named_args,
    Name_index_map const   *name_arg_indexes)
{
    MDL_ASSERT(name_arg_indexes != NULL);

    error_reported = false;

    Memory_arena arena(m_module.get_allocator());

    Signature_list best_matches(&arena);

    Bitset used_names(m_module.get_allocator(), num_named_args);
    int best_value = 0;

    VLA<IType const *> signature(m_module.get_allocator(), num_pos_args + num_named_args);
    VLA<bool>          bounds(m_module.get_allocator(), num_pos_args + num_named_args);

    // because it is not possible to pass an argument by position AND by name
    // we must have at least this amount of arguments
    size_t min_params = num_pos_args > num_named_args ? num_pos_args : num_named_args;

    for (Definition_list::iterator it(def_list.begin()), end(def_list.end());
         it != end;
         ++it)
    {
        Definition const     *candidate = *it;
        IType const          *type = candidate->get_type();

        // TODO: there should be no errors at this place
        if (is<IType_error>(type)) {
            continue;
        }

        IType_function const *func_type = cast<IType_function>(type);
        size_t num_params = func_type->get_parameter_count();

        if (is_overloaded && num_params < min_params) {
            continue;
        }

        int this_value = 0;
        bool type_must_match = candidate->has_flag(Definition::DEF_OP_LVALUE);
        for (size_t k = 0; k < num_pos_args && k < num_params; ++k, type_must_match = false) {
            IType const   *param_type;
            ISymbol const *name;

            IType const *arg_type = arg_types[k];
            func_type->get_parameter(k, param_type, name);

            param_type = param_type->skip_type_alias();

            bool new_bound          = false;
            bool need_explicit_conv = false;
            if (equal_types(param_type, arg_type)) {
                // types match
                this_value += 3;
            } else if (!type_must_match &&
                       can_assign_param(param_type, arg_type, new_bound, need_explicit_conv)) {
                // can be implicitly converted (or explicit with warning)
                this_value += need_explicit_conv ? 1 : 2;
            } else {
                if (! is_overloaded) {
                    string func_name = get_short_signature(candidate);

                    if (is<IType_function>(arg_type->skip_type_alias())) {
                        warning(
                            CANNOT_CONVERT_POS_ARG_TYPE_FUNCTION,
                            anno->get_argument(k)->access_position(),
                            Error_params(*this)
                            .add(func_name.c_str())
                            .add_numword(k + 1));  // count from 1
                        error_reported = true;
                    } else {
                        IType const *bound_type = get_bound_type(param_type);
                        warning(
                            CANNOT_CONVERT_POS_ARG_TYPE,
                            anno->get_argument(k)->access_position(),
                            Error_params(*this)
                                .add(func_name.c_str())
                                .add_numword(k + 1)  // count from 1
                                .add(arg_type)
                                .add(bound_type));
                        error_reported = true;
                        if (bound_type != param_type) {
                            add_note(
                                BOUND_IN_CONTEXT,
                                anno->get_argument(k)->access_position(),
                                Error_params(*this).add(param_type).add(bound_type));
                        }
                    }
                }
                this_value = -1;
                break;
            }
            signature[k] = get_bound_type(param_type);
            bounds[k]    = new_bound;
        }
        if (!is_overloaded && num_params < num_pos_args) {
            Definition const *candidate = *(def_list.begin());

            warning(
                TOO_MANY_ARGUMENTS,
                anno->get_argument(num_params)->access_position(),
                Error_params(*this).add_signature(candidate));
            this_value = -1;
            error_reported = true;
        }

        if (this_value >= 0) {
            // check named parameters
            for (size_t k = num_pos_args; k < num_params; ++k) {
                IType const   *param_type;
                ISymbol const *name;

                func_type->get_parameter(k, param_type, name);

                IType const    *arg_type      = NULL;
                size_t         idx            = k;
                bool           is_default_arg = false;
                Position const *def_pos = NULL;

                // find this parameter
                Name_index_map::const_iterator it = name_arg_indexes->find(name);
                if (it == name_arg_indexes->end()) {
                    // not found, must have a default value
                    IExpression const *def_expr = get_default_param_initializer(candidate, k);
                    if (def_expr == NULL) {
                        this_value = -1;
                        if (! is_overloaded) {
                            string func_name = get_short_signature(candidate);

                            warning(
                                MISSING_PARAMETER,
                                anno->access_position(),
                                Error_params(*this)
                                    .add(func_name.c_str())
                                    .add(name));
                            error_reported = true;
                        }
                        break;
                    } else {
                        // remember this default parameter
                        is_default_arg = true;

                        // beware: the default parameter might be owned by another module,
                        // import its type
                        arg_type = def_expr->get_type();
                        arg_type = m_module.import_type(arg_type->skip_type_alias());
                        def_pos  = &def_expr->access_position();
                    }
                } else {
                    // found the position of a named parameter
                    idx = it->second;
                    arg_type = arg_types[idx];

                    if (used_names.test_bit(idx - num_pos_args)) {
                        if (! is_overloaded) {
                            string func_name = get_short_signature(candidate);

                            warning(
                                PARAMETER_PASSED_TWICE,
                                anno->get_argument(idx)->access_position(),
                                Error_params(*this)
                                    .add(func_name.c_str())
                                    .add(name));
                            error_reported = true;
                        }
                        this_value = -1;
                        break;
                    }

                    used_names.set_bit(idx - num_pos_args);
                }

                // check types and calculate type binding
                param_type = param_type->skip_type_alias();

                bool new_bound          = false;
                bool need_explicit_conv = false;
                if (is<IType_error>(arg_type)) {
                    // some error occurred on the (default) argument expression
                    this_value = -1;
                    break;
                } else if (equal_types(param_type, arg_type)) {
                    // types match
                    if (!is_default_arg)
                        this_value += 3;
                } else if (can_assign_param(param_type, arg_type, new_bound, need_explicit_conv)) {
                    // can be implicitly converted (or explicit with warning)
                    if (!is_default_arg)
                        this_value += need_explicit_conv ? 1 : 2;
                } else {
                    if (! is_overloaded) {
                        string func_name = get_short_signature(candidate);

                        if (is<IType_function>(arg_type->skip_type_alias())) {
                            if (is_default_arg) {
                                warning(
                                    CANNOT_CONVERT_DEFAULT_ARG_TYPE_FUNCTION,
                                    anno->access_position(),
                                    Error_params(*this)
                                        .add(func_name.c_str())
                                        .add(name));
                                add_note(
                                    DEFAULT_ARG_DECLARED_AT,
                                    *def_pos,
                                    Error_params(*this).add(name));
                            } else {
                                warning(
                                    CANNOT_CONVERT_ARG_TYPE_FUNCTION,
                                    anno->get_argument(idx)->access_position(),
                                    Error_params(*this)
                                        .add(func_name.c_str())
                                        .add(name));
                            }
                            error_reported = true;
                        } else {
                            IType const *bound_type = get_bound_type(param_type);
                            if (is_default_arg) {
                                warning(
                                    CANNOT_CONVERT_DEFAULT_ARG_TYPE,
                                    anno->access_position(),
                                    Error_params(*this)
                                        .add(func_name.c_str())
                                        .add(name)
                                        .add(arg_type)
                                        .add(bound_type));
                                // this should only happen with bounding
                                MDL_ASSERT(bound_type != param_type);
                                add_note(
                                    BOUND_IN_CONTEXT,
                                    anno->access_position(),
                                    Error_params(*this).add(param_type).add(bound_type));
                                add_note(
                                    DEFAULT_ARG_DECLARED_AT,
                                    *def_pos,
                                    Error_params(*this).add(name));
                            } else {
                                warning(
                                    CANNOT_CONVERT_ARG_TYPE,
                                    anno->get_argument(idx)->access_position(),
                                    Error_params(*this)
                                        .add(func_name.c_str())
                                        .add(name)
                                        .add(arg_type)
                                        .add(bound_type));
                                if (bound_type != param_type) {
                                    add_note(
                                        BOUND_IN_CONTEXT,
                                        anno->get_argument(idx)->access_position(),
                                        Error_params(*this).add(param_type).add(bound_type));
                                }
                            }
                            error_reported = true;
                        }
                    }
                    this_value = -1;
                    break;
                }
                if (!is_default_arg) {
                    signature[idx] = get_bound_type(param_type);
                    bounds[idx]    = new_bound;
                }
            }
        }
        if (this_value >= 0) {
            // check if all args where used
            for (size_t k = 0; k < num_named_args; ++k) {
                if (! used_names.test_bit(k)) {
                    // unused arg
                    this_value = -1;
                    IArgument_named const *arg =
                        cast<IArgument_named>(anno->get_argument(num_pos_args + k));

                    if (! is_overloaded) {
                        string func_name = get_short_signature(candidate);
                        ISymbol const *p_name = arg->get_parameter_name()->get_symbol();

                        // this parameter is either already given by pos/name OR
                        // completely unknown
                        if (is_known_parameter(candidate, func_type, p_name)) {
                            error(
                                PARAMETER_PASSED_TWICE,
                                arg->access_position(),
                                Error_params(*this)
                                    .add(func_name.c_str())
                                    .add(p_name));
                            error_reported = true;
                        } else {
                            error(
                                UNKNOWN_NAMED_PARAMETER,
                                arg->access_position(),
                                Error_params(*this)
                                    .add(func_name.c_str())
                                    .add(p_name));
                            error_reported = true;
                        }
                    }
                }
            }
        }
        used_names.clear_bits();
        clear_type_bindings();

        if (this_value > best_value) {
            // higher values mean "more specific"
            best_matches.clear();
            Signature_entry entry(
                (IType const *const *)Arena_memdup(
                    arena, signature.data(), signature.size() * sizeof(signature[0])),
                (bool const *const)Arena_memdup(
                    arena, bounds.data(), bounds.size() * sizeof(bounds[0])),
                signature.size(),
                candidate);
            best_matches.push_back(entry);
            best_value = this_value;
        } else if (this_value == best_value) {
            Signature_entry entry(
                (IType const *const *)Arena_memdup(
                    arena, signature.data(), signature.size() * sizeof(signature[0])),
                (bool const *const)Arena_memdup(
                    arena, bounds.data(), bounds.size() * sizeof(bounds[0])),
                signature.size(),
                candidate);
            if (kill_less_specific(best_matches, entry)) {
                best_matches.push_back(entry);
            }
        }
    }

    Definition_list res(m_module.get_allocator());

    for (Signature_list::iterator it(best_matches.begin()), end(best_matches.end());
         it != end;
         ++it)
    {
        Signature_entry const &entry = *it;
        res.push_back(entry.def);
    }
    return res;
}

// Resolve overloads for an annotation
Definition const *NT_analysis::find_annotation_overload(
    Definition const *def,
    IAnnotation      *anno,
    bool             bound_to_scope)
{
    clear_type_bindings();

    if (is_error(def)) {
        // name resolving failed, nothing to do here
        return def;
    }

    IAllocator *alloc = m_module.get_allocator();

    m_string_buf->clear();
    m_printer->print(anno->get_name());
    m_printer->print("(");

    size_t arg_count = anno->get_argument_count();

    VLA<IType const *> arg_types(alloc, arg_count);
    VLA<IType const *> mod_arg_types(alloc, arg_count);
    VLA<IArgument const *> arguments(alloc, arg_count);

    Name_index_map name_arg_indexes(
        0, Name_index_map::hasher(), Name_index_map::key_equal(), alloc);

    size_t pos_arg_count = 0, named_arg_count = 0;

    // do NOT issue an error if one of the types is the error type, silently
    // return the error definition

    ISymbol const *last_named_arg = NULL;
    bool argument_error           = false;

    for (size_t i = 0; i < arg_count; ++i) {
        IArgument const   *arg    = anno->get_argument(i);
        IExpression const *expr   = arg->get_argument_expr();
        IType const       *arg_tp = expr->get_type();

        // save arguments for later
        arguments[i] = arg;

        if (i > 0) {
            m_printer->print(", ");
        }

        if (is<IType_error>(arg_tp)) {
            // don't stop here, process all errors
            argument_error = true;
        }
        if (arg->get_kind() == IArgument::AK_POSITIONAL) {
            // positional argument
            if (last_named_arg != NULL) {
                // we switch from named to positional mode, this is not allowed
                argument_error = true;
                warning(
                    POS_ARGUMENT_AFTER_NAMED,
                    arg->access_position(),
                    Error_params(*this).add(i + 1).add(last_named_arg));
            }
            ++pos_arg_count;
        } else {
            // named argument
            IArgument_named const *named_arg = cast<IArgument_named>(arg);
            ISimple_name const    *sname     = named_arg->get_parameter_name();
            ISymbol const         *arg_sym   = sname->get_symbol();

            last_named_arg = arg_sym;
            ++named_arg_count;

            if (name_arg_indexes.find(arg_sym) != name_arg_indexes.end()) {
                warning(
                    PARAMETER_PASSED_TWICE,
                    anno->get_argument(i)->access_position(),
                    Error_params(*this)
                        .add(anno->get_name())
                        .add(arg_sym));
                argument_error = true;
            } else {
                name_arg_indexes[arg_sym] = i;

                m_printer->print(arg_sym);
                m_printer->print(" : ");
            }
        }
        arg_types[i]     = arg_tp;
        mod_arg_types[i] = arg_tp->skip_type_alias();
        m_printer->print(arg_tp);
    }
    m_printer->print(")");

    if (argument_error) {
        // could not resolve due to argument errors
        return get_error_definition();
    }

    string signature(m_string_buf->get_data(), alloc);

    for (;;) {
        // collect the possible set
        Definition_list function_list(alloc);

        size_t cnt = 0;
        for (Definition const *curr_def = def;
             curr_def != NULL;
             curr_def = curr_def->get_prev_def())
        {
            if (! curr_def->has_flag(Definition::DEF_IGNORE_OVERLOAD)) {
                function_list.push_back(curr_def);
                ++cnt;
            }
        }

        // resolve it
        bool warnings_reported = false;
        Definition_list overloads = resolve_annotation_overload(
            warnings_reported,
            anno,
            function_list,
            cnt > 1,
            mod_arg_types.data(),
            pos_arg_count,
            named_arg_count,
            &name_arg_indexes);

        if (overloads.empty()) {
            // no matching overload found
            def = def->get_outer_def();
            if (!bound_to_scope && def) {
                // try outer if we are not bound
                continue;
            }
            // else finished
            if (!warnings_reported) {
                // we have an overload set
                warning(
                    NO_MATCHING_OVERLOAD,
                    anno->access_position(),
                    Error_params(*this).add(signature.c_str()));
            }
            return get_error_definition();
        }
        Definition_list::iterator it = overloads.begin();
        def = *it;
        if (++it != overloads.end()) {
            // ambiguous overload found
            warning(
                AMBIGUOUS_OVERLOAD,
                anno->access_position(),
                Error_params(*this).add(signature.c_str()));
            add_candidates(overloads);
            return get_error_definition();
        } else {
            clear_type_bindings();

            // All fine, found only ONE possible overload.
            // Now add argument conversions if needed and reorder the arguments.
            reformat_annotation_arguments(
                def, anno, mod_arg_types.data(), pos_arg_count,
                name_arg_indexes, arguments.data(), named_arg_count);
            return def;
        }
    }
}

// Report an error type error.
void NT_analysis::report_array_type_error(
    IType const    *type,
    Position const &pos)
{
    if (as<IType_array>(type) != NULL) {
        error(
            FORBIDDEN_NESTED_ARRAY,
            pos,
            Error_params(*this));
    } else {
        error(
            FORBIDDEN_ARRAY_TYPE,
            pos,
            Error_params(*this).add(type));
    }
}

// Check if the given call is an array copy constructor.
bool NT_analysis::handle_array_copy_constructor(
    IExpression_call const *call)
{
    IExpression_reference const *callee = as<IExpression_reference>(call->get_reference());
    if (callee == NULL) {
        return false;
    }

    int n_args = call->get_argument_count();

    if (n_args != 1) {
        return false;
    }

    IArgument const *arg = call->get_argument(0);

    // all array constructor arguments must be positional
    if (arg->get_kind() == IArgument::AK_NAMED) {
        return false;
    }

    IExpression const *expr = arg->get_argument_expr();

    // we know that we are searching a constructor here ...
    IType const *arg_type = expr->get_type();

    IType_array const *a_type = as<IType_array>(arg_type);
    if (a_type == NULL) {
        return false;
    }

    IType_name const *type_name = callee->get_name();
    IExpression const *arr_size = type_name->get_array_size();

    IType const *res_type = a_type;

    bool is_invalid = false;
    if (arr_size != NULL) {
        // must match
        res_type = a_type->get_element_type();

        IDefinition const *array_size_def = NULL;
        if (!is_const_array_size(arr_size, array_size_def, is_invalid)) {
            // must be a const expression
            if (!is_invalid) {
                error(
                    ARRAY_SIZE_NON_CONST,
                    arr_size->access_position(),
                    Error_params(*this));
                is_invalid = true;
            }
        } else if (array_size_def != NULL) {
            // an abstract array
            IType_array_size const *size = get_array_size(array_size_def, NULL);
            res_type = m_tc.create_array(res_type, size);
        } else {
            // fold it
            m_exc_handler.clear_error_state();
            IValue const *val = arr_size->fold(
                &m_module, m_module.get_value_factory(), &m_exc_handler);

            // convert to int
            val = val->convert(m_module.get_value_factory(), m_tc.int_type);
            if (IValue_int const *iv = as<IValue_int>(val)) {
                int size = iv->get_value();

                if (!is<IExpression_literal>(arr_size)) {
                    Position const *pos = &arr_size->access_position();
                    arr_size = m_module.create_literal(val, pos);
                    const_cast<IType_name *>(type_name)->set_array_size(arr_size);
                }

                // FIXME: do we allow arrays of size 0?
                if (size < 0) {
                    error(
                        ARRAY_SIZE_NEGATIVE,
                        arr_size->access_position(),
                        Error_params(*this));
                    is_invalid = true;
                } else {
                    // all fine: create the array type
                    res_type = m_tc.create_array(res_type, size);
                }
            } else if (m_exc_handler.has_error()) {
                is_invalid = true;
            } else {
                MDL_ASSERT(!"const folding failed");
                is_invalid = true;
            }
        }
        if (is_invalid)
            res_type = m_tc.error_type;

        if (!is<IType_error>(res_type)) {
            if (!equal_types(res_type, a_type)) {
                error(
                    NO_ARRAY_CONVERSION,
                    call->access_position(),
                    Error_params(*this).add(a_type).add(res_type));
                res_type = m_tc.error_type;
            }
        }
    }

    const_cast<IExpression_call *>(call)->set_type(
        check_performance_restriction(res_type, call->access_position()));

    return true;
}

// Find an array constructor.
bool NT_analysis::handle_array_constructor(
    Definition const *def,
    IExpression_call *call)
{
    if (is_error(def)) {
        // name resolving failed, nothing to do here
        return false;
    }

    // check if it's a function
    IType_function const *func_type = as<IType_function>(def->get_type());

    if (def->get_kind() != Definition::DK_CONSTRUCTOR || func_type == NULL) {
        error(
            NOT_A_TYPE_NAME,
            call->get_reference()->access_position(),
            Error_params(*this).add(def->get_sym()));
        if (const Position *def_pos = def->get_position()) {
            add_note(
                DEFINED_AT, *def_pos, Error_params(*this).add_signature(def));
        }
        return false;
    }

    IType const *ret_type = func_type->get_return_type();
    if (!is_allowed_array_type(ret_type)) {
        // try to create a forbidden array type, ignore further errors
        report_array_type_error(ret_type, call->access_position());
        return false;
    }

    IAllocator *alloc = m_module.get_allocator();

    // collect the possible set
    Definition_list function_list(alloc);

    int n_args = call->get_argument_count();

    size_t cnt = 0;
    if (n_args > 0) {
        for (Definition const *curr_def = def;
            curr_def != NULL;
            curr_def = curr_def->get_prev_def())
        {
            if (! curr_def->has_flag(Definition::DEF_IGNORE_OVERLOAD)) {
                function_list.push_back(curr_def);
                ++cnt;
            }
        }
    }

    bool res = true;
    for (int i = 0; i < n_args; ++i) {
        IArgument const *arg = call->get_argument(i);

        // all array constructor arguments must be positional
        if (arg->get_kind() == IArgument::AK_NAMED) {
            IArgument_named const *narg = cast<IArgument_named>(arg);
            error(
                ARRAY_CONSTRUCTOR_DOESNT_HAVE_NAMED_ARG,
                narg->access_position(),
                Error_params(*this).add(narg->get_parameter_name()->get_symbol()));
            res = false;
        }

        IExpression const *expr = arg->get_argument_expr();

        // we know that we are searching a constructor here ...
        IType const *arg_type = expr->get_type();

        if (is<IType_error>(arg_type)) {
            res = false;
            continue;
        }

        m_string_buf->clear();
        m_printer->print(ret_type);
        m_printer->print("(");

        IType const *mod_arg_types[1];

        // we do not allow array of pointers, so produce a better error message here
        if (IType_function const *f_type = as<IType_function>(arg_type)) {
            IType const *ret_type = f_type->get_return_type();
            if (ret_type != NULL && is_material_type(ret_type)) {
                error(
                    MATERIAL_PTR_USED,
                    expr->access_position(),
                    Error_params(*this));
            } else {
                error(
                    FUNCTION_PTR_USED,
                    expr->access_position(),
                    Error_params(*this));
            }
            res = false;
            continue;
        }

        mod_arg_types[0] = arg_type->skip_type_alias();
        m_printer->print(arg_type);

        m_printer->print(")");

        string signature(m_string_buf->get_data(), alloc);

        for (;;) {
            // resolve it
            bool errors_reported = false;
            Definition_list overloads = resolve_overload(
                errors_reported,
                call,
                function_list,
                /*is_overloaded=*/cnt > 1,
                mod_arg_types,
                /*num_pos_args=*/1,
                /*num_named_args=*/0,
                /*name_arg_indexes=*/NULL,
                /*arg_idx=*/i);

            if (overloads.empty()) {
                // no matching overload found
                if (!errors_reported) {
                    // we have an overload set
                    error(
                        NO_MATCHING_OVERLOAD,
                        expr->access_position(),
                        Error_params(*this).add(signature.c_str()));
                    res = false;
                }
                break;
            }
            Definition_list::iterator it = overloads.begin();
            def = *it;
            if (++it != overloads.end()) {
                // ambiguous overload found
                error(
                    AMBIGUOUS_OVERLOAD,
                    expr->access_position(),
                    Error_params(*this).add(signature.c_str()));
                add_candidates(overloads);
                res = false;
            } else {
                if (def->get_semantics() != Definition::DS_COPY_CONSTRUCTOR) {
                    IArgument const *arg    = call->get_argument(i);
                    IExpression const *expr = arg->get_argument_expr();

                    IExpression_call *conv =
                        create_type_conversion_call(expr, def, /*warn_if_explicit=*/true);

                    const_cast<IArgument *>(arg)->set_argument_expr(conv);

                    IType_function const *f_type = cast<IType_function>(def->get_type());
                    IType const          *c_arg_type;
                    ISymbol const        *dummy;
                    f_type->get_parameter(0, c_arg_type, dummy);


                    // All fine, found only ONE possible overload.
                    // Now check if the current argument can be converted if needed.
                    if (!convert_array_cons_argument(conv, 0, c_arg_type))
                        res = false;
                }

                update_call_graph(def);
            }
            break;
        }
    }
    return res;
}

// Find constructor for an initializer.
IExpression const *NT_analysis::find_init_constructor(
    IType const        *type,
    IType_name const   *tname,
    IExpression const  *init_expr,
    Position const     &pos)
{
    if (is<IType_error>(type)) {
        // already error state
        return init_expr;
    }
    type = type->skip_type_alias();

    if (init_expr != NULL) {
        if (IExpression_call const *call = as<IExpression_call>(init_expr)) {
            if (IExpression_reference const *ref =
                as<IExpression_reference>(call->get_reference()))
            {
                MDL_ASSERT(!ref->is_array_constructor() && "cannot convert from array type");

                IDefinition const *call_def = ref->get_definition();
                if (call_def->get_kind() == Definition::DK_CONSTRUCTOR &&
                    get_result_type(call_def)->skip_type_alias() == type)
                {
                    // the init expression is already a call to var_type's constructor,
                    // nothing to do
                    return init_expr;
                }
            }
        }

        IType const *from_type = init_expr->get_type()->skip_type_alias();
        if (IType_function const *f_type = as<IType_function>(from_type)) {
            IType const *ret_type = f_type->get_return_type();

            if (ret_type != NULL && is_material_type(ret_type)) {
                error(
                    MATERIAL_PTR_USED,
                    init_expr->access_position(),
                    Error_params(*this));
            } else {
                error(
                    FUNCTION_PTR_USED,
                    init_expr->access_position(),
                    Error_params(*this));
            }
            const_cast<IExpression *>(init_expr)->set_type(m_tc.error_type);
            return init_expr;
        }
    }

    // construct a call
    IExpression_factory &fact = *m_module.get_expression_factory();

    IExpression_reference *callee    = fact.create_reference(tname);
    IExpression_call      *call_expr = fact.create_call(callee, POS(pos));

    // visit the reference, that sets the definitions
    (void)visit(callee);

    if (init_expr != NULL) {
        Position const &pos = init_expr->access_position();
        IArgument_positional const *arg = fact.create_positional_argument(init_expr, POS(pos));
        call_expr->add_argument(arg);
    }

    Definition const *initial_def = impl_cast<Definition>(callee->get_definition());
    MDL_ASSERT(is_error(initial_def) || initial_def->get_kind() == Definition::DK_CONSTRUCTOR);

    // we are always looking up an constructor, so we are bound to scope
    Definition const *def = find_overload(initial_def, call_expr, /*bound_to_scope=*/true);

    const_cast<IExpression_reference *>(callee)->set_definition(def);

    IType const *res_type = get_result_type(def);

    // the result type might be bound here
    res_type = get_bound_type(res_type->skip_type_alias());

    call_expr->set_type(check_performance_restriction(res_type, pos));

    if (def->get_kind() == IDefinition::DK_CONSTRUCTOR &&
        is<IType_resource>(res_type) &&
        def->get_semantics() != IDefinition::DS_COPY_CONSTRUCTOR)
    {
        return handle_resource_constructor(call_expr);
    }

    return call_expr;
}

// Check if an expression has side effects.
IDefinition const *NT_analysis::has_side_effect(
    Origin_map        &origins,
    IExpression const *expr,
    int               index,
    int               &dep_index)
{
    class Side_effect_checker : public Module_visitor {
        typedef Module_visitor Base;
    public:
        explicit Side_effect_checker(Origin_map &origins, int index)
        : Base()
        , m_origins(origins)
        , m_index(index)
        , m_def_index(-1)
        , m_dep_def(NULL)
        {}

        IExpression *post_visit(IExpression_binary *expr) MDL_FINAL
        {
            switch (expr->get_operator()) {
            case IExpression_binary::OK_ASSIGN:
            case IExpression_binary::OK_MULTIPLY_ASSIGN:
            case IExpression_binary::OK_DIVIDE_ASSIGN:
            case IExpression_binary::OK_MODULO_ASSIGN:
            case IExpression_binary::OK_PLUS_ASSIGN:
            case IExpression_binary::OK_MINUS_ASSIGN:
            case IExpression_binary::OK_SHIFT_LEFT_ASSIGN:
            case IExpression_binary::OK_SHIFT_RIGHT_ASSIGN:
            case IExpression_binary::OK_UNSIGNED_SHIFT_RIGHT_ASSIGN:
            case IExpression_binary::OK_BITWISE_AND_ASSIGN:
            case IExpression_binary::OK_BITWISE_XOR_ASSIGN:
            case IExpression_binary::OK_BITWISE_OR_ASSIGN:
                {
                    IExpression const *l = expr->get_left_argument();

                    // now handle the write
                    if (IDefinition const *def = get_lvalue_base(l)) {
                        m_origins[def] = m_index;
                    }
                }
                break;
            default:
                break;
            }
            return expr;
        }

        IExpression *post_visit(IExpression_unary *expr) MDL_FINAL
        {
            switch (expr->get_operator()) {
            case IExpression_unary::OK_BITWISE_COMPLEMENT:
            case IExpression_unary::OK_LOGICAL_NOT:
            case IExpression_unary::OK_POSITIVE:
            case IExpression_unary::OK_NEGATIVE:
            case IExpression_unary::OK_CAST:
                break;
            case IExpression_unary::OK_PRE_INCREMENT:
            case IExpression_unary::OK_PRE_DECREMENT:
            case IExpression_unary::OK_POST_INCREMENT:
            case IExpression_unary::OK_POST_DECREMENT:
                {
                    IExpression const *arg = expr->get_argument();

                    // now handle the write
                    if (IDefinition const *def = get_lvalue_base(arg)) {
                        m_origins[def] = m_index;
                    }
                }
                break;
            }
            return expr;
        }

        IExpression *post_visit(IExpression_reference *ref) MDL_FINAL
        {
            IDefinition const *def = ref->get_definition();

            Origin_map::const_iterator it = m_origins.find(def);
            if (it != m_origins.end()) {
                int index = it->second;

                if (index != m_index) {
                    MDL_ASSERT(index < m_index);
                    m_def_index = index;
                    m_dep_def   = def;
                }
            }
            return ref;
        }

        IDefinition const *depends_on(int &index) const
        {
            index = m_def_index;
            return m_dep_def;
        }

    private:
        Origin_map &m_origins;

        /// Argument index of the current expression.
        int const m_index;

        /// The index of the defining argument.
        int m_def_index;

        /// Definition of the dependent entity.
        IDefinition const *m_dep_def;
    };

    Side_effect_checker checker(origins, index);

    (void)checker.visit(expr);
    return checker.depends_on(dep_index);
}

// Check the the arguments of a call are independent on evaluation order.
void NT_analysis::check_arguments_evaluation_order(
    IExpression_call const *call)
{
    Origin_map origins(
        0, Origin_map::hasher(), Origin_map::key_equal(), m_module.get_allocator());

    for (int i = 0, n = call->get_argument_count(); i < n; ++i) {
        IArgument const   *arg  = call->get_argument(i);
        IExpression const *expr = arg->get_argument_expr();

        int dep_index;
        if (IDefinition const *ent_def = has_side_effect(origins, expr, i, dep_index)) {
            IArgument const *dep_arg = call->get_argument(dep_index);

            if (IArgument_named const *narg = as<IArgument_named>(arg)) {
                warning(
                    NAME_ARGUMENT_VALUE_DEPENDS_ON_EVALUATION_ORDER,
                    arg->access_position(),
                    Error_params(*this)
                        .add(narg->get_parameter_name()->get_symbol())
                        .add_signature(ent_def));
            } else {
                warning(
                    POS_ARGUMENT_VALUE_DEPENDS_ON_EVALUATION_ORDER,
                    arg->access_position(),
                    Error_params(*this)
                        .add_numword(i + 1)
                        .add_signature(ent_def));
            }

            if (IArgument_named const *narg = as<IArgument_named>(dep_arg)) {
                add_note(
                    ASSIGNED_AT_PARAM_NAME,
                    dep_arg->access_position(),
                    Error_params(*this)
                    .add_signature(ent_def)
                    .add(narg->get_parameter_name()->get_symbol()));
            } else {
                add_note(
                    ASSIGNED_AT_PARAM_POS,
                    dep_arg->access_position(),
                    Error_params(*this)
                        .add_signature(ent_def)
                        .add_numword(dep_index + 1));
            }
        }
    }
}

// Check argument range.
void NT_analysis::check_argument_range(
    IDeclaration const *decl,
    IExpression const  *expr,
    int                idx)
{
    if (decl == NULL) {
        return;
    }

    if (IDeclaration_function const *f_decl = as<IDeclaration_function>(decl)) {
        if (f_decl->is_preset()) {
            // FIXME: check must be done on original
            return;
        }
        IParameter const   *param  = f_decl->get_parameter(idx);
        ISimple_name const *p_name = param->get_name();

        if (IAnnotation_block const *block = param->get_annotations()) {
            IAnnotation const *range =
                find_annotation_by_semantics(block, IDefinition::DS_HARD_RANGE_ANNOTATION);

            if (range == NULL) {
                return;
            }

            check_expression_range(range, expr, ARGUMENT_OUTSIDE_HARD_RANGE, p_name->get_symbol());
        }
    } else if (IDeclaration_type_struct const *s_decl = as<IDeclaration_type_struct>(decl)) {
        // when calling a struct constructor, the declaration of the struct is passed
        ISimple_name const *f_name = s_decl->get_field_name(idx);

        if (IAnnotation_block const *block = s_decl->get_annotations(idx)) {
            IAnnotation const *range =
                find_annotation_by_semantics(block, IDefinition::DS_HARD_RANGE_ANNOTATION);

            if (range == NULL) {
                return;
            }

            check_expression_range(range, expr, ARGUMENT_OUTSIDE_HARD_RANGE, f_name->get_symbol());
        }
    }
}

// replace default struct constructors by elemental constructors
Definition const *NT_analysis::reformat_default_struct_constructor(
    Definition const *callee_def,
    IExpression_call *call)
{
    IType_function const *f_type = cast<IType_function>(callee_def->get_type());
    IType_struct const   *s_type = cast<IType_struct>(f_type->get_return_type());

    // first step: find the elementary constructor inside THIS module
    Definition const *c_def = NULL;
    for (c_def = m_module.get_first_constructor(s_type);
        c_def != NULL;
        c_def = m_module.get_next_constructor(c_def))
    {
        if (c_def->get_semantics() == IDefinition::DS_ELEM_CONSTRUCTOR) {
            // found
            break;
        }
    }
    MDL_ASSERT(c_def != NULL && "could not find default constructor");

    // lookup the original definition to get default arguments
    Module const *origin = NULL;
    Definition const *orig_c_def = m_module.get_original_definition(callee_def, origin);

    f_type = cast<IType_function>(c_def->get_type());
    size_t param_count = f_type->get_parameter_count();

    // Use origin to rewrite resource URLs
    Default_initializer_modifier def_modifier(*this, param_count, origin);

    Expression_factory *fact = m_module.get_expression_factory();
    for (size_t i = 0; i < param_count; ++i) {
        IExpression const *expr = NULL;

        if (expr = orig_c_def->get_default_param_initializer(i)) {
            // this struct field has an initializer, clone it
            m_module.clone_expr(expr, &def_modifier);
        } else {
            // this struct field is default initialized
            IType const   *ptype;
            ISymbol const *psym;

            f_type->get_parameter(i, ptype, psym);

            IValue const *pval =
                m_module.create_default_value(m_module.get_value_factory(), ptype);
            expr = fact->create_literal(pval);
        }
        def_modifier.set_parameter_value(i, expr);

        IArgument_positional const *new_arg = fact->create_positional_argument(expr);
        call->add_argument(new_arg);
    }
    return c_def;
}

// Reformat and reorder the arguments of a call.
void NT_analysis::reformat_arguments(
     Definition const     *callee_def,
     IExpression_call     *call,
     IType const          *pos_arg_types[],
     size_t               pos_arg_count,
     Name_index_map const &name_arg_indexes,
     IArgument const      *arguments[],
     size_t               named_arg_count)
{
    bool is_preset_match = call == m_preset_overload;

    check_arguments_evaluation_order(call);
    bool function_mode = true;

    if (is_preset_match) {
        // always "named" mode
        function_mode = false;
    } else if (callee_def->get_kind() == Definition::DK_CONSTRUCTOR) {
        IType const *constr_type = get_result_type(callee_def);
        IType::Kind kind         = constr_type->get_kind();

        // constructors for builtin-types use "named" mode
        if (kind == IType::TK_BSDF || kind == IType::TK_HAIR_BSDF ||
            kind == IType::TK_VDF || kind == IType::TK_EDF)
        {
            function_mode = false;
        } else if (kind == IType::TK_STRUCT) {
            IType_struct const *s_type = cast<IType_struct>(constr_type);
            if (s_type->get_predefined_id() == IType_struct::SID_MATERIAL)
                function_mode = false;
        }
    }

    IType_function const *func_type = cast<IType_function>(callee_def->get_type());
    if (IType_struct const *s_type = as<IType_struct>(func_type->get_return_type())) {
        if (s_type->get_predefined_id() == IType_struct::SID_MATERIAL) {
            function_mode = false;
        }
    }

    Module const *origin = NULL;
    if (Definition const *orig_callee_def =
        m_module.get_original_definition(callee_def, origin))
    {
        // get the original definition, so we can access annotations and default arguments
        callee_def = orig_callee_def;
    }
    IDeclaration const *decl = callee_def->get_declaration();

    // Use origin to rewrite resource URLs
    Default_initializer_modifier def_modifier(*this, func_type->get_parameter_count(), origin);

    if (function_mode) {
        // make all arguments positional

        // first the positional arguments
        for (size_t k = 0; k < pos_arg_count; ++k) {
            IType const   *param_type;
            ISymbol const *param_sym;
            func_type->get_parameter(k, param_type, param_sym);

            param_type = param_type->skip_type_alias();

            IArgument const   *arg = call->get_argument(k);
            IExpression const *expr = arg->get_argument_expr();

            IType const *arg_type = pos_arg_types[k];
            if (need_type_conversion(arg_type, param_type)) {
                expr = convert_to_type_implicit(expr, param_type);

                MDL_ASSERT(expr != NULL && "Failed to find conversion constructor");
                const_cast<IArgument *>(arg)->set_argument_expr(expr);
            } else if (IType_array const *a_type = as<IType_array>(param_type)) {
                // bind the type here, the binding might be used for the return type
                if (IType_array const *a_param_type = as<IType_array>(arg_type)) {
                    if (needs_binding(a_type, a_param_type))
                        bind_array_type(a_type, a_param_type);
                }
            }

            check_argument_range(decl, expr, int(k));

            def_modifier.set_parameter_value(k, expr);
        }

        // now convert all names arguments into positional
        size_t n_params = func_type->get_parameter_count();

        Expression_factory *fact = m_module.get_expression_factory();

        for (size_t k = pos_arg_count; k < n_params; ++k) {
            IType const   *param_type;
            ISymbol const *param_sym;
            func_type->get_parameter(k, param_type, param_sym);

            param_type = param_type->skip_type_alias();

            Position const    *pos = NULL;
            IExpression const *expr;
            bool              arg_was_given = true;

            // find this parameter
            IArgument const *arg = NULL;
            Name_index_map::const_iterator it = name_arg_indexes.find(param_sym);
            if (it == name_arg_indexes.end()) {
                // not found, must have a default value
                expr = callee_def->get_default_param_initializer(k);
                MDL_ASSERT(expr && "Missing default expression");

                expr = m_module.clone_expr(expr, &def_modifier);
                arg_was_given = false;
            } else {
                size_t idx = it->second;
                arg = arguments[idx];
                pos = &arg->access_position();
                expr = arg->get_argument_expr();
            }

            IType const *arg_type = expr->get_type()->skip_type_alias();
            if (need_type_conversion(arg_type, param_type)) {
                expr = convert_to_type_implicit(expr, param_type);

                MDL_ASSERT(expr != NULL && "Failed to find conversion constructor");
                if (arg != NULL)
                    const_cast<IArgument *>(arg)->set_argument_expr(expr);
            } else if (IType_array const *a_type = as<IType_array>(param_type)) {
                // bind the type here, the binding might be used for the return type
                if (IType_array const *a_arg_type = as<IType_array>(arg_type)) {
                    if (needs_binding(a_type, a_arg_type))
                        bind_array_type(a_type, a_arg_type);
                }
            }

            def_modifier.set_parameter_value(k, expr);

            // create a new positional argument
            int sl = 0, sc = 0, el = 0, ec = 0;
            if (pos != NULL) {
                sl = pos->get_start_line();
                sc = pos->get_start_column();
                el = pos->get_end_line();
                ec = pos->get_end_column();
            }
            IArgument_positional const *new_arg = fact->create_positional_argument(
                expr, sl, sc, el, ec);

            if (k < pos_arg_count + named_arg_count) {
                call->replace_argument(k, new_arg);
            } else {
                call->add_argument(new_arg);
            }

            if (arg_was_given) {
                check_argument_range(decl, expr, int(k));
            }
        }
    } else {
        // make all arguments named
        Name_factory       *name_fact = m_module.get_name_factory();
        Expression_factory *expr_fact = m_module.get_expression_factory();

        // first the positional arguments
        for (size_t k = 0; k < pos_arg_count; ++k) {
            IType const   *param_type;
            ISymbol const *param_sym;
            func_type->get_parameter(k, param_type, param_sym);

            param_type = param_type->skip_type_alias();

            IArgument const   *arg  = call->get_argument(k);
            IExpression const *expr = arg->get_argument_expr();
            IType const *arg_type   = pos_arg_types[k];
            if (need_type_conversion(arg_type, param_type)) {
                expr = convert_to_type_implicit(expr, param_type);

                MDL_ASSERT(expr != NULL && "Failed to find conversion constructor");
                const_cast<IArgument *>(arg)->set_argument_expr(expr);
            } else if (IType_array const *a_type = as<IType_array>(param_type)) {
                // bind the type here, the binding might be used for the return type
                if (IType_array const *a_arg_type = as<IType_array>(arg_type)) {
                    if (needs_binding(a_type, a_arg_type))
                        bind_array_type(a_type, a_arg_type);
                }
            }
            def_modifier.set_parameter_value(k, expr);

            Position const &pos = arg->access_position();

            // create a new named argument
            int sl = pos.get_start_line(), sc = pos.get_start_column(),
                el = pos.get_end_line(), ec = pos.get_end_column();

            ISimple_name const    *p_name  = name_fact->create_simple_name(param_sym);
            IArgument_named const *new_arg = expr_fact->create_named_argument(
                p_name, expr, sl, sc, el, ec);

            if (k < pos_arg_count + named_arg_count) {
                call->replace_argument(k, new_arg);
            } else {
                call->add_argument(new_arg);
            }
        }

        // now reorder all names arguments and create default ones
        size_t n_params = func_type->get_parameter_count();

        size_t n_idx = pos_arg_count;
        for (size_t k = pos_arg_count; k < n_params; ++k) {
            IType const   *param_type;
            ISymbol const *param_sym;
            func_type->get_parameter(k, param_type, param_sym);

            param_type = param_type->skip_type_alias();

            IArgument const *new_arg;

            // find this parameter
            Name_index_map::const_iterator it = name_arg_indexes.find(param_sym);
            if (it == name_arg_indexes.end()) {
                if (is_preset_match) {
                    // this parameter is not set
                    continue;
                } else {
                    // not found, must have a default value
                    IExpression const *expr = callee_def->get_default_param_initializer(k);
                    MDL_ASSERT(expr && "Missing default expression");

                    expr = m_module.clone_expr(expr, &def_modifier);

                    ISimple_name const *p_name  = name_fact->create_simple_name(param_sym);
                    new_arg = expr_fact->create_named_argument(p_name, expr);
                }
            } else {
                size_t idx = it->second;
                new_arg = arguments[idx];
            }

            // check if a conversion is needed
            IExpression const *expr     = new_arg->get_argument_expr();
            IType const       *arg_type = expr->get_type()->skip_type_alias();
            if (need_type_conversion(arg_type, param_type)) {
                expr = convert_to_type_implicit(expr, param_type);

                MDL_ASSERT(expr != NULL && "Failed to find conversion constructor");
                const_cast<IArgument *>(new_arg)->set_argument_expr(expr);
            } else if (IType_array const *a_type = as<IType_array>(param_type)) {
                // bind the type here, the binding might be used for the return type
                if (IType_array const *a_arg_type = as<IType_array>(arg_type)) {
                    if (needs_binding(a_type, a_arg_type))
                        bind_array_type(a_type, a_arg_type);
                }
            }

            def_modifier.set_parameter_value(k, expr);

            if (n_idx < pos_arg_count + named_arg_count) {
                call->replace_argument(n_idx, new_arg);
            } else {
                call->add_argument(new_arg);
            }
            ++n_idx;
        }
    }
}

// Reformat and reorder the arguments of an annotation.
void NT_analysis::reformat_annotation_arguments(
     Definition const     *anno_def,
     IAnnotation          *anno,
     IType const          *pos_arg_types[],
     size_t               pos_arg_count,
     Name_index_map const &name_arg_indexes,
     IArgument const      *arguments[],
     size_t               named_arg_count)
{
    IType_function const *func_type = cast<IType_function>(anno_def->get_type());

    Module const *origin = NULL;
    if (Definition const *orig_callee_def =
        m_module.get_original_definition(anno_def, origin))
    {
        // get the original definition, so we can access default arguments
        anno_def = orig_callee_def;
    }

    Default_initializer_modifier def_modifier(*this, func_type->get_parameter_count(), NULL);

    // first the positional arguments
    for (size_t k = 0; k < pos_arg_count; ++k) {
        IType const   *param_type;
        ISymbol const *param_sym;
        func_type->get_parameter(k, param_type, param_sym);

        param_type = param_type->skip_type_alias();

        IType const *arg_type = pos_arg_types[k];
        if (need_type_conversion(arg_type, param_type)) {
            IArgument const   *arg = anno->get_argument(k);
            IExpression const *expr = arg->get_argument_expr();
            IExpression const *conv = convert_to_type_implicit(expr, param_type);

            MDL_ASSERT(conv != NULL && "Failed to find conversion constructor");
            const_cast<IArgument *>(arg)->set_argument_expr(conv);
        } else if (IType_array const *a_type = as<IType_array>(param_type)) {
            // bind the type here, the binding might be used for the return type
            if (IType_array const *a_param_type = as<IType_array>(arg_type)) {
                if (needs_binding(a_type, a_param_type)) {
                    bind_array_type(a_type, a_param_type);
                }
            }
        }
    }

    // now convert all names arguments into positional
    size_t n_params = func_type->get_parameter_count();

    Expression_factory *fact = m_module.get_expression_factory();

    for (size_t k = pos_arg_count; k < n_params; ++k) {
        IType const   *param_type;
        ISymbol const *param_sym;
        func_type->get_parameter(k, param_type, param_sym);

        param_type = param_type->skip_type_alias();

        Position const    *pos = NULL;
        IExpression const *expr;

        // find this parameter
        Name_index_map::const_iterator it = name_arg_indexes.find(param_sym);
        if (it == name_arg_indexes.end()) {
            // not found, must have a default value
            expr = anno_def->get_default_param_initializer(k);
            MDL_ASSERT(expr && "Missing default expression");

            expr = m_module.clone_expr(expr, &def_modifier);
        } else {
            size_t idx = it->second;
            IArgument const *arg = arguments[idx];
            pos = &arg->access_position();
            expr = arg->get_argument_expr();
        }

        IType const *arg_type = expr->get_type()->skip_type_alias();
        if (need_type_conversion(arg_type, param_type)) {
            IExpression const *conv = convert_to_type_implicit(expr, param_type);

            MDL_ASSERT(conv != NULL && "Failed to find conversion constructor");
            expr = conv;
        } else if (IType_array const *a_type = as<IType_array>(param_type)) {
            // bind the type here, the binding might be used for the return type
            if (IType_array const *a_arg_type = as<IType_array>(arg_type)) {
                if (needs_binding(a_type, a_arg_type))
                    bind_array_type(a_type, a_arg_type);
            }
        }

        // create a new positional argument
        int sl = 0, sc = 0, el = 0, ec = 0;
        if (pos != NULL) {
            sl = pos->get_start_line();
            sc = pos->get_start_column();
            el = pos->get_end_line();
            ec = pos->get_end_column();
        }
        IArgument_positional const* new_arg = fact->create_positional_argument(
            expr, sl, sc, el, ec);

        if (k < pos_arg_count + named_arg_count) {
            anno->replace_argument(k, new_arg);
        } else {
            anno->add_argument(new_arg);
        }
    }
}

// Convert one argument of an array constructor call.
bool NT_analysis::convert_array_cons_argument(
     IExpression_call     *call,
     int                  idx,
     IType const          *e_type)
{
    IArgument const   *arg    = call->get_argument(idx);
    IExpression const *expr   = arg->get_argument_expr();
    IType const       *arg_tp = expr->get_type();

    if (need_type_conversion(arg_tp, e_type)) {
        IExpression const *conv = convert_to_type_implicit(expr, e_type);

        if (conv != NULL) {
            const_cast<IArgument *>(arg)->set_argument_expr(conv);
        } else {
            error(
                CANNOT_CONVERT_ARRAY_ELEMENT,
                expr->access_position(),
                Error_params(*this)
                    .add_numword(idx + 1)
                    .add(expr->get_type())
                    .add(e_type));
            return false;
        }
    }
    return true;
}

// Start an import declaration.
bool NT_analysis::pre_visit(IDeclaration_import *import_decl)
{
    bool is_exported = import_decl->is_exported();
    if (IQualified_name const *mod_name = import_decl->get_module_name()) {
        // using <mod_name> import ..
        // import the following entities from this module

        // check for pre 1.3 restrictions
        if (m_module.get_mdl_version() < IMDL::MDL_VERSION_1_3) {
            ISimple_name const *first_name = mod_name->get_component(0);
            ISymbol const      *sym        = first_name->get_symbol();
            size_t             id          = sym->get_id();

            if (id == ISymbol::SYM_DOT || id == ISymbol::SYM_DOTDOT) {
                int major = 1, minor = 0;
                m_module.get_version(major, minor);

                error(
                    RELATIVE_IMPORTS_NOT_SUPPORTED,
                    first_name->access_position(),
                    Error_params(*this).add(major).add(minor));
            }
        }

        if (is_error(mod_name)) {
            // suppress extra error on erroneous name

            // do not visit children anymore
            return false;
        }

        mi::base::Handle<const Module> imp_mod(
            load_module_to_import(mod_name, /*ignore_last=*/false));

        if (imp_mod) {
            for (size_t i = 0, n = import_decl->get_name_count(); i < n; ++i) {
                IQualified_name const *ent_name = import_decl->get_name(i);

                if (is_star_import(ent_name)) {
                    import_all_entities(
                        mod_name, imp_mod.get(), is_exported, ent_name->access_position());
                } else {
                    // error will be reported in import_entity
                    import_entity(mod_name, imp_mod.get(), ent_name, is_exported);
                }
            }
        } else {
            // errors already reported
        }
    } else {
        // import ...
        if (is_exported) {
            error(
                REEXPORTING_QUALIFIED_NAMES_FORBIDDEN,
                import_decl->access_position(),
                Error_params(*this));
        }
        for (size_t i = 0, n = import_decl->get_name_count(); i < n; ++i) {
            IQualified_name const *mod_name = import_decl->get_name(i);

            // check for pre 1.3 restrictions
            if (m_module.get_mdl_version() < IMDL::MDL_VERSION_1_3) {
                ISimple_name const *first_name = mod_name->get_component(0);
                ISymbol const      *sym        = first_name->get_symbol();
                size_t             id          = sym->get_id();

                if (id == ISymbol::SYM_DOT || id == ISymbol::SYM_DOTDOT) {
                    int major = 1, minor = 0;
                    m_module.get_version(major, minor);

                    error(
                        RELATIVE_IMPORTS_NOT_SUPPORTED,
                        first_name->access_position(),
                        Error_params(*this).add(major).add(minor));
                }
            }

            if (is_error(mod_name)) {
                // suppress extra error on erroneous name
                continue;
            }
            import_qualified(mod_name, mod_name->access_position());
        }
    }

    // do not visit children anymore
    return false;
}

// Start a constant declaration.
bool NT_analysis::pre_visit(IDeclaration_constant *con_decl)
{
    IType_name const *t_name = con_decl->get_type_name();
    visit(t_name);

    Qualifier q = t_name->get_qualifier();
    if (q != FQ_NONE) {
        error(
            FORBIDDEN_QUALIFIER_ON_CONST_DECL,
            t_name->access_position(),
            Error_params(*this).add(q));
    }

    bool is_incomplete_arr = t_name->is_incomplete_array();

    IType const *type = as_type(t_name);

    IType const *bad_type = has_forbidden_variable_type(type);
    if (bad_type != NULL) {
        error(
            FORBIDDEN_CONSTANT_TYPE,
            t_name->access_position(),
            Error_params(*this).add(type));
        if (bad_type->skip_type_alias() != type->skip_type_alias()) {
            add_note(
                TYPE_CONTAINS_FORBIDDEN_SUBTYPE,
                t_name->access_position(),
                Error_params(*this)
                    .add(type)
                    .add(bad_type));
        }
        type = m_tc.error_type;
    }

    for (size_t i = 0, n = con_decl->get_constant_count(); i < n; ++i) {
        ISimple_name const *c_name = con_decl->get_constant_name(i);
        ISymbol const *sym = c_name->get_symbol();

        // check if this is already defined
        Definition *def = get_definition_at_scope(sym);
        bool def_is_error = false;
        if (def != NULL) {
            err_redeclaration(
                Definition::DK_CONSTANT,
                def,
                c_name->access_position(),
                ENT_REDECLARATION);
            def = get_error_definition();
            def_is_error = true;
        } else {
            IType const *def_type = is_incomplete_arr ? m_tc.error_type : type;
            def = m_def_tab->enter_definition(
                Definition::DK_CONSTANT, sym, def_type, &c_name->access_position());
            def->set_declaration(con_decl);

            if (con_decl->is_exported()) {
                export_definition(def);
            }

            // set he bad value until another one is set
            def->set_constant_value(m_module.get_value_factory()->create_bad());
        }
        const_cast<ISimple_name *>(c_name)->set_definition(def);

        Flag_store expected_const_expr(m_in_expected_const_expr, true);
        IExpression const *init = con_decl->get_constant_exp(i);

        if (init != NULL) {
            Definition::Scope_flag scope(*def, Definition::DEF_IS_INCOMPLETE);
            IExpression const *n_init = visit(init);
            if (n_init != init) {
                con_decl->set_variable_init(i, n_init);
                init = n_init;
            }

            if (def->has_flag(Definition::DEF_USED_INCOMPLETE)) {
                error(
                    FORBIDDEN_SELF_REFERENCE,
                    def,
                    Error_params(*this)
                        .add_signature(def));
            }

            bool is_invalid = false;
            if (! is_const_expression(init, is_invalid)) {
                if (! is_invalid) {
                    error(
                        NON_CONSTANT_INITIALIZER,
                        c_name->access_position(),
                        Error_params(*this).add(sym));
                }
            } else {
                if (!def_is_error) {
                    IType const *init_type = init->get_type();
                    if (is_incomplete_arr) {
                        // we do not allow array conversions
                        if (is<IType_error>(init_type)) {
                            // error already reported
                        } else if (is<IType_array>(init_type)) {
                            IType_array const *a_type = cast<IType_array>(init_type);
                            if (!equal_types(type, a_type->get_element_type())) {
                                error(
                                    NO_ARRAY_CONVERSION,
                                    init->access_position(),
                                    Error_params(*this).add(init_type).add(type, -1));
                            } else {
                                // element types are equal, fix the incomplete type
                                def->set_type(a_type);
                            }
                        } else {
                            error(
                                NO_ARRAY_CONVERSION,
                                init->access_position(),
                                Error_params(*this).add(init_type).add(type, -1));
                        }
                    } else if (IType_array const *a_type = as<IType_array>(type)) {
                        // we do not allow array conversions
                        if (is<IType_error>(init_type)) {
                            // error already reported
                        } else if (! equal_types(a_type, init_type)) {
                            error(
                                NO_ARRAY_CONVERSION,
                                init->access_position(),
                                Error_params(*this).add(init_type).add(type));
                        }
                    } else {
                        if (is<IType_array>(init_type)) {
                            error(
                                NO_ARRAY_CONVERSION,
                                init->access_position(),
                                Error_params(*this).add(init_type).add(type));
                        } else {
                            // find the default constructor
                            init = find_init_constructor(
                                type,
                                m_module.clone_name(t_name, /*modifier=*/NULL),
                                init,
                                init->access_position());
                        }
                    }

                    if (!is<IType_error>(init->get_type())) {
                        // we have a valid init constructor
                        m_exc_handler.clear_error_state();
                        IValue const *val = init->fold(
                            &m_module, m_module.get_value_factory(), &m_exc_handler);
                        if (! m_exc_handler.has_error()) {
                            MDL_ASSERT(
                                !is<IValue_bad>(val) &&
                                "const folding failed for constant expression");

                            def->set_constant_value(val);

                            if (! is<IExpression_literal>(init)) {
                                Position const *pos = &init->access_position();
                                init = m_module.create_literal(val, pos);
                            }
                        }
                    }
                }
                con_decl->set_variable_init(i, init);
            }
        } else {
            error(
                MISSING_CONSTANT_INITIALIZER,
                c_name->access_position(),
                Error_params(*this).add(sym));
        }

        if (IAnnotation_block const *anno = con_decl->get_annotations(i)) {
            Definition_store store(m_annotated_def, def);
            visit(anno);
        }

        if (!m_is_stdlib && !is_error(def)) {
            calc_mdl_versions(def);
        }
    }
    // do not visit children anymore
    return false;
}

// start an annotation declaration
bool NT_analysis::pre_visit(IDeclaration_annotation *anno_decl)
{
    ISimple_name const *a_name = anno_decl->get_name();
    ISymbol const      *a_sym  = a_name->get_symbol();

    bool               has_error = false;

    Definition *a_def = get_definition_at_scope(a_sym);
    if (a_def != NULL && a_def->get_kind() != Definition::DK_ANNOTATION) {
        err_redeclaration(
            Definition::DK_ANNOTATION, a_def, anno_decl->access_position(), ENT_REDECLARATION);
        has_error = true;
    }

    // for now, set the error definition, update it later
    a_def = get_error_definition();

    // create new scope for the parameters
    Scope *anno_scope = NULL;
    m_next_param_idx = 0;
    {
        Definition_table::Scope_enter scope(*m_def_tab, a_def);

        anno_scope = m_def_tab->get_curr_scope();

        // annotation parameters are NEVER unused. Mark them.
        Flag_store params_are_used(m_params_are_used, true);

        for (int i = 0, n = anno_decl->get_parameter_count(); i < n; ++i) {
            IParameter const *param = anno_decl->get_parameter(i);

            visit(param);
        }
    }

    size_t n_params = anno_decl->get_parameter_count();
    VLA<IType_factory::Function_parameter> params(get_allocator(), n_params);
    VLA<IExpression const *>               inits(get_allocator(), n_params);

    bool has_initializers = false;

    // handle function parameter
    for (size_t i = 0; i < n_params; ++i) {
        IParameter const   *param = anno_decl->get_parameter(i);

        IType_name const   *tname = param->get_type_name();
        IType const        *ptype = tname->get_type();

        ISimple_name const *pname = param->get_name();
        ISymbol const      *psym  = pname->get_symbol();

        IType const *bad_type = has_forbidden_annotation_parameter_type(ptype);
        if (bad_type != NULL) {
            error(
                FORBIDDEN_ANNOTATION_PARAMETER_TYPE,
                pname->access_position(),
                Error_params(*this).add(ptype));
            if (bad_type->skip_type_alias() != ptype->skip_type_alias()) {
                add_note(
                    TYPE_CONTAINS_FORBIDDEN_SUBTYPE,
                    pname->access_position(),
                    Error_params(*this)
                    .add(ptype)
                    .add(bad_type));
            }
            ptype = m_tc.error_type;
        }

        params[i].p_type = ptype;
        params[i].p_sym  = psym;
        inits[i]         = NULL;

        if (is<IType_error>(ptype)) {
            has_error = true;
        }

        // there should be no init expression here
        if (IExpression const *init = param->get_init_expr()) {
            has_initializers = true;

            // must be a const expression
            bool is_invalid = false;
            if (!is_const_expression(init, is_invalid)) {
                has_error = true;

                if (!is_invalid) {
                    error(
                        ANNONATION_NON_CONST_DEF_PARAM,
                        pname->access_position(),
                        Error_params(*this).add(psym));
                }
                // set the type to error, so it will not be reported again
                const_cast<IExpression *>(init)->set_type(m_tc.error_type);
            }

            inits[i] = init;
        }
    }

    if (!has_error) {
        IType_function const *anno_type = m_tc.create_function(NULL, params);
        a_def = m_def_tab->enter_definition(
            Definition::DK_ANNOTATION, a_sym, anno_type, &anno_decl->access_position());
        a_def->set_own_scope(anno_scope);
        if (m_is_stdlib)
            a_def->set_flag(Definition::DEF_IS_STDLIB);
        a_def->set_declaration(anno_decl);

        if (has_initializers)
            m_module.allocate_initializers(a_def, n_params);

        // check default arguments
        bool need_default_arg = false;
        for (size_t i = 0; i < n_params; ++i) {
            if (IExpression const *init = inits[i]) {
                a_def->set_default_param_initializer(i, init);
                need_default_arg = true;
            } else if (need_default_arg) {
                IParameter const   *param  = anno_decl->get_parameter(i);
                ISimple_name const *p_name = param->get_name();

                error(
                    MISSING_DEFAULT_ARGUMENT,
                    p_name->access_position(),
                    Error_params(*this).add(i + 1)
                        .add_signature(a_def));
            }
        }

        if (!m_is_stdlib)
            calc_mdl_versions(a_def);
        if (anno_decl->is_exported())
            export_definition(a_def);
    }

    const_cast<ISimple_name *>(a_name)->set_definition(a_def);
    anno_decl->set_definition(a_def);

    if (IAnnotation_block const *anno = anno_decl->get_annotations()) {
        Definition_store store(m_annotated_def, a_def);

        if (m_module.get_version() < IMDL::MDL_VERSION_1_5) {
            int major, minor;
            m_module.get_version(major, minor);

            warning(
                ANNOS_ON_ANNO_DECL_NOT_SUPPORTED,
                anno->access_position(),
                Error_params(*this).add(major).add(minor));

            anno_decl->set_annotations(NULL);
            m_annotated_def = get_error_definition();
        }

        visit(anno);
    }

    if (m_is_stdlib) {
        // handle standard annotations known by the compiler
        string abs_name(m_module.get_name(), m_module.get_allocator());
        abs_name += "::";
        abs_name += a_def->get_sym()->get_name();

        IDefinition::Semantics sema = m_compiler->get_builtin_semantic(abs_name.c_str());

        if (sema != IDefinition::DS_UNKNOWN) {
            a_def->set_semantic(sema);

            switch (sema) {
            case IDefinition::DS_SOFT_RANGE_ANNOTATION:
            case  IDefinition::DS_HARD_RANGE_ANNOTATION:
                {
                    for (size_t i = 0; i < n_params; ++i) {
                        if (!is<IType_atomic>(params[i].p_type)) {
                            // range annotations for non-atomic types are available from MDL 1.2
                            replace_since_version(a_def, 1, 2);
                            break;
                        }
                    }
                }
                break;
            case IDefinition::DS_VERSION_NUMBER_ANNOTATION:
                // removed from MDL 1.3
                replace_removed_version(a_def, 1, 3);
                break;
            case IDefinition::DS_DEPRECATED_ANNOTATION:
            case IDefinition::DS_VERSION_ANNOTATION:
            case IDefinition::DS_DEPENDENCY_ANNOTATION:
                // these annotations are available from MDL 1.3
                replace_since_version(a_def, 1, 3);
                break;
            case IDefinition::DS_UI_ORDER_ANNOTATION:
            case IDefinition::DS_USAGE_ANNOTATION:
            case IDefinition::DS_ENABLE_IF_ANNOTATION:
            case IDefinition::DS_THUMBNAIL_ANNOTATION:
                // these annotations are available from MDL 1.4
                replace_since_version(a_def, 1, 4);
                break;
            case IDefinition::DS_ORIGIN_ANNOTATION:
                // these annotations are available from MDL 1.5
                replace_since_version(a_def, 1, 5);
                break;
            default:
                break;
            }
        }
    }

    // don't visit children anymore
    return false;
}

// Start of an alias declaration
bool NT_analysis::pre_visit(IDeclaration_type_alias *alias_decl)
{
    IType_name const *tname = alias_decl->get_type_name();
    visit(tname);

    if (tname->is_incomplete_array()) {
        error(
            INCOMPLETE_ARRAY_SPECIFICATION,
            tname->access_position(),
            Error_params(*this));
        const_cast<IType_name*>(tname)->set_type(m_tc.error_type);
    }

    IType const *type = as_type(tname);

    ISimple_name const *name = alias_decl->get_alias_name();
    ISymbol const      *sym  = name->get_symbol();

    // create a fully qualified name
    string alias_name(get_current_scope_name());
    if (!alias_name.empty())
        alias_name += "::";
    alias_name += sym->get_name();

    ISymbol const *fq_sym = m_module.get_symbol_table().get_user_type_symbol(alias_name.c_str());

    Definition *def = get_definition_at_scope(sym);
    if (def != NULL) {
        err_redeclaration(Definition::DK_TYPE, def, name->access_position(), TYPE_REDECLARATION);
        def = get_error_definition();
    } else {
        // create an alias type here, it will map to error if type is the error type
        IType const *alias_type = m_tc.create_alias(type, fq_sym, IType::MK_NONE);
        def = m_def_tab->enter_definition(
            Definition::DK_TYPE, sym, alias_type, &name->access_position());
        def->set_declaration(alias_decl);

        if (alias_decl->is_exported())
            export_definition(def);
    }
    const_cast<ISimple_name *>(name)->set_definition(def);

    // don't visit children anymore
    return false;
}

// Start an enum type declaration
bool NT_analysis::pre_visit(IDeclaration_type_enum *enum_decl)
{
    ISymbol const *sym = enum_decl->get_name()->get_symbol();

    bool is_enum_class = enum_decl->is_enum_class();

    // create a fully qualified name
    string enum_name(get_current_scope_name());
    if (!enum_name.empty())
        enum_name += "::";
    enum_name += sym->get_name();

    ISymbol const *fq_sym = m_module.get_symbol_table().get_user_type_symbol(enum_name.c_str());

    // UGLY: because ::tex::gamma_mode type must be a builtin-type, detect its "declaration"
    // and handle it gracefully
    bool is_builtin_enum_type =
        fq_sym->get_id() == ISymbol::SYM_TYPE_TEX_GAMMA_MODE &&
        m_module.is_stdlib() &&
        strcmp("::tex", m_module.get_name()) == 0;

    IType_enum *e_type = is_builtin_enum_type ?
        const_cast<IType_enum *>(m_tc.tex_gamma_mode_type) : m_tc.create_enum(fq_sym);

    Definition *type_def = get_definition_at_scope(sym);
    if (type_def != NULL) {
        err_redeclaration(
            Definition::DK_TYPE, type_def, enum_decl->access_position(), TYPE_REDECLARATION);
        type_def = get_error_definition();
    } else {
        type_def = m_def_tab->enter_definition(
            Definition::DK_TYPE, sym, e_type, &enum_decl->access_position());
        type_def->set_declaration(enum_decl);

        if (enum_decl->is_exported())
            export_definition(type_def);
    }
    enum_decl->set_definition(type_def);

    if (IAnnotation_block const *anno = enum_decl->get_annotations()) {
        Definition_store store(m_annotated_def, type_def);
        visit(anno);
    }

    // create a type scope
    Scope *type_scope = m_def_tab->enter_scope(e_type, type_def);
    if (!is_error(type_def)) {
        type_def->set_own_scope(type_scope);

        // create the default operators here, we might need them for the enum value initializers
        create_default_operators(e_type, sym, type_def, enum_decl);
    }

    if (! is_enum_class) {
        // non-class Enumerations are defined in the scope of the declaration itself, not inside
        // the scope of the declaration, so leave it again.
        m_def_tab->leave_scope();
    }

    // another work-around for the builtin enum types: Add values to the builtin enum types only
    // the first time we encounter the declaration. This happens when the standard library is
    // parsed, so the time frame where builtin enum types have no values is luckily short.
    bool add_values = !is_builtin_enum_type;
    if (is_builtin_enum_type && e_type->get_value_count() == 0) {
        // first time we encounter this enum
        add_values = true;
    }

    int code = 0;
    bool had_errors = false;
    for (int idx = 0, n = enum_decl->get_value_count(); idx < n; ++idx) {
        ISimple_name const *v_name = enum_decl->get_value_name(idx);
        ISymbol const      *v_sym  = v_name->get_symbol();

        if (IExpression const *init = enum_decl->get_value_init(idx)) {
            Flag_store expected_const_expr(m_in_expected_const_expr, true);
            IExpression const *n_init = visit(init);
            if (n_init != init) {
                enum_decl->set_value_init(idx, n_init);
                init = n_init;
            }

            // evaluate to int
            IType const *init_tp = init->get_type()->skip_type_alias();

            if (!is_integer_or_enum_type(init_tp)) {
                if (!is<IType_error>(init_tp) && !is_syntax_error(v_sym)) {
                    error(
                        ENUMERATOR_VALUE_NOT_INT,
                        v_name->access_position(),
                        Error_params(*this).add(v_sym));
                }
            } else {
                bool is_invalid = false;
                if (!is_const_expression(init, is_invalid)) {
                    if (!is_invalid) {
                        error(
                            NON_CONSTANT_INITIALIZER,
                            v_name->access_position(),
                            Error_params(*this).add(v_sym));
                    }
                } else {
                    m_exc_handler.clear_error_state();
                    IValue const *val = init->fold(
                        &m_module, m_module.get_value_factory(), &m_exc_handler);
                    if (IValue_int const *iv = as<IValue_int>(val)) {
                        code = iv->get_value();
                    } else if (IValue_enum const *ev = as<IValue_enum>(val)) {
                        code = ev->get_value();
                    }
                    MDL_ASSERT(
                        (m_exc_handler.has_error() || !is<IValue_bad>(val)) &&
                        "const folding failed for constant expression");

                    if (! is<IExpression_literal>(init)) {
                        Position const *pos = &init->access_position();
                        init = m_module.create_literal(val, pos);
                        enum_decl->set_value_init(idx, init);
                    }
                }
            }
        }

        Definition *v_def = get_definition_at_scope(v_sym);
        if (v_def != NULL) {
            err_redeclaration(
                Definition::DK_ENUM_VALUE, v_def, v_name->access_position(), ENT_REDECLARATION);
            v_def = get_error_definition();
        } else {
            if (is_syntax_error(v_sym)) {
                // parse errors on names are expressed by error symbols
                v_def = get_error_definition();
            } else {
                v_def = m_def_tab->enter_definition(
                    Definition::DK_ENUM_VALUE, v_sym, e_type, &v_name->access_position());

                if (enum_decl->is_exported())
                    export_definition(v_def);
            }
        }

        // set the definition for the value
        const_cast<ISimple_name *>(v_name)->set_definition(v_def);

        int t_idx = idx;
        if (is_error(v_def)) {
            had_errors = true;
        } else {
            if (add_values) {
                // builtin enum types are already created, do not modify them
                t_idx = e_type->add_value(v_sym, code);
                MDL_ASSERT(t_idx == idx || had_errors);
            }
            IValue const *enum_value = m_module.get_value_factory()->create_enum(e_type, t_idx);
            v_def->set_constant_value(enum_value);
        }
        ++code;

        if (IAnnotation_block const *anno = enum_decl->get_annotations(idx)) {
            Definition_store store(m_annotated_def, v_def);
            visit(anno);
        }
    }

    if (is_enum_class) {
        m_def_tab->leave_scope();
    }

    create_default_constructors(e_type, sym, type_def, enum_decl);

    const_cast<ISimple_name *>(enum_decl->get_name())->set_definition(type_def);

    // don't visit children anymore
    return false;
}

// Start a struct type declaration.
bool NT_analysis::pre_visit(IDeclaration_type_struct *struct_decl)
{
    ISymbol const *sym = struct_decl->get_name()->get_symbol();

    // create a fully qualified name to name the type
    string struct_name(get_current_scope_name());
    if (! struct_name.empty())
        struct_name += "::";
    struct_name += sym->get_name();

    ISymbol const *fq_sym = m_module.get_symbol_table().get_user_type_symbol(struct_name.c_str());

    IType_struct *s_type = m_tc.create_struct(fq_sym);

    Definition *type_def = get_definition_at_scope(sym);
    if (type_def) {
        err_redeclaration(
            Definition::DK_TYPE, type_def, struct_decl->access_position(), TYPE_REDECLARATION);
        type_def = get_error_definition();
    } else {
        type_def = m_def_tab->enter_definition(
            Definition::DK_TYPE, sym, s_type, &struct_decl->access_position());
        type_def->set_declaration(struct_decl);

        // cannot create instances of this type until it is completed
        type_def->set_flag(Definition::DEF_IS_INCOMPLETE);

        if (struct_decl->is_exported())
            export_definition(type_def);
    }
    struct_decl->set_definition(type_def);

    if (IAnnotation_block const *anno = struct_decl->get_annotations()) {
        Definition_store store(m_annotated_def, type_def);
        visit(anno);
    }

    // create a new type scope
    {
        Definition_table::Scope_enter scope(*m_def_tab, s_type, type_def);

        for (int i = 0, n = struct_decl->get_field_count(); i < n; ++i) {
            IType_name const *t_name = struct_decl->get_field_type_name(i);
            visit(t_name);

            if (t_name->is_incomplete_array()) {
                error(
                    INCOMPLETE_ARRAY_SPECIFICATION,
                    t_name->access_position(),
                    Error_params(*this));
                const_cast<IType_name*>(t_name)->set_type(m_tc.error_type);
            }

            IType const        *f_type = as_type(t_name);
            ISimple_name const *f_name = struct_decl->get_field_name(i);
            ISymbol const      *f_sym  = f_name->get_symbol();

            if (IType_array const *a_type = as<IType_array>(f_type)) {
                if (!a_type->is_immediate_sized()) {
                    error(
                        ABSTRACT_ARRAY_INSIDE_STRUCT,
                        t_name->access_position(),
                        Error_params(*this).add(f_type).add(f_sym));
                    f_type = m_tc.error_type;
                }
            }
            if (!m_is_stdlib && !is_allowed_field_type(f_type)) {
                error(
                    FORBIDDEN_FIELD_TYPE,
                    t_name->access_position(),
                    Error_params(*this).add(f_type));
                f_type = m_tc.error_type;
            }

            Definition *f_def  = NULL;

            if (f_sym == sym) {
                // we do not allow a field of the name of the constructor
                if (!is<IType_error>(f_type)) {
                    error(
                        CONSTRUCTOR_SHADOWED,
                        f_name->access_position(),
                        Error_params(*this).add(f_type).add(sym).add(f_sym).add("struct"));
                }
                f_def = get_error_definition();
            } else {
                f_def = get_definition_at_scope(f_sym);
                if (f_def != NULL) {
                    err_redeclaration(
                        Definition::DK_MEMBER,
                        f_def,
                        f_name->access_position(),
                        ENT_REDECLARATION);
                    f_def = get_error_definition();
                } else {
                    f_def = m_def_tab->enter_definition(
                        Definition::DK_MEMBER, f_sym, f_type, &f_name->access_position());

                    f_def->set_field_index(i);
                    s_type->add_field(f_type, f_sym);
                }
            }

            // set the definition for the field
            const_cast<ISimple_name *>(f_name)->set_definition(f_def);

            IExpression const *init = struct_decl->get_field_init(i);
            if (init != NULL) {
                Definition::Scope_flag scope(*f_def, Definition::DEF_IS_INCOMPLETE);
                IExpression const *n_init = visit(init);
                if (n_init != init) {
                    struct_decl->set_field_init(i, n_init);
                    init = n_init;
                }
            }

            if (!is_error(f_def)) {
                if (IType_array const *a_type = as<IType_array>(f_type)) {
                    if (init != NULL) {
                        IType const *init_type = init->get_type();

                        // we do not allow array conversions
                        if (is<IType_error>(init_type)) {
                            // error already reported
                        } else if (!equal_types(a_type, init_type)) {
                            error(
                                NO_ARRAY_CONVERSION,
                                init->access_position(),
                                Error_params(*this).add(init_type).add(f_type));
                        }
                    }
                } else if (init != NULL) {
                    IType const *init_type = init->get_type();

                    if (is<IType_array>(init_type)) {
                        error(
                            NO_ARRAY_CONVERSION,
                            init->access_position(),
                            Error_params(*this).add(init_type).add(f_type));
                    } else {
                        if (!equal_types(init_type, f_type->skip_type_alias())) {
                            // find the constructor
                            Position const &pos = init->access_position();

                            init = find_init_constructor(
                                f_type, m_module.clone_name(t_name, /*modifier=*/NULL), init, pos);
                        }
                        struct_decl->set_field_init(i, init);
                    }
                }
            }

            if (IAnnotation_block const *anno = struct_decl->get_annotations(i)) {
                Definition_store store(m_annotated_def, f_def);
                visit(anno);

                // only check init expression if there is at least an annotation block
                if (IExpression const *init = struct_decl->get_field_init(i)) {
                    if (!is_error(init)) {
                        check_field_range(anno, init);
                    }
                }
            }
        }

        create_default_members(s_type, sym, type_def, struct_decl);
    }

    // clear the incomplete flag
    if (!is_error(type_def)) {
        type_def->clear_flag(Definition::DEF_IS_INCOMPLETE);
    }

    const_cast<ISimple_name *>(struct_decl->get_name())->set_definition(type_def);

    // don't visit children anymore
    return false;
}

// Start of a parameter.
bool NT_analysis::pre_visit(IParameter *param)
{
    IType_name const *t_name = param->get_type_name();
    {
        Store<Match_restriction> restrict_type(m_curr_restriction, MR_TYPE);
        visit(t_name);
    }

    if (t_name->is_incomplete_array()) {
        error(
            INCOMPLETE_ARRAY_SPECIFICATION,
            t_name->access_position(),
            Error_params(*this));
        const_cast<IType_name*>(t_name)->set_type(m_tc.error_type);
    }

    IType const *p_type = as_type(t_name);

    ISimple_name const *p_name = param->get_name();
    ISymbol      const *p_sym  = p_name->get_symbol();

    Definition *p_def = get_definition_at_scope(p_sym);
    if (p_def != NULL) {
        err_redeclaration(
            Definition::DK_PARAMETER, p_def, p_name->access_position(), PARAMETER_REDECLARATION);
        p_def = get_error_definition();
    } else {
        p_def = m_def_tab->enter_definition(
            Definition::DK_PARAMETER, p_sym, p_type, &p_name->access_position());
        p_def->set_parameter_index(m_next_param_idx);
        if (m_params_are_used)
            p_def->set_flag(Definition::DEF_IS_USED);
    }
    const_cast<ISimple_name *>(p_name)->set_definition(p_def);

    if (IExpression const *init = param->get_init_expr()) {
        Definition::Scope_flag scope(*p_def, Definition::DEF_IS_INCOMPLETE);
        Flag_store             inside_param_initializer(m_inside_param_initializer, true);

        IExpression const *n_init = visit(init);
        if (n_init != init) {
            param->set_init_expr(n_init);
            init = n_init;
        }

        if (p_def->has_flag(Definition::DEF_USED_INCOMPLETE)) {
            error(
                FORBIDDEN_SELF_REFERENCE,
                p_def,
                Error_params(*this)
                    .add_signature(p_def));
        }
    }

    if (IAnnotation_block const *anno = param->get_annotations()) {
        Definition_store store(m_annotated_def, p_def);
        visit(anno);
    }

    ++m_next_param_idx;

    // don't visit children anymore
    return false;
}

// Start of a function or a material.
bool NT_analysis::pre_visit(IDeclaration_function *fkt_decl)
{
    bool is_function = true;

    IType_name const *ret_name = fkt_decl->get_return_type_name();
    if (!ret_name->is_array()) {
        IQualified_name const *q_name = ret_name->get_qualified_name();

        if (is_material_qname(q_name)) {
            is_function = false;
        }
    }

    if (fkt_decl->is_preset()) {
        if (is_function && m_module.get_mdl_version() >= IMDL::MDL_VERSION_1_3) {
            // handle function presets in MDL 1.3+
            declare_function_preset(fkt_decl);
        } else {
            // handle material presets
            declare_material_preset(fkt_decl);
        }
    } else {
        if (is_function) {
            // handle functions
            declare_function(fkt_decl);
        } else {
            // handle materials
            declare_material(fkt_decl);
        }
    }

    // don't visit children anymore
    return false;
}

// Start of module declaration
bool NT_analysis::pre_visit(IDeclaration_module *d)
{
    if (IAnnotation_block const *anno = d->get_annotations()) {
        Flag_store is_module_annotation(m_is_module_annotation, true);

        visit(anno);
    }

    // don't visit children anymore
    return false;
}

/// Check restrictions on package names.
static char check_allowed_chars(char const *s)
{
    /// The package names and the module name are regular MDL identifiers(Section 5.5)
    /// or string literals (Section 5.7.5).The string literals enable the use of Unicode names
    /// for modules and packages.They must not contain control codes(ASCII code < 32), delete
    /// (ASCII code 127), colon ':', back slash '\', and forward slash '/', which is reserved
    /// as a path separator.These restrictions match the ones for file paths in Section 2.2.
    for (char const *p = s; *p != '\0';) {
        unsigned res = 0;
        p = utf8_to_unicode_char(p, res);

        switch (res) {
        case 127:
        case ':':
        case '\\':
        case '/':
            return res;
        default:
            if (res < 32)
                return res;
            break;
        }
    }
    return 'A';
}

// Start of a namespace alias
bool NT_analysis::pre_visit(IDeclaration_namespace_alias *alias_decl)
{
    if (m_module.get_mdl_version() < IMDL::MDL_VERSION_1_6) {
        // this feature does not exists prior to 1.6;
        // we could either handle it, assuming just the version number is wrong, or
        // ignore it (which probably creates more errors)
        error(
            USING_ALIAS_DECL_FORBIDDEN,
            alias_decl->access_position(),
            Error_params(*this));
    }

    ISimple_name const *alias = alias_decl->get_alias();
    ISymbol const      *a_sym = alias->get_symbol();

    IQualified_name const *ns = alias_decl->get_namespace();

    bool is_absolute = ns->is_absolute();

    string ns_name(is_absolute ? "::" : "", m_builder.get_allocator());

    for (int i = 0, n = ns->get_component_count(); i < n; ++i) {
        if (i > 0)
            ns_name.append("::");
        char const *s = ns->get_component(i)->get_symbol()->get_name();

        char bad = check_allowed_chars(s);
        if (bad != 'A') {
            error(
                PACKAGE_NAME_CONTAINS_FORBIDDEN_CHAR,
                ns->get_component(i)->access_position(),
                Error_params(*this).add_char(bad));
        }
        ns_name.append(s);
    }

    Definition const *def = m_def_tab->get_namespace_alias(a_sym);
    if (def != NULL) {
        err_redeclaration(
            IDefinition::DK_NAMESPACE,
            def,
            alias->access_position(),
            USING_ALIAS_REDECLARATION);
        def = get_error_definition();
    } else {
        ISymbol const *ns = m_st->create_shared_symbol(ns_name.c_str());
        def = m_def_tab->enter_namespace_alias(a_sym, ns, alias_decl);
    }
    const_cast<ISimple_name *>(alias)->set_definition(def);

    // don't visit children anymore
    return false;
}

// Start of a variable declaration
bool NT_analysis::pre_visit(IDeclaration_variable *var_decl)
{
    IType_name const *t_name = var_decl->get_type_name();
    visit(t_name);

    bool is_incomplete_arr = t_name->is_incomplete_array();

    IType const *var_type = as_type(t_name);

    if (!m_inside_material_constr) {
        IType const *bad_type = has_forbidden_variable_type(var_type);
        if (bad_type != NULL) {
            error(
                FORBIDDEN_VARIABLE_TYPE,
                t_name->access_position(),
                Error_params(*this).add(var_type));
            if (bad_type->skip_type_alias() != var_type->skip_type_alias()) {
                add_note(
                    TYPE_CONTAINS_FORBIDDEN_SUBTYPE,
                    t_name->access_position(),
                    Error_params(*this)
                        .add(var_type)
                        .add(bad_type));
            }
            var_type = m_tc.error_type;
        }
    }

    for (size_t i = 0, n = var_decl->get_variable_count(); i < n; ++i) {
        ISimple_name  *v_name = const_cast<ISimple_name *>(var_decl->get_variable_name(i));
        ISymbol const *v_sym  = v_name->get_symbol();

        Definition *v_def = m_def_tab->get_definition(v_sym);
        if (v_def != NULL) {
            if (v_def->get_def_scope() == m_def_tab->get_curr_scope()) {
                if (v_def->get_kind() == Definition::DK_PARAMETER) {
                    // suppress "redeclaration as different kind ..." for params
                    // by saying that we declaring another param
                    err_redeclaration(
                        Definition::DK_PARAMETER,
                        v_def,
                        v_name->access_position(),
                        PARAMETER_REDECLARATION);
                } else {
                    // normal case
                    err_redeclaration(
                        Definition::DK_VARIABLE,
                        v_def,
                        v_name->access_position(),
                        ENT_REDECLARATION);
                }
                v_def = get_error_definition();
            } else {
                // warn if we shadow a parameter
                if (v_def->get_kind() == Definition::DK_PARAMETER)
                    warn_shadow(v_def, v_name->access_position(), PARAMETER_SHADOWED);
                v_def = NULL;
            }
        }

        if (v_def == NULL) {
            v_def = m_def_tab->enter_definition(
                Definition::DK_VARIABLE, v_sym, var_type, &v_name->access_position());

            if (m_inside_let_decls) {
                // Variables inside a let are temporaries only and cannot be reassigned.
                // We could model this by making its type const, but this would have a bad
                // effect on the auto type analysis, hence use a special flag.
                v_def->set_flag(Definition::DEF_IS_LET_TEMPORARY);
            }

            v_def->set_declaration(var_decl);
        }
        v_name->set_definition(v_def);

        IExpression const *init = var_decl->get_variable_init(i);
        if (init != NULL) {
            Definition::Scope_flag scope(*v_def, Definition::DEF_IS_INCOMPLETE);
            IExpression const *n_init = visit(init);
            if (n_init != init) {
                var_decl->set_variable_init(i, n_init);
                init = n_init;
            }

            if (v_def->has_flag(Definition::DEF_USED_INCOMPLETE)) {
                error(
                    FORBIDDEN_SELF_REFERENCE,
                    v_def,
                    Error_params(*this)
                        .add_signature(v_def));
            }
        }

        if (! is_error(v_def)) {
            if (is_incomplete_arr) {
                if (init != NULL) {
                    // we do not allow array conversions
                    IType const *init_type = init->get_type();

                    if (is<IType_error>(init_type)) {
                        // error already reported
                    } else if (is<IType_array>(init_type)) {
                        IType_array const *a_type = cast<IType_array>(init_type);
                        if (!equal_types(var_type, a_type->get_element_type())) {
                            error(
                                NO_ARRAY_CONVERSION,
                                init->access_position(),
                                Error_params(*this).add(init_type).add(var_type, -1));
                        } else {
                            // element types are equal, fix the incomplete type
                            v_def->set_type(a_type);
                        }
                    }
                } else {
                    // we need an initializer for incomplete arrays
                    error(
                        INCOMPLETE_ARRAY_SPECIFICATION,
                        *v_def->get_position(),
                        Error_params(*this));
                }
            } else if (IType_array const *a_type = as<IType_array>(var_type)) {
                if (init != NULL) {
                    // we do not allow array conversions
                    IType const *init_type = init->get_type();

                    if (is<IType_error>(init_type)) {
                        // error already reported
                    } else if (! equal_types(a_type, init_type)) {
                        error(
                            NO_ARRAY_CONVERSION,
                            init->access_position(),
                            Error_params(*this).add(init_type).add(var_type));
                    }
                }
            } else {
                bool init_err = false;
                if (init != NULL) {
                    IType const *init_type = init->get_type();
                    if (is<IType_array>(init_type)) {
                        error(
                            NO_ARRAY_CONVERSION,
                            init->access_position(),
                            Error_params(*this).add(init_type).add(var_type));
                        init_err = true;
                    }
                }

                if (!init_err) {
                    // find the constructor
                    Position const &pos = init != NULL ?
                        init->access_position() : v_name->access_position();

                    init = find_init_constructor(
                        var_type, m_module.clone_name(t_name, /*modifier=*/NULL), init, pos);
                    var_decl->set_variable_init(i, init);
                }
            }
        }

        if (IAnnotation_block const *anno = var_decl->get_annotations(i)) {
            Definition_store store(m_annotated_def, v_def);
            visit(anno);
        }
    }

    // don't visit children anymore
    return false;
}

// start of compound statement
bool NT_analysis::pre_visit(IStatement_compound *block) {
    m_def_tab->enter_scope(NULL);
    return true;
}

// end of compound statement
void NT_analysis::post_visit(IStatement_compound *block)
{
    Scope *scope = m_def_tab->get_curr_scope();
    m_def_tab->leave_scope();

    if (scope->is_empty()) {
        // drop it to save some space
        m_def_tab->remove_empty_scope(scope);
    }
}

// declaration statement is handled automatically
// expression statement is handled automatically

// end of an if statement
void NT_analysis::post_visit(IStatement_if *if_stmt)
{
    IExpression const *cond = if_stmt->get_condition();
    bool has_error = false;
    if (IExpression *conv = check_bool_condition(cond, has_error, /*inside_enable_if=*/false)) {
        if_stmt->set_condition(conv);
    }
}

// start of a switch statement
bool NT_analysis::pre_visit(IStatement_switch *switch_stmt)
{
    IExpression const *control_expr = switch_stmt->get_condition();
    IExpression const *n_control_expr = visit(control_expr);
    if (n_control_expr != control_expr) {
        switch_stmt->set_condition(n_control_expr);
        control_expr = n_control_expr;
    }

    // check control type first
    IType const *control_type = control_expr->get_type();
    control_type = control_type->skip_type_alias();
    if (!is<IType_error>(control_type) && !is_integer_or_enum_type(control_type)) {
        if (IExpression const *conv = convert_to_type_implicit(control_expr, m_tc.int_type)) {
            switch_stmt->set_condition(conv);
            control_expr = conv;
            control_type = m_tc.int_type;
        } else {
            error(
                ILLEGAL_SWITCH_CONTROL_TYPE,
                control_expr->access_position(),
                Error_params(*this).add(control_type));
            control_type = m_tc.error_type;
        }
    }

    bool has_case = false;
    IStatement_case const *first_default = NULL;

    typedef ptr_hash_map<IValue const, IStatement_case const *>::Type Case_map;
    Case_map case_map(0, Case_map::hasher(), Case_map::key_equal(), m_module.get_allocator());

    for (size_t i = 0, n = switch_stmt->get_case_count(); i < n; ++i) {
        IStatement const *st = switch_stmt->get_case(i);

        switch (st->get_kind()) {
        case IStatement::SK_INVALID:
            // syntax error, ignore
            continue;
        case IStatement::SK_CASE: {
            IStatement_case const *case_stmt = cast<IStatement_case>(st);
            IExpression const *label_expr = case_stmt->get_label();

            // visit children
            if (label_expr != NULL) {
                IExpression const *n_label_expr = visit(label_expr);
                if (n_label_expr != label_expr) {
                    const_cast<IStatement_case *>(case_stmt)->set_label(n_label_expr);
                    label_expr = n_label_expr;
                }
            }
            for (size_t i = 0, n = case_stmt->get_statement_count(); i < n; ++i) {
                visit(case_stmt->get_statement(i));
            }

            if (label_expr != NULL) {
                // real case
                has_case = true;

                bool has_error = false;

                IType const *label_type = label_expr->get_type();
                label_type = label_type->skip_type_alias();
                if (is<IType_error>(control_type)) {
                    // do not check the label type
                } else if (is<IType_error>(label_type)) {
                    has_error = true;
                } else if (!is_integer_or_enum_type(label_type)) {
                    has_error = true;
                    if (is_integer_type(control_type)) {
                        // check if we can convert implicit to the control type
                        IExpression const *conv
                            = convert_to_type_implicit(label_expr, control_type);
                        if (conv != NULL) {
                            const_cast<IStatement_case *>(case_stmt)->set_label(conv);
                            label_expr = conv;
                            label_type = m_tc.int_type;
                            has_error = false;
                        }
                    }
                    if (has_error) {
                        error(
                            ILLEGAL_CASE_LABEL_TYPE,
                            control_expr->access_position(),
                            Error_params(*this).add(control_type));
                    }
                } else {
                    // type is ok, check against control type
                    if (control_type != label_type && !is<IType_error>(control_type)) {
                        // cannot convert to control type
                        error(
                            CANNOT_CONVERT_CASE_LABEL_TYPE,
                            label_expr->access_position(),
                            Error_params(*this).add(control_type));
                        has_error = true;
                    }
                }
                if (!has_error) {
                    bool is_invalid = false;
                    if (!is_const_expression(label_expr, is_invalid)) {
                        if (! is_invalid) {
                            error(
                                CASE_LABEL_NOT_CONSTANT,
                                label_expr->access_position(),
                                Error_params(*this));
                        }
                        has_error = true;
                    } else {
                        // fold it
                        IValue const *val  = label_expr->fold(
                            &m_module, m_module.get_value_factory(), NULL);
                        IValue const *ival =
                            val->convert(m_module.get_value_factory(), m_tc.int_type);

                        MDL_ASSERT(
                            !is<IValue_bad>(ival) &&
                            "const folding failed for constant expression");

                        Case_map::const_iterator it = case_map.find(ival);
                        if (it == case_map.end()) {
                            // good, new case
                            case_map[ival] = case_stmt;
                        } else {
                            // bad, already used
                            error(
                                MULTIPLE_CASE_VALUE,
                                case_stmt->access_position(),
                                Error_params(*this).add(val));
                            IStatement_case const *prev = it->second;
                            add_note(
                                PREVIOUSLY_USED,
                                prev->access_position(),
                                Error_params(*this));
                        }

                        if (!is<IExpression_literal>(label_expr)) {
                            Position const *pos = &label_expr->access_position();
                            label_expr = m_module.create_literal(val, pos);
                        }
                    }
                    const_cast<IStatement_case *>(case_stmt)->set_label(label_expr);
                }
            } else {
                // a default case
                if (first_default != NULL) {
                    error(
                        MULTIPLE_DEFAULT_LABEL,
                        case_stmt->access_position(),
                        Error_params(*this));
                    add_note(
                        FIRST_DEFAULT,
                        first_default->access_position(),
                        Error_params(*this));
                } else {
                    first_default = case_stmt;
                }
            }
            break;
        }
        default:
            MDL_ASSERT(!"Unsupported child of a switch expression");
            break;
        }
    }

    if (first_default != NULL) {
        if (!has_case) {
            warning(DEFAULT_WITHOUT_CASE, switch_stmt->access_position(), Error_params(*this));
        }
    } else {
        // no default:
        if (IType_enum const *e_type = as<IType_enum>(control_type)) {
            Value_factory *value_fact = m_module.get_value_factory();

            for (int idx = 0, n = e_type->get_value_count(); idx < n; ++idx) {
                ISymbol const *sym;
                int           code;

                e_type->get_value(idx, sym, code);

                // must be handled as int here, otherwise two enum values with equal
                // codes will be threated as two different values
                IValue const *v = value_fact->create_int(code);

                if (case_map.find(v) == case_map.end()) {
                    // found an unhandled case
                    warning(
                        ENUM_VALUE_NOT_HANDLED_IN_SWITCH,
                        switch_stmt->access_position(),
                        Error_params(*this).add(sym));
                }
            }
        }
    }

    // don't visit children anymore
    return false;
}

// end of a while statement
void NT_analysis::post_visit(IStatement_while *while_stmt)
{
    bool has_error = false;
    if (IExpression const *cond = while_stmt->get_condition()) {
        if (IExpression *conv = check_bool_condition(cond, has_error, /*inside_enable_if=*/false)) {
            while_stmt->set_condition(conv);
        }
    }
}

// end of a do-while statement
void NT_analysis::post_visit(IStatement_do_while *do_while_stmt)
{
    bool has_error = false;
    if (IExpression const *cond = do_while_stmt->get_condition()) {
        if (IExpression *conv = check_bool_condition(cond, has_error, /*inside_enable_if=*/false)) {
            do_while_stmt->set_condition(conv);
        }
    }
}

// start of a for expression
bool NT_analysis::pre_visit(IStatement_for *for_stmt)
{
    IStatement const *init = for_stmt->get_init();
    if (init && is<IStatement_declaration>(init)) {
        // this for expression creates a scope
        m_def_tab->enter_scope(NULL);
    }
    return true;
}

// end of a for expression
void NT_analysis::post_visit(IStatement_for *for_stmt)
{
    IStatement const *init = for_stmt->get_init();
    if (init && is<IStatement_declaration>(init)) {
        // this for expression creates a scope
        m_def_tab->leave_scope();
    }

    bool has_error = false;
    if (IExpression const *cond = for_stmt->get_condition()) {
        if (IExpression *conv = check_bool_condition(cond, has_error, /*inside_enable_if=*/false)) {
            for_stmt->set_condition(conv);
        }
    }
}

// end of a return statement
void NT_analysis::post_visit(IStatement_return *ret_stmt)
{
    IExpression const *expr      = ret_stmt->get_expression();
    IType const       *expr_type = expr->get_type();

    if (is<IType_error>(expr_type))
        return;

    // check the return type and add conversions if necessary

    Definition const *func_def = tos_function();
    IType const      *ret_type = get_result_type(func_def)->skip_type_alias();

    if (is<IType_error>(ret_type))
        return;

    bool has_error          = true;
    bool new_bound          = false;
    bool need_explicit_conv = false;
    if (can_assign_param(ret_type, expr_type, new_bound, need_explicit_conv)) {
        has_error = false;
        if (need_type_conversion(ret_type, expr_type)) {
            if (is<IType_array>(ret_type->skip_type_alias())) {
                // cannot convert arrays
                has_error = true;
            } else if (IExpression const *conv = convert_to_type_implicit(expr, ret_type)) {
                ret_stmt->set_expression(conv);
            } else {
                // no conversion possible
                has_error = true;
            }
        }
    }
    if (has_error) {
        if (is<IType_function>(expr_type->skip_type_alias())) {
            error(
                FUNCTION_PTR_USED,
                expr->access_position(),
                Error_params(*this));
        } else {
            error(
                INVALID_CONVERSION_RETURN,
                expr->access_position(),
                Error_params(*this).add(expr_type).add(ret_type));
        }
    }
}

// end of an invalid expression
IExpression *NT_analysis::post_visit(IExpression_invalid *inv_expr)
{
    // the type of an invalid expression is always the error type
    inv_expr->set_type(m_tc.error_type);
    return inv_expr;
}

IExpression *NT_analysis::post_visit(IExpression_literal *lit)
{
    // No need to set the type for literal ...
    if (IValue_resource const *r_val = as<IValue_resource>(lit->get_value())) {
        // ensure resource are processed if we analyse "hand-written" ASTs which
        // already use resource values instead of calls to resource constructors
        handle_resource_url(r_val, lit, r_val->get_type());
    }
    return lit;
}

// end of a reference expression
IExpression *NT_analysis::post_visit(IExpression_reference *ref)
{
    // get the definition from the type name
    IType_name const *type_name = ref->get_name();
    IQualified_name const *qual_name = type_name->get_qualified_name();
    Definition const *tdef = impl_cast<Definition>(qual_name->get_definition());

    Definition const *def = tdef;
    Definition::Kind kind = tdef->get_kind();
    if (kind == Definition::DK_TYPE) {
        // we are referencing a type/material, must be a constructor: look if there is one
        IType const *type = type_name->get_type()->skip_type_alias();

        if (type_name->is_array()) {
            ref->set_array_constructor();
        }

        if (is<IType_error>(type)) {
            def = get_error_definition();
        } else {
            if (IType_array const *a_type = as<IType_array>(type)) {
                // folded array constructor, i.e. T[size] a(args)
                type = a_type->get_element_type()->skip_type_alias();
            }

            Scope *type_scope = m_def_tab->get_type_scope(type);

            def = type_scope != NULL ?
                type_scope->find_definition_in_scope(type_scope->get_scope_name()) : NULL;
            if (def == NULL) {
                error(
                    INTERNAL_COMPILER_ERROR,
                    ref->access_position(),
                    Error_params(*this)
                        .add("Could not find constructor for type '")
                        .add(tdef->get_sym())
                        .add("'"));
                def = get_error_definition();
            }
        }
    }

    if (def->get_kind() != Definition::DK_ERROR) {
        if (def->has_flag(Definition::DEF_IS_INCOMPLETE)) {
            // was used when in incomplete state
            const_cast<Definition *>(def)->set_flag(Definition::DEF_USED_INCOMPLETE);
        }
    }

    ref->set_definition(def);
    ref->set_type(check_performance_restriction(def->get_type(), ref->access_position()));
    return ref;
}

// end of unary expression
IExpression *NT_analysis::post_visit(IExpression_unary *un_expr)
{
    IExpression_unary::Operator op = un_expr->get_operator();

    if (op == IExpression_unary::OK_CAST) {
        handle_cast_expression(un_expr);
        return un_expr;
    }

    IExpression const           *arg = un_expr->get_argument();
    IType const                 *arg_types[1];

    if (m_inside_material_constr) {
        if (is_assign_operator(IExpression::Operator(op))) {
            error(
                ASSIGNMENT_INSIDE_MATERIAL,
                un_expr->access_position(),
                Error_params(*this).add(op));
        }
    }

    arg_types[0] = arg->get_type();
    Definition const *def = find_operator(
        un_expr,
        IExpression::Operator(op),
        arg_types,
        dimension_of(arg_types));

    if (def->has_flag(Definition::DEF_OP_LVALUE)) {
        // operand must be an lvalue
        if (! is_lvalue(arg)) {
            error(
                ILLEGAL_LVALUE_ASSIGNMENT,
                arg->access_position(),
                Error_params(*this)
                    .add(op)
                    .add(op == IExpression_unary::OK_POST_DECREMENT ||
                         op == IExpression_unary::OK_POST_INCREMENT ?
                         Error_params::DIR_LEFT : Error_params::DIR_RIGHT));
            add_let_temporary_note(arg);
        }
    }

    IType const *res_type = get_result_type(def);
    un_expr->set_type(check_performance_restriction(res_type, un_expr->access_position()));
    return un_expr;
}

// start of a binary expression
bool NT_analysis::pre_visit(IExpression_binary *bin_expr)
{
    IExpression_binary::Operator op = bin_expr->get_operator();

    switch (op) {
    case IExpression_binary::OK_SELECT:
        handle_select_scopes(bin_expr);
        // don't visit children anymore
        return false;
    default:
        return true;
    }
}

/// Check if the given opcode can throw a divide by zero exception.
static bool can_throw_divzero(IExpression_binary::Operator op)
{
    switch (op) {
    case mi::mdl::IExpression_binary::OK_DIVIDE:
    case mi::mdl::IExpression_binary::OK_MODULO:
    case mi::mdl::IExpression_binary::OK_DIVIDE_ASSIGN:
    case mi::mdl::IExpression_binary::OK_MODULO_ASSIGN:
        return true;
    default:
        return false;
    }
}

// end of a binary expression
IExpression *NT_analysis::post_visit(IExpression_binary *bin_expr)
{
    IExpression_binary::Operator op   = bin_expr->get_operator();
    IExpression const            *lhs = bin_expr->get_left_argument();
    IExpression const            *rhs = bin_expr->get_right_argument();

    IType const                  *res_type = NULL;
    IType const                  *arg_types[2];

    if (m_inside_material_constr) {
        if (is_assign_operator(IExpression::Operator(op))) {
            error(
                ASSIGNMENT_INSIDE_MATERIAL,
                bin_expr->access_position(),
                Error_params(*this).add(op));
        }
    }

    switch (op) {
    case IExpression_binary::OK_SELECT:
        // the type of the select expression is the type of its rhs
        res_type = rhs->get_type();
        break;
    case IExpression_binary::OK_ARRAY_INDEX:
        {
            IType const *lhs_type = lhs->get_type();

            // matrices and vectors can be treated as an array
            IType::Modifiers mod_cu = lhs_type->get_type_modifiers();
            lhs_type = lhs_type->skip_type_alias();

            int arr_size = -1;
            switch (lhs_type->get_kind()) {
            case IType::TK_VECTOR:
            {
                IType_vector const *t = cast<IType_vector>(lhs_type);
                res_type = t->get_element_type();
                arr_size = t->get_size();
                break;
            }
            case IType::TK_MATRIX:
            {
                IType_matrix const *t = cast<IType_matrix>(lhs_type);
                res_type = t->get_element_type();
                arr_size = t->get_columns() * t->get_element_type()->get_size();
                break;
            }
            case IType::TK_ARRAY:
            {
                IType_array const *t = cast<IType_array>(lhs_type);
                res_type = t->get_element_type();
                if (t->is_immediate_sized())
                    arr_size = t->get_size();
                break;
            }
            case IType::TK_INCOMPLETE:
                // suppress error message, it will be reported at occurrences
                res_type = m_tc.error_type;
                break;
            case IType::TK_ERROR:
                // suppress error message
                res_type = m_tc.error_type;
                break;
            default:
                error(
                    INDEX_OF_NON_ARRAY,
                    bin_expr->access_position(),
                    Error_params(*this).add(lhs_type));
                res_type = m_tc.error_type;
                break;
            }

            if (!is<IType_error>(res_type)) {
                // const/uniform modifier do not get lost on an index expression
                res_type = m_tc.decorate_type(res_type, mod_cu);
            }

            // check indices
            IType const *index_type = rhs->get_type();

            if (is_integer_or_enum_type(index_type)) {
                // we can also catch an out of range reference if the index can be folded
                bool is_invalid = false;
                if (!m_in_expected_const_expr && is_const_expression(rhs, is_invalid)) {
                    IValue const *val = rhs->fold(&m_module, m_module.get_value_factory(), NULL);
                    val = val->convert(m_module.get_value_factory(), m_tc.int_type);

                    if (is<IValue_int>(val)) {
                        int index = cast<IValue_int>(val)->get_value();

                        if (index < 0) {
                            warning(
                                ARRAY_INDEX_OUT_OF_RANGE,
                                rhs->access_position(),
                                Error_params(*this).add(index).add("<").add(0));
                        }
                        if (arr_size >= 0 && index >= arr_size) {
                            warning(
                                ARRAY_INDEX_OUT_OF_RANGE,
                                rhs->access_position(),
                                Error_params(*this).add(index).add(">=").add(arr_size));
                        }
                    }
                }
            } else if (! is<IType_error>(index_type)) {
                // neither int nor enum
                error(
                    ILLEGAL_INDEX_TYPE,
                    rhs->access_position(),
                    Error_params(*this).add(index_type));
            }

            // for now, set the can_thow property if an index operator is found
            m_can_throw_bounds = true;

            break;
        }
    case IExpression_binary::OK_MULTIPLY:
    case IExpression_binary::OK_DIVIDE:
    case IExpression_binary::OK_MODULO:
    case IExpression_binary::OK_PLUS:
    case IExpression_binary::OK_MINUS:
    case IExpression_binary::OK_SHIFT_LEFT:
    case IExpression_binary::OK_SHIFT_RIGHT:
    case IExpression_binary::OK_UNSIGNED_SHIFT_RIGHT:
    case IExpression_binary::OK_LESS:
    case IExpression_binary::OK_LESS_OR_EQUAL:
    case IExpression_binary::OK_GREATER_OR_EQUAL:
    case IExpression_binary::OK_GREATER:
    case IExpression_binary::OK_EQUAL:
    case IExpression_binary::OK_NOT_EQUAL:
    case IExpression_binary::OK_BITWISE_AND:
    case IExpression_binary::OK_BITWISE_XOR:
    case IExpression_binary::OK_BITWISE_OR:
    case IExpression_binary::OK_LOGICAL_AND:
    case IExpression_binary::OK_LOGICAL_OR:
    case IExpression_binary::OK_ASSIGN:
    case IExpression_binary::OK_MULTIPLY_ASSIGN:
    case IExpression_binary::OK_DIVIDE_ASSIGN:
    case IExpression_binary::OK_MODULO_ASSIGN:
    case IExpression_binary::OK_PLUS_ASSIGN:
    case IExpression_binary::OK_MINUS_ASSIGN:
    case IExpression_binary::OK_SHIFT_LEFT_ASSIGN:
    case IExpression_binary::OK_SHIFT_RIGHT_ASSIGN:
    case IExpression_binary::OK_UNSIGNED_SHIFT_RIGHT_ASSIGN:
    case IExpression_binary::OK_BITWISE_AND_ASSIGN:
    case IExpression_binary::OK_BITWISE_XOR_ASSIGN:
    case IExpression_binary::OK_BITWISE_OR_ASSIGN:
        {
            arg_types[0] = lhs->get_type();
            arg_types[1] = rhs->get_type();

            Definition const *def = NULL;
            if (m_has_array_assignment && op == IExpression_binary::OK_ASSIGN) {
                // check, if it is an array assignment
                def = find_array_assignment(bin_expr, arg_types[0], arg_types[1]);
            }

            if (def == NULL) {
                def = find_operator(
                    bin_expr, IExpression::Operator(op), arg_types, dimension_of(arg_types));
            }

            if (def->has_flag(Definition::DEF_OP_LVALUE)) {
                // left operand must be an lvalue
                if (! is_lvalue(lhs)) {
                    error(
                        ILLEGAL_LVALUE_ASSIGNMENT,
                        lhs->access_position(),
                        Error_params(*this)
                            .add(op)
                            .add(Error_params::DIR_LEFT));
                    add_let_temporary_note(lhs);
                }
            }

            res_type = get_result_type(def);

            if (can_throw_divzero(op) && !is_error(def)) {
                // check if it is an integer division
                IType_function const *f_type = cast<IType_function>(def->get_type());
                MDL_ASSERT(f_type->get_parameter_count() == 2);
                IType const *r_type;
                ISymbol const *dummy;
                f_type->get_parameter(1, r_type, dummy);

                if (is_integer_or_integer_vector_type(r_type)) {
                    // check if the right parameter is NOT a non-zero literal
                    bool can_trow = true;
                    IValue const *v = rhs->fold(&m_module, m_module.get_value_factory(), NULL);
                    if (!is<IValue_bad>(v)) {
                        if (IValue_vector const *vv = as<IValue_vector>(v)) {
                            // check every component
                            can_trow = false;
                            for (int i = 0, n = vv->get_component_count(); i < n; ++i) {
                                IValue const *comp = vv->get_value(i);
                                if (comp->is_zero()) {
                                    can_trow = true;
                                    break;
                                }
                            }
                        } else {
                            can_trow = v->is_zero();
                        }
                        if (can_trow) {
                            // we know here is a division by zero
                            warning(
                                DIVISION_BY_ZERO,
                                bin_expr->access_position(),
                                Error_params(*this));

                        }
                    }
                    m_can_throw_divzero |= can_trow;
                }
            }
        }
        break;
    case IExpression_binary::OK_SEQUENCE:
        // the type of a sequence expression is the type of the last one
        res_type = rhs->get_type();
        break;
    }

    bin_expr->set_type(check_performance_restriction(res_type, bin_expr->access_position()));

    if (!is<IType_error>(res_type) && op == IExpression_binary::OK_ASSIGN) {
        // a valid assignment
        IExpression_binary const *sel = as<IExpression_binary>(bin_expr->get_left_argument());
        if (sel != NULL && sel->get_operator() == IExpression_binary::OK_SELECT) {
            // is a a.b = c assignment, check for valid range
            IExpression_reference const *ref =
                as<IExpression_reference>(sel->get_right_argument());
            if (ref != NULL) {
                IDefinition const *ref_def = ref->get_definition();
                if (!is_error(ref_def)) {
                    MDL_ASSERT(ref_def->get_kind() == IDefinition::DK_MEMBER);

                    check_field_assignment_range(ref_def, bin_expr->get_right_argument());
                }
            }
        }
    }
    return bin_expr;
}

// end of a conditional expression
IExpression *NT_analysis::post_visit(IExpression_conditional *cond_expr)
{
    IExpression const *cond      = cond_expr->get_condition();
    IType const       *cond_type = cond->get_type();
    bool              has_error  = false;

    if (is<IType_error>(cond_type)) {
        has_error = true;
    } else {
        if (IExpression *conv = check_bool_condition(cond, has_error, /*inside_enable_if=*/false))
        {
            cond_expr->set_condition(conv);
        }
    }

    IExpression const *true_expr  = cond_expr->get_true();
    IExpression const *false_expr = cond_expr->get_false();
    IType const       *true_type  = true_expr->get_type()->skip_type_alias();
    IType const       *false_type = false_expr->get_type()->skip_type_alias();

    if (is<IType_error>(true_type) || is<IType_error>(false_type)) {
        has_error = true;
    } else {
        if (true_type != false_type) {
            // If the "then" and "else" types do not match but there
            // is an implicit conversion between the two, insert
            // an explicit constructor so that the types match, and
            // assign that type to the conditional expression.
            if (!is<IType_array>(true_type) && !is<IType_array>(false_type)) {
                if (IExpression const *conv_f =
                    convert_to_type_implicit(true_expr, false_type)) {
                    true_type = false_type;
                    cond_expr->set_true(conv_f);
                    true_expr = conv_f;
                } else if (IExpression const *conv_t =
                    convert_to_type_implicit(false_expr, true_type)) {
                    false_type = true_type;
                    cond_expr->set_false(conv_t);
                    false_expr = conv_t;
                }
            }
            if (true_type != false_type) {
                if (!is<IType_error>(true_type) && !is<IType_error>(false_type)) {
                    // expression types cannot be combined
                    error(
                        MISMATCHED_CONDITIONAL_EXPR_TYPES,
                        cond_expr->access_position(),
                        Error_params(*this));
                }
                has_error = true;
            }
        }
    }

    cond_expr->set_type(
        has_error ?
            m_tc.error_type :
            check_performance_restriction(true_type, cond_expr->access_position()));
    return cond_expr;
}

// start of a call
bool NT_analysis::pre_visit(IExpression_call *call_expr)
{
    IExpression_binary const *callee = as<IExpression_binary>(call_expr->get_reference());
    if (callee == NULL) {
        return true;
    }

    if (callee->get_operator() != IExpression_binary::OK_ARRAY_INDEX) {
        return true;
    }

    IExpression_reference const *lhs = as<IExpression_reference>(callee->get_left_argument());
    if (lhs == NULL) {
        return true;
    }

    if (lhs->is_array_constructor()) {
        return true;
    }

    // x[y](...) assume this is an array constructor
    IType_name *tn = const_cast<IType_name *>(lhs->get_name());
    tn->set_array_size(callee->get_right_argument());

    Position &pos = tn->access_position();
    IExpression_reference *ref = m_module.get_expression_factory()->create_reference(tn, POS(pos));

    call_expr->set_reference(ref);

    {
        // ref should be a callable
        Store<Match_restriction> restrict_callable(m_curr_restriction, MR_CALLABLE);
        (void)visit(ref);
    }

    for (size_t i = 0, n = call_expr->get_argument_count(); i < n; ++i) {
        IArgument const *arg = call_expr->get_argument(i);
        visit(arg);
    }

    // do not visit children anymore
    return false;
}

// end of a call
IExpression *NT_analysis::post_visit(IExpression_call *call_expr)
{
    IExpression const *callee = call_expr->get_reference();

    if (is<IExpression_invalid>(callee)) {
        // syntax error
        call_expr->set_type(m_tc.error_type);
        return call_expr;
    }

    if (!is<IExpression_reference>(callee)) {
        error(
            CALLED_OBJECT_NOT_A_FUNCTION,
            callee->access_position(),
            Error_params(*this)
                .add("")
                .add("")
                .add(""));
        call_expr->set_type(m_tc.error_type);
        return call_expr;
    }

    IExpression_reference const *callee_ref = cast<IExpression_reference>(callee);

    if (callee_ref->in_parenthesis()) {
        warning(
            EXTRA_PARENTHESIS_AROUND_FUNCTION_NAME,
            callee_ref->access_position(),
            Error_params(*this).add(callee_ref));
    }

    bool bound_to_scope = false;
    IType_name const *type_name = callee_ref->get_name();
    if (type_name->is_absolute()) {
        bound_to_scope = true;
    } else {
        IQualified_name const *qname = type_name->get_qualified_name();
        if (qname->get_component_count() > 1) {
            bound_to_scope = true;
        }
    }

    Definition const *initial_def = impl_cast<Definition>(callee_ref->get_definition());
    if (initial_def->get_kind() == Definition::DK_CONSTRUCTOR) {
        // if we identified a constructor, we are always bound to scope
        bound_to_scope = true;
    }

    if (callee_ref->is_array_constructor()) {
        if (handle_array_copy_constructor(call_expr)) {
            // not a real call, but a collection of calls
            const_cast<IExpression_reference *>(callee_ref)->set_definition(NULL);
        } else if (handle_array_constructor(initial_def, call_expr)) {
            IType const *res_type = get_result_type(initial_def);

            if (!is<IType_error>(res_type)) {
                bool is_invalid = false;

                // the result of an array constructor is a array, create it here
                int n_args = call_expr->get_argument_count();

                if (IExpression const *arr_size = type_name->get_array_size()) {
                    IDefinition const *array_size_def = NULL;
                    if (!is_const_array_size(arr_size, array_size_def, is_invalid)) {
                        // must be a const expression
                        if (!is_invalid) {
                            error(
                                ARRAY_SIZE_NON_CONST,
                                arr_size->access_position(),
                                Error_params(*this));
                            is_invalid = true;
                        }
                    } else if (array_size_def != NULL) {
                        // an abstract array
                        if (n_args > 0) {
                            error(
                                ABSTRACT_ARRAY_CONSTRUCTOR_ARGUMENT,
                                call_expr->access_position(),
                                Error_params(*this));
                            is_invalid = true;
                        } else {
                            IType_array_size const *size = get_array_size(array_size_def, NULL);
                            res_type = m_tc.create_array(res_type, size);
                        }
                    } else {
                        // fold it
                        m_exc_handler.clear_error_state();
                        IValue const *val = arr_size->fold(
                            &m_module, m_module.get_value_factory(), &m_exc_handler);

                        // convert to int
                        val = val->convert(m_module.get_value_factory(), m_tc.int_type);
                        if (IValue_int const *iv = as<IValue_int>(val)) {
                            int size = iv->get_value();

                            if (!is<IExpression_literal>(arr_size)) {
                                Position const *pos = &arr_size->access_position();
                                arr_size = m_module.create_literal(val, pos);
                                const_cast<IType_name *>(type_name)->set_array_size(arr_size);
                            }

                            // FIXME: do we allow arrays of size 0?
                            if (size < 0) {
                                error(
                                    ARRAY_SIZE_NEGATIVE,
                                    arr_size->access_position(),
                                    Error_params(*this));
                                is_invalid = true;
                            } else {
                                if (n_args > 0 && size != n_args) {
                                    error(
                                        ARRAY_CONSTRUCTOR_ARGUMENT_MISMATCH,
                                        call_expr->access_position(),
                                        Error_params(*this)
                                            .add(res_type)
                                            .add(size)
                                            .add(size));
                                    is_invalid = true;
                                } else {
                                    // all fine: create the array type
                                    res_type = m_tc.create_array(res_type, size);
                                }
                            }
                        } else if (m_exc_handler.has_error()) {
                            is_invalid = true;
                        } else {
                            MDL_ASSERT(!"const folding failed");
                            is_invalid = true;
                        }
                    }
                    if (is_invalid) {
                        res_type = m_tc.error_type;
                    }
                } else {
                    // simplest case: a T[]() constructor
                    res_type = m_tc.create_array(res_type, n_args);
                }
            }

            call_expr->set_type(
                check_performance_restriction(res_type, call_expr->access_position()));
        } else {
            // error occurred
            call_expr->set_type(m_tc.error_type);
        }
        // not a real call, but a collection of calls
        const_cast<IExpression_reference *>(callee_ref)->set_definition(NULL);
    } else {
        // not an array constructor: handle overloads
        Definition const *def = find_overload(initial_def, call_expr, bound_to_scope);

        const_cast<IExpression_reference *>(callee_ref)->set_definition(def);

        IType const *res_type = get_result_type(def);

        // the result type might be bound here
        res_type = get_bound_type(res_type->skip_type_alias());
        call_expr->set_type(
            check_performance_restriction(res_type, call_expr->access_position()));

        update_call_graph(def);

        if (m_inside_param_initializer && !is_error(def)) {
            const_cast<Definition *>(def)->set_flag(Definition::DEF_REF_BY_DEFAULT_INIT);
        }

        if (def->get_kind() == IDefinition::DK_CONSTRUCTOR &&
            is<IType_resource>(res_type) &&
            def->get_semantics() != IDefinition::DS_COPY_CONSTRUCTOR)
        {
            return handle_resource_constructor(call_expr);
        }

    }
    return call_expr;
}

// start of let expression
bool NT_analysis::pre_visit(IExpression_let *let_expr)
{
    {
        // a let expression creates a scope for its declarations and there uses
        // inside the expression
        Definition_table::Scope_enter scope(*m_def_tab);
        Flag_store                    inside_let_decl(m_inside_let_decls, true);

        for (size_t i = 0, n = let_expr->get_declaration_count(); i < n; ++i) {
            IDeclaration const *decl = let_expr->get_declaration(i);
            visit(decl);
        }

        IExpression const *ex = let_expr->get_expression();
        IExpression const *n_ex = visit(ex);
        if (n_ex != ex) {
            let_expr->set_expression(n_ex);
        }
    }

    // The type of a let expression is the type of its sub-expression
    IExpression const *sub_expr = let_expr->get_expression();

    // no performance type check needed here
    let_expr->set_type(sub_expr->get_type());

    if (!m_inside_material_constr && !m_allow_let_expression) {
        error(
            LET_USED_OUTSIDE_MATERIAL,
            let_expr->access_position(),
            Error_params(*this));
    }
    // do not visit children anymore
    return false;
}

// Handle qualified names
void NT_analysis::post_visit(IQualified_name *qual_name)
{
    if (m_inside_material_defaults) {
        // some references are already set, especially state::normal,
        // so do not lookup it again, as it might be fail
        IDefinition const *def = qual_name->get_definition();
        if (def != NULL) {
            return;
        }
    }

    // ignore definitions of deferred array sizes if enabled
    Definition *def = find_definition_for_qualified_name(
        qual_name, m_in_array_size && m_ignore_deferred_size);
    qual_name->set_definition(def);
}

// Note that the name type_name is misleading here, because it is used by reference expressions.

bool NT_analysis::pre_visit(IType_name *type_name)
{
    IQualified_name const *qual_name = type_name->get_qualified_name();
    visit(qual_name);

    if (IExpression const *size = type_name->get_array_size()) {
        Flag_store expexted_const_expr(m_in_expected_const_expr, true);
        Flag_store in_array_size(m_in_array_size, true);

        IExpression const *n_size = visit(size);
        if (n_size != size) {
            type_name->set_array_size(n_size);
        }
    }

    Definition const *def = impl_cast<Definition>(qual_name->get_definition());

    IType const *type = def->get_type();
    if (is<IType_error>(type)) {
        // error definitions could be types, so set one
        type_name->set_type(type);
    } else if (def->get_kind() == Definition::DK_TYPE) {
        // really describes a type, handle qualifier
        Qualifier qual = type_name->get_qualifier();
        type = qualify_type(qual, type);

        // check for arrays
        if (type_name->is_array()) {
            bool is_invalid = false;

            if (!is_allowed_array_type(type)) {
                report_array_type_error(type, type_name->access_position());
                is_invalid = true;
            }

            if (type_name->is_concrete_array()) {
                IExpression const *arr_size = type_name->get_array_size();

                if (arr_size == NULL) {
                    // incomplete array spec, T[], handled later
                    // for now just let the element type ...
                } else if (is<IExpression_invalid>(arr_size)) {
                    // parse error, replace by error type
                    type = m_tc.error_type;
                } else {
                    // fold the array size
                    IType const *size_tp = arr_size->get_type();
                    if (!is_integer_or_enum_type(size_tp)) {
                        // must be int or enum type
                        is_invalid = true;
                        if (!is<IType_error>(size_tp)) {
                            error(
                                ARRAY_SIZE_NOT_INTEGRAL,
                                arr_size->access_position(),
                                Error_params(*this).add(size_tp));
                        }
                    }

                    IDefinition const *array_size_def = NULL;
                    if (is_invalid || !is_const_array_size(arr_size, array_size_def, is_invalid)) {
                        // must be a const expression
                        if (! is_invalid) {
                            error(
                                ARRAY_SIZE_NON_CONST,
                                arr_size->access_position(),
                                Error_params(*this));
                            is_invalid = true;
                        }
                    } else if (array_size_def != NULL) {
                        // an abstract array
                        IType_array_size const *size = get_array_size(array_size_def, NULL);
                        type = m_tc.create_array(type, size);
                    } else {
                        m_exc_handler.clear_error_state();
                        IValue const *val = arr_size->fold(
                            &m_module, m_module.get_value_factory(), &m_exc_handler);

                        // convert to int
                        val = val->convert(m_module.get_value_factory(), m_tc.int_type);
                        if (IValue_int const *iv = as<IValue_int>(val)) {
                            int size = iv->get_value();

                            if (!is<IExpression_literal>(arr_size)) {
                                Position const *pos = &arr_size->access_position();
                                arr_size = m_module.create_literal(val, pos);
                                type_name->set_array_size(arr_size);
                            }

                            // Note: we allow arrays of size 0 in MDL (aka "the void value")
                            if (size < 0) {
                                error(
                                    ARRAY_SIZE_NEGATIVE,
                                    arr_size->access_position(),
                                    Error_params(*this));
                                is_invalid = true;
                            } else {
                                // all fine: create the array type
                                type = m_tc.create_array(type, size);
                            }
                        } else if (m_exc_handler.has_error()) {
                            is_invalid = true;
                        } else {
                            MDL_ASSERT(!"const folding failed");
                            is_invalid = true;
                        }
                    }
                }
            } else {
                ISimple_name const *size_name = type_name->get_size_name();
                ISymbol const      *size_sym  = size_name->get_symbol();

                // enter definition for this symbol
                Definition *def = get_definition_at_scope(size_sym);
                if (def != NULL) {
                    err_redeclaration(
                        Definition::DK_ARRAY_SIZE,
                        def,
                        size_name->access_position(),
                        ENT_REDECLARATION);
                    // without the size symbol we cannot do much here, so use the error type
                    type = m_tc.error_type;
                } else {
                    def = m_def_tab->enter_definition(
                        Definition::DK_ARRAY_SIZE, size_sym,
                        m_tc.int_type, &size_name->access_position());

                    // create a fully qualified name for this symbol
                    string fq_name(get_current_scope_name());
                    if (!fq_name.empty()) {
                        fq_name += "::";
                    }
                    fq_name += size_sym->get_name();

                    ISymbol const *fq_sym =
                        m_module.get_symbol_table().get_user_type_symbol(fq_name.c_str());

                    const_cast<ISimple_name *>(size_name)->set_definition(def);

                    IType_array_size const *size = get_array_size(def, fq_sym);
                    type = m_tc.create_array(type, size);
                }
            }

            if (is_invalid) {
                type = m_tc.error_type;
            }
        }
        // set the type, so it will be recognized as a type
        type_name->set_type(type);
    }
    return false;
}

// Enter a new semantic version for given module.
Position const *NT_analysis::set_module_sem_version(
    Module         &module,
    int            major,
    int            minor,
    int            patch,
    char const     *prelease,
    Position const &pos)
{
    if (m_sema_version_pos != NULL) {
        return m_sema_version_pos;
    }
    m_sema_version_pos = &pos;
    m_module.set_semantic_version(major, minor, patch, prelease);
    return NULL;
}

/// Check if name is a valid module name.
///
/// \param name  the name to check
static bool valid_module_path(char const *name)
{
    bool dotdot_allowed = true;

    if (name[0] == '.') {
        ++name;
        if (name[0] == '.') {
            ++name;
            // started with '..'
        } else {
            // started with '.', no more . or ..
            dotdot_allowed = false;
        }
        if (name[0] == ':' && name[1] == ':') {
            name += 2;
        } else {
            // missing scope operator
            return false;
        }
    } else {
        // other begin, no '..'
        dotdot_allowed = false;

        if (name[0] == ':' && name[1] == ':') {
            name += 2;
        }
    }

    if (dotdot_allowed) {
        // follow '..'
        for (;;) {
            if (name[0] != '.')
                break;

            if (name[1] == '.') {
                // '..' found
                name += 2;
            } else {
                // no '.' at this point allowed
                return false;
            }

            if (name[0] == ':' && name[1] == ':') {
                name += 2;
            } else {
                // missing scope operator
                return false;
            }
        }
    }

    for (;;) {
        char c = name[0];

        // note that we assume UTF-8 encoding here
        if (('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z')) {
            do {
                ++name;
                c = name[0];
            } while(
                ('a' <= c && c <= 'z') ||
                ('A' <= c && c <= 'Z') ||
                ('_' == c) ||
                ('0' <= c && c <= '9'));
        } else {
            // not an identifier
            return false;
        }

        if (c == '\0') {
            return true;
        }
        if (c != ':' || name[1] != ':') {
            // not a scope operator
            return false;
        }
        name += 2;
    }
}

static string convert_to_absolute(
    IAllocator   *alloc,
    char const   *module_name,
    Module const &module)
{
    if (module_name[0] == ':') {
        MDL_ASSERT(module_name[1] == ':');

        // already absolute
        return string(module_name, alloc);
    }

    if (module_name[0] == '.') {
        // prepare package name
        string res(module.get_name(), alloc);
        size_t pos = res.rfind("::");

        MDL_ASSERT(pos != string::npos);
        res = res.substr(0, pos);

        if (module_name[1] == '.') {
            // started with '..'
            do {
                // this should be checked earlier
                MDL_ASSERT(module_name[1] == '.');

                size_t pos = res.rfind("::");

                if (pos == string::npos) {
                    // could not go further up, error
                    res.clear();
                    return res;
                }
                res = res.substr(0, pos);

                module_name += 2;

                // this should be checked earlier
                MDL_ASSERT(module_name[0] == ':' && module_name[1] == ':');
                module_name += 2;
            } while (module_name[0] == '.');

            // ready
            res.append("::");
            res.append(module_name);
            return res;
        } else {
            // started with '.
            module_name += 1;

            // this should be checked earlier
            MDL_ASSERT(module_name[0] == ':' && module_name[1] == ':');

            // ready
            res.append(module_name);
            return res;
        }
    }

    // either relative or absolute, we cannot decide at this point
    return string(module_name, alloc);
}

/// Check if the given semantic version is lesser than the given version.
///
/// \param ver    the semantic version
/// \param major  the major version to check
/// \param minor  the minor version to check
/// \param patch  the patch version to check
static bool is_less(
    ISemantic_version const *ver,
    int                     major,
    int                     minor,
    int                     patch,
    char const              *prerel)
{
    int v_major = ver->get_major();
    if (v_major < major) {
        return true;
    }
    if (v_major > major) {
        return false;
    }

    int v_minor = ver->get_minor();
    if (v_minor < minor) {
        return true;
    }
    if (v_minor > minor) {
        return false;
    }

    int v_patch = ver->get_patch();
    if (v_patch < patch) {
        return true;
    }
    if (v_patch > patch) {
        return false;
    }

    char const *v_prerl = ver->get_prerelease();

    if (v_prerl[0] != '\0' && prerel[0] == '\0') {
        // a non-empty prerelease is always "lesser" then an empty prerelease
        return true;
    }
    if (v_prerl[0] == '\0' && prerel[0] != '\0') {
        // a non-empty prerelease is always "bigger" then an empty prerelease
        return false;
    }

    return strcmp(v_prerl, prerel) < 0;
}

// Enter a new semantic dependency for the current module.
bool NT_analysis::check_module_dependency(
    IAnnotation const *anno)
{
    IExpression const *module_name_expr = anno->get_argument(0)->get_argument_expr();
    IValue const      *module_name_val  =
        module_name_expr->fold(&m_module, m_module.get_value_factory(), NULL);
    char const        *module_name = cast<IValue_string>(module_name_val)->get_value();

    if (!valid_module_path(module_name)) {
        warning(NOT_A_MODULE_NAME,
            module_name_expr->access_position(),
            Error_params(*this).add(module_name));
        return false;
    }

    string abs_name = convert_to_absolute(get_allocator(), module_name, m_module);

    // try to find it
    Module const *dep_module = NULL;
    bool         direct      = false;

    if (!abs_name.empty() && abs_name[0] != ':') {
        // try relative first

        string res(m_module.get_name(), get_allocator());
        size_t pos = res.rfind("::");

        MDL_ASSERT(pos != string::npos);
        res = res.substr(0, pos + 2);

        res.append(abs_name);

        dep_module = m_module.find_imported_module(res.c_str(), direct);

        if (dep_module == NULL) {
            // try absolute
            abs_name = "::" + abs_name;
        } else {
            abs_name = res;
        }
    }

    if (dep_module == NULL) {
        dep_module = m_module.find_imported_module(abs_name.c_str(), direct);
    }

    if (dep_module == NULL) {
        warning(
            NON_EXISTANT_DEPENDENCY,
            anno->access_position(),
            Error_params(*this).add(module_name)
        );
        return false;
    }

    if (!direct) {
        warning(
            NON_DIRECT_DEPENDENCY,
            anno->access_position(),
            Error_params(*this).add(dep_module->get_name())
        );
    }

    IExpression const *major_expr       = anno->get_argument(1)->get_argument_expr();
    IValue const      *major_val        =
        major_expr->fold(&m_module, m_module.get_value_factory(), NULL);
    int               major             = cast<IValue_int>(major_val)->get_value();
    IExpression const *minor_expr       = anno->get_argument(2)->get_argument_expr();
    IValue const      *minor_val        =
        minor_expr->fold(&m_module, m_module.get_value_factory(), NULL);
    int               minor             = cast<IValue_int>(minor_val)->get_value();
    IExpression const *patch_expr       = anno->get_argument(3)->get_argument_expr();
    IValue const      *patch_val        =
        patch_expr->fold(&m_module, m_module.get_value_factory(), NULL);
    int               patch             = cast<IValue_int>(patch_val)->get_value();
    IExpression const *prerl_expr       = anno->get_argument(4)->get_argument_expr();
    IValue const      *prerl_val        =
        prerl_expr->fold(&m_module, m_module.get_value_factory(), NULL);
    char const        *prerl            = cast<IValue_string>(prerl_val)->get_value();


    ISemantic_version const *version = dep_module->get_semantic_version();
    if (version == NULL) {
        // the modules does not define a version at all, assume ALWAYS it is older
        warning(
            DEPENDENCY_VERSION_MISSING,
            anno->access_position(),
            Error_params(*this)
                .add(dep_module->get_name())
                .add(major)
                .add(minor)
                .add(patch)
                .add_dot_string(prerl));
        return false;
    }

    // the version of the imported module must not be lesser
    if (is_less(version, major, minor, patch, prerl)) {
        warning(
            DEPENDENCY_VERSION_REQUIRED,
            anno->access_position(),
            Error_params(*this)
            .add(dep_module->get_name())
            .add(major)
            .add(minor)
            .add(patch)
            .add_dot_string(prerl)
            .add(version->get_major())
            .add(version->get_minor())
            .add(version->get_patch())
            .add_dot_string(version->get_prerelease()));
        return false;
    }

    // the major number should match
    if (version->get_major() != major) {
        warning(
            DEPENDENCY_VERSION_MAJOR_MISMATCH,
            anno->access_position(),
            Error_params(*this)
            .add(dep_module->get_name())
            .add(major)
            .add(version->get_major())
            .add(version->get_minor())
            .add(version->get_patch())
            .add_dot_string(version->get_prerelease()));
        return false;
    }
    return true;
}

// Given a range annotation definition find the overload for the given type.
Definition const *NT_analysis::select_range_overload(
    Definition const *def,
    IType const      *type)
{
    Definition const *p = def;

    for (;;) {
        Definition const *n = p->get_next_def();
        if (n == NULL)
            break;
        p = n;
    }
    for (Definition const *curr_def = p;
        curr_def != NULL;
        curr_def = curr_def->get_prev_def())
    {
        if (curr_def->has_flag(Definition::DEF_IGNORE_OVERLOAD))
            continue;

        IType_function const *ftype = cast<IType_function>(curr_def->get_type());
        MDL_ASSERT(ftype->get_parameter_count() == 2);

        IType const   *p_type = NULL;
        ISymbol const *p_sym = NULL;
        ftype->get_parameter(0, p_type, p_sym);

        if (p_type == type) {
            // found it
            return curr_def;
        }
    }
    MDL_ASSERT(!"select_range_overload() not found");
    return def;
}

// Handle known annotations.
Definition const *NT_analysis::handle_known_annotation(
    Definition const *def,
    IAnnotation      *anno)
{
    if (m_annotated_def != NULL && is_error(m_annotated_def)) {
        // the annotated entity is already wrong, ignore
        return get_error_definition();
    }

    if (m_annotated_def == NULL) {
        if (m_is_module_annotation) {
            // check for annotations allowed on modules
            switch (def->get_semantics()) {
            case IDefinition::DS_VERSION_ANNOTATION:
                {
                    IExpression const *major_expr = anno->get_argument(0)->get_argument_expr();
                    IValue const      *major      =
                        major_expr->fold(&m_module, m_module.get_value_factory(), NULL);
                    IExpression const *minor_expr = anno->get_argument(1)->get_argument_expr();
                    IValue const      *minor      =
                        minor_expr->fold(&m_module, m_module.get_value_factory(), NULL);
                    IExpression const *patch_expr = anno->get_argument(2)->get_argument_expr();
                    IValue const      *patch      =
                        patch_expr->fold(&m_module, m_module.get_value_factory(), NULL);
                    IExpression const *prerl_expr = anno->get_argument(3)->get_argument_expr();
                    IValue const      *prerl      =
                        prerl_expr->fold(&m_module, m_module.get_value_factory(), NULL);

                    Position const *prev_pos = set_module_sem_version(
                        m_module,
                        cast<IValue_int>(major)->get_value(),
                        cast<IValue_int>(minor)->get_value(),
                        cast<IValue_int>(patch)->get_value(),
                        cast<IValue_string>(prerl)->get_value(),
                        anno->access_position());

                    if (prev_pos != NULL) {
                        warning(
                            ADDITIONAL_ANNOTATION_IGNORED,
                            anno->access_position(),
                            Error_params(*this).add(def->get_symbol()));
                        add_note(
                            PREVIOUSLY_ANNOTATED,
                            Err_location(*prev_pos),
                            Error_params(*this).add(def->get_symbol()));
                        // remove it
                        def = get_error_definition();
                    }
                }
                return def;
            case IDefinition::DS_DEPENDENCY_ANNOTATION:
                {
                    // check dependency, at this point imports are done
                    // FIXME: auto-import might add other modules
                    if (!check_module_dependency(anno)) {
                        // error already reported
                        return get_error_definition();
                    }
                }
                return def;
            case Definition::DS_THROWS_ANNOTATION:
            case Definition::DS_SINCE_ANNOTATION:
            case Definition::DS_REMOVED_ANNOTATION:
            case Definition::DS_CONST_EXPR_ANNOTATION:
            case Definition::DS_DERIVABLE_ANNOTATION:
            case Definition::DS_NATIVE_ANNOTATION:
            case Definition::DS_EXPERIMENTAL_ANNOTATION:
            case Definition::DS_LITERAL_PARAM_ANNOTATION:
            case Definition::DS_UNUSED_ANNOTATION:
            case Definition::DS_NOINLINE_ANNOTATION:
            case Definition::DS_SOFT_RANGE_ANNOTATION:
            case Definition::DS_HARD_RANGE_ANNOTATION:
            case Definition::DS_DEPRECATED_ANNOTATION:
            case Definition::DS_UI_ORDER_ANNOTATION:
            case Definition::DS_USAGE_ANNOTATION:
            case Definition::DS_ENABLE_IF_ANNOTATION:
            case Definition::DS_THUMBNAIL_ANNOTATION:
                // not allowed on modules
                warning(
                    MODULE_DOES_NOT_SUPPORT_ANNOTATION,
                    anno->access_position(),
                    Error_params(*this).add(def->get_sym()));
                break;
            case Definition::DS_ORIGIN_ANNOTATION:
                // allowed on ALL entities.
                return def;
            default:
                // do nothing
                return def;
            }
        }
        MDL_ASSERT(!"Unexpected annotation without definition");
        return get_error_definition();
    }

    if (m_is_return_annotation) {
        // check for annotations allowed on return types
        switch (def->get_semantics()) {
        case Definition::DS_DERIVABLE_ANNOTATION:
            m_annotated_def->set_flag(Definition::DEF_IS_DERIVABLE);
            // ensure the annotation is deleted from the source for later ...
            return get_error_definition();

        case Definition::DS_USAGE_ANNOTATION:
        case Definition::DS_DESCRIPTION_ANNOTATION:
        case Definition::DS_DISPLAY_NAME_ANNOTATION:
            return def;

        default:
            // not allowed on return values
            warning(
                RETURN_VALUE_DOES_NOT_SUPPORT_ANNOTATION,
                anno->access_position(),
                Error_params(*this).add(def->get_sym()));
            break;
        }
        return get_error_definition();
    }

    switch (def->get_semantics()) {
    case Definition::DS_INTRINSIC_ANNOTATION:
        if (m_annotated_def->get_kind() == Definition::DK_FUNCTION) {
            // handle the intrinsic() annotation
            string abs_name(m_module.get_name(), m_module.get_allocator());
            abs_name += "::";
            abs_name += m_annotated_def->get_sym()->get_name();

            IDefinition::Semantics sema = m_compiler->get_builtin_semantic(abs_name.c_str());

            if (sema != IDefinition::DS_UNKNOWN) {
                m_annotated_def->set_semantic(sema);

                // set additional flags depending on semantics
                switch (sema) {
                case IDefinition::DS_INTRINSIC_STATE_OBJECT_ID:
                    m_annotated_def->set_flag(Definition::DEF_USES_OBJECT_ID);
                    break;
                case IDefinition::DS_INTRINSIC_STATE_TRANSFORM:
                case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_POINT:
                case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_VECTOR:
                case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_NORMAL:
                case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_SCALE:
                    m_annotated_def->set_flag(Definition::DEF_USES_TRANSFORM);
                    break;
                case IDefinition::DS_INTRINSIC_STATE_NORMAL:
                    m_annotated_def->set_flag(Definition::DEF_USES_NORMAL);
                    break;
                case IDefinition::DS_INTRINSIC_TEX_WIDTH:
                case IDefinition::DS_INTRINSIC_TEX_HEIGHT:
                case IDefinition::DS_INTRINSIC_TEX_DEPTH:
                case IDefinition::DS_INTRINSIC_TEX_TEXTURE_ISVALID:
                    // these functions read only texture attributes
                    m_annotated_def->set_flag(Definition::DEF_READ_TEXTURE_ATTR);
                    break;
                case IDefinition::DS_INTRINSIC_TEX_LOOKUP_FLOAT:
                case IDefinition::DS_INTRINSIC_TEX_LOOKUP_FLOAT2:
                case IDefinition::DS_INTRINSIC_TEX_LOOKUP_FLOAT3:
                case IDefinition::DS_INTRINSIC_TEX_LOOKUP_FLOAT4:
                case IDefinition::DS_INTRINSIC_TEX_LOOKUP_COLOR:
                    // these functions from the tex module use textures and derivatives
                    m_annotated_def->set_flag(Definition::DEF_USES_TEXTURES);
                    m_annotated_def->set_flag(Definition::DEF_USES_DERIVATIVES);
                    break;
                case IDefinition::DS_INTRINSIC_TEX_TEXEL_FLOAT:
                case IDefinition::DS_INTRINSIC_TEX_TEXEL_FLOAT2:
                case IDefinition::DS_INTRINSIC_TEX_TEXEL_FLOAT3:
                case IDefinition::DS_INTRINSIC_TEX_TEXEL_FLOAT4:
                case IDefinition::DS_INTRINSIC_TEX_TEXEL_COLOR:
                    // these functions from the tex module use textures
                    m_annotated_def->set_flag(Definition::DEF_USES_TEXTURES);
                    break;
                case IDefinition::DS_INTRINSIC_MATH_DX:
                case IDefinition::DS_INTRINSIC_MATH_DY:
                    // these functions from the math module use derivatives
                    m_annotated_def->set_flag(Definition::DEF_USES_DERIVATIVES);
                    break;
                case IDefinition::DS_INTRINSIC_DF_LIGHT_PROFILE_ISVALID:
                case IDefinition::DS_INTRINSIC_DF_LIGHT_PROFILE_MAXIMUM:
                case IDefinition::DS_INTRINSIC_DF_LIGHT_PROFILE_POWER:
                    // these functions from df read the light profile attributes
                    m_annotated_def->set_flag(Definition::DEF_READ_LP_ATTR);
                    break;
                case IDefinition::DS_INTRINSIC_DF_BSDF_MEASUREMENT_ISVALID:
                    // these functions from df read the bsdf measurement attributes,
                    // but so far we have only one, so we reuse the LP flag
                    m_annotated_def->set_flag(Definition::DEF_READ_LP_ATTR);
                    break;
                case IDefinition::DS_INTRINSIC_SCENE_DATA_ISVALID:
                case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_INT:
                case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_INT2:
                case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_INT3:
                case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_INT4:
                case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT:
                case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT2:
                case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT3:
                case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT4:
                case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_COLOR:
                case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT:
                case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT2:
                case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT3:
                case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT4:
                case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT:
                case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT2:
                case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT3:
                case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT4:
                case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_COLOR:
                    // these functions from the scene module use scene data
                    m_annotated_def->set_flag(Definition::DEF_USES_SCENE_DATA);
                    break;
                default:
                    break;
                }
            } else {
                error(
                    INTERNAL_COMPILER_ERROR,
                    anno->access_position(),
                    Error_params(*this)
                    .add_signature(m_annotated_def)
                    .add(" is an unknown intrinsic"));
                MDL_ASSERT(!"Unknown intrinsic");
            }
        }
        // ensure the INTRINSIC annotation is deleted from the source for later ...
        def = get_error_definition();
        break;
    case Definition::DS_THROWS_ANNOTATION:
        if (m_annotated_def->get_kind() == Definition::DK_FUNCTION) {
            // handle the throws() annotation
            m_annotated_def->set_flag(Definition::DEF_CAN_THROW_BOUNDS);
        }
        // ensure the THROWS annotation is deleted from the source for later ...
        def = get_error_definition();
        break;
    case Definition::DS_NATIVE_ANNOTATION:
        if (m_annotated_def->get_kind() != Definition::DK_FUNCTION) {
            warning(
                IGNORED_ON_NON_FUNCTION,
                anno->access_position(),
                Error_params(*this)
                .add(def->get_sym())
                .add_signature(m_annotated_def));
            // ensure the NATIVE annotation is deleted from the source for later ...
            def = get_error_definition();
        } else {
            // handle the native() annotation
            m_annotated_def->set_flag(Definition::DEF_IS_NATIVE);
            // mark this function so it will not be inlined
            m_annotated_def->set_flag(Definition::DEF_NO_INLINE);
        }
        break;
    case Definition::DS_UNUSED_ANNOTATION:
        m_annotated_def->set_flag(Definition::DEF_IS_UNUSED);
        break;
    case Definition::DS_NOINLINE_ANNOTATION:
        if (m_annotated_def->get_kind() != Definition::DK_FUNCTION) {
            warning(
                IGNORED_ON_NON_FUNCTION,
                anno->access_position(),
                Error_params(*this)
                    .add(def->get_sym())
                    .add_signature(m_annotated_def));
            // ensure the NOINLINE annotation is deleted from the source for later ...
            def = get_error_definition();
        } else {
            // mark this function so it will not be inlined
            m_annotated_def->set_flag(Definition::DEF_NO_INLINE);
        }
        break;
    case Definition::DS_SINCE_ANNOTATION:
        {
            IExpression const *major_expr = anno->get_argument(0)->get_argument_expr();
            IValue const      *major      =
                major_expr->fold(&m_module, m_module.get_value_factory(), NULL);
            IExpression const *minor_expr = anno->get_argument(1)->get_argument_expr();
            IValue const      *minor      =
                minor_expr->fold(&m_module, m_module.get_value_factory(), NULL);

            if (!is<IValue_int>(major) || !is<IValue_int>(minor)) {
                error(
                    INTERNAL_COMPILER_ERROR,
                    anno->access_position(),
                    Error_params(*this)
                    .add("Cannot fold since() parameter"));
                MDL_ASSERT(!"since() failed");
            } else {
                int v_major = cast<IValue_int>(major)->get_value();
                int v_minor = cast<IValue_int>(minor)->get_value();

                replace_since_version(m_annotated_def, v_major, v_minor);
            }
            // ensure the SINCE annotation is deleted from the source for later ...
            def = get_error_definition();
        }
        break;
    case Definition::DS_REMOVED_ANNOTATION:
        {
            IExpression const *major_expr = anno->get_argument(0)->get_argument_expr();
            IValue const      *major      =
                major_expr->fold(&m_module, m_module.get_value_factory(), NULL);
            IExpression const *minor_expr = anno->get_argument(1)->get_argument_expr();
            IValue const      *minor      =
                minor_expr->fold(&m_module, m_module.get_value_factory(), NULL);

            if (!is<IValue_int>(major) || !is<IValue_int>(minor)) {
                error(
                    INTERNAL_COMPILER_ERROR,
                    anno->access_position(),
                    Error_params(*this)
                    .add("Cannot fold removed() parameter"));
                MDL_ASSERT(!"since() failed");
            } else {
                int v_major = cast<IValue_int>(major)->get_value();
                int v_minor = cast<IValue_int>(minor)->get_value();

                replace_removed_version(m_annotated_def, v_major, v_minor);
            }
            // ensure the REMOVED annotation is deleted from the source for later ...
            def = get_error_definition();
        }
        break;
    case Definition::DS_EXPERIMENTAL_ANNOTATION:
        {
            int v_major = 0;
            int v_minor = 0;

            Module::get_version(IMDL::MDL_version(IMDL::MDL_LATEST_VERSION + 1), v_major, v_minor);

            replace_since_version(m_annotated_def, v_major, v_minor);

            // ensure the experimental annotation is deleted from the source for later ...
            def = get_error_definition();
        }
        break;
    case Definition::DS_LITERAL_PARAM_ANNOTATION:
        {
            // just mark it
            m_annotated_def->set_flag(Definition::DEF_LITERAL_PARAM);

            // ensure the literal_param annotation is deleted from the source for later ...
            def = get_error_definition();
        }
        break;
    case Definition::DS_SOFT_RANGE_ANNOTATION:
    case Definition::DS_HARD_RANGE_ANNOTATION:
        {
            IExpression const *min_expr = anno->get_argument(0)->get_argument_expr();
            IExpression const *max_expr = anno->get_argument(1)->get_argument_expr();

            Definition::Kind kind = m_annotated_def->get_kind();
            if (kind != Definition::DK_PARAMETER && kind != Definition::DK_MEMBER) {
                warning(
                    ENTITY_DOES_NOT_SUPPORT_ANNOTATION,
                    anno->access_position(),
                    Error_params(*this)
                        .add_signature(m_annotated_def)
                        .add(def->get_sym()));
                def = get_error_definition();
            } else {
                // check the type
                IType const *ent_type  = m_annotated_def->get_type()->skip_type_alias();
                IType const *anno_type = min_expr->get_type()->skip_type_alias();

                if (ent_type != anno_type) {
                    // the annotation range type does not match the type of the annotated
                    // entity

                    // check if we support this type in general
                    if (is_supported_by_range_anno(ent_type, m_module.get_mdl_version())) {
                        // yes, the type can be annotated, try to convert
                        min_expr = convert_to_type_and_fold(min_expr, ent_type);
                        if (min_expr != NULL) {
                            max_expr = convert_to_type_and_fold(max_expr, ent_type);

                            const_cast<IArgument *>(anno->get_argument(0))
                                ->set_argument_expr(min_expr);
                            const_cast<IArgument *>(anno->get_argument(1))
                                ->set_argument_expr(max_expr);
                            anno_type = min_expr->get_type()->skip_type_alias();

                            def = select_range_overload(def, anno_type);
                        }
                    }
                    if (ent_type != anno_type) {
                        if (!is<IType_error>(ent_type)) {
                            warning(
                                TYPE_DOES_NOT_SUPPORT_RANGE,
                                anno->access_position(),
                                Error_params(*this)
                                .add(ent_type)
                                .add(anno_type));
                        }
                        def = get_error_definition();
                    }
                }
            }

            if (!is_error(def)) {
                // check if the range is valid
                IValue const *min_v = min_expr->fold(&m_module, m_module.get_value_factory(), NULL);
                IValue const *max_v = max_expr->fold(&m_module, m_module.get_value_factory(), NULL);
                if (element_wise_compare(min_v, max_v) & ~IValue::CR_LE) {
                    warning(
                        WRONG_ANNOTATION_RANGE_INTERVAL,
                        anno->access_position(),
                        Error_params(*this)
                        .add(min_v)
                        .add(max_v)
                        .add(def->get_sym()));
                    def = get_error_definition();
                } else {
                    // ensure that the range is represented by literals, later checks depend
                    // on this
                    if (!is<IExpression_literal>(min_expr)) {
                        Position const            *pos = &min_expr->access_position();
                        IExpression_literal const *l   = m_module.create_literal(min_v, pos);

                        const_cast<IArgument*>(anno->get_argument(0))->set_argument_expr(l);
                    }
                    if (!is<IExpression_literal>(max_expr)) {
                        Position const            *pos = &max_expr->access_position();
                        IExpression_literal const *l   = m_module.create_literal(max_v, pos);

                        const_cast<IArgument*>(anno->get_argument(1))->set_argument_expr(l);
                    }
                }
            }
        }
        break;
    case Definition::DS_HIDDEN_ANNOTATION:
        // no special processing so far
        break;
    case Definition::DS_DEPRECATED_ANNOTATION:
        if (has_deprecated_anno(m_module.get_mdl_version())) {
            m_annotated_def->set_flag(Definition::DEF_IS_DEPRECATED);

            if (anno->get_argument_count() == 1) {
                IExpression const   *msg_expr = anno->get_argument(0)->get_argument_expr();
                IValue_string const *msg = as<IValue_string>(
                    msg_expr->fold(&m_module, m_module.get_value_factory(), NULL));

                if (msg != NULL) {
                    m_module.set_deprecated_message(m_annotated_def, msg);
                }
            }
        }
        break;
    case Definition::DS_CONST_EXPR_ANNOTATION:
        if (m_annotated_def->get_kind() != Definition::DK_FUNCTION) {
            warning(
                IGNORED_ON_NON_FUNCTION,
                anno->access_position(),
                Error_params(*this)
                .add(def->get_sym())
                .add_signature(m_annotated_def));
            // ensure the const_expr annotation is deleted from the source for later ...
            def = get_error_definition();
        } else {
            // handle the const_expr() annotation
            m_annotated_def->set_flag(Definition::DEF_IS_CONST_EXPR);
        }
        break;
    case Definition::DS_VERSION_NUMBER_ANNOTATION:
    case Definition::DS_DISPLAY_NAME_ANNOTATION:
    case Definition::DS_IN_GROUP_ANNOTATION:
    case Definition::DS_DESCRIPTION_ANNOTATION:
    case Definition::DS_AUTHOR_ANNOTATION:
    case Definition::DS_CONTRIBUTOR_ANNOTATION:
    case Definition::DS_COPYRIGHT_NOTICE_ANNOTATION:
    case Definition::DS_CREATED_ANNOTATION:
    case Definition::DS_MODIFIED_ANNOTATION:
    case Definition::DS_KEYWORDS_ANNOTATION:
        // no semantic value
        break;
    case Definition::DS_VERSION_ANNOTATION:
    case Definition::DS_DEPENDENCY_ANNOTATION:
        // FIXME: unhandled so far
        break;
    case Definition::DS_USAGE_ANNOTATION:
        {
            // only allowed on material and function return types (handled above), materials, and
            // function and material parameters
            bool has_error = false;
            switch (m_annotated_def->get_kind()) {
            case Definition::DK_PARAMETER:
                // allowed
                break;
            case Definition::DK_FUNCTION:
                {
                    IType_function const *fkt_tp =
                        cast<IType_function>(m_annotated_def->get_type());
                    if (!is_material_type(fkt_tp->get_return_type())) {
                        has_error = true;
                    }
                }
                break;
            default:
                has_error = true;
                break;
            }

            if (has_error) {
                warning(
                    ENTITY_DOES_NOT_SUPPORT_ANNOTATION,
                    anno->access_position(),
                    Error_params(*this)
                    .add_signature(m_annotated_def)
                    .add(def->get_sym()));
                // ensure the usage annotation is deleted from the source for later ...
                def = get_error_definition();
            }
        }
        break;
    case Definition::DS_THUMBNAIL_ANNOTATION:
        // only allowed on materials and functions
        if (m_annotated_def->get_kind() != Definition::DK_FUNCTION) {
            warning(
                ENTITY_DOES_NOT_SUPPORT_ANNOTATION,
                anno->access_position(),
                Error_params(*this)
                .add_signature(m_annotated_def)
                .add(def->get_sym()));
            // ensure the thumbnail annotation is deleted from the source for later ...
            def = get_error_definition();
        } else {
            // check file path
            IArgument const *arg0 = anno->get_argument(0);
            IExpression_literal const *name_lit = as<IExpression_literal>(
                arg0->get_argument_expr());
            if (name_lit != NULL) {
                if (IValue_string const *sval = as<IValue_string>(name_lit->get_value())) {
                    if (char const *str = sval->get_value()) {
                        size_t len = strlen(str);
                        bool valid = false;
                        if (len > 4 && str[len - 4] == '.' && (
                                strcmp(str + len - 3, "png") == 0 ||
                                strcmp(str + len - 3, "PNG") == 0 ||
                                strcmp(str + len - 3, "jpg") == 0 ||
                                strcmp(str + len - 3, "JPG") == 0))
                            valid = true;
                        else if (len > 5 && str[len - 5] == '.' && (
                                strcmp(str + len - 4, "jpeg") == 0 ||
                                strcmp(str + len - 4, "JPEG") == 0))
                            valid = true;
                        if (!valid) {
                            warning(
                                INVALID_THUMBNAIL_EXTENSION,
                                name_lit->access_position(),
                                Error_params(*this));
                            // ensure the thumbnail annotation is deleted from the source for later
                            def = get_error_definition();
                        }
                    }
                }
            } else {
                warning(
                    RESOURCE_NAME_NOT_LITERAL,
                    arg0->access_position(),
                    Error_params(*this));
                // ensure the thumbnail annotation is deleted from the source for later ...
                def = get_error_definition();
            }
        }
        break;
    case Definition::DS_UI_ORDER_ANNOTATION:
        // only allowed on parameters
        if (m_annotated_def->get_kind() != IDefinition::DK_PARAMETER) {
            warning(
                IGNORED_ON_NON_PARAMETER,
                anno->access_position(),
                Error_params(*this)
                .add(def->get_sym())
                .add_signature(m_annotated_def));
            // ensure the ui_order annotation is deleted from the source for later ...
            def = get_error_definition();
        }
        break;
    case Definition::DS_ENABLE_IF_ANNOTATION:
        // only allowed on parameters
        if (m_annotated_def->get_kind() != IDefinition::DK_PARAMETER) {
            warning(
                IGNORED_ON_NON_PARAMETER,
                anno->access_position(),
                Error_params(*this)
                    .add(def->get_sym())
                    .add_signature(m_annotated_def));
            // ensure the enable_if annotation is deleted from the source for later ...
            def = get_error_definition();
        } else {
            // parse the condition argument
            IArgument const *arg0 = anno->get_argument(0);
            if (!is<IExpression_literal>(arg0->get_argument_expr())) {
                warning(
                    CONDITION_NOT_LITERAL,
                    arg0->access_position(),
                    Error_params(*this));
                // ensure the enable_if annotation is deleted from the source for later ...
                def = get_error_definition();
            }
        }
        break;
    case Definition::DS_DERIVABLE_ANNOTATION:
        if (m_annotated_def->get_kind() == Definition::DK_PARAMETER) {
            // handle the derivable() annotation
            m_annotated_def->set_flag(Definition::DEF_IS_DERIVABLE);
        }
        // ensure the annotation is deleted from the source for later ...
        def = get_error_definition();
        break;
    case Definition::DS_ORIGIN_ANNOTATION:
        // allowed on any entity
        break;
    default:
        MDL_ASSERT(!"missing handler for known annotation");
        break;
    }
    return def;
}

// Check whether the condition expression of an enable_if annotation conforms to the
// MDL specification, i.e. uses the specified subset of MDL expressions only
// and does not reference the annotated parameter.
void NT_analysis::check_enable_if_condition(
    IExpression const *expr,
    ISymbol const     *param_sym)
{
    class Enable_if_cond_checker : public Module_visitor {
        typedef Module_visitor Base;
    public:
        // Constructor.
        Enable_if_cond_checker(
            Analysis      &ana,
            ISymbol const *param_sym)
        : Base()
        , m_ana(ana)
        , m_param_sym(param_sym)
        {
        }

        IExpression *post_visit(IExpression_binary *expr) MDL_FINAL
        {
            IExpression_binary::Operator op = expr->get_operator();

            switch (op) {
            // Valid binary operators: same as const_expr
            case IExpression_binary::OK_LOGICAL_AND:
            case IExpression_binary::OK_LOGICAL_OR:
            case IExpression_binary::OK_LESS:
            case IExpression_binary::OK_LESS_OR_EQUAL:
            case IExpression_binary::OK_GREATER_OR_EQUAL:
            case IExpression_binary::OK_GREATER:
            case IExpression_binary::OK_EQUAL:
            case IExpression_binary::OK_NOT_EQUAL:
            case IExpression_binary::OK_SELECT:
            case IExpression_binary::OK_ARRAY_INDEX:
            case IExpression_binary::OK_MULTIPLY:
            case IExpression_binary::OK_DIVIDE:
            case IExpression_binary::OK_MODULO:
            case IExpression_binary::OK_PLUS:
            case IExpression_binary::OK_MINUS:
            case IExpression_binary::OK_SHIFT_LEFT:
            case IExpression_binary::OK_SHIFT_RIGHT:
            case IExpression_binary::OK_UNSIGNED_SHIFT_RIGHT:
            case IExpression_binary::OK_BITWISE_AND:
            case IExpression_binary::OK_BITWISE_XOR:
            case IExpression_binary::OK_BITWISE_OR:
                break;

            // Invalid binary operators
            case IExpression_binary::OK_ASSIGN:
            case IExpression_binary::OK_MULTIPLY_ASSIGN:
            case IExpression_binary::OK_DIVIDE_ASSIGN:
            case IExpression_binary::OK_MODULO_ASSIGN:
            case IExpression_binary::OK_PLUS_ASSIGN:
            case IExpression_binary::OK_MINUS_ASSIGN:
            case IExpression_binary::OK_SHIFT_LEFT_ASSIGN:
            case IExpression_binary::OK_SHIFT_RIGHT_ASSIGN:
            case IExpression_binary::OK_UNSIGNED_SHIFT_RIGHT_ASSIGN:
            case IExpression_binary::OK_BITWISE_AND_ASSIGN:
            case IExpression_binary::OK_BITWISE_XOR_ASSIGN:
            case IExpression_binary::OK_BITWISE_OR_ASSIGN:
            case IExpression_binary::OK_SEQUENCE:
                m_ana.error(
                    FORBIDDEN_BINARY_OP_IN_ENABLE_IF,
                    expr->access_position(),
                    Error_params(m_ana).add(op));
                break;
            }
            return expr;
        }

        IExpression *post_visit(IExpression_unary *expr) MDL_FINAL
        {
            IExpression_unary::Operator op = expr->get_operator();
            switch (op) {
            // Valid unary operators: same as const_expr
            case IExpression_unary::OK_LOGICAL_NOT:
            case IExpression_unary::OK_BITWISE_COMPLEMENT:
            case IExpression_unary::OK_POSITIVE:
            case IExpression_unary::OK_NEGATIVE:
            case IExpression_unary::OK_CAST:
                break;

            // Invalid unary operators
            case IExpression_unary::OK_PRE_INCREMENT:
            case IExpression_unary::OK_PRE_DECREMENT:
            case IExpression_unary::OK_POST_INCREMENT:
            case IExpression_unary::OK_POST_DECREMENT:
                m_ana.error(
                    FORBIDDEN_UNARY_OP_IN_ENABLE_IF,
                    expr->access_position(),
                    Error_params(m_ana).add(op));
                break;
            }
            return expr;
        }

        IExpression *post_visit(IExpression_conditional *expr) MDL_FINAL
        {
            // The ternary operator is invalid
            m_ana.error(
                FORBIDDEN_TERNARY_OP_IN_ENABLE_IF,
                expr->access_position(),
                Error_params(m_ana));
            return expr;
        }

        bool pre_visit(IExpression_call *call) MDL_FINAL
        {
            IExpression const *callee = call->get_reference();
            if (callee->get_kind() == IExpression::EK_REFERENCE) {
                IExpression_reference const *ref =
                    static_cast<IExpression_reference const *>(callee);
                if (ref->is_array_constructor()) {
                    // visit children
                    return true;
                } else if (Definition const *def = impl_cast<Definition>(ref->get_definition())) {
                    bool is_valid = false;
                    if (def->has_flag(Definition::DEF_IS_CONST_EXPR)) {
                        // call to const_expr
                        is_valid = true;
                    } else {
                        // allow the *_isvalid() functions
                        switch (def->get_semantics()) {
                        case IDefinition::DS_INTRINSIC_TEX_TEXTURE_ISVALID:
                        case IDefinition::DS_INTRINSIC_DF_LIGHT_PROFILE_ISVALID:
                        case IDefinition::DS_INTRINSIC_DF_BSDF_MEASUREMENT_ISVALID:
                            // *_isvalid() is allowed
                        case IDefinition::DS_CONV_OPERATOR:
                        case IDefinition::DS_CONV_CONSTRUCTOR:
                            // conversion operator/constructor is allowed
                            is_valid = true;
                            break;
                        default:
                            is_valid = false;
                            break;
                        }
                    }

                    // All other calls are invalid
                    if (!is_valid && !is_error(def)) {
                        m_ana.error(
                            FORBIDDEN_CALL_IN_ENABLE_IF,
                            callee->access_position(),
                            Error_params(m_ana).add_signature(def));
                    }
                }
            } else {
                // report more errors
                IExpression const *n_callee = visit(callee);
                if (n_callee != callee) {
                    call->set_reference(n_callee);
                }
            }

            for (size_t i = 0, n = call->get_argument_count(); i < n; ++i) {
                IArgument const *arg = call->get_argument(i);
                visit(arg);
            }

            // do not visit children anymore
            return false;
        }

        IExpression *post_visit(IExpression_let *expr) MDL_FINAL
        {
            // Let expressions are invalid
            m_ana.error(
                LET_USED_OUTSIDE_MATERIAL,
                expr->access_position(),
                Error_params(m_ana));
            return expr;
        }

        IExpression *post_visit(IExpression_reference *ref) MDL_FINAL
        {
            if (ref->is_array_constructor()) {
                // syntactically wrong but no extra error here
                return ref;
            }

            IDefinition const *def = ref->get_definition();
            if (is_error(def)) {
                // already reported
                return ref;
            }

            // Must point to another parameter of the same function or material definition
            // or must be a fully qualified enum value name.
            // Call expressions also contain references, so allow constructors here.
            IDefinition::Kind kind = def->get_kind();
            if (kind == IDefinition::DK_PARAMETER) {
                if (def->get_symbol() == m_param_sym) {
                    // The condition may not reference the annotated parameter
                    m_ana.error(
                        FORBIDDEN_SELF_REF_IN_ENABLE_IF,
                        ref->access_position(),
                        Error_params(m_ana).add_signature(def));
                }
            }
            // we do not need extra checks here: Constants and enums are allowed, function
            // (addresses) are reported as errors anyhow, variables cannot exists in this scope
            return ref;
        }

    private:
        /// The analysis to issue errors.
        Analysis &m_ana;

        /// The name of the annotated parameter.
        ISymbol const *m_param_sym;
    };

    Enable_if_cond_checker checker(*this, param_sym);

    (void)checker.visit(expr);
}

// Check the given type for any possible performance restrictions and warn if necessary.
IType const *NT_analysis::check_performance_restriction(
    IType const    *type,
    Position const &pos)
{
    if (m_is_stdlib) {
        // no performance warnings in the stdlib
        return type;
    }

    IType const *t = type->skip_type_alias();

    switch (t->get_kind()) {
    case IType::TK_DOUBLE:
        warning(
            DOUBLE_TYPE_USED,
            pos,
            Error_params(*this));
        break;
    default:
        // no further restrictions so far
        break;
    }
    return type;
}

// Check if the given source type can be casted into the destination type.
IType const *NT_analysis::check_cast_conversion(
    Position const &pos,
    IType const    *src_tp,
    IType const    *dst_tp,
    bool           dst_is_incomplete,
    bool           report_error)
{
    IType const *src_type = src_tp->skip_type_alias();
    IType const *dst_type = dst_tp->skip_type_alias();

    if (is<IType_error>(dst_type))
        return dst_type;

    if (src_type == dst_type)
        return dst_tp;

    IType::Kind kind = src_type->get_kind();
    if (kind != dst_type->get_kind() && (kind != IType::TK_ARRAY || !dst_is_incomplete)) {
        if (report_error) {
            error(
                CAST_TYPES_UNRELATED,
                pos,
                Error_params(*this)
                    .add(src_tp)
                    .add(dst_tp, dst_is_incomplete)
                    .add(dst_is_incomplete ? "[]" : "")
            );
        }
        return m_tc.error_type;
    }

    bool has_error = false;

    if (kind == IType::TK_ENUM) {
        IType_enum const *s_type = cast<IType_enum>(src_type);
        IType_enum const *d_type = cast<IType_enum>(dst_type);

        typedef map<int, ISymbol const *>::Type EV_map;

        EV_map enum_values(EV_map::key_compare(), get_allocator());

        for (int i = 0, n_values = s_type->get_value_count(); i < n_values; ++i) {
            ISymbol const *sym;
            int           code;

            s_type->get_value(i, sym, code);

            enum_values.insert(EV_map::value_type(code, sym));
        }

        for (int i = 0, n_values = d_type->get_value_count(); i < n_values; ++i) {
            ISymbol const *sym;
            int           code;

            d_type->get_value(i, sym, code);

            EV_map::iterator it(enum_values.find(code));
            if (it == enum_values.end()) {
                has_error = true;

                if (report_error) {
                    // destination enum has more values ...
                    error(
                        CAST_MISSING_ENUM_VALUE_DST,
                        pos,
                        Error_params(*this)
                            .add(src_tp)
                            .add(dst_type)
                            .add(sym)
                            .add(code)
                    );
                }
            } else {
                it->second = NULL;
            }
        }

        for (EV_map::iterator it(enum_values.begin()), end(enum_values.end()); it != end; ++it) {
            if (ISymbol const *sym = it->second) {
                // found unused value
                has_error = true;

                if (report_error) {
                    int code = 0;
                    for (int i = 0, n = s_type->get_value_count(); i < n; ++i) {
                        ISymbol const *e_sym;

                        s_type->get_value(i, e_sym, code);
                        if (e_sym == sym)
                            break;
                    }

                    error(
                        CAST_MISSING_ENUM_VALUE_SRC,
                        pos,
                        Error_params(*this)
                        .add(src_tp)
                        .add(dst_type)
                        .add(sym)
                        .add(code)
                    );
                }
            }
        }

        return has_error ? m_tc.error_type : dst_tp;
    } else if (kind == IType::TK_STRUCT) {
        IType_struct const *s_type = cast<IType_struct>(src_type);
        IType_struct const *d_type = cast<IType_struct>(dst_type);

        int n_fields = s_type->get_field_count();

        if (n_fields != d_type->get_field_count()) {
            if (report_error) {
                error(
                    CAST_STRUCT_FIELD_COUNT,
                    pos,
                    Error_params(*this)
                    .add(src_tp)
                    .add(dst_tp)
                );
            }
            return m_tc.error_type;
        }

        for (int i = 0; i < n_fields; ++i) {
            IType const   *fs_type;
            ISymbol const *fs_sym;

            s_type->get_field(i, fs_type, fs_sym);

            IType const   *fd_type;
            ISymbol const *fd_sym;

            d_type->get_field(i, fd_type, fd_sym);

            if (is<IType_error>(
                check_cast_conversion(
                    pos,
                    fs_type,
                    fd_type,
                    /*dst_is_incomplete=*/false,
                    /*report_error=*/false))
            ) {
                if (report_error) {
                     // field conversion failed
                    error(
                        CAST_STRUCT_FIELD_INCOMPATIBLE,
                        pos,
                        Error_params(*this)
                        .add(src_tp)
                        .add(dst_tp)
                        .add(fs_sym)
                        .add(fd_sym)
                    );
                }
                has_error = true;
            }
        }
        return has_error ? m_tc.error_type : dst_tp;
    } else if (kind == IType::TK_ARRAY) {
        IType_array const *s_arr_type = cast<IType_array>(src_type);
        IType const       *s_el_type  = s_arr_type->get_element_type();
        IType_array const *d_arr_type = NULL;
        IType const       *d_el_type  = dst_type;

        if (!dst_is_incomplete) {
            d_arr_type    = cast<IType_array>(dst_type);
            d_el_type = d_arr_type->get_element_type();
        }

        bool s_imm = s_arr_type->is_immediate_sized();

        if (is<IType_error>(
            check_cast_conversion(
                pos,
                s_el_type,
                d_el_type,
                /*dst_is_incomplete=*/false,
                /*report_error=*/false))
        ) {
            if (report_error) {
                error(
                    CAST_ARRAY_ELEMENT_INCOMPATIBLE,
                    pos,
                    Error_params(*this)
                    .add(src_tp)
                    .add(dst_tp)
                    .add(dst_is_incomplete ? "[]" : "")
                );
            }
            has_error = true;
        } else if (dst_is_incomplete) {
            // compute destination type
            if (s_imm) {
                size_t l = s_arr_type->get_size();
                d_arr_type = cast<IType_array>(m_tc.create_array(d_el_type, l));
            } else {
                IType_array_size const *l = s_arr_type->get_deferred_size();
                d_arr_type = cast<IType_array>(m_tc.create_array(d_el_type, l));
            }
        }

        bool d_imm = d_arr_type != NULL ? d_arr_type->is_immediate_sized() : s_imm;
        if (s_imm != d_imm) {
            // deferred size must be the same
            if (d_imm) {
                if (report_error) {
                    // cannot cast from deferred size
                    error(
                        CAST_ARRAY_DEFERRED_TO_IMM,
                        pos,
                        Error_params(*this)
                        .add(src_tp)
                        .add(dst_tp)
                    );
                }
                has_error = true;
            } else {
                if (report_error) {
                    // cannot cast from deferred size
                    error(
                        CAST_ARRAY_IMM_TO_DEFERRED,
                        pos,
                        Error_params(*this)
                        .add(src_tp)
                        .add(dst_tp)
                    );
                }
                has_error = true;
            }
        } else if (!dst_is_incomplete) {
            if (s_imm) {
                // both array are immediate sized
                if (s_arr_type->get_size() != d_arr_type->get_size()) {
                    if (report_error) {
                        // different array size
                        error(
                            CAST_ARRAY_DIFFERENT_LENGTH,
                            pos,
                            Error_params(*this)
                            .add(src_tp)
                            .add(dst_tp)
                        );
                    }
                    has_error = true;
                }
            } else {
                // both array are deferred sized
                if (s_arr_type->get_deferred_size() != d_arr_type->get_deferred_size()) {
                    if (report_error) {
                        // different array size
                        error(
                            CAST_ARRAY_DIFFERENT_LENGTH,
                            pos,
                            Error_params(*this)
                            .add(src_tp)
                            .add(dst_tp)
                        );
                    }
                    has_error = true;
                }
            }
        }
        // return the newly computed type here
        return has_error ? m_tc.error_type : (IType const *)d_arr_type;
    }

    if (report_error) {
        // completely unrelated
        error(
            CAST_TYPES_UNRELATED,
            pos,
            Error_params(*this)
            .add(src_tp)
            .add(dst_tp, dst_is_incomplete)
            .add(dst_is_incomplete ? "[]" : "")
        );
    }

    return  m_tc.error_type;
}

// Handle a cast expression.
void NT_analysis::handle_cast_expression(IExpression_unary *cast_expr)
{
    IType_name const *tn            = cast_expr->get_type_name();
    IType const      *dst_type      = m_tc.error_type;
    bool              d_incomplete  = false;

    if (tn == NULL) {
        // this should not happen, wrong AST
        MDL_ASSERT(!"incomplete cast expression, missing type name");
    } else {
        dst_type = tn->get_type();

        d_incomplete = tn->is_incomplete_array();
    }

    if (!is<IType_error>(dst_type)) {
        IType const *arg_type = cast_expr->get_argument()->get_type();
        dst_type = check_cast_conversion(
            cast_expr->access_position(),
            arg_type,
            dst_type,
            d_incomplete,
            /*report_error=*/true);
    }

    cast_expr->set_type(dst_type);
}

// end of a annotation
bool NT_analysis::pre_visit(IAnnotation *anno)
{
    IQualified_name const *qname = anno->get_name();

    Definition const *def = find_definition_for_annotation_name(qname);
    const_cast<IQualified_name *>(qname)->set_definition(def);
    if (is_error(def)) {
        if (!is_syntax_error(qname)) {
            // suppress "annotation '<ERROR>' will be ignored"
            warning(
                ANNOTATION_IGNORED,
                anno->access_position(),
                Error_params(*this).add(qname));
        }
        return false;
    } else if (def->get_kind() != Definition::DK_ANNOTATION) {
        error(
            NOT_A_ANNOTATION,
            qname->access_position(),
            Error_params(*this).add_signature(def));
        if (Position const *def_pos = def->get_position()) {
            add_note(
                DEFINED_AT, *def_pos, Error_params(*this).add_signature(def));
        }
        def = get_error_definition();
        const_cast<IQualified_name *>(qname)->set_definition(def);

        return false;
    }

    bool arg_error = false;
    for (size_t i = 0, n = anno->get_argument_count(); i < n; ++i) {
        IArgument const *arg = anno->get_argument(i);

        visit(arg);

        IExpression const *expr = arg->get_argument_expr();

        if (is<IType_error>(expr->get_type())) {
            arg_error = true;
            continue;
        }

        bool is_invalid = false;
        if (!is_const_expression(expr, is_invalid)) {
            arg_error = true;
            if (!is_invalid) {
                if (is<IArgument_named>(arg)) {
                    IArgument_named const *narg  = cast<IArgument_named>(arg);
                    ISimple_name const    *sname = narg->get_parameter_name();
                    ISymbol const         *sym   = sname->get_symbol();

                    warning(
                        NON_CONSTANT_ANNOTATION_ARG_NAME,
                        arg->access_position(),
                        Error_params(*this).add(sym));
                } else {
                    warning(
                        NON_CONSTANT_ANNOTATION_ARG_POS,
                        arg->access_position(),
                        Error_params(*this).add_numword(i + 1));
                }
            }
        }
    }
    if (arg_error)
        def = get_error_definition();

    if (!is_error(def)) {
        bool bound_to_scope = false;
        if (qname->is_absolute())
            bound_to_scope = true;
        else {
            if (qname->get_component_count() > 1)
                bound_to_scope = true;
        }

        def = find_annotation_overload(def, anno, bound_to_scope);
    }

    if (is_error(def)) {
        warning(
            ANNOTATION_IGNORED,
            anno->access_position(),
            Error_params(*this).add(qname));
    } else if (def->get_semantics() != IDefinition::DS_UNKNOWN) {
        def = handle_known_annotation(def, anno);
    }

    const_cast<IQualified_name *>(qname)->set_definition(def);

    return false;
}

// end of an annotation block
void NT_analysis::post_visit(IAnnotation_block *anno_blk)
{
    IAnnotation const *hard_range = NULL;
    IAnnotation const *soft_range = NULL;
    int sr_index = 0;

    for (int i = 0, n = anno_blk->get_annotation_count(); i < n; ++i) {
        IAnnotation const     *anno  = anno_blk->get_annotation(i);
        IQualified_name const *qname = anno->get_name();
        IDefinition const     *def   = qname->get_definition();

        // allow only one soft/hard_range
        IDefinition::Semantics sema = def->get_semantics();
        if (sema == IDefinition::DS_HARD_RANGE_ANNOTATION) {
            if (hard_range == NULL) {
                hard_range = anno;
            } else {
                warning(
                    ADDITIONAL_ANNOTATION_IGNORED,
                    anno->access_position(),
                    Error_params(*this).add(def->get_symbol()));
                add_note(
                    PREVIOUSLY_ANNOTATED,
                    hard_range->access_position(),
                    Error_params(*this).add(def->get_symbol()));
                def = get_error_definition();
            }
        } else if (sema == IDefinition::DS_SOFT_RANGE_ANNOTATION) {
            if (soft_range == NULL) {
                soft_range = anno;
                sr_index   = i;
            } else {
                warning(
                    ADDITIONAL_ANNOTATION_IGNORED,
                    anno->access_position(),
                    Error_params(*this).add(def->get_symbol()));
                add_note(
                    PREVIOUSLY_ANNOTATED,
                    soft_range->access_position(),
                    Error_params(*this).add(def->get_symbol()));
                def = get_error_definition();
            }
        }

        if (is_error(def)) {
            // delete erroneous annotations
            anno_blk->delete_annotation(i);
            --i;
            --n;
        }
    }

    if (hard_range != NULL && soft_range != NULL) {
        IExpression_literal const *h_min_expr =
            cast<IExpression_literal>(hard_range->get_argument(0)->get_argument_expr());
        IValue const *h_min = h_min_expr->get_value();
        IExpression_literal const *h_max_expr =
            cast<IExpression_literal>(hard_range->get_argument(1)->get_argument_expr());
        IValue const *h_max = h_max_expr->get_value();

        IExpression_literal const *s_min_expr =
            cast<IExpression_literal>(soft_range->get_argument(0)->get_argument_expr());
        IValue const *s_min = s_min_expr->get_value();
        IExpression_literal const *s_max_expr =
            cast<IExpression_literal>(soft_range->get_argument(1)->get_argument_expr());
        IValue const *s_max = s_max_expr->get_value();

        if ((element_wise_compare(h_min, s_min) & IValue::CR_LE) == 0 ||
            (element_wise_compare(s_max, h_max) & IValue::CR_LE) == 0)
        {
            warning(
                SOFT_RANGE_OUTSIDE_HARD_RANGE,
                soft_range->access_position(),
                Error_params(*this)
                    .add(s_min)
                    .add(s_max)
                    .add(h_min)
                    .add(h_max));
            anno_blk->delete_annotation(sr_index);
        }
    }
}

/// visit all default arguments of the material constructors.
void NT_analysis::visit_material_default(Module_visitor &visitor)
{
    Flag_store store(m_inside_material_defaults, true);

    IType const *material_types[] = {
        m_tc.material_emission_type,
        m_tc.material_surface_type,
        m_tc.material_geometry_type,
        m_tc.material_emission_type,
        m_tc.material_volume_type,
        m_tc.material_type
    };

    for (size_t tp_idx = 0, n_tp = dimension_of(material_types); tp_idx < n_tp; ++tp_idx) {
        IType const *type = material_types[tp_idx];
        Scope *scope = m_def_tab->get_type_scope(type);

        for (Definition const *def = scope->get_first_definition_in_scope();
             def != NULL;
             def = def->get_next_def_in_scope())
        {
            if (def->get_kind() != Definition::DK_CONSTRUCTOR)
                continue;

            for (Definition const *cd = def; cd != NULL; cd = cd->get_next_def()) {
                IType_function const *ftype = cast<IType_function>(cd->get_type());

                for (int i = 0, n = ftype->get_parameter_count(); i < n; ++i) {
                    MDL_ASSERT(
                        cd->get_original_import_idx() == 0 &&
                        "material definition not from this module");

                    if (IExpression const *expr = cd->get_default_param_initializer(i)) {
                        (void)visitor.visit(expr);
                    }
                }
            }
        }
    }
}

// Update the call graph by a call to a function.
void NT_analysis::update_call_graph(Definition const *callee)
{
    if (is_error(callee))
        return;
    if (Definition *caller = tos_function()) {
        m_cg.add_call(caller, const_cast<Definition *>(callee));
    }
}

// Called for every called function in the semantic analysis.
void NT_analysis::check_called_function_existance(Definition const *def)
{
    if (is_error(def))
        return;

    if (def->has_flag(Definition::DEF_IS_DECL_ONLY)) {
        // a missing definition

        if (def->has_flag(Definition::DEF_IS_COMPILER_GEN) ||
            def->has_flag(Definition::DEF_IS_STDLIB))
        {
            // compiler generated constructors and operators doesn't have
            // definitions
            return;
        }
        MDL_ASSERT(def->get_position() != NULL);
        error(
            CALL_OF_UNDEFINED_FUNCTION,
            def,
            Error_params(*this)
                .add_signature(def));
    }
}

// Called for every global definition.
void NT_analysis::check_used_function(Definition const *def)
{
    if (is_error(def))
        return;

    if (def->get_kind() != IDefinition::DK_FUNCTION)
        return;

    if (def->has_flag(Definition::DEF_IS_DECL_ONLY))
        return;

    if (def->has_flag(Definition::DEF_IS_IMPORTED)) {
        // suppress warning for imported entities
        return;
    }

    if (def->has_flag(Definition::DEF_IS_USED)) {
        if (def->has_flag(Definition::DEF_IS_UNUSED)) {
            IType_function const *func_type  = cast<IType_function>(def->get_type());
            bool                 is_material = func_type->get_return_type() == m_tc.material_type;
            warning(
                is_material ? USED_MATERIAL_MARKED_UNUSED : USED_FUNCTION_MARKED_UNUSED,
                def,
                Error_params(*this).add_signature(def));
        }
    } else {
        if (!def->has_flag(Definition::DEF_IS_UNUSED)) {
            IType_function const *func_type  = cast<IType_function>(def->get_type());
            bool                 is_material = func_type->get_return_type() == m_tc.material_type;
            warning(
                is_material ? UNREFERENCED_MATERIAL_REMOVED : UNREFERENCED_FUNCTION_REMOVED,
                def,
                Error_params(*this).add_signature(def));
            if (def->has_flag(Definition::DEF_REF_BY_DEFAULT_INIT)) {
                add_note(
                    ONLY_REFERNCED_INSIDE_DEFAULT_INITIALIZER,
                    def,
                    Error_params(*this).add_signature(def));
            }
        }
    }
}

// Report a recursion.
void NT_analysis::process_scc(Call_node_vec const &scc)
{
    Definition *def = scc[0]->get_definition();

    error(
        RECURSION_CALL_NOT_ALLOWED,
        def,
        Error_params(*this)
            .add_signature(def));

    size_t n = scc.size();

    if (n > 1) {
        // dump the chain
        for (size_t i = 0; i < n; ++i) {
            Definition *caller = scc[i]->get_definition();

            int j = (i + 1) % n;
            Definition *callee = scc[j]->get_definition();

            add_note(
                RECURSION_CHAIN,
                callee,
                Error_params(*this)
                    .add_signature(callee)
                    .add_signature(caller));
        }
    } else {
        add_note(
            SELF_RECURSION,
            def,
            Error_params(*this)
                .add_signature(def));
    }
}

// Check that all called functions do exists.
void NT_analysis::check_called_functions()
{
    /// Helper class for the semantic analysis to traverse the call graph.
    class Existance_checker : public ICallgraph_visitor {
    public:
        Existance_checker(NT_analysis &ana)
        : m_ana(ana)
        , m_curr_mod_id(ana.m_module.get_unique_id())
        {
        }

        /// Call graph visitor.
        void visit_cg_node(Call_node *node, ICallgraph_visitor::Order order) {
            Definition *def = node->get_definition();

            if (def->get_owner_module_id() != m_curr_mod_id) {
                // FIXME: found an entity that will be auto-imported later. Ignore it,
                // it's from another module.
                return;
            }

            if (order == ICallgraph_visitor::POST_ORDER) {
                m_ana.check_called_function_existance(def);
            } else {
                if (def->has_flag(Definition::DEF_IS_USED)) {
                    // propagate the USED flag
                    for (Callee_iterator it(node->callee_begin()), end(node->callee_end());
                         it != end;
                         ++it)
                    {
                        Call_node  *callee = *it;
                        Definition *def    = callee->get_definition();

                        if (def->get_owner_module_id() != m_curr_mod_id) {
                            // FIXME: found an entity that will be auto-imported later. Ignore it,
                            // it's from another module. This is safe, because it must be exported
                            // anyway, so it *is* already marked as used.
                            continue;
                        }
                        def->set_flag(Definition::DEF_IS_USED);
                    }
                }
            }
        }

    private:
        // The analysis.
        NT_analysis &m_ana;

        /// The ID of the current module.
        size_t m_curr_mod_id;
    };

    /// Helper class to check that all functions are used
    class Usage_checker : public IDefinition_visitor {
    public:
        Usage_checker(NT_analysis &ana) : m_ana(ana)
        {
        }

        /// Call visitor.
        void visit(Definition const *def) const MDL_FINAL {
            m_ana.check_used_function(def);
        }

    private:
        // The analysis.
        NT_analysis &m_ana;
    };

    Existance_checker ec(*this);
    Usage_checker     uc(*this);

    // check if the call graph has loops and report them
    m_cg.finalize(this);

    // mark all exported roots as used
    Call_node_vec const &root_set = m_cg.get_root_set();
    for (size_t i = 0, n = root_set.size(); i < n; ++i) {
        Call_node  *root = root_set[i];
        Definition *def  = root->get_definition();

        if (def->has_flag(Definition::DEF_IS_EXPORTED))
            def->set_flag(Definition::DEF_IS_USED);
    }

    // check that every called function exists and mark used ones
    Call_graph_walker::walk(m_cg, &ec);

    if (m_opt_dump_cg) {
        dump_call_graph(get_allocator(), m_module.get_name(), m_cg);
    }

    Scope *file_scope = m_def_tab->get_global_scope();

    // report unused functions
    file_scope->walk(&uc);
}

// Dump a call graph to a file.
void NT_analysis::dump_call_graph(
    IAllocator       *alloc,
    char const       *module_name,
    Call_graph const &cg)
{
    // dump the call graph
    string fname("cg_", alloc);
    char const *abs_name = module_name;
    if (abs_name[0] == ':' && abs_name[1] == ':')
        abs_name += 2;
    fname += abs_name;
    for (size_t i = 0, n = fname.size(); i < n; ++i) {
        if (fname[i] == ':')
            fname[i] = '_';
    }
    fname += ".gv";

    if (FILE *f = fopen(fname.c_str(), "w")) {
        Allocator_builder builder(alloc);

        mi::base::Handle<File_Output_stream> out(
            builder.create<File_Output_stream>(alloc, f, /*close_at_destroy=*/true));

        cg.dump(out.get());
    }
}

// Check that all exported entities exist.
void NT_analysis::check_exported_for_existance()
{
    // we have already put all declaration only in a list, so just check if
    // every definition in the list has now a Definition
    for (size_t i = 0, n = m_exported_decls_only.size(); i < n; ++i) {
        Definition const *def = m_exported_decls_only[i];
        Definition const *def_def = def->get_definite_definition();

        if (def_def != NULL)
            def = def_def;
        if (def->has_flag(Definition::DEF_IS_STDLIB)) {
            // standard library entries are intrinsics and known by the compiler
            // even without a definition
            continue;
        }
        if (def->has_flag(Definition::DEF_IS_DECL_ONLY)) {
            // no definition found
            error(
                EXPORTED_FUNCTION_NOT_DEFINED,
                def,
                Error_params(*this)
                .add_signature(def));
        }
    }
}

// Check restriction on the given file path.
bool NT_analysis::check_file_path(char const *path, Position const &pos)
{
    bool res = true;

    // work-around for MDLe's misuse of the MDL path name
    char const *p = strstr(path, "mdle:");
    if (p != NULL) {
        path = p + 5;
    }

    if (m_module.get_mdl_version() < IMDL::MDL_VERSION_1_3) {
        // . and .. are not allowed before MDL 1.3

        int major = 1, minor = 0;
        m_module.get_version(major, minor);

        bool start = true;
        for (char const *p = path; p[0] != '\0'; ++p) {
            if (start) {
                if (p[0] == '.') {
                    if (p[1] == '/' || p[1] == '\0') {
                        // found '.'
                        warning(
                            CURR_DIR_FORBIDDEN_IN_RESOURCE,
                            pos,
                            Error_params(*this).add(major).add(minor));
                    } else if (p[1] == '.' && (p[2] == '/' || p[2] == '\0')) {
                        // found '..'
                        warning(
                            PARENT_DIR_FORBIDDEN_IN_RESOURCE,
                            pos,
                            Error_params(*this).add(major).add(minor));
                    }
                }
                start = false;
            }
            if (p[0] == '/')
                start = true;
        }
    }

    // check restrictions
    u32string u32(get_allocator());
    utf8_to_utf32(u32, path);

    for (size_t i = 0, l = u32.size(); i < l; ++i) {
        unsigned c32 = u32[i];

        if (c32 < 32 || c32 == 127 || c32 == ':' || c32 == '\\') {
            char buffer[8];

            if (c32 == ':') {
                buffer[0] = ':';
                buffer[1] = '\0';
            } else if (c32 == '\\') {
                buffer[0] = '\\';
                buffer[1] = '\0';
            } else {
                snprintf(buffer, 8, "\\u%04u", c32);
            }

            error(
                INVALID_CHARACTER_IN_RESOURCE,
                pos,
                Error_params(*this).add(buffer));
            res = false;
        }
    }
    return res;
}

// Handle resource constructors.
IExpression *NT_analysis::handle_resource_constructor(IExpression_call *call_expr)
{
    IType const *res_type = call_expr->get_type()->skip_type_alias();

    switch (res_type->get_kind()) {
    case IType::TK_TEXTURE:
    case IType::TK_LIGHT_PROFILE:
    case IType::TK_BSDF_MEASUREMENT:
        // the resource name is always the first argument
        if (call_expr->get_argument_count() >= 1) {
            IArgument const           *arg0 = call_expr->get_argument(0);
            IExpression_literal const *lit = as<IExpression_literal>(arg0->get_argument_expr());
            if (lit != NULL) {
                if (IValue_string const *sval = as<IValue_string>(lit->get_value())) {
                    handle_resource_url(
                        sval,
                        const_cast<IExpression_literal *>(lit),
                        cast<IType_resource>(res_type));
                }
            } else {
                // the spec forbids the uses of non-literals here
                error(
                    RESOURCE_NAME_NOT_LITERAL,
                    arg0->access_position(),
                    Error_params(*this));
            }
        }
        break;
    default:
        break;
    }

    // always fold resource constructors
    IValue const *v = call_expr->fold(&m_module, m_module.get_value_factory(), /*handler=*/NULL);
    if (!is<IValue_bad>(v)) {
        return m_module.get_expression_factory()->create_literal(
            v, POS(call_expr->access_position()));
    }
    return call_expr;
}

// Add a new resource entry.
void NT_analysis::add_resource_entry(
    char const           *url,
    char const           *file_name,
    int                  resource_tag,
    IType_resource const *type,
    bool                 exists)
{
    Resource_entry re(get_allocator(), url, file_name, resource_tag, type, exists);

    const char *found = strstr(file_name, ".mdle:");
    if (found) {
        Resource_table_key key(convert_os_separators_to_slashes(re.m_filename), re.m_res_tag);
        m_resource_entries.insert(Resource_table::value_type(key, re));
    } else {
        Resource_table_key key(re.m_url, re.m_res_tag);
        m_resource_entries.insert(Resource_table::value_type(key, re));
    }

}

// Copy the given resolver messages to the current module.
void NT_analysis::copy_resolver_messages_to_module(
    Messages_impl const &messages,
    bool is_resource)
{
    bool map_res_err_to_warn = false;
    if (is_resource && m_module.get_mdl_version() < IMDL::MDL_VERSION_1_4) {
        map_res_err_to_warn = true;
    }

    for (size_t i = 0, n = messages.get_error_message_count(); i < n; ++i) {
        IMessage const *msg = messages.get_message(i);

        IMessage::Severity sev = msg->get_severity();

        if (map_res_err_to_warn) {
            // map resource error to warning for MDL < 1.4
            if (sev == IMessage::MS_ERROR)
                sev = IMessage::MS_WARNING;
        }

        size_t index = m_module.access_messages_impl().add_message(
            sev,
            msg->get_code(),
            msg->get_class(),
            msg->get_string(),
            msg->get_file(),
            msg->get_position()->get_start_line(),
            msg->get_position()->get_start_column(),
            msg->get_position()->get_end_line(),
            msg->get_position()->get_end_column());

        for (size_t j = 0, m = msg->get_note_count(); j < m; ++j) {
            IMessage const *note = msg->get_note(j);

            m_module.access_messages_impl().add_note(
                index,
                note->get_severity(),
                note->get_code(),
                note->get_class(),
                note->get_string(),
                note->get_file(),
                note->get_position()->get_start_line(),
                note->get_position()->get_start_column(),
                note->get_position()->get_end_line(),
                note->get_position()->get_end_column());
        }
        m_last_msg_idx = index;
    }
}

namespace {

/// Helper class to change the qualified name of auto-imported entities.
class Reference_replace : public Module_visitor {
public:
    typedef vector<Definition *>::Type Def_vec;

    /// Constructor.
    Reference_replace(
        NT_analysis   &ana,
        Module        &mod,
        Def_vec const &initializer_to_fix)
    : m_ana(ana)
    , m_mod(mod)
    , m_def_tab(mod.get_definition_table())
    , m_name_fact(*mod.get_name_factory())
    , m_initializer_to_fix(initializer_to_fix)
    {
    }

    /// Run the replacement.
    void run(Module &mod)
    {
        // replace in the AST
        visit(&mod);

        // replace in the default initializers
        for (Def_vec::const_iterator
             it(m_initializer_to_fix.begin()), end(m_initializer_to_fix.end());
             it != end;
             ++it)
        {
            Definition *def = *it;

            IType_function const *ftype = cast<IType_function>(def->get_type());
            for (int i = 0, n = ftype->get_parameter_count(); i < n; ++i) {
                IExpression const *init = def->get_default_param_initializer(i);

                if (init != NULL) {
                    (void)visit(init);
                }
            }
        }
    }

    /// Visit and fix a reference expression.
    IExpression *post_visit(IExpression_reference *ref) MDL_FINAL
    {
        IDefinition const *idef = ref->get_definition();
        Definition const  *def  = impl_cast<Definition>(idef);

        if (Definition const *imp_def = m_ana.get_auto_imported_definition(def)) {
            size_t q_len = 0;
            Scope *scope = imp_def->get_def_scope();

            IType_name const *type_name = ref->get_name();

            if (imp_def->get_kind() == Definition::DK_CONSTRUCTOR) {
                // constructors are inside the type scope, skip it
                scope = scope->get_parent();
            }

            Position const &pos = type_name->get_qualified_name()->access_position();

            IQualified_name *qname = m_name_fact.create_qualified_name(POS(pos));

            if (scope != m_def_tab.get_predef_scope()) {
                Scope const *global = m_def_tab.get_global_scope();

                for (Scope const *s = scope; s != global; s = s->get_parent()) {
                    ++q_len;
                }
                VLA<ISymbol const *> syms(m_mod.get_allocator(), q_len);

                size_t i = 1;
                for (Scope const *s = scope; s != global; s = s->get_parent()) {
                    syms[q_len - i] = s->get_scope_name();
                    ++i;
                }

                for (size_t i = 0; i < q_len; ++i) {
                    qname->add_component(m_name_fact.create_simple_name(syms[i]));
                }
            }
            qname->add_component(m_name_fact.create_simple_name(imp_def->get_sym()));
            qname->set_definition(imp_def);

            IType_name *res = m_name_fact.create_type_name(qname);
            res->set_qualifier(type_name->get_qualifier());

            ref->set_name(res);
            ref->set_definition(imp_def);
        }
        return ref;
    }

private:
    /// The analysis.
    NT_analysis &m_ana;

    // The module.
    Module &m_mod;

    /// The definition table of the module.
    Definition_table const &m_def_tab;

    /// The Name factory of the module.
    Name_factory &m_name_fact;

    /// List of initializer that must be fixed.
    Def_vec const &m_initializer_to_fix;
};

}  // anonymous

// Try to import the symbol
bool NT_analysis::auto_import(
    Module const  *imp_mod,
    ISymbol const *sym)
{
    Definition_table const &imp_deftab = imp_mod->get_definition_table();
    size_t                 imp_idx     = m_module.get_import_index(imp_mod);
    Scope                  *imp_scope  = imp_deftab.get_global_scope();

    ISymbol const *imp_sym = imp_mod->lookup_symbol(sym);
    Definition *def = imp_scope->find_definition_in_scope(imp_sym);
    if (def == NULL) {
        // should not happen
        return false;
    }
    if (!def->has_flag(Definition::DEF_IS_EXPORTED)) {
        // should not happen
        return false;
    }

    // try to import into named scope
    {
        Definition_table::Scope_transition transition(*m_def_tab, m_def_tab->get_global_scope());
        IQualified_name const *imp_name = imp_mod->get_qualified_name();

        enter_import_scope(imp_name, 0, /*ignore_last=*/false);
        // qualified imports are never exported

        Definition const *clash_def = import_definition(
            imp_mod, imp_idx, def, /*is_exported=*/false, Position_impl(0, 0, 0, 0));
        if (clash_def != NULL) {
            // bad, cannot import
            return false;
        }

        // if it is a enum type, its enum values are at the same scope ... must check for them
        Scope::Definition_list values(get_allocator());
        if (def->get_kind() == IDefinition::DK_TYPE) {
            if (IType_enum const *e_type = as<IType_enum>(def->get_type())) {
                imp_scope->collect_enum_values(e_type, values);
            }
        }

        for (Scope::Definition_list::const_iterator it(values.begin()), end(values.end());
            it != end;
            ++it)
        {
            Definition const *ev_def = *it;

            Definition const *clash_def = import_definition(
                imp_mod, imp_idx, ev_def, /*is_exported=*/false, Position_impl(0, 0, 0, 0));
            if (clash_def != NULL) {
                // bad, cannot import associated enum value
                return false;
            }
        }
        return true;
    }
}

// Add an import declaration to the current module.
void NT_analysis::add_auto_import_declaration(
    Module const  *imp_mod,
    ISymbol const *sym)
{
    IDeclaration_import   *import    = m_module.get_declaration_factory()->create_import();
    IQualified_name const *mod_name  = imp_mod->get_qualified_name();
    Name_factory          &name_fact = *m_module.get_name_factory();

    IQualified_name *q_name = name_fact.create_qualified_name();
    for (int i = 0, n = mod_name->get_component_count(); i < n; ++i) {
        ISymbol const *s = mod_name->get_component(i)->get_symbol();
        q_name->add_component(name_fact.create_simple_name(m_module.import_symbol(s)));
    }
    ISymbol const *imp_sym = m_module.import_symbol(sym);
    q_name->add_component(name_fact.create_simple_name(imp_sym));

    // is always an absolute import
    q_name->set_absolute();

    import->add_name(q_name);
    m_module.add_auto_import(import);
}

// Insert all auto imports from default initializers.
void NT_analysis::fix_auto_imports()
{
    if (m_auto_imports.empty())
        return;

    // first step: do the imports
    for (size_t i = 0, n = m_auto_imports.size(); i < n; ++i) {
        Auto_imports::Entry &entry = m_auto_imports[i];
        Definition const    *def   = entry.def;

        if (Definition const *imp_def = m_module.find_imported_definition(def)) {
            entry.imported = imp_def;
        } else {
            // do auto-import here
            bool already_imported = false;

            size_t original_id = 0;
            Module const *imp_mod = m_module.find_imported_module(
                def->get_owner_module_id(), already_imported);

            if (def->get_kind() == IDefinition::DK_ENUM_VALUE) {
                if (imp_mod == NULL) {
                    // this should really not happen here, it it does, then something
                    // in the imported modules goes wrong, probably in auto-import
                    error(
                        INTERNAL_COMPILER_ERROR,
                        def,
                        Error_params(*this)
                        .add("auto import of '")
                        .add_signature(def)
                        .add("' failed"));
                    continue;
                }

                // Get the enum type definition for this value here.
                // unfortunately enum value definitions does NOT live inside
                // the enum type scope, so we must lookup the enum type scope
                // first using the definition table of the imported module.
                IType const *e_type = def->get_type();
                Definition_table const &deftab = imp_mod->get_definition_table();
                Scope const *scope = deftab.get_type_scope(e_type);
                MDL_ASSERT(scope != NULL && "Cannot find enum type for enum value");
                Definition const *type_def = scope->get_owner_definition();

                // just add a new auto-import for the type and stop here, the enum values
                // will be fixed in the next step
                if (m_auto_imports.insert(type_def)) {
                    // first insert
                    ++n;
                }
                continue;
            }

            if (imp_mod != NULL) {
                // Check if the def was imported by the imported module itself.
                // if yes, we assume the imported module was correct itself AND it was then
                // exported by the import of the imported module.
                size_t import_idx = def->get_original_import_idx();
                if (import_idx != 0)
                    imp_mod = imp_mod->get_import_entry(import_idx)->get_module();

                original_id = imp_mod->get_unique_id();
            }

            imp_mod = m_module.find_imported_module(original_id, already_imported);
            if (imp_mod != NULL) {
                if (!already_imported)
                    m_module.register_import(imp_mod);

                bool res = auto_import(imp_mod, def->get_sym());
                if (res) {
                    // add an import declaration
                    add_auto_import_declaration(imp_mod, def->get_sym());

                    entry.imported = m_module.find_imported_definition(def);
                    MDL_ASSERT(entry.imported != NULL && "Could not find autoimported entity");
                }
            } else {
                // this should not happen
                error(
                    INTERNAL_COMPILER_ERROR,
                    def,
                    Error_params(*this)
                        .add("auto import of '")
                        .add_signature(def)
                        .add("' failed"));
            }
        }
    }

    // second step: handle enum values, they still might be not resolved
    for (size_t i = 0, n = m_auto_imports.size(); i < n; ++i) {
        Auto_imports::Entry &entry = m_auto_imports[i];
        Definition const    *def = entry.def;

        if (entry.imported == NULL && def->get_kind() == IDefinition::DK_ENUM_VALUE) {
            // the enum type must be auto-imported already, hence the values should be found now
            Definition const *imp_def = m_module.find_imported_definition(def);
            entry.imported = imp_def;

            MDL_ASSERT(imp_def != NULL && "second step of auto-import failed");
        }
    }

    // last step: walk over the AST and replace the definitions
    Reference_replace ref_replace(*this, m_module, m_initializers_must_be_fixed);

    ref_replace.run(m_module);
    if (!m_module.is_stdlib() || m_module.is_builtins()) {
        // update builtin types
        visit_material_default(ref_replace);
    }
}

// Return the Type from a IType_name, handling errors if the IType_name
// does not name a type.
IType const *NT_analysis::as_type(IType_name const *type_name)
{
    IType const *type = type_name->get_type();
    if (type == NULL) {
        // typename does NOT name a type, but an entity
        error(
            NOT_A_TYPE_NAME,
            type_name->access_position(),
            Error_params(*this)
                .add(type_name->get_qualified_name()));

        // must be a type here
        type = m_tc.error_type;
        const_cast<IType_name *>(type_name)->set_type(type);
    }
    return type;
}

// Get a array size from a definition.
IType_array_size const *NT_analysis::get_array_size(
    IDefinition const *def,
    ISymbol const     *abs_name)
{
    if (abs_name != NULL) {
        IType_array_size const *size = m_tc.get_array_size(abs_name, def->get_symbol());

        MDL_ASSERT(m_array_size_map.find(def) == m_array_size_map.end() &&
               "array size collision");

        m_array_size_map[def] = size;
        return size;
    } else {
        Array_size_map::const_iterator it = m_array_size_map.find(def);

        MDL_ASSERT(it != m_array_size_map.end() && "Cannot find array size");
        return it->second;
    }
}

// Collect all builtin entities and put them into the current module.
void NT_analysis::collect_builtin_entities()
{
    Scope const *scope = m_def_tab->get_predef_scope();

    Definition const *mat_definitions[IType_struct::SID_LAST + 1];
    memset(mat_definitions, 0, sizeof(mat_definitions));

    for (Definition const *def = scope->get_first_definition_in_scope();
        def != NULL;
        def = def->get_next_def_in_scope())
    {
        switch (def->get_kind()) {
        case IDefinition::DK_FUNCTION:
        case IDefinition::DK_OPERATOR:
        case IDefinition::DK_CONSTANT:
            // functions, types, operators and constants are visible
            for (Definition const *odef = def; odef != NULL; odef = odef->get_next_def()) {
                m_module.m_builtin_definitions.push_back(odef);
            }
            break;
        case IDefinition::DK_TYPE:
            {
                IType const *type = def->get_type();
                IType::Kind kind = type->get_kind();

                if (kind != IType::TK_STRUCT && kind != IType::TK_ENUM) {
                    // Neither a struct nor an enum type. Add constructors for those types
                    // directly, constructors for struct and enum types can be retrieved from
                    // the types itself.
                    Scope const *def_scope = def->get_own_scope();
                    for (Definition const *idef = def_scope->get_first_definition_in_scope();
                         idef != NULL;
                         idef = idef->get_next_def_in_scope())
                    {
                        if (idef->get_kind() == IDefinition::DK_CONSTRUCTOR) {
                            // add constructors
                            for (Definition const *odef = idef;
                                 odef != NULL;
                                 odef = odef->get_next_def())
                            {
                                m_module.m_builtin_definitions.push_back(odef);
                            }
                        }
                    }
                }
                if (kind != IType::TK_STRUCT) {
                    // add the type itself
                    m_module.m_builtin_definitions.push_back(def);
                } else {
                    // work-around for missing dependency due to default initializers
                    // in the DAG backend: ensure the material types are presorted in dependency
                    // order
                    IType_struct const *s_type = as<IType_struct>(def->get_type());
                    IType_struct::Predefined_id id = s_type->get_predefined_id();
                    MDL_ASSERT(id != IType_struct::SID_USER);
                    mat_definitions[id] = def;
                }
            }
        default:
            // ignore others
            break;
        }
    }
    for (size_t i = 0; i <= IType_struct::SID_LAST; ++i) {
        Definition const *def = mat_definitions[i];

        MDL_ASSERT(def != NULL);
        m_module.m_builtin_definitions.push_back(def);
    }
}

// Calculate, which function uses the state.
void NT_analysis::calc_state_usage()
{
    class State_spreader : public ICallgraph_visitor {
    public:
        /// Visit a node of the call graph.
        void visit_cg_node(Call_node *node, ICallgraph_visitor::Order order) MDL_FINAL {
            if (order == POST_ORDER) {
                enum Attributes {
                    need_state         = 1 << 0,
                    need_varying_state = 1 << 1,
                    need_tex           = 1 << 2,
                    can_thow_bounds    = 1 << 3,
                    can_thow_divzero   = 1 << 4,
                    read_tex           = 1 << 5,
                    read_lp            = 1 << 6,
                    uses_debug         = 1 << 7,
                    uses_object_id     = 1 << 8,
                    uses_transform     = 1 << 9,
                    uses_normal        = 1 << 10,
                    uses_tex_attr      = 1 << 11,
                    uses_lp_attr       = 1 << 12,
                    need_scene_data    = 1 << 13,
                };
                unsigned attr = 0;

                for (Callee_iterator it = node->callee_begin(), end = node->callee_end();
                     it != end;
                     ++it)
                {
                    Call_node        *callee     = *it;
                    Definition const *callee_def = callee->get_definition();

                    if (callee_def->has_flag(Definition::DEF_USES_STATE)) {
                        attr |= need_state;
                    }
                    if (callee_def->has_flag(Definition::DEF_USES_VARYING_STATE)) {
                        attr |= need_varying_state;
                    }
                    if (callee_def->has_flag(Definition::DEF_USES_TEXTURES)) {
                        attr |= need_tex;
                    }
                    if (callee_def->has_flag(Definition::DEF_USES_DERIVATIVES)) {
                        switch (callee_def->get_semantics()) {
                        case IDefinition::DS_INTRINSIC_TEX_LOOKUP_FLOAT:
                        case IDefinition::DS_INTRINSIC_TEX_LOOKUP_FLOAT2:
                        case IDefinition::DS_INTRINSIC_TEX_LOOKUP_FLOAT3:
                        case IDefinition::DS_INTRINSIC_TEX_LOOKUP_FLOAT4:
                        case IDefinition::DS_INTRINSIC_TEX_LOOKUP_COLOR:
                        case IDefinition::DS_INTRINSIC_MATH_DX:
                        case IDefinition::DS_INTRINSIC_MATH_DY:
                            {
                                Definition *def = node->get_definition();
                                def->set_flag(Definition::DEF_USES_DERIVATIVES);
                            }
                            break;
                        default:
                            break;
                        }
                    }
                    if (callee_def->has_flag(Definition::DEF_CAN_THROW_BOUNDS)) {
                        attr |= can_thow_bounds;
                    }
                    if (callee_def->has_flag(Definition::DEF_CAN_THROW_DIVZERO)) {
                        attr |= can_thow_divzero;
                    }
                    if (callee_def->has_flag(Definition::DEF_READ_TEXTURE_ATTR)) {
                        attr |= read_tex;
                    }
                    if (callee_def->has_flag(Definition::DEF_READ_LP_ATTR)) {
                        attr |= read_lp;
                    }
                    if (callee_def->has_flag(Definition::DEF_USES_DEBUG_CALLS)) {
                        attr |= uses_debug;
                    }
                    if (callee_def->has_flag(Definition::DEF_USES_OBJECT_ID)) {
                        attr |= uses_object_id;
                    }
                    if (callee_def->has_flag(Definition::DEF_USES_TRANSFORM)) {
                        attr |= uses_transform;
                    }
                    if (callee_def->has_flag(Definition::DEF_USES_NORMAL)) {
                        attr |= uses_normal;
                    }
                    if (callee_def->has_flag(Definition::DEF_READ_TEXTURE_ATTR)) {
                        attr |= uses_tex_attr;
                    }
                    if (callee_def->has_flag(Definition::DEF_READ_LP_ATTR)) {
                        attr |= uses_lp_attr;
                    }
                    if (callee_def->has_flag(Definition::DEF_USES_SCENE_DATA)) {
                        attr |= need_scene_data;
                    }
                }
                if (attr & need_state) {
                    Definition *def = node->get_definition();
                    def->set_flag(Definition::DEF_USES_STATE);
                }
                if (attr & need_varying_state) {
                    Definition *def = node->get_definition();
                    def->set_flag(Definition::DEF_USES_VARYING_STATE);
                }
                if (attr & need_tex) {
                    Definition *def = node->get_definition();
                    def->set_flag(Definition::DEF_USES_TEXTURES);
                }
                if (attr & can_thow_bounds) {
                    Definition *def = node->get_definition();
                    def->set_flag(Definition::DEF_CAN_THROW_BOUNDS);
                }
                if (attr & can_thow_divzero) {
                    Definition *def = node->get_definition();
                    def->set_flag(Definition::DEF_CAN_THROW_DIVZERO);
                }
                if (attr & read_tex) {
                    Definition *def = node->get_definition();
                    def->set_flag(Definition::DEF_READ_TEXTURE_ATTR);
                }
                if (attr & read_lp) {
                    Definition *def = node->get_definition();
                    def->set_flag(Definition::DEF_READ_LP_ATTR);
                }
                if (attr & uses_debug) {
                    Definition *def = node->get_definition();
                    def->set_flag(Definition::DEF_USES_DEBUG_CALLS);
                }
                if (attr & uses_object_id) {
                    Definition *def = node->get_definition();
                    def->set_flag(Definition::DEF_USES_OBJECT_ID);
                }
                if (attr & uses_transform) {
                    Definition *def = node->get_definition();
                    def->set_flag(Definition::DEF_USES_TRANSFORM);
                }
                if (attr & uses_normal) {
                    Definition *def = node->get_definition();
                    def->set_flag(Definition::DEF_USES_NORMAL);
                }
                if (attr & uses_tex_attr) {
                    Definition *def = node->get_definition();
                    def->set_flag(Definition::DEF_READ_TEXTURE_ATTR);
                }
                if (attr & uses_lp_attr) {
                    Definition *def = node->get_definition();
                    def->set_flag(Definition::DEF_READ_LP_ATTR);
                }
                if (attr & need_scene_data) {
                    Definition *def = node->get_definition();
                    def->set_flag(Definition::DEF_USES_SCENE_DATA);
                }
            }
        }
    };

    State_spreader spreader;
    Call_graph_walker::walk(m_cg, &spreader);
}


// Run the name and type analysis on this module.
void NT_analysis::run()
{
    if (m_module.get_owner_archive_version() != NULL) {
        if (m_module.m_mdl_version > m_module.m_arc_mdl_version) {
            Position_impl zero(0, 0, 0, 0);
            error(
                FORBIDDEN_MDL_VERSION_IN_ARCHIVE,
                zero,
                Error_params(*this)
                    .add_mdl_version(m_module.m_mdl_version)
                    .add_mdl_version(m_module.m_arc_mdl_version)
            );
        }
    }

    // clear up, analyze might be called more than once
    m_module.m_imported_modules.clear();
    m_module.m_exported_definitions.clear();

    // clear the table here, this might be a re-analyze
    m_def_tab->clear();

    // enter the outer scope
    m_def_tab->reopen_scope(m_def_tab->get_predef_scope());

    // enter all predefined entities
    enter_predefined_entities(m_module, m_tc, m_compiler->build_predefined_types());

    // and create the error definition here, so the set of
    // predefined entities is constant
    ISymbol const *err_sym = m_st->get_error_symbol();
    m_def_tab->enter_error(err_sym, m_tc.error_type);

    enter_builtin_constants();

    if (m_module.is_stdlib())
        enter_builtin_annotations();
    if (m_module.is_native())
        enter_native_annotations();

    // enter the global scope
    m_def_tab->reopen_scope(m_def_tab->get_global_scope());

    if (m_module.m_is_stdlib) {
        // stdlib constants must live at global scope
        enter_stdlib_constants();

        if (m_module.is_builtins()) {
            visit_material_default(*this);
            collect_builtin_entities();
        }
    } else {
        // every NON-stdlib module hidden imports the ::<builtins> module
        // because this contains all the builtin-entities, see above
        m_module.register_import(m_compiler->find_builtin_module(
            string("::<builtins>", get_allocator())));

        visit_material_default(*this);
    }

    visit(&m_module);

    // leave global scope
    m_def_tab->leave_scope();

    // leave outer scope
    m_def_tab->leave_scope();

    check_exported_for_existance();
    check_called_functions();

    fix_auto_imports();

    calc_state_usage();


    // run auto-typing
    AT_analysis::run(m_compiler, m_module, m_ctx, m_cg);

    // check additional restrictions on resources if we have a handler
    if (IResource_restriction_handler *rrh = m_ctx.get_resource_restriction_handler()) {
        File_resolver resolver(
            *m_compiler,
            m_module_cache,
            m_compiler->get_external_resolver(),
            m_compiler->get_search_path(),
            m_compiler->get_search_path_lock(),
            m_module.access_messages_impl(),
            m_ctx.get_front_path());

        m_module.check_referenced_resources(*this, resolver, *rrh);
    }

    // copy the resource table.
    if (!m_resource_entries.empty()) {
        m_module.m_res_table.clear();
        m_module.m_res_table.reserve(m_resource_entries.size());

        for (Resource_table::const_iterator
                it(m_resource_entries.begin()), end(m_resource_entries.end());
             it != end;
             ++it)
        {
            Resource_entry const &re = it->second;

            m_module.add_resource_entry(
                re.m_url.c_str(),
                re.m_filename.empty() ? NULL : re.m_filename.c_str(),
                re.m_type,
                re.m_exists);
        }
    }
}

// Process a resource url.
void NT_analysis::handle_resource_url(
    IValue const         *val,
    IExpression_literal  *lit,
    IType_resource const *res_type)
{
    IValue_string const *sval   = as<IValue_string>(val);
    IValue_resource const *rval = as<IValue_resource>(val);
    MDL_ASSERT(sval != NULL || rval != NULL);

    char const *url = sval != NULL ? sval->get_value() : rval->get_string_value();

    if (!check_file_path(url, lit->access_position())) {
        return;
    }

    string abs_url(get_allocator());
    string abs_file_name(get_allocator());

    bool resource_exists = true;
    int resource_tag = 0;
    if (m_resolve_resources) {
        resource_exists = true;
        if (abs_url.empty() && rval != NULL && rval->get_tag_value() != 0) {
            // this is a "neuray-style" in-memory resource, represented by a tag;
            // assume existing
            resource_tag = rval->get_tag_value();
        } else {
            // try to resolve it
            Messages_impl messages(get_allocator(), m_module.get_filename());

            File_resolver resolver(
                *m_compiler,
                m_module_cache,
                m_compiler->get_external_resolver(),
                m_compiler->get_search_path(),
                m_compiler->get_search_path_lock(),
                messages,
                m_ctx.get_front_path());

            string murl(url, get_allocator());

            bool is_weak_16 = false;
            if (m_module.get_mdl_version() >= IMDL::MDL_VERSION_1_6) {
                // from MDL 1.6 weak imports do not exists
                if (murl[0] != '/' &&
                    murl[0] != '.' &&
                    murl.rfind(".mdle:") == string::npos) {
                        is_weak_16 = true;
                        // previous weak imports are relative now
                        murl = "./" + murl;
                }
            }

            mi::base::Handle<IMDL_resource_set> res(resolver.resolve_resource(
                lit->access_position(),
                murl.c_str(),
                m_module.get_name(),
                m_module.get_filename()));

            // copy messages
            copy_resolver_messages_to_module(messages, /*is_resource=*/ true);

            if (res.is_valid_interface()) {
                abs_url = res->get_mdl_url_mask();
                if (char const *s = res->get_filename_mask())
                    abs_file_name = s;
            }

            if (messages.get_error_message_count() > 0 || !res.is_valid_interface()) {
                resource_exists = false;

                if (is_weak_16 &&
                    !res.is_valid_interface() &&
                    messages.get_error_message_count() == 1)
                {
                    // resolving a formally weak reference failed, check if it is absolute
                    mi::base::Handle<IMDL_resource_set> res(resolver.resolve_resource(
                        lit->access_position(),
                        murl.c_str() + 1,
                        m_module.get_name(),
                        m_module.get_filename()));
                    if (res.is_valid_interface()) {
                        // .. and add a note
                        add_note(
                            POSSIBLE_ABSOLUTE_IMPORT,
                            lit->access_position(),
                            Error_params(*this).add(murl.c_str() + 1));
                    }
                }

                // Resolver failed. This is bad, because letting the name unchanged
                // might lead to "wrong" fixes later. One possible solution would be to
                // return an invalid resource here, but then the user will not see ANY
                // error.
                // Hence we do a "stupid" transformation here, i.e. compute an absolute
                // path that will point into PA. Note that the resource still will fail,
                // otherwise the resolver had found it there.
                if (url[0] != '/') {
                    abs_url = make_absolute_package(
                        get_allocator(), url, m_module.get_name());
                } else {
                    abs_url = url;
                }
            }
        }
    } else {
        // do NOT resolve resources
        if (url[0] == '/') {
            // an absolute url, keep it.
            abs_url = url;
        } else if ((url[0] == '.' && url[1] == '/') ||
                 (url[0] == '.' && url[1] == '.' && url[2] == '/' )) {
            // strict relative, make absolute
            abs_url = make_absolute_package(
                get_allocator(), url, m_module.get_name());
        } else {
            if (m_module.get_mdl_version() >= IMDL::MDL_VERSION_1_6) {
                // formally weak relatives are mapped to strict
                string murl("./", get_allocator());
                murl += url;
                abs_url = make_absolute_package(
                    get_allocator(), murl.c_str(), m_module.get_name());
            } else {
                // weak relative, prepend module name separated by "::"
                abs_url = m_module.get_name();
                abs_url += "::";
                abs_url += url;
            }
        }
        // unknown file name
        abs_file_name = "";

        // assume the resource exists
        resource_exists = true;
    }

    // in case of MDLE, the absolute URL also contains the MDLE file path
    size_t p = abs_file_name.rfind(".mdle:");
    if (p != string::npos) {
        abs_url = convert_os_separators_to_slashes(abs_file_name);
    }

    // add to the resource table
    add_resource_entry(
        abs_url.c_str(),
        abs_file_name.c_str(),
        resource_tag,
        res_type,
        resource_exists);

    if (m_opt_keep_original_resource_file_paths)
        return;

    // rewrite if necessary
    if (string(url, get_allocator()) != abs_url) {
        switch (val->get_kind()) {
        case IValue::VK_STRING:
            val = m_module.get_value_factory()->create_string(abs_url.c_str());
            break;
        case IValue::VK_TEXTURE:
            {
                IValue_texture const *tval = cast<IValue_texture>(val);
                IType_texture const *t = tval->get_type();
                MDL_ASSERT(t->get_shape() != IType_texture::TS_BSDF_DATA);
                val = m_module.get_value_factory()->create_texture(
                    t,
                    abs_url.c_str(),
                    tval->get_gamma_mode(),
                    tval->get_tag_value(),
                    tval->get_tag_version());
            }
            break;
        case IValue::VK_LIGHT_PROFILE:
            {
                IValue_light_profile const *lval = cast<IValue_light_profile>(val);
                val = m_module.get_value_factory()->create_light_profile(
                    lval->get_type(),
                    abs_url.c_str(),
                    lval->get_tag_value(),
                    lval->get_tag_version());
            }
            break;
        case IValue::VK_BSDF_MEASUREMENT:
            {
                IValue_bsdf_measurement const *bval = cast<IValue_bsdf_measurement>(val);
                val = m_module.get_value_factory()->create_bsdf_measurement(
                    bval->get_type(),
                    abs_url.c_str(),
                    bval->get_tag_value(),
                    bval->get_tag_version());
            }
            break;
        default:
            MDL_ASSERT(!"unexpected resource type");
            break;
        }
        lit->set_value(val);
    }
}

// Constructor.
Error_params::Error_params(
    Analysis const &ana)
: m_alloc(ana.get_allocator())
, m_args(m_alloc)
, m_possible_match(NULL)
{
}

// Constructor.
Error_params::Error_params(IAllocator *alloc)
: m_alloc(alloc)
, m_args(alloc)
, m_possible_match(NULL)
{
}

// -------------------------------- AST checker --------------------------------

// Constructor.
AST_checker::AST_checker(IAllocator *alloc)
: Base()
, m_stmts(0, Stmt_set::hasher(), Stmt_set::key_equal(), alloc)
, m_exprs(0, Expr_set::hasher(), Expr_set::key_equal(), alloc)
, m_decls(0, Decl_set::hasher(), Decl_set::key_equal(), alloc)
, m_errors(false)
{
}

// Checker, asserts on failure.
bool AST_checker::check_module(IModule const *module)
{
    Module const *mod = impl_cast<Module>(module);

    AST_checker checker(mod->get_allocator());

    checker.visit(module);

    return !checker.m_errors;
}

void AST_checker::post_visit(IStatement *stmt)
{
    if (!m_stmts.insert(stmt).second) {
        // already visited, DAG detected
        m_errors = true;
        MDL_ASSERT(!"DAG detected in AST, statement reused");
    }
}

IExpression *AST_checker::post_visit(IExpression *expr)
{
    if (!m_exprs.insert(expr).second) {
        // already visited, DAG detected
        m_errors = true;
        MDL_ASSERT(!"DAG detected in AST, expression reused");
    }
    return expr;
}

void AST_checker::post_visit(IDeclaration *decl)
{
    if (!m_decls.insert(decl).second) {
        // already visited, DAG detected
        m_errors = true;
        MDL_ASSERT(!"DAG detected in AST, declaration reused");
    }
}

}  // mdl
}  // mi

