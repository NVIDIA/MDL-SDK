/******************************************************************************
 * Copyright (c) 2016-2023, NVIDIA CORPORATION. All rights reserved.
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

#include "mdl/compiler/compilercore/compilercore_malloc_allocator.h"

#include "compiler_glsl_analysis.h"
#include "compiler_glsl_builtins.h"
#include "compiler_glsl_compiler.h"
#include "compiler_glsl_compilation_unit.h"
#include "compiler_glsl_definitions.h"
#include "compiler_glsl_errors.h"
#include "compiler_glsl_symbols.h"

namespace mi {
namespace mdl {
namespace glsl {

// Constructor.
Analysis::Analysis(
    Compiler         &compiler,
    Compilation_unit &unit)
: m_builder(unit.get_allocator())
, m_compiler(compiler)
, m_unit(unit)
, m_st(unit.get_symbol_table())
, m_tc(unit.get_type_factory())
, m_def_tab(unit.get_definition_table())
, m_last_msg_idx(-1)
, m_string_buf(m_builder.create<Buffer_output_stream>(unit.get_allocator()))
, m_printer(m_builder.create<Printer>(unit.get_allocator(), m_string_buf.get()))
, m_disabled_warnings()
, m_warnings_are_errors()
, m_all_warnings_are_errors(false)
, m_strict(unit.get_glslang_context().is_strict())
{
}

// Format a message.
string Analysis::format_msg(
    GLSL_compiler_error code,
    Error_params const  &params)
{
    m_string_buf->clear();

    print_error_message(code, params, m_printer.get());
    return string(m_string_buf->get_data(), m_builder.get_allocator());
}

// Creates a new error.
void Analysis::error(
    GLSL_compiler_error code,
    Err_location const  &loc,
    Error_params const  &params)
{
    Messages &msgs = m_unit.get_messages();

    string msg(format_msg(code, params));
    m_last_msg_idx = msgs.add_error_message(
        code, loc.get_location(), msg.c_str());
}

// Creates a new warning.
void Analysis::warning(
    GLSL_compiler_error code,
    Err_location const  &loc,
    Error_params const  &params)
{
    bool marked_as_error = m_warnings_are_errors.test_bit(code);

    if (!marked_as_error && m_disabled_warnings.test_bit(code)) {
        // suppress this warning
        return;
    }

    bool is_error = marked_as_error || m_all_warnings_are_errors;

    Messages &msgs = m_unit.get_messages();

    string msg(format_msg(code, params));
    m_last_msg_idx = is_error ?
        msgs.add_error_message(code, loc.get_location(), msg.c_str()) :
        msgs.add_warning_message(code, loc.get_location(), msg.c_str());
}

// Add a note to the last error.
void Analysis::add_note(
    GLSL_compiler_error code,
    Err_location const  &loc,
    Error_params const  &params)
{
    Messages &msgs = m_unit.get_messages();

    string msg(format_msg(code, params));
    msgs.add_note(
        m_last_msg_idx,
        Message::MS_INFO,
        code,
        loc.get_location(),
        msg.c_str());
}

// Add a compiler note for a previous definition.
void Analysis::add_prev_definition_note(Definition const *prev_def)
{
    if (prev_def->get_location() != NULL) {
        add_note(
            PREVIOUS_DEFINITION,
            prev_def,
            Error_params(*this)
                .add_signature(prev_def));
    }
}

// Issue an error for some previously defined entity.
void Analysis::err_redeclaration(
    Definition::Kind    kind,
    Definition const    *def,
    Err_location const  &loc,
    GLSL_compiler_error err)
{
    if (is_error(def)) {
        // already error
        return;
    }

    if (kind != def->get_kind()) {
        err = REDECLARATION_OF_DIFFERENT_KIND;
    }

    error(
        err,
        loc,
        Error_params(*this).add_signature(def));
    add_prev_definition_note(def);
}

// Issue a warning if some previously defined entity is shadowed.
void Analysis::warn_shadow(
    Definition const    *def,
    Location const      &loc,
    GLSL_compiler_error warn)
{
    warning(
        warn,
        loc,
        Error_params(*this).add_signature(def));
    add_prev_definition_note(def);
}

// Get the one and only "error definition" of the processed compilation unit.
Definition *Analysis::get_error_definition() const
{
    Symbol     *err_sym      = m_st.get_error_symbol();
    Scope      *predef_scope = m_def_tab.get_predef_scope();
    Definition *def          = predef_scope->find_definition_in_scope(err_sym);

    MDL_ASSERT(def != NULL && "Cannot find the error definition");
    return def;
}

// Find the definition for a name.
Definition *Analysis::find_definition_for_name(Name *name)
{
    Symbol     *sym = name->get_symbol();
    Definition *def = NULL;

    bool error_reported = false;
    def = m_def_tab.get_definition(sym);
    if (def == NULL) {
        if (sym->get_id() == Symbol::SYM_ERROR) {
            // parse errors on names are expressed by error symbols
            error_reported = true;
        }
        if (!error_reported) {
            // FIXME: NYI
#if 0
            if (m_in_select != NULL) {
                if (!is<Type_error>(m_in_select)) {
                    error(
                        UNKNOWN_MEMBER,
                        name->get_location(),
                        Error_params(*this).add(m_in_select).add(sym));
                }
            } else
#endif
            {
                error(
                    UNKNOWN_IDENTIFIER,
                    name->get_location(),
                    Error_params(*this).add(sym));
            }
        }

        def = get_error_definition();
    }
    return def;
}

// Apply a type qualifier to a given type.
Type *Analysis::qualify_type(Type_qualifier *qual, Type *type)
{
    // FIXME: NYI

    Type_qualifier::Storage_qualifiers sq = qual->get_storage_qualifiers();

    if (sq & SQ_CONST) {
        type = m_tc.decorate_type(type, Type::MK_CONST);
    }

    // in/out ignored, do we need an error here?

    if (sq & SQ_ATTRIBUTE) {
        // FIXME: NYI
    }

    if (sq & SQ_UNIFORM) {
        type = m_tc.decorate_type(type, Type::MK_UNIFORM);
    }

    if (sq & SQ_VARYING) {
        type = m_tc.decorate_type(type, Type::MK_VARYING);
    }

    if (sq & SQ_BUFFER) {
        // FIXME: NYI
    }

    if (sq & SQ_SHARED) {
        // FIXME: NYI
    }

    if (sq & (SQ_CENTROID | SQ_SAMPLE | SQ_PATCH)) {
        // FIXME: NYI
    }

    if (sq & SQ_COHERENT) {
        // FIXME: NYI
    }

    if (sq & SQ_VOLATILE) {
        // FIXME: NYI
    }
    if (sq & SQ_RESTRICT) {
        // FIXME: NYI
    }
    if (sq & SQ_READONLY) {
        // FIXME: NYI
    }
    if (sq & SQ_WRITEONLY) {
        // FIXME: NYI
    }

    // layout qualifier ignored for the type

    Precision_qualifier pq = qual->get_precision_qualifier();
    switch (pq) {
    case PQ_NONE:
        break;
    case PQ_HIGHP:
        type = m_tc.decorate_type(type, Type::MK_HIGHP);
        break;
    case PQ_MEDIUMP:
        type = m_tc.decorate_type(type, Type::MK_MEDIUMP);
        break;
    case PQ_LOWP:
        type = m_tc.decorate_type(type, Type::MK_LOWP);
        break;
    }

    // FIXME: interpolation qualifier ignored for now

    return type;
}

// Apply array specifiers to a type.
Type *Analysis::apply_array_specifiers(Type *type, Array_specifiers const &arrs)
{
    for (Array_specifiers::const_iterator it(arrs.begin()), end(arrs.end()); it != end; ++it) {
        Array_specifier const &spec = *it;
        Expr const *size = spec.get_size();

        // fold it
        Value *val  = size->fold(m_unit.get_value_factory());
        Value *ival = val->convert(m_unit.get_value_factory(), m_tc.int_type);

        size_t a_size      = 0;
        bool   have_a_size = false;

        if (Value_int_32 *i32v = as<Value_int_32>(ival)) {
            have_a_size = true;
            int32_t s = i32v->get_value();
            if (s >= 1) {
                a_size = size_t(s);
            }
        } else if (Value_int_64 *i64v = as<Value_int_64>(ival)) {
            have_a_size = true;
            int64_t s = i64v->get_value();
            if (s >= 1) {
                a_size = size_t(s);
            }
        }
        if (have_a_size) {
            if (a_size >= 1) {
                type = m_tc.get_array(type, a_size);
            } else {
                // FIXME: error
                GLSL_ASSERT(!"NYI");
                type = m_tc.error_type;
            }
        } else {
            // FIXME: report error
            GLSL_ASSERT(!"NYI");
            type = m_tc.error_type;
        }
    }
    return type;
}

// ---------------------------- Name end type analysis ----------------------------

// Constructor.
NT_analysis::NT_analysis(
    Compiler         &compiler,
    Compilation_unit &unit)
: Base(compiler, unit)
{
}

// Returns the definition of a symbol at the at a given scope.
Definition *NT_analysis::get_definition_at_scope(
    Symbol *sym,
    Scope  *scope) const
{
    if (Definition *def = m_def_tab.get_definition(sym)) {
        if (def->get_def_scope() == scope) {
            return def;
        }
    }
    return NULL;
}

// Returns the definition of a symbol at the current scope only.
Definition *NT_analysis::get_definition_at_scope(Symbol *sym) const
{
    return get_definition_at_scope(sym, m_def_tab.get_curr_scope());
}

// Return the Type from a IType_name, handling errors if the IType_name
// does not name a type.
Type *NT_analysis::as_type(Type_name *type_name)
{
    Type *type = type_name->get_type();
    if (type == NULL) {
        // typename does NOT name a type, but an entity
        error(
            NOT_A_TYPE_NAME,
            type_name->get_location(),
            Error_params(*this)
                .add(type_name->get_name()->get_symbol()));

        // must be a type here
        type = m_tc.error_type;
        type_name->set_type(type);
    }
    return type;
}

// end of a name
void NT_analysis::post_visit(Name *name)
{
    Definition *def = find_definition_for_name(name);
    name->set_definition(def);
}

// start of a type name
bool NT_analysis::pre_visit(Type_name *tn)
{
    Type_qualifier *tq  = &tn->get_qualifier();
    Definition     *def = NULL;

    if (Name *name = tn->get_name()) {
        visit(name);
        def = name->get_definition();
    } else if (Declaration *sdecl = tn->get_struct_decl()) {
        visit(sdecl);
        def = sdecl->get_definition();
    }

    Array_specifiers &as = tn->get_array_specifiers();
    for (Array_specifiers::iterator it(as.begin()), end(as.end()); it != end; ++it) {
        Array_specifier *spec = it;

        visit(spec);
    }

    Type *type = def->get_type();
    if (is<Type_error>(type)) {
        // error definitions could be types, so set one
        tn->set_type(type);
    } else if (def->get_kind() == Definition::DK_TYPE) {
        type = qualify_type(tq, type);

        // check for arrays
        if (tn->is_array()) {
            // FIXME: NYI
            GLSL_ASSERT(!"NYI");
        }
        // set the type, so it will be recognized as a type
        tn->set_type(type);
    }

    // do not visit children anymore
    return false;
}

// start of compound statement
bool NT_analysis::pre_visit(Stmt_compound *block)
{
    m_def_tab.enter_scope(NULL);
    return true;
}

// end of compound statement
void NT_analysis::post_visit(Stmt_compound *block)
{
    Scope *scope = m_def_tab.get_curr_scope();
    m_def_tab.leave_scope();

    if (scope->is_empty()) {
        // drop it to save some space
        m_def_tab.remove_empty_scope(scope);
    }
}

// start of a for statement
bool NT_analysis::pre_visit(Stmt_for *for_stmt)
{
    Stmt *init = for_stmt->get_init();
    if (init != NULL && is<Stmt_decl>(init)) {
        // this for statement creates a scope
        m_def_tab.enter_scope(NULL);
    }
    return true;
}

// end of a for statement
void NT_analysis::post_visit(Stmt_for *for_stmt)
{
    Stmt const *init = for_stmt->get_init();
    if (init != NULL && is<Stmt_decl>(init)) {
        // this for statement creates a scope
        m_def_tab.leave_scope();
    }

    // FIXME: check condition
}

// End of a invalid declaration.
void NT_analysis::post_visit(Declaration_invalid *decl)
{
    // error already reported
    decl->set_definition(get_error_definition());
}

// Start of a struct declaration.
bool NT_analysis::pre_visit(Declaration_struct *sdecl)
{
    Name       *name     = sdecl->get_name();
    Definition *type_def = get_error_definition();

    sdecl->set_definition(type_def);

    Symbol *s_sym  = NULL;
    Type   *s_type = m_tc.error_type;

    if (name != NULL) {
        s_sym = name->get_symbol();

        type_def = get_definition_at_scope(s_sym);
        if (type_def) {
            err_redeclaration(
                Definition::DK_TYPE, type_def, sdecl->get_location(), TYPE_REDECLARATION);
            type_def = get_error_definition();
        } else {
            Def_type *def = m_def_tab.enter_type_definition(
                s_sym, s_type, &sdecl->get_location());
            def->set_declaration(sdecl);
            type_def = def;

#if 0
            // cannot create instances of this type until it is completed
            def->set_flag(Definition::DEF_IS_INCOMPLETE);
#endif
            sdecl->set_definition(def);
        }
    } else {
        // anonymous struct
        s_sym = m_st.get_anonymous_symbol();
        Def_type *def = m_def_tab.enter_type_definition(
            s_sym, s_type, &sdecl->get_location());
        def->set_declaration(sdecl);
        type_def = def;
        sdecl->set_definition(def);
    }

    // create a new type scope
    bool has_error = false;
    {
        Definition_table::Scope_enter scope(m_def_tab, type_def);
        Scope *type_scope = m_def_tab.get_curr_scope();

        vector<Type_struct::Field>::Type fields(get_allocator());
        for (Declaration_struct::iterator it(sdecl->begin()), end(sdecl->end()); it != end; ++it) {
            Declaration *decl = it;
            visit(decl);

            if (!is<Declaration_field>(decl)) {
                has_error = true;
                continue;
            }

            Declaration_field *field = cast<Declaration_field>(decl);

            Type_name *tn          = field->get_type_name();
            Type      *fields_type = tn->get_type();

            for (Declaration_field::const_iterator fit(field->begin()), fend(field->end());
                fit != fend;
                ++fit)
            {
                Field_declarator const &fd = *fit;

                Name const *f_name = fd.get_name();
                Symbol     *f_sym  = f_name->get_symbol();

                Array_specifiers const &as = fd.get_array_specifiers();

                Type *f_type = apply_array_specifiers(fields_type, as);

                fields.push_back(Type_struct::Field(f_type, f_sym));
            }
        }

        if (!has_error) {
            // all went fine, create the struct type
            s_type = m_tc.get_struct(fields, s_sym);
        }

        if (Def_type *def = as<Def_type>(type_def)) {
            def->set_type(s_type);
            def->set_own_scope(type_scope);
        }
    }

    // do not visit children
    return false;
}

// Start of a variable declaration.
bool NT_analysis::pre_visit(Declaration_variable *vdecl)
{
    Type_name *tname = vdecl->get_type_name();
    visit(tname);

    Type *base_type = as_type(tname);

    for (Declaration_variable::iterator it(vdecl->begin()), end(vdecl->end()); it != end; ++it) {
        Init_declarator        *idecl  = it;
        Name                   *v_name = idecl->get_name();
        Symbol                 *v_sym  = v_name->get_symbol();
        Array_specifiers const &arrs   = idecl->get_array_specifiers();
        Type                   *v_type = apply_array_specifiers(base_type, arrs);

        Definition *v_def = m_def_tab.get_definition(v_sym);
        if (v_def != NULL) {
            if (v_def->get_def_scope() == m_def_tab.get_curr_scope()) {
                if (v_def->get_kind() == Definition::DK_PARAMETER) {
                    // suppress "redeclaration as different kind ..." for params
                    // by saying that we declaring another param
                    err_redeclaration(
                        Definition::DK_PARAMETER,
                        v_def,
                        v_name->get_location(),
                        PARAMETER_REDECLARATION);
                } else {
                    // normal case
                    err_redeclaration(
                        Definition::DK_VARIABLE,
                        v_def,
                        v_name->get_location(),
                        ENT_REDECLARATION);
                }
                v_def = get_error_definition();
            } else {
                // warn if we shadow a parameter
                if (v_def->get_kind() == Definition::DK_PARAMETER) {
                    warn_shadow(v_def, v_name->get_location(), PARAMETER_SHADOWED);
                }
                v_def = NULL;
            }
        }

        if (v_def == NULL) {
            Def_variable *d = m_def_tab.enter_variable_definition(
                v_sym, v_type, &v_name->get_location());

            d->set_declaration(vdecl);
            v_def = d;
        }
        v_name->set_definition(v_def);

        Expr *v_init = idecl->get_initializer();
        if (v_init != NULL) {
            // FIXME:
//            Definition::Scope_flag scope(*v_def, Definition::DEF_IS_INCOMPLETE);
            visit(v_init);

            if (!is_error(v_def)) {
                // FIXME: check init type
            }
        }
    }

    // don't visit children anymore
    return false;
}

// Start of a function.
bool NT_analysis::pre_visit(Declaration_function *fdecl)
{
    Name   *f_name = fdecl->get_identifier();
    Symbol *f_sym  = f_name->get_symbol();

    bool       has_error = false;
    Definition *def    = get_definition_at_scope(f_sym);
    if (def != NULL && def->get_kind() != Definition::DK_FUNCTION) {
        err_redeclaration(
            Definition::DK_FUNCTION,
            def,
            fdecl->get_location(),
            ENT_REDECLARATION);
        has_error = true;
    }

    // for now, set the error definition, update it later
    def = get_error_definition();
    f_name->set_definition(def);
    fdecl->set_definition(def);

    Type_name *ret_name = fdecl->get_ret_type();
    visit(ret_name);

    // FIXME: check forbidden return type

    Type *ret_type = as_type(ret_name);
    if (is<Type_error>(ret_type)) {
        has_error = true;
    }

    // create new scope for the parameters and the body
    Scope         *f_scope = NULL;
    Type_function *f_type  = NULL;

    {
        Definition_table::Scope_enter scope(m_def_tab, def);

        typedef Type_function::Parameter Parameter;
        VLA<Parameter> params(get_allocator(), fdecl->get_param_count());

        f_scope = m_def_tab.get_curr_scope();
        size_t i = 0;
        for (Declaration_function::iterator it(fdecl->begin()), end(fdecl->end()); it != end; ++it)
        {
            Declaration *p = it;
            visit(p);

            if (!is<Declaration_param>(p)) {
                // error already reported by the parser
                continue;
            }
            Declaration_param *param = cast<Declaration_param>(p);

            Type_name                          *tn = param->get_type_name();
            Type_qualifier const               &tq = tn->get_qualifier();
            Type_qualifier::Storage_qualifiers sq  = tq.get_storage_qualifiers();

            Parameter::Modifier mod = Parameter::PM_IN;

            switch (sq & SQ_INOUT) {
            case SQ_OUT:
                mod = Parameter::PM_OUT;
                break;
            case SQ_INOUT:
                mod = Parameter::PM_INOUT;
                break;
            default:
                mod = Parameter::PM_IN;
                break;
            }

            Name   *p_name = param->get_name();
            Type   *p_type = NULL;

            if (p_name != NULL) {
                p_type = p_name->get_type();
            } else {
                // nameless parameter
                p_type = param->get_type_name()->get_type();
            }

            // FIXME: check for forbidden types

            params[i] = Parameter(p_type, mod);

            if (is<Type_error>(p_type)) {
                has_error = true;
            }

            if (Expr *init_expr = param->get_default_argument()) {
                // FIXME: handle default
                (void)init_expr;
            }
        }

        // create the function type
        f_type = m_tc.get_function(ret_type, params);

        if (!has_error) {
            // create a new definition for this function
            Def_function *f_def = m_def_tab.enter_function_definition(
                f_sym, f_type, Def_function::DS_UNKNOWN, &fdecl->get_location());
            f_def->set_own_scope(f_scope);
            f_def->set_declaration(fdecl);
            f_name->set_definition(f_def);
            fdecl->set_definition(f_def);
            def = f_def;
        }

        if (Stmt *body = fdecl->get_body()) {
            visit(body);
        }
    }

    // don't visit children anymore
    return false;
}

// Run the name and type analysis on the current unit.
void NT_analysis::run(
    Compiler         &compiler,
    Compilation_unit &unit)
{
    NT_analysis ana(compiler, unit);

    // clear the table here, this might be a re-analyze
    ana.m_def_tab.clear();

    // enter the outer scope
    ana.m_def_tab.reopen_scope(ana.m_def_tab.get_predef_scope());

    // enter all predefined entities
    enter_predefined_entities(unit, ana.m_tc);

    // and create the error definition here, so the set of
    // predefined entities is constant
    Symbol *err_sym = ana.m_st.get_error_symbol();
    ana.m_def_tab.enter_error(err_sym, ana.m_tc.error_type);

    // enter the global scope
    ana.m_def_tab.reopen_scope(ana.m_def_tab.get_global_scope());

    ana.visit(&unit);

    // leave global scope
    ana.m_def_tab.leave_scope();

    // leave outer scope
    ana.m_def_tab.leave_scope();
}

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
/// For debugging. If true, dump_ast() prints locations.
static bool g_location_enabled = false;

/// Debug helper: Enables the dumping of locations.
///
/// \param enable  if true, following dump_ast() calls
///                will include locations
void dump_enable_locations(bool enable)
{
    g_location_enabled = enable;
}

// Debug helper : Dump the AST of a compilation unit.
void dump_ast(ICompilation_unit const *iunit)
{
    if (iunit == NULL) {
        return;
    }

    Compilation_unit const *unit = impl_cast<Compilation_unit>(iunit);
    Alloc alloc(unit->get_allocator());

    mi::base::Handle<Debug_Output_stream> dbg(alloc.dbg());
    mi::base::Handle<Printer>             printer(alloc.prt(dbg.get()));

    printer->enable_locations(g_location_enabled);
    printer->print(unit);
}

// Debug helper: Dump the AST of a declaration.
void dump_ast(Declaration const *decl)
{
    if (decl == NULL) {
        return;
    }

    mi::base::Handle<IAllocator> mallocator(mdl::MallocAllocator::create_instance());
    Alloc alloc(mallocator.get());

    mi::base::Handle<Debug_Output_stream> dbg(alloc.dbg());
    mi::base::Handle<Printer>             printer(alloc.prt(dbg.get()));

    printer->enable_locations(g_location_enabled);
    printer->print_decl(decl, /*is_toplevel=*/ true);
}

// Debug helper: Dump the AST of an expression.
void dump_expr(Expr const *expr)
{
    if (expr == NULL) {
        return;
    }

    mi::base::Handle<IAllocator> mallocator(MallocAllocator::create_instance());
    Alloc alloc(mallocator.get());

    mi::base::Handle<Debug_Output_stream> dbg(alloc.dbg());
    mi::base::Handle<Printer>             printer(alloc.prt(dbg.get()));

    printer->enable_locations(g_location_enabled);
    printer->print(expr);
    printer->print("\n");
}

// Debug helper: Dump a definition.
void dump_def(Definition const *def)
{
    mi::base::Handle<IAllocator> mallocator(MallocAllocator::create_instance());
    Alloc alloc(mallocator.get());

    mi::base::Handle<Debug_Output_stream> dbg(alloc.dbg());
    mi::base::Handle<Printer>             printer(alloc.prt(dbg.get()));

    printer->print(def);
}

}  // glsl
}  // mdl
}  // mi

