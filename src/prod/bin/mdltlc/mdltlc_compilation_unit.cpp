/******************************************************************************
 * Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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

#include <ios>
#include <sstream>

#include <mdl/compiler/compilercore/compilercore_memory_arena.h>
#include <mdl/compiler/compilercore/compilercore_streams.h>
#include <mdl/compiler/compilercore/compilercore_tools.h>
#include <mdl/compiler/compilercore/compilercore_file_utils.h>
#include <mdl/compiler/compilercore/compilercore_wchar_support.h>
#include <mdl/compiler/compilercore/compilercore_mdl.h>

#include <mi/mdl/mdl_distiller_rules.h>

#include <mi/mdl/mdl_types.h>

#include "mdltlc_compilation_unit.h"
#include "mdltlc_expr_walker.h"
#include "mdltlc_analysis.h"
#include "mdltlc_codegen.h"

#include "Scanner.h"
#include "Parser.h"

// We just pull in this one source file for the Node_types instead of
// linking the distiller library and many other dependencies.
#include <mdl/codegenerators/generator_dag/generator_dag_distiller_node_types.cpp>

void dump_vars(Var_set const &vars) {
    printf("vars:\n");
    for (Var_set::const_iterator it(vars.begin()), end(vars.end());
         it != end; ++it) {
        printf("  %s\n", (*it)->get_name());
    }
}

/// Return a pointer to the filename portion of the given path, or the
/// path itself if it does not have a directory component.
char const *basename(char const *filename) {
    char const *slash_ptr = strrchr(filename, '/');
    char const *backslash_ptr = strrchr(filename, '\\');

    return slash_ptr ? slash_ptr + 1 :
        (backslash_ptr ? backslash_ptr + 1 : filename);
}

/// Basic implementation of the Errors interface in the Coco/R
/// scanner.
class Syntax_error : public Errors {
private:
    mi::mdl::Memory_arena *m_global_arena;

    mi::mdl::vector<Message*>::Type &m_messages;

    const char* m_filename;

    unsigned m_error_count;
    unsigned m_warning_count;

public:
    /// Report a syntax error at given line, column pair.
    ///
    /// \param la      the current look-ahead token
    /// \param s       the human readable error message
    void Error(Token const* la, wchar_t const* s) {
        mi::mdl::string tmp(m_global_arena->get_allocator());
        mi::mdl::utf16_to_utf8(tmp, s);

        m_error_count++;

        mi::mdl::Arena_builder builder(*m_global_arena);
        Message *m = builder.create<Message>(m_global_arena, Message::Severity::SEV_ERROR,
                                             m_filename, la->line, la->col, tmp.c_str());
        m_messages.push_back(m);
    }

    /// Report a syntax warning at given line, column pair.
    ///
    /// \param line  the start line of the syntax error
    /// \param col   the start column of the syntax error
    /// \param s     the human readable error message
    void Warning(int line, int col, wchar_t const* s) {
        mi::mdl::string tmp(m_global_arena->get_allocator());
        mi::mdl::utf16_to_utf8(tmp, s);

        m_warning_count++;

        mi::mdl::Arena_builder builder(*m_global_arena);
        Message *m = builder.create<Message>(m_global_arena, Message::Severity::SEV_WARNING,
                                             m_filename, line, col, tmp.c_str());
        m_messages.push_back(m);
    }

    /// Construct a Syntax_error object.
    ///
    /// \param alloc    Allocator to use for temporary storage allocations.
    /// \param filename Use this filename in generated error messages.
    ///
    explicit Syntax_error(//mi::mdl::IAllocator * alloc,
                          mi::mdl::Memory_arena *global_arena,
                          mi::mdl::vector<Message*>::Type &messages,
                          const char* filename)
        : Errors()
        , m_global_arena(global_arena)
        , m_messages(messages)
        , m_filename(filename)
        , m_error_count(0)
        , m_warning_count(0)
        {
        }

    unsigned error_count() {
        return m_error_count;
    }

    unsigned warning_count() {
        return m_warning_count;
    }
};

// Constructor.
Compilation_unit::Compilation_unit(
    mi::mdl::IAllocator *alloc,
    mi::mdl::Memory_arena *global_arena,
    mi::mdl::IMDL *imdl,
    Symbol_table *symbol_table,
    char const *file_name,
    Compiler_options const *comp_options,
    Message_list *messages,
    Builtin_type_map *builtins)
    : Base(alloc)
    , m_global_arena(*global_arena)
    , m_arena(alloc)
    , m_arena_builder(m_arena)
    , m_imdl(imdl)
    , m_filename(Arena_strdup(m_arena, file_name))
    , m_filename_only(Arena_strdup(m_arena, basename(m_filename)))
    , m_comp_options(comp_options)
    , m_symbol_table(symbol_table)
    , m_type_factory(m_arena, *m_symbol_table)
    , m_value_factory(&m_arena, this, m_type_factory)
    , m_expr_factory(m_arena, this, m_value_factory)
    , m_rule_factory(m_arena)
    , m_rulesets()
    , m_messages(messages)
    , m_builtins(builtins)
    , m_error_count(0)
    , m_warning_count(0)
    , m_api_class("IDistiller_plugin_api")
    , m_rule_matcher_class("IRule_matcher")
    , m_error_type(m_type_factory.get_error())
    , m_attr_counter(0)
    , m_attribute_env(m_arena, Environment::Kind::ENV_ATTRIBUTE, nullptr)
{
}

// Get the absolute name of the file from which this unit was loaded.
char const *Compilation_unit::get_filename() const
{
    return m_filename;
}

// Get the expression factory.
Expr_factory &Compilation_unit::get_expression_factory()
{
    return m_expr_factory;
}

// Get the type factory.
Type_factory &Compilation_unit::get_type_factory()
{
    return m_type_factory;
}

// Get the value factory.
Value_factory &Compilation_unit::get_value_factory()
{
    return m_value_factory;
}

// Get the rule factory.
Rule_factory &Compilation_unit::get_rule_factory()
{
    return m_rule_factory;
}

// Get the symbol table of this compilation unit.
Symbol_table &Compilation_unit::get_symbol_table()
{
    return *m_symbol_table;
}

void Compilation_unit::reset_attr_counter() {
    m_attr_counter = 0;
}

int Compilation_unit::next_attr_counter() {
    int r = m_attr_counter;
    m_attr_counter++;
    return r;
}

/// Compile the mdltl file from `input_stream` according to the
/// compiler options passed to the constructor.
unsigned Compilation_unit::compile(mi::mdl::IInput_stream *input_stream) {

    Syntax_error error(&m_global_arena, *m_messages, m_filename);

    Scanner scanner(m_arena.get_allocator(), &error, input_stream);
    Parser parser(&scanner, &error);

    parser.set_compilation_unit(this);

    parser.Parse();

    if (error.error_count() > 0) {
        return error.error_count();
    }

    m_error_count += error.error_count();

    {
        Environment builtin_env(m_arena, Environment::Kind::ENV_BUILTIN, nullptr);

        init_builtin_env(builtin_env);
        process_imports(builtin_env);

        if (m_error_count > 0)
            return m_error_count;

        type_check(builtin_env);
    }

    if (m_error_count == 0) {
        if (m_comp_options->get_normalize_mixers()) {
            normalize_mixers();
        }
    }

    if (m_error_count == 0)
        lint_rules();

    calculate_rule_ids();

    if (m_comp_options->get_verbosity() >= 1) {
        std::stringstream sstr;
        pp::Pretty_print p(m_arena, sstr);
//        p.set_flags(pp::Pretty_print::Flags::PRINT_RETURN_TYPES);

        if (m_comp_options->get_verbosity() >= 3)
            p.set_flags(pp::Pretty_print::Flags::PRINT_ATTRIBUTE_TYPES);

        if (m_rulesets.size() == 0) {
            printf("[info] No rulesets defined.\n");
        } else {
            printf("[info] Defined rulesets:\n");
            for (mi::mdl::Ast_list<Ruleset>::iterator it(m_rulesets.begin()),
                     end(m_rulesets.end());
                 it != end;
                 ++it) {
                if (m_comp_options->get_verbosity() >= 2) {
                    it->pp(p);
                } else {
                    printf("[info]   %s\n", it->get_name());
                }
            }
        }
      std::cout << sstr.str() << std::endl;
    }

    if (m_comp_options->get_generate()) {
        if (m_error_count == 0) {
            output();
        }
    }

    return m_error_count;
}

void Compilation_unit::normalize_mixers() {
    for (mi::mdl::Ast_list<Ruleset>::iterator it(m_rulesets.begin()),
             end(m_rulesets.end());
         it != end; ++it) {
        for (mi::mdl::Ast_list<Rule>::iterator rit(it->get_rules().begin()),
                 rend(it->get_rules().end());
             rit != rend; ++rit) {
            Expr *lhs = rit->get_lhs();
            rit->set_lhs(normalize_mixer_pattern(lhs));
        }
    }
}

bool more_general_pattern(Expr const *p1, Expr const *p2) {
    if (p1->get_kind() == Expr::EK_REFERENCE && p2->get_kind() != Expr::EK_REFERENCE) {
        return true;
    }

    if (p1->get_kind() != p2->get_kind()) {
        return false;
    }

    switch (p1->get_kind()) {
    case Expr::EK_REFERENCE:
        return true;

    case Expr::EK_CALL:
    {
        Expr_call const *call_1 = cast<Expr_call>(p1);
        Expr_call const *call_2 = cast<Expr_call>(p2);
        if (cast<Expr_ref>(call_1->get_callee())->get_name() !=
            cast<Expr_ref>(call_2->get_callee())->get_name()) {
            return false;
        }
        if (call_1->get_argument_count() != call_2->get_argument_count())
            return false;
        for (int i = 0; i < call_1->get_argument_count(); i++) {
            Expr const *arg1 = call_1->get_argument(i);
            Expr const *arg2 = call_2->get_argument(i);

            if (!more_general_pattern(arg1, arg2)) {
                return false;
            }
        }
        break;
    }
    default:
        MDL_ASSERT(!"Unexpected node type in more_general_pattern");
        break;
    }
    return true;
}

void Compilation_unit::lint_rules() {
    for (mi::mdl::Ast_list<Ruleset>::iterator it(m_rulesets.begin()),
             end(m_rulesets.end());
         it != end; ++it) {

        bool warned_unused = false;

        // Check for unused variables.
        for (mi::mdl::Ast_list<Rule>::iterator rit(it->get_rules().begin()),
                 rend(it->get_rules().end());
             rit != rend; ++rit) {

            Var_set defined(get_allocator());
            Var_set used(get_allocator());

            defined_vars(rit->get_lhs(), defined);

            Argument_list const &bindings = rit->get_bindings();
            for (Argument_list::const_iterator ait(bindings.begin()), aend(bindings.end());
                 ait != aend; ++ait) {

                Expr const *assign_expr = ait->get_expr();
                MDL_ASSERT(assign_expr->get_kind() == Expr::Kind::EK_BINARY);
                Expr_binary const *assign = cast<Expr_binary>(assign_expr);
                MDL_ASSERT(assign->get_operator() == Expr_binary::Operator::OK_ASSIGN);

                defined_vars(assign->get_left_argument(), defined);
                used_vars(assign->get_right_argument(), used);
            }

            lhs_used_vars(rit->get_lhs(), used);
            used_vars(rit->get_rhs(), used);
            Expr const *g_expr = rit->get_guard();
            if (g_expr) {
                MDL_ASSERT(g_expr->get_kind() == Expr::Kind::EK_UNARY);
                Expr_unary const *u = cast<Expr_unary>(g_expr);
                MDL_ASSERT((u->get_operator() == Expr_unary::Operator::OK_IF_GUARD) ||
                           (u->get_operator() == Expr_unary::Operator::OK_MAYBE_GUARD));
                used_vars(u->get_argument(), used);
            }

            Debug_out_list const &deb_outs = rit->get_debug_out();
            for (Debug_out_list::const_iterator ait(deb_outs.begin()), aend(deb_outs.end());
                ait != aend; ++ait) {
                used.insert(ait->get_symbol());
            }
            for (Var_set::const_iterator uit(defined.begin()), uend(defined.end());
                 uit != uend; ++uit) {

                Symbol const *def_sym = *uit;
                if (def_sym->get_name()[0] != '_' && used.find(def_sym) == used.end()) {
                    mi::mdl::string msg(m_arena.get_allocator());
                    msg = "unused variable: ";
                    msg += def_sym->get_name();
                    warning(rit->get_location(), msg.c_str());
                    if (!warned_unused) {
                        warned_unused = true;
                        hint(rit->get_location(),
                             "you can suppress warnings for unused variables by prefixing their names with '_'");
                    }
                }
            }
        }

        // Check for overlapping patterns, if requested.
        if (m_comp_options->get_warn_overlapping_patterns()) {
            for (mi::mdl::Ast_list<Rule>::iterator rit(it->get_rules().begin()),
                     rend(it->get_rules().end());
                 rit != rend; ++rit) {

                for (mi::mdl::Ast_list<Rule>::iterator rit2(it->get_rules().begin());
                     rit2 != rit; ++rit2) {

                    Rule *rule1 = rit2;
                    Rule *rule2 = rit;
                    if (!rule1->get_guard()) {
                        if (more_general_pattern(rule1->get_lhs(), rule2->get_lhs())) {
                            error(rule1->get_lhs()->get_location(),
                                  "rule pattern is more general than later pattern");
                            hint(rule2->get_lhs()->get_location(),
                                 "this is the more specific later pattern");
                        }
                    }
                }
            }
        }
    }
}

void Compilation_unit::calculate_rule_ids() {
    // For each rule set, generate methods and rule table.
    for (Ruleset_list::iterator it(m_rulesets.begin()), end(m_rulesets.end());
         it != end; ++it) {

        for (Rule_list::iterator rit(it->get_rules().begin()), rend(it->get_rules().end());
             rit != rend; ++rit) {
            rit->calc_hash(m_arena.get_allocator(), it->get_name());

            for (Rule_list::iterator rit2(it->get_rules().begin()), rend2(rit);
                 rit2 != rend2; ++rit2) {
                if (rit->get_uid() == rit2->get_uid()) {
                    warning(rit->get_location(),
                            "duplicate rule id - possible rule duplication");
                }
            }
        }
    }
}

/// Add a ruleset to the compilation unit. This is called by the parser.
void Compilation_unit::add_ruleset(Ruleset *ruleset)
{
    m_rulesets.push(ruleset);
}

/// Add a binding for `name` to `type` to the given environment.
///
/// If a binding for `name` to the same type already exists, the
/// environment is not changed.
void Compilation_unit::add_binding(Symbol const *name, Type *type, Environment &builtin_env) {
    Environment::Type_list *types = builtin_env.find(name);

    Type_function *this_tf = as<Type_function>(type);

    bool already_bound = false;

    if (types) {
        for (Environment::Type_list::iterator it(types->begin()), end(types->end());
             it != end; ++it) {
            Type *test_type = *it;

            // FIXME: Right now, we do only get selector information for
            // node types specially marked in the distiller. What we do
            // here is a hack: whenever we encounter a function type which
            // has assigned a selector (or node type), and a previous
            // binding does not have a selector/node type, we simply copy
            // it over the older binding. Also, the other way around.

            if (this_tf) {
                if (Type_function *that_tf = as<Type_function>(test_type)) {
                    if (this_tf->get_selector() && !that_tf->get_selector()) {
                        that_tf->set_selector(this_tf->get_selector());
                    }
                    if (that_tf->get_selector() && !this_tf->get_selector()) {
                        this_tf->set_selector(that_tf->get_selector());
                    }
                    if (this_tf->get_node_type() && !that_tf->get_node_type()) {
                        that_tf->set_node_type(this_tf->get_node_type());
                    }
                    if (that_tf->get_node_type() && !this_tf->get_node_type()) {
                        this_tf->set_node_type(that_tf->get_node_type());
                    }
                }
            }
            if (m_type_factory.types_equal(type, test_type)) {
                already_bound = true;
            }
        }
    }
    if (!already_bound) {
        builtin_env.bind(name, type);
    }
}

/// Return the type matching the builtin type named `type_name`.
Type *Compilation_unit::builtin_type_for(const char *type_name) {
    Type * t_bool = m_type_factory.get_bool();
    Type * t_int = m_type_factory.get_int();
    Type * t_float = m_type_factory.get_float();
    Type * t_double = m_type_factory.get_double();

    if (!type_name) {
        printf("[BUG] NULL type name in builtin_type_for\n");
        return nullptr;
    }
    if (!strcmp(type_name, "")) {
        printf("[BUG] empty type name in builtin_type_for\n");
        return nullptr;
    }
    if (!strcmp(type_name, "bool"))
        return t_bool;
    if (!strcmp(type_name, "bool2"))
        return m_type_factory.get_vector(2, t_bool);
    if (!strcmp(type_name, "bool3"))
        return m_type_factory.get_vector(3, t_bool);
    if (!strcmp(type_name, "bool4"))
        return m_type_factory.get_vector(4, t_bool);

    if (!strcmp(type_name, "color"))
        return m_type_factory.get_color();

    if (!strcmp(type_name, "int"))
        return m_type_factory.get_int();
    if (!strcmp(type_name, "int2"))
        return m_type_factory.get_vector(2, t_int);
    if (!strcmp(type_name, "int3"))
        return m_type_factory.get_vector(3, t_int);
    if (!strcmp(type_name, "int4"))
        return m_type_factory.get_vector(4, t_int);

    if (!strcmp(type_name, "float"))
        return m_type_factory.get_float();
    if (!strcmp(type_name, "float2"))
        return m_type_factory.get_vector(2, t_float);
    if (!strcmp(type_name, "float3"))
        return m_type_factory.get_vector(3, t_float);
    if (!strcmp(type_name, "float4"))
        return m_type_factory.get_vector(4, t_float);

    if (!strcmp(type_name, "double"))
        return m_type_factory.get_double();
    if (!strcmp(type_name, "double2"))
        return m_type_factory.get_vector(2, t_double);
    if (!strcmp(type_name, "double3"))
        return m_type_factory.get_vector(3, t_double);
    if (!strcmp(type_name, "double4"))
        return m_type_factory.get_vector(4, t_double);

    if (!strcmp(type_name, "string"))
        return m_type_factory.get_string();

    if (!strcmp(type_name, "float2x2"))
        return m_type_factory.get_matrix(2, m_type_factory.get_vector(2, t_float));
    if (!strcmp(type_name, "float2x3"))
        return m_type_factory.get_matrix(3, m_type_factory.get_vector(2, t_float));
    if (!strcmp(type_name, "float2x4"))
        return m_type_factory.get_matrix(4, m_type_factory.get_vector(2, t_float));
    if (!strcmp(type_name, "float3x2"))
        return m_type_factory.get_matrix(2, m_type_factory.get_vector(3, t_float));
    if (!strcmp(type_name, "float3x3"))
        return m_type_factory.get_matrix(3, m_type_factory.get_vector(3, t_float));
    if (!strcmp(type_name, "float3x4"))
        return m_type_factory.get_matrix(4, m_type_factory.get_vector(3, t_float));
    if (!strcmp(type_name, "float4x2"))
        return m_type_factory.get_matrix(2, m_type_factory.get_vector(4, t_float));
    if (!strcmp(type_name, "float4x3"))
        return m_type_factory.get_matrix(3, m_type_factory.get_vector(4, t_float));
    if (!strcmp(type_name, "float4x4"))
        return m_type_factory.get_matrix(4, m_type_factory.get_vector(4, t_float));

    if (!strcmp(type_name, "double2x2"))
        return m_type_factory.get_matrix(2, m_type_factory.get_vector(2, t_double));
    if (!strcmp(type_name, "double2x3"))
        return m_type_factory.get_matrix(3, m_type_factory.get_vector(2, t_double));
    if (!strcmp(type_name, "double2x4"))
        return m_type_factory.get_matrix(4, m_type_factory.get_vector(2, t_double));
    if (!strcmp(type_name, "double3x2"))
        return m_type_factory.get_matrix(2, m_type_factory.get_vector(3, t_double));
    if (!strcmp(type_name, "double3x3"))
        return m_type_factory.get_matrix(3, m_type_factory.get_vector(3, t_double));
    if (!strcmp(type_name, "double3x4"))
        return m_type_factory.get_matrix(4, m_type_factory.get_vector(3, t_double));
    if (!strcmp(type_name, "double4x2"))
        return m_type_factory.get_matrix(2, m_type_factory.get_vector(4, t_double));
    if (!strcmp(type_name, "double4x3"))
        return m_type_factory.get_matrix(3, m_type_factory.get_vector(4, t_double));
    if (!strcmp(type_name, "double4x4"))
        return m_type_factory.get_matrix(4, m_type_factory.get_vector(4, t_double));

    if (!strcmp(type_name, "light_profile"))
        return m_type_factory.get_light_profile();

    if (!strcmp(type_name, "bsdf"))
        return m_type_factory.get_bsdf();
    if (!strcmp(type_name, "edf"))
        return m_type_factory.get_edf();
    if (!strcmp(type_name, "vdf"))
        return m_type_factory.get_vdf();
    if (!strcmp(type_name, "hair_bsdf"))
        return m_type_factory.get_hair_bsdf();

    if (!strcmp(type_name, "texture_2d"))
        return m_type_factory.get_texture(Type_texture::Texture_kind::TK_2D);
    if (!strcmp(type_name, "texture_3d"))
        return m_type_factory.get_texture(Type_texture::Texture_kind::TK_3D);
    if (!strcmp(type_name, "texture_cube"))
        return m_type_factory.get_texture(Type_texture::Texture_kind::TK_CUBE);
    if (!strcmp(type_name, "texture_ptex"))
        return m_type_factory.get_texture(Type_texture::Texture_kind::TK_PTEX);

    if (!strcmp(type_name, "bsdf_measurement"))
        return m_type_factory.get_bsdf_measurement();
    if (!strcmp(type_name, "material_emission"))
        return m_type_factory.get_material_emission();
    if (!strcmp(type_name, "material_surface"))
        return m_type_factory.get_material_surface();
    if (!strcmp(type_name, "material_volume"))
        return m_type_factory.get_material_volume();
    if (!strcmp(type_name, "material_geometry"))
        return m_type_factory.get_material_geometry();
    if (!strcmp(type_name, "material"))
//        return m_type_factory.get_material_volume();
        return m_type_factory.get_material();

    // if (!strcmp(type_name, "tex_gamma_mode"))
    //     return m_type_factory.get_tex_gamma_mode();
    if (!strcmp(type_name, "intensity_mode"))
        return m_type_factory.create_enum(m_symbol_table->get_symbol("::df::intensity_mode"));
    if (!strcmp(type_name, "::df::intensity_mode"))
        return m_type_factory.create_enum(m_symbol_table->get_symbol("::df::intensity_mode"));
    if (!strcmp(type_name, "scatter_mode"))
        return m_type_factory.create_enum(m_symbol_table->get_symbol("::df::scatter_mode"));
    if (!strcmp(type_name, "::df::scatter_mode"))
        return m_type_factory.create_enum(m_symbol_table->get_symbol("::df::scatter_mode"));

    char const *brack = strstr(type_name, "[<N>]");
    if (brack) {
        if (!strncmp(type_name, "color", 5)) {
            return m_type_factory.get_array(m_type_factory.get_color());
        }
    }
    printf("[BUG] unknown type in builtin_type_for: %s\n", type_name);
    return nullptr;
}

/// Declare the functions that are built into the MDL compiler.
///
/// Note: these are automatically derived from the known definitions
/// (builtins) of the MDL compiler.
void Compilation_unit::declare_builtins(Environment &env) {
    int num_args = 0;

    (void) num_args;


#define BUILTIN_TYPE_BEGIN(typename, flags)                             \
    {                                                                   \
        Symbol *symbol = m_symbol_table->get_symbol(#typename);         \
        Type *builtin_type = builtin_type_for(#typename);               \
        MDL_ASSERT(builtin_type && "unknown builtin type " #typename);  \
        if (!builtin_type) {                                            \
            printf("[error]: unknown builtin type %s\n", #typename);    \
            builtin_type = m_error_type;                                \
        }

#define ARG0()                              num_args = 0;
#define ARG1(a1)                            num_args = 1; a1
#define ARG2(a1, a2)                        num_args = 2; a1 a2
#define ARG3(a1, a2, a3)                    num_args = 3; a1 a2 a3
#define ARG4(a1, a2, a3, a4)                num_args = 4; a1 a2 a3 a4
#define ARG5(a1, a2, a3, a4, a5)            num_args = 5; a1 a2 a3 a4 a5
#define ARG6(a1, a2, a3, a4, a5, a6)        num_args = 6; a1 a2 a3 a4 a5 a6
#define ARG7(a1, a2, a3, a4, a5, a6, a7)    num_args = 7; a1 a2 a3 a4 a5 a6 a7
#define ARG8(a1, a2, a3, a4, a5, a6, a7, a8) \
    num_args = 8; a1 a2 a3 a4 a5 a6 a7 a8
#define ARG9(a1, a2, a3, a4, a5, a6, a7, a8, a9) \
    num_args = 9; a1 a2 a3 a4 a5 a6 a7 a8 a9
#define ARG12(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12) \
    num_args = 12; a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12
#define ARG16(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16) \
    num_args = 16; a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 a16

#define ARG(type, name, arr)                                            \
            {                                                           \
                Type *arg_type = builtin_type_for(#type);               \
                if (arg_type) {                                         \
                    Type_list_elem *type_list_elem = m_arena_builder.create<Type_list_elem>(arg_type); \
                    constr_type->add_parameter(type_list_elem);         \
                }                                                       \
            }

#define DEFARG(type, name, arr, expr) \
    ARG(type, name, arr)

#define CONSTRUCTOR(kind, classname, args, sema, flags)                 \
        {                                                               \
                                \
            Type_function *constr_type = m_type_factory.create_function(builtin_type); \
            constr_type->set_semantics(mi::mdl::IDefinition::Semantics:: sema); \
            args                                                        \
            add_binding(symbol, constr_type, env);                      \
        }

#define BUILTIN_TYPE_END(typename)              \
    }

#include "mdl/compiler/compilercore/compilercore_known_defs.h"
}

/// Declare all functions that have been loaded from the standard
/// libraries.
void Compilation_unit::declare_stdlib(Environment &builtin_env) {
    for (Builtin_type_map::iterator it(m_builtins->begin()), end(m_builtins->end());
         it != end; ++it) {
        Symbol const *symbol = it->first;
        mi::mdl::IDefinition::Semantics sema = it->second.get_semantics();

        // FIXME: This is a hack to assign special semantics and a
        // bsdf return type to the local_normal function, which gets
        // special handling in the distiller.
        bool is_local_normal = false;
        const char *n = symbol->get_name();
        if (strlen(n) >= strlen("local_normal") &&
            !strcmp(n + (strlen(n) - strlen("local_normal")), "local_normal")) {
            is_local_normal = true;
        }

        if (is_local_normal) {
            Type_function *tf = m_type_factory.create_function(m_type_factory.get_bsdf());
            sema = static_cast<mi::mdl::IDefinition::Semantics>(mi::mdl::Distiller_extended_node_semantics::DS_DIST_LOCAL_NORMAL);
            tf->set_semantics(sema);
            tf->set_selector("mi::mdl::DS_DIST_LOCAL_NORMAL");
            Type_list_elem *tle = m_type_factory.create_type_list_elem(
                                      m_type_factory.get_float());
            tf->add_parameter(tle);
            tle = m_type_factory.create_type_list_elem(
                m_type_factory.get_vector(3, m_type_factory.get_float()));
            tf->add_parameter(tle);
            tf->set_node_type(mi::mdl::Node_types::static_type_from_idx(mi::mdl::local_normal));
            add_binding(symbol, tf, builtin_env);

            // Skip normal handling for local_normal.
            continue;
        }

        Type_ptr_list const &v = it->second.get_type_list();

        char const *selector = it->second.get_selector();
        mi::mdl::string selector_str(m_arena.get_allocator());
        if (!strcmp(selector, "::NONE::")) {
            selector_str = "mi::mdl::IDefinition::Semantics::";
            selector_str += get_semantics_name(sema);
        }

        // Add a function type for each overload.
        for (Type_ptr_list::const_iterator ti(v.begin()), tend(v.end());
             ti != tend; ++ti) {

            mi::mdl::IType const *mdl_type = *ti;
            Type *type = m_type_factory.import_type(mdl_type);

            if (is<Type_error>(type)) {
                printf("[error] could not import type for %s\n",
                       symbol->get_name());
                continue;
            }

            // In case of functions, assign semantics and selector strings.
            if (Type_function *tf = as<Type_function>(type)) {
                tf->set_semantics(sema);
                tf->set_selector(Arena_strdup(m_arena, selector_str.c_str()));
            }

            // Add the type of this overload to the builtin
            // environmen.
            add_binding(symbol, type, builtin_env);

            // In case of enums, add all variant names as global
            // constants typed as that enum.
            if (is<mi::mdl::IType_enum>(mdl_type)) {
                // For enums, add the values as global identifiers.
                mi::mdl::IType_enum const *te = as<mi::mdl::IType_enum>(mdl_type);

                for (int i = 0; i < te->get_value_count(); i++) {
                    mi::mdl::ISymbol const *sym;
                    int code;

                    if (te->get_value(i, sym, code)) {
                        add_binding(m_symbol_table->get_symbol(sym->get_name()),
                                    type, builtin_env);
                    } else {
                        // Ignore "impossible" error.
                    }
                }
            }

            // For structs, add a constructor function with all the
            // fields as parameters.
            if (is<mi::mdl::IType_struct>(mdl_type)) {
                mi::mdl::IType_struct const *ts = as<mi::mdl::IType_struct>(mdl_type);

                Type_function *tf = m_type_factory.create_function(type);

                for (int i = 0; i < ts->get_field_count(); i++) {
                    mi::mdl::IType const *param_type;
                    mi::mdl::ISymbol const *param_name;

                    ts->get_field(i, param_type, param_name);
                    Type *t = m_type_factory.import_type(param_type);

                    Type_list_elem *tle = m_type_factory.create_type_list_elem(t);
                    tf->add_parameter(tle);
                }
                add_binding(symbol, tf, builtin_env);
            }
        }
    }
}

/// Declare all the identifiers that the distiller defines as node
/// types.
void Compilation_unit::declare_dist_nodes(Environment &builtin_env) {

    for (int idx = 0; ; idx++) {
        bool error = false;
        mi::mdl::Node_type const *nt = mi::mdl::Node_types::static_type_from_idx(idx);
        // static_type_from_idx returns nullptr if the idx is larger
        // than the last supported node type index.
        if (!nt) {
            break;
        }

        mi::mdl::string nt_ret_type(m_arena.get_allocator());

        // FIXME: For the following node type, we get an invalid
        // string from Node_type::get_return_type().
        // Therefore, we hardcode them for now.

        if (nt->type_name == "color_measured_curve_layer") {
            nt_ret_type = "bsdf";
        } else if (nt->type_name == "edf_color_unbounded_mix_3") {
            nt_ret_type = "bsdf";
        } else if (nt->type_name == "vdf_color_unbounded_mix_3") {
            nt_ret_type = "bsdf";
        } else if (nt->type_name == "hair_bsdf_tint") {
            nt_ret_type = "bsdf";
        } else if (nt->type_name == "material_surface") {
            nt_ret_type = "material_surface";
        } else if (nt->type_name == "material_emission") {
            nt_ret_type = "material_emission";
        } else if (nt->type_name == "material_geometry") {
            nt_ret_type = "material_geometry";
        } else if (nt->type_name == "nvidia::df::simple_glossy_bsdf_legacy") {
            nt_ret_type = "bsdf";
        } else if (nt->type_name == "material_conditional_operator") {
            nt_ret_type = "material";
        } else if (nt->type_name == "bsdf_conditional_operator") {
            nt_ret_type = "bsdf";
        } else if (nt->type_name == "edf_conditional_operator") {
            nt_ret_type = "edf";
        } else if (nt->type_name == "vdf_conditional_operator") {
            nt_ret_type = "vdf";
        } else {
            nt_ret_type = nt->get_return_type().c_str();
        }

        Type *ret_type = builtin_type_for(nt_ret_type.c_str());

        if (!ret_type) {
            printf("[warning] ignoring distiller function '%s' (unknown return type: '%s' for '%s' [%s])\n",
                   nt->type_name.c_str(), nt_ret_type.c_str(), nt->get_return_type().c_str(), nt->get_signature().c_str());
            continue;
        }

        Symbol* name = m_symbol_table->get_symbol(nt->type_name.c_str());

        // For builtins important from the distiller node definitions,
        // we expand the function types for all supported parameter
        // counts.  That means that we add types that have arities
        // from the minimum parameter count of the node up to the
        // maximum number of parameters, one by one.
        for (int max_param = nt->min_parameters; max_param <= nt->parameters.size(); max_param++) {
            Type_function *t = m_type_factory.create_function(ret_type);
            t->set_semantics(nt->semantics);
            t->set_selector(nt->selector_enum.c_str());
            t->set_node_type(nt);

            Type_list_elem *type_list_elem;

            int i = 1;
            for (std::vector<mi::mdl::Node_param>::const_iterator it(nt->parameters.begin()), end(nt->parameters.end());
                 it != end && i <= max_param; ++it, ++i) {
                Type *pt = builtin_type_for(it->param_type.c_str());
                if (!pt) {
                    printf("[warning] ignoring distiller function %s (unknown type for parameter %d: %s)\n",
                           nt->type_name.c_str(), i, nt->get_signature().c_str());
                    error = true;
                    break;
                }

                type_list_elem = m_arena_builder.create<Type_list_elem>(pt);
                t->add_parameter(type_list_elem);
            }

            if (error)
                continue;

            add_binding(name, t, builtin_env);
        }
    }
}

/// Declare all exported materials from the MDL module called
/// `module_name`.
void Compilation_unit::declare_materials_from_module(
    char const *module_name,
    mi::mdl::Module const *module,
    Environment &builtin_env)
{

    int def_count = module->get_exported_definition_count();

    for (int i = 0; i < def_count; i++) {
        mi::mdl::Definition const *def = module->get_exported_definition(i);

        switch (def->get_kind()) {
        case mi::mdl::IDefinition::Kind::DK_FUNCTION:
        {
            mi::mdl::ISymbol const *sym = def->get_symbol();
            mi::mdl::IType_function const *func_type =
                cast<mi::mdl::IType_function>(def->get_type());
            mi::mdl::IType const *ret_type = func_type->get_return_type();

            // Make sure that the return type is a struct.
            switch (ret_type->get_kind()) {
            case mi::mdl::IType::Kind::TK_STRUCT:
            {
                mi::mdl::IType_struct const *strct = cast<mi::mdl::IType_struct>(ret_type);

                // The returned struct must be a material, or the
                // function will be ignored.
                if (strct->get_predefined_id() == mi::mdl::IType_struct::Predefined_id::SID_MATERIAL) {
                    Type *mdltl_type = m_type_factory.import_type(func_type);

                    if (is<Type_error>(mdltl_type)) {
                        printf("[error] could not import type for %s\n",
                               sym->get_name());
                        continue;
                    }

                    // Bind the fully qualified name of the material
                    // in the environment.

                    mi::mdl::string fq_name(get_allocator());
                    fq_name = module_name;
                    fq_name += "::";
                    fq_name += sym->get_name();

                    Symbol const *mdltl_fq_sym = m_symbol_table->get_symbol(fq_name.c_str());
                    add_binding(mdltl_fq_sym, mdltl_type, builtin_env);
                }
                break;
            }
            default:
                // Ignore functions not returning a struct.
                break;
            }
            break;
        }
        default:
            // Ignore non-function exports.
            break;
        }

    }
}

void Compilation_unit::dump_env(Environment &env) {
    pp::Pretty_print p(m_arena, std::cerr);
    env.pp(p);
}

/// Initialize the environment with all builtin, stdlib and distiller
/// types and functions.
void Compilation_unit::init_builtin_env(Environment &builtin_env) {
    declare_builtins(builtin_env);
    declare_stdlib(builtin_env);
    declare_dist_nodes(builtin_env);

    if (m_comp_options->get_debug_dump_builtins()) {
        dump_env(builtin_env);
    }
}

/// Load all MDL modules that are referenced in any rule set of the
/// current compilation unit.
void Compilation_unit::process_imports(Environment &env) {

    mi::mdl::MDL *mdl = mi::mdl::impl_cast<mi::mdl::MDL>(m_imdl);

    mi::base::Handle<mi::mdl::Thread_context> thread_ctx(mdl->create_thread_context());

    for (mi::mdl::Ast_list<Ruleset>::iterator it(m_rulesets.begin()),
             end(m_rulesets.end());
         it != end; ++it) {

        Ruleset &ruleset = *it;

        Import_list const &imports = ruleset.get_imports();

        for (mi::mdl::Ast_list<Import>::const_iterator it(imports.begin()),
                 end(imports.end());
             it != end;
             ++it) {
            char const *name = it->get_name();

            mi::base::Handle<mi::mdl::Module const>
                imp_mod(mdl->load_module(thread_ctx.get(), name, NULL));

            if (!imp_mod) {
                mi::mdl::Messages const &msgs = thread_ctx->access_messages();
                size_t msg_count = msgs.get_message_count();
                for (size_t i = 0; i < msg_count; i++) {
                    mi::mdl::IMessage const *msg = msgs.get_error_message(i);

                    error(it->get_location(), msg->get_string());
                }
                continue;
            }

            declare_materials_from_module(name, imp_mod.get(), env);
        }
    }
}

void Compilation_unit::error(Location const &location, const char *msg) {
    mi::mdl::Arena_builder builder(m_global_arena);
    Message *m = builder.create<Message>(
        &m_global_arena, Message::Severity::SEV_ERROR,
        m_filename, location.get_line(), location.get_column(), msg);
    m_messages->push_back(m);
    m_error_count++;
}

void Compilation_unit::warning(Location const &location, const char *msg) {
    mi::mdl::Arena_builder builder(m_global_arena);
    Message *m = builder.create<Message>(
        &m_global_arena, Message::Severity::SEV_WARNING,
        m_filename, location.get_line(), location.get_column(), msg);
    m_messages->push_back(m);
    m_warning_count++;
}

void Compilation_unit::hint(Location const &location, const char *msg) {
    mi::mdl::Arena_builder builder(m_global_arena);
    Message *m = builder.create<Message>(
        &m_global_arena, Message::Severity::SEV_HINT,
        m_filename, location.get_line(), location.get_column(), msg);
    m_messages->push_back(m);
}

void Compilation_unit::info(Location const &location, const char *msg) {
    mi::mdl::Arena_builder builder(m_global_arena);
    Message *m = builder.create<Message>(
        &m_global_arena, Message::Severity::SEV_INFO,
        m_filename, location.get_line(), location.get_column(), msg);
    m_messages->push_back(m);
}

/// Return the Distiller selector for the given expression.
int Compilation_unit::get_node_selector(Expr *expr, Selector_kind &sel_kind) {
    int selector = 0;

    sel_kind = SK_NORMAL;

    switch ( expr->get_kind()) {
    case Expr::EK_REFERENCE:
    {
        Expr_ref const *c = cast<Expr_ref>(expr);
        if (c->get_name() == m_symbol_table->get_symbol("_")) {
            sel_kind = SK_WILDCARD;
        } else {
            sel_kind = SK_VARIABLE;
        }
        break;
    }

    case Expr::EK_ATTRIBUTE:
    {
        Expr_attribute *c = cast<Expr_attribute>(expr);
        Expr *arg = c->get_argument();
        return get_node_selector(arg, sel_kind);
    }

    case Expr::EK_LITERAL:
    {
        Expr_literal const *c = cast<Expr_literal>(expr);
        switch (deref(c->get_type())->get_kind()) {
        case Type::Kind::TK_BSDF:
            selector = mi::mdl::DS_DIST_DEFAULT_BSDF;
            break;
        case Type::Kind::TK_HAIR_BSDF:
            selector = mi::mdl::DS_DIST_DEFAULT_HAIR_BSDF;
            break;
        case Type::Kind::TK_EDF:
            selector = mi::mdl::DS_DIST_DEFAULT_EDF;
            break;
        case Type::Kind::TK_VDF:
            selector = mi::mdl::DS_DIST_DEFAULT_VDF;
            break;
        case Type::Kind::TK_STRUCT:
        {
            Symbol const *name = cast<Type_struct>(deref(c->get_type()))->get_name();
            if (name == m_symbol_table->get_symbol("material_emission")) {
                selector = mi::mdl::DS_DIST_STRUCT_MATERIAL_EMISSION;
            } else if (name == m_symbol_table->get_symbol("material_surface")) {
                selector = mi::mdl::DS_DIST_STRUCT_MATERIAL_SURFACE;
            } else if (name == m_symbol_table->get_symbol("material_volume")) {
                selector = mi::mdl::DS_DIST_STRUCT_MATERIAL_VOLUME;
            } else if (name == m_symbol_table->get_symbol("material_geometry")) {
                selector = mi::mdl::DS_DIST_STRUCT_MATERIAL_GEOMETRY;
            } else if (name == m_symbol_table->get_symbol("material")) {
                selector = mi::mdl::DS_DIST_STRUCT_MATERIAL;
            }
            break;
        }
        default:
            break;
        }
        break;
    }

    case Expr::EK_CALL:
    {
        Expr_call const *call = cast<Expr_call>(expr);
        MDL_ASSERT(is<Expr_ref>(call->get_callee()));
        Expr_ref const *callee = cast<Expr_ref>(call->get_callee());
        MDL_ASSERT(is<Type_function>(deref(callee->get_type())));
        Type_function const *callee_type = cast<Type_function>(deref(callee->get_type()));
        Type const *return_type = callee_type->get_return_type();

        selector = callee_type->get_semantics();

        if ( selector == mi::mdl::IDefinition::DS_ELEM_CONSTRUCTOR) {
            if (is<Type_struct>(deref(expr->get_type()))) {
                MDL_ASSERT(is<Type_struct>(expr->get_type()));
                Symbol const *name = cast<Type_struct>(deref(expr->get_type()))->get_name();
                if (name == m_symbol_table->get_symbol("material_emission")) {
                    selector = mi::mdl::DS_DIST_STRUCT_MATERIAL_EMISSION;
                } else if (name == m_symbol_table->get_symbol("material_surface")) {
                    selector = mi::mdl::DS_DIST_STRUCT_MATERIAL_SURFACE;
                } else if (name == m_symbol_table->get_symbol("material_volume")) {
                    selector = mi::mdl::DS_DIST_STRUCT_MATERIAL_VOLUME;
                } else if (name == m_symbol_table->get_symbol("material_geometry")) {
                    selector = mi::mdl::DS_DIST_STRUCT_MATERIAL_GEOMETRY;
                } else if (name == m_symbol_table->get_symbol("material")) {
                    selector = mi::mdl::DS_DIST_STRUCT_MATERIAL;
                }
            }
        } else if ((selector == mi::mdl::IDefinition::DS_INTRINSIC_DF_NORMALIZED_MIX)
                   || (selector == mi::mdl::IDefinition::DS_INTRINSIC_DF_CLAMPED_MIX)
                   || (selector == mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_NORMALIZED_MIX)
                   || (selector == mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_CLAMPED_MIX)
                   || (selector == mi::mdl::IDefinition::DS_INTRINSIC_DF_UNBOUNDED_MIX)
                   || (selector == mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_UNBOUNDED_MIX)) {
            if (selector == mi::mdl::IDefinition::DS_INTRINSIC_DF_NORMALIZED_MIX)
                selector = mi::mdl::DS_DIST_BSDF_MIX_1;
            else if (selector == mi::mdl::IDefinition::DS_INTRINSIC_DF_CLAMPED_MIX)
                selector = mi::mdl::DS_DIST_BSDF_CLAMPED_MIX_1;
            else if (selector == mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_NORMALIZED_MIX)
                selector = mi::mdl::DS_DIST_BSDF_COLOR_MIX_1;
            else if (selector == mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_CLAMPED_MIX)
                selector = mi::mdl::DS_DIST_BSDF_COLOR_CLAMPED_MIX_1;
            else if (selector == mi::mdl::IDefinition::DS_INTRINSIC_DF_UNBOUNDED_MIX)
                selector = mi::mdl::DS_DIST_BSDF_UNBOUNDED_MIX_1;
            else if (selector == mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_UNBOUNDED_MIX)
                selector = mi::mdl::DS_DIST_BSDF_COLOR_UNBOUNDED_MIX_1;
            switch ( return_type->get_kind()) {
            case Type::TK_BSDF:
                break;
            case Type::TK_EDF:
                selector += mi::mdl::DS_DIST_EDF_MIX_1 - mi::mdl::DS_DIST_BSDF_MIX_1;
                break;
            case Type::TK_VDF:
                selector += mi::mdl::DS_DIST_VDF_MIX_1 - mi::mdl::DS_DIST_BSDF_MIX_1;
                break;
            default:
                MDL_ASSERT(!"Malformed AST with a mixer node whose array is none of the DFs.");
            }

            sel_kind = SK_MIXER;

        } else if (selector == mi::mdl::IDefinition::DS_INTRINSIC_DF_TINT) {
            selector = mi::mdl::DS_DIST_BSDF_TINT;
            if (call->get_argument_count() == 2) {
                switch (deref(call->get_argument(1)->get_type())->get_kind()) {
                case Type::TK_BSDF:
                    selector = mi::mdl::DS_DIST_BSDF_TINT; break;
                case Type::TK_COLOR:
                    selector = mi::mdl::DS_DIST_BSDF_TINT; break;
                case Type::TK_EDF:
                    selector = mi::mdl::DS_DIST_EDF_TINT; break;
                case Type::TK_VDF:
                    selector = mi::mdl::DS_DIST_VDF_TINT; break;
                case Type::TK_HAIR_BSDF:
                    selector = mi::mdl::DS_DIST_HAIR_BSDF_TINT; break;
                default:
                    MDL_ASSERT(!"Unsupported tint modifier");
                }
            } else {
                MDL_ASSERT(call->get_argument_count() == 3 && "Unsupported tint overload");
                selector = mi::mdl::DS_DIST_BSDF_TINT2;
            }
        } else if (selector == (mi::mdl::IDefinition::DS_OP_BASE + mi::mdl::IExpression::OK_TERNARY)) {
            Expr_call const *c1 = as<Expr_call>(call->get_argument(1));
            if ( c1) {
                Type_function *tf1 = cast<Type_function>(deref(c1->get_callee()->get_type()));
                if (tf1->get_semantics() == mi::mdl::IDefinition::DS_ELEM_CONSTRUCTOR
                    && is<Type_struct>(tf1->get_return_type())
                    && cast<Type_struct>(tf1->get_return_type())->get_name() == m_symbol_table->get_symbol("material")) {
                    selector = mi::mdl::DS_DIST_MATERIAL_CONDITIONAL_OPERATOR;
                } else if (deref(call->get_argument(1)->get_type())->get_kind() == Type::TK_BSDF) {
                    selector = mi::mdl::DS_DIST_BSDF_CONDITIONAL_OPERATOR;
                } else if (deref(call->get_argument(1)->get_type())->get_kind() == Type::TK_EDF) {
                    selector = mi::mdl::DS_DIST_EDF_CONDITIONAL_OPERATOR;
                } else if (deref(call->get_argument(1)->get_type())->get_kind() == Type::TK_VDF) {
                    selector = mi::mdl::DS_DIST_VDF_CONDITIONAL_OPERATOR;
                }
            }
        } else if (selector == mi::mdl::IDefinition::DS_UNKNOWN) {
            if (callee->get_name() == m_symbol_table->get_symbol("::nvidia::distilling_support::local_normal") ||
                callee->get_name() == m_symbol_table->get_symbol("local_normal")) {
                selector = mi::mdl::DS_DIST_LOCAL_NORMAL;
            }
        }
        break;
    }
    default:
        MDL_ASSERT(!"invalid node type in pattern");
        break;
    }
    return selector;
}

Expr *Compilation_unit::normalize_mixer_pattern(Expr *expr) {
    switch (expr->get_kind()) {
    case Expr::EK_REFERENCE:
    case Expr::Kind::EK_TYPE_ANNOTATION:
        break;

    case Expr::EK_CALL:
    {
        Expr_call *call = cast<Expr_call>(expr);

        // First, we recurse and construct a new call with (possibly)
        // rearranged children.

        Expr_call *new_call = m_expr_factory.create_call(call->get_type(), call->get_callee());
        for (int i = 0; i < call->get_argument_count(); i++) {
            Expr *new_arg = normalize_mixer_pattern(call->get_argument(i));
            Argument *a = m_expr_factory.create_argument(new_arg);
            new_call->add_argument(a);
        }

        expr = new_call;

        Selector_kind sel_kind;

        // We can ignore the result here. We do not need the selector
        // at this point, only need to know whether it is a mixer or
        // not.
        (void) get_node_selector(expr, sel_kind);

        switch (sel_kind) {
        case SK_WILDCARD:
        case SK_VARIABLE:
            break;

        case SK_MIXER:
        {
            // When the current expression is a mixer call, we check
            // whether the arguments are call expressions, and if they
            // are we check that they are ordered correctly. If not,
            // we rearrange them.

            if (call->get_argument_count() > 2) {
                int last_sel = -1000000000;
                bool unnormalized = false;
                bool some_vars = false;
                bool all_vars = true;
                for (int i = 0; i < call->get_argument_count() / 2; i++) {
                    Expr *arg = call->get_argument(i * 2 + 1);

                    MDL_ASSERT(arg);

                    int sel = get_node_selector(arg, sel_kind);

                    if (sel_kind == SK_WILDCARD || sel_kind == SK_VARIABLE) {
                        some_vars = true;
                    } else {
                        all_vars = false;
                    }
                    if (sel < last_sel) {
                        unnormalized = true;
                        break;
                    }
                    last_sel = sel;
                }
                if (m_comp_options->get_warn_non_normalized_mixers()) {
                    if (some_vars && !all_vars) {
                        warning(expr->get_location(), "mix of variable and non-variable mixer parameters - cannot normalize pattern");
                    } else if (some_vars && unnormalized) {
                        warning(expr->get_location(), "mixer arguments not in normalized order");
                    }
                }

                // Only attempt argument sorting if all of them are
                // call expressions.

                if (!some_vars && unnormalized) {

                    constexpr int MAX_MIXER_ARG_COUNT = 6;

                    int argc = new_call->get_argument_count();
                    MDL_ASSERT(argc <= MAX_MIXER_ARG_COUNT);

                    Expr *args[MAX_MIXER_ARG_COUNT] = {0};

                    // The following is a simple insertion sort. We
                    // can only have a small number of arguments, so
                    // this is very cheap.

                    for (int i = 0; i < argc / 2; i++) {
                        Expr *weight = new_call->get_argument(i * 2);
                        Expr *bsdf = new_call->get_argument(i * 2 + 1);

                        // Find the insertion spot, moving larger
                        // argument pairs up in the array while
                        // searching.

                        int j = i;
                        while (j > 0) {
                            int idx = j * 2 - 1;
                            int sel1 = get_node_selector(bsdf, sel_kind);
                            int sel2 = get_node_selector(args[idx], sel_kind);
                            if (sel1 >= sel2) {
                                break;
                            }
                            args[j * 2 + 0] = args[j * 2 - 2];
                            args[j * 2 + 1] = args[j * 2 - 1];
                            j--;
                        }

                        // Put the new entry into the correct spot.
                        args[j * 2] = weight;
                        args[j * 2 + 1] = bsdf;
                    }

                    // Now, create a new call expression with the
                    // rearranged arguments.

                    Expr_call *new_call2 = m_expr_factory.create_call(call->get_type(), call->get_callee());
                    for (int i = 0; i < new_call->get_argument_count(); i++) {
                        Expr *new_arg = args[i];
                        Argument *a = m_expr_factory.create_argument(new_arg);
                        new_call2->add_argument(a);
                    }

                    expr = new_call2;

                    // Let the user know that we did some magic.
                     info(call->get_location(), "normalized mixer by ordering arguments");
                }
            }
            break;
        }
        default:
            break;
        }
        break;
    }

    case Expr::Kind::EK_BINARY:
    {
        Expr_binary *eb = cast<Expr_binary>(expr);
        if (eb->get_operator() != Expr_binary::Operator::OK_TILDE) {
            MDL_ASSERT(!"unexpected binary operator in normalize_mixer_pattern");
            return expr;
        }
        Expr *new_right_arg = normalize_mixer_pattern(eb->get_right_argument());
        expr = m_expr_factory.create_binary(expr->get_location(),
                                            expr->get_type(),
                                            eb->get_operator(),
                                            eb->get_left_argument(),
                                            new_right_arg);
        break;
    }

    case Expr::Kind::EK_ATTRIBUTE:
    {
        Expr_attribute *e = cast<Expr_attribute>(expr);
        Expr *new_pat = normalize_mixer_pattern(e->get_argument());

        Expr_attribute::Expr_attribute_vector &attrs = e->get_attributes();
        Expr_attribute::Expr_attribute_vector new_attrs;

        for (size_t i = 0; i < attrs.size(); i++) {
            Expr_attribute::Expr_attribute_entry &p = attrs[i];

            Expr *att_pat = p.expr ? normalize_mixer_pattern(p.expr) : nullptr;
            Expr_attribute::Expr_attribute_entry new_entry;

            // Copy all fields from old entry...
            new_entry = p;
            // ...except for renormalized expression.
            new_entry.expr = att_pat;

            new_attrs.push_back(new_entry);
        }
        expr = m_expr_factory.create_attribute(e->get_location(),
                                               e->get_type(), new_pat, new_attrs,
                                               e->get_node_name());
        break;
    }

    default:
        MDL_ASSERT(!"unexpected expression kind in normalize_mixer_pattern");
        break;
    }

    return expr;
}

