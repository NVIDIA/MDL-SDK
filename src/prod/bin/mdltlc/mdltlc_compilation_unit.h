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

#ifndef MDLTLC_COMPILATION_UNIT_H
#define MDLTLC_COMPILATION_UNIT_H 1

#include <mi/mdl/mdl_mdl.h>

#include <mdl/compiler/compilercore/compilercore_memory_arena.h>

#include <mi/mdl/mdl_distiller_node_types.h>

#include "mdltlc_message.h"
#include "mdltlc_pprint.h"
#include "mdltlc_symbols.h"
#include "mdltlc_types.h"
#include "mdltlc_values.h"
#include "mdltlc_exprs.h"
#include "mdltlc_env.h"
#include "mdltlc_rules.h"
#include "mdltlc_compiler_options.h"

namespace mi {
namespace mdl {
    class Module;
}
}

typedef mi::mdl::vector<Type *>::Type Mdl_type_vector;

/// Interface of the mdltl compilation unit.
class ICompilation_unit : public
    mi::base::Interface_declare<0x05b3fbca,0x5a35,0x4432,0xab,0xd3,0xd7,0x71,0xd8,0x17,0x65,0xde,
    mi::base::IInterface>
{
public:
    /// Load mdltl definitions from the given input stream.
    ///
    /// \param input_stream Input stream to read the mdltl definitions
    ///                     from.
    virtual unsigned compile(mi::mdl::IInput_stream *input_stream) = 0;
};

/// Implementation of an mdltl compilation unit.
class Compilation_unit : public mi::mdl::Allocator_interface_implement<ICompilation_unit>
{
    typedef mi::mdl::Allocator_interface_implement<ICompilation_unit> Base;
    friend class mi::mdl::Allocator_builder;

    enum Pattern_context {
        PC_EXPRESSION,
        PC_PATTERN
    };

public:
    /// Get the relative name of the file from which this unit was
    /// loaded.
    ///
    /// \returns The relative name of the file from which the module
    ///          was loaded, or null if no such file exists.

    char const *get_filename() const;

    /// Get the expression factory.
    Expr_factory &get_expression_factory();

    /// Get the type factory.
    Type_factory &get_type_factory();

    /// Get the value factory.
    Value_factory &get_value_factory();

    /// Get the symbol table of this module.
    Symbol_table &get_symbol_table();

    /// Get the rule factory.
    Rule_factory &get_rule_factory();

    /// Load mdltl definitions from the given input stream.
    ///
    /// \param input_stream Input stream to read the mdltl definitions
    ///                     from.
    unsigned compile(mi::mdl::IInput_stream *input_stream);

    /// Generate an error message for the given location.
    void error(Location const &location, const char *msg);

    /// Generate a warning message for the given location.
    void warning(Location const &location, const char *msg);

    /// Generate an hint message for the given location.
    void hint(Location const &location, const char *msg);

    /// Generate an info message for the given location.
    void info(Location const &location, const char *msg);

    /// Add a ruleset to the compilation unit. This is called by the parser.
    ///
    /// \param ruleset    ruleset to be added.
    void add_ruleset(Ruleset *ruleset);

    // --------------------------- non interface methods ---------------------------

private:
    /// Constructor.
    ///
    /// \param alloc        the allocator
    /// \param file_name    the file name of the module
    explicit Compilation_unit(
        mi::mdl::IAllocator *alloc,
        mi::mdl::Memory_arena *global_arena,
        mi::mdl::IMDL *imdl,
        mi::mdl::Node_types *node_types,
        Symbol_table *symbol_table,
        char const *file_name,
        Compiler_options const *comp_options,
        Message_list *messages,
        Builtin_type_map *builtins);

    /// Return the type for the builtin type named `type_name`.
    Type *builtin_type_for(const char *type_name);

    /// Typecheck the AST of the mdltl file.
    void type_check(Environment &builtin_env);

    /// Initialize the environment with all builtin, stdlib and
    /// distiller types and functions.
    void init_builtin_env(Environment &builtin_env);

    /// Load all MDL modules that are referenced in any rule set of
    /// the current compilation unit.
    void process_imports(Environment &env);

    /// Add the type `type` in environment `builtin_env` to symbol
    /// `name`.
    void add_binding(Symbol const *name, Type *type, Environment &builtin_env);

    /// Declare the functions that are built into the MDL compiler.
    void declare_builtins(Environment &env);

    /// Declare all functions that have been loaded from the standard
    /// libraries.
    void  declare_stdlib(Environment &builtin_env);

    /// Declare all exported materials from the MDL module called
    /// `module_name`.
    void declare_materials_from_module(
        char const *module_name,
        mi::mdl::Module const *module,
        Environment &builtin_env);

    /// Declare all functions defined on the state object.
    void declare_state_functions(Environment &builtin_env);

    /// Declare all functions declared as node types in the distiller.
    void declare_dist_nodes(Environment &builtin_env);

    /// Perform type checking on the given rule set, using the given
    /// builtin definitions.
    void type_check_ruleset(Ruleset &, Environment &builtin_env);

    /// Perform type checking on a single rule, using the given
    /// builtin definitions.
    void type_check_rule(Rule &, Environment &builtin_env);

    Type *type_check_attribute(Expr *expr, Expr_attribute *e, Environment &env,
                               Pattern_context pattern_ctx);

    /// Type check the LHS of a rule. `builtin_env` are the builtin
    /// definitions available and `env` is the local environment used for type
    /// checking. Variables introduced by the pattern are added to `env`.
    Type *type_check_pattern(Expr *expr, Environment &env);

    /// Type check an expression. See `type_check_pattern` for a
    /// description of the `Env` parameters.
    Type *type_check_expr(Expr *expr, Environment &env);

    /// Type check a rule guard expression. See `type_check_pattern` for a
    /// description of the `Env` parameters.
    void type_check_guard(Expr *guard, Environment &env);

    /// Type check a reference to a named entity. See
    /// `type_check_pattern` for a description of the `Env`
    /// parameters.
    Type *type_check_reference(Expr *expr,
                               Environment &env,
                               Pattern_context pattern_ctx);

    /// Type check a reference in call position. These never introduce
    /// variables and must be defined in the builtin environment.
    /// `expr` is the called reference, `call_expr` is the surrounding
    /// call expression and `arg_type` is a vector of the types of all
    /// arguments of the call, which are used for overload
    /// resolution. If overload resolution is successful, all argument
    /// types that are type variables will be bound to the
    /// corresponding parameter types of the resolved function
    /// overload.
    Type *type_check_called_reference(Expr *expr,
                                      Expr_call *call_expr,
                                      Mdl_type_vector &arg_type,
                                      Environment &env);

    /// Type check a call expression.
    Type *type_check_call(Expr *expr,
                          Environment &env,
                          Pattern_context pattern_ctx);

    /// Type check a where clause.
    void type_check_where(Argument_list &list,
                          Environment &env);

    /// Type check a rule set postcondition.
    void type_check_postcond(Postcond &list, Environment &builtin_env);

    /// Look up a reference in the environments and return true if it
    /// exists, false otherwise. If it does not exist, issue an error.
    bool check_reference_exists(Expr *expr, Environment &env);

    /// Helper function to generate hints for the user when overload
    /// resolution fails.
    void generate_overload_hints(Expr *expr,
                                 Expr_call *call_expr,
                                 Environment::Type_list &ts,
                                 Mdl_type_vector &arg_types);

    /// Calculates the result type of a binary arithmetic expression
    /// of operands with the given types.
    Type *types_compatible_arith(Expr *expr,
                                 Expr_binary::Operator op,
                                 Type *type1,
                                 Type *type2);

    /// Calculates the result type of a binary comparison expression
    /// of operands with the given types.
    Type *types_compatible_cmp(Expr *expr,
                               Expr_binary::Operator op,
                               Type *type1,
                               Type *type2);

    /// Calculates the result type of a general binary expression of
    /// operands with the given types. Calls `types_compatible_arith`
    /// and `types_compatible_cmp` for appropriate operations.
    Type *types_compatible(Expr *expr,
                           Expr_binary::Operator op,
                           Type *type1,
                           Type *type2);

    /// Perform overload resolution. `name` is the name of the called
    /// function, `expr` is the callee, `call_expr` the surrounding
    /// call expression, `ts` the list of available overload types for
    /// `name` and `arg_types` a vector of the actual argument types
    /// in the call (which may contain type variables).
    Type *resolve_overload(Symbol const *name,
                           Expr *expr,
                           Expr_call *call_expr,
                           Environment::Type_list *ts,
                           Mdl_type_vector &arg_types);

    enum Selector_kind {
        SK_NORMAL,
        SK_WILDCARD,
        SK_VARIABLE,
        SK_MIXER
    };

    /// Get the distiller node selector for the given call expression.
    int get_node_selector(Expr *expr, Selector_kind &sel_kind);

    /// Bring all mixer calls in the given pattern into normal
    /// form. This means that the arguments are ordered by the numeric
    /// value of the semantics of their root functions.
    Expr *normalize_mixer_pattern(Expr *expr);

    void normalize_mixers();

    void lint_rules();

    /// Generate the output .cpp/.h files for the parsed and type
    /// checked compilation unit. This should only be called when no
    /// errors were encountered during the earlier phases.
    void output();

    /// Generate the output .h file.
    void output_h(mi::mdl::string const &stem_name,
                  mi::mdl::string const &h_name);

    /// Helper function for output_h.
    void output_h_postcond_expr(pp::Pretty_print &p,
                                Expr *expr,
                                int &idx);

    /// Generate the output .cpp file.
    void output_cpp(mi::mdl::string const &stem_name,
                    mi::mdl::string const &cpp_name);

    /// Helper function for output_cpp. Outputs the bindings for
    /// matched variables.
    void output_cpp_match_variables(pp::Pretty_print &p,
                                    Expr const *expr,
                                    mi::mdl::string const &prefix,
                                    Var_set &used_vars);

    /// Helper function for output_cpp. Outputs the matcher function.
    void output_cpp_matcher(pp::Pretty_print &p,
                            Ruleset &ruleset,
                            mi::mdl::vector<Rule const *>::Type &rules);
    void output_cpp_matcher_body(pp::Pretty_print &p,
                                 Rule const &rule,
                                 size_t rule_index,
                                 mi::mdl::string &pfx);

    /// Helper function for output_cpp. Outputs the helper function
    /// definitions for postcondition checks.
    void output_cpp_postcond_helpers(pp::Pretty_print &p,
                                     Ruleset &ruleset,
                                     Expr *expr,
                                     int &idx);

    /// Helper function for output_cpp. Outputs a postcondition
    /// expression which calls the helper functions generated by
    /// `output_cpp_postcond_helpers`.
    void output_cpp_postcond_expr(pp::Pretty_print &p,
                                  Ruleset &ruleset,
                                  Expr *expr,
                                  int &idx);

    /// Helper function for output_cpp. Outputs the code for checking
    /// the postcondition.
    void output_cpp_postcond(pp::Pretty_print &p,
                             Ruleset &ruleset);

    /// Helper function for output_cpp. Outputs a pattern condition
    /// expression continuation for matcher and postcondition checks.
    void output_cpp_pattern_condition(pp::Pretty_print &p,
                                      Expr const *expr,
                                      mi::mdl::string const &prefix);

    /// Helper function for output_cpp. Outputs the event handler definitions.
    void output_cpp_event_handler(pp::Pretty_print &p,
                                  Ruleset &ruleset);

    /// Helper function for output_cpp. Outputs the code to construct
    /// the equivalent of the given expression.
    void output_cpp_expr(pp::Pretty_print &p,
                         Expr const *expr);

    /// Helper function for output_cpp. Outputs the code to bind
    /// variables for nodes with attached attributes.  the equivalent
    /// of the given expression.
    void output_cpp_expr_bindings(pp::Pretty_print &p,
                                  Expr const *expr);

    /// Return the selector for the given expression. This is a
    /// variant of mi::mdl::IDefition::Semantics as a string, to be
    /// used in generated code.
    char const *find_selector(Expr const *expr);

    /// Return the semantic for the given expression. This is a
    /// variant of mi::mdl::IDefition::Semantics as a string, to be
    /// used in generated code.
    int find_semantics(Expr const* expr);

    /// Helper to output a list of where bindings in reverse.
    void output_reversed(pp::Pretty_print &p,
                         Argument_list::const_iterator it,
                         Argument_list::const_iterator end);

    /// Sort the rules from the given ruleset into the vector,
    /// bringing rules with the same top-level function symbol
    /// together.
    void sort_rules(mi::mdl::vector<Rule const *>::Type &rules, Ruleset const &ruleset);

    void output_cpp_literal_value(
        pp::Pretty_print &p,
        Expr_literal const *lit,
        bool for_color = false);
    void output_cpp_literal(
        pp::Pretty_print &p,
        Expr const *lit);
    void output_cpp_mixer_call(
        pp::Pretty_print &p,
        Expr_call const *call,
        int n_ary_mixer,
        Type_function const *callee_type);
    void output_cpp_constant_call(
        pp::Pretty_print &p,
        Expr_call const *call,
        char const *call_name);
    void output_cpp_function_call(
        pp::Pretty_print &p,
        Expr_call const *call,
        Expr_ref const *callee_ref);
    void output_cpp_bsdf_call(
        pp::Pretty_print &p,
        Expr_call const *call,
        Expr_ref const *callee_ref,
        mi::mdl::Node_type const *node_type,
        int node_type_idx);
    void output_cpp_call(
        pp::Pretty_print &p,
        Expr_call const *call);

    /// Return true if all arguments of the call expressions are
    /// constant (e.g., literals).
    bool all_arguments_constants(Expr_call const *expr_call);

    void calculate_rule_ids();

    void reset_attr_counter();
    int next_attr_counter();

private:
    // non copyable
    Compilation_unit(Compilation_unit const &) = delete;
    Compilation_unit &operator=(Compilation_unit const &) = delete;

    bool is_material_struct(Type const *t);
    bool is_material_or_bsdf(Type const *t);

    void dump_env(Environment &env);

private:
    /// The memory arena of for the global compilation process, used
    /// to allocate elements that must survive the compilation unit.
    mi::mdl::Memory_arena &m_global_arena;

    /// The memory arena of this module, used to allocate all elements
    /// of this module.
    mi::mdl::Memory_arena m_arena;

    /// The Arena bulder used;
    mi::mdl::Arena_builder m_arena_builder;

    /// The MDL implementation to use for access to intrinsics.
    mi::mdl::IMDL *m_imdl;

    /// Pointer to node type information. This is the information the distiller
    /// uses for matching and constructing nodes.
    mi::mdl::Node_types *m_node_types;

    /// The name of the file from which the module was loaded.
    char const *m_filename;

    /// The file name of the loaded module without any directories.
    char const *m_filename_only;

    /// Options for the compiler.
    Compiler_options const *m_comp_options;

    /// The symbol table to use for this compilation.
    Symbol_table *m_symbol_table;

    /// The type factory to use for this compilation.
    Type_factory m_type_factory;

    /// The value factory to use for this compilation.
    Value_factory m_value_factory;

    /// The expression factory to use for this compilation.
    Expr_factory m_expr_factory;

    /// The rule factory to use for this compilation.
    Rule_factory m_rule_factory;

    /// List of rulesets in the compilation unit.
    Ruleset_list m_rulesets;

    /// Pointer to message collector managed in the Compiler class.
    Message_list *m_messages;

    /// Pointer to map from builtin names to their MDL types.
    Builtin_type_map *m_builtins;

    /// The number of errors encountered while compiling an mdltl
    /// file.
    int m_error_count;

    /// The number of warnings encountered while compiling an mdltl
    /// file.
    int m_warning_count;

    /// Name of the API class to use in generated files.
    char const *m_api_class;

    /// Name of the rule matcher base class to use in generated files.
    char const *m_rule_matcher_class;

    /// Error type from type factory. We have this because it is used
    /// very frequently.
    Type *m_error_type;

    /// Used to generate identifiers for node attribute mappings during
    /// the generation of matcher code.
    int m_attr_counter;

    /// Environment to hold attribute types across different rule
    /// sets.
    Environment m_attribute_env;
};

#endif
