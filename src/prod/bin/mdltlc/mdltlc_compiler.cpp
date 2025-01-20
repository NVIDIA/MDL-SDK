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

#include <mdl/compiler/compilercore/compilercore_memory_arena.h>
#include <mdl/compiler/compilercore/compilercore_streams.h>
#include <mdl/compiler/compilercore/compilercore_tools.h>
#include <mdl/compiler/compilercore/compilercore_file_utils.h>
#include <mdl/compiler/compilercore/compilercore_wchar_support.h>
#include <mdl/compiler/compilercore/compilercore_mdl.h>
#include <mdl/compiler/compilercore/compilercore_factories.h>
#include <mdl/compiler/compilercore/compilercore_symbols.h>
#include <mdl/compiler/compilercore/compilercore_mangle.h>

#include "mdltlc_compiler.h"
#include "mdltlc_module.h"

constexpr int SHOW_MESSAGE_COUNT = 10;

// We just pull in this one source file for the Node_types instead of
// linking the distiller library and many other dependencies.
#include <mdl/codegenerators/generator_dag/generator_dag_distiller_node_types.cpp>

// Constructor.
Compiler::Compiler(mi::mdl::IMDL *imdl)
    : Base(imdl->get_mdl_allocator())
    , m_imdl(imdl)
    , m_mdl(*mi::mdl::impl_cast<mi::mdl::MDL>(m_imdl))
    , m_node_types()
    , m_mdl_type_factory(*mi::mdl::impl_cast<mi::mdl::Type_factory>(imdl->get_type_factory()))
    , m_mdl_symbol_table(*m_mdl_type_factory.get_symbol_table())
    , m_allocator(imdl->get_mdl_allocator())
    , m_arena(m_allocator)
    , m_arena_builder(m_arena)
    , m_comp_options(&m_arena)
    , m_symbol_table(m_arena)
    , m_type_factory(m_arena, m_symbol_table)
    , m_messages(m_allocator)
    , m_mdl_type_cache(*mi::mdl::impl_cast<mi::mdl::MDL>(imdl)->get_type_factory())
    , m_def_table(m_arena)
{
}

mi::base::Handle<Compilation_unit> Compiler::create_unit(const char *fname)
{
    mi::mdl::Allocator_builder builder(m_allocator);
    Compilation_unit *unit =
        builder.create<Compilation_unit>(
            m_allocator,
            &m_arena,
            m_imdl,
            &m_node_types,
            &m_symbol_table,
            &m_type_factory,
            fname,
            &m_comp_options,
            &m_messages,
            &m_def_table);
    return mi::base::make_handle(unit);
}

void Compiler::run(unsigned& err_count) {

    // Load ::nvidia::distilling_support as a builtin module to make
    // its definitions available to mdltl files.
    {
        // register nvidia/distilling_support.mdl
        bool res = m_imdl->add_builtin_module(
            "::nvidia::distilling_support",
            (char const *)mi::mdl::mdl_module_distilling_support,
            sizeof(mi::mdl::mdl_module_distilling_support),
            /*is_encoded=*/true,
            /*is_native=*/false);

        MDL_ASSERT(res && "Registration of ::nvidia::distilling_support failed");
        (void)res;
    }

    // Declare types built into the MDL compiler.
    declare_builtins();

    // Declare types built into the distiller.
    declare_dist_nodes();

    // Load standard libraries.
    load_builtins();

    //pp::Pretty_print p(m_arena, std::cout);
    //m_def_table.dump(p);

    // Make the MDL paths specified in the compiler options available
    // to the core compiler.
    {
        mi::mdl::IAllocator *allocator = m_imdl->get_mdl_allocator();

        mi::mdl::Allocator_builder builder(allocator);

        Mdltlc_search_path *search_path = builder.create<Mdltlc_search_path>(allocator,
                                                                             &m_comp_options);

        m_imdl->install_search_path(search_path);
    }

    // Compile each mdltl file on the command line, all with the same
    // options, each possibly generating its own .h/.cpp files.
    for (int i = 0; i < m_comp_options.get_filename_count(); i++) {
        const char *filename = m_comp_options.get_filename(i);
        if (m_comp_options.get_verbosity() >= 1) {
            printf("[info] Compiling %s...\n", filename);
        }
        mi::base::Handle<mi::mdl::IInput_stream> is(m_imdl->create_file_input_stream(filename));

        if (!is.get()) {
          fprintf(stderr, "error: file not found: %s\n", filename);
          err_count += 1;
          continue;
        }

        mi::base::Handle<Compilation_unit> unit = create_unit(filename);

        unsigned this_err_count = unit->compile(is.get());

        err_count += this_err_count;
    }

    // For unit tests, we allow error messages to be suppressed.
    if (!m_comp_options.get_silent())
        print_messages();
}

Message_list const &Compiler::get_messages() const {
    return m_messages;
}

void Compiler::print_messages() {

    mi::base::Handle<mi::mdl::IOutput_stream> outs =
        mi::base::make_handle(m_imdl->create_std_stream(mi::mdl::IMDL::Std_stream::OS_STDERR));

    mi::base::Handle<mi::mdl::IPrinter> printer =
        mi::base::make_handle(m_imdl->create_printer(outs.get()));

    printer->enable_color(true);

    int cnt = 0;
    int total_message_cnt = m_messages.size();
    for (mi::mdl::vector<Message*>::Type::const_iterator it(m_messages.begin()),
             end(m_messages.end());
         it != end && (m_comp_options.get_all_errors() || cnt < SHOW_MESSAGE_COUNT);
         ++it, ++cnt) {
        const Message* m = *it;
        printer->print(m->get_filename());
        printer->print(":");
        printer->print(m->get_line());
        printer->print(":");
        printer->print(m->get_column());
        printer->print(": ");
        printer->print(m->get_severity_str());
        printer->print(": ");
        printer->print(m->get_message());
        printer->print("\n");
    }
    if (cnt < total_message_cnt) {
        printer->print("... and ");
        printer->print(total_message_cnt - cnt);
        printer->print(" more message(s)... (use --all-errors to see the full list)\n");
    }
}

Compiler_options &Compiler::get_compiler_options() {
    return m_comp_options;
}

Compiler_options const &Compiler::get_compiler_options() const {
    return m_comp_options;
}

void Compiler::add_builtin(
    Symbol *symbol,
    Symbol *fq_symbol,
    Type const *type,
    mi::mdl::IDefinition::Semantics sema,
    char const *signature,
    mi::mdl::IType const *mdl_type) {
    m_def_table.add(symbol, fq_symbol, type, signature, sema, mdl_type);
}

void Compiler::add_builtin(
    Symbol *symbol,
    Symbol *fq_symbol,
    mi::mdl::IType const *mdl_type,
    mi::mdl::IDefinition::Semantics sema,
    char const *signature)
{
    Type const *t = m_type_factory.import_type(mdl_type);
    add_builtin(symbol, fq_symbol, t, sema, signature, mdl_type);
}

/// Load the given builtin module using the MDL-SDK and print some
/// information about its exports.
void Compiler::load_builtin_module(const char *name) {
    if (m_comp_options.get_debug_builtin_loading()) {
        printf("[info] Loading builtin module %s...\n", name);
    }

    mi::mdl::MDL *mdl = mi::mdl::impl_cast<mi::mdl::MDL>(m_imdl);
    mi::mdl::IMDL::MDL_version current_version =
        mi::mdl::IMDL::MDL_version::MDL_LATEST_VERSION;

    if (m_comp_options.get_debug_builtin_loading()) {
        printf("[info] current MDL version: %d\n", current_version);
    }


    mi::base::Handle<mi::mdl::Thread_context> thread_ctx(mdl->create_thread_context());

    mi::base::Handle<const mi::mdl::Module>
        df_mod(mdl->load_module(thread_ctx.get(), name, NULL));

    if (!df_mod) {
        printf("[error] cannot load builtin module: %s\n", name);
        return;
    }

    mi::mdl::Name_printer printer(m_allocator);
    mi::mdl::string signature(m_allocator);

    int exported_count = df_mod->get_exported_definition_count();

    // Add all builtins to the type map.
    for (int i = 0; i < exported_count; i++) {
        mi::mdl::Definition const *def = df_mod->get_exported_definition(i);
        unsigned version_flags = def->get_version_flags();

        bool removed = false;
        int added_in = 0;
        int removed_in = 0;
        bool added = false;

        for (int version = 1; version <= current_version; version++) {
            if (((version_flags >> 8) & 0x0f) == version) {
                removed = true;
                removed_in = version;
            }
            if (version_flags == version) {
                added = true;
                added_in = version;
            }
        }
        bool available = (!added && !removed)
            || (!removed && added_in <= current_version)
            || (removed_in > current_version);

        if (m_comp_options.get_debug_builtin_loading()) {
            printf("[info] %s: ", def->get_symbol()->get_name());
            if (added)
                printf("added in: %d", added_in);
            if (removed)
                printf(" removed in: %d", removed_in);
            printf(" => %s\n", available ? "loaded" : "NOT loaded");
        }

        // Don't add builtins that are not available in the current version.
        if (!available) {
            continue;
        }

        // Intern the name of the builtin in our own symbol table.

        // Create the unqualified name for the export.
        Symbol *symbol = m_symbol_table.get_symbol(def->get_symbol()->get_name());

        // Create the qualified name for the export;
        mi::mdl::string tmp(m_allocator);
        tmp = (name + 2);
        tmp += "::";
        tmp += def->get_symbol()->get_name();
        Symbol *q_symbol = m_symbol_table.get_symbol(tmp.c_str());

        // Create the fully qualified (absolute) name for the export.
        tmp = name;
        tmp += "::";
        tmp += def->get_symbol()->get_name();
        Symbol *fq_symbol = m_symbol_table.get_symbol(tmp.c_str());


        mi::mdl::IType const *type = def->get_type();

        char const *c_signature = nullptr;

        if (mi::mdl::IType_function const *f_type = as<mi::mdl::IType_function>(type)) {
            // Create the signature, if the definition is a function.
            signature = tmp;
            signature += "(";
            for (size_t i = 0; i < f_type->get_parameter_count(); ++i) {
                mi::mdl::IType const *p_type;
                mi::mdl::ISymbol const *p_name;
                f_type->get_parameter(i, p_type, p_name);
                if (i > 0) {
                    signature += ',';
                }
                printer.print(p_type->skip_type_alias());
                signature += printer.get_line();
            }
            signature += ")";
            c_signature = mi::mdl::Arena_strdup(m_arena, signature.c_str());
        }

        mi::mdl::IDefinition::Semantics sema = m_imdl->get_builtin_semantic(fq_symbol->get_name());

        // Register the type for all unqualified and qualified names.
        add_builtin(symbol, fq_symbol, type, sema, c_signature);
        add_builtin(q_symbol, fq_symbol, type, sema, c_signature);
        add_builtin(fq_symbol, fq_symbol, type, sema, c_signature);
    }
}

/// Load the builtin MDL modules the rule compiler is interested in.
void Compiler::load_builtins() {

     load_builtin_module("::df");
     load_builtin_module("::tex");
     load_builtin_module("::math");
     load_builtin_module("::state");
     load_builtin_module("::nvidia::distilling_support");
}

void Compiler::declare_builtins() {
    int num_args = 0;

    (void)num_args;


#define BUILTIN_TYPE_BEGIN(typenam, flags)                               \
    {                                                                    \
        Symbol *symbol = m_symbol_table.get_symbol(#typenam);            \
        (void) symbol;                                                   \
        std::string s = "::";\
        s += #typenam;\
        Symbol *fq_symbol = m_symbol_table.get_symbol(s.c_str()); \
        (void) fq_symbol;                                                \
        Type *builtin_type = m_type_factory.builtin_type_for(#typenam);  \
        (void) builtin_type;

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
                Type *arg_type = m_type_factory.builtin_type_for(#type);               \
                signature += #type;                                     \
                signature += ',';                                       \
                Type_list_elem *type_list_elem = m_arena_builder.create<Type_list_elem>(arg_type); \
                constr_type->add_parameter(type_list_elem);             \
            }

#define UARG(type, name, arr) \
    ARG(type, name, arr)

#define DEFARG(type, name, arr, expr) \
    ARG(type, name, arr)

#define UDEFARG(type, name, arr, expr) \
    ARG(type, name, arr)

#define XDEFARG(type, name, arr, expr) \
    ARG(type, name, arr)

#define CONSTRUCTOR(kind, classname, args, sema, flags)                 \
        {                                                               \
            using mi::mdl::Version_flags::REMOVED_1_1; \
            using mi::mdl::Version_flags::SINCE_1_1; \
            using mi::mdl::Version_flags::REMOVED_1_5; \
            using mi::mdl::Version_flags::SINCE_1_5; \
            using mi::mdl::Version_flags::REMOVED_1_7; \
            using mi::mdl::Version_flags::SINCE_1_7; \
            unsigned version_removed = (flags >> 8) & 0xff; \
            if (version_removed == 0) { \
            Type_function *constr_type = m_type_factory.create_function(builtin_type); \
            constr_type->set_semantics(mi::mdl::IDefinition::Semantics:: sema);        \
            std::string signature;   \
            signature += #classname; \
            signature += '(';        \
            args                     \
            if (signature.back() == ',') { signature.pop_back(); } \
            signature += ')';        \
            add_builtin(symbol, fq_symbol, constr_type, mi::mdl::IDefinition::Semantics::sema, signature.c_str(), nullptr); \
            } \
        }

#define BUILTIN_TYPE_END(typenam)              \
    }

#include "mdl/compiler/compilercore/compilercore_known_defs.h"
}

void Compiler::declare_dist_nodes() {

    for (int idx = 0; ; idx++) {
        bool error = false;
        mi::mdl::Node_type const *nt = m_node_types.type_from_idx(idx);
        // static_type_from_idx returns nullptr if the idx is larger
        // than the last supported node type index.
        if (!nt) {
            break;
        }

        //mi::mdl::string nt_ret_type(m_arena.get_allocator());
        char const *nt_ret_type = m_node_types.get_return_type(idx);

        Type *ret_type = m_type_factory.builtin_type_for(nt_ret_type);

        if (!ret_type) {
            printf("[warning] ignoring distiller function '%s' (unknown return type: '%s' for '%s' [%s])\n",
                nt->type_name, nt_ret_type, m_node_types.get_return_type(idx), nt->get_signature().c_str());
            continue;
        }

        Symbol *name = m_symbol_table.get_symbol(nt->type_name);

        // For builtins important from the distiller node definitions,
        // we expand the function types for all supported parameter
        // counts.  That means that we add types that have arities
        // from the minimum parameter count of the node up to the
        // maximum number of parameters, one by one.
        for (int max_param = nt->min_parameters; max_param <= nt->parameters.size(); max_param++) {
            Type_function *t = m_type_factory.create_function(ret_type);
            t->set_semantics(nt->semantics);
            t->set_selector(nt->selector_enum);
            t->set_node_type(nt);

            Type_list_elem *type_list_elem;

            int i = 1;
            for (std::vector<mi::mdl::Node_param>::const_iterator it(nt->parameters.begin()), end(nt->parameters.end());
                it != end && i <= max_param; ++it, ++i) {
                Type *pt = m_type_factory.builtin_type_for(it->param_type);
                if (!pt) {
                    printf("[warning] ignoring distiller function %s (unknown type for parameter %d)\n",
                        nt->type_name, i);
                    error = true;
                    break;
                }

                type_list_elem = m_arena_builder.create<Type_list_elem>(pt);
                t->add_parameter(type_list_elem);
            }

            if (error)
                continue;

            add_builtin(name, name, t, nt->semantics, Arena_strdup(m_arena, nt->get_signature().c_str()), nullptr);
        }
    }
}
