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

#include <mdl/compiler/compilercore/compilercore_memory_arena.h>
#include <mdl/compiler/compilercore/compilercore_streams.h>
#include <mdl/compiler/compilercore/compilercore_tools.h>
#include <mdl/compiler/compilercore/compilercore_file_utils.h>
#include <mdl/compiler/compilercore/compilercore_wchar_support.h>
#include <mdl/compiler/compilercore/compilercore_mdl.h>
#include <mdl/compiler/compilercore/compilercore_factories.h>
#include <mdl/compiler/compilercore/compilercore_symbols.h>

#include "mdltlc_compiler.h"
#include "mdltlc_module.h"

constexpr int SHOW_MESSAGE_COUNT = 10;

// Constructor.
Compiler::Compiler(mi::mdl::IMDL *imdl)
    : Base(imdl->get_mdl_allocator())
    , m_imdl(imdl)
    , m_mdl(*mi::mdl::impl_cast<mi::mdl::MDL>(m_imdl))
    , m_mdl_type_factory(*mi::mdl::impl_cast<mi::mdl::Type_factory>(imdl->get_type_factory()))
    , m_mdl_symbol_table(*m_mdl_type_factory.get_symbol_table())
    , m_allocator(imdl->get_mdl_allocator())
    , m_arena(m_allocator)
    , m_arena_builder(m_arena)
    , m_comp_options(&m_arena)
    , m_builtin_type_map(0, Builtin_type_map::hasher(), Builtin_type_map::key_equal(), m_allocator)
    , m_symbol_table(m_arena)
    , m_messages(m_allocator)
    , m_mdl_type_cache(*mi::mdl::impl_cast<mi::mdl::MDL>(imdl)->get_type_factory())
{
}

mi::base::Handle<Compilation_unit> Compiler::create_unit( const char *fname)
{
    mi::mdl::Allocator_builder builder(m_allocator);
    Compilation_unit *unit =
        builder.create<Compilation_unit>(m_allocator, &m_arena, m_imdl,
                                         &m_symbol_table, fname,
                                         &m_comp_options, &m_messages,
                                         &m_builtin_type_map);
    return mi::base::make_handle(unit);
}

void Compiler::run(unsigned& err_count) {

    // Initialize the Distiller-specific BSDF node table.
    mi::mdl::Node_types::init();

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

    // Load standard libraries.
    load_builtins();

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

    // Shut down the Distiller-specific node table, so that multiple
    // Compiler instances can be created in series.
    mi::mdl::Node_types::exit();
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
        printer->print(static_cast<long int>(m->get_line()));
        printer->print(":");
        printer->print(static_cast<long int>(m->get_column()));
        printer->print(": ");
        printer->print(m->get_severity_str());
        printer->print(": ");
        printer->print(m->get_message());
        printer->print("\n");
    }
    if (cnt < total_message_cnt) {
        printer->print("... and ");
        printer->print(static_cast<long int>(total_message_cnt - cnt));
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
    mi::mdl::IType const *mdl_type,
    mi::mdl::IDefinition::Semantics sema)
{
    Builtin_type_map::iterator it = m_builtin_type_map.find(symbol);
    if (it != m_builtin_type_map.end()) {
        Type_ptr_list &vec = it->second.get_type_list();
        vec.push_back(mdl_type);
        if (m_comp_options.get_debug_builtin_loading()) {
            printf("[info] %s: was overloaded\n", symbol->get_name());
        }
    } else {
        mi::mdl::Allocator_builder builder(m_allocator);
        Type_ptr_list vec(m_allocator);
        vec.push_back(mdl_type);

        Builtin_entry entry(fq_symbol, vec, sema);

        std::pair<Symbol *, Builtin_entry> p(symbol, entry);

        std::pair<Builtin_type_map::iterator, bool> newly_inserted =
            m_builtin_type_map.insert(p);
        MDL_ASSERT(newly_inserted.second);
        (void) newly_inserted.second;
    }
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

        mi::mdl::IDefinition::Semantics sema = m_imdl->get_builtin_semantic(fq_symbol->get_name());

        // Register the type for all unqualified and qualified names.
        add_builtin(symbol, fq_symbol, type, sema);
        add_builtin(q_symbol, fq_symbol, type, sema);
        add_builtin(fq_symbol, fq_symbol, type, sema);
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
