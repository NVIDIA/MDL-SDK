/******************************************************************************
 * Copyright (c) 2011-2020, NVIDIA CORPORATION. All rights reserved.
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
 ******************************************************************************/

#ifndef _MDLC_
#define _MDLC_ 1

#include <mi/base/handle.h>

#include <string>
#include <list>

namespace mi {
    namespace mdl {
        class IMDL;
        class IModule;
        class IGenerated_code;
        class ISyntax_coloring;
        class Options;
    }
}

/// The MDL command line compiler application.
class Mdlc
{
    /// Target languages.
    enum Target_language {
        TL_NONE,        ///< Don't output any target code.
        TL_MDL,         ///< MDL.
        TL_DAG,         ///< DAG
        TL_GLSL,        ///< GLSL
        TL_HLSL,        ///< HLSL
        TL_JIT,         ///< Executable machine code
        TL_PTX,         ///< PTX code
        TL_BIN          ///< binary output (serialization result)
    };

public:

    //! Constructor.
    ///
    /// \param program_name  The name of the command line application.
    Mdlc(char const *program_name);

    //! Destructor.
    ~Mdlc();

    //! Run the application.
    ///
    /// \param  argc    The argument count.
    /// \param  argv    The argument values.
    ///
    /// \returns    The number of compilation types that failed,
    ///             or a negative number if there was a problem
    ///             with the option settings.
    int run(int argc, char *argv[]);

private:
    /// Prints usage.
    void usage();

    /// Compile one module.
    /// \param      module_name     The name of the module to compile.
    /// \param      errors          The number of errors detected during compilation.
    /// \returns                    NULL: Some serious error occurred and no modules was created.
    ///                             The created module.
    mi::mdl::IModule const *compile(char const *module_name, size_t &errors);

    // Apply backend options.
    void apply_backend_options(mi::mdl::Options &opts);

    /// Compile a module to a target language.
    /// \param      module          The module to compile.
    /// \returns                    false: Some serious error occurred.
    ///                             true: compiled to target
    bool backend(mi::mdl::IModule const *module);

    /// Check if the given filename exists and if it represents a binary,
    ///
    /// \param filename  The name of the file to check.
    bool is_binary(char const *filename) const;

    /// Load a module binary.
    /// \param      filename        The name of the binary file.
    /// \param      errors          The number of errors detected during compilation.
    /// \returns                    NULL: Some serious error occurred and no modules was created.
    ///                             The created module.
    mi::mdl::IModule const *load_binary(
        char const *filename,
        size_t     &errors);

    /// Prints colorized code to stdout.
    void print_generated_code(mi::mdl::IModule const *mod);

    /// Prints colorized code to stdout.
    void print_generated_code(mi::mdl::IGenerated_code const *code);

    /// Find all modules in a library.
    void find_all_modules(char const *root, char const *package);

private:

    /// The program name.
    const char *m_program;

    /// If set, dump DAGs in the DAG backend.
    bool m_dump_dag;

    /// True if verbose mode enabled.
    bool m_verbose;

    /// True if syntax coloring is enabled.
    bool m_syntax_coloring;

    /// True if position should be printed.
    bool m_show_positions;

    /// The MDL compiler interface.
    mi::base::Handle<mi::mdl::IMDL> m_imdl;

    /// If non empty, a library root to check
    std::string m_check_root;

    /// The internal space to be used.
    std::string m_internal_space;

    typedef std::list<std::string> String_list;

    /// The list of backend options.
    String_list m_backend_options;

    /// The target language
    Target_language m_target_lang;

    /// The list of modules to compile.
    String_list m_input_modules;


    /// If set and target equals MDL, inline all imports except for stdlib/builtins
    bool m_inline;
};

#endif

