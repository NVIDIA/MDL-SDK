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

#ifndef MDLTLC_COMPILER_OPTIONS_H
#define MDLTLC_COMPILER_OPTIONS_H 1

#include <mdl/compiler/compilercore/compilercore_memory_arena.h>

class Compiler_options {
    friend class mi::mdl::Arena_builder;
public:
    explicit Compiler_options(mi::mdl::Memory_arena *arena);

    /// Set the verbosity level.
    ///
    /// Must be an integer between 0 and 9.
    void set_verbosity(int verbosity);

    /// Return the verbosity level.
    int get_verbosity() const;

    /// Set to true to generate .h/.cpp files
    void set_generate(bool generate);

    /// Get the --generate flag.
    bool get_generate() const;

    /// Set to true to show all error messages. If false, only the
    /// first 10 messages are displayed.
    void set_all_errors(bool all_errors);

    /// Get the --all-errors flag.
    bool get_all_errors() const;

    /// Set to true to output debug messages for intrinsic loading.
    void set_debug_builtin_loading(bool debug_intrinsics);

    /// Get the --debug=builtin-loading flag.
    bool get_debug_builtin_loading() const;

    /// Set to true to dump the environment of builtin functions.
    void set_debug_dump_builtins(bool debug_dump_builtins);

    /// Get the --debug=dump-builtins flag.
    bool get_debug_dump_builtins() const;

    /// Set to true to warn for non-normalized mixer patterns
    void set_warn_non_normalized_mixers(bool warn_non_normalized_mixers);

    /// Get the --warn=non-normalized-mixers flag.
    bool get_warn_non_normalized_mixers() const;

    /// Set to true to warn for non-normalized mixer patterns
    void set_warn_overlapping_patterns(bool warn_overlapping_patterns);

    /// Get the --warn=overlapping-patterns flag.
    bool get_warn_overlapping_patterns() const;

    /// Set to true to enable mixer normalization.
    void set_normalize_mixers(bool normalize_mixers);

    /// Get the --normalize-mixers flag.
    bool get_normalize_mixers() const;

    /// Add the filename of an mdltl file to the options.
    void add_filename(const char *filename);

    /// Get the number of filenames that have been configured.
    int get_filename_count() const;

    /// Get the filename at the given index (0 <= index <
    /// get_filename_count()).
    char const *get_filename(int index) const;

    /// Set the silent flag, which suppresses error output. This is
    /// used internally for testing and not available from the command
    /// line.
    void set_silent(bool silent);

    /// Get the silent flag.
    bool get_silent() const;

    /// Set the output directory for generated files.
    void set_output_dir(char const *dirname);

    /// Get the value of the --output-dir flag.
    char const *get_output_dir() const;

    /// Add the directory name to the MDL search path.
    void add_mdl_path(const char *dirname);

    /// Return the number of MDL path elements.
    size_t get_mdl_path_count() const;

    /// Return the MDL path at index i.
    char const *get_mdl_path(size_t index) const;

private:
    // non copyable
    Compiler_options(Compiler_options const &) = delete;
    Compiler_options &operator=(Compiler_options const &) = delete;

private:
    /// Memory arena to use for allocating dynamic memory for this
    /// option object. Must outlive the compiler option object.
    mi::mdl::Memory_arena *m_arena;

    /// -v/--verbosity option.
    int m_verbosity { 0 };

    /// --generate flag.
    bool m_generate { false };

    /// --all-errors flag.
    bool m_all_errors { false };

    /// --debug=builtin-loading flag.
    bool m_debug_builtin_loading { false };

    /// --debug=dump-builtins flag.
    bool m_debug_dump_builtins { false };

    /// --warn=non-normalized-mixers flag.
    bool m_warn_non_normalized_mixers { false };

    /// --warn=overlapping-patterns flag.
    bool m_warn_overlapping_patterns { false };

    /// --normalize-mixers flag.
    bool m_normalize_mixers { false };

    /// Configured file names.
    mi::mdl::vector<char const *>::Type m_filenames;

    /// Silent flag. Not available from the command line.
    bool m_silent { false };

    /// --output-dir=DIR flag.
    char const *m_output_dir { nullptr };

    /// --mdl-path=DIR flag.
    mi::mdl::vector<char const *>::Type m_mdl_path;
};
#endif
