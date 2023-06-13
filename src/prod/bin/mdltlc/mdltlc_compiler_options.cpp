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

#include "mdltlc_compiler_options.h"

// Constructor.
Compiler_options::Compiler_options(mi::mdl::Memory_arena *arena)
    : m_arena(arena)
    , m_verbosity(0)
    , m_generate(false)
    , m_all_errors(false)
    , m_debug_builtin_loading(false)
    , m_debug_dump_builtins(false)
    , m_warn_non_normalized_mixers(false)
    , m_warn_overlapping_patterns(false)
    , m_normalize_mixers(false)
    , m_filenames(m_arena->get_allocator())
    , m_silent(false)
    , m_output_dir(nullptr)
    , m_mdl_path(m_arena->get_allocator())
{
}

void Compiler_options::set_verbosity(int verbosity) {
    MDL_ASSERT(verbosity >= 0 && verbosity < 10);
    m_verbosity = verbosity;
}

int Compiler_options::get_verbosity() const {
    return m_verbosity;
}

void Compiler_options::set_generate(bool generate) {
    m_generate = generate;
}

bool Compiler_options::get_generate() const {
    return m_generate;
}

void Compiler_options::set_all_errors(bool all_errors) {
    m_all_errors = all_errors;
}

bool Compiler_options::get_all_errors() const {
    return m_all_errors;
}

void Compiler_options::set_debug_builtin_loading(bool debug_builtin_loading) {
    m_debug_builtin_loading = debug_builtin_loading;
}

bool Compiler_options::get_debug_builtin_loading() const {
    return m_debug_builtin_loading;
}

void Compiler_options::set_debug_dump_builtins(bool debug_dump_builtins) {
    m_debug_dump_builtins = debug_dump_builtins;
}

bool Compiler_options::get_debug_dump_builtins() const {
    return m_debug_dump_builtins;
}

void Compiler_options::set_warn_non_normalized_mixers(bool warn_non_normalized_mixers) {
    m_warn_non_normalized_mixers = warn_non_normalized_mixers;
}

bool Compiler_options::get_warn_non_normalized_mixers() const {
    return m_warn_non_normalized_mixers;
}

void Compiler_options::set_warn_overlapping_patterns(bool warn_overlapping_patterns) {
    m_warn_overlapping_patterns = warn_overlapping_patterns;
}

bool Compiler_options::get_warn_overlapping_patterns() const {
    return m_warn_overlapping_patterns;
}

void Compiler_options::set_normalize_mixers(bool normalize_mixers) {
    m_normalize_mixers = normalize_mixers;
}

bool Compiler_options::get_normalize_mixers() const {
    return m_normalize_mixers;
}

void Compiler_options::add_filename(char const *filename) {
    m_filenames.push_back(mi::mdl::Arena_strdup(*m_arena, filename));
}

int Compiler_options::get_filename_count() const {
    return m_filenames.size();
}

char const *Compiler_options::get_filename(int index) const {
    MDL_ASSERT(index >= 0 && index < get_filename_count());
    return m_filenames[index];
}

void Compiler_options::set_silent(bool silent) {
    m_silent = silent;
}

bool Compiler_options::get_silent() const {
    return m_silent;
}

void Compiler_options::set_output_dir(char const *dirname) {
    m_output_dir = mi::mdl::Arena_strdup(*m_arena, dirname);
}

char const *Compiler_options::get_output_dir() const {
    return m_output_dir;
}

void Compiler_options::add_mdl_path(const char *dirname) {
    m_mdl_path.push_back(mi::mdl::Arena_strdup(*m_arena, dirname));
}

size_t Compiler_options::get_mdl_path_count() const {
    return m_mdl_path.size();
}

char const *Compiler_options::get_mdl_path(size_t index) const {
    MDL_ASSERT(index >= 0 && index < get_mdl_path_count());
    return m_mdl_path[index];
}
