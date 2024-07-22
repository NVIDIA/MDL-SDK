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

#define MI_TEST_AUTO_SUITE_NAME "Basic Test Suite prod/bin/mdltlc"
#define MI_TEST_IMPLEMENT_TEST_MAIN_INSTEAD_OF_MAIN

#include <base/system/test/i_test_auto_driver.h>
#include <base/system/test/i_test_auto_case.h>
#include <mi/mdl/mdl_mdl.h>

#include <mdl/compiler/compilercore/compilercore_mdl.h>
#include <mdl/compiler/compilercore/compilercore_assert.h>

#include <sys/stat.h>

#include <filesystem>
#include <fstream>

#include "mdltlc_compiler_options.h"
#include "mdltlc_compiler.h"

#define DIR_PREFIX "test_output"

namespace fs = std::filesystem;

static char const *success_files[] = {
    "000_empty.mdltl",
    "001_comments.mdltl",
    "002_empty_ruleset.mdltl",
    "003_simple.mdltl",
    "004_two_rules.mdltl",
    "005_where.mdltl",
    "006_postcond.mdltl",
    "007_complex_where.mdltl",
    "008_numbers.mdltl",
    "009_expr.mdltl",
    "010_mixer_norm.mdltl",
    "014_tc.mdltl",
    "016_dependent_where.mdltl",
    "017_create_constants.mdltl",
    "024_attr.mdltl",
    "025_local_normal.mdltl",
    "027_lm2000.mdltl",
    "028_attr_syntax.mdltl",
    "029_imports.mdltl",
    "030_custom_material.mdltl",
    "031_attr_binding.mdltl",
    "040_lhs_alias.mdltl",
    "041_complex_alias.mdltl",
    "043_mixer_error.mdltl",
    "044_debug.mdltl",
    "046_conditionals.mdltl"
};

static char const *success_generate_files[] = {
    "000_empty.mdltl",
    "001_comments.mdltl",
    "002_empty_ruleset.mdltl",
    "003_simple.mdltl",
    "004_two_rules.mdltl",
    "005_where.mdltl",
    "006_postcond.mdltl",
    "007_complex_where.mdltl",
    "008_numbers.mdltl",
    "009_expr.mdltl",
    "010_mixer_norm.mdltl",
    "013_mixer_norm_reorder.mdltl",
    "014_tc.mdltl",
    "016_dependent_where.mdltl",
    "017_create_constants.mdltl",
    "024_attr.mdltl",
    "025_local_normal.mdltl",
    "027_lm2000.mdltl",
    "028_attr_syntax.mdltl",
    "029_imports.mdltl",
    "030_custom_material.mdltl",
    "031_attr_binding.mdltl",
    "040_lhs_alias.mdltl",
    "041_complex_alias.mdltl",
    "043_mixer_error.mdltl",
    "044_debug.mdltl",
    "046_conditionals.mdltl"
};

struct Expected_failure_message {
    int m_message_index;
    char const *m_expected_error;

    Expected_failure_message(int message_index, const char *expected_error)
        : m_message_index(message_index)
        , m_expected_error(expected_error) {}
};

struct Expected_failure {
    char const *m_filename;
    int m_error_count;
    int m_warning_count;
    int m_info_count;
    int m_hint_count;
    Expected_failure_message *m_messages;
    int m_message_count;

    Expected_failure(char const *filename, int error_count, int warning_count,
                     int info_count, int hint_count,
                     Expected_failure_message *messages, int message_count)
        : m_filename(filename)
        , m_error_count(error_count)
        , m_warning_count(warning_count)
        , m_info_count(info_count)
        , m_hint_count(hint_count)
        , m_messages(messages)
        , m_message_count(message_count) {}
};

static Expected_failure_message failure_messages_1[] = {
    Expected_failure_message(0, "variable `a` cannot be redefined")
};

static Expected_failure_message failure_messages_2[] = {
    Expected_failure_message(0, "mix of variable and non-variable mixer parameters")
};

static Expected_failure_message failure_messages_3[] = {
    Expected_failure_message(0, "rule pattern is more general than later pattern"),
    Expected_failure_message(1, "this is the more specific later pattern")
};

static Expected_failure_message failure_messages_4[] = {
    Expected_failure_message(0, "normalized mixer by ordering arguments")
};

static Expected_failure_message failure_messages_5[] = {
    Expected_failure_message(0, "unknown name: x")
};

static Expected_failure_message failure_messages_6[] = {
    Expected_failure_message(0, "call in pattern must have bsdf or material return type")
};

// This does not include all messages, only the ones related to unused
// variables.
static Expected_failure_message failure_messages_7[] = {
    Expected_failure_message(30, "unused variable: c"),
    Expected_failure_message(31, "unused variable: c"),
    Expected_failure_message(32, "unused variable: c"),
    Expected_failure_message(33, "unused variable: c"),
    Expected_failure_message(34, "unused variable: c"),
    Expected_failure_message(35, "unused variable: e"),
    Expected_failure_message(36, "unused variable: e"),
    Expected_failure_message(37, "you can suppress warnings")
};

static Expected_failure_message failure_messages_8[] = {
    Expected_failure_message(0, "type mismatch for attribute: a")
};

static Expected_failure_message failure_messages_9[] = {
    Expected_failure_message(0, "literals are not allowed in patterns")
};

static Expected_failure_message failure_messages_10[] = {
    Expected_failure_message(0, "type mismatch in comparison operation")
};

static Expected_failure failure_files[] = {
    Expected_failure("015_where_invalid.mdltl", 1, 0, 0, 0,
                     failure_messages_1, sizeof(failure_messages_1) / sizeof(failure_messages_1[0])),
    Expected_failure("011_mixer_norm_invalid.mdltl", 0, 1, 0, 0,
                     failure_messages_2, sizeof(failure_messages_2) / sizeof(failure_messages_2[0])),
    Expected_failure("012_overlap_invalid.mdltl", 1, 0, 0, 1,
                     failure_messages_3, sizeof(failure_messages_3) / sizeof(failure_messages_3[0])),
    Expected_failure("013_mixer_norm_reorder.mdltl", 0, 0, 1, 0,
                     failure_messages_4, sizeof(failure_messages_4) / sizeof(failure_messages_4[0])),
    Expected_failure("019_undefined_vars.mdltl", 1, 0, 0, 0,
                     failure_messages_5, sizeof(failure_messages_5) / sizeof(failure_messages_5[0])),
    Expected_failure("020_pattern_invalid.mdltl", 1, 0, 0, 0,
                     failure_messages_6, sizeof(failure_messages_6) / sizeof(failure_messages_6[0])),

    // Note: the error and hint count includes messages about pattern
    // overlap, which we are not interested in for this test.
    Expected_failure("018_unused_vars.mdltl", 15, 7, 0, 16,
                     failure_messages_7, sizeof(failure_messages_7) / sizeof(failure_messages_7[0])),

    Expected_failure("026_attr_type_mismatch.mdltl", 1, 0, 0, 0,
                     failure_messages_8, sizeof(failure_messages_8) / sizeof(failure_messages_8[0])),

    Expected_failure("032_attr_const_pattern.mdltl", 1, 0, 0, 0,
                     failure_messages_9, sizeof(failure_messages_9) / sizeof(failure_messages_9[0])),

    Expected_failure("042_wrong_comparisons.mdltl", 1, 0, 0, 0,
                     failure_messages_10, sizeof(failure_messages_10) / sizeof(failure_messages_10[0]))
};

// Test the Compiler_options class.
MI_TEST_AUTO_FUNCTION( test_compiler_options )
{
    mi::base::Handle<mi::mdl::IMDL> imdl(mi::mdl::initialize(true));
    mi::mdl::IAllocator *allocator = imdl->get_mdl_allocator();
    mi::mdl::Memory_arena arena(allocator);

    Compiler_options opts(&arena);

    opts.set_generate(true);
    opts.set_all_errors(true);
    opts.set_debug_builtin_loading(true);
    opts.set_debug_dump_builtins(true);
    opts.set_verbosity(9);
    opts.add_filename("tests/testname1.mdltl");
    opts.add_filename("tests/testname2.mdltl");

    MI_CHECK_EQUAL(opts.get_generate(), true);
    MI_CHECK_EQUAL(opts.get_all_errors(), true);
    MI_CHECK_EQUAL(opts.get_debug_builtin_loading(), true);
    MI_CHECK_EQUAL(opts.get_debug_dump_builtins(), true);
    MI_CHECK_EQUAL(opts.get_verbosity(), 9);
    MI_CHECK_EQUAL(opts.get_filename_count(), 2);
    MI_CHECK_EQUAL(std::string(opts.get_filename(0)), "tests/testname1.mdltl");
    MI_CHECK_EQUAL(std::string(opts.get_filename(1)), "tests/testname2.mdltl");

    opts.set_generate(false);
    opts.set_all_errors(false);
    opts.set_debug_builtin_loading(false);
    opts.set_debug_dump_builtins(false);
    opts.set_verbosity(1);
    opts.add_filename("tests/testname3.mdltl");

    MI_CHECK_EQUAL(opts.get_generate(), false);
    MI_CHECK_EQUAL(opts.get_all_errors(), false);
    MI_CHECK_EQUAL(opts.get_debug_builtin_loading(), false);
    MI_CHECK_EQUAL(opts.get_debug_dump_builtins(), false);
    MI_CHECK_EQUAL(opts.get_verbosity(), 1);
    MI_CHECK_EQUAL(opts.get_filename_count(), 3);
    MI_CHECK_EQUAL(std::string(opts.get_filename(0)), "tests/testname1.mdltl");
    MI_CHECK_EQUAL(std::string(opts.get_filename(1)), "tests/testname2.mdltl");
    MI_CHECK_EQUAL(std::string(opts.get_filename(2)), "tests/testname3.mdltl");
}

// Create a compiler and run it in check-only mode on an empty mdltl
// file under tests/ as a basic smoke test. For each file, a new
// compiler is created.
MI_TEST_AUTO_FUNCTION( test_compiler_creation )
{
    mi::base::Handle<mi::mdl::IMDL> imdl(mi::mdl::initialize(true));
    mi::mdl::IAllocator *allocator = imdl->get_mdl_allocator();

    mi::mdl::Allocator_builder builder(allocator);

    mi::base::Handle<Compiler> compiler(builder.create<Compiler>(imdl.get()));

    std::string test_dir(MI::TEST::mi_src_path("prod/bin/mdltlc") + "/tests/");
    std::string filename(test_dir + "000_empty.mdltl");

    Compiler_options &comp_options = compiler->get_compiler_options();
    comp_options.add_filename(filename.c_str());

    unsigned err_count = 0;

    compiler->run(err_count);

    MI_CHECK_EQUAL(err_count, 0);
}

// Run compiler in check-only mode on all mdltl files in tests/
// directory where we expect success.
MI_TEST_AUTO_FUNCTION( test_expected_success )
{
    for (size_t i = 0; i < sizeof(success_files) / sizeof(success_files[0]); i++) {
        mi::base::Handle<mi::mdl::IMDL> imdl(mi::mdl::initialize(true));
        mi::mdl::IAllocator *allocator = imdl->get_mdl_allocator();

        mi::mdl::Allocator_builder builder(allocator);

        std::string test_dir(MI::TEST::mi_src_path("prod/bin/mdltlc") + "/tests/");

        std::string filename(test_dir + success_files[i]);

        mi::base::Handle<Compiler> compiler(builder.create<Compiler>(imdl.get()));

        Compiler_options &comp_options = compiler->get_compiler_options();
        comp_options.add_filename(filename.c_str());
        comp_options.add_mdl_path(test_dir.c_str());

        unsigned err_count = 0;

#if 0
        std::cerr << filename << "\n";
#endif
        compiler->run(err_count);

        MI_CHECK_EQUAL(err_count, 0);
    }
}

// Run compiler in check-only mode on all mdltl files in tests/
// directory where we diagnostics are expected. For each file, a new
// compiler is created.
MI_TEST_AUTO_FUNCTION( test_expected_failure )
{

    struct
    {
        bool operator()(char const* a, char const* b) const { return strcmp(a, b) < 0; }
    }
    char_ptr_less;

    for (size_t i = 0; i < sizeof(failure_files) / sizeof(failure_files[0]); i++) {
        mi::base::Handle<mi::mdl::IMDL> imdl(mi::mdl::initialize(true));
        mi::mdl::IAllocator *allocator = imdl->get_mdl_allocator();

        mi::mdl::Allocator_builder builder(allocator);

        std::string test_dir(MI::TEST::mi_src_path("prod/bin/mdltlc") + "/tests/");

        std::string filename(test_dir + failure_files[i].m_filename);

        mi::base::Handle<Compiler> compiler(builder.create<Compiler>(imdl.get()));

        Compiler_options &comp_options = compiler->get_compiler_options();
        comp_options.add_filename(filename.c_str());
        comp_options.set_silent(true);
        comp_options.set_normalize_mixers(true);
        comp_options.set_warn_overlapping_patterns(true);
        comp_options.set_warn_non_normalized_mixers(true);

        unsigned err_count = 0;

#if 0
        std::cerr << filename.c_str() << "\n";
#endif

        compiler->run(err_count);

        Message_list const &messages = compiler->get_messages();

        int act_error_count = 0;
        int act_warning_count = 0;
        int act_info_count = 0;
        int act_hint_count = 0;

        std::vector<char const *> sorted_messages;
        for (int i = 0; i < messages.size(); i++) {
            switch (messages[i]->get_severity()) {
            case Message::SEV_ERROR:
                act_error_count += 1;
                break;
            case Message::SEV_WARNING:
                act_warning_count += 1;
                break;
            case Message::SEV_INFO:
                act_info_count += 1;
                break;
            case Message::SEV_HINT:
                act_hint_count += 1;
                break;
            }
            sorted_messages.push_back(messages[i]->get_message());
        }
        MI_CHECK_EQUAL(act_error_count, failure_files[i].m_error_count);
        MI_CHECK_EQUAL(act_warning_count, failure_files[i].m_warning_count);
        MI_CHECK_EQUAL(act_info_count, failure_files[i].m_info_count);
        MI_CHECK_EQUAL(act_hint_count, failure_files[i].m_hint_count);

        std::sort(sorted_messages.begin(), sorted_messages.end(), char_ptr_less);
        for (size_t j = 0; j < failure_files[i].m_message_count; j++) {
            int exp_message_index = failure_files[i].m_messages[j].m_message_index;
            char const *exp_message = failure_files[i].m_messages[j].m_expected_error;

            char const *pos = strstr(sorted_messages[exp_message_index], exp_message);
            if (!pos) {
                int m = 0;
                for (auto s : sorted_messages) {
                    std::cout << "]]" << m << "[[ " << s << "\n";
                    m++;
                }
                std::cout << "[ERROR] Checking i: " << i << ", file: " <<
                    failure_files[i].m_filename
                          << ", j: " << j << ", exp_message_index: " << exp_message_index <<
                    ", exp_message: " << exp_message <<
                    ", sorted_messages[exp_message_index]: " <<
                    sorted_messages[exp_message_index] << "\n";
            }
            MI_CHECK(pos);
        }
    }
}

// Run compiler in check-only mode on all mdltl files in tests/
// directory where we expect success, but run the compiler on all
// files at once, not one per compiler creation.
MI_TEST_AUTO_FUNCTION( test_multi_files )
{
    mi::base::Handle<mi::mdl::IMDL> imdl(mi::mdl::initialize(true));
    mi::mdl::IAllocator *allocator = imdl->get_mdl_allocator();

    mi::mdl::Allocator_builder builder(allocator);

    std::string test_dir(MI::TEST::mi_src_path("prod/bin/mdltlc") + "/tests/");

    mi::base::Handle<Compiler> compiler(builder.create<Compiler>(imdl.get()));

    Compiler_options &comp_options = compiler->get_compiler_options();

    comp_options.add_mdl_path(test_dir.c_str());
    comp_options.set_silent(true);

    for (size_t i = 0; i < sizeof(success_files) / sizeof(success_files[0]); i++) {
        std::string filename(test_dir + success_files[i]);

        comp_options.add_filename(filename.c_str());

    }
    unsigned err_count = 0;

    compiler->run(err_count);

    MI_CHECK_EQUAL(err_count, 0);
}

bool compare_files(std::ifstream &golden_f, std::ifstream &under_test_f) {
    size_t golden_line_no = 1;
    size_t under_test_line_no = 1;
    std::string golden_line;
    std::string under_test_line;

    // First, skip copyrights.
    while (true) {
        bool golden_eof = !std::getline(golden_f, golden_line);
        if (golden_eof) {
            std::cerr << "error: line" << golden_line_no << ": missing start marker in golden file\n";
            return false;
        }
        if (golden_line == "// Generated by mdltlc") {
            break;
        }
        golden_line_no++;
    }

    while (true) {
        bool under_test_eof = !std::getline(under_test_f, under_test_line);
        if (under_test_eof) {
            std::cerr << "error: line" << under_test_line_no << ": missing start marker in under_test file\n";
            return false;
        }
        if (under_test_line == "// Generated by mdltlc") {
            break;
        }
        under_test_line_no++;
    }

    while (true) {
        bool golden_eof = !std::getline(golden_f, golden_line);
        bool under_test_eof = !std::getline(under_test_f, under_test_line);
        if (golden_eof && under_test_eof) {
            if (golden_f.eof() && under_test_f.eof()) {
                break;
            }
            if (golden_f.bad()) {
                std::cerr << "error: line" << golden_line_no << ": error reading golden file\n";
            }
            if (under_test_f.bad()) {
                std::cerr << "error: line" << under_test_line_no << ": error reading file under test\n";
            }
            return false;
        }
        if (golden_eof) {
            std::cerr << "error: line " << golden_line_no << ": golden file too short\n";
            return false;
        }
        if (under_test_eof) {
            std::cerr << "error: line " << under_test_line_no << ": file under test too short\n";
            return false;
        }
        if (golden_line != under_test_line) {
            std::cerr << "error: files differ\n";
            std::cerr << "  golden: line " << golden_line_no << ":    \"";
            for (auto c : golden_line) {
                if (::isprint(c)) {
                    std::cerr << c;
                } else {
                    std::cerr << std::hex << "\\x" << unsigned(c);
                }
            }
            std::cerr << "\"\n";
            std::cerr << "  under test: line " << under_test_line_no << ": \"";
            for (auto c : under_test_line) {
                if (::isprint(c)) {
                    std::cerr << c;
                } else {
                    std::cerr << std::hex << "\\x" << unsigned(c);
                }
            }
            std::cerr << "\"\n";
            return false;
        }
        golden_line_no++;
        under_test_line_no++;
    }
    return true;
}

// Run compiler in generation mode on all mdltl files in tests/
// directory where we expect success and compare them to their
// "golden" version.
MI_TEST_AUTO_FUNCTION( test_golden_files )
{
    fs::remove_all(DIR_PREFIX);
    fs::create_directory(DIR_PREFIX);
    fs::path cwd(fs::current_path());

    for (size_t i = 0; i < sizeof(success_generate_files) / sizeof(success_generate_files[0]); i++) {

        mi::base::Handle<mi::mdl::IMDL> imdl(mi::mdl::initialize(true));
        mi::mdl::IAllocator *allocator = imdl->get_mdl_allocator();

        mi::mdl::Allocator_builder builder(allocator);

        std::string test_dir(MI::TEST::mi_src_path("prod/bin/mdltlc") + "/tests/");

        std::string basename(success_generate_files[i]);
        std::string filename(test_dir + basename);
        std::string stemname = basename.substr(0, basename.rfind('.'));
        std::string golden_stemname = test_dir + "golden/" + stemname;
        std::string under_test_stemname = cwd.string() + "/" + DIR_PREFIX "/" + stemname;

        mi::base::Handle<Compiler> compiler(builder.create<Compiler>(imdl.get()));

        Compiler_options &comp_options = compiler->get_compiler_options();
        comp_options.add_filename(filename.c_str());
        comp_options.set_silent(true);
        comp_options.set_generate(true);
        comp_options.set_normalize_mixers(true);
        comp_options.set_output_dir(DIR_PREFIX);
        comp_options.add_mdl_path(test_dir.c_str());

        unsigned err_count = 0;

        compiler->run(err_count);

        MI_CHECK_EQUAL(err_count, 0);

        {
            std::ifstream f_golden ((golden_stemname + ".cpp"), std::ifstream::in);
            std::ifstream f_under_test ((under_test_stemname + ".cpp"), std::ifstream::in);
            bool result = compare_files(f_golden, f_under_test);
            if (!result) {
                printf(".cpp file comparison failed:\n");
                printf("golden file:     %s\n", (golden_stemname + ".cpp").c_str());
                printf("file under test: %s\n", (under_test_stemname + ".cpp").c_str());
            }
            MI_CHECK(result);
        }
        {
            std::ifstream f_golden ((golden_stemname + ".cpp"), std::ifstream::in);
            std::ifstream f_under_test ((under_test_stemname + ".cpp"), std::ifstream::in);
            bool result = compare_files(f_golden, f_under_test);
            if (!result) {
                printf(".h file comparison failed:\n");
                printf("golden file:     %s\n", (golden_stemname + ".h").c_str());
                printf("file under test: %s\n", (under_test_stemname + ".h").c_str());
            }


            MI_CHECK(result);
        }
    }
}

MI_TEST_MAIN_CALLING_TEST_MAIN();
