/******************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
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

#define MI_TEST_AUTO_SUITE_NAME "Parser Test Suite prod/bin/mdltlc"
#define MI_TEST_IMPLEMENT_TEST_MAIN_INSTEAD_OF_MAIN

#include <base/system/test/i_test_auto_driver.h>
#include <base/system/test/i_test_auto_case.h>
#include <mi/mdl/mdl_mdl.h>

#include <mdl/compiler/compilercore/compilercore_mdl.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "mdltlc_compiler.h"
#include "mdltlc_ast_compare.h"
#include "mdltlc_parser_coco.h"
#include "mdltlc_parser_rd.h"

namespace fs = std::filesystem;

struct Parse_result {
    bool parsed;
    std::string message;
    unsigned line;
    unsigned column;

    Parse_result()
        : parsed(false)
        , message()
        , line(0)
        , column(0)
    {
    }
};

enum Parser_kind {
    PK_LEGACY_COCO,
    PK_RECURSIVE_DESCENT
};

static std::string read_file(fs::path const &path)
{
    std::ifstream file(path, std::ios::in | std::ios::binary);
    MI_CHECK(file.good());

    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

static std::vector<fs::path> mdltl_files_in(char const *subdir)
{
    fs::path dir = fs::path(MI::TEST::mi_src_path("prod/bin/mdltlc")) / "tests" / subdir;
    MI_CHECK(fs::is_directory(dir));

    std::vector<fs::path> files;
    for (fs::directory_entry const &entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".mdltl") {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    MI_CHECK(!files.empty());
    return files;
}

static char const *parser_name(Parser_kind parser)
{
    switch (parser) {
    case PK_LEGACY_COCO:
        return "legacy Coco/R";
    case PK_RECURSIVE_DESCENT:
        return "recursive descent";
    }
    return "<unknown>";
}

static Parse_result parse_source(
    Compilation_unit  &unit,
    std::string const &source,
    Parser_kind        parser)
{
    Parse_result result;
    if (parser == PK_LEGACY_COCO) {
        result.parsed = mdltlc_parse_coco(
            unit,
            source.data(),
            source.size(),
            result.message,
            result.line,
            result.column);
        return result;
    }

    result.parsed = mdltlc_parse_rd(
        unit,
        source.data(),
        source.size(),
        result.message,
        result.line,
        result.column);
    return result;
}

MI_TEST_AUTO_FUNCTION( test_parser_ok_files )
{
    mi::base::Handle<mi::mdl::IMDL> imdl(mi::mdl::initialize(true));
    mi::mdl::Allocator_builder builder(imdl->get_mdl_allocator());
    mi::base::Handle<Compiler> compiler(builder.create<Compiler>(imdl.get()));

    for (fs::path const &path : mdltl_files_in("ok")) {
        std::string source = read_file(path);
        mi::base::Handle<Compilation_unit> legacy_unit =
            compiler->create_unit(path.string().c_str());
        mi::base::Handle<Compilation_unit> rd_unit =
            compiler->create_unit(path.string().c_str());
        Parse_result legacy = parse_source(*legacy_unit.get(), source, PK_LEGACY_COCO);
        Parse_result rd = parse_source(*rd_unit.get(), source, PK_RECURSIVE_DESCENT);

        for (Parser_kind parser : { PK_LEGACY_COCO, PK_RECURSIVE_DESCENT }) {
            Parse_result const &result = parser == PK_LEGACY_COCO ? legacy : rd;
            if (!result.parsed) {
                std::cout << parser_name(parser) << " parser rejected ok fixture "
                          << path.string()
                          << " at " << result.line << ":" << result.column
                          << ": " << result.message << "\n";
            }
            MI_CHECK(result.parsed);
        }
        MI_CHECK_EQUAL(legacy.parsed, rd.parsed);
        if (legacy.parsed && rd.parsed) {
            Mdltlc_ast_compare_result compare_result;
            bool equal = mdltlc_compare_asts(
                legacy_unit->get_rulesets(),
                rd_unit->get_rulesets(),
                compare_result);
            if (!equal) {
                std::cout << "AST mismatch for ok fixture " << path.string()
                          << ": " << compare_result.message() << "\n";
            }
            MI_CHECK(equal);
        }
    }
}

MI_TEST_AUTO_FUNCTION( test_parser_error_files )
{
    mi::base::Handle<mi::mdl::IMDL> imdl(mi::mdl::initialize(true));
    mi::mdl::Allocator_builder builder(imdl->get_mdl_allocator());
    mi::base::Handle<Compiler> compiler(builder.create<Compiler>(imdl.get()));

    for (fs::path const &path : mdltl_files_in("error")) {
        std::string source = read_file(path);
        mi::base::Handle<Compilation_unit> legacy_unit =
            compiler->create_unit(path.string().c_str());
        mi::base::Handle<Compilation_unit> rd_unit =
            compiler->create_unit(path.string().c_str());
        Parse_result legacy = parse_source(*legacy_unit.get(), source, PK_LEGACY_COCO);
        Parse_result rd = parse_source(*rd_unit.get(), source, PK_RECURSIVE_DESCENT);

        if (legacy.parsed != rd.parsed) {
            std::cout << "parser disagreement for error fixture " << path.string()
                      << ": legacy=" << legacy.parsed
                      << ", recursive_descent=" << rd.parsed << "\n";
            if (!legacy.parsed) {
                std::cout << "legacy error at " << legacy.line << ":" << legacy.column
                          << ": " << legacy.message << "\n";
            }
            if (!rd.parsed) {
                std::cout << "recursive descent error at " << rd.line << ":" << rd.column
                          << ": " << rd.message << "\n";
            }
        }

        for (Parser_kind parser : { PK_LEGACY_COCO, PK_RECURSIVE_DESCENT }) {
            Parse_result const &result = parser == PK_LEGACY_COCO ? legacy : rd;
            if (result.parsed) {
                std::cout << parser_name(parser) << " parser accepted error fixture "
                          << path.string() << "\n";
            }
            MI_CHECK(!result.parsed);
            MI_CHECK(!result.message.empty());
        }
        MI_CHECK_EQUAL(legacy.parsed, rd.parsed);
    }
}

MI_TEST_AUTO_FUNCTION( test_compiler_error_message_has_source_excerpt )
{
    mi::base::Handle<mi::mdl::IMDL> imdl(mi::mdl::initialize(true));
    mi::mdl::Allocator_builder builder(imdl->get_mdl_allocator());
    mi::base::Handle<Compiler> compiler(builder.create<Compiler>(imdl.get()));

    fs::path path = fs::path(MI::TEST::mi_src_path("prod/bin/mdltlc")) /
        "tests" / "error" / "bad_hex_without_digits.mdltl";

    Compiler_options &options = compiler->get_compiler_options();
    options.add_filename(path.string().c_str());
    options.set_silent(true);

    unsigned err_count = 0;
    compiler->run(err_count);

    Message_list const &messages = compiler->get_messages();
    MI_CHECK(err_count > 0);
    MI_CHECK(!messages.empty());

    Message const *message = messages[0];
    MI_CHECK(message->has_source_excerpt());
    MI_CHECK(std::strstr(message->get_source_line(), "result(0x)") != nullptr);

    std::string underline(message->get_source_underline());
    MI_CHECK(underline.find('^') != std::string::npos);
    MI_CHECK_EQUAL(underline.find('^'), size_t(message->get_column() - 1));
}

MI_TEST_MAIN_CALLING_TEST_MAIN();
