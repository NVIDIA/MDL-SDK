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

#define MI_TEST_AUTO_SUITE_NAME "Real-World MDLTL Test Suite prod/bin/mdltlc"
#define MI_TEST_IMPLEMENT_TEST_MAIN_INSTEAD_OF_MAIN

#include <base/system/test/i_test_auto_driver.h>
#include <base/system/test/i_test_auto_case.h>
#include <mi/mdl/mdl_mdl.h>

#include <mdl/compiler/compilercore/compilercore_mdl.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "mdltlc_compiler.h"

namespace fs = std::filesystem;

static std::vector<fs::path> mdltl_files_in_real_world_dir()
{
    fs::path dir = fs::path(MI::TEST::mi_src_path("prod/bin/mdltlc")) /
        "tests" / "real_world";
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

static std::string read_file(fs::path const &path)
{
    std::ifstream file(path, std::ios::in | std::ios::binary);
    MI_CHECK(file.good());

    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

static std::string strip_nvidia_df_rules(std::string const &source)
{
    std::istringstream input(source);
    std::ostringstream output;
    std::string line;
    bool copy = true;

    while (std::getline(input, line)) {
        if (line.find("nvidia::df rules start") != std::string::npos) {
            copy = false;
        } else if (line.find("nvidia::df rules end") != std::string::npos) {
            copy = true;
        } else if (copy) {
            output << line << "\n";
        }
    }

    return output.str();
}

static fs::path write_stripped_real_world_file(fs::path const &source_path, fs::path const &out_dir)
{
    fs::create_directories(out_dir);

    fs::path out_path = out_dir / source_path.filename();
    std::ofstream file(out_path, std::ios::out | std::ios::binary | std::ios::trunc);
    MI_CHECK(file.good());
    file << strip_nvidia_df_rules(read_file(source_path));
    MI_CHECK(file.good());
    return out_path;
}

MI_TEST_AUTO_FUNCTION( test_real_world_mdltl_files_compile_without_errors )
{
    mi::base::Handle<mi::mdl::IMDL> imdl(mi::mdl::initialize(true));
    mi::mdl::Allocator_builder builder(imdl->get_mdl_allocator());
    mi::base::Handle<Compiler> compiler(builder.create<Compiler>(imdl.get()));

    Compiler_options &options = compiler->get_compiler_options();
    options.set_silent(true);
    options.set_all_errors(true);
    options.add_mdl_path(MI::TEST::mi_src_path("shaders/mdl").c_str());

    std::vector<fs::path> files = mdltl_files_in_real_world_dir();
    fs::path stripped_dir = fs::current_path() / "test_real_world_stripped";
    for (fs::path const &path : files) {
        fs::path stripped_path = write_stripped_real_world_file(path, stripped_dir);
        options.add_filename(stripped_path.string().c_str());
    }

    unsigned err_count = 0;
    compiler->run(err_count);

    Message_list const &messages = compiler->get_messages();
    for (Message const *message : messages) {
        if (message->get_severity() == Message::SEV_ERROR) {
            std::cout << message->get_filename() << ":"
                      << message->get_line() << ":"
                      << message->get_column() << ": "
                      << message->get_severity_str() << ": "
                      << message->get_message() << "\n";
            if (message->has_source_excerpt()) {
                std::cout << "    " << message->get_source_line() << "\n"
                          << "    " << message->get_source_underline() << "\n";
            }
        }
    }

    MI_CHECK_EQUAL(err_count, 0);
}

MI_TEST_MAIN_CALLING_TEST_MAIN();
