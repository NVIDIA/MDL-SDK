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

#include <string>

#include <mi/base/handle.h>
#include <mdl/compiler/compilercore/compilercore_streams.h>

#include "mdltlc_compilation_unit.h"
#include "mdltlc_parser_coco.h"

#include "Scanner.h"
#include "Parser.h"

namespace {

class Coco_parse_error : public Errors {
public:
    Coco_parse_error()
        : Errors()
        , m_message()
        , m_line(1)
        , m_column(1)
        , m_count(0)
    {
    }

    void Error(Token const *token, char const *message) MDL_FINAL
    {
        record(message, token != nullptr ? token->line : 1, token != nullptr ? token->col : 1);
    }

    void Warning(int, int, char const *) MDL_FINAL
    {
    }

    void Error(int line, int column, int code) MDL_FINAL
    {
        std::string message("scanner error ");
        message += std::to_string(code);
        record(message.c_str(), unsigned(line), unsigned(column));
    }

    unsigned error_count() const { return m_count; }
    std::string const &message() const { return m_message; }
    unsigned line() const { return m_line; }
    unsigned column() const { return m_column; }

private:
    void record(char const *message, unsigned line, unsigned column)
    {
        ++m_count;
        if (!m_message.empty()) {
            return;
        }
        m_message = message != nullptr ? message : "legacy parser error";
        m_line = line;
        m_column = column;
    }

private:
    std::string m_message;
    unsigned    m_line;
    unsigned    m_column;
    unsigned    m_count;
};

}  // namespace

bool mdltlc_parse_coco(
    Compilation_unit &unit,
    char const       *source,
    size_t            length,
    std::string      &error_message,
    unsigned         &error_line,
    unsigned         &error_column)
{
    mi::mdl::Allocator_builder builder(unit.get_allocator());
    mi::base::Handle<mi::mdl::Buffer_Input_stream> input_stream(
        builder.create<mi::mdl::Buffer_Input_stream>(
            unit.get_allocator(),
            source,
            length,
            unit.get_filename()));

    Coco_parse_error error;
    Scanner scanner(unit.get_allocator(), &error, input_stream.get());
    Parser parser(&scanner, &error);
    parser.set_compilation_unit(&unit);
    parser.Parse();

    if (error.error_count() == 0) {
        return true;
    }

    error_message = error.message();
    error_line = error.line();
    error_column = error.column();
    return false;
}
