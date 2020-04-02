/******************************************************************************
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_GENERATOR_JIT_STREAMS_H
#define MDL_GENERATOR_JIT_STREAMS_H 1

#include "mdl/compiler/compilercore/compilercore_cc_conf.h"
#include "mdl/compiler/compilercore/compilercore_allocator.h"

#include <llvm/Support/raw_ostream.h>

namespace mi {
namespace mdl {

/// Wrapper to "stream" an LLVM object into a string.
class raw_string_ostream : public llvm::raw_ostream
{
    /// write_impl - See raw_ostream::write_impl.
    void write_impl(char const *Ptr, size_t Size) MDL_FINAL {
        // FIXME: slow implementation
        m_string.append(Ptr, Size);
    }

    /// current_pos - Return the current position within the stream, not
    /// counting the bytes currently in the buffer.
    uint64_t current_pos() const MDL_FINAL { return m_string.size(); }

public:
    explicit raw_string_ostream(string &str) : m_string(str) {}

    ~raw_string_ostream() { flush(); }

    /// str - Flushes the stream contents to the target string and returns
    ///  the string's reference.
    string &str() {
        flush();
        return m_string;
    }

private:
    string &m_string;
};

/// Implementation of the IOutput_stream interface.
class String_stream_writer : public mi::base::Interface_implement<mi::mdl::IOutput_stream> {
public:
    /// Constructor.
    /// \param str  String object which will be written to
    String_stream_writer(string &str)
    : m_string(str)
    {
    }

    /// Write a character to the output stream.
    void write_char(char c) MDL_FINAL { m_string.append(1, c); }

    /// Write a string to the stream.
    void write(const char *string) MDL_FINAL { m_string.append(string); }

    /// Flush stream.
    void flush() MDL_FINAL {}

    /// Remove the last character from output stream if possible.
    ///
    /// \param c  remove this character from the output stream
    ///
    /// \return true if c was the last character in the stream and it was successfully removed,
    /// false otherwise
    bool unput(char c) MDL_FINAL {
        size_t l = m_string.size();
        if (l > 0 && m_string[l - 1] == c) {
            m_string.erase(m_string.begin() + l - 1);
            return true;
        }
        return false;
    }

private:
    string &m_string;
};

}  // mdl
}  // mi

#endif // MDL_GENERATOR_JIT_STREAMS_H
