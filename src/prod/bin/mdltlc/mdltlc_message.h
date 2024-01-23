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

#ifndef MDLTLC_MESSAGE_H
#define MDLTLC_MESSAGE_H 1

#include <mi/mdl/mdl_mdl.h>

#include <mdl/compiler/compilercore/compilercore_memory_arena.h>

/// Representation of compiler messages (hints, infos, warnings and
/// errors).
class Message {
  public:
    enum Severity {
        SEV_HINT,
        SEV_INFO,
        SEV_WARNING,
        SEV_ERROR,
    };

  private:
    const Severity m_severity;
    const char *m_filename;
    const int m_line;
    const int m_column;
    const char *m_message;

  public:

    /// Constructor.
    Message(mi::mdl::Memory_arena *arena,
            Severity severity,
            const char *filename,
            int line,
            int column,
            const char *message);

    /// Return the message's severity;
    Severity get_severity() const { return m_severity; }

    /// Return a string representation of the message's severity.
    const char *get_severity_str() const;

    /// Return the message's file name;
    const char *get_filename() const { return m_filename; }

    /// Return the message's line number;
    int get_line() const { return m_line; }

    /// Return the message's column number;
    int get_column() const { return m_column; }

    /// Return the message's message;
    const char *get_message() const { return m_message; }
};

typedef mi::mdl::vector<Message*>::Type Message_list;

#endif
