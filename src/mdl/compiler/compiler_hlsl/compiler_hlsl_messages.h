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

#ifndef MDL_COMPILER_HLSL_MESSAGES_H
#define MDL_COMPILER_HLSL_MESSAGES_H 1

#include "mdl/compiler/compilercore/compilercore_allocator.h"
#include "mdl/compiler/compilercore/compilercore_memory_arena.h"
#include "compiler_hlsl_locations.h"

namespace mi {
namespace mdl {
namespace hlsl {

class Messages_impl;
class Message;
class Definition;

/// The type of vectors of messages.
typedef vector<Message *>::Type       Message_vector;
typedef Arena_vector<Message *>::Type Arena_message_vector;

class Messages;

/// Implementation of a compiler message.
class Message
{
    friend class mi::mdl::Arena_builder;
    friend class Messages;
public:
    /// The possible severities of messages.
    enum Severity {
        MS_ERROR,       ///< An error message.
        MS_WARNING,     ///< A warning message.
        MS_INFO         ///< An informational message.
    };

    /// Get the message severity.
    Severity get_severity() const;

    /// Get the message code.
    int get_code() const;

    /// Get the message string.
    char const *get_string() const;

    /// Get the file location.
    char const *get_file() const;

    /// Get the location to which the message is associated.
    Location const &get_location() const;

    /// Get the number of notes attached to this message.
    ///
    /// \returns        The number of notes attached to this message.
    size_t get_note_count() const;

    /// Get the note at index attached to the message at message_index.
    ///
    /// \param index    The index of the note to get.
    /// \returns        The note.
    Message *get_note(size_t index) const;

    /// Add a note to this message.
    ///
    /// \param note     The note to add.
    /// \returns        The index of the added note.
    size_t add_note(Message *note);

private:
    /// Constructor for a message.
    ///
    /// \param owner     the message list owner of the new message
    /// \param sev       the severity of the message
    /// \param code      the error code of the message
    /// \param fname_id  the id for the file name of the message
    /// \param loc       the location of the message
    /// \param msg       the human readable message text
    explicit Message(
        Messages       *owner,
        Severity       sev,
        int            code,
        size_t         fname_id,
        Location const *loc,
        char const     *msg);

private:
    /// The severity of this message.
    Severity m_severity;

    /// The message code.
    int const m_code;

    /// The id of the file name in the owner's file name table.
    size_t const m_filename_id;

    /// The location of this message.
    Location const m_loc;

    /// The human readable message, stored on the message arena.
    char const *const m_msg;

    /// The owner
    Messages * const m_owner;

    /// The list of notes for this messages.
    Arena_message_vector m_notes;
};

/// Implementation of the Messages interface (serving as a factory and a store for IMessage's).
class Messages
{
    friend class Compilation_unit;
    friend class Compiler;
public:
    /// Get the number of messages.
    size_t get_message_count() const;

    /// Get the message at index.
    Message *get_message(size_t index) const;

    /// Get number of error messages.
    size_t get_error_message_count() const;

    /// Get the error message at index.
    Message *get_error_message(size_t index) const;

    /// Retrieve the message arena.
    Memory_arena &get_msg_arena() { return m_msg_arena; }

    /// Add a message.
    ///
    /// \param severity  The severity of the message.
    /// \param code      The code of the message.
    /// \param str       The message string.
    /// \param file      The file from which the message originates.
    /// \param line      The line on which the message starts.
    /// \param column    The column on which the message starts.
    /// \returns         The index of the message.
    size_t add_message(
        Message::Severity severity,
        int               code,
        const char        *str,
        const char        *file,
        unsigned          line,
        unsigned          column);

    /// Add a note to a message.
    /// \param  index     The index of the message to which to add the note.
    /// \param  severity  The severity of the message.
    /// \param  code      The code of the message.
    /// \param  str       The message string.
    /// \param  file      The file from which the message originates.
    /// \param  line      The line on which the message starts.
    /// \param  column    The column on which the message starts.
    /// \returns          The index of the message.
    size_t add_note(
        size_t index,
        Message::Severity severity,
        int               code,
        const char        *str,
        const char        *file,
        unsigned          line,
        unsigned          column);

    /// Add a message.
    /// \param sev     severity of the message to add
    /// \param code    error code of the message
    /// \param file_id  the file id for the message location
    /// \param loc     location of the message
    /// \param str     message text of the message
    ///
    /// \returns The index of the added message.
    size_t add_message(
        Message::Severity sev,
        int               code,
        size_t            file_id,
        Location const    *loc,
        char const        *str);

    /// Add a note to a message.
    ///
    /// \param message_index  the index of the message this note will be added
    /// \param sev            severity of the node to add
    /// \param code           error code of the note
    /// \param file_id        the file id for the message location
    /// \param loc            location of the note
    /// \param str            message text of the note
    /// 
    /// \returns The index of the added note.
    size_t add_note(
        size_t            message_index,
        Message::Severity sev,
        int               code,
        size_t            file_id,
        Location const    *loc,
        char const        *str);

    /// Add an error message.
    ///
    /// \param code     error code of the message
    /// \param file_id  the file id for the message location
    /// \param loc      location of the message
    /// \param str      message text of the message
    /// 
    /// \returns The index of the added message.
    size_t add_error_message(
        int            code,
        size_t         file_id,
        Location const *loc,
        char const     *str);

    /// Add a warning message.
    ///
    /// \param code     error code of the warning
    /// \param file_id  the module id for the warning position
    /// \param loc      location of the warning
    /// \param str      message text of the warning
    /// 
    /// \returns The index of the added message.
    size_t add_warning_message(
        int            code,
        size_t         file_id,
        Location const *pos,
        char const     *str);

    /// Add an informational message.
    ///
    /// \param code     error code of the message
    /// \param file_id  the file id for the message location
    /// \param loc      location of the message
    /// \param str      message text of the message
    /// 
    /// \returns The index of the added message.
    size_t add_info_message(
        int            code,
        size_t         file_id,
        Location const *loc,
        const char     *str);

    /// Returns a file name from the file name table.
    ///
    /// \note the zero id always returns the owner file name of the module
    char const *get_fname(size_t id) const;

    /// Set a file name into the file name table.
    void set_fname(size_t id, char const *s);

    /// Register an external file name.
    ///
    /// \param filename  the name to add
    ///
    /// \return the file ID of this file name
    size_t register_fname(char const *filename);

    /// Constructor.
    explicit Messages(IAllocator *alloc, char const *owner_fname);

private:
    /// The memory arena all messages are allocated on.
    Memory_arena m_msg_arena;

    /// The builder for messages.
    Arena_builder m_builder;

    /// The list of messages.
    Arena_message_vector m_msgs;

    /// The list of errors only.
    Arena_message_vector m_err;

    typedef vector<char const *>::Type Filename_vec;

    /// The file name table.
    Filename_vec m_filenames;
};

///
/// Represents an error location. A tuple of a Position and the module ID
/// of this position.
///
class Err_location {
public:
    /// Create an error location from a position in the current module.
    /* implicit */ Err_location(Location const &pos);

    /// Create an error location from a definition.
    /* implicit */ Err_location(Definition const *def);

    /// Get the error location.
    Location const *get_location() const { return m_loc; }

private:
    /// The location of the error location.
    Location const *m_loc;
};

}  // hlsl
}  // mdl
}  // mi

#endif // MDL_COMPILER_HLSL_MESSAGES_H
