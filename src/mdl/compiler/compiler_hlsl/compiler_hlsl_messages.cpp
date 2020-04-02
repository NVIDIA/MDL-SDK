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

#include "pch.h"

#include <string>
#include <vector>

#include "compiler_hlsl_definitions.h"
#include "compiler_hlsl_messages.h"
#include "compiler_hlsl_locations.h"

namespace mi {
namespace mdl {
namespace hlsl {

static size_t insert_message(Arena_message_vector &messages, Message *message)
{
    Location const &loc = message->get_location();
    unsigned line = loc.get_line();
    unsigned column = loc.get_column();
    Arena_message_vector::iterator it = messages.begin();
    for (size_t i = 0; it != messages.end(); ++i, ++it) {
        Message *msg = *it;
        Location const &loc= msg->get_location();
        if (loc.get_line() < line)
            continue;
        if (loc.get_line() == line && loc.get_column() <= column)
            continue;
        // insert AFTER it
        messages.insert(it, message);
        return i;
    }
    messages.push_back(message);
    return messages.size() - 1;
}

// Get the message severity.
Message::Severity Message::get_severity() const
{
    return m_severity;
}

// Get the message code.
int Message::get_code() const
{
    return m_code;
}

// Get the file location.
const char *Message::get_file() const
{
    return m_owner->get_fname(m_filename_id);
}

// Get the message string.
const char *Message::get_string() const
{
    return m_msg;
}

/// Get the location to which the message is associated.
Location const &Message::get_location() const
{
    return m_loc;
}

// Get the number of notes attached to this message.
size_t Message::get_note_count() const
{
    return m_notes.size();
}

// Get the note at note_index attached to the message at message_index.
Message *Message::get_note(size_t index) const
{
    return m_notes.at(index);
}

// Add a note to this message.
size_t Message::add_note(Message *note)
{
    return insert_message(m_notes, note);
}

/// Constructor.
Message::Message(
    Messages  *owner,
    Severity       sev,
    int            code,
    size_t         fname_id,
    Location const *loc,
    char const     *msg)
: m_severity(sev)
, m_code(code)
, m_filename_id(fname_id)
, m_loc(*loc)
, m_msg(Arena_strdup(owner->get_msg_arena(), msg))
, m_owner(owner)
, m_notes(&owner->get_msg_arena())
{
}

// Get the number of messages.
size_t Messages::get_message_count() const
{
    return m_msgs.size();
}

// Get the message at index.
Message *Messages::get_message(size_t index) const
{
    return m_msgs.at(index);
}

// Get number of error messages.
size_t Messages::get_error_message_count() const
{
    return m_err.size();
}

// Get the error message at index.
Message *Messages::get_error_message(size_t index) const
{
    return m_err.at(index);
}

// Add a message.
size_t Messages::add_message(
    Message::Severity severity,
    int               code,
    const char        *str,
    const char        *file,
    unsigned          line,
    unsigned          column)
{
    Location loc(line, column);
    return add_message(severity, code, register_fname(file), &loc, str);
}

// Add a note to a message.
size_t Messages::add_note(
    size_t            index,
    Message::Severity severity,
    int               code,
    const char        *str,
    const char        *file,
    unsigned          line,
    unsigned          column)
{
    Location loc(line, column);
    return add_note(index, severity, code, register_fname(file), &loc, str);
}

// Add a message.
size_t Messages::add_message(
    Message::Severity sev,
    int               code,
    size_t            file_id,
    Location const    *loc,
    char const        *str)
{
    Message *msg = m_builder.create<Message>(this, sev, code, file_id, loc, str);

    if (sev == Message::MS_ERROR)
        insert_message(m_err, msg);
    return insert_message(m_msgs, msg);
}

// Add a note to a message.
size_t Messages::add_note(
    size_t            message_index,
    Message::Severity sev,
    int               code,
    size_t            file_id,
    Location const    *pos,
    char const        *str)
{
    Message *note = m_builder.create<Message>(this, Message::MS_INFO, code, file_id, pos, str);
    return m_msgs.at(message_index)->add_note(note);
}

// Add an error message.
size_t Messages::add_error_message(
    int            code,
    size_t         file_id,
    Location const *pos,
    const char     *str)
{
    return add_message(Message::MS_ERROR, code, file_id, pos, str);
}

// Add a warning message.
size_t Messages::add_warning_message(
    int            code,
    size_t         file_id,
    Location const *pos,
    const char     *str)
{
    return add_message(Message::MS_WARNING, code, file_id, pos, str);
}

// Add an informational message.
size_t Messages::add_info_message(
    int            code,
    size_t         file_id,
    Location const *pos,
    const char     *str)
{
    return add_message(Message::MS_INFO, code, file_id, pos, str);
}

// Register an external file name.
size_t Messages::register_fname(
    char const *filename)
{
    // slow linear search is not expected to by a bottle neck here
    size_t n = m_filenames.size();
    for (size_t i = 0; i < n; ++i) {
        if (strcmp(filename, m_filenames[i]) == 0)
            return i;
    }

    char *s = Arena_strdup(m_msg_arena, filename);
    m_filenames.push_back(s);
    return n;
}

// Returns a file name from the file name table.
char const *Messages::get_fname(size_t id) const
{
    if (id < m_filenames.size())
        return m_filenames[id];
    return "<invalid>";
}

// Set a file name into the file name table.
void Messages::set_fname(size_t id, char const *s)
{
    if (id < m_filenames.size())
        m_filenames[id] = Arena_strdup(m_msg_arena, s);
}


Messages::Messages(IAllocator *alloc, char const *owner_fname)
:m_msg_arena(alloc)
, m_builder(m_msg_arena)
, m_msgs(&m_msg_arena)
, m_err(&m_msg_arena)
, m_filenames(alloc)
{
    char const *s = Arena_strdup(m_msg_arena, owner_fname);
    m_filenames.push_back(s);
}

// ------------------------ Error location ------------------------

// Create an error location from a position in the current module.
Err_location::Err_location(Location const &loc)
: m_loc(&loc)
{
}

// Create an error location from a definition.
Err_location::Err_location(Definition const *def)
: m_loc(def->get_location())
{
}

}  // hlsl
}  // mdl
}  // mi
