/******************************************************************************
 * Copyright (c) 2011-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <mi/mdl/mdl_messages.h>
#include <mi/base/interface_implement.h>

#include <string>
#include <vector>

#include "compilercore_def_table.h"
#include "compilercore_errors.h"
#include "compilercore_messages.h"
#include "compilercore_positions.h"
#include "compilercore_printers.h"
#include "compilercore_serializer.h"
#include "compilercore_streams.h"

namespace mi {
namespace mdl {

/// Inserts a message into a ordered message vector, using its line and position.
///
/// \param messages  the vector of messages
/// \param message   the message to insert
template<typename A>
static size_t insert_message(std::vector<Message *, A> &messages, Message *message)
{
    Position const *pos = message->get_position();
    size_t ID     = message->get_filename_id();
    int    line   = pos->get_start_line();
    int    column = pos->get_start_column();

    typename std::vector<Message *, A>::iterator it = messages.begin();

    for (int i = 0; it != messages.end(); ++i, ++it) {
        Message *msg = *it;
        Position const *pos = msg->get_position();

        if (msg->get_filename_id() < ID)
            continue;
        if ((msg->get_filename_id() == ID) && (pos->get_start_line() < line))
            continue;
        if ((msg->get_filename_id() == ID) &&
            (pos->get_start_line() == line) &&
            (pos->get_start_column() <= column))
            continue;

        // insert BEFORE the current one
        messages.insert(it, message);
        return i;
    }
    // insert last
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
char const *Message::get_file() const
{
    return m_owner->get_fname(m_filename_id);
}

// Get the message string.
char const *Message::get_string() const
{
    return m_msg;
}

/// Get he position to which the message is associated.
Position_impl const *Message::get_position() const
{
    return &m_pos;
}

// Get the number of notes attached to this message.
size_t Message::get_note_count() const
{
    return m_notes.size();
}

// Get the note at note_index attached to the message at message_index.
IMessage const *Message::get_note(size_t index) const
{
    return m_notes.at(index);
}

// Add a note to this message.
size_t Message::add_note(Message *note)
{
    return insert_message(m_notes, note);
}

// Get the message class.
char Message::get_class() const
{
    return m_class;
}

// Constructor.
Message::Message(
    Messages_impl  *owner,
    Severity       sev,
    int            code,
    char           msg_class,
    size_t         fname_id,
    Position const *pos,
    char const     *msg)
: Base()
, m_severity(sev)
, m_code(code)
, m_class(msg_class)
, m_filename_id(fname_id)
, m_pos(pos)
, m_msg(Arena_strdup(owner->get_msg_arena(), msg))
, m_owner(owner)
, m_notes(&owner->get_msg_arena())
{
}

Message::Message(
    Messages_impl  *owner,
    IMessage const *msg,
    size_t         fname_id)
: Base()
, m_severity(msg->get_severity())
, m_code(msg->get_code())
, m_class(msg->get_class())
, m_filename_id(fname_id)
, m_pos(msg->get_position())
, m_msg(Arena_strdup(owner->get_msg_arena(), msg->get_string()))
, m_owner(owner)
, m_notes(&owner->get_msg_arena())
{
}

// Get the number of messages.
size_t Messages_impl::get_message_count() const
{
    return m_msgs.size();
}

// Get the message at index.
IMessage const *Messages_impl::get_message(size_t index) const
{
    return m_msgs.at(index);
}

// Get number of error messages.
size_t Messages_impl::get_error_message_count() const
{
    return m_err.size();
}

// Get the error message at index.
IMessage const *Messages_impl::get_error_message(size_t index) const
{
    return m_err.at(index);
}

// Add a message.
size_t Messages_impl::add_message(
    IMessage::Severity severity,
    int                code,
    char               msg_class,
    char const         *str,
    char const         *file,
    int                start_line,
    int                start_column,
    int                end_line,
    int                end_column)
{
    Position_impl pos(start_line,start_column,end_line,end_column);
    return add_message(severity, code, msg_class, register_fname(file), &pos, str);
}

// Add a note to a message.
size_t Messages_impl::add_note(
    size_t             index,
    IMessage::Severity severity,
    int                code,
    char               msg_class,
    char const         *str,
    char const         *file,
    int                start_line,
    int                start_column,
    int                end_line,
    int                end_column)
{
    Position_impl pos(start_line,start_column,end_line,end_column);
    return add_note(index, severity, code, msg_class, register_fname(file), &pos, str);
}

// Add a message.
size_t Messages_impl::add_message(
    IMessage::Severity sev,
    int                code,
    char               msg_class,
    size_t             mod_id,
    Position const     *pos,
    char const         *str)
{
    Message *msg = m_builder.create<Message>(this, sev, code, msg_class, mod_id, pos, str);

    if (sev == Message::MS_ERROR)
        insert_message(m_err,msg);
    int index = insert_message(m_msgs,msg);
    return index;
}

// Add a note to a message.
size_t Messages_impl::add_note(
    size_t             message_index,
    IMessage::Severity sev,
    int                code,
    char               msg_class,
    size_t             mod_id,
    Position const     *pos,
    char const         *str)
{
    Message *note = m_builder.create<Message>(
        this, IMessage::MS_INFO, code, msg_class, mod_id, pos, str);
    m_msgs.at(message_index)->add_note(note);
    return 0;
}

// Add an error message.
size_t Messages_impl::add_error_message(
    int            code,
    char           msg_class,
    size_t         mod_id,
    Position const *pos,
    char const     *str)
{
    return add_message(Message::MS_ERROR, code, msg_class, mod_id, pos, str);
}

// Add a warning message.
size_t Messages_impl::add_warning_message(
    int            code,
    char           msg_class,
    size_t         mod_id,
    Position const *pos,
    char const     *str)
{
    return add_message(Message::MS_WARNING, code, msg_class, mod_id, pos, str);
}

// Add an informational message.
size_t Messages_impl::add_info_message(
    int            code,
    char           msg_class,
    size_t         mod_id,
    Position const *pos,
    char const     *str)
{
    return add_message(Message::MS_INFO, code, msg_class, mod_id, pos, str);
}

// Add an imported message to a message.
size_t Messages_impl::add_imported(
     size_t         message_index,
     size_t         fname_id,
     IMessage const *msg)
{
    Message *note = m_builder.create<Message>(this, msg, fname_id);
    m_msgs.at(message_index)->add_note(note);
    return 0;
}

// Format a message.
string Messages_impl::format_msg(int code, char msg_class, Error_params const &params)
{
    IAllocator *alloc = m_msg_arena.get_allocator();
    Allocator_builder builder(alloc);

    mi::base::Handle<Buffer_output_stream> os(builder.create<Buffer_output_stream>(alloc));
    mi::base::Handle<Printer> printer(builder.create<Printer>(alloc, os.get()));

    print_error_message(code, msg_class, params, printer.get());
    return string(os->get_data(), alloc);
}

// Register an external file name.
size_t Messages_impl::register_fname(
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

// Get the dynamic memory consumption of this message list.
size_t Messages_impl::get_dynamic_memory_consumption() const
{
    size_t res = m_msg_arena.get_chunks_size();
    res += dynamic_memory_consumption(m_msgs);
    res += dynamic_memory_consumption(m_err);
    res += dynamic_memory_consumption(m_filenames);

    return res;
}

// Clear messages.
void Messages_impl::clear_messages()
{
    m_msgs.clear();
}

// Returns a file name from the file name table.
char const *Messages_impl::get_fname(size_t id) const
{
    if (id < m_filenames.size())
        return m_filenames[id];
    return "<invalid>";
}

// Set a file name into the file name table.
void Messages_impl::set_fname(size_t id, char const *s)
{
    if (id < m_filenames.size())
        m_filenames[id] = Arena_strdup(m_msg_arena, s);
}

// Drop all messages.
void Messages_impl::clear()
{
    m_msg_arena.drop(NULL);
    m_msgs.clear();
    m_err.clear();
    m_filenames.clear();

    // must always have an owner name
    m_filenames.push_back("");
}

// Copy messages from another list to this.
void Messages_impl::copy_messages(Messages const &src)
{
    for (size_t i = 0, n = src.get_message_count(); i < n; ++i) {
        IMessage const *msg = src.get_message(i);

        char const *fname = msg->get_file();
        size_t mod_id = register_fname(fname);

        int idx = add_message(
            msg->get_severity(),
            msg->get_code(),
            msg->get_class(),
            mod_id,
            msg->get_position(),
            msg->get_string());

        for (size_t j = 0, m = msg->get_note_count(); j < m; ++j) {
            IMessage const *note = msg->get_note(j);

            char const *fname = note->get_file();
            size_t mod_id = register_fname(fname);

            add_note(
                idx,
                note->get_severity(),
                note->get_code(),
                note->get_class(),
                mod_id,
                note->get_position(),
                note->get_string());
        }
    }
}

// Serialize a message list.
template<typename A>
void Messages_impl::serialize_msg_list(
    Entity_serializer                 &serializer,
    std::vector<Message *, A> const &msgs) const
{
    size_t l = msgs.size();
    serializer.write_encoded_tag(l);

    for (size_t i = 0; i < l; ++i) {
        Message const *msg = msgs[i];

        serializer.write_unsigned(msg->m_severity);
        serializer.write_unsigned(msg->m_code);
        serializer.write_byte(msg->m_code);
        serializer.write_encoded_tag(msg->m_filename_id);
        msg->m_pos.serialize(serializer);
        serializer.write_cstring(msg->m_msg);
        // no need to serializer owner, will be automatically restored

        serialize_msg_list(serializer, msg->m_notes);
    }
}

// Deserialize a message list.
template<typename A>
void Messages_impl::deserialize_msg_list(
    Entity_deserializer         &deserializer,
    std::vector<Message *, A> &msgs)
{
    size_t l = deserializer.read_encoded_tag();
    msgs.reserve(l);

    for (size_t i = 0; i < l; ++i) {
        IMessage::Severity sev  = IMessage::Severity(deserializer.read_unsigned());
        int                code = deserializer.read_unsigned();
        char               cls  = deserializer.read_byte();
        size_t             f_id = deserializer.read_encoded_tag();
        Position_impl      pos(deserializer);
        string             s(deserializer.read_cstring(), deserializer.get_allocator());
        // no need to serializer owner, will be automatically restored

        Message *msg = m_builder.create<Message>(this, sev, code, cls, f_id, &pos, s.c_str());
        deserialize_msg_list(deserializer, msg->m_notes);

        msgs.push_back(msg);
    }
}

// Serialize the message list.
void Messages_impl::serialize(Entity_serializer &serializer) const
{
    serialize_msg_list(serializer, m_msgs);
    serialize_msg_list(serializer, m_err);

    size_t n_fnames = m_filenames.size();
    serializer.write_encoded_tag(n_fnames);

    for (size_t i = 0; i < n_fnames; ++i) {
        char const *fname = m_filenames[i];
        serializer.write_cstring(fname);
    }
}

// Deserialize this message list.
void Messages_impl::deserialize(Entity_deserializer &deserializer)
{
    deserialize_msg_list(deserializer, m_msgs);
    deserialize_msg_list(deserializer, m_err);

    size_t n_fnames = deserializer.read_encoded_tag();
    m_filenames.reserve(n_fnames);

    for (size_t i = 0; i < n_fnames; ++i) {
        char const *fname = deserializer.read_cstring();

        char *s = Arena_strdup(m_msg_arena, fname);
        m_filenames.push_back(s);
    }
}

// Constructor.
Messages_impl::Messages_impl(IAllocator *alloc, char const *owner_fname)
: Base()
, m_msg_arena(alloc)
, m_builder(m_msg_arena)
, m_msgs(alloc)
, m_err(alloc)
, m_filenames(alloc)
{
    char const *s = Arena_strdup(m_msg_arena, owner_fname);
    m_filenames.push_back(s);
}

// Create an error location from a position in the current module.
Err_location::Err_location(Position const &pos)
: m_pos(&pos), m_import_idx(0)
{
}

// Create an error location from a definition.
Err_location::Err_location(Definition const *def)
: m_pos(def->get_position()), m_import_idx(def->get_original_import_idx())
{
}

}  // mdl
}  // mi
