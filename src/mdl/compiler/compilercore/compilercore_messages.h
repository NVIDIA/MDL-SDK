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

#ifndef MDL_COMPILERCORE_MESSAGES_H
#define MDL_COMPILERCORE_MESSAGES_H 1

#include <mi/base/interface_implement.h>
#include <mi/mdl/mdl_messages.h>

#include "compilercore_cc_conf.h"
#include "compilercore_allocator.h"
#include "compilercore_memory_arena.h"
#include "compilercore_positions.h"
#include "compilercore_dynamic_memory.h"

namespace mi {
namespace mdl {

class Messages_impl;
class Message;
class Definition;
class Entity_serializer;
class Entity_deserializer;
class Error_params;

/// The type of vectors of messages.
typedef vector<Message *>::Type       Message_vector;
typedef Arena_vector<Message *>::Type Arena_message_vector;

/// Implementation of a Message.
class Message : public IMessage
{
    typedef IMessage Base;
    friend class Arena_builder;
    friend class Messages_impl;

public:

    /// Get the message severity.
    Severity get_severity() const MDL_FINAL;

    /// Get the message code.
    int get_code() const MDL_FINAL;

    /// Get the message string.
    char const *get_string() const MDL_FINAL;

    /// Get the file location.
    char const *get_file() const MDL_FINAL;

    /// Get he position to which the message is associated.
    Position_impl const *get_position() const MDL_FINAL;

    /// Get the number of notes attached to this message.
    /// \returns        The number of notes attached to this message.
    size_t get_note_count() const MDL_FINAL;

    /// Get the note at index attached to the message at message_index.
    /// \param index    The index of the note to get.
    /// \returns        The note.
    IMessage const *get_note(size_t index) const MDL_FINAL;

    /// Get the message class.
    char get_class() const MDL_FINAL;

    /// Add a note to this message.
    /// \param note     The note to add.
    /// \returns        The index of the added note.
    size_t add_note(Message *note);

    /// Get the file ID of this message.
    size_t get_filename_id() const { return m_filename_id; }

private:
    /// Constructor for a message.
    ///
    /// \param owner     the message list owner of the new message
    /// \param sev       the severity of the message
    /// \param code      the error code of the message
    /// \param msg_class the message class
    /// \param fname_id  the id for the file name of the message
    /// \param pos       the position of the message
    /// \param msg       the human readable message text
    explicit Message(
        Messages_impl  *owner,
        Severity       sev,
        int            code,
        char           msg_class,
        size_t         fname_id,
        Position const *pos,
        char const     *msg);

    /// Constructor for an imported message.
    ///
    /// \param owner     the message list owner of the new message
    /// \param msg       the imported message
    /// \param fname_id  the id for the file name of the imported message
    explicit Message(
        Messages_impl  *owner,
        IMessage const *msg,
        size_t         fname_id);

private:
    /// The severity of this message.
    Severity m_severity;

    /// The message code.
    int const m_code;

    /// The message class, 'C' for compiler.
    char const m_class;

    /// The id of the file name in the owner's file name table.
    size_t const m_filename_id;

    /// The position of this message.
    Position_impl const m_pos;

    /// The human readable message, stored on the message arena.
    char *const m_msg;

    /// The owner
    Messages_impl * const m_owner;

    /// The list of notes for this messages.
    Arena_message_vector m_notes;
};

/// Implementation of the Messages interface (serving as a factory and a store for IMessage's).
class Messages_impl : public Messages {
    typedef Messages Base;
    friend class Module;
public:

    /// Get the number of messages.
    size_t get_message_count() const MDL_FINAL;

    /// Get the message at index.
    IMessage const *get_message(size_t index) const MDL_FINAL;

    /// Get number of error messages.
    size_t get_error_message_count() const MDL_FINAL;

    /// Get the error message at index.
    IMessage const *get_error_message(size_t index) const MDL_FINAL;

    /// Add a message.
    ///
    /// \param severity        The severity of the message.
    /// \param code            The code of the message.
    /// \param msg_class       The class of the message.
    /// \param str             The message string.
    /// \param file            The file from which the message originates.
    /// \param start_line      The line on which the message starts.
    /// \param start_column    The column on which the message starts.
    /// \param end_line        The line on which the message ends.
    /// \param end_column      The column on which the message ends.
    /// \returns               The index of the message.
    size_t add_message(
        IMessage::Severity severity,
        int                code,
        char               msg_class,
        char const         *str,
        char const         *file,
        int                start_line,
        int                start_column,
        int                end_line,
        int                end_column) MDL_FINAL;

    /// Add a note to a message.
    ///
    /// \param index           The index of the message to which to add the note.
    /// \param severity        The severity of the message.
    /// \param code            The code of the message.
    /// \param msg_class       The class of the message.
    /// \param str             The message string.
    /// \param file            The file from which the message originates.
    /// \param start_line      The line on which the message starts.
    /// \param start_column    The column on which the message starts.
    /// \param end_line        The line on which the message ends.
    /// \param end_column      The column on which the message ends.
    /// \returns               The index of the message.
    size_t add_note(
        size_t             index,
        IMessage::Severity severity,
        int                code,
        char               msg_class,
        char const         *str,
        char const         *file,
        int                start_line,
        int                start_column,
        int                end_line,
        int                end_column) MDL_FINAL;

    // --------------- non-interface members ---------------

    /// Retrieve the message arena.
    Memory_arena &get_msg_arena() { return m_msg_arena; }

    /// Add a message.
    ///
    /// \param sev            severity of the message to add
    /// \param code           error code of the message
    /// \param msg_class      the class of the message
    /// \param mod_id         the module id for the message position
    /// \param pos            position of the message
    /// \param str            message text of the message
    ///
    /// \returns The index of the added message.
    size_t add_message(
        IMessage::Severity sev,
        int                code,
        char               msg_class,
        size_t             mod_id,
        Position const     *pos,
        char const         *str);

    /// Add a note to a message.
    ///
    /// \param message_index  the index of the message this note will be added
    /// \param sev            severity of the node to add
    /// \param code           error code of the note
    /// \param msg_class      the class of the note
    /// \param mod_id         the module id for the error position
    /// \param pos            position of the note
    /// \param str            message text of the note
    ///
    /// \returns The index of the added note.
    size_t add_note(
        size_t             message_index,
        IMessage::Severity sev,
        int                code,
        char               msg_class,
        size_t             mod_id,
        Position const     *pos,
        char const         *str);

    /// Add an error message.
    ///
    /// \param code           error code of the message
    /// \param msg_class      the class of the message
    /// \param mod_id         the module id for the error position
    /// \param pos            position of the message
    /// \param str            message text of the message
    ///
    /// \returns The index of the added message.
    size_t add_error_message(
        int            code,
        char           msg_class,
        size_t         mod_id,
        Position const *pos,
        char const     *str);

    /// Add a warning message.
    ///
    /// \param code           error code of the warning
    /// \param msg_class      the class of the warning
    /// \param mod_id         the module id for the warning position
    /// \param pos            position of the warning
    /// \param str            message text of the warning
    ///
    /// \returns The index of the added message.
    size_t add_warning_message(
        int            code,
        char           msg_class,
        size_t         mod_id,
        Position const *pos,
        char const     *str);

    /// Add an informational message.
    ///
    /// \param code           error code of the message
    /// \param msg_class      the class of the message
    /// \param mod_id         the module id for the message position
    /// \param pos            position of the message
    /// \param str            message text of the message
    ///
    /// \returns The index of the added message.
    size_t add_info_message(
        int            code,
        char           msg_class,
        size_t         mod_id,
        Position const *pos,
        const char     *str);

    /// Format a message.
    ///
    /// \param code           error code of the message
    /// \param msg_class      the class of the message
    /// \param params         the parameters for the message
    string format_msg(int code, char msg_class, Error_params const &params);

    /// Clear messages.
    void clear_messages();

    /// Returns a file name from the file name table.
    ///
    /// \note the zero id always returns the owner file name of the module
    char const *get_fname(size_t id) const;

    /// Set a file name into the file name table.
    void set_fname(size_t id, char const *s);

    /// Add an imported message to a message.
    ///
    /// \param message_index  the index of the message this note will be added
    /// \param fname_id       the id for the file name of the imported message
    /// \param msg            the message to add
    /// 
    /// \returns The index of the added msg.
    size_t add_imported(
        size_t         message_index,
        size_t         fname_id,
        IMessage const *msg);

    /// Register an external file name.
    ///
    /// \param filename  the name to add
    ///
    /// \return the file ID of this file name
    size_t register_fname(char const *filename);

    /// Get the number of registered filenames.
    size_t get_fname_count() const { return m_filenames.size(); }

    /// Get the dynamic memory consumption of this message list.
    size_t get_dynamic_memory_consumption() const;

    /// Drop all messages.
    ///
    /// \note This also sets the owner filename to "".
    void clear();

    /// Copy messages from another list to this.
    ///
    /// \param src  the original list
    void copy_messages(Messages const &src);

    /// Serialize the message list.
    ///
    /// \param serializer  an entity serializer
    void serialize(Entity_serializer &serializer) const;

    /// Deserialize this message list.
    ///
    /// \param deserializer  an entity deserializer
    void deserialize(Entity_deserializer &deserializer);

public:
    /// Constructor.
    ///
    /// \param alloc            the allocator
    /// \param owner_filename   the file name of the owner of this message list
    explicit Messages_impl(IAllocator *alloc, char const *owner_fname);

private:
    // non copyable
    Messages_impl(Messages_impl const &) MDL_DELETED_FUNCTION;
    Messages_impl &operator=(Messages_impl const &) MDL_DELETED_FUNCTION;

private:
    /// Serialize a message list.
    ///
    /// \param serializer  an entity serializer
    /// \param msgs        a message list
    template<typename A>
    void serialize_msg_list(
        Entity_serializer                 &serializer,
        std::vector<Message *, A> const &msgs) const;

    /// Deserialize a message list.
    ///
    /// \param deserializer  an entity deserializer
    /// \param msgs          a message list
    template<typename A>
    void deserialize_msg_list(
        Entity_deserializer         &deserializer,
        std::vector<Message *, A> &msgs);

private:
    /// The memory arena all messages are allocated on.
    Memory_arena m_msg_arena;

    /// The builder for messages.
    Arena_builder m_builder;

    /// The list of messages.
    Message_vector m_msgs;

    /// The list of errors only.
    Message_vector m_err;

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
    /* implicit */ Err_location(Position const &pos);

    /// Create an error location from a definition.
    /* implicit */ Err_location(Definition const *def);

    /// Get the error position.
    Position const *get_position() const { return m_pos; }

    /// Get the error module import index, 0 if the error location is in the current module.
    size_t get_module_import_idx() const { return m_import_idx; }

private:
    /// The position of the error location.
    Position const *m_pos;

    /// The module import index of the error location.
    size_t const m_import_idx;
};

// Helper shims for calculation the dynamic memory consumption
inline bool has_dynamic_memory_consumption(Messages_impl const &msg) { return true; }
inline size_t dynamic_memory_consumption(Messages_impl const &msgs) {
    return msgs.get_dynamic_memory_consumption();
}

}  // mdl
}  // mi

#endif // MDL_COMPILERCORE_MESSAGES_H
