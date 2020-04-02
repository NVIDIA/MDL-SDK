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

#ifndef MDL_COMPILERCORE_STREAMS_H
#define MDL_COMPILERCORE_STREAMS_H 1

#include <mi/mdl/mdl_streams.h>
#include <cstdio>

#include "compilercore_cc_conf.h"
#include "compilercore_allocator.h"

namespace mi {
namespace mdl {

/// Implementation of the IInput_stream interface using FILE I/O.
class File_Input_stream : public Allocator_interface_implement<IInput_stream>
{
    typedef Allocator_interface_implement<IInput_stream> Base;
public:
    /// Read a character from the input stream.
    /// \returns    The code of the character read, or -1 on the end of the stream.
    int read_char() MDL_FINAL;

    /// Get the name of the file on which this input stream operates.
    /// \returns    The name of the file or null if the stream does not operate on a file.
    char const *get_filename() MDL_FINAL;

    /// Constructor.
    ///
    /// \param alloc             the allocator
    /// \param f                 the file handle
    /// \param close_at_destroy  if true, the file handle will be closed
    ///                          if this object is destroyed
    explicit File_Input_stream(
        IAllocator *alloc,
        FILE       *f,
        bool       close_at_destroy,
        char const *filename);

private:
    // non copyable
    File_Input_stream(File_Input_stream const &) MDL_DELETED_FUNCTION;
    File_Input_stream &operator=(File_Input_stream const &) MDL_DELETED_FUNCTION;

private:
    ~File_Input_stream() MDL_FINAL;

private:
    /// The file handle.
    FILE *m_file;

    /// Set if file must be closed at destroy time.
    bool m_close_at_destroy;

    /// The filename.
    string m_filename;
};

/// Implementation of the IInput_stream interface using a buffer.
class Buffer_Input_stream : public Allocator_interface_implement<IInput_stream>
{
    typedef Allocator_interface_implement<IInput_stream> Base;
public:
    /// Read a character from the input stream.
    /// \returns    The code of the character read, or -1 on the end of the stream.
    int read_char() MDL_OVERRIDE;

    /// Get the name of the file on which this input stream operates.
    /// \returns    The name of the file or null if the stream does not operate on a file.
    char const *get_filename() MDL_FINAL;

    /// Construct an input stream from a character buffer.
    /// Does NOT copy the buffer, so it must stay until the lifetime of the
    /// Input stream object!
    ///
    /// \param alloc     the allocator
    /// \param buffer    the character buffer
    /// \param length    the length of the buffer
    /// \param filename  the name of the buffer or NULL
    explicit Buffer_Input_stream(
        IAllocator *alloc,
        char const *buffer,
        size_t     length,
        char const *filename);

private:
    // non copyable
    Buffer_Input_stream(Buffer_Input_stream const &) MDL_DELETED_FUNCTION;
    Buffer_Input_stream &operator=(Buffer_Input_stream const &) MDL_DELETED_FUNCTION;

protected:
    ~Buffer_Input_stream() MDL_OVERRIDE;

private:
    /// Current position.
    char const *m_curr_pos;

    /// End position.
    char const *m_end_pos;

    /// The name of this stream.
    string m_file_name;
};

/// Implementation of the IInput_stream interface using an encrypted buffer.
class Encoded_buffer_Input_stream : public Buffer_Input_stream
{
    typedef Buffer_Input_stream Base;
public:
    /// Read a character from the input stream.
    /// \returns    The code of the character read, or -1 on the end of the stream.
    int read_char() MDL_FINAL;

    /// Construct an input stream from a character buffer.
    /// Does NOT copy the buffer, so it must stay until the lifetime of the
    /// Input stream object!
    ///
    /// \param alloc     the allocator
    /// \param buffer    the character buffer
    /// \param length    the length of the buffer
    /// \param filename  the name of the buffer or NULL
    explicit Encoded_buffer_Input_stream(
        IAllocator          *alloc,
        unsigned char const *buffer,
        size_t              length,
        char const          *filename,
        char const          *key = "P4ssW0rT");

private:
    ~Encoded_buffer_Input_stream() MDL_FINAL;

private:
    /// The name of this stream.
    string m_key;

    /// The current read index.
    size_t m_index;
};

/// Implementation of the IOutput_stream_colored interface using FILE I/O.
class File_Output_stream : public Allocator_interface_implement<IOutput_stream_colored>
{
    typedef Allocator_interface_implement<IOutput_stream_colored> Base;
public:

    /// Write a char to the stream.
    void write_char(char c) MDL_FINAL;

    /// Write a string to the stream.
    void write(char const *string) MDL_FINAL;

    /// Flush stream.
    void flush() MDL_FINAL;

    /// Remove the last character from output stream if possible.
    ///
    /// \param c  remove this character from the output stream
    ///
    /// \return true if c was the last character in the stream and it was successfully removed,
    /// false otherwise
    bool unput(char c) MDL_FINAL;

    /// Returns true if this stream supports color.
    bool has_color() const MDL_FINAL;

    /// Set the color.
    /// \param color      the color
    /// \param bold       if true, set bold
    /// \para background  if true, set background color, else foreground color
    void set_color(Color color, bool bold = false, bool background = false) MDL_FINAL;

    /// Reset the color to the default
    void reset_color() MDL_FINAL;

    /// Constructor.
    ///
    /// \param alloc             the allocator
    /// \param f                 the file handle
    /// \param close_at_destroy  if true, the file handle will be closed
    ///                          if this object is destroyed
    explicit File_Output_stream(IAllocator *alloc, FILE *f, bool close_at_destroy);

private:
    // non copyable
    File_Output_stream(File_Output_stream const &) MDL_DELETED_FUNCTION;
    File_Output_stream &operator=(File_Output_stream const &) MDL_DELETED_FUNCTION;

private:
    ~File_Output_stream();

private:
    /// The file handle.
    FILE *m_file;

    /// Set if file must be closed at destroy time.
    bool m_close_at_destroy;

    /// Set if this stream is attached to a console (or tty in Unix).
    bool m_console_attached;

    /// The current color.
    unsigned m_curr_color;
};

/// Implementation of the IOutput_stream_colored interface for the debug output stream.
class Debug_Output_stream : public Allocator_interface_implement<IOutput_stream>
{
    typedef Allocator_interface_implement<IOutput_stream> Base;
public:

    /// Write a char to the stream.
    void write_char(char c) MDL_FINAL;

    /// Write a string to the stream.
    void write(char const *string) MDL_FINAL;

    /// Flush stream.
    void flush() MDL_FINAL;

    /// Remove the last character from output stream if possible.
    ///
    /// \param c  remove this character from the output stream
    ///
    /// \return true if c was the last character in the stream and it was successfully removed,
    /// false otherwise
    bool unput(char c) MDL_FINAL;

    /// Constructor.
    ///
    /// \param alloc  the allocator to be used
    explicit Debug_Output_stream(IAllocator *alloc);

private:
    // non copyable
    Debug_Output_stream(Debug_Output_stream const &) MDL_DELETED_FUNCTION;
    Debug_Output_stream &operator=(Debug_Output_stream const &) MDL_DELETED_FUNCTION;

private:
    ~Debug_Output_stream();

private:
    /// The file handle.
    FILE *m_file;
};

/// Implementation of the IOutput_stream interface using a growing buffer.
class Buffer_output_stream : public Allocator_interface_implement<IOutput_stream>
{
    typedef Allocator_interface_implement<IOutput_stream> Base;
public:

    /// Write a char to the stream.
    void write_char(char c) MDL_FINAL;

    /// Write a string to the stream.
    void write(char const *string) MDL_FINAL;

    /// Flush stream.
    void flush() MDL_FINAL;

    /// Remove the last character from output stream if possible.
    ///
    /// \param c  remove this character from the output stream
    ///
    /// \return true if c was the last character in the stream and it was successfully removed,
    /// false otherwise
    bool unput(char c) MDL_FINAL;

    /// Retrieve the written data len.
    size_t get_data_size() const { return m_data_length; }

    /// Retrieve the buffer itself.
    char const *get_data() const { return m_data; }

    /// Retrieve the (writable) buffer itself.
    char *get_data() { return m_data; }

    /// Clear the buffer.
    void clear();

    /// Constructor.
    ///
    /// \param alloc  the allocator to be used
    explicit Buffer_output_stream(IAllocator *alloc);

private:
    ~Buffer_output_stream();

private:
    // non copyable
    Buffer_output_stream(Buffer_output_stream const &) MDL_DELETED_FUNCTION;
    Buffer_output_stream &operator=(Buffer_output_stream const &) MDL_DELETED_FUNCTION;

private:
    /// The data.
    char *m_data;

    /// The data length.
    size_t m_data_length;

    /// The data chunk length.
    size_t m_data_chunk_length;
};

}  // mdl
}  // mi

#endif
