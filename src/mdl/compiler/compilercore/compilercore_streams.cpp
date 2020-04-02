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

#include <cstring>
#include "compilercore_streams.h"
#include "compilercore_wchar_support.h"

using namespace mi::mdl;

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <io.h>
#include <wincon.h>

static bool is_console_attached(int fd)
{
    return isatty(fd) != 0;
}

static bool set_color_need_flush() { return true; }

static char const *reset_color(int fd, unsigned color)
{
    SetConsoleTextAttribute((HANDLE)_get_osfhandle(fd), (WORD)color);

    return NULL;
}

static unsigned get_current_color(int fd)
{
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    if (GetConsoleScreenBufferInfo((HANDLE)_get_osfhandle(fd), &csbi))
        return csbi.wAttributes;
    return 0;
}


static char const *set_color(
    int                           fd,
    IOutput_stream_colored::Color color,
    bool                          bold,
    bool                          bg,
    unsigned                      def_color)
{
    WORD attr = get_current_color(fd);

    if (bg) {
        attr &= ~(BACKGROUND_RED | BACKGROUND_GREEN | BACKGROUND_BLUE | BACKGROUND_INTENSITY);
        switch (color) {
        case IOutput_stream_colored::BLACK:
            break;
        case IOutput_stream_colored::RED:
            attr |= BACKGROUND_RED;
            break;
        case IOutput_stream_colored::GREEN:
            attr |= BACKGROUND_GREEN;
            break;
        case IOutput_stream_colored::YELLOW:
            attr |= BACKGROUND_GREEN | BACKGROUND_RED;
            break;
        case IOutput_stream_colored::BLUE:
            attr |= BACKGROUND_BLUE;
            break;
        case IOutput_stream_colored::MAGENTA:
            attr |= BACKGROUND_BLUE | BACKGROUND_RED;
            break;
        case IOutput_stream_colored::CYAN:
            attr |= BACKGROUND_BLUE | BACKGROUND_GREEN;
            break;
        case IOutput_stream_colored::WHITE:
            attr |= BACKGROUND_RED | BACKGROUND_GREEN | BACKGROUND_BLUE;
            break;
        case IOutput_stream_colored::DEFAULT:
            attr |= def_color & (BACKGROUND_RED | BACKGROUND_GREEN | BACKGROUND_BLUE);
            break;
        }
        if (bold)
            attr |= BACKGROUND_INTENSITY;
    } else {
        attr &= ~(FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY);
        switch (color) {
        case IOutput_stream_colored::BLACK:
            break;
        case IOutput_stream_colored::RED:
            attr |= FOREGROUND_RED;
            break;
        case IOutput_stream_colored::GREEN:
            attr |= FOREGROUND_GREEN;
            break;
        case IOutput_stream_colored::YELLOW:
            attr |= FOREGROUND_GREEN | FOREGROUND_RED;
            break;
        case IOutput_stream_colored::BLUE:
            attr |= FOREGROUND_BLUE;
            break;
        case IOutput_stream_colored::MAGENTA:
            attr |= FOREGROUND_BLUE | FOREGROUND_RED;
            break;
        case IOutput_stream_colored::CYAN:
            attr |= FOREGROUND_BLUE | FOREGROUND_GREEN;
            break;
        case IOutput_stream_colored::WHITE:
            attr |= FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE;
            break;
        case IOutput_stream_colored::DEFAULT:
            attr |= def_color & (FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
            break;
        }
        if (bold)
            attr |= FOREGROUND_INTENSITY;
    }

    return reset_color(fd, attr);
}

#else

#include <unistd.h>
#include <cstdlib>

static bool is_console_attached(int fd)
{
    if (isatty(fd) == 0)
        return false;
    // do not color "dump" terminals
    if (char const *term = getenv("TERM")) {
        return strcmp(term, "dumb") != 0;
    }
    return true;
}

static bool set_color_need_flush() { return false; }

static char const *reset_color(int fd, unsigned color)
{
    return "\033[0m";
}

static unsigned get_current_color(int fd)
{
    return 0;
}

static char const *set_color(
    int                           fd,
    IOutput_stream_colored::Color color,
    bool                          bold,
    bool                          bg,
    unsigned                      def_color)
{
    if (bold) {
        if (bg) {
            switch (color) {
            case IOutput_stream_colored::BLACK:
                return "\033[0;1;40m";
            case IOutput_stream_colored::RED:
                return "\033[0;1;41m";
            case IOutput_stream_colored::GREEN:
                return "\033[0;1;42m";
            case IOutput_stream_colored::YELLOW:
                return "\033[0;1;43m";
            case IOutput_stream_colored::BLUE:
                return "\033[0;1;44m";
            case IOutput_stream_colored::MAGENTA:
                return "\033[0;1;45m";
            case IOutput_stream_colored::CYAN:
                return "\033[0;1;46m";
            case IOutput_stream_colored::WHITE:
                return "\033[0;1;47m";
            case IOutput_stream_colored::DEFAULT:
                return NULL;
            }
        } else {
            switch (color) {
            case IOutput_stream_colored::BLACK:
                return "\033[0;1;30m";
            case IOutput_stream_colored::RED:
                return "\033[0;1;31m";
            case IOutput_stream_colored::GREEN:
                return "\033[0;1;32m";
            case IOutput_stream_colored::YELLOW:
                return "\033[0;1;33m";
            case IOutput_stream_colored::BLUE:
                return "\033[0;1;34m";
            case IOutput_stream_colored::MAGENTA:
                return "\033[0;1;35m";
            case IOutput_stream_colored::CYAN:
                return "\033[0;1;36m";
            case IOutput_stream_colored::WHITE:
                return "\033[0;1;37m";
            case IOutput_stream_colored::DEFAULT:
                return NULL;
            }
        }
    } else {
        if (bg) {
            switch (color) {
            case IOutput_stream_colored::BLACK:
                return "\033[0;40m";
            case IOutput_stream_colored::RED:
                return "\033[0;41m";
            case IOutput_stream_colored::GREEN:
                return "\033[0;42m";
            case IOutput_stream_colored::YELLOW:
                return "\033[0;43m";
            case IOutput_stream_colored::BLUE:
                return "\033[0;44m";
            case IOutput_stream_colored::MAGENTA:
                return "\033[0;45m";
            case IOutput_stream_colored::CYAN:
                return "\033[0;46m";
            case IOutput_stream_colored::WHITE:
                return "\033[0;47m";
            case IOutput_stream_colored::DEFAULT:
                return NULL;
            }
        } else {
            switch (color) {
            case IOutput_stream_colored::BLACK:
                return "\033[0;30m";
            case IOutput_stream_colored::RED:
                return "\033[0;31m";
            case IOutput_stream_colored::GREEN:
                return "\033[0;32m";
            case IOutput_stream_colored::YELLOW:
                return "\033[0;33m";
            case IOutput_stream_colored::BLUE:
                return "\033[0;34m";
            case IOutput_stream_colored::MAGENTA:
                return "\033[0;35m";
            case IOutput_stream_colored::CYAN:
                return "\033[0;36m";
            case IOutput_stream_colored::WHITE:
                return "\033[0;37m";
            case IOutput_stream_colored::DEFAULT:
                return NULL;
            }
        }
    }
    return NULL;
}

#endif


namespace mi {
namespace mdl {

// Read a character from the input stream.
int File_Input_stream::read_char()
{
    return fgetc(m_file);
}

// Get the name of the file on which this input stream operates.
char const *File_Input_stream::get_filename()
{
    return m_filename.empty() ? 0 : m_filename.c_str();
}

// Constructor.
File_Input_stream::File_Input_stream(
    IAllocator *alloc,
    FILE       *f,
    bool       close_at_destroy,
    char const *filename)
: Base(alloc)
, m_file(f)
, m_close_at_destroy(close_at_destroy)
, m_filename(filename, alloc)
{
}

// Destructor.
File_Input_stream::~File_Input_stream()
{
    if (m_close_at_destroy)
        fclose(m_file);
}

// Read a character from the input stream.
int Buffer_Input_stream::read_char()
{
    return m_curr_pos < m_end_pos ? (unsigned char)*m_curr_pos++ : -1;
}

// Get the name of the file on which this input stream operates.
char const *Buffer_Input_stream::get_filename()
{
    return m_file_name.empty() ? NULL : m_file_name.c_str();
}

// Constructor.
Buffer_Input_stream::Buffer_Input_stream(
    IAllocator *alloc,
    char const *buffer,
    size_t     length,
    char const *filename)
: Base(alloc)
, m_curr_pos(buffer)
, m_end_pos(buffer + length)
, m_file_name(filename, alloc)
{
}

// Destructor.
Buffer_Input_stream::~Buffer_Input_stream()
{
}

// Constructor.
Encoded_buffer_Input_stream::Encoded_buffer_Input_stream(
    IAllocator          *alloc,
    unsigned char const *buffer,
    size_t              length,
    char const          *filename,
    char const          *key)
: Base(alloc, (char const *)buffer, length, filename)
, m_key(key, alloc)
, m_index(0)
{
}

// Destructor.
Encoded_buffer_Input_stream::~Encoded_buffer_Input_stream()
{
}


// Read a character from the input stream.
int Encoded_buffer_Input_stream::read_char()
{
    int c = Base::read_char();

    if (c != -1) {
        c = c ^ ((unsigned char)m_index) ^ m_key[m_index % m_key.size()];
        ++m_index;
        return c & 0xFF;
    }
    return -1;
}

// Write a char to the stream.
void File_Output_stream::write_char(char c)
{
#ifdef WIN32
    if (m_console_attached) {
        DWORD dummy;

        WriteConsoleA((HANDLE)_get_osfhandle(fileno(m_file)), &c, 1, &dummy, NULL);
        return;
    }
#endif
    fputc(c, m_file);
}

// Write a string to the stream.
void File_Output_stream::write(char const *string)
{
#ifdef WIN32
    if (m_console_attached) {
        DWORD dummy;

        wstring t(get_allocator());
        utf8_to_utf16(t, string);
        WriteConsoleW((HANDLE)_get_osfhandle(fileno(m_file)), t.c_str(), t.length(), &dummy, NULL);
        return;
    }
#endif
    fwrite(string, strlen(string), 1, m_file);
}

// Flush stream.
void File_Output_stream::flush()
{
    fflush(m_file);
}

// Remove the last character from output stream if possible.
bool File_Output_stream::unput(char c)
{
    // unsupported
    return false;
}

// File output streams do not support color for now.
bool File_Output_stream::has_color() const
{
    return m_console_attached;
}

// Set the color.
void File_Output_stream::set_color(Color color, bool bold, bool background)
{
    if (! has_color())
        return;

    if (set_color_need_flush())
        flush();

    if (char const *p = ::set_color(fileno(m_file), color, bold, background, m_curr_color))
        write(p);
}

// Reset the color to the default
void File_Output_stream::reset_color()
{
    if (! has_color())
        return;

    if (set_color_need_flush())
        flush();

    if (char const *p = ::reset_color(fileno(m_file), m_curr_color))
        write(p);
}

// Constructor.
File_Output_stream::File_Output_stream(IAllocator *alloc, FILE *f, bool close_at_destroy)
: Base(alloc)
, m_file(f)
, m_close_at_destroy(close_at_destroy)
, m_console_attached(is_console_attached(fileno(f)))
, m_curr_color(m_console_attached ? get_current_color(fileno(f)) : 0)
{
}

// Destructor.
File_Output_stream::~File_Output_stream()
{
    if (m_close_at_destroy)
        fclose(m_file);
}

// ------------------------------------ Debug output ------------------------------------

// Write a char to the stream.
void Debug_Output_stream::write_char(char c)
{
#ifdef _WIN32
    char buf[2] = { c, '\0' };
    OutputDebugString(buf);
#else
    fwrite(&c, 1, 1, m_file);
#endif
}

// Write a string to the stream.
void Debug_Output_stream::write(char const *string)
{
#ifdef _WIN32
    OutputDebugString(string);
#else
    fwrite(string, strlen(string), 1, m_file);
#endif
}

// Flush stream.
void Debug_Output_stream::flush()
{
#ifdef _WIN32
    // no flush
#else
    fflush(m_file);
#endif
}

// Remove the last character from output stream if possible.
bool Debug_Output_stream::unput(char c)
{
    // unsupported
    return false;
}

// Constructor.
Debug_Output_stream::Debug_Output_stream(IAllocator *alloc)
: Base(alloc)
#ifdef _WIN32
, m_file(NULL)
#else
, m_file(stderr)
#endif
{
}

// Destructor.
Debug_Output_stream::~Debug_Output_stream()
{
}

// Write a char to the stream.
void Buffer_output_stream::write_char(char c)
{
    if (m_data == NULL) {
        m_data_chunk_length = 1024;

        m_data = reinterpret_cast<char *>(get_allocator()->malloc(m_data_chunk_length));
    }

    size_t len = m_data_length;
    m_data[len++] = c;

    if (len >= m_data_chunk_length) {
        m_data_chunk_length += 1024;
        IAllocator *alloc = get_allocator();
        char *buf = reinterpret_cast<char *>(alloc->malloc(m_data_chunk_length));
        memcpy(buf, m_data, len);

        char *t = m_data;
        m_data = buf;
        alloc->free(t);
    }
    m_data_length = len;
}

// Write a string to the stream.
void Buffer_output_stream::write(char const *string)
{
    if (m_data == NULL) {
        m_data_chunk_length = 1024;
        
        m_data = reinterpret_cast<char *>(get_allocator()->malloc(m_data_chunk_length));
    }

    bool end = false;
    size_t len = m_data_length;
    for (size_t i = 0; !end; ++i) {
        m_data[len++] = string[i];

        if (len >= m_data_chunk_length) {
            m_data_chunk_length += 1024;
            IAllocator *alloc = get_allocator();
            char *buf = reinterpret_cast<char *>(alloc->malloc(m_data_chunk_length));
            memcpy(buf, m_data, len);

            char *t = m_data;
            m_data = buf;
            alloc->free(t);
        }
        end = string[i] == '\0';
    }
    // '\0' is always written but NOT counted
    m_data_length = len - 1;
}

// Flush stream.
void Buffer_output_stream::flush()
{
    // always flushed, do nothing
}

// Remove the last character from output stream if possible.
bool Buffer_output_stream::unput(char c)
{
    if (m_data_length > 0 && m_data[m_data_length - 1] == c) {
        --m_data_length;
        return true;
    }
    return false;
}

// Clear the buffer.
void Buffer_output_stream::clear()
{
    m_data_length = 0;
    if (m_data)
        m_data[0] = '\0';
}

// Constructor.
Buffer_output_stream::Buffer_output_stream(IAllocator *alloc)
: Base(alloc)
, m_data(NULL)
, m_data_length(0)
, m_data_chunk_length(0)
{
}

// Destructor.
Buffer_output_stream::~Buffer_output_stream()
{
    if (m_data != NULL) {
        get_allocator()->free(m_data);
        m_data = NULL;
    }
}

// The global debug logger.
IOutput_stream *i_debug_log = NULL;

}  // mdl
}  // mi
