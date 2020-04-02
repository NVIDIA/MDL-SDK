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
/// \file mi/mdl/mdl_streams.h
/// \brief Input and Output streams in the MDL Core API
#ifndef MDL_STREAMS_H
#define MDL_STREAMS_H 1

#include <mi/base/iinterface.h>
#include <mi/base/interface_declare.h>

namespace mi {
namespace mdl {

class IArchive_manifest;

/// The interface of an input stream.
class IInput_stream : public
    mi::base::Interface_declare<0xee6a9a18,0x654d,0x4d23,0xb3,0xb8,0xf0,0xc1,0x32,0xd8,0xd9,0x4e,
    mi::base::IInterface>
{
public:
    /// Read a character from the input stream.
    ///
    /// \returns    The code of the character read, or -1 on the end of the stream.
    virtual int read_char() = 0;

    /// Get the name of the file on which this input stream operates.
    ///
    /// \returns    The name of the file or NULL if the stream does not operate on a file.
    virtual char const *get_filename() = 0;
};

/// The interface of an input stream from an archive.
class IArchive_input_stream : public
    mi::base::Interface_declare<0x6cce8433,0xb727,0x4445,0x9b,0x9d,0x46,0xd8,0xa4,0x9f,0xf6,0x1e,
    IInput_stream>
{
public:
    /// Get the manifest of the owning archive.
    virtual IArchive_manifest const *get_manifest() const = 0;
};


/// The interface of an input stream from an mdle.
class IMdle_input_stream : public
    mi::base::Interface_declare<0x67e4a0f9,0xeee2,0x4987,0xa2,0x84,0xb8,0x2e,0xb7,0xf4,0x6a,0x92,
    IInput_stream>
{
};

/// The interface of an output stream.
class IOutput_stream : public
    mi::base::Interface_declare<0x1a7c81ca,0xf7f8,0x4db7,0x94,0xe2,0x22,0xcc,0x14,0xd4,0x1f,0xbe,
    mi::base::IInterface>
{
public:
    /// Write a character to the output stream.
    ///
    /// \param c  the character to be written
    virtual void write_char(char c) = 0;

    /// Write a C-string to the stream.
    ///
    /// \param string  the string to be written
    virtual void write(char const *string) = 0;

    /// Flush the stream.
    virtual void flush() = 0;

    /// Remove the last character from output stream if possible.
    ///
    /// \param c  remove this character from the output stream
    ///
    /// \return true if c was the last character in the stream and it was successfully removed,
    /// false otherwise
    virtual bool unput(char c) = 0;
};

/// The interface of an output stream that supports colored output.
class IOutput_stream_colored : public
    mi::base::Interface_declare<0xdbe72eeb,0xfdbb,0x487a,0x95,0x26,0xe0,0x70,0x8f,0x51,0x6c,0x7e,
    IOutput_stream>
{
public:
    /// supported colors
    enum Color {
        BLACK,
        RED,
        GREEN,
        YELLOW,
        BLUE,
        MAGENTA,
        CYAN,
        WHITE,
        DEFAULT
    };

    /// Returns true if this stream supports color output.
    virtual bool has_color() const = 0;

    /// Set the color for the next entity written to this stream.
    ///
    /// \param color      the color
    /// \param bold       if true, set bold
    /// \param background if true, set background color, else foreground color
    virtual void set_color(
        Color color,
        bool  bold = false,
        bool  background = false) = 0;

    /// Reset the output color to the default.
    virtual void reset_color() = 0;
};

/// The global output stream used for debug::* messages.
extern IOutput_stream *i_debug_log;

}  // mdl
}  // mi

#endif
