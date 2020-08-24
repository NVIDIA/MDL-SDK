/***************************************************************************************************
 * Copyright (c) 2003-2020, NVIDIA CORPORATION. All rights reserved.
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
 **************************************************************************************************/

/// \file
/// \brief The inline implementations.


namespace MI {
namespace DISK {

//-------------------------------------------------------- File ---------------
//
// return true if the file has been successfully opened. This is useful for
// debugging code that loops over all open files, and prints the open ones.
//

inline bool File::is_open() const
{
    return !!m_fp;
}


//
// return the path of the open file, or 0 if it isn't open.
//

inline const char *File::path() const
{
    return m_path.c_str();
}


//
// return last system error code
//

inline int File::error() const
{
    return m_error;
}


//
// return file pointer
//

inline FILE *File::get_file_pointer() const
{
    return m_fp;
}


//-------------------------------------------------------- Directory ----------
//
// return true if we are at the end of the directory. Unknown before reading or
// writing for the first time.
//

inline bool Directory::eof() const
{
    return m_eof;
}


//
// return the path of the open file, or 0 if it isn't open.
//

inline const char *Directory::path() const
{
    return m_path.c_str();
}


//
// return last system error code
//

inline int Directory::error() const
{
    return m_error;
}


template <typename InputIterator>
std::string find_file_on_paths(
    const char* file_name,
    InputIterator iter,
    InputIterator end)
{
    std::string fullpath = find_file_on_path(file_name);

    if (fullpath.empty()) {
        while (iter != end) {
            fullpath = find_file_on_path(file_name, *iter);
            if (!fullpath.empty())
                break;
            ++iter;
        }
    }
    return fullpath;
}

}
}
