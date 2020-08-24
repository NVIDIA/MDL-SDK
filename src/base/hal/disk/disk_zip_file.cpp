/***************************************************************************************************
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
 **************************************************************************************************/

/// \file
/// \brief Sketchy support for zip files with the disk file interface
///
/// This is not complete yet and only a sketch. Will be implemented properly later.
/// Functions which are not supported yet will abort.

#include "pch.h"
#include "i_disk_zip_file.h"
#include "i_disk_file.h"

namespace MI {
namespace DISK {

Zip_file::Zip_file()
  : m_file(0)
{}


Zip_file::~Zip_file()
{
    close();
}


bool Zip_file::open(const std::string& path, IFile::Mode mode)
{
    if (mode != IFile::M_READ)
        return false;

    // Briefly open the file to get the size of the file and store it.
    File* file = new File();
    if (!file->open(path, mode))
    {
        delete file;
        return false;
    }
    m_file_size = file->filesize();
    delete file;

    m_file = gzopen(path.c_str(), get_mode_string(mode));
    if (m_file == NULL)
        return false;
    m_path = path;
    return true;
}


bool Zip_file::open(const char* path, IFile::Mode mode)
{
    if (!path)
        return false;
    return open(std::string(path), mode);
}


bool Zip_file::close()
{
    if (m_file != NULL)
        gzclose(m_file);

    m_file = NULL;
    return true;
}


Sint64 Zip_file::read(char* buf, Uint64 num)
{
    return gzread(m_file, buf, static_cast<unsigned int>(num));
}


bool Zip_file::eof() const
{
    return gzeof(m_file) ? true : false;
}


bool Zip_file::seek(Sint64 offset, int whence)
{
    gzseek(m_file, static_cast<long>(offset), whence);
    return true;
}


Sint64 Zip_file::tell()
{
    return (Sint64)gztell(m_file);
}


Sint64 Zip_file::filesize()
{
    return m_file_size;
}


bool Zip_file::is_open() const
{
    return m_file != NULL;
}


bool Zip_file::is_file() const
{
    return true;
}


const char* Zip_file::path() const
{
    return m_path.c_str();
}


int Zip_file::error() const
{
    return 0;
}


bool Zip_file::is_zipped_file() const
{
    return gzdirect(m_file) == 0;
}

}
}
