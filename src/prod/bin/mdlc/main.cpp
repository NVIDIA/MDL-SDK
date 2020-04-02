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

#include "mdlc.h"

#ifdef MI_PLATFORM_WINDOWS

#include <string>
#include <vector>

#include <mi/base/miwindows.h>

namespace {

template<typename T>
class Array_holder {
public:
    Array_holder(size_t size) : m_arr(new T[size]) {}
    ~Array_holder() { delete[] m_arr; }
    T &operator[](size_t index) { return m_arr[index]; }
    T *data() { return m_arr; }
private:
    T *m_arr;
};
}

int wmain(int argc, wchar_t *argv[])
{
    std::vector<std::string> utf8_args(argc);
    Array_holder<char *> argv_utf8(argc);

    for (int i = 0; i < argc; ++i) {
        wchar_t *warg = argv[i];
        DWORD size = WideCharToMultiByte(
            CP_UTF8,
            0,
            warg,
            -1,
            NULL,
            0,
            NULL,
            NULL);
        if (size > 0) {
            std::string res(size, '\0');
            DWORD result = WideCharToMultiByte(
                CP_UTF8,
                0,
                warg,
                -1,
                const_cast<char *>(res.c_str()),
                size,
                NULL,
                NULL);
            if (result == size) {
                utf8_args[i] = res;
                argv_utf8[i] = const_cast<char *>(utf8_args[i].c_str());
            } else {
                argv_utf8[i] = const_cast<char *>("");
            }
        }
    }
    SetConsoleOutputCP(CP_UTF8);

    Mdlc the_app(argv_utf8[0]);
    return the_app.run(argc, argv_utf8.data());
}

#else

int main(int argc, char *argv[])
{
    Mdlc the_app(argv[0]);
    return the_app.run(argc, argv);
}
#endif
