/***************************************************************************************************
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
 **************************************************************************************************/
/// \file neuray_factory.h
/// \brief Load and unload for SDK shared library

#ifndef MI_NEURAY_NEURAY_FACTORY_H
#define MI_NEURAY_NEURAY_FACTORY_H

#include <mi/base.h>

#include <cstdio>
#include <cstdlib>

#ifdef MI_PLATFORM_WINDOWS
#include <mi/base/miwindows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

namespace mi {

    namespace neuraylib {

        class INeuray;

        /// Loads the shared library and calls the main factory function at construction time,
        /// and releases the SDK at the end of the scope in the destructor.
        ///
        class Neuray_factory {
        public:
            /// Enum type to encode possible error results of the INeuray interface creation.
            enum Result {
                /// The INeuray interface was successfully created.
                RESULT_SUCCESS = 0,
                /// The shared library failed to load.
                RESULT_LOAD_FAILURE,
                /// The shared library does not contain the expected \c mi_factory symbol.
                RESULT_SYMBOL_LOOKUP_FAILURE,
                /// The requested INeuray interface has a different IID than the ones 
                /// that can be served by the \c mi_factory function.
                RESULT_VERSION_MISMATCH,
                /// The requested INeuray interface cannot be served by the \c mi_factory 
                /// function and neither can the IVersion interface for better diagnostics.
                RESULT_INCOMPATIBLE_LIBRARY,
                //  Undocumented, for alignment only
                RESULT_FORCE_32_BIT = 0xffffffffU
            };

            /// The constructor loads the shared library, locates and calls the 
            /// #mi_factory() function. It store an instance of the main 
            /// #mi::neuraylib::INeuray interface for later access. 
            ///
            /// \param filename    The file name of the DSO. If \c NULL, the built-in
            ///                    default name of the SDK library is used.
            /// \param logger      Interface to report any errors during construction as well
            ///                    as during destruction. The logger interface needs to have 
            ///                    a suitable lifetime. If \c NULL, no error diagnostic will 
            ///                    be reported. The result code can be used for a diagnostic 
            ///                    after the construction.
            Neuray_factory(mi::base::ILogger* logger = 0,
                const char*        filename = 0);


            /// Returns the result code of loading the shared library. If the return value
            /// is one of #RESULT_LOAD_FAILURE or #RESULT_SYMBOL_LOOKUP_FAILURE on a Windows
            /// operating system, a call to \c GetLastError can provide more detail.
            Result get_result_code() const { return m_result_code; }

            /// Returns the pointer to an instance of the main #mi::neuraylib::INeuray 
            /// interface if loading the shared library was successful, or \c NULL otherwise.
            /// Does not retain the interface.
            mi::neuraylib::INeuray* get() const {
                return m_neuray.get();
            }

            /// Releases the #mi::neuraylib::INeuray interface and unloads the shared library.
            ~Neuray_factory();

        private:
            mi::base::Handle<mi::base::ILogger>      m_logger;
            Result                                   m_result_code;
            const char*                              m_filename;
            void*                                    m_dso_handle;
            mi::base::Handle<mi::neuraylib::INeuray> m_neuray;
        };


        // Inline implementation to make this helper class completely client code

        // printf() format specifier for arguments of type LPTSTR (Windows only).
#ifdef MI_PLATFORM_WINDOWS
#ifdef UNICODE
#define FMT_LPTSTR "%ls"
#else // UNICODE
#define FMT_LPTSTR "%s"
#endif // UNICODE
#endif // MI_PLATFORM_WINDOWS

        inline Neuray_factory::Neuray_factory(mi::base::ILogger* logger, const char* filename)
            : m_logger(logger, mi::base::DUP_INTERFACE),
            m_result_code(RESULT_SUCCESS),
            m_filename(0),
            m_dso_handle(0),
            m_neuray(0)
        {
            if (!filename)
#ifdef MI_MDL_SDK_H
                filename = "libmdl_sdk" MI_BASE_DLL_FILE_EXT;
#else
                filename = "libneuray" MI_BASE_DLL_FILE_EXT;
#endif
            m_filename = filename;
#ifdef MI_PLATFORM_WINDOWS
            void* handle = LoadLibraryA((LPSTR)filename);
            if (!handle) {
                m_result_code = RESULT_LOAD_FAILURE;
                if (logger) {
                    DWORD error_code = GetLastError();
                    LPTSTR buffer = 0;
                    LPCTSTR message = TEXT("unknown failure");
                    if (FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER 
                        | FORMAT_MESSAGE_FROM_SYSTEM 
                        | FORMAT_MESSAGE_IGNORE_INSERTS, 0, error_code,
                        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR)&buffer, 0, 0))
                        message = buffer;
                    logger->printf(mi::base::MESSAGE_SEVERITY_FATAL, "MAIN",
                        "Failed to load library (%u): " FMT_LPTSTR,
                        error_code, message);
                    if (buffer)
                        LocalFree(buffer);
                }
                return;
            }
            void* symbol = GetProcAddress((HMODULE)handle, "mi_factory");
            if (!symbol) {
                m_result_code = RESULT_SYMBOL_LOOKUP_FAILURE;
                if (logger) {
                    DWORD error_code = GetLastError();
                    LPTSTR buffer = 0;
                    LPCTSTR message = TEXT("unknown failure");
                    if (FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER 
                        | FORMAT_MESSAGE_FROM_SYSTEM 
                        | FORMAT_MESSAGE_IGNORE_INSERTS, 0, error_code,
                        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR)&buffer, 0, 0))
                        message = buffer;
                    logger->printf(mi::base::MESSAGE_SEVERITY_FATAL, "MAIN",
                        "GetProcAddress error (%u): " FMT_LPTSTR, error_code, message);
                    if (buffer)
                        LocalFree(buffer);
                }
                return;
            }
#else // MI_PLATFORM_WINDOWS
#ifdef MI_PLATFORM_MACOSX
            void* handle = dlopen(filename, RTLD_LAZY);
#else // MI_PLATFORM_MACOSX
            void* handle = dlopen(filename, RTLD_LAZY | RTLD_DEEPBIND);
#endif // MI_PLATFORM_MACOSX
            if (!handle) {
                m_result_code = RESULT_LOAD_FAILURE;
                if (logger)
                    logger->message(mi::base::MESSAGE_SEVERITY_FATAL, "MAIN", dlerror());
                return;
            }
            void* symbol = dlsym(handle, "mi_factory");
            if (!symbol) {
                m_result_code = RESULT_SYMBOL_LOOKUP_FAILURE;
                if (logger)
                    logger->message(mi::base::MESSAGE_SEVERITY_FATAL, "MAIN", dlerror());
                return;
            }
#endif // MI_PLATFORM_WINDOWS
            m_dso_handle = handle;

            m_neuray = mi::neuraylib::mi_factory<mi::neuraylib::INeuray>(symbol);
            if (!m_neuray) {
                mi::base::Handle<mi::neuraylib::IVersion> version(
                    mi::neuraylib::mi_factory<mi::neuraylib::IVersion>(symbol));
                if (!version) {
                    m_result_code = RESULT_INCOMPATIBLE_LIBRARY;
                    if (logger)
                        logger->message(mi::base::MESSAGE_SEVERITY_FATAL, "MAIN",
                            "Incompatible SDK shared library. Could not retrieve INeuray "
                            "nor IVersion interface.");
                }
                else {
                    m_result_code = RESULT_VERSION_MISMATCH;
                    if (logger)
                        logger->printf(mi::base::MESSAGE_SEVERITY_FATAL, "MAIN",
                            "SDK shared library version mismatch: Header version "
                            "%s does not match library version %s.",
                            MI_NEURAYLIB_PRODUCT_VERSION_STRING,
                            version->get_product_version());
                }
            }
        }

        inline Neuray_factory::~Neuray_factory()
        {
            // destruct neuray before unloading the shared library
            m_neuray = 0;

#ifdef MI_PLATFORM_WINDOWS
            int result = FreeLibrary((HMODULE)m_dso_handle);
            if (m_logger && result == 0) {
                LPTSTR buffer = 0;
                LPCTSTR message = TEXT("unknown failure");
                DWORD error_code = GetLastError();
                if (FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                    FORMAT_MESSAGE_IGNORE_INSERTS, 0, error_code,
                    MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR)&buffer, 0, 0))
                    message = buffer;
                m_logger->printf(mi::base::MESSAGE_SEVERITY_FATAL, "MAIN",
                    "Failed to unload library (%u): " FMT_LPTSTR, error_code, message);
                if (buffer)
                    LocalFree(buffer);
            }
#else
            int result = dlclose(m_dso_handle);
            if (m_logger && result != 0)
                m_logger->message(mi::base::MESSAGE_SEVERITY_FATAL, "MAIN", dlerror());
#endif
        }

    } // namespace neuraylib

} // namespace mi

#endif // MI_NEURAY_NEURAY_FACTORY_H
