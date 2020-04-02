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
 ******************************************************************************/
#pragma once

#include <mi/mdl_sdk.h>
#include "options.h"

 /// Forward declarations.
namespace mi 
{
    namespace neuraylib 
    { 
        class Neuray_factory;
    }
}

namespace mdlm
{
    class Command;

    /// Wrapper for the MDL SDK based application.
    ///
    class Application
    {
    public:
        /// Options class
        class Options
        {
        public:
            std::vector<std::string> m_paths;   ///< MDL search paths
            std::vector<std::string> m_command; ///< Command and arguments
            int m_verbosity;                    ///< log level: 0 = off, 3 = show errs and warns
            bool m_nostdpath;
            bool m_quiet; /// Quiet mode, see Application::report()

        public:
            /// Create options with default settings.
            Options()
                : m_verbosity(3), m_nostdpath(false), m_quiet(false)
            {}
        };
    private:
        Options m_options;
        Command * m_command;
        mi::base::Handle<mi::base::ILogger> m_logger;
        std::string m_name;
        mi::neuraylib::Neuray_factory * m_factory;
        bool m_freeimage_loaded;

        Application(const Application&);//prevent copy ctor
        // Setup the application options from the command line arguments
        mi::Sint32 setup_options(int argc, char *argv[]);

    public:
        /// Singleton
        static Application & theApp();

    public:
        /// Member initialization
        Application();

        /// Shutdown neuray
        ~Application();

        /// Parse command line, setup neuray, start neuray
        ///
        /// \param  argc    The argument count.
        /// \param  argv    The argument values.
        /// \return
        ///		-  0: Success
        ///		-  -?: See INeuray->start()
        mi::Sint32 initialize(int argc, char *argv[]);

        /// Shutdown neuray
        void shutdown();

        /// Access neuray 
        mi::neuraylib::INeuray * neuray();

        /// Access logger 
        const mi::base::Handle<mi::base::ILogger> & logger();

        /// Return the command which the program should invoke
        Command * get_command();

        /// Name of the application without extension and without directory
        const std::string & name() { return m_name; }

        /// Utility routines for logging messages
        /// This can be disbaled using option quiet mode
        void report(const std::string & msg) const;

        // Check if freeimage is available
        bool freeimage_available() const { return m_freeimage_loaded; }
    };

} // namespace mdlm
