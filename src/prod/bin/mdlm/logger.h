//*****************************************************************************
// Copyright 2018 NVIDIA Corporation. All rights reserved.
//*****************************************************************************
/// \file logger.h
/// \brief Logger class implementation to re-direct log output.
///
//*****************************************************************************
#pragma once

#include <mi/mdl_sdk.h>

namespace mdlm
{
    /// Custom logger to re-direct log output.
    ///
    class Logger : public mi::base::Interface_implement<mi::base::ILogger>
    {
        int  m_level;           ///< Logging level up to which messages are reported

        /// Returns a string label corresponding to the log level severity.
        static const char* get_log_level(mi::base::Message_severity level);

    public:
        /// Logger where only messages of level lower than the \p level parameter
        /// are written to stderr, i.e., \p level = 0 will disable all logging, and
        /// \p level = 1 logs only fatal messages, \p level = 2 logs errors, 
        /// \p level = 3 logs warnings, \p level = 4 logs info, \p level = 5 logs debug.
        Logger(int level) :
            m_level(level) {}

        /// Callback function logging a message.
        void message(
            mi::base::Message_severity level,
            const char* module_category,
            const char* message);
    };

} // namespace mdlm
