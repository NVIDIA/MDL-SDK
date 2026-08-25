/***************************************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
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

#ifndef BASE_DATA_DBLIGHT_DBLIGHT_ADMIN_SERVER_H
#define BASE_DATA_DBLIGHT_DBLIGHT_ADMIN_SERVER_H

#include "dblight_util.h"

#ifdef DBLIGHT_ENABLE_ADMIN_SERVER

#include <map>
#include <string>
#include <thread>

#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>

namespace MI {

namespace DBLIGHT {

class Database_impl;

/// An HTTP server that allows to inspect the state of the database.
///
/// The server listens on port 12345 on localhost.
class Admin_server
{
public:
    /// Constructor.
    ///
    /// \param database   Instance of the database this manager belongs to.
    /// \param port       Port to listen on.
    Admin_server( Database_impl* database, mi::Uint16 port);

    /// Destructor.
    ~Admin_server();

    /// Indicates whether the server is running.
    bool is_running() { return m_thread.joinable(); }

    /// Stops the server if it is running.
    void stop();

private:
    /// Type to hold a parsed URL query.
    using Params = std::map<std::string, std::string>;

    void do_accept();
    void handle_session( boost::asio::ip::tcp::socket socket);

    /// Generates the index page (just the link menu at the top).
    static std::string page_index();

    /// Generates various parameter-less pages reachable by the link menu at the top.
    std::string page_database();
    std::string page_scopes();
    std::string page_transactions();
    std::string page_tags();
    std::string page_names();
    std::string page_garbage_collection();
    std::string page_statistics();

    /// Generates the page for a tag given by the parameter with key "id".
    std::string page_tag( const Params& params);

    /// Generates the page for a name given by the parameter with key "id".
    std::string page_name( const Params& params);

    /// Generates a HTML page with the given title and content.
    static std::string get_page( const std::string& title, const std::string& content);

    /// Encodes the given string for HTML.
    static std::string html_encode( const std::string& s);

    /// Encodes the given string for an URL (percent encoding).
    static std::string url_encode( const std::string& s);

    /// Decodes the given string for an URL (percent encoding).
    static std::string url_decode( const std::string& s);

    /// Parses an URL query.
    static Params parse_query( std::string query);

    /// Returns a HTML for a bad request with an optional reason.
    static std::string bad_request( const std::string& reason = "");

    /// Content of the link menu at the top.
    static const std::pair<std::string, std::string> s_links[];

    /// Instance of the database this admin server belongs to.
    Database_impl* const m_database;

    /// Callbacks and static information needed by methods generating the HTML pages.
    const Html_context m_html_context;

    boost::asio::io_context        m_ioc;
    boost::asio::ip::tcp::acceptor m_acceptor;
    std::thread                    m_thread;
};

} // namespace DBLIGHT

} // namespace MI

#endif // DBLIGHT_ENABLE_ADMIN_SERVER

#endif // BASE_DATA_DBLIGHT_DBLIGHT_ADMIN_SERVER_H
