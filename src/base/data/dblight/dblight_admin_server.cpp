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

#include "pch.h"

#include "dblight_admin_server.h"

#ifdef DBLIGHT_ENABLE_ADMIN_SERVER

#include <limits>

#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>

#include <base/lib/log/i_log_logger.h>

#include "dblight_database.h"
#include "dblight_info.h"
#include "dblight_transaction.h"
#include "dblight_scope.h"

namespace beast = boost::beast;
namespace http  = beast::http;
namespace asio  = boost::asio;
using tcp       = asio::ip::tcp;

namespace MI {

namespace DBLIGHT {

const std::pair<std::string, std::string> Admin_server::s_links[] = {
    { "Database",           "/database.html"           },
    { "Scopes",             "/scopes.html"             },
    { "Transactions",       "/transactions.html"       },
    { "Tags",               "/tags.html"               },
    { "Names",              "/names.html"              },
    { "Garbage collection", "/garbage_collection.html" },
    { "Statistics",         "/statistics.html"         }
};

Admin_server::Admin_server( Database_impl* database, mi::Uint16 port)
  : m_database( database),
    m_html_context{
        Admin_server::html_encode,
        Admin_server::url_encode,
        "/tag.html?id=",
        "/name.html?id="},
    m_acceptor( m_ioc)
{
    std::string address = "0.0.0.0";
    tcp::endpoint endpoint{ asio::ip::make_address( address), port};

    beast::error_code ec;
    m_acceptor.open( endpoint.protocol(), ec);
    if( !ec)
        m_acceptor.set_option( tcp::acceptor::reuse_address( true), ec);
    if( !ec)
        m_acceptor.bind( endpoint, ec);
    if( !ec)
        m_acceptor.listen( asio::socket_base::max_listen_connections, ec);

    if( ec) {
        LOG::mod_log->error( M_DB, LOG::Mod_log::C_DATABASE,
            "Admin server failed to listen on %s:%hu: %s",
            address.c_str(), port, ec.message().c_str());
        return;
    }

    do_accept();
    m_thread = std::thread( [this]{ m_ioc.run(); });
}

Admin_server::~Admin_server()
{
    stop();
}

void Admin_server::stop()
{
    if( !m_thread.joinable())
        return;

    // handle_session() is synchronous, so m_ioc.run() only returns once
    // the in-flight session (if any) has finished — no sessions are cut short.
    m_ioc.stop();
    m_thread.join();
}

void Admin_server::do_accept()
{
    m_acceptor.async_accept( [this]( beast::error_code ec, tcp::socket socket) {
        if( !ec)
            handle_session( std::move( socket));
        else
            LOG::mod_log->error( M_DB, LOG::Mod_log::C_DATABASE,
                "Admin server accept error: %s", ec.message().c_str());
        if( m_acceptor.is_open())
            do_accept();
    });
}

void Admin_server::handle_session( tcp::socket socket)
{
    try
    {
        beast::flat_buffer buffer{ 8192 };
        http::request<http::string_body> request;

        http::read( socket, buffer, request);

        std::string target( request.target().data(), request.target().size());
        auto pos = target.find( '?');
        Params params;
        if( pos != std::string::npos) {
            params = parse_query( target.substr( pos + 1));
            target.resize( pos);
        }

        std::string body;
        http::status status = http::status::ok;

        if( target == "/scopes.html") {
            body = page_scopes();
        } else if( target == "/database.html") {
            body = page_database();
        } else if( target == "/transactions.html") {
            body = page_transactions();
        } else if( target == "/tags.html") {
            body = page_tags();
        } else if( target == "/names.html") {
            body = page_names();
        } else if( target == "/garbage_collection.html") {
            body = page_garbage_collection();
        } else if( target == "/statistics.html") {
            body = page_statistics();
        } else if( target == "/tag.html") {
            body = page_tag( params);
        } else if( target == "/name.html") {
            body = page_name( params);
        } else {
            body = page_index();
        }

        http::response<http::string_body> response( status, request.version());
        response.set( http::field::content_type, "text/html");
        response.body() = body;
        response.prepare_payload();
        http::write( socket, response);
        beast::error_code ec;
        socket.shutdown( tcp::socket::shutdown_send, ec);
    }
    catch( const beast::system_error& e)
    {
        LOG::mod_log->error( M_DB, LOG::Mod_log::C_DATABASE,
            "Session error: %s", e.what());
    }
}

std::string Admin_server::page_index()
{
    return get_page( "", "");
}

std::string Admin_server::page_database()
{
    std::ostringstream s;
    {
        THREAD::Block block( &m_database->get_lock());
        m_database->dump_html( s, m_html_context);
    }
    return get_page( "Database", s.str());
}

std::string Admin_server::page_scopes()
{
    std::ostringstream s;
    {
        THREAD::Block block( &m_database->get_lock());
        m_database->get_scope_manager()->dump_html( s, m_html_context);
    }
    return get_page( "Scopes", s.str());
}

std::string Admin_server::page_transactions()
{
    std::ostringstream s;
    {
        THREAD::Block block( &m_database->get_lock());
        m_database->get_transaction_manager()->dump_html( s, m_html_context);
    }
    return get_page( "Transactions", s.str());
}

std::string Admin_server::page_tags()
{
    std::ostringstream s;
    {
        THREAD::Block block( &m_database->get_lock());
        m_database->get_info_manager()->dump_html_tags( s, m_html_context);
    }
    return get_page( "Tags", s.str());
}

std::string Admin_server::page_names()
{
    std::ostringstream s;
    {
        THREAD::Block block( &m_database->get_lock());
        m_database->get_info_manager()->dump_html_names( s, m_html_context);
    }
    return get_page( "Names", s.str());
}

std::string Admin_server::page_garbage_collection()
{
    std::ostringstream s;
    {
        THREAD::Block block( &m_database->get_lock());
        m_database->get_info_manager()->dump_html_garbage_collection( s, m_html_context);
   }

    return get_page( "Garbage collection", s.str());
}

std::string Admin_server::page_statistics()
{
    std::ostringstream s;
    {
        THREAD::Block block( &m_database->get_lock());
        m_database->get_info_manager()->dump_html_statistics( s, m_html_context);
    }
    return get_page( "Statistics", s.str());
}

std::string Admin_server::page_tag( const Params& params)
{
    auto it = params.find( "id");
    if( it == params.end())
        return bad_request( "missing id");

    const std::string& value_str = it->second;
    unsigned long value;
    try {
        value = std::stoul( value_str);
    } catch( std::exception& ) {
        return bad_request( "invalid id " + value_str);
    }
    if( value == 0 || value > std::numeric_limits<mi::Uint32>::max())
        return bad_request( "invalid id " + value_str);
    DB::Tag tag( value);

    std::string title = "Tag " + std::to_string( tag());
    std::ostringstream s;
    {
        THREAD::Block block( &m_database->get_lock());
        m_database->get_info_manager()->dump_html_tag( s, m_html_context, tag);
    }
    return get_page( title, s.str());
}

std::string Admin_server::page_name( const Params& params)
{
    auto it = params.find( "id");
    if( it == params.end())
        return bad_request( "missing id");

    const std::string& name = it->second;
    if( name.empty())
       return bad_request( "invalid id " + name);

    std::string title = "Name " + name;
    std::ostringstream s;
    {
        THREAD::Block block( &m_database->get_lock());
        m_database->get_info_manager()->dump_html_name( s, m_html_context, name);
    }
    return get_page( title, s.str());
}

std::string Admin_server::get_page( const std::string& title, const std::string& content)
{
    std::string result = "<!DOCTYPE html>\n";
    result += "<html>\n";
    result += "<head>\n";
    result += "<title>" + html_encode( title) + "</title>\n";
    result += "</head>\n";
    result += "<body>\n";

    bool is_first = true;
    for( const auto& link: s_links) {
        if( !is_first)
            result += " &mdash; ";
        bool do_link = link.first != title;
        if( do_link)
            result += "<a href=\"" + link.second + "\">";
        result += html_encode( link.first);
        if( do_link)
            result += "</a>";
        result += "\n";
        is_first = false;
    }
    result += "<p></p>\n";

    if( !title.empty())
        result += "<h1>" + html_encode( title) + "</h1>\n";
    result += content;
    result += "</body>\n</html>\n";

    return result;
}

std::string Admin_server::html_encode( const std::string& s)
{
    std::string result;
    result.reserve( s.size());
    for( char c: s) {
        switch( c) {
            case '&':  result += "&amp;";  break;
            case '<':  result += "&lt;";   break;
            case '>':  result += "&gt;";   break;
            case '"':  result += "&quot;"; break;
            case '\'': result += "&#39;";  break;
            default:   result += c;        break;
        }
    }
    return result;
}

std::string Admin_server::url_encode( const std::string& s)
{
    static const char hex[] = "0123456789ABCDEF";

    std::string result;
    result.reserve( s.size());
    for( unsigned char c: s) {
        if(    (c >= 'A' && c <= 'Z')
            || (c >= 'a' && c <= 'z')
            || (c >= '0' && c <= '9')
            || c == '-'
            || c == '_'
            || c == '.'
            || c == '~') {
            result += static_cast<char>( c);
        } else {
            result += '%';
            result += hex[c >> 4];
            result += hex[c & 0xF];
        }
    }
    return result;
}

std::string Admin_server::url_decode( const std::string& s)
{
    std::string result;
    result.reserve( s.size());

    auto hex_to_int = []( char c) -> int {
        if( c >= '0' && c <= '9') return c - '0';
        if( c >= 'A' && c <= 'F') return c - 'A' + 10;
        if( c >= 'a' && c <= 'f') return c - 'a' + 10;
        return -1;
    };

    for( size_t i = 0; i < s.size(); ++i) {
        if( s[i] == '+') {
            result += ' ';
            continue;
        }
        if( s[i] == '%' && i + 2 < s.size()) {
            int hi = hex_to_int( s[i + 1]);
            int lo = hex_to_int( s[i + 2]);
            if( hi >= 0 && lo >= 0) {
                result += static_cast<char>( (hi << 4) | lo);
                i += 2;
                continue;
            }
        }
        result += s[i];
    }

    return result;
}

Admin_server::Params Admin_server::parse_query( std::string query)
{
    Params result;

    while( !query.empty()) {

        auto amp         = query.find( '&');
        std::string pair = query.substr( 0, amp);
        auto eq          = pair.find( '=');

        if( eq != std::string::npos) {
            std::string key   = url_decode( pair.substr( 0, eq));
            std::string value = url_decode( pair.substr( eq + 1));
            result[key] = value;
        } else if( !pair.empty()) {
            std::string key   = url_decode( pair);
            result[key] = "";
        }

        if( amp == std::string::npos)
            break;
        query = query.substr( amp + 1);
    }

    return result;
}

std::string Admin_server::bad_request( const std::string& reason)
{
    std::string result = "<html><body><h1>400 Bad Request";
    if( !reason.empty())
        result += ": " + html_encode( reason);
    result += "</h1></body></html>";
    return result;
}

} // namespace DBLIGHT

} // namespace MI

#endif // DBLIGHT_ENABLE_ADMIN_SERVER
