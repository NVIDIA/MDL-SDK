/***************************************************************************************************
 * Copyright (c) 2006-2018, NVIDIA CORPORATION. All rights reserved.
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
/// \brief The implementation of useful path' utilities.

#include "pch.h"

#include "i_hal_ospath.h"

#include <base/lib/log/i_log_assert.h>
#include <base/util/string_utils/i_string_utils.h>
#include <mi/base/config.h>

#include <sstream>
#include <algorithm>

#ifndef WIN_NT
 #include <unistd.h>
#endif


namespace MI {
namespace HAL {

namespace {

// Replace all occurrence of a string with another one
MISTD::string replace_all(
    const MISTD::string& str,		// input string
    const MISTD::string& fstr,		// find string
    const MISTD::string& rstr)		// replace string
{
    if (str.empty())
	return MISTD::string("");

    MISTD::string fs(str);
    size_t pos = 0;
    while(pos != MISTD::string::npos) {
	pos = fs.find(fstr, pos);
	if (pos != MISTD::string::npos) {
	    fs.replace(pos, fstr.size(), rstr);
	}
    }

    return fs;
}

}


// Return the OS independent separator. On Windows the separator is '\',
// while on the other operating system is is '/'.
MISTD::string Ospath::sep()
{
#ifdef WIN_NT
	return MISTD::string("\\");
#else
	return MISTD::string("/");
#endif
}

// Return the OS independent path set separator.
// on Linux and MacOSX, path set is separated by ":", e.g.,
// "/home/user0:/home/user0", on Windows,
// "C:\home\user0;C:\home\user0"
MISTD::string Ospath::get_path_set_separator()
{
#ifdef _WIN32
    return ";";
#else
    return ":";
#endif
}

// Return the base name of pathname path. This is the second half of the
// pair returned by split(path). Note that the result of this function is
// different from the Unix basename program; where basename for '/foo/bar/'
// returns 'bar', the basename() function returns an empty string ('').
MISTD::string Ospath::basename(
    const MISTD::string& path)			// Incoming path
{
    MISTD::string head, tail;
    split(path, head, tail);
    return tail;
}

// Return the directory name of pathname path. This is the first half of
// the pair returned by split(path).
MISTD::string Ospath::dirname(
    const MISTD::string& path)			// Incoming path
{
    MISTD::string head, tail;
    split(path, head, tail);
    return head;
}

MISTD::string Ospath::join(
    const MISTD::string& path,
    const MISTD::string& path2)
{
    if (path.empty())
	return path2;
    else if (path2.empty())
	return path;
    else
	return path + sep() + path2;
}

MISTD::string Ospath::join_v2(const MISTD::string& path1, const MISTD::string& path2)
{
    if (path1 == ".")
        return path2;
    if (path2 == ".")
        return path1;
    return path1 + sep() + path2;
}

MISTD::string Ospath::normpath(
    const MISTD::string& orig_in_path)
{
    MISTD::vector<MISTD::string> token_list;
    const MISTD::string separator = HAL::Ospath::sep();
    const MISTD::string in_path = Ospath::normpath_only(orig_in_path);
    MISTD::string out_path;

    MI::STRING::split(in_path, separator, token_list);

    MISTD::vector<MISTD::string> last_dir;
    MISTD::vector<MISTD::string>::const_iterator it = token_list.begin();

    // iterate over all elements, skipping . and resolving ..
    while(it != token_list.end()) {
	if (it->empty()) { // We do not care about empty tokens (i.e. //)
	    ++it;
	    continue;
	}
	// valid dir names are appended and saved on a stack (last_dir)
	if (*it != ".." && *it != ".") {
	    last_dir.push_back(*it);
	    if (!out_path.empty() && out_path[out_path.length() - 1] != separator[0])
		out_path.append(separator);
	    out_path.append(*it);
	}
	// ".." means we have to pop the last valid dir name from the stack if we can
	else if (*it == "..") {
	    if (last_dir.empty())
		return orig_in_path;
	    size_t ind = out_path.rfind(last_dir.back());
	    if (ind == MISTD::string::npos)
		return orig_in_path;
	    out_path = out_path.substr(0, ind);
	    last_dir.pop_back();
	}
	++it;
    }

    // special case: path ended with /
    if (!out_path.empty() && !in_path.empty() && in_path[in_path.length() - 1] == separator[0])
	out_path.append(separator);
    // special case: path was absolute, i.e. started with /
    if (!in_path.empty() && in_path[0] == separator[0])
      out_path.insert(0, separator);

#if WIN_NT
    // Check for UNC and put back the '\\' which have been removed.
    // First convert back to backslashes since siteconfig converts
    // paths to forward slashes which means the test against sep()
    // which returns backslashes on windows would not work.
    MISTD::string win_path = convert_to_backward_slashes(orig_in_path);
    if (win_path.length() >= 2 &&
	(win_path[0] == sep()[0]) && (win_path[1] == sep()[0]))
	out_path.insert(0, sep());
    return convert_to_backward_slashes(out_path);
#else
    return out_path;
#endif
}

MISTD::string Ospath::normpath_only(
    const MISTD::string& path)
{
    MISTD::string npath = convert_to_forward_slashes(path);
    npath = replace_all(npath, MISTD::string("//"), MISTD::string("/"));

#if WIN_NT
    // Check for UNC and put back the '\\' which have been removed.
    // First convert back to backslashes since siteconfig converts
    // paths to forward slashes which means the test against sep()
    // which returns backslashes on windows would not work.
    MISTD::string win_path = convert_to_backward_slashes(path);
    if (win_path.length() >= 2 &&
	(win_path[0] == sep()[0]) && (win_path[1] == sep()[0]))
	npath.insert(0, sep());
    return convert_to_backward_slashes(npath);
#else
    return npath;
#endif
}

MISTD::string Ospath::normpath_v2(const MISTD::string& path)
{
    const MISTD::string& separator = sep();
    ASSERT(M_HAL, separator.size() == 1);

    MISTD::vector<MISTD::string> path_components;
    MI::STRING::split(path, separator, path_components);

    // resolved current and parent directory references
    MISTD::vector<MISTD::string> result_components;
    for (size_t i = 0; i < path_components.size(); ++i) {
        const MISTD::string& component = path_components[i];

        // skip empty path components or current directory references
        if (component.empty() || component == ".")
            continue;

        // handle parent directory references
        if (component == "..") {
            if (result_components.empty())
                result_components.push_back("..");
            else
                result_components.pop_back();
            continue;
        }

        // handle regular path components
        result_components.push_back(component);
    }

    // convert result_components into a string
    MISTD::string result = result_components.empty() ? "" : result_components[0];
    for (size_t i = 1; i < result_components.size(); ++i)
        result += separator + result_components[i];

    // re-add leading separator for absolute paths
    if (!path.empty() && path[0] == separator[0])
        result.insert(0, separator);

#if WIN_NT
    // re-add another leading separator for UNC paths on Windows
    if (path.length() >= 2 && path[0] == separator[0] && path[1] == separator[0])
	result.insert(0, separator);
#endif

    return result.empty() ? "." : result;
}

// Split the pathname path into a pair, (head, tail) where tail is the last
// pathname component and head is everything leading up to that. The tail
// part will never contain a slash; if path ends in a slash, tail will be
// empty. If there is no slash in path, head will be empty. If path is
// empty, both head and tail are empty. Trailing slashes are stripped from
// head unless it is the root (one or more slashes only). In nearly all
// cases, join(head, tail) equals path (the only exception being when there
// were multiple slashes separating head from tail).
void Ospath::split(
    const MISTD::string& path,			// Incoming path
    MISTD::string& head_ref,			// Out head part
    MISTD::string& tail_ref)			// Out tail part
{
    MISTD::string head, tail;

    if (path.empty()) {
	head_ref = head;
	tail_ref = tail;
	return;
    }

    MISTD::string filepath;
    filepath = convert_to_forward_slashes(path);

    MISTD::string::size_type i = filepath.find_last_of('/');
    if (i == MISTD::string::npos) {
	head_ref = MISTD::string("");
	tail_ref = path;
	return;
    }

    // Remove from one behind '/' to the end
    head = path;
    head.erase(i+1, MISTD::string::npos);
    // remove trailing '/' iff size()>1 and if this is not a Windows drive, eg C:/
    if (head.size() > 1
#ifdef WIN_NT
	// watch out for a drive
	&& !(head.size() > 2 && head[head.size()-2] == ':')
#endif
	)
	head.erase(i);

    // Remove from the beginning of the string to the '/' included
    tail = path;
    tail.erase(0, i+1);

    head_ref = head;
    tail_ref = tail;
}

// Split the pathname path into a pair, (head, tail) where tail is the
// last pathname component and head is everything leading up to that.
// This version leaves both parts untouched and does only the splitting.
void Ospath::split_only(
    const MISTD::string& path,		// Incoming path
    MISTD::string& head,			// Out head part
    MISTD::string& tail)			// Out tail part
{
    // set pos to the 1st delimiter
    MISTD::string::size_type pos_win = path.rfind('\\');
    MISTD::string::size_type pos_lin = path.rfind('/');
    MISTD::string::size_type pos = MISTD::string::npos;
    if (pos_win == MISTD::string::npos && pos_lin != MISTD::string::npos)
	pos = pos_lin;
    else if (pos_win != MISTD::string::npos && pos_lin == MISTD::string::npos)
	pos = pos_win;
    else
	pos = MISTD::max(pos_win, pos_lin);

    head = path.substr(0, pos);				// this spares the trailing delimiter
    tail = path.substr(pos == MISTD::string::npos? 0 : pos+1);
}

// Split the pathname path into a pair (drive, tail) where drive is either
// a drive specification or the empty string. On systems which do not use
// drive specifications, drive will always be the empty string. In all
// cases, drive + tail will be the same as path.
void Ospath::splitdrive(
    const MISTD::string& path,		// Incoming path
    MISTD::string& drive,			// Out drive part
    MISTD::string& tail)			// Out tail part
{
    drive = MISTD::string("");
    tail = MISTD::string("");
    if (path.empty())
	return;

    MISTD::string::size_type i = path.find_last_of(':');
    if (i == MISTD::string::npos)
    {
	tail = path;
	return;
    }

    drive = path.substr(0,i+1);
    tail = path.substr(i+1);
}

// Split the pathname path into a pair (root, ext) such that
// root + ext == path, and ext is empty or begins with a period and
// contains at most one period.
void Ospath::splitext(
    const MISTD::string& path,		// Incoming path
    MISTD::string& root,			// Out root part
    MISTD::string& ext)			// Out extension part
{
    root = MISTD::string("");
    ext = MISTD::string("");
    if (path.empty())
	return;

    MISTD::string::size_type i = path.find_last_of('.');
    if (i == MISTD::string::npos)
	return;

    root = path;
    root.erase(i, MISTD::string::npos);
    ext = path;
    ext.erase(0, i);
}


MISTD::string Ospath::get_ext(
    const MISTD::string& path)
{
    MISTD::string::size_type i = path.find_last_of('.');
    if (i == MISTD::string::npos)
	return MISTD::string();
    return path.substr(i);
}


// Convert the given path to use forward slashes
MISTD::string Ospath::convert_to_forward_slashes(
    const MISTD::string& path)		// Convert this path
{
    return replace_all(path, MISTD::string("\\"), MISTD::string("/"));
}


// Convert the given path to use forward slashes
MISTD::string Ospath::convert_to_backward_slashes(
    const MISTD::string& path)		// Convert this path
{
    return replace_all(path, MISTD::string("/"), MISTD::string("\\"));
}


// Convert all separators to the current platform's separators.
MISTD::string Ospath::convert_to_platform_specific_path(
    const MISTD::string& path)
{
    return
#ifdef MI_PLATFORM_WINDOWS
	convert_to_backward_slashes(path)
#else
	convert_to_forward_slashes(path)
#endif
	;
}

}
}
