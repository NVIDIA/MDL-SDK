/******************************************************************************
 * Copyright (c) 2004-2024, NVIDIA CORPORATION. All rights reserved.
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

 /// \file
 /// \brief Tests various aspects of hal: colors, system errors

#include "pch.h"

#include <base/lib/log/i_log_logger.h>
#include <base/system/test/i_test_auto_driver.h>
#include <base/system/main/access_module.h>

#include <cstring>
#include <cerrno>

#include "i_hal_ospath.h"
#include "hal.h"

using namespace MI;
using namespace MI::LOG;
using namespace MI::HAL;

MI_TEST_AUTO_FUNCTION( verify_hal_module )
{
    // test error
    FILE* f = fopen("/proc/non_existing==@@!&^%$", "r");
    MI_REQUIRE(!f);
    int err = get_errno();
    MI_REQUIRE_EQUAL(err, ENOENT);
    MI_REQUIRE(!strcmp(MI::HAL::strerror(err).c_str(), "No such file or directory"));

    // test color
    int c;
    for (c=0; c < C_NUM; c++)
    {
        set_console_color(Color(c));
        // Disable useless msg for automatic unit testing
        // mod_log->debug(M_MAIN, Mod_log::C_MAIN, "This is color %d", c);
        set_console_color(C_DEFAULT);
    }
}

MI_TEST_AUTO_FUNCTION( test_ospath_basename )
{
    std::string dir = "C:\\Users\\user\\src";
    MI_REQUIRE_EQUAL(Ospath::basename(dir), "src");
    dir = "/foo/bar/";
    MI_REQUIRE_EQUAL(Ospath::basename(dir), "");
}

MI_TEST_AUTO_FUNCTION( test_ospath_split )
{
    std::string cwd = std::string("C:\\a\\b\\c\\d\\e.f");

    std::string tail, head;
    Ospath::split(cwd, head, tail);
    MI_REQUIRE_EQUAL(head, "C:\\a\\b\\c\\d");
    MI_REQUIRE_EQUAL(tail, "e.f");
    MI_REQUIRE_EQUAL(tail, Ospath::basename(cwd));
    MI_REQUIRE_EQUAL(head, Ospath::dirname(cwd));
    cwd = head;
    Ospath::split(cwd, head, tail);
    MI_REQUIRE_EQUAL(head, "C:\\a\\b\\c");
    MI_REQUIRE_EQUAL(tail, "d");
    MI_REQUIRE_EQUAL(tail, Ospath::basename(cwd));
    MI_REQUIRE_EQUAL(head, Ospath::dirname(cwd));
}

MI_TEST_AUTO_FUNCTION( test_ospath_join )
{
    std::string tail, head;
    head = "", tail = "foo";
    std::string cwd = Ospath::join(head, tail);
    MI_REQUIRE_EQUAL(cwd, std::string("foo"));

    cwd = Ospath::join(std::string("a\\b\\"), std::string("f///y"));
    cwd = Ospath::normpath(cwd);
#ifdef WIN_NT
    MI_REQUIRE_EQUAL(cwd, "a\\b\\f\\y");
#else
    MI_REQUIRE_EQUAL(cwd, "a/b/f/y");
#endif
}

MI_TEST_AUTO_FUNCTION( test_ospath_join_v2 )
{
    MI_REQUIRE_EQUAL("a", Ospath::join_v2("a", "."));
    MI_REQUIRE_EQUAL("b", Ospath::join_v2(".", "b"));
    MI_REQUIRE_EQUAL(".", Ospath::join_v2(".", "."));
#ifdef WIN_NT
    MI_REQUIRE_EQUAL("a\\b", Ospath::join_v2("a", "b"));
#else
    MI_REQUIRE_EQUAL("a/b", Ospath::join_v2("a", "b"));
#endif
}

MI_TEST_AUTO_FUNCTION( test_ospath_normpath )
{
    std::string cwd = Ospath::normpath("\\\\a\\b\\");
#ifdef WIN_NT
    MI_REQUIRE_EQUAL(cwd, "\\\\a\\b\\");
#else
#endif

    cwd = Ospath::normpath("./a/./b");
#ifdef WIN_NT
    MI_REQUIRE_EQUAL(cwd, "a\\b");
#else
    MI_REQUIRE_EQUAL(cwd, "a/b");
#endif

    cwd = Ospath::normpath("./a/../b");
    MI_REQUIRE_EQUAL(cwd, "b");

    cwd = Ospath::normpath("./a/c/../../b");
    MI_REQUIRE_EQUAL(cwd, "b");

    {
    // normpath should handle relative paths with a '/../' in it -- prior to 19780
    // it sometimes did not (result was a hang, not a bad value), such as with this case:
    std::string path =
#ifdef WIN_NT
            "..\\..\\data\\textures";
#else
            "../../data/textures";
#endif
    std::string normed = Ospath::normpath(path);
    MI_REQUIRE_EQUAL(path, normed);
    }

    // test normpath with relative paths (.., . etc.)
    {
    std::string ret;
    std::string p;

    // normal case
    p = "apa/foo/bil.html";
    ret = Ospath::normpath(p);
    MI_REQUIRE_EQUAL(ret, Ospath::normpath_only("apa/foo/bil.html"));

    // one ..
    p = "apa/foo/../knark/bil.html";
    ret = Ospath::normpath(p);
    MI_REQUIRE_EQUAL(ret, Ospath::normpath_only("apa/knark/bil.html"));

    // going outside with .. and a valid file
    p = "../bil.html";
    ret = Ospath::normpath(p);
    MI_REQUIRE_EQUAL(ret, p);

    // resulting in the last file with ..
    p = "apa/../bil.html";
    ret = Ospath::normpath(p);
    MI_REQUIRE_EQUAL(ret, "bil.html");

    // no path, just one file
    p = "lastbil.html";
    ret = Ospath::normpath(p);
    MI_REQUIRE_EQUAL(ret, "lastbil.html");

    // empty path
    p = "";
    ret = Ospath::normpath(p);
    MI_REQUIRE_EQUAL(ret, "");

    // going outside with .. and no valid directories at all
    p = "../../../";
    ret = Ospath::normpath(p);
    MI_REQUIRE_EQUAL(ret, p);

    // relative with multiple .. and resulting in the last file
    p = "abc/def/ghi/../../../jkl";
    ret = Ospath::normpath(p);
    MI_REQUIRE_EQUAL(ret, "jkl");

    // same as above but with an ending /, i.e. ending with a dir
    p = "123/456/789/../../../jkl/";
    ret = Ospath::normpath(p);
    MI_REQUIRE_EQUAL(ret, Ospath::normpath_only("jkl/"));

    // going outside with several .. and several dirs
    p = "123/abc/foobar/../../../../jkl/";
    ret = Ospath::normpath(p);
    MI_REQUIRE_EQUAL(ret, p);

    // mixing .. and dirs
    p = "abc/def/ghi/../mno/../../jkl/";
    ret = Ospath::normpath(p);
    MI_REQUIRE_EQUAL(ret, Ospath::normpath_only("abc/jkl/"));

    // mixing .. and dirs with starting /
    p = "/abc/def/ghi/../mno/../../jkl/";
    ret = Ospath::normpath(p);
    MI_REQUIRE_EQUAL(ret, Ospath::normpath_only("/abc/jkl/"));

    // more mixing
    p = "abc/def/123/../mno/../pqr/../stu/../../jkl/";
    ret = Ospath::normpath(p);
    MI_REQUIRE_EQUAL(ret, Ospath::normpath_only("abc/jkl/"));

    // test handling of "."
    p = ".";
    ret = Ospath::normpath(p);
    // maybe this is debatable..
    MI_REQUIRE_EQUAL(ret, "");

    // test handling of "abc/.."
    p = "abc/..";
    ret = Ospath::normpath(p);
    // maybe this is debatable..
    MI_REQUIRE_EQUAL(ret, "");

    // test that we do ignore the . when we have a relative path
    p = "./";
    ret = Ospath::normpath(p);
    // maybe this is debatable..
    MI_REQUIRE_EQUAL(ret, "");

    // test that we do ignore the . when we have a path
    p = "/./";
    ret = Ospath::normpath(p);
    MI_REQUIRE_EQUAL(ret, Ospath::normpath_only("/"));

    // test that we ignore the . as a path entry also in the middle
    p = "abc/./foobar/./././";
    ret = Ospath::normpath(p);
    MI_REQUIRE_EQUAL(ret, Ospath::normpath_only("abc/foobar/"));

    // .. and another variant
    p = "././abc/../././foobar/./././";
    ret = Ospath::normpath(p);
    MI_REQUIRE_EQUAL(ret, Ospath::normpath_only("foobar/"));

    // We do not try to handle * and ? in any way and leave them untouched
    p = "././?/../././*/./././";
    ret = Ospath::normpath(p);
    MI_REQUIRE_EQUAL(ret, Ospath::normpath_only("*/"));

#ifndef WIN_NT
    // Test does not work on windows since normpath() thinks a beginning
    // '\\' is special and will make sure it is inserted again.
    // lots of /////// should just be ignored
    p = "///////////////////abc//////////////////";
    ret = Ospath::normpath(p);
    MI_REQUIRE_EQUAL(ret, "/abc/");
#endif
    }
}

MI_TEST_AUTO_FUNCTION( test_ospath_normpath_v2 )
{
#ifdef WIN_NT
    // relative paths
    MI_REQUIRE_EQUAL(".",             Ospath::normpath_v2(""));
    MI_REQUIRE_EQUAL("a",             Ospath::normpath_v2("a"));
    MI_REQUIRE_EQUAL("a\\b",          Ospath::normpath_v2("a\\b"));
    MI_REQUIRE_EQUAL("a\\b",          Ospath::normpath_v2("a\\.\\b"));
    MI_REQUIRE_EQUAL("a\\b",          Ospath::normpath_v2("a\\c\\..\\b"));
    MI_REQUIRE_EQUAL(".",             Ospath::normpath_v2("a\\.."));
    MI_REQUIRE_EQUAL("..",            Ospath::normpath_v2("a\\..\\.."));
    MI_REQUIRE_EQUAL("..\\a",         Ospath::normpath_v2("..\\a"));
    MI_REQUIRE_EQUAL("..\\..\\a",     Ospath::normpath_v2("..\\..\\a"));

    // absolute paths
    MI_REQUIRE_EQUAL("\\",            Ospath::normpath_v2("\\"));
    MI_REQUIRE_EQUAL("\\a",           Ospath::normpath_v2("\\a"));
    MI_REQUIRE_EQUAL("\\a\\b",        Ospath::normpath_v2("\\a\\b"));
    MI_REQUIRE_EQUAL("\\a\\b",        Ospath::normpath_v2("\\a\\.\\b"));
    MI_REQUIRE_EQUAL("\\a\\b",        Ospath::normpath_v2("\\a\\c\\..\\b"));
    MI_REQUIRE_EQUAL("\\",            Ospath::normpath_v2("\\a\\.."));
    MI_REQUIRE_EQUAL("\\..",          Ospath::normpath_v2("\\a\\..\\.."));
    MI_REQUIRE_EQUAL("\\..\\a",       Ospath::normpath_v2("\\..\\a"));
    MI_REQUIRE_EQUAL("\\..\\..\\a",   Ospath::normpath_v2("\\..\\..\\a"));

    // separator at the end
    MI_REQUIRE_EQUAL("\\a",           Ospath::normpath_v2("\\a"));
    MI_REQUIRE_EQUAL("a",             Ospath::normpath_v2("a\\"));
    MI_REQUIRE_EQUAL("a\\b",          Ospath::normpath_v2("a\\b\\"));
    MI_REQUIRE_EQUAL("a\\b",          Ospath::normpath_v2("a\\.\\b\\"));
    MI_REQUIRE_EQUAL("a\\b",          Ospath::normpath_v2("a\\c\\..\\b\\"));
    MI_REQUIRE_EQUAL(".",             Ospath::normpath_v2("a\\..\\"));
    MI_REQUIRE_EQUAL("..",            Ospath::normpath_v2("a\\..\\..\\"));
    MI_REQUIRE_EQUAL("..\\a",         Ospath::normpath_v2("..\\a\\"));
    MI_REQUIRE_EQUAL("..\\..\\a",     Ospath::normpath_v2("..\\..\\a\\"));

    // UNC paths
    MI_REQUIRE_EQUAL("\\\\",          Ospath::normpath_v2("\\\\"));
    MI_REQUIRE_EQUAL("\\\\a",         Ospath::normpath_v2("\\\\a"));
    MI_REQUIRE_EQUAL("\\\\a\\b",      Ospath::normpath_v2("\\\\a\\b"));
    MI_REQUIRE_EQUAL("\\\\a\\b",      Ospath::normpath_v2("\\\\a\\.\\b"));
    MI_REQUIRE_EQUAL("\\\\a\\b",      Ospath::normpath_v2("\\\\a\\c\\..\\b"));
    MI_REQUIRE_EQUAL("\\\\",          Ospath::normpath_v2("\\\\a\\.."));
    MI_REQUIRE_EQUAL("\\\\..",        Ospath::normpath_v2("\\\\a\\..\\.."));
    MI_REQUIRE_EQUAL("\\\\..\\a",     Ospath::normpath_v2("\\\\..\\a"));
    MI_REQUIRE_EQUAL("\\\\..\\..\\a", Ospath::normpath_v2("\\\\..\\..\\a"));
#else
    // relative paths
    MI_REQUIRE_EQUAL(".",             Ospath::normpath_v2(""));
    MI_REQUIRE_EQUAL("a",             Ospath::normpath_v2("a"));
    MI_REQUIRE_EQUAL("a/b",           Ospath::normpath_v2("a/b"));
    MI_REQUIRE_EQUAL("a/b",           Ospath::normpath_v2("a/./b"));
    MI_REQUIRE_EQUAL("a/b",           Ospath::normpath_v2("a/c/../b"));
    MI_REQUIRE_EQUAL(".",             Ospath::normpath_v2("a/.."));
    MI_REQUIRE_EQUAL("..",            Ospath::normpath_v2("a/../.."));
    MI_REQUIRE_EQUAL("../a",          Ospath::normpath_v2("../a"));
    MI_REQUIRE_EQUAL("../../a",       Ospath::normpath_v2("../../a"));

    // absolute paths
    MI_REQUIRE_EQUAL("/",             Ospath::normpath_v2("/"));
    MI_REQUIRE_EQUAL("/a",            Ospath::normpath_v2("/a"));
    MI_REQUIRE_EQUAL("/a/b",          Ospath::normpath_v2("/a/b"));
    MI_REQUIRE_EQUAL("/a/b",          Ospath::normpath_v2("/a/./b"));
    MI_REQUIRE_EQUAL("/a/b",          Ospath::normpath_v2("/a/c/../b"));
    MI_REQUIRE_EQUAL("/",             Ospath::normpath_v2("/a/.."));
    MI_REQUIRE_EQUAL("/..",           Ospath::normpath_v2("/a/../.."));
    MI_REQUIRE_EQUAL("/../a",         Ospath::normpath_v2("/../a"));
    MI_REQUIRE_EQUAL("/../../a",      Ospath::normpath_v2("/../../a"));

    // separator at the end
    MI_REQUIRE_EQUAL("a",             Ospath::normpath_v2("a/"));
    MI_REQUIRE_EQUAL("a/b",           Ospath::normpath_v2("a/b/"));
    MI_REQUIRE_EQUAL("a/b",           Ospath::normpath_v2("a/./b/"));
    MI_REQUIRE_EQUAL("a/b",           Ospath::normpath_v2("a/c/../b/"));
    MI_REQUIRE_EQUAL(".",             Ospath::normpath_v2("a/../"));
    MI_REQUIRE_EQUAL("..",            Ospath::normpath_v2("a/../../"));
    MI_REQUIRE_EQUAL("../a",          Ospath::normpath_v2("../a/"));
    MI_REQUIRE_EQUAL("../../a",       Ospath::normpath_v2("../../a/"));
#endif
}

MI_TEST_AUTO_FUNCTION( verify_disk_functions )
{
    std::string q = HAL::Ospath::basename("a/b/c.d");
    MI_REQUIRE(!q.empty());
    MI_REQUIRE(q == "c.d");

    q = HAL::Ospath::basename("a.b");
    MI_REQUIRE(!q.empty());
    MI_REQUIRE(q == "a.b");

    q = HAL::Ospath::dirname("");
    MI_REQUIRE(q.empty());

    q = HAL::Ospath::dirname("/");
    MI_REQUIRE(q == "/");

    q = HAL::Ospath::dirname("\\");
    MI_REQUIRE(q == "\\");

    q = HAL::Ospath::dirname("/a");
    MI_REQUIRE(q == "/");

    q = HAL::Ospath::dirname("\\a");
    MI_REQUIRE(!q.empty());
    MI_REQUIRE(q == "\\");

    q = HAL::Ospath::dirname("/a/");
    MI_REQUIRE(!q.empty());
    MI_REQUIRE(q == "/a");

    q = HAL::Ospath::dirname("\\a\\");
    MI_REQUIRE(!q.empty());
    MI_REQUIRE(q == "\\a");

    q = HAL::Ospath::dirname("a");
    MI_REQUIRE(q.empty());
}

MI_TEST_AUTO_FUNCTION( test_ospath_dirname )
{
    {
    std::string values[] = {"C:\\Users\\user\\test\\foo.mi", "C:\\Users\\user\\test"};
    std::string dirname = Ospath::dirname(values[0]);
    MI_REQUIRE_EQUAL(dirname, values[1]);
    }
    {
    std::string values[] = {"/home/user/test/foo.mi", "/home/user/test"};
    std::string dirname = Ospath::dirname(values[0]);
    MI_REQUIRE_EQUAL(dirname, values[1]);
    }
    {
    std::string values[] = {"C:\\Users\\user\\test\\", "C:\\Users\\user\\test"};
    std::string dirname = Ospath::dirname(values[0]);
    MI_REQUIRE_EQUAL(dirname, values[1]);
    }
    {
    std::string values[] = {"/home/user/test/", "/home/user/test"};
    std::string dirname = Ospath::dirname(values[0]);
    MI_REQUIRE_EQUAL(dirname, values[1]);
    }
    {
    std::string file("");
    std::string dirname = Ospath::dirname(file);
    MI_REQUIRE_EQUAL(dirname, "");
    }
    {
    std::string file(".\\foo\\bar.txt");
    std::string dirname = Ospath::normpath(Ospath::dirname(file));
    MI_REQUIRE_EQUAL(dirname, "foo");
    }
    {
    std::string file("./foo/bar.txt");
    std::string dirname = Ospath::normpath(Ospath::dirname(file));
    MI_REQUIRE_EQUAL(dirname, "foo");
    }
    {
    std::string values[] = { "\\", "\\" };
    std::string dirname = Ospath::dirname(values[0]);
    MI_REQUIRE_EQUAL(dirname, values[1]);
    }
    {
    std::string values[] = { "/", "/" };
    std::string dirname = Ospath::dirname(values[0]);
    MI_REQUIRE_EQUAL(dirname, values[1]);
    }
    {
    std::string values[] = { "/a", "/" };
    std::string dirname = Ospath::dirname(values[0]);
    MI_REQUIRE_EQUAL(dirname, values[1]);
    }
}

MI_TEST_AUTO_FUNCTION( test_ospath_split_only )
{
    // test path splitting
    {
    std::string path = "C:\\foo\\bar/../baz/file.txt";
    std::string root, name;
    Ospath::split_only(path, root, name);
    MI_REQUIRE_EQUAL(root, "C:\\foo\\bar/../baz");
    MI_REQUIRE_EQUAL(name, "file.txt");
    }

    // test with spaces
    {
    std::string path = "C:\\fo o\\bar/../baz/file.txt";
    std::string root, name;
    Ospath::split_only(path, root, name);
    MI_REQUIRE_EQUAL(root, "C:\\fo o\\bar/../baz");
    MI_REQUIRE_EQUAL(name, "file.txt");
    }
}

MI_TEST_AUTO_FUNCTION( test_ospath_splitext )
{
    std::string root, ext;
    Ospath::splitext("a/b.ext", root, ext);
    MI_REQUIRE_EQUAL(ext, ".ext");
}

MI_TEST_AUTO_FUNCTION( test_ospath_splitdrive )
{
    std::string drive, tail;
#ifndef WIN_NT
    //Ospath::splitdrive("/a/b.ext", drive, tail);
    //MI_REQUIRE_EQUAL(drive, "/a");
#else
    Ospath::splitdrive("C:\\Users\\user", drive, tail);
    MI_REQUIRE_EQUAL(drive, "C:");
    MI_REQUIRE_EQUAL(tail, "\\Users\\user");

    Ospath::splitdrive("C:\\\\\\", drive, tail);
    MI_REQUIRE_EQUAL(drive, "C:");
    MI_REQUIRE_EQUAL(tail, "\\\\\\");
#endif
}

MI_TEST_AUTO_FUNCTION( test_ospath_get_ext )
{
    MI_REQUIRE_EQUAL(Ospath::get_ext("a/b.ext"), ".ext");
}
