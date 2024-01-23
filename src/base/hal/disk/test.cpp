/***************************************************************************************************
 * Copyright (c) 2003-2024, NVIDIA CORPORATION. All rights reserved.
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
/// \brief The unit tests of DISK

#include "pch.h"

#include <mi/base/config.h>
#include <base/lib/log/i_log_logger.h>
#include <base/hal/hal/hal.h>
#include <base/hal/disk/disk.h>
#include <base/system/test/i_test_auto_driver.h>
#include <base/system/main/access_module.h>
#include <base/system/main/access_module.h>

#include <cstdio>
#include <cstring>
#include <string>

using namespace MI;
using namespace MI::LOG;
using namespace MI::DISK;
using namespace MI::HAL;
using namespace MI::SYSTEM;


// The sole purpose of this seems to be the creation of the file "Testfile". So, it probably should
// be put into a setup() function of a test suite?
MI_TEST_AUTO_FUNCTION( setup_hal )
{
    std::fclose(std::fopen("Testfile", "w"));
}


MI_TEST_AUTO_FUNCTION( verify_access )
{
    std::string src_path = TEST::mi_src_path("");
    std::string file = TEST::mi_src_path("base/hal/disk/test.cpp");

    // Require a readable file.
    MI_REQUIRE(DISK::access(file.c_str(), false));
    // We do not really care if it is writable so just log it.
    DISK::access(file.c_str(), true);

    // Require our current directory to be readable and writable.
    MI_REQUIRE(DISK::access(".", false));
    MI_REQUIRE(DISK::access(".", true));

    // We created this file so it must still exist and be readable.
    MI_REQUIRE(DISK::access("Testfile"));
}


MI_TEST_AUTO_FUNCTION( verify_copy )
{
    // Copying our test file must succeed.
    MI_REQUIRE(DISK::file_copy("Testfile", "Testcopy"));
    // ..and we must be able to access the copy
    MI_REQUIRE(DISK::access("Testcopy"));
}


MI_TEST_AUTO_FUNCTION( verify_rename )
{
    // to avoid failures with an existing "Testrename"
    DISK::file_remove("Testrename");

    // Renaming our testfile must succeed.
    MI_REQUIRE(DISK::rename("Testcopy", "Testrename"));
    // ..and we must be able to access it after renaming
    MI_REQUIRE(DISK::access("Testrename"));
}


MI_TEST_AUTO_FUNCTION( verify_directory )
{
    std::string src_path = TEST::mi_src_path("base/hal/disk");
    std::string file = TEST::mi_src_path("base/hal/disk/test.cpp");

    Directory dir, dir2;

    // Open MI_SRC dir
    MI_REQUIRE(dir.open(src_path.c_str()));

    int i, n[2], ok[2];
    for (i=0; i < 2; i++) {
        // rewind() must be ok.
        MI_REQUIRE(dir.rewind());
        // eof() should not happen here, yet.
        MI_REQUIRE(!dir.eof());

        // The argument to dir.read() is whether to return files
        // that start with . or not.
        n[i]=ok[i]=0;
        std::string name = dir.read(!i);
        while (!name.empty()) {
            ok[i] += (name == "test.cpp");
            n[i]++;
            name = dir.read(!i);
        }

        // No errors should have occured.
        MI_REQUIRE(!dir.error());

        // Should not be zero at this point, we must have found
        // at least one file.
        MI_REQUIRE(n[i]);

        // We should have found exactly one test file
        MI_REQUIRE(ok[i] == 1);

        // And now we should have an EOF condition
        MI_REQUIRE(dir.eof());
    }

    // There should be atleast two less entries between the two runs in
    // unix and windows with and without filtering out files that start
    // with dots.
    MI_REQUIRE(!(n[0] > n[1]-2));

    // Closing must succeed.
    MI_REQUIRE(dir.close());
}

MI_TEST_AUTO_FUNCTION( verify_file_raw )
{
    std::string src_path = TEST::mi_src_path("");
    std::string file = TEST::mi_src_path("base/hal/disk/test.cpp");

    File rfile, wfile;
    char line[1024];
    mi::Sint64 i, j;

    // We must be able to open this file.
    MI_REQUIRE(rfile.open(file.c_str()));

    // Opening a file for writing in CWD should be possible.
    MI_REQUIRE(wfile.open("Testfile-r", File::M_WRITE));

    // The initial state of the opened files must be correct, no eof flag.
    MI_REQUIRE(rfile.eof() ==false && wfile.eof() == false);

    for (;;) {
        // Reading should not fail from this file.
        MI_REQUIRE((i = rfile.read(line, sizeof(line))) >= 0);
        // And there should not be any errors.
        MI_REQUIRE(rfile.error() == 0);

        // if read returned eof, break out.
        if (!i)
            break;

        // It should not be possible to read more than specified.
        MI_REQUIRE(i <= (int)sizeof(line));

        // Writing should not fail.
        MI_REQUIRE((j = wfile.write(line, i)) >= 0);
        // All should have been written.
        MI_REQUIRE(i == j);
        // No Write errors should have occured.
        MI_REQUIRE(wfile.error() == 0);
    }

    // Now rfile should be flagged with eof
    MI_REQUIRE(rfile.eof());

    i = rfile.filesize();
    j = wfile.filesize();
    // Sanity checks
    MI_REQUIRE(i >= 10 && i <= 100000);
    MI_REQUIRE(i == j);

    // Closing files must succeed
    MI_REQUIRE(rfile.close());
    MI_REQUIRE(wfile.close());

    // Remove test file should work.
    MI_REQUIRE(DISK::file_remove("Testfile-r"));
}


MI_TEST_AUTO_FUNCTION( verify_file_copy )
{
    std::string src_path = TEST::mi_src_path("");
    std::string file = TEST::mi_src_path("base/hal/disk/test.cpp");

    File rfile, wfile;
    char line[1024];
    mi::Sint64 i, j;

    // Open our test file should work
    MI_REQUIRE(rfile.open(file.c_str()));

    // Creating a test file in CWD must work
    MI_REQUIRE(wfile.open("Testfile-l", File::M_WRITE));

    // In the beginning, none of the files can have the EOF flag set
    MI_REQUIRE(rfile.eof() == false && wfile.eof() == false);

    for (;;) {
        // Read a line from the test file
        MI_REQUIRE(rfile.readline(line, sizeof(line)));
        // ..should not have caused any errors.
        MI_REQUIRE(rfile.error() == 0);

        // If that was the last line, break.
        if (rfile.eof())
            break;

        // Write the read line to the temp file.
        MI_REQUIRE(wfile.writeline(line));
        // ..should not cause any errors.
        MI_REQUIRE(wfile.error() == 0);
    }

    i = rfile.filesize();
    j = wfile.filesize();
    // Sanity checks
    MI_REQUIRE(i >= 10 && i <= 100000);
    MI_REQUIRE(i == j);

    // The write files file position should be the same as the written
    // byte count.
    Sint64 pos = wfile.tell();
    MI_REQUIRE(pos == j);

    // Closing the files should work fine
    MI_REQUIRE(rfile.close());
    MI_REQUIRE(wfile.close());
}


MI_TEST_AUTO_FUNCTION( verify_rewrite_function )
{
    File wfile;
    char line[1024];

    // Open our temp file again
    MI_REQUIRE(wfile.open("Testfile-l", File::M_READWRITE));

    // Seek to the end (2 == SEEK_END)
    MI_REQUIRE(wfile.seek(0, 2));
    Sint64 pos = wfile.tell();

    // Write at a certain position.
    MI_REQUIRE(wfile.printf("hello, %s\n", "world"));
    // Jump back.
    MI_REQUIRE(wfile.seek(pos));
    // Read it.
    MI_REQUIRE(wfile.readline(line, sizeof(line)));
    // Make sure the content is the same.
    MI_REQUIRE(strcmp(line, "hello, world\n") == 0);

    // Close and remove our temp file.
    MI_REQUIRE(wfile.close());
    MI_REQUIRE(DISK::file_remove("Testfile-l"));
}


MI_TEST_AUTO_FUNCTION( verify_open_function )
{
    std::string src_path = TEST::mi_src_path("base");
    std::string file = TEST::mi_src_path("base/hal/disk/test.cpp");

    Directory dir, dir2;
    File rfile, wfile;

    MI_REQUIRE(rfile.open(file.c_str()));
    MI_REQUIRE(dir.open(src_path.c_str()));
    MI_REQUIRE(!wfile.open("/non_existing"));
    MI_REQUIRE(!dir2.open("/non_existing"));

    // closing a dir will always return true and
    // closing a file which is not open will also always return true.
    MI_REQUIRE(dir.close());
    MI_REQUIRE(dir2.close());
    MI_REQUIRE(rfile.close());
    MI_REQUIRE(wfile.close());
}


MI_TEST_AUTO_FUNCTION( verify_mkdir_function )
{
    // Create a directory.
    DISK::rmdir("Testdir");
    MI_REQUIRE(DISK::mkdir("Testdir"));
    // And we must be able to access our newly created dir.
    MI_REQUIRE(DISK::access("Testdir"));
    // ..and stat it must succeed.
    DISK::Stat file_stat;
    MI_REQUIRE(DISK::stat("Testdir", &file_stat));
    // Since we created a dir it should be reported as a dir.
    MI_REQUIRE(file_stat.m_is_dir);

    DISK::rmdir("Testdir");
}


MI_TEST_AUTO_FUNCTION( verify_rmdir_function )
{
    DISK::mkdir("Testdir");
    // Now remove dir.
    MI_REQUIRE(DISK::rmdir("Testdir"));
    // access() should not fail.
    MI_REQUIRE(!DISK::access("Testdir"));
}


MI_TEST_AUTO_FUNCTION( verify_isdir_function )
{
    std::string src_path = TEST::mi_src_path("");
    std::string file = TEST::mi_src_path("base/hal/disk/test.cpp");

    // Stat MI_SRC dir should be ok.
    DISK::Stat file_stat;
    MI_REQUIRE(DISK::stat(src_path.c_str(), &file_stat));
    // It must be a dir
    MI_REQUIRE(file_stat.m_is_dir);

    // Stat the test file
    MI_REQUIRE(DISK::stat(file.c_str(), &file_stat));
    // Is is NOT a dir, fail if it is reported as such.
    MI_REQUIRE(!file_stat.m_is_dir);
}


MI_TEST_AUTO_FUNCTION( verify_chdir_function )
{
    std::string cwd = get_cwd();
#ifdef MI_PLATFORM_WINDOWS
    char dir_path[] = "c:\\windows\\";
#elif defined(MI_PLATFORM_MACOSX)
    char dir_path[] = "/Applications/";
#else
    char dir_path[] = "/tmp/";
#endif
    // Test if well known directory is a directory, first test it with trailing
    // separator since Windows _stat cannot handle that.
    MI_REQUIRE(DISK::is_directory(dir_path));
    // ..and then without trailing separator.
    dir_path[strlen(dir_path) - 1] = '\0';
    MI_REQUIRE(DISK::is_directory(dir_path));

    // and then check a NULL pointer.
    MI_REQUIRE(DISK::is_directory(NULL) == false);

    // Test that chdir and getcurdir work.
    MI_REQUIRE(DISK::chdir(dir_path));
    MI_REQUIRE_EQUAL(DISK::get_cwd(), dir_path);

    DISK::chdir(cwd.c_str());
}


MI_TEST_AUTO_FUNCTION( remove_files )
{
    // Remove our test file should work.
    MI_REQUIRE(DISK::file_remove("Testrename"));
    // access() should now return an error.
    MI_REQUIRE(!DISK::access("Testrename"));

    // Remove original test file.
    MI_REQUIRE(DISK::file_remove("Testfile"));
    // after remove it should not be possible to access it.
    MI_REQUIRE(!DISK::access("Testfile"));
}

MI_TEST_AUTO_FUNCTION( verify_is_path_absolute )
{
    MI_REQUIRE(DISK::is_path_absolute("/foo/bar/"));
    MI_REQUIRE(DISK::is_path_absolute("/foo/bar"));
    MI_REQUIRE(DISK::is_path_absolute("C:\\Users"));
    MI_REQUIRE(DISK::is_path_absolute("C:\\boot.ini"));

    MI_REQUIRE(!DISK::is_path_absolute("./foo"));
    MI_REQUIRE(!DISK::is_path_absolute("./foo/"));
    MI_REQUIRE(!DISK::is_path_absolute("boot.ini"));

#ifndef WIN_NT
    MI_REQUIRE(DISK::is_path_absolute("/"));
#else
    MI_REQUIRE(DISK::is_path_absolute("C:\\"));
    MI_REQUIRE(DISK::is_path_absolute("C:/"));
    MI_REQUIRE(!DISK::is_path_absolute("C:"));
#endif
}

MI_TEST_AUTO_FUNCTION( verify_is_directory )
{
#ifndef WIN_NT
    MI_REQUIRE(DISK::is_directory("/"));
#else
    MI_REQUIRE(DISK::is_directory("C:\\"));
    MI_REQUIRE(DISK::is_directory("C:\\\\\\"));
    MI_REQUIRE(DISK::is_directory("C:///"));
    MI_REQUIRE(!DISK::is_directory("C:"));
#endif
    MI_REQUIRE(DISK::is_directory("."));
    MI_REQUIRE(DISK::is_directory(".."));

    DISK::rmdir("Testdir");
    DISK::mkdir("Testdir");
    MI_REQUIRE(DISK::is_directory("Testdir"));
    DISK::rmdir("Testdir");
}

MI_TEST_AUTO_FUNCTION( verify_is_file )
{
    std::string file = TEST::mi_src_path("base/hal/disk/test.cpp");
    MI_REQUIRE(DISK::is_file(file.c_str()));
    file = TEST::mi_src_path(std::string());
    MI_REQUIRE(!DISK::is_file(file.c_str()));
}

