/******************************************************************************
 * Copyright (c) 2004-2023, NVIDIA CORPORATION. All rights reserved.
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
 /// \brief Test dynamic arrays.

#include "test.h"
#include "i_cont_array.h"

#include <base/lib/log/i_log_logger.h>
#include <base/system/main/i_module_id.h>

using namespace MI::CONT;
using MI::LOG::Mod_log;

static int global_counter = 0;

struct Foo
{
    Foo() 		: first(10), second(11)	{ ++global_counter; }
    Foo(int i, int j) 	: first(i), second(j) 	{ ++global_counter; }
    Foo(const Foo& foo)
      : first(foo.first), second(foo.second)	{ ++global_counter; }

    ~Foo() 					{ --global_counter; }

    int first;
    int second;
};

struct Bar
{
    Bar() : m_i(0){
	//mod_log->info(M_CONT, Mod_log::C_MEMORY, "Bar()");
    }
    explicit Bar(int i) : m_i(i){
	//mod_log->info(M_CONT, Mod_log::C_MEMORY, "Bar(int)");
    }
    Bar(const Bar& bar) {
	//mod_log->info(M_CONT, Mod_log::C_MEMORY, "Bar(const Bar&)");
	m_i = bar.m_i;
    }
    ~Bar() {
	//mod_log->info(M_CONT, Mod_log::C_MEMORY, "~Bar()");
    }
    Bar& operator=(const Bar& bar) {
	if (this != &bar)
	    m_i = bar.m_i;

	//mod_log->info(M_CONT, Mod_log::C_MEMORY, "operator=()");
	return *this;
    }

    int m_i;
};


class Test_Array : public Test
{
  public:
    virtual ~Test_Array() {};

    bool test() {
	//mod_log->progress(M_CONT, Mod_log::C_MEMORY, 100, "-- Allocators --");
	{
	    Array<int> array_01;
	    MI_REQUIRE_EQUAL(array_01.size(), 0);
	    print(array_01);

	    Array<int> array_02(0);
	    MI_REQUIRE_EQUAL(array_02.size(), 0);
	    print(array_02);

	    Array<int> array_03(10);
	    MI_REQUIRE_EQUAL(array_03.size(), 10);
	    print(array_03);

	    Array<int> array_04(array_03);
	    MI_REQUIRE_EQUAL(array_04.size(), array_03.size());
	    print(array_04);

	    Array<int> array_05(5, 1);
	    MI_REQUIRE_EQUAL(array_05.size(), 5);
	    MI_REQUIRE_EQUAL(array_05[0], 1);
	    MI_REQUIRE_EQUAL(array_05[array_05.size()-1], 1);

	    // assignment
	    Array<int> array_06;
	    array_06 = array_05;
	    MI_REQUIRE_EQUAL(array_06.size(), array_05.size());
	    MI_REQUIRE_EQUAL(array_06[0], array_05[0]);
	    MI_REQUIRE(
		   array_06[array_06.size()-1] ==
		   array_05[array_06.size()-1]);
	    print(array_06);

	    Array<int> array_07(20, 2);
	    array_07 = array_05;
	    MI_REQUIRE_EQUAL(array_07.size(), array_05.size());
	    MI_REQUIRE_EQUAL(array_07[0], array_05[0]);
	    MI_REQUIRE(
		   array_07[array_07.size()-1] ==
		   array_05[array_07.size()-1]);
	    print(array_07);

	}

	//mod_log->progress(M_CONT, Mod_log::C_MEMORY, 100, "-- Insertions --");
	{
	    //mod_log->progress(M_CONT, Mod_log::C_MEMORY, 100,	"Insertions via insert()");
	    Array<int> array_02;
	    array_02.insert(10, -1);
	    MI_REQUIRE_EQUAL(array_02[array_02.size()-1], -1);
	    MI_REQUIRE_EQUAL(array_02.size(), 11);
	    array_02.insert(10, -2);
	    MI_REQUIRE_EQUAL(array_02[array_02.size()-1], -1);
	    MI_REQUIRE_EQUAL(array_02[array_02.size()-2], -2);
	    array_02.insert(0, -3);
	    MI_REQUIRE_EQUAL(array_02[array_02.size()-1], -1);
	    MI_REQUIRE_EQUAL(array_02[0], -3);
	    array_02.insert(array_02.size()-1, -4);
	    MI_REQUIRE_EQUAL(array_02[array_02.size()-1], -1);
	    MI_REQUIRE_EQUAL(array_02[array_02.size()-2], -4);
	    array_02.insert(array_02.size(), -5);
	    MI_REQUIRE_EQUAL(array_02[array_02.size()-1], -5);
	    MI_REQUIRE_EQUAL(array_02[array_02.size()-2], -1);
	    print(array_02);
	}

	//mod_log->progress(M_CONT, Mod_log::C_MEMORY, 100, "-- Reserve --");
	{
	    Array<int> array_01(10);
	    array_01.reserve(1000);
	}

	MI_REQUIRE_EQUAL(global_counter, 0);
	//mod_log->progress(M_CONT, Mod_log::C_MEMORY, 100, "-- Foo Tests --");
	{
	    Array<Foo> array_01;
	    Array<Foo> array_02(200);
	    Array<Foo> array_03(array_02);
	    Array<Foo> array_04(111, Foo(11, 22));

	    MI_REQUIRE(global_counter != 0);

	}
	MI_REQUIRE_EQUAL(global_counter, 0);

	return true;
    }

  private:
    template <typename T>
    void print(const Array<T>& array) {
#ifdef MI_TEST_VERBOSE
	mod_log->debug(M_CONT, Mod_log::C_MEMORY, "Array has %d elements", (int)array.size());

	typename Array<T>::Const_iterator iter(array);
	int i=0;
	for (iter.to_first(); !iter.at_end(); iter.to_next(), ++i)
	    mod_log->debug(M_CONT, Mod_log::C_MEMORY,"[%d]\t%d", i, *iter);
#endif
    }

};
