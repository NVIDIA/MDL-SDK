/******************************************************************************
 * Copyright (c) 2006-2024, NVIDIA CORPORATION. All rights reserved.
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
 /// \brief Test bitvector arrays.

#include "test.h"
#include "i_cont_bitvector.h"

#include <base/lib/log/i_log_logger.h>
#include <base/system/main/i_module_id.h>

using namespace MI::CONT;
using MI::LOG::Mod_log;

class Test_Bitvector : public Test
{
  public:
    virtual ~Test_Bitvector() {};

    bool test() {
#ifdef MI_TEST_VERBOSE
        mod_log->progress(M_CONT, Mod_log::C_MEMORY, 100, "-- Allocators --");
#endif
        {
            Bitvector bits_01(0);
            MI_REQUIRE_EQUAL(bits_01.size(), 0);
            // print(bits_01);

            Bitvector bits_02(10);
            MI_REQUIRE_EQUAL(bits_02.size(), 10);
            // print(bits_02);
            MI_REQUIRE(is_cleared(bits_02));
            bits_02.set(5);
            bits_02.set(9);
            MI_REQUIRE(!is_cleared(bits_02));

            // copy constructor
            Bitvector bits_03(bits_02);
            // print(bits_03);
            MI_REQUIRE(is_equal(bits_03, bits_02));

            // clear
            bits_02.clear();
            MI_REQUIRE(is_cleared(bits_02));

            // assignment
            Bitvector bits_04(0);
            bits_04 = bits_03;
            MI_REQUIRE_EQUAL(bits_04.size(), bits_03.size());
            // print(bits_04);
            MI_REQUIRE(is_equal(bits_04, bits_03));
            bits_04.set(2);
            bits_04.set(3);

            Bitvector bits_05(20);
            bits_05 = bits_04;
            MI_REQUIRE_EQUAL(bits_05.size(), bits_04.size());
            MI_REQUIRE_EQUAL(bits_05.capacity(), 24);
            MI_REQUIRE(is_equal(bits_04, bits_05));
            // print(bits_05);
        }

#ifdef MI_TEST_VERBOSE
        mod_log->progress(M_CONT, Mod_log::C_MEMORY, 100, "-- Resize --");
#endif
        {
            Bitvector bits_01(19030);
            bits_01.resize(19031);
            MI_REQUIRE(is_cleared(bits_01));
            bits_01.set(19030);
            bits_01.resize(10);
            bits_01.resize(19031);
            MI_REQUIRE(!bits_01.is_set(19030));

            Bitvector bits_02(10);
            bits_02.set(8);
            bits_02.set(2);
            bits_02.resize(6);
            MI_REQUIRE(bits_02.is_set(2));
            bits_02.resize(10);
            MI_REQUIRE(bits_02.is_set(2));
            for (int i=6; i<10; ++i)
                MI_REQUIRE(!bits_02.is_set(i));

            int bsize = 19031;
            Bitvector bits_03(bsize-2);
            bits_03.resize(bsize);
            MI_REQUIRE(is_cleared(bits_03));
            bits_03.set(bsize-3);
            bits_03.set(bsize-2, false);
            bits_03.set(bsize-1);
            bits_03.resize(bsize-1);
            MI_REQUIRE(bits_03.is_set(bsize-3));
            MI_REQUIRE(!bits_03.is_set(bsize-2));
            bits_03.resize(bsize);
            MI_REQUIRE(!bits_01.is_set(bsize-1));
        }

#ifdef MI_TEST_VERBOSE
        mod_log->progress(M_CONT, Mod_log::C_MEMORY, 100, "-- Binary data --");
#endif
        {
            Bitvector pixel_mask(0);
            MI_REQUIRE_EQUAL(pixel_mask.size(), 0);
            MI_REQUIRE_EQUAL(pixel_mask.get_binary_size(), 0);
            pixel_mask.resize(10);
            pixel_mask.clear();
            MI_REQUIRE_EQUAL(pixel_mask.size(), 10);
            MI_REQUIRE_EQUAL(pixel_mask.get_binary_size(), 2);
            pixel_mask.set(5);
            MI_REQUIRE(pixel_mask.is_set(5));
        }

        return true;
    }

  private:
    void print(const Bitvector& bits) {
#ifdef MI_TEST_VERBOSE
        mod_log->debug(M_CONT, Mod_log::C_MEMORY,
                       "Bitvector has %d elements", (int)bits.size());

        for (size_t i=0; i < bits.size(); ++i)
            mod_log->debug(M_CONT, Mod_log::C_MEMORY,"[%d]\t%d",
                (int)i, (int)bits.is_set(i));
#endif
    }

    bool is_cleared(const Bitvector& bits) {
        for (size_t i=0; i < bits.size(); ++i) {
            if (bits.is_set(i))
                return false;
        }
        return true;
    }

    bool is_equal(const Bitvector& bits0, const Bitvector& bits1) {
        MI_REQUIRE_EQUAL(bits0.size(), bits1.size());
        if ( bits0.size() != bits1.size())
            return false;
        for (size_t i=0; i < bits0.size(); ++i) {
            MI_REQUIRE_EQUAL(bits0.is_set(i), bits1.is_set(i));
            if (bits0.is_set(i) != bits1.is_set(i))
                return false;
        }
        return true;
    }
};
