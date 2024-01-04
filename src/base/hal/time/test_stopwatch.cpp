/******************************************************************************
 * Copyright (c) 2008-2023, NVIDIA CORPORATION. All rights reserved.
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

/// \file
/// \brief stopwatch test/examples.

#include "pch.h"

#include "i_time.h"

#include <base/system/test/i_test_auto_case.h>
#ifdef MI_TEST_VERBOSE
#include <iostream>
#endif
using namespace MI::TIME;

//----------------------------------------------------------------------
// we spend some time here.
static void spend_time()
{
    // use volatile here to avoid optimizing out the loop.
    volatile MI::Sint32 count = 0;
    for(int i = 0; i < 500; ++i){
        for(int j = 0; j < 1000; ++j){
            for(int k = 0; k < 1000; ++k){
                count = 2 * (count - 1) - count + 3; // anything do
            }
        }
    }
#ifdef MI_TEST_VERBOSE
    std::cout << "spend time: count " << count << std::endl;
#endif
}

//----------------------------------------------------------------------
/// test for stopwatch
MI_TEST_AUTO_FUNCTION( test_stopwatch )
{
    // how to get the local time.
    Stopwatch sw0;
    sw0.start();
    spend_time();
    sw0.stop();
#ifdef MI_TEST_VERBOSE
    std::cout << "Elapsed time = " << sw0.elapsed() << std::endl;
#endif
    // spend_time should spend some time.
    MI_CHECK(sw0.elapsed() > 0.0);

    Stopwatch sw1;
    {
        // scoped start/stop.
        Stopwatch::Scoped_run timer(sw1);
        spend_time();
    }
#ifdef MI_TEST_VERBOSE
    std::cout << "Elapsed time = " << sw1.elapsed() << std::endl;
#endif
    // spend_time should spend some time.
    MI_CHECK(sw1.elapsed() > 0.0);
}

//----------------------------------------------------------------------
