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
 *****************************************************************************/

#include <cstdlib>
#include <cstring>
#include <cstdio>

#include "getopt.h"

namespace mi {
namespace getopt {

char const *optarg = NULL;
int optind = 0;
int opterr = 1;
int optopt = 0;

static char *next = NULL;

static int handle_long(
    int                 argc,
    char * const        argv[],
    struct option const *longopts,
    char const          *opt,
    int                 *longindex)
{
    char const *e = strchr(opt, '=');
    size_t l = e == NULL ? strlen(opt) : e - opt;

    ++optind;
    for (int i = 0; longopts[i].name != NULL; ++i) {
        struct option const &o = longopts[i];

        if (strncmp(o.name, opt, l) == 0 && o.name[l] == '\0') {
            // found
            if (o.has_arg) {
                if (e != NULL) {
                    optarg = e + 1;
                } else if (optind < argc) {
                    optarg = argv[optind];
                    ++optind;
                } else {
                    if (opterr) {
                        fprintf(stderr, "%s: option '--%s' requires an argument\n",
                            argv[0], opt);
                    }
                    return ':';
                }
            } else {
                if (e != NULL) {
                    // extra argument given
                    if (opterr) {
                        fprintf(stderr, "%s: option ' --%s' does not require an argument\n",
                            argv[0], opt);
                    }
                    return ':';
                }
            }
            if (longindex != NULL)
                *longindex = i;
            if (o.flag != NULL) {
                *o.flag = o.val;
                return 0;
            }
            return o.val;
        }
    }
    if (opterr) {
        fprintf(stderr, "%s: unknown option '-%s'\n", argv[0], opt);
    }

    return '?';
}

int getopt_long(
    int                 argc,
    char * const        argv[],
    const char          *optstring,
    struct option const *longopts,
    int                 *longindex)
{
    if (longindex != NULL)
        *longindex = -1;
    if (argv == NULL || optstring == NULL || longopts == NULL)
        return -1;

    if (next == NULL || *next == '\0') {
        if (optind == 0)
            ++optind;

        if (optind >= argc || argv[optind][0] != '-' || argv[optind][1] == '\0')
            return -1;

        if (argv[optind][1] == '-') {
            if (argv[optind][2] == '\0') {
                // a '--' stops parsing
                ++optind;
                return -1;
            }

            // else a long option
            next = NULL;
            return handle_long(argc, argv, longopts, argv[optind] + 2, longindex);
        }
        next = &argv[optind][1];
        ++optind;
    }

    char option = *next++;

    char const *found = strchr(optstring, option);

    if (found == NULL || option == ':') {
        if (opterr) {
            fprintf(stderr, "%s: unknown option '-%c'\n", argv[0], option);
        }
        return '?';
    }

    if (found[1] == ':') {
        // extra argument expected
        if (*next != '\0') {
            optarg = next;
            next   = NULL;
        } else if (optind < argc) {
            optarg = argv[optind];
            ++optind;
        } else {
            if (opterr) {
                fprintf(stderr, "%s: option '-%c' requires an argument\n", argv[0], option);
            }
            return ':';
        }
    }
    return option;
}


}  // getopt
}  // mi
