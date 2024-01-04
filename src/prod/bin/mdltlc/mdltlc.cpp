/******************************************************************************
 * Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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

#include <mdl/compiler/compilercore/compilercore_mdl.h>

#include <base/system/version/i_version.h>

#include "getopt.h"
#include "mdltlc.h"

#include "mdltlc_compiler.h"

#define MDLTLC_VERSION "0.1"

Mdltlc::Mdltlc(char const *program_name)
    : m_program(program_name)
    , m_imdl()
{
}

/// Print a short usage summary to stderr.
void Mdltlc::usage()
{
    fprintf(
        stderr,
        "Usage: %s [options] modules\n"
        "Options are:\n"
        "  -?"
        "\t\t\t\tThis help.\n"
        "  -v level"
        "\t\t\tverbosity level, 0 is quiet (default)\n"
        "  --plugin"
        "\t\t\tgenerate code for a distiller plugin (legacy option, ignored).\n"
        "  --output-dir=DIR"
        "\t\tgenerate files in the given directory.\n"
        "  --mdl-path=DIR"
        "\t\tadd the given directory to the MDL search path.\n"
        "  --generate"
        "\t\t\tgenerate .h/.cpp files (off by default).\n"
        "  --normalize-mixers"
        "\t\tenable mixer normalization (off by default).\n"
        "  --all-errors"
        "\t\t\tdo not cut off list of error messages (off by default).\n"
        "  --warn=non-normalized-mixers"
        "\temit warnings for mixer call patterns that are not normalized.\n"
        "  --warn=overlapping-patterns"
        "\temit warnings for possibly overlapping patterns.\n"
        "  --debug=builtin-loading"
        "\toutput debug messages related to builtin loading.\n"
        "  --debug=dump-builtins"
        "\t\toutput the environment containing builtin definitions.\n",
        m_program);
}

mi::base::Handle<Compiler> Mdltlc::create_compiler() {
    mi::mdl::IAllocator *allocator = m_imdl->get_mdl_allocator();

    mi::mdl::Allocator_builder builder(allocator);

    return mi::base::make_handle(builder.create<Compiler>(m_imdl.get()));
}


int Mdltlc::run(int argc, char *argv[])
{
    static mi::getopt::option const long_options[] = {
        /* 0*/ { "help",                   mi::getopt::NO_ARGUMENT,       NULL, '?' },
        /* 1*/ { "verbose",                mi::getopt::REQUIRED_ARGUMENT, NULL, 'v' },
        /* 2*/ { "plugin",                 mi::getopt::NO_ARGUMENT,       NULL, 0 },
        /* 3*/ { "generate",               mi::getopt::NO_ARGUMENT,       NULL, 'g' },
        /* 4*/ { "normalize-mixers",       mi::getopt::NO_ARGUMENT,       NULL, 0 },
        /* 5*/ { "all-errors",             mi::getopt::NO_ARGUMENT,       NULL, 0 },
        /* 6*/ { "output-dir",             mi::getopt::REQUIRED_ARGUMENT, NULL, 0 },
        /* 7*/ { "mdl-path",               mi::getopt::REQUIRED_ARGUMENT, NULL, 0 },
        /* 8*/ { "warn",                   mi::getopt::REQUIRED_ARGUMENT, NULL, 0 },
        /* 9*/ { "debug",                  mi::getopt::REQUIRED_ARGUMENT, NULL, 0 },
        /*10*/ { NULL,                     0,                             NULL, 0 }
    };

    bool opt_error = false;
    bool show_version = false;

    m_imdl = mi::mdl::initialize();

    mi::base::Handle<Compiler> compiler = create_compiler();

    Compiler_options &comp_options = compiler->get_compiler_options();

    int  c, longidx;

    while ((c = mi::getopt::getopt_long(argc, argv, "?v:g", long_options, &longidx)) != -1) {
        switch (c) {
        case '?':
            usage();
            return EXIT_SUCCESS;

        case 'v':
        {
            char varg = mi::getopt::optarg[0];
            if (mi::getopt::optarg[0] == '\0' || mi::getopt::optarg[1] != '\0' || varg < '0' || varg > '9') {
                opt_error = true;
                fprintf(stderr, "%s: argument to -v/--verbose must be between 0 and 9\n", argv[0]);
            }
            else {
                comp_options.set_verbosity(std::atoi(mi::getopt::optarg));
            }
        break;
        }

        case 'g':
        {
            comp_options.set_generate(true);
            break;
        }

        case '\0':
            switch (longidx) {
            case 2: /* --plugin*/

                // This is a legacy option and is ignored. We
                // recognize it for compatibility for now.

                break;

            case 4: /* --normalize-mixers */
                comp_options.set_normalize_mixers(true);
                break;

            case 5: /* --all-errors */
                comp_options.set_all_errors(true);
                break;

            case 6: /* --output-dir */
                comp_options.set_output_dir(mi::getopt::optarg);
                break;

            case 7: /* --mdl-path */
                comp_options.add_mdl_path(mi::getopt::optarg);
                break;

            case 8: /* --warn */
            {
                std::string arg(mi::getopt::optarg);
                if (arg == "non-normalized-mixers") {
                    comp_options.set_warn_non_normalized_mixers(true);
                    continue;
                }
                if (arg == "overlapping-patterns") {
                    comp_options.set_warn_overlapping_patterns(true);
                    continue;
                }
                fprintf(
                    stderr,
                    "%s error: unknown argument to --warn: '%s'\n",
                    argv[0],
                    mi::getopt::optarg);
                opt_error = true;
                break;
            }

            case 9: /* --debug */
            {
                std::string arg(mi::getopt::optarg);
                if (arg == "builtin-loading") {
                    comp_options.set_debug_builtin_loading(true);
                    continue;
                }
                if (arg == "dump-builtins") {
                    comp_options.set_debug_dump_builtins(true);
                    continue;
                }
                fprintf(
                    stderr,
                    "%s error: unknown argument to --debug: '%s'\n",
                    argv[0],
                    mi::getopt::optarg);
                opt_error = true;
                break;
            }

            default:
                fprintf(
                    stderr,
                    "%s error: unknown option '%s'\n",
                    argv[0],
                    argv[mi::getopt::optind]);
                opt_error = true;
                break;
            }
        }
    }

    if (opt_error) {
        return EXIT_FAILURE;
    }

    if (show_version) {
        fprintf(
            stderr,
            "mdltlc version " MDLTLC_VERSION ", build %s.\n",
            MI::VERSION::get_platform_version());
        return EXIT_SUCCESS;
    }

    if (mi::getopt::optind >= argc) {
        fprintf(stderr,"%s: no input rule file specified\n", argv[0]);
        return EXIT_FAILURE;
    }

    for (int i = mi::getopt::optind; i < argc; i++) {
        comp_options.add_filename(argv[i]);
    }

    unsigned err_count = 0;

    if (comp_options.get_verbosity() >= 3) {
        printf(
            "verbosity: %d\n"
            "output-dir: %s\n"
            "generate: %d\n"
            "normalize-mixers: %d\n",
            comp_options.get_verbosity(),
            comp_options.get_output_dir() ? comp_options.get_output_dir() : "<unspecified>",
            comp_options.get_generate(),
            comp_options.get_normalize_mixers()
            );
        printf("Input files:\n");
        for (int i = 0; i < comp_options.get_filename_count(); i++) {
            printf("  %s\n", comp_options.get_filename(i));
        }
        printf("MDL path:\n");
        for (int i = 0; i < comp_options.get_mdl_path_count(); i++) {
            printf("  %s\n", comp_options.get_mdl_path(i));
        }
        fflush(stdout);
    }

    compiler->run(err_count);

    return err_count == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
