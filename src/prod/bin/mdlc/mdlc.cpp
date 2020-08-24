/******************************************************************************
 * Copyright (c) 2011-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "mdlc.h"

#if defined(LINUX)
#include <dirent.h>
#include <unistd.h>
#elif defined(WIN_NT)
#include <mi/base/miwindows.h>
#include <io.h>
#define access(a, b) _access(a, b)
#else
#include <sys/dir.h>
#include <unistd.h>
#endif

#include <cerrno>
#include <sys/types.h>
#include <sys/stat.h>   // For stat().

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <mi/base/handle.h>
#include <mi/mdl/mdl_generated_dag.h>
#include <mi/mdl/mdl_code_generators.h>
#include <mi/mdl/mdl_mdl.h>
#include <mi/mdl/mdl_module_transformer.h>



#include <string>
#include <algorithm>
#include <base/system/version/i_version.h>

#include <mdl/compiler/compilercore/compilercore_mdl.h>

#include "search_path.h"
#include "getopt.h"

#ifdef WIN_NT
#define strcasecmp(s1, s2) _stricmp(s1, s2)
#endif

using namespace std;

using mi::mdl::IMDL;
using mi::mdl::ISymbol;
using mi::mdl::ISimple_name;
using mi::mdl::IQualified_name;
using mi::mdl::IModule;
using mi::mdl::Messages;
using mi::mdl::IMessage;
using mi::mdl::IPrinter;
using mi::mdl::ISyntax_coloring;
using mi::mdl::IMDL_exporter;
using mi::mdl::ICode_generator;
using mi::mdl::ICode_generator_dag;
using mi::mdl::ICode_generator_jit;
using mi::mdl::IGenerated_code;
using mi::mdl::IGenerated_code_dag;
using mi::mdl::IGenerated_code_executable;
using mi::mdl::IOutput_stream;
using mi::mdl::IInput_stream;
using mi::mdl::IThread_context;
using mi::mdl::IMDL_module_transformer;


using namespace std;

/// Print messages to a printer.
///
/// \param msgs     the messages
/// \param printer  the printer
static void print_messages(Messages const &msgs, IPrinter *printer)
{
    for (size_t i = 0, n = msgs.get_message_count(); i < n; ++i) {
        IMessage const *msg = msgs.get_message(i);
        printer->print(msg, /*include_notes=*/true);
    }
}


Mdlc::Mdlc(char const *program_name)
: m_program(program_name)
, m_dump_dag(false)
, m_verbose(false)
, m_syntax_coloring(false)
, m_show_positions(false)
, m_imdl()
, m_check_root()
, m_internal_space("coordinate_object")
, m_backend_options()
, m_target_lang(TL_NONE)
, m_input_modules()
, m_inline(false)
{
}

Mdlc::~Mdlc()
{
}

void Mdlc::usage()
{
    fprintf(
        stderr,
        "Usage: %s [options] modules\n"
        "Options are:\n"
        "  --optimize <level>\n"
        "  -O <level>\n"
        "\tSet the optimization level:\n"
        "\t\t0\tDisable all optimizations\n"
        "\t\t1\tEnable intraprocedural optimizations\n"
        "\t\t2\tEnable intra- and interprocedural optimizations\n"
        "  --warn <options>\n"
        "  -W <option>\n"
        "\tSet a warning option, one of\n"
        "\t\terr\tThreat all warnings as errors\n"
        "\t\tnum=off\tDisable warning Wnum\n"
        "\t\tnum=on\tEnable warning Wnum\n"
        "\t\tnum=err\tTreat warning Wnum as an error\n"
        "  --strict\n"
        "  --no-strict\n"
        "\tEnables (default) or disables strict compilation mode.\n"
        "  --version\n"
        "  -V\n"
        "\tPrint version of the MDLC compiler.\n"
        "  --verbose\n"
        "  -v\n"
        "\tEnable verbose mode.\n"
        "  --path <path>\n"
        "  -p <path>\n"
        "\tSpecify the MDL module search path.\n"
        "\tThis option can be specified multiple times.\n"
        "  --syntax-coloring\n"
        "  -C\n"
        "\tColor the output.\n"
        "  --check-lib <root>\n"
        "\tCheck a library stored at root.\n"
        "  --target <target>\n"
        "  -t <target>\n"
        "\tSet target language.\n"
        "\tTarget language can be one of\n"
        "\t\tNONE\t(the default)\n"
        "\t\tMDL\n"
        "\t\tDAG\n"
        "\t\tBIN\n"
        "  --dump <option>\n"
        "  -d <option>\n"
        "\tSet dump option.\n"
        "\tDump option can be one of\n"
        "\t\tDG\tauto-typing dependence graph\n"
        "\t\tCG\tcall-graph\n"
        "\t\tDAG\tmaterial expression DAG\n"
        "  --backend <option=value>\n"
        "  -B <option>=<value>\n"
        "\tSet the backend option <option> to value <value>.\n"
        "  --internal-space <space>\n"
        "\tSet internal space, can be one of\n"
        "\t\tinternal\n"
        "\t\tobject\t(the default)\n"
        "\t\tworld\n"
        "  --show-positions\n"
        "\tShow source code position in target output.\n"
        "  --inline\n"
        "  -i\n"
        "\tInlines the given module (if target is set to MDL).\n"
        "  --plugin <filename>\n"
        "  -l\n"
        "\tLoads the given plugin.\n"
        "  --help\n"
        "  -?"
        "\tThis help.\n",
        m_program);
}

int Mdlc::run(int argc, char *argv[])
{
    static mi::getopt::option const long_options[] = {
        /* 0*/ { "optimize",               mi::getopt::REQUIRED_ARGUMENT, NULL, 'O' },
        /* 1*/ { "warn",                   mi::getopt::REQUIRED_ARGUMENT, NULL, 'W' },
        /* 2*/ { "strict",                 mi::getopt::NO_ARGUMENT,       NULL, 0 },
        /* 3*/ { "no-strict",              mi::getopt::NO_ARGUMENT,       NULL, 0 },
        /* 4*/ { "version",                mi::getopt::NO_ARGUMENT,       NULL, 'V' },
        /* 5*/ { "verbose",                mi::getopt::NO_ARGUMENT,       NULL, 'v' },
        /* 6*/ { "path",                   mi::getopt::REQUIRED_ARGUMENT, NULL, 'p' },
        /* 7*/ { "syntax-coloring",        mi::getopt::NO_ARGUMENT,       NULL, 'C' },
        /* 8*/ { "check-lib",              mi::getopt::REQUIRED_ARGUMENT, NULL, 0 },
        /* 9*/ { "target",                 mi::getopt::REQUIRED_ARGUMENT, NULL, 't' },
        /*10*/ { "dump",                   mi::getopt::REQUIRED_ARGUMENT, NULL, 'd' },
        /*11*/ { "backend",                mi::getopt::REQUIRED_ARGUMENT, NULL, 'B' },
        /*12*/ { "internal-space",         mi::getopt::REQUIRED_ARGUMENT, NULL, 0 },
        /*13*/ { "show-positions",         mi::getopt::NO_ARGUMENT,       NULL, 0 },
        /*15*/{ "inline",                  mi::getopt::NO_ARGUMENT,       NULL, 'i' },
        /*16*/{ "plugin",                  mi::getopt::REQUIRED_ARGUMENT, NULL, 'l' },
        /*17*/ { "help",                   mi::getopt::NO_ARGUMENT,       NULL, '?' },
       
        /*18*/ { NULL,                     0,                             NULL, 0 }
    };

    bool opt_error = false;
    bool show_version = false;
    int  c, longidx;
    std::string warn_options;

    MDL_search_path *search_path(new MDL_search_path);

    m_imdl = mi::mdl::initialize();


    mi::mdl::Options &comp_options = m_imdl->access_options();


    std::vector<std::string> plugin_filenames;

    while (
        (c = mi::getopt::getopt_long(argc, argv, "O:W:Vvip:Ct:d:B:l:?", long_options, &longidx)) != -1
    ) {
        switch (c) {
        case 'O':
            {
                char const *s = mi::getopt::optarg;
                char level = s[0];

                bool valid = false;
                if (level == '0' || level == '1' || level == '2' || level == '3') {
                    if (s[1] == '\0') {
                        valid = true;
                        comp_options.set_option(MDL_OPTION_OPT_LEVEL, mi::getopt::optarg);
                    }
                }
                if (!valid) {
                    fprintf(
                        stderr,
                        "%s error: unknown optimization option (%s)\n",
                        argv[0],
                        s);
                    opt_error = true;
                }
            }
            break;
        case 'W':
            {
                const char *s = mi::getopt::optarg;
                unsigned value;
                if (strcasecmp(s, "err") == 0) {
                    // ok
                } else if (sscanf(s, "%u=off", &value) == 1) {
                    // ok
                } else if (sscanf(s, "%u=on", &value) == 1) {
                    // ok
                } else if (sscanf(s, "%u=err", &value) == 1) {
                    // ok
                } else {
                    fprintf(
                        stderr,
                        "%s error: unknown value of warning option (%s)\n",
                        argv[0],
                        s);
                    opt_error = true;
                }
                if (!warn_options.empty())
                    warn_options += ",";
                warn_options += s;
            }
            break;
        case 'V':
            show_version = true;
            break;
        case 'v':
            m_verbose = true;
            break;
        case 'p':
            search_path->add_path(mi::getopt::optarg);
            break;
        case 'C':
            m_syntax_coloring = true;
            break;
        case 't':
            if (strcasecmp(mi::getopt::optarg, "none") == 0) {
                m_target_lang = TL_NONE;
            } else if (strcasecmp(mi::getopt::optarg, "mdl") == 0) {
                m_target_lang = TL_MDL;
            } else if (strcasecmp(mi::getopt::optarg, "dag") == 0) {
                m_target_lang = TL_DAG;
            } else if (strcasecmp(mi::getopt::optarg, "bin") == 0) {
                m_target_lang = TL_BIN;
            } else {
                fprintf(
                    stderr,
                    "%s error: unknown target language '%s'\n",
                    argv[0],
                    mi::getopt::optarg);
                opt_error = true;
            }
            break;
        case 'd':
            if (strcasecmp(mi::getopt::optarg, "dg") == 0) {
                comp_options.set_option(MDL_OPTION_DUMP_DEPENDENCE_GRAPH, "true");
            } else if (strcasecmp(mi::getopt::optarg, "cg") == 0) {
                comp_options.set_option(MDL_OPTION_DUMP_CALL_GRAPH, "true");
            } else if (strcasecmp(mi::getopt::optarg, "dag") == 0) {
                m_dump_dag = true;
            } else {
                fprintf(
                    stderr,
                    "%s error: unknown dump option '%s'\n",
                    argv[0],
                    mi::getopt::optarg);
                opt_error = true;
            }
            break;
        case 'B':
            m_backend_options.push_back(mi::getopt::optarg);
            break;
        case '?':
            usage();
            return EXIT_SUCCESS;
        case 'i':
            m_inline = true;
            break;
        case 'l':
            plugin_filenames.push_back(mi::getopt::optarg);
            break;
        case '\0':
            switch (longidx) {
            case 2:
                comp_options.set_option(MDL_OPTION_STRICT, "true");
                break;
            case 3:
                comp_options.set_option(MDL_OPTION_STRICT, "false");
                break;
            case 8:
                m_check_root = mi::getopt::optarg;
                break;
            case 12:
                if (strcasecmp(mi::getopt::optarg, "internal") == 0) {
                    // ok
                } else if (strcasecmp(mi::getopt::optarg, "object") == 0) {
                    // ok
                } else if (strcasecmp(mi::getopt::optarg, "world") == 0) {
                    // ok
                } else {
                    fprintf(
                        stderr,
                        "%s error: unsupported internal space '%s'\n",
                        argv[0],
                        mi::getopt::optarg);
                    opt_error = true;
                }
                m_internal_space = mi::getopt::optarg;
                break;
            case 13:
                m_show_positions = true;
                break;
            case 16:
                plugin_filenames.push_back(mi::getopt::optarg);
                break;
            default:
                fprintf(
                    stderr,
                    "%s error: unknown option '%s'\n",
                    argv[0],
                    argv[mi::getopt::optind]);
                opt_error = true;
                break;
            }
            break;
        }
    }

    if (opt_error) {
        return EXIT_FAILURE;
    }

    if (!warn_options.empty()) {
        std::string s(warn_options);
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        comp_options.set_option(MDL_OPTION_WARN, s.c_str());
    }


    if (show_version) {
        fprintf(
            stderr,
            "mdlc version 1.0, build %s.\n",
            MI::VERSION::get_platform_version());
        return EXIT_SUCCESS;
    }

    if (!m_check_root.empty()) {
        // add the library root itself
        search_path->add_path(m_check_root.c_str());
    } else if (search_path->get_search_path_count(MDL_search_path::MDL_SEARCH_PATH) == 0) {
        search_path->add_path(".");
    }

    m_imdl->install_search_path(search_path);

    if (mi::getopt::optind >= argc) {
        fprintf(stderr,"%s: no source modules specified\n", argv[0]);
        return EXIT_FAILURE;
    }

    unsigned err_count = 0;

    if (!m_check_root.empty()) {
        find_all_modules(m_check_root.c_str(), NULL);
    }

    for (int i = mi::getopt::optind; i < argc; ++i) {
        m_input_modules.push_back(argv[i]);
    }

    for (String_list::const_iterator it(m_input_modules.begin()), end(m_input_modules.end());
         it != end;
         ++it)
    {
        std::string const &input_module = *it;

        size_t errors = 0;
        mi::base::Handle<IModule const> module;

        if (is_binary(input_module.c_str())) {
            module = mi::base::make_handle(load_binary(input_module.c_str(), errors));
        } else {
            module = mi::base::make_handle(compile(input_module.c_str(), errors));
        }
        if (!module.is_valid_interface())
            return EXIT_FAILURE;
        err_count += errors;

        if (m_check_root.empty()) {
            // compile
            if (!backend(module.get()))
                return EXIT_FAILURE;
        }
    }

    if (!m_check_root.empty()) {
        if (err_count > 0) {
            fprintf(
                stderr,
                "%s: Library '%s' contains %u errors\n",
                m_program,
                m_check_root.c_str(),
                err_count);
        } else {
            fprintf(
                stderr,
                "%s: Successfully checked library '%s'\n",
                m_program,
                m_check_root.c_str());
        }
    }

    return err_count == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

// Compile one module.
IModule const *Mdlc::compile(char const *module_name, size_t &errors)
{
    mi::base::Handle<IThread_context> ctx(m_imdl->create_thread_context());
    IModule const *module = m_imdl->load_module(ctx.get(), module_name, /*cache=*/NULL);

    mi::base::Handle<IOutput_stream> os_stderr(m_imdl->create_std_stream(IMDL::OS_STDERR));
    mi::base::Handle<IPrinter> printer(m_imdl->create_printer(os_stderr.get()));

    printer->enable_color(m_syntax_coloring);

    Messages const &msgs = ctx->access_messages();
    print_messages(msgs, printer.get());

    size_t err_count = msgs.get_error_message_count();
    if (0 < err_count) {
        fprintf(stderr, "%s: %u errors detected in module %s\n",
            m_program, unsigned(err_count), module_name);
    } else if (m_verbose && m_check_root.empty()) {
        fprintf(stderr,"%s: successfully compiled module %s\n", m_program, module_name);
    }

    errors = err_count;
    return module;
}

// Apply backend options.
void Mdlc::apply_backend_options(mi::mdl::Options &opts)
{
    String_list const &bo = m_backend_options;
    for (String_list::const_iterator it(bo.begin()), end(bo.end()); it != end; ++it) {
        std::string const &t = *it;

        size_t pos = t.find('=');

        if (pos != std::string::npos) {
            std::string key(t.substr(0, pos));
            std::string val(t.substr(pos + 1));

            bool res = opts.set_option(key.c_str(), val.c_str());

            if (!res) {
                fprintf(stderr, "Selected backend does not support option '%s'\n", key.c_str());
            }
        } else {
            fprintf(stderr, "Malformed backend option '%s' ignored\n", t.c_str());
        }
    }
}


// Compile a module to a target language.
bool Mdlc::backend(IModule const *module)
{
    mi::base::Handle<IOutput_stream> os_stderr(m_imdl->create_std_stream(IMDL::OS_STDERR));
    mi::base::Handle<IPrinter> printer(m_imdl->create_printer(os_stderr.get()));

    printer->enable_color(m_syntax_coloring);

    switch (m_target_lang) {
    case TL_NONE:
        break;
    case TL_MDL:
        if (m_inline) {
            mi::base::Handle<IMDL_module_transformer> transformer(
                m_imdl->create_module_transformer());
            mi::base::Handle<IModule const> inlined_module(
                transformer->inline_imports(module));
            if (inlined_module.is_valid_interface()) {
                print_generated_code(inlined_module.get());
            } else {
                fprintf(
                    stderr,
                    "%s error: failed to inline module %s\n",
                    m_program, module->get_name());

                Messages const &msgs = transformer->access_messages();
                print_messages(msgs, printer.get());

                return false;
            }
        } else {
            print_generated_code(module);
        }
        break;
    case TL_DAG:
        if (module->is_valid()) {
            mi::base::Handle<ICode_generator_dag> generator =
                mi::base::make_handle(m_imdl->load_code_generator("dag"))
                    .get_interface<ICode_generator_dag>();
            if (!generator.is_valid_interface()) {
                fprintf(
                    stderr,
                    "%s error: failed to load code generator for target language dag\n",
                    m_program);
                return false;
            }

            mi::mdl::Options &dag_opts = generator->access_options();
            dag_opts.set_option(
                MDL_CG_OPTION_INTERNAL_SPACE, m_internal_space.c_str());
            if (m_dump_dag)
                dag_opts.set_option(
                    MDL_CG_DAG_OPTION_DUMP_MATERIAL_DAG, "true");

            // We support local entity usage inside MDL materials in neuray, but ...
            dag_opts.set_option( MDL_CG_DAG_OPTION_NO_LOCAL_FUNC_CALLS, "false");
            // ... we need entries for those in the DB, hence generate them
            dag_opts.set_option( MDL_CG_DAG_OPTION_INCLUDE_LOCAL_ENTITIES, "true");
            // We enable unsafe math optimizations
            dag_opts.set_option(MDL_CG_DAG_OPTION_UNSAFE_MATH_OPTIMIZATIONS, "true");

            apply_backend_options(dag_opts);

            mi::base::Handle<IGenerated_code_dag> dag(generator->compile(module));
            if (!dag.is_valid_interface()) {
                fprintf(stderr, "%s error: failed to generate dag code for module %s\n",
                                m_program, module->get_name());
                return false;
            }

            Messages const &msgs = dag->access_messages();
            print_messages(msgs, printer.get());

            size_t err_count = msgs.get_error_message_count();
            if (0 < err_count) {
                fprintf(stderr, "%s: %u errors detected in dag code generated for module %s\n",
                                m_program, unsigned(err_count), module->get_name());
                return false;
            } else {
                print_generated_code(dag.get());
            }
        }
        break;
    case TL_GLSL:
        break;
    case TL_HLSL:
        break;
    case TL_JIT:
        break;
    case TL_PTX:
        break;
    case TL_BIN:
        if (module->is_valid()) {
            mi::base::Handle<IOutput_stream> os(m_imdl->create_file_output_stream("output.bin"));
            mi::mdl::Stream_serializer stream_serializer(os.get());

            if (os.is_valid_interface()) {
                m_imdl->serialize_module(module, &stream_serializer, true);
            }
        }
        break;
    }

    return true;
}

// Check if the given filename exists and if it represents a binary,
bool Mdlc::is_binary(char const *filename) const
{
    if (FILE *f = fopen(filename, "rb")) {
        bool res = false;
        char sig[4];

        if (fread(sig, 1, 4, f) == 4) {
            res = sig[0] == 'm' && sig[1] == 'd' && sig[2] == 'l' && sig[3] == 'S';
        }
        fclose(f);
        return res;
    }
    return false;
}

// Load a module binary.
IModule const *Mdlc::load_binary(char const *filename, size_t &errors)
{
    mi::base::Handle<IInput_stream> is(m_imdl->create_file_input_stream(filename));
    mi::base::Handle<mi::base::IAllocator> allocator(m_imdl->get_mdl_allocator());
    mi::mdl::Stream_deserializer stream_deserializer(allocator.get(), is.get());

    IModule const *module = m_imdl->deserialize_module(&stream_deserializer);
    if (module == NULL) {
        fprintf(
            stderr, "%s: failed to open binary '%s' for reading\n", m_program, filename);
        return NULL;
    }

    Messages const &msgs = module->access_messages();
    size_t err_count = msgs.get_error_message_count();
    errors = err_count;
    return module;
}


// Prints colorized code to stdout.
void Mdlc::print_generated_code(IModule const *mod)
{
    mi::base::Handle<IOutput_stream> os_stdout(m_imdl->create_std_stream(IMDL::OS_STDOUT));
    if (mod->is_valid()) {
        // use the exporter
        mi::base::Handle<IMDL_exporter> exporter(m_imdl->create_exporter());
        exporter->enable_color(m_syntax_coloring);
        exporter->export_module(os_stdout.get(), mod, /*resource_cb=*/NULL);
    } else if (m_verbose) {
        // use the printer, this module contains errors
        mi::base::Handle<IPrinter> printer(m_imdl->create_printer(os_stdout.get()));
        printer->enable_color(m_syntax_coloring);
        printer->show_positions(m_show_positions);
        printer->print(mod);
    }
}

// Prints colorized code to stdout.
void Mdlc::print_generated_code(IGenerated_code const *code)
{
    mi::base::Handle<IOutput_stream> os_stdout(m_imdl->create_std_stream(IMDL::OS_STDOUT));
    mi::base::Handle<IPrinter> printer(m_imdl->create_printer(os_stdout.get()));
    printer->enable_color(m_syntax_coloring);
    printer->show_positions(m_show_positions);
    printer->print(code);
}

namespace {

/// Represents a single directory in a OS independent way.
class Directory
{
public:
    /// Constructor.
    Directory();

    /// Destructor.
    ~Directory();

    /// Open a directory for reading names in it
    /// \param path to open
    /// \return success
    bool open(char const *path);

    /// Close directory
    /// \return success
    bool close();

    /// Read the next filename from the directory. Names are unsorted.
    /// \return the next filename, or 0 if at eof
    char const *read();

    /// Retrieve whether reading has hit the end of the directory.
    /// \return true if reading has hit the end of the directory
    bool eof() const { return m_eof; }

    /// Retrieve last system error code.
    /// \return last system error code
    int error() const { return m_error; }

    /// Check if the given name is a directory.
    bool isdir(char const *name) const;

private:
    struct Hal_dir;

    std::string   m_path;         ///< last path passed to open()
    int             m_error;        ///< last error, 0 if none
    bool            m_eof;          ///< hit EOF while reading?
#ifdef WIN_NT
    Hal_dir         *m_dir;         ///< information for windows-based dir searching

                                    /// An internal, windows-specific helper method to encapsulate
                                    /// the directory reading code.
                                    /// \return success
    bool read_next_file();

    /// Retrieve whether a given path exists.
    /// For now this is simply a (static) helper method, but might be a useful
    /// method to expose publicly and for all platforms
    /// \param path path in question
    /// \return true, if path exists
    static bool exists(char const *path);
#else
    Hal_dir         *m_dp_wrapper;  ///< open directory, 0 if not open
#endif
};

#ifdef WIN_NT

struct Directory::Hal_dir
{
    WIN32_FIND_DATA m_find_data;
    HANDLE m_first_handle;
    bool m_opened;
    bool m_first_file;
};

Directory::Directory()
    : m_path()
    , m_error(0)
    , m_eof(true)
    , m_dir(NULL)
{
    m_dir = new Hal_dir;
    memset(&(m_dir->m_find_data), 0, sizeof(WIN32_FIND_DATA));
    m_dir->m_first_handle = INVALID_HANDLE_VALUE;
    m_dir->m_opened = false;
    m_dir->m_first_file = true;
}

Directory::~Directory()
{
    close();
    delete m_dir;
}

bool Directory::open(
    char const *path)
{
    m_eof = false;
    if (m_dir->m_opened && !close())
        return false;

    std::string new_path(path ? path : "");

    // if we find a '*', just leave things as they are
    // note that this will likely not work for a 'c:/users/*/log' call
    if (strchr(new_path.c_str(), '*') == NULL) {
        size_t len = strlen(path);

        // need this as m_path is const char *
        std::string temp_path(new_path);

        if (len == 0) { // empty string -- assume they just want the curr dir
            temp_path = "*";
        } else if (new_path[len - 1] == '/' || new_path[len - 1] == '\\') {
            // there is a trailing delimiter, so we just add the wildcard
            temp_path += "*";
        } else {
            // no trailing delimiter -- add one (and also the '*')
            temp_path += "/*";
        }

        m_path = temp_path;
    } else
        m_path = new_path;

    // check for existence -- user is not going to be able to find anything
    // in a directory that isn't there
    if (!Directory::exists(new_path.c_str())) {
        return false;
    }

    // This flag tells the readdir method whether it should invoke
    // FindFirstFile or FindNextFile
    m_dir->m_first_file = true;
    m_dir->m_first_handle = INVALID_HANDLE_VALUE;

    // and now we indicate we've been opened -- we don't really
    // do much with this open call, it's the first search that matters
    m_dir->m_opened = true;
    return true;
}

bool Directory::close()
{
    bool ret_val = true;

    if (m_dir->m_opened && m_dir->m_first_handle != INVALID_HANDLE_VALUE) {
        // FindClose returns BOOL not bool, so we check this way
        ret_val = (::FindClose(m_dir->m_first_handle) != 0);
    }

    m_dir->m_opened = false;
    m_dir->m_first_file = true;
    m_dir->m_first_handle = INVALID_HANDLE_VALUE;
    return ret_val;
}

bool Directory::read_next_file()
{
    // return if we haven't been opened already
    if (!m_dir->m_opened)
        return false;

    bool success = false;
    if (m_dir->m_first_file) {
        m_dir->m_first_handle = ::FindFirstFile(
            m_path.c_str(),			// our path
            &m_dir->m_find_data);	// where windows puts the results

        if (m_dir->m_first_handle != INVALID_HANDLE_VALUE) {
            success = true;
            // so we don't call this block again
            m_dir->m_first_file = false;
        } else {
            m_error = GetLastError();
            if (m_error == ERROR_NO_MORE_FILES) { // not really an error
                m_error = 0;
                m_eof = true;
            }
        }
    } else {
        // FindNextFile returns BOOL not bool, so we check this way
        if (::FindNextFile(
            m_dir->m_first_handle,  // what we got before
            &m_dir->m_find_data)    // where windows puts the results
            != 0)
            success = true;
        else {
            m_error = GetLastError();
            if (m_error == ERROR_NO_MORE_FILES) { // not really an error
                m_error = 0;
                m_eof = true;
            }
        }
    }
    return success;
}

char const *Directory::read()
{
    m_error = 0;
    if (m_dir->m_opened && read_next_file()) {
        // We don't dup the returned data
        return m_dir->m_find_data.cFileName;
    }

    return NULL;
}

bool Directory::exists(
    char const *path)
{
    std::string newpath = path ? path : "";

    // let's strip off any trailing *'s, forward- or back-slashes
    size_t len = newpath.size();
    while (len > 0) {
        char c = newpath[len - 1];
        if (c != '*' && c != '/' && c != '\\')
            break;
        --len;
    }
    newpath = newpath.substr(0, len);

    // CreateFile will fail on a directory under Win95/98/Me, what is our
    // minimum spec?
    HANDLE hDir = ::CreateFile(
        newpath.c_str(),    // what are we opening
        0,			        // access, we can use 0 for existence test
        FILE_SHARE_READ,    // share mode
        NULL,               // security attributes
        OPEN_EXISTING,      // creation disposition
        FILE_FLAG_BACKUP_SEMANTICS,// flags & attrs, need this one for a dir
        0);                 // template file

    if (hDir == INVALID_HANDLE_VALUE)
        return false;
    else {
        ::CloseHandle(hDir);
        return true;
    }
}


// Check if the given name is a directory.
bool Directory::isdir(char const *name) const
{
    struct stat sb;
    if (stat(name, &sb) != 0)
        return false;
    return (sb.st_mode & _S_IFDIR) != 0;
}

#else

// Wrapper for the Unix DIR structure.
struct Directory::Hal_dir
{
    DIR *m_dp;			// open directory, 0 if not open
};

Directory::Directory()
    : m_path()
    , m_error(0)
    , m_eof(true)
    , m_dp_wrapper(NULL)
{
    m_dp_wrapper = new Hal_dir;
    m_dp_wrapper->m_dp = 0;
}

Directory::~Directory()
{
    close();
    delete m_dp_wrapper;
}

bool Directory::open(
    char const *path)
{
    m_eof = false;
    if (m_dp_wrapper->m_dp && !close())
        return false;

    m_path = path ? path : "";

    if ((m_dp_wrapper->m_dp = opendir(m_path.c_str())) != 0) {
        m_error = 0;
        return true;
    } else {
        m_error = errno;
        return false;
    }
}

bool Directory::close()
{
    if (m_dp_wrapper->m_dp) {
        closedir(m_dp_wrapper->m_dp);
        m_dp_wrapper->m_dp = 0;
    }
    return true;
}

const char *Directory::read()
{
    m_error = 0;
    for (;;) {
        struct dirent *entry = readdir(m_dp_wrapper->m_dp);
        if (entry == NULL) {
            m_eof = true;
            return NULL;
        }
        return entry->d_name;
    }
}

// Check if the given name is a directory.
bool Directory::isdir(char const *name) const
{
    struct stat sb;
    if (stat(name, &sb) != 0)
        return false;
    return sb.st_mode & S_IFDIR;
}

#endif

} // anonymous

// Find all modules in a library.
void Mdlc::find_all_modules(
    char const *root,
    char const *package)
{
    Directory dir;

    dir.open(root);

    for (char const *fname = dir.read(); fname != NULL; fname = dir.read()) {
        if (fname[0] == '.')
            continue;

        size_t len = strlen(fname);
        if (len >= 5 &&
            fname[len - 4] == '.' &&
            fname[len - 3] == 'm' &&
            fname[len - 2] == 'd' &&
            fname[len - 1] == 'l')
        {
            std::string mod_name;

            if (package != NULL) {
                mod_name = package;
                mod_name += "::";
            }

            mod_name += std::string(fname, len - 4);

            m_input_modules.push_back(mod_name);
        } else {
            // not an mdl file
            std::string full_name(root);
            full_name += "/";
            full_name += fname;

            if (dir.isdir(full_name.c_str())) {
                std::string package_name;

                if (package != NULL) {
                    package_name = package;
                    package_name += "::";
                }
                package_name += fname;

                find_all_modules(full_name.c_str(), package_name.c_str());
            }
        }
    }
}

