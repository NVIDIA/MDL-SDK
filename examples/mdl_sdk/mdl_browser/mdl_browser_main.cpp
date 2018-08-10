/******************************************************************************
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

//Defines the entry point for the console application.

// system
#include <iostream>

// Qt
#include <QtQuick>
#include <QtQuickControls2/QQuickStyle>
#include <QQmlApplicationEngine>
#include <QtSvg>

// MDL
#include "mdl_sdk_wrapper.h"

// project
#include "view_model/view_model.h"

#include "view_model/navigation/vm_nav_stack.h"
#include "view_model/navigation/vm_nav_stack_level_model.h"
#include "view_model/navigation/vm_nav_stack_level_proxy_model.h"
#include "view_model/navigation/vm_nav_package.h"

#include "view_model/selection/vm_sel_model.h"
#include "view_model/selection/vm_sel_proxy_model.h"
#include "view_model/selection/vm_sel_element.h"
#include "mdl_browser_settings.h"
#include "utilities/platform_helper.h"
#include "utilities/qt/mdl_archive_image_provider.h"

void keep_console_open();


enum Mdl_browser_state
{
    STATE_NOT_OPENED_YET,
    STATE_OPEN,
    STATE_CONFIRMED,
    STATE_ABORTED,
    STATE_ERROR = -1
};


// interface for communicating between the application and the "dialogue"
struct Mdl_browser_context
{
    // path that contains the 'resource'-folder
    const char* application_folder;

    Mdl_browser_state state;
    const char* result;

    Mdl_browser_context()
        : application_folder(nullptr)
        , state(Mdl_browser_state::STATE_NOT_OPENED_YET)
        , result(nullptr)
    {
    }

    ~Mdl_browser_context()
    {
        if (result) delete[] result;
    }
};


// Command line options structure.
struct Mdl_browser_command_line_options
{
    // search paths
    std::vector<std::string> search_paths;

    bool cache_rebuild;
    bool keep_open;

    // The constructor.
    Mdl_browser_command_line_options(int argc, char* argv[])
        : search_paths()
        , cache_rebuild(false)
        , keep_open(false)
        , prog_name(argv[0])
    {
        for (int i = 1; i < argc; ++i)
        {
            char const *opt = argv[i];
            if (opt[0] == '-')
            {
                // options without argument
                if (strcmp(opt, "-h") == 0 || strcmp(opt, "--help") == 0)
                    print_usage();
 
                else if (strcmp(opt, "-c") == 0 || strcmp(opt, "--cache_rebuild") == 0)
                    cache_rebuild = true;

                else if (strcmp(opt, "-k") == 0 || strcmp(opt, "--keep_open") == 0)
                    keep_open = true;

                // options with one argument
                else if(i < argc - 1)
                {
                    if (strcmp(opt, "-p") == 0 || strcmp(opt, "--mdl_path") == 0)
                        search_paths.emplace_back(argv[++i]);
                }
                else
                {
                    std::cerr << "Invalid number of arguments." << std::endl;
                    print_usage();
                    keep_console_open();
                    exit(EXIT_FAILURE);
                }
            }
            else
            {
                std::cerr << "Invalid parameter." << std::endl;
                print_usage();
                keep_console_open();
                exit(EXIT_FAILURE);
            }
        }
    }

    // print print_usage to help specifying the options corretly
    void print_usage()
    {
        std::cout
            << "Usage: " << prog_name.c_str() << " [options] [<value>]\n"
            << "Options:\n"
            << "  -h|--help                     prints these usage instructions\n"
            << "  -c|--cache_rebuild            force a rebuild of the cache file\n"
            << "  -k|--keep_open                reopens the browser until the console is closed.\n"
            << "  -p|--mdl_path <path>          mdl search path, can occur multiple times.\n"
            << std::endl;
        keep_console_open();
        exit(EXIT_FAILURE);
    }

private:
    std::string prog_name;
};



bool open_mdl_browser_dialog(Mdl_browser_context& context, 
                             const Mdl_browser_command_line_options& options)
{
    if (!context.application_folder)
    {
        std::cerr << "[open_mdl_browser_dialog] context.application_folder not set.\n";
        context.state = STATE_ERROR;
        return false;
    }

    // Setup MDL
    Mdl_sdk_wrapper mdl(options.search_paths, options.cache_rebuild);

    // Back-end
    View_model modelData(&mdl, context.application_folder);


    // Setup and run Qt App
    {
        #if defined(Q_OS_WIN)
            QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
        #endif

        int argc = 0;
        QGuiApplication app(argc, nullptr);
        app.setWindowIcon(
            QIcon(context.application_folder + QString("/resources/graphics/nvidia_icon.svg")));

        qmlRegisterType<View_model>("mdl.browser.navigation", 1, 0, "ViewModel");
        qmlRegisterType<VM_nav_stack>("mdl.browser.navigation", 1, 0, "NavStack");
        qmlRegisterType<VM_nav_stack_level_model>("mdl.browser.navigation",
                                                  1, 0, "NavStackLevelModel");
        qmlRegisterType<VM_nav_stack_level_proxy_model>("mdl.browser.navigation",
                                                        1, 0, "NavStackLevelProxyModel");
        qmlRegisterType<VM_nav_package>("mdl.browser.navigation", 1, 0, "NavPackage");
        qmlRegisterType<VM_sel_model>("mdl.browser.selection", 1, 0, "SelModel");
        qmlRegisterType<VM_sel_proxy_model>("mdl.browser.selection", 1, 0, "SelProxyModel");
        qmlRegisterType<VM_sel_element>("mdl.browser.selection", 1, 0, "SelElement");
        qmlRegisterType<Mdl_browser_settings>("mdl.browser.selection", 1, 0, "MdlBrowserSettings");

        QQmlApplicationEngine engine;

        // attach c++ back-end
        engine.rootContext()->setContextProperty("view_model", &modelData);

        // add components
        engine.addImageProvider(QLatin1String("mdl_archive"), new Mdl_archive_image_provider(&mdl));

        // load main window
        engine.load(
            QUrl::fromLocalFile(context.application_folder + QString("/resources/Main.qml")));

        // Run the Application
        context.state = STATE_OPEN;
        if (app.exec() != 0)
        {
            std::cerr << "[open_mdl_browser_dialog] Qt application crashed.\n";
            context.state = STATE_ERROR;
            return false;
        }
    }

    // copy the result into the context object
    std::string result = modelData.get_result().toUtf8().constData();
    if (result.length() > 0)
    {
        char* copy = new char[result.length() + 1];

        #pragma warning( disable : 4996 )
        strncpy(copy, result.c_str(), result.length() + 1);
        #pragma warning( default : 4996 )

        context.result = copy;

        context.state = STATE_CONFIRMED;
        return true;
    }

    context.state = STATE_ABORTED;
    return false;
}



int main(int argc, char* argv[])
{
    Mdl_browser_command_line_options options(argc, argv);

    QQuickStyle::setStyle("Material");
    std::string executable_dir = Platform_helper::get_executable_directory();

    while (true)
    {
        // ----------------------------------------------------------------------------------------
        // Setup Context
        // ----------------------------------------------------------------------------------------
        Mdl_browser_context context;
        context.application_folder = executable_dir.c_str();

        // ----------------------------------------------------------------------------------------
        // Open Dialog
        // ----------------------------------------------------------------------------------------
        open_mdl_browser_dialog(context, options);

        // ----------------------------------------------------------------------------------------
        // Get Result
        // ----------------------------------------------------------------------------------------

        std::cerr << "\n-----------------------------------------------------------------------\n";
        switch (context.state)
        {
            case STATE_NOT_OPENED_YET:
            case STATE_OPEN:
            case STATE_ERROR:
                std::cerr << "[main] browsing failed because of error or invalid usage.\n";
                break;

            case STATE_ABORTED:
                std::cerr << "[main] browsing was aborted by the user.\n";
                break;

            case STATE_CONFIRMED:
                std::cerr << "[main] browsing done, user picked: " << context.result << "\n";
                break;
        }
        std::cerr << "-----------------------------------------------------------------------\n\n";

        if(!options.keep_open)
        {
            Platform_helper::keep_console_open();
            return 0;
        }
    }

    return 0;
}

