#include "example_shared.h"

// the helper classes currently require the following global variables which
// have to be in one link unit. So either linking the example shared library
// or adding this .cpp file to the build is required.

namespace mi { namespace examples { namespace mdl {

mi::base::Handle<mi::base::ILogger> g_logger;

// required for loading and unloading the SDK
#ifdef MI_PLATFORM_WINDOWS
    HMODULE g_dso_handle = 0;
#else
    void* g_dso_handle = 0;
#endif

}}}
