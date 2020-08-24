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

// examples/mdl_core/shared/example_shared.h
//
// Code shared by all examples

#ifndef EXAMPLE_SHARED_H
#define EXAMPLE_SHARED_H

#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <vector>

#include <mi/base.h>
#include <mi/mdl/mdl.h>

#ifndef MI_PLATFORM_WINDOWS
#include <dlfcn.h>
#include <unistd.h>
#include <dirent.h>
#endif

#ifdef MI_PLATFORM_MACOSX
#include <mach-o/dyld.h>   // _NSGetExecutablePath
#endif

#ifndef MDL_SAMPLES_ROOT
#define MDL_SAMPLES_ROOT "."
#endif

typedef mi::mdl::IMDL *(IMDL_factory)(mi::base::IAllocator *);


/// Pointer to the DSO handle. Cached here for unload().
void* g_dso_handle = 0;

/// Returns the value of the given environment variable.
///
/// \param env_var   environment variable name
/// \return          the value of the environment variable or an empty string
///                  if that variable does not exist or does not have a value.
std::string get_environment(const char* env_var)
{
    std::string value;
#ifdef MI_PLATFORM_WINDOWS
    char* buf = nullptr;
    size_t sz = 0;
    if (_dupenv_s(&buf, &sz, env_var) == 0 && buf != nullptr) {
        value = buf;
        free(buf);
    }
#else
    const char* v = getenv(env_var);
    if (v)
        value = v;
#endif
    return value;
}

// Checks if the given directory exists.
//
// \param  directory path to check
// \return true, of the path points to a directory, false if not
bool dir_exists(const char* path)
{
#ifdef MI_PLATFORM_WINDOWS
    DWORD attrib = GetFileAttributesA(path);
    return (attrib != INVALID_FILE_ATTRIBUTES) && (attrib & FILE_ATTRIBUTE_DIRECTORY);
#else
    DIR* dir = opendir(path);
    if (dir == nullptr)
        return false;

    closedir(dir);
    return true;
#endif
}

/// Returns a string pointing to the directory relative to which the MDL-Core examples
/// expect their resources, e. g. materials or textures.
std::string get_samples_root()
{
    std::string samples_root = get_environment("MDL_SAMPLES_ROOT");
    if (samples_root.empty()) {
        samples_root = MDL_SAMPLES_ROOT;
    }
    if (dir_exists(samples_root.c_str()))
        return samples_root;

    return ".";
}

/// Returns a string pointing to the MDL search root for the MDL-Core examples
std::string get_samples_mdl_root()
{
    return get_samples_root() + "/mdl";
}

/// Ensures that the console with the log messages does not close immediately. On Windows, the user
/// is asked to press enter. On other platforms, nothing is done as the examples are most likely
/// started from the console anyway.
void keep_console_open() {
#ifdef MI_PLATFORM_WINDOWS
    if (IsDebuggerPresent()) {
        fprintf(stderr, "Press enter to continue . . . \n");
        fgetc(stdin);
    }
#endif // MI_PLATFORM_WINDOWS
}

/// Helper macro. Checks whether the expression is true and if not prints a message and exits.
#define check_success(expr) \
    do { \
        if (!(expr)) { \
            fprintf(stderr, "Error in file %s, line %u: \"%s\".\n", __FILE__, __LINE__, #expr); \
            keep_console_open(); \
            exit(EXIT_FAILURE); \
        } \
    } while (false)

// printf() format specifier for arguments of type LPTSTR (Windows only).
#ifdef MI_PLATFORM_WINDOWS
#ifdef UNICODE
#define FMT_LPTSTR "%ls"
#else // UNICODE
#define FMT_LPTSTR "%s"
#endif // UNICODE
#endif // MI_PLATFORM_WINDOWS


/// Loads the MDL compiler library and calls the MDL factory function.
///
/// This convenience function loads the mdl_core DSO, locates and calls the #mi_mdl_factory()
/// function. It returns an instance of the main #mi::mdl::IMDL interface.
/// The function may be called only once.
///
/// \param filename    The file name of the DSO. It is feasible to pass \c nullptr, which uses a
///                    built-in default value.
/// \return            A pointer to an instance of the main #mi::mdl::IMDL interface
mi::mdl::IMDL* load_mdl_compiler(const char* filename = 0)
{
    if (!filename)
        filename = "libmdl_core" MI_BASE_DLL_FILE_EXT;
#ifdef MI_PLATFORM_WINDOWS
    void* handle = LoadLibraryA((LPSTR) filename);
    if (!handle) {
        LPTSTR buffer = 0;
        LPCTSTR message = TEXT("unknown failure");
        DWORD error_code = GetLastError();
        if (FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS, 0, error_code,
            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR) &buffer, 0, 0))
            message = buffer;
        fprintf(stderr, "Failed to load library (%u): " FMT_LPTSTR, error_code, message);
        if (buffer)
            LocalFree(buffer);
        return nullptr;
    }
    void* symbol = GetProcAddress((HMODULE) handle, "mi_mdl_factory");
    if (!symbol) {
        LPTSTR buffer = 0;
        LPCTSTR message = TEXT("unknown failure");
        DWORD error_code = GetLastError();
        if (FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS, 0, error_code,
            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR) &buffer, 0, 0))
            message = buffer;
        fprintf(stderr, "GetProcAddress error (%u): " FMT_LPTSTR, error_code, message);
        if (buffer)
            LocalFree(buffer);
        return nullptr;
    }
#else // MI_PLATFORM_WINDOWS
    void* handle = dlopen(filename, RTLD_LAZY);
    if (!handle) {
        fprintf(stderr, "%s\n", dlerror());
        return nullptr;
    }
    void* symbol = dlsym(handle, "mi_mdl_factory");
    if (!symbol) {
        fprintf(stderr, "%s\n", dlerror());
        return nullptr;
    }
#endif // MI_PLATFORM_WINDOWS
    g_dso_handle = handle;

    return ((IMDL_factory *)(symbol))(nullptr);
}

/// Unloads the mdl_core lib.
bool unload()
{
#ifdef MI_PLATFORM_WINDOWS
    int result = FreeLibrary((HMODULE)g_dso_handle);
    if (result == 0) {
        LPTSTR buffer = 0;
        LPCTSTR message = TEXT("unknown failure");
        DWORD error_code = GetLastError();
        if (FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS, 0, error_code,
            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR) &buffer, 0, 0))
            message = buffer;
        fprintf(stderr, "Failed to unload library (%u): " FMT_LPTSTR, error_code, message);
        if (buffer)
            LocalFree(buffer);
        return false;
    }
    return true;
#else
    int result = dlclose(g_dso_handle);
    if (result != 0) {
        printf("%s\n", dlerror());
        return false;
    }
    return true;
#endif
}

/// Sleep the indicated number of seconds.
void sleep_seconds(mi::Float32 seconds)
{
#ifdef MI_PLATFORM_WINDOWS
    Sleep(static_cast<DWORD>(seconds * 1000));
#else
    usleep(static_cast<useconds_t>(seconds * 1000000));
#endif
}

// Map snprintf to _snprintf on Windows.
#ifdef MI_PLATFORM_WINDOWS
#define snprintf _snprintf
#endif

/// Returns the folder path of the current executable.
std::string get_executable_folder()
{
#ifdef MI_PLATFORM_WINDOWS
    char path[MAX_PATH];
    if (!GetModuleFileNameA(nullptr, path, MAX_PATH))
        return "";

    const char sep = '\\';
#else  // MI_PLATFORM_WINDOWS
    char path[4096];

#ifdef MI_PLATFORM_MACOSX
    uint32_t buflen(sizeof(path));
    if (_NSGetExecutablePath(path, &buflen) != 0)
        return "";
#else  // MI_PLATFORM_MACOSX
    char proc_path[64];
#ifdef __FreeBSD__
    snprintf(proc_path, sizeof(proc_path), "/proc/%d/file", getpid());
#else
    snprintf(proc_path, sizeof(proc_path), "/proc/%d/exe", getpid());
#endif

    ssize_t written = readlink(proc_path, path, sizeof(path));
    if (written < 0 || size_t(written) >= sizeof(path))
        return "";
    path[written] = 0;  // add terminating null
#endif // MI_PLATFORM_MACOSX

    const char sep = '/';
#endif // MI_PLATFORM_WINDOWS

    char *last_sep = strrchr(path, sep);
    if (last_sep == nullptr) return "";
    return std::string(path, last_sep);
}


/// An implementation of the MDL search path interface.
class MDL_search_path : public mi::base::Interface_implement<mi::mdl::IMDL_search_path>
{
public:
    /// Get the number of search paths.
    ///
    /// \param set  the path set
    virtual size_t get_search_path_count(Path_set set) const final
    {
        // supports only the compliant path
        return set == mi::mdl::IMDL_search_path::MDL_SEARCH_PATH ? m_roots.size() : 0;
    }

    /// Get the i'th search path.
    ///
    /// \param set  the path set
    /// \param i    index of the path
    virtual char const *get_search_path(Path_set set, size_t i) const final
    {
        // supports only the compliant path
        if (set != mi::mdl::IMDL_search_path::MDL_SEARCH_PATH)
            return nullptr;

        return m_roots[i].c_str();
    }

    /// Add a search path.
    void add_path(const char *path)
    {
        m_roots.push_back(path);
    }

private:
    /// The type of vectors of strings.
    typedef std::vector<std::string> String_vector;

    /// The search path roots.
    String_vector m_roots;
};


/// A helper class for handling MDL compiler messages.
class Msg_context
{
public:
    /// Constructor.
    ///
    /// \param mdl_compiler  the MDL compiler interface
    Msg_context(mi::mdl::IMDL *mdl_compiler)
    : m_out(mdl_compiler->create_std_stream(mi::mdl::IMDL::OS_STDOUT))
    , m_printer(mi::base::make_handle(mdl_compiler->create_printer(m_out.get())))
    , m_ctx(mdl_compiler->create_thread_context())
    {
        m_printer->enable_color(true);
    }

    /// Get the associated thread context.
    operator mi::mdl::IThread_context *() const
    {
        return m_ctx.get();
    }

    /// Get the compiler messages of the associated thread context.
    mi::mdl::Messages const &access_messages() const
    {
        return m_ctx->access_messages();
    }

    /// Print current compiler messages to stdout.
    void print_messages()
    {
        mi::mdl::Messages const &msgs = m_ctx->access_messages();
        for (size_t i = 0, n = msgs.get_message_count(); i < n; ++i) {
            m_printer->print(msgs.get_message(i));
        }
    }

    /// Get the associated printer.
    mi::base::Handle<mi::mdl::IPrinter> get_printer()
    {
        return m_printer;
    }

private:
    mi::base::Handle<mi::mdl::IOutput_stream>  m_out;
    mi::base::Handle<mi::mdl::IPrinter>        m_printer;
    mi::base::Handle<mi::mdl::IThread_context> m_ctx;
};


/// A module manager implementing a call name resolver and a module cache for the MDL compiler.
class Module_manager : public mi::mdl::ICall_name_resolver, public mi::mdl::IModule_cache
{
    typedef std::unordered_map<std::string, mi::base::Handle<mi::mdl::IModule const> > Module_map;
    typedef std::unordered_map<std::string, mi::base::Handle<mi::mdl::IGenerated_code_dag const> > DAG_map;
public:
    /// Constructor.
    ///
    /// \param mdl_compiler  the MDL compiler interface
    Module_manager(mi::mdl::IMDL *mdl_compiler)
      : m_dag_be(mi::base::make_handle(mdl_compiler->load_code_generator("dag"))
        .get_interface<mi::mdl::ICode_generator_dag>())
    {
    }

    /// Adds a module and all its imports to the module manager.
    ///
    /// \param module  the MDL module
    void add_module(mi::mdl::IModule const *module)
    {
        if (module->is_builtins()) {
            m_builtins_module = mi::base::make_handle_dup(module);
        }

        Module_map::mapped_type &entry = m_modules[module->get_name()];
        bool not_found = !entry;

        if (not_found) {
            // set/update module
            entry = mi::base::make_handle_dup(module);

            // if the module was not known before, add all imported modules recursively
            for (int i = 0, n = module->get_import_count(); i < n; ++i) {
                mi::base::Handle<mi::mdl::IModule const> imported_module(module->get_import(i));
                add_module(imported_module.get());
            }
            
            mi::base::Handle<mi::mdl::IGenerated_code_dag const> dag(m_dag_be->compile(module));
            m_dags[module->get_name()] = dag;

            for (size_t i = 0, n = dag->get_material_count(); i < n; ++i) {
                const char* material_name = dag->get_material_name(i);
                m_entities[material_name] = make_handle_dup(module);
            }

            for (size_t i = 0, n = dag->get_function_count(); i < n; ++i) {
                const char* function_name = dag->get_function_name(i);
                m_entities[function_name] = make_handle_dup(module);
            }
        }
    }

    /// Returns the DAG for a previously added module.
    mi::mdl::IGenerated_code_dag const *get_dag(mi::mdl::IModule const *module)
    {
        auto it = m_dags.find(module->get_name());
        if (it == m_dags.end())
            return nullptr;

        it->second->retain();
        return it->second.get();
    }
    
    /// Find the owner module of a given entity name.
    /// If the entity name does not contain a colon, you should return the builtins module,
    /// which you can identify by IModule::is_builtins().
    ///
    /// \param entity_name    the entity name
    ///
    /// \returns the owning module of this entity if found, nullptr otherwise
    mi::mdl::IModule const *get_owner_module(char const *entity_name) const final
    {
        auto it = m_entities.find(entity_name);
        if (it == m_entities.end()) {
            if (strchr(entity_name, ':'))
                return nullptr;
            m_builtins_module->retain();
            return m_builtins_module.get();
        }

        it->second->retain();
        return it->second.get();
    }

    /// Find the owner code DAG of a given entity name.
    /// If the entity name does not contain a colon, you should return the builtins DAG,
    /// which you can identify by calling its owner module's IModule::is_builtins().
    ///
    /// \param entity_name    the entity name
    ///
    /// \returns the owning module of this entity if found, NULL otherwise
    mi::mdl::IGenerated_code_dag const *get_owner_dag(char const *entity_name) const final
    {
        return nullptr;
    }

    /// Create an \c IModule_cache_lookup_handle for this \c IModule_cache implementation.
    /// Has to be freed using \c free_lookup_handle.
    mi::mdl::IModule_cache_lookup_handle* create_lookup_handle() const final
    {
        return nullptr;
    }

    /// Free a handle created by \c create_lookup_handle.
    /// \param handle       a handle created by this module cache.
    void free_lookup_handle(mi::mdl::IModule_cache_lookup_handle* handle) const final
    {
    }

    /// Lookup a module.
    ///
    /// \param absname  the absolute name of an MDL module as returned by the module resolver
    ///
    /// \return  If this module is already known, return it, otherwise nullptr.
    mi::mdl::IModule const *lookup(
        char const *absname,
        mi::mdl::IModule_cache_lookup_handle * /*handle*/) const final
    {
        // search for the module name
        auto it = m_modules.find(absname);
        if (it != m_modules.end()) {
            it->second->retain();
            return it->second.get();
        }

        return nullptr;
    }

    /// Get the module loading callback.
    mi::mdl::IModule_loaded_callback *get_module_loading_callback() const final
    {
        return nullptr;
    }

private:
    mi::base::Handle<mi::mdl::ICode_generator_dag> m_dag_be;
    mi::base::Handle<mi::mdl::IModule const> m_builtins_module;
    Module_map m_modules;
    DAG_map m_dags;
    Module_map m_entities;
};


/// Helper class to handle array passing safely.
template<typename T>
class Array_ref {
public:
    typedef const T *iterator;
    typedef const T *const_iterator;

    /// Construct an empty array reference.
    /*implicit*/ Array_ref()
        : m_arr(nullptr)
        , m_n(0)
    {
    }

    /// Construct from a pointer and a length.
    ///
    /// \param arr  the array
    /// \param n    length of the array
    Array_ref(T const *arr, size_t n)
        : m_arr(arr)
        , m_n(n)
    {
    }

    /// Construct from a start and an end pointer.
    ///
    /// \param begin  the start pointer
    /// \param end    the end pointer
    Array_ref(T const *begin, T const *end)
        : m_arr(begin)
        , m_n(end - begin)
    {
    }

    /// Construct from one element.
    ///
    /// \param elem  the element
    /*implicit*/ Array_ref(T const &elem)
        : m_arr(&elem)
        , m_n(1)
    {
    }

    /// Construct a descriptor from a C array.
    template <size_t N>
    /*implicit*/ Array_ref(T const (&arr)[N])
        : m_arr(arr), m_n(N)
    {
    }

    /// Construct a descriptor from a C array.
    /*implicit*/ Array_ref(std::initializer_list<T> arr)
        : m_arr(arr.begin()), m_n(arr.size())
    {
    }

    /// Construct from an vector.
    template <typename A>
    /*implicit*/ Array_ref(std::vector<T, A> const &v)
        : m_arr(v.empty() ? nullptr : &v[0]), m_n(v.size())
    {
    }

    /// Get the begin iterator.
    iterator begin() const { return m_arr; }

    /// Get the end iterator.
    iterator end() const { return m_arr + m_n; }

    /// Get the array size.
    size_t size() const { return m_n; }

    /// Index operator.
    T const &operator[](size_t i) const
    {
        return m_arr[i];
    }

    /// Chop off the first N elements of the array.
    Array_ref<T> slice(size_t n) const
    {
        return Array_ref<T>(&m_arr[n], m_n - n);
    }

    /// slice(n, m) - Chop off the first N elements of the array, and keep M
    /// elements in the array.
    Array_ref<T> slice(size_t N, size_t M) const
    {
        return Array_ref<T>(data() + N, M);
    }

    /// Get the data.
    T const *data() const { return m_arr; }

private:
    T const *m_arr;
    size_t  m_n;
};


/// Helper class grouping an IMaterial_instance together with its DAG and material index.
class Material_instance
{
    using Call_argument = mi::mdl::DAG_call::Call_argument;

public:
    /// Construct an invalid material instance.
    Material_instance()
        : m_material_index(0)
    {}

    /// Constructor.
    Material_instance(
        mi::mdl::IGenerated_code_dag const *dag,
        size_t material_index,
        mi::mdl::IGenerated_code_dag::IMaterial_instance *instance)
    : m_dag(dag, mi::base::DUP_INTERFACE)
    , m_material_index(material_index)
    , m_inst(instance, mi::base::DUP_INTERFACE)
    {}

    /// Provides transparent access to the material instance mdl_core object.
    /// The reference count is not changed by this function.
    mi::mdl::IGenerated_code_dag::IMaterial_instance *operator->() const
    {
        return m_inst.get();
    }

    /// Returns true, if the material instance is valid.
    operator bool() const
    {
        return m_inst.is_valid_interface();
    }

    /// Get the name of the material.
    char const *get_dag_material_name() const
    {
        return m_dag->get_material_name(m_material_index);
    }

    /// Get the number of annotations of the material.
    ///
    /// \returns The number of annotations.
    size_t get_dag_annotation_count() const
    {
        return m_dag->get_material_annotation_count(m_material_index);
    }

    /// Get the annotation at annotation_index of the material.
    ///
    /// \param annotation_index  The index of the annotation.
    ///
    /// \returns The annotation.
    mi::mdl::DAG_node const *get_dag_annotation(int annotation_index) const
    {
        return m_dag->get_material_annotation(m_material_index, annotation_index);
    }

    /// Get the parameter count of the material in the DAG.
    ///
    /// \returns The number of parameters of the material.
    size_t get_dag_parameter_count() const
    {
        return m_dag->get_material_parameter_count(m_material_index);
    }

    /// Get the type of a parameter of the material in the DAG.
    ///
    /// \param parameter_index  The index of the parameter.
    ///
    /// \returns The type of the parameter.
    mi::mdl::IType const *get_dag_parameter_type(size_t parameter_index) const
    {
        return m_dag->get_material_parameter_type(m_material_index, parameter_index);
    }

    /// Get the name of a parameter of the material in the DAG.
    ///
    /// \param parameter_index  The index of the parameter.
    ///
    /// \returns The name of the parameter.
    char const *get_dag_parameter_name(size_t parameter_index) const
    {
        return m_dag->get_material_parameter_name(m_material_index, parameter_index);
    }

    /// Get the index of the material parameter with the given name in the DAG.
    ///
    /// \param param_name  The name of the DAG parameter
    ///
    /// \returns The index of the DAG parameter or -1, if it was not found
    size_t get_dag_parameter_index(char const *param_name) const
    {
        return m_dag->get_material_parameter_index(m_material_index, param_name);
    }

    /// Get the default initializer of a parameter in the DAG.
    ///
    /// \param parameter_index  The index of the parameter.
    ///
    /// \returns The type of the parameter.
    mi::mdl::DAG_node const *get_dag_parameter_default(size_t parameter_index) const
    {
        return m_dag->get_material_parameter_default(m_material_index, parameter_index);
    }

    /// Get the number of annotations of a parameter of the material in the DAG.
    ///
    /// \param parameter_index  The index of the parameter.
    ///
    /// \returns The number of annotations.
    size_t get_dag_parameter_annotation_count(size_t parameter_index) const
    {
        return m_dag->get_material_parameter_annotation_count(m_material_index, parameter_index);
    }

    /// Get an annotation of a parameter of the material in the DAG.
    ///
    /// \param parameter_index   The index of the parameter.
    /// \param annotation_index  The index of the annotation.
    ///
    /// \returns The annotation or nullptr, if the indices are invalid.
    mi::mdl::DAG_node const *get_dag_parameter_annotation(
        size_t parameter_index,
        size_t annotation_index) const
    {
        return m_dag->get_material_parameter_annotation(
            m_material_index, parameter_index, annotation_index);
    }

    /// Get the DAG containing the material of the material instance.
    mi::base::Handle<mi::mdl::IGenerated_code_dag::IMaterial_instance> get_material_instance()
        const
    {
        return m_inst;
    }

    /// Get the DAG containing the material of the material instance.
    mi::base::Handle<mi::mdl::IGenerated_code_dag const> get_dag() const {
        return m_dag;
    }

    /// Get the material index inside the containing DAG.
    size_t get_material_index() const {
        return m_material_index;
    }

    /// Get the first index of a function with the given name (with or without signature).
    ///
    /// \param module_dag  The module containing the function
    /// \param func_name   The name of the function
    ///
    /// \return The index of the found function or -1 if not found.
    static size_t find_first_function(
        mi::mdl::IGenerated_code_dag const *module_dag,
        char const                         *func_name)
    {
        size_t len = strlen(func_name);
        for (size_t i = 0, n = module_dag->get_function_count(); i < n; ++i) {
            char const *cur_func_name = module_dag->get_function_name(i);
            if (strncmp(cur_func_name, func_name, len) == 0 &&
                    (cur_func_name[len] == '(' || cur_func_name[len] == 0))
                return i;
        }
        return ~0;
    }

    /// Create a DAG call with the given name (with or without signature) and the list of arguments.
    /// Note: currently the arguments are not type-checked.
    ///
    /// \param module_dag     the DAG of the module containing the function
    /// \param function_name  the name of the function to call
    /// \param args           the list of named arguments, arguments for parameters with default
    ///                       values may be omitted
    ///
    /// \returns The created node which may already be optimized to something else or nullptr
    ///     on error (unknown function name, missing or too many arguments, creation failed)
    mi::mdl::DAG_node const *create_call(
        mi::mdl::IGenerated_code_dag const *module_dag,
        char const                         *function_name,
        Array_ref<Call_argument> const     &args)
    {
        size_t func_index = find_first_function(module_dag, function_name);
        if (func_index == ~0)
            return nullptr;

        // Create list of call arguments: all arguments and in the right order
        std::vector<Call_argument> real_args;
        size_t used_args = 0;
        size_t num_args = args.size();
        for (size_t i = 0, n = module_dag->get_function_parameter_count(func_index); i < n; ++i) {
            char const *param_name = module_dag->get_function_parameter_name(func_index, i);

            // First check, whether the argument already exists at the right position
            if (i < num_args && strcmp(param_name, args[i].param_name) == 0) {
                real_args.push_back(args[i]);
                ++used_args;
                continue;
            }

            // If not, search for it
            size_t j = 0;
            for (; j < num_args; ++j) {
                if (j == i) continue;  // already checked
                if (strcmp(param_name, args[j].param_name) == 0) {
                    real_args.push_back(args[j]);
                    ++used_args;
                    break;
                }
            }
            if (j < num_args) continue;  // found it

            // Not found, use default parameter
            mi::mdl::DAG_node const *def_param =
                module_dag->get_function_parameter_default(func_index, i);
            if (def_param == nullptr)
                return nullptr;  // no default for a missing parameter -> fail

            real_args.push_back(Call_argument(def_param, param_name));
        }

        // Not all arguments were used?
        if (used_args != args.size())
            return nullptr;

        // Check types of call arguments.
        // Note: In this example we do not check for conformance of the arguments regarding
        //    uniform/varying properties.
        for (size_t i = 0, n = real_args.size(); i < n; ++i) {
            mi::mdl::IType const *param_type =
                module_dag->get_function_parameter_type(func_index, int(i))->skip_type_alias();
            mi::mdl::IType const *arg_type = real_args[i].arg->get_type()->skip_type_alias();
            if (param_type != arg_type)
                return nullptr;
        }

        return m_inst->create_call(
            function_name,
            module_dag->get_function_semantics(func_index),
            real_args.data(),
            int(real_args.size()),
            module_dag->get_function_return_type(func_index));
    }

private:
    mi::base::Handle<mi::mdl::IGenerated_code_dag const> m_dag;
    size_t m_material_index;
    mi::base::Handle<mi::mdl::IGenerated_code_dag::IMaterial_instance> m_inst;
};

/// Base class for compiling materials into specific targets.
class Material_compiler
{
    typedef mi::mdl::DAG_call::Call_argument Call_argument;

public:
    /// Constructor.
    ///
    /// \param mdl_compiler  the MDL compiler interface
    Material_compiler(mi::mdl::IMDL *mdl_compiler)
    : m_mdl_compiler(mi::base::make_handle_dup(mdl_compiler))
    , m_dag_be(mi::base::make_handle(mdl_compiler->load_code_generator("dag"))
        .get_interface<mi::mdl::ICode_generator_dag>())
    , m_search_path(new MDL_search_path())
    , m_msg_context(mdl_compiler)
    , m_module_manager(mdl_compiler)
    {
        // increment reference counter, as the MDL compiler will take ownership (and not increment it),
        // but we will keep a reference
        m_search_path->retain();

        m_mdl_compiler->install_search_path(m_search_path.get());
    }

    /// Get the index of a material in a given module DAG.
    ///
    /// \param module_dag     the DAG of a MDL module
    /// \param material_name  the name of the requested material
    ///
    /// \return the index of the material or -1 if the requested material was not found
    static size_t get_material_index(
        mi::mdl::IGenerated_code_dag const *module_dag,
        char const                         *material_name)
    {
        for (size_t i = 0, n = module_dag->get_material_count(); i < n; ++i) {
            if (strcmp(module_dag->get_material_name(i), material_name) == 0)
                return i;
        }
        return ~0;
    }

    /// Helper function to extract the module name from a fully-qualified material name.
    ///
    /// \param material_name  a fully qualified MDL material name
    static std::string get_module_name(std::string const &material_name)
    {
        std::size_t p = material_name.rfind("::");
        return material_name.substr(0, p);
    }

    /// Helper function to ensure the material name is prefixed with "::".
    static std::string get_full_material_name(std::string const &material_name)
    {
        if (material_name.size() < 2 || material_name[0] != ':' || material_name[1] != ':')
            return "::" + material_name;
        return material_name;
    }

    /// Add a module search path used to find MDL modules.
    ///
    /// \param path  an MDL search path to be added
    void add_module_path(char const *path)
    {
        m_search_path->add_path(path);
    }

    /// Get the module manager.
    Module_manager &get_module_manager()
    {
        return m_module_manager;
    }

    /// Access the compiler messages.
    mi::mdl::Messages const &access_messages() const
    {
        return m_msg_context.access_messages();
    }

    /// Check whether any errors occurred during compilation.
    bool has_errors() const
    {
        return access_messages().get_error_message_count() > 0;
    }

    /// Print all available compiler messages to the console.
    void print_messages()
    {
        m_msg_context.print_messages();
    }

    /// Get the stdout printer.
    mi::base::Handle<mi::mdl::IPrinter> get_printer()
    {
        return mi::base::Handle<mi::mdl::IPrinter>(m_msg_context.get_printer());
    }

    /// Get a DAG node along the given path.
    ///
    /// \param node         the root node for the path
    /// \param path         the list of path elements to walk along
    /// \param dag_builder  the owner of any new constants generated (in case sub-elements of
    ///                     constants are accessed)
    ///
    /// \returns the requested DAG node or nullptr if not found
    mi::mdl::DAG_node const *get_dag_arg(
        mi::mdl::DAG_node const       *node,
        Array_ref<char const *> const &path,
        mi::mdl::IDag_builder         *dag_builder)
    {
        if (node == nullptr)
            return node;
        if (path.size() == 0)
            return node;

        switch (node->get_kind()) {
        case mi::mdl::DAG_node::EK_CONSTANT:
            {
                mi::mdl::DAG_constant const *c = mi::mdl::as<mi::mdl::DAG_constant>(node);
                if (auto const *vc = mi::mdl::as<mi::mdl::IValue_compound>(c->get_value())) {
                    if (mi::mdl::IValue const *subval = vc->get_value(path[0])) {
                        node = dag_builder->create_constant(subval);
                        return get_dag_arg(node, path.slice(1), dag_builder);
                    }
                }
                return nullptr;
            }
        case mi::mdl::DAG_node::EK_TEMPORARY:
            {
                mi::mdl::DAG_temporary const *temp = mi::mdl::as<mi::mdl::DAG_temporary>(node);
                return get_dag_arg(temp->get_expr(), path, dag_builder);
            }
        case mi::mdl::DAG_node::EK_CALL:
            {
                mi::mdl::DAG_call const *call = static_cast<mi::mdl::DAG_call const *>(node);
                node = call->get_argument(path[0]);
                return get_dag_arg(node, path.slice(1), dag_builder);
            }
        case mi::mdl::DAG_node::EK_PARAMETER:
            return nullptr;
        }
        check_success(!"unknown DAG node kind");
        return nullptr;
    }

    /// Compile a module to a DAG.
    ///
    /// \param module_name  a fully qualified MDL module name.
    mi::base::Handle<mi::mdl::IGenerated_code_dag const> compile_module(
        char const *module_name)
    {
        // Load the MDL module
        mi::base::Handle<mi::mdl::IModule const> module(
            m_mdl_compiler->load_module(m_msg_context, module_name, &m_module_manager));
        if (!module || !module->is_valid()) {
            m_msg_context.print_messages();
            check_success(!"Loading module failed");
        }
        m_module_manager.add_module(module.get());

        return make_handle(m_module_manager.get_dag(module.get()));
    }

    /// Return the list of all material names in the given MDL module.
    std::vector<std::string> get_material_names(const std::string& module_name)
    {
        mi::base::Handle<mi::mdl::IGenerated_code_dag const> code_dag(
            compile_module(module_name.c_str()));

        size_t num_materials = code_dag->get_material_count();
        std::vector<std::string> material_names(num_materials);
        for (size_t i = 0; i < num_materials; ++i) {
            material_names[i] = code_dag->get_material_name(i);
        }
        return material_names;
    }

    /// Creates an instance of the given material.
    ///
    /// \param code_dag       the DAG of a MDL module
    /// \param material_name  the name of the material to instantiate
    Material_instance create_material_instance(
        mi::mdl::IGenerated_code_dag const *code_dag,
        std::string const                  &material_name)
    {
        size_t mat_index = get_material_index(code_dag, get_full_material_name(material_name).c_str());
        if (mat_index == ~0)
            return Material_instance();

        mi::base::Handle<mi::mdl::IGenerated_code_dag::IMaterial_instance> mat_instance(
            code_dag->create_material_instance(mat_index));
        if (!mat_instance)
            return Material_instance();

        return Material_instance(code_dag, mat_index, mat_instance.get());
    }

    /// Creates an instance of the given material.
    ///
    /// \param material_name  a fully qualified MDL material name
    Material_instance create_material_instance(
        std::string const &material_name)
    {
        // Load the MDL module
        std::string module_name = get_module_name(material_name);
        mi::base::Handle<mi::mdl::IGenerated_code_dag const> code_dag(compile_module(module_name.c_str()));

        return create_material_instance(code_dag.get(), material_name);
    }

    /// Create a DAG call with the given name (with or without signature) and the list of arguments.
    ///
    /// Any missing arguments will be added with default values, if provided by the function.
    /// Note: currently the arguments are not type-checked.
    ///
    /// \param mat_inst           the material instance
    /// \param args               material instance arguments
    /// \param class_compilation  if true, create a material instance in class compilation mode
    /// \param use_zero_defaults  if true, set open parameters to zero
    mi::mdl::IGenerated_code_dag::Error_code initialize_material_instance(
        Material_instance              &mat_inst,
        Array_ref<Call_argument> const &args,
        bool                           class_compilation,
        bool                           use_zero_defaults = false)
    {
        std::vector<mi::mdl::DAG_node const *> real_args;
        size_t used_args = 0;
        size_t num_args = args.size();
        for (size_t i = 0, n = mat_inst.get_dag_parameter_count(); i < n; ++i) {
            char const *param_name = mat_inst.get_dag_parameter_name(i);

            // First check, whether the argument already exists at the right position
            if (i < num_args && strcmp(param_name, args[i].param_name) == 0) {
                real_args.push_back(args[i].arg);
                ++used_args;
                continue;
            }

            // If not, search for it
            size_t j = 0;
            for (; j < num_args; ++j) {
                if (j == i) continue;  // already checked
                if (strcmp(param_name, args[j].param_name) == 0) {
                    real_args.push_back(args[j].arg);
                    ++used_args;
                    break;
                }
            }
            if (j < num_args)
                continue;  // found it

            // Not found, use default parameter
            mi::mdl::DAG_node const *def_param = mat_inst.get_dag_parameter_default(i);
            if (def_param == nullptr) {
                if (!use_zero_defaults)
                    return mi::mdl::IGenerated_code_dag::EC_TOO_FEW_ARGUMENTS;
                def_param = mat_inst->create_constant(
                    mat_inst->get_value_factory()->create_zero(
                        mat_inst.get_dag_parameter_type(i)));
            }

            real_args.push_back(def_param);
        }

        // Not all arguments were used?
        if (used_args != args.size())
            return mi::mdl::IGenerated_code_dag::EC_TOO_MANY_ARGUMENTS;

        return mat_inst->initialize(
            &m_module_manager,
            /*resource_modifier=*/ nullptr,
            mat_inst.get_dag().get(),
            int(real_args.size()),
            real_args.data(),
            /*use_temporaries=*/ false,
            class_compilation
                ? mi::mdl::IGenerated_code_dag::IMaterial_instance::DEFAULT_CLASS_COMPILATION
                : mi::mdl::IGenerated_code_dag::IMaterial_instance::INSTANCE_COMPILATION,
            /*evaluator=*/ nullptr,
            /*fold_meters_per_scene_unit=*/ true,
            /*mdl_meters_per_scene_unit=*/ 1.0f,
            /*wavelength_min=*/ 380.0f,
            /*wavelength_max=*/ 780.0f,
            /*fold_params=*/ nullptr,
            /*num_fold_params=*/ 0);
    }

protected:
    mi::base::Handle<mi::mdl::IMDL>                m_mdl_compiler;
    mi::base::Handle<mi::mdl::ICode_generator_dag> m_dag_be;
    mi::base::Handle<MDL_search_path>              m_search_path;
    Msg_context                                    m_msg_context;
    Module_manager                                 m_module_manager;
};

#ifdef MI_PLATFORM_WINDOWS

#define MAIN_UTF8 main_utf8

#define COMMANDLINE_TO_UTF8 \
int wmain(int argc, wchar_t* argv[]) { \
    char** argv_utf8 = new char*[argc]; \
    for (int i = 0; i < argc; i++) { \
        LPWSTR warg = argv[i]; \
        DWORD size = WideCharToMultiByte(CP_UTF8, 0, warg, -1, NULL, 0, NULL, NULL); \
        check_success(size > 0); \
        argv_utf8[i] = new char[size]; \
        DWORD result = WideCharToMultiByte(CP_UTF8, 0, warg, -1, argv_utf8[i], size, NULL, NULL); \
        check_success(result > 0); \
    } \
    SetConsoleOutputCP(CP_UTF8); \
    int result = main_utf8(argc, argv_utf8); \
    delete[] argv_utf8; \
    return result; \
}

#else

#define MAIN_UTF8 main
#define COMMANDLINE_TO_UTF8

#endif

#endif // MI_EXAMPLE_SHARED_H
