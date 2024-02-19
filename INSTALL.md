# Building the MDL SDK from Source

MDL uses [CMake](http://www.cmake.org/) to generate build files for a
particular development environment. It is required to use CMake 3.12 or later
(3.21 or later on Windows), which can be downloaded from the
[CMake Website](https://cmake.org/download/). When using a Unix-like system,
you can install the *cmake* package using the respective package management
systems. On the Mac OS X platform, third party dependencies can be resolved
using the [Homebrew Package Manager](https://brew.sh/index_de).


## Dependencies

The source code requires a C++17 compiler and several third-party libraries
and tools to build the MDL SDK. Additional third-party libraries are used in
the examples.

The build with the following 64-bit platform-compiler combinations has been
successfully tested:

-   **Windows 10:**            Microsoft Visual Studio 2022 (msvc v143)
-   **CentOS 7, Debian 12:**   GCC 12 (on x86-64 or aarch64)
-   **Mac OS X 12.6:**         Xcode 14.2 (Apple Clang 14.0.0, on x86-64 or aarch64)

The versions listed with the following dependencies have been
successfully tested. Where not mentioned otherwise, other versions
might work as well.

<a name="thirdparty-dependencies-libs"></a>
The following third-party libraries and tools are required to build the MDL SDK:

-   <a name="vcpkg">**vcpkg**</a> *(git commit ID fe1e9f5)*  
    [Vcpkg](https://vcpkg.io/en/getting-started.html) is the recommended way to
    install other dependencies like Boost, OpenImageIO, GLEW, and GLFW. The
    vcpkg version mentioned above corresponds to the versions mentioned for
    these dependencies below.  
    Windows: It is strongly recommended to select the same toolset that is used
    later to build the MDL SDK, e.g., by adding
    *set(VCPKG_PLATFORM_TOOLSET v143)* (or similar) to
    *triplets/x64-windows-static.cmake*. See the corresponding section in the
    [vcpkg documentation](https://learn.microsoft.com/en-us/vcpkg/users/triplets#windows-specific-variables)
    for further details. Add the vcpkg option *--triplet=x64-windows-static* to
    the *install* command. There is no need to run the *integrate* command.

-   **Boost** *(1.83.0)*  
    Installation via [vcpkg](#vcpkg) is strongly recommended. Install the vcpkg
    packages *boost-any* and *boost-uuid*.  

-   **OpenImageIO** *(2.4.14.0)*  
    Installation via [vcpkg](#vcpkg) is strongly recommended. Install the vcpkg
    packages *openimageio[gif,openjpeg,tools,webp]*.  

-   **Python3** *(3.8.0)*  
    Linux: Install the *python* package.  
    Windows and Max OS X: Download and install Python 3.8 from
    [python.org](https://www.python.org/downloads/).

-   **Clang** *(12.0.1)*  
    Using version 12.0.1 is mandatory.  
    Pre-compiled binaries can be found on
    [llvm.org](http://releases.llvm.org/download.html#12.0.1).

None of the remaining dependencies are strictly required. If not available, the
corresponding functionality needs to be disabled, as indicated by the error
message. See also the [CMake Options](#cmake-options).

<a name="doc-build-tools"></a>
The following tools are used to build the API reference documentation:

-   **Doxygen** *(1.9.4)*  
    See the [Doxygen project page](https://sourceforge.net/projects/doxygen/) and
    the [archive of all releases](https://sourceforge.net/projects/doxygen/files/).

-   **dot from GraphViz** *(2.40.1)*  
    The `dot` tool from GraphViz is optional: it is used to generate nicer
    inheritance diagrams. See the
    [GraphViz project page](https://www.graphviz.org/).

For generating and compiling the MDL Python Bindings, the following additional
dependencies are required:

-   **Python3 Development Package** *(3.8.0)*  
    Linux: Install the *python-dev* package in addition to *python*.  
    Windows and Max OS X: Download and install Python 3.8 from
    [python.org](https://www.python.org/downloads/).

-   **SWIG** *(4.0.2)*  
    Follow the instructions for downloading or building on
    [swig.org](http://www.swig.org/download.html).

The following third-party dependencies are used by several and/or major
examples. Installation is strongly recommended unless the corresponding group
of examples is of no interest to you.

-   **DirectX Raytracing support** *(Windows only)*  
    Building the DXR example requires Windows 10 version 1909 and the
    corresponding SDK 10.0.18362.0. Additionally the optional *Graphic Tools*
    feature has to be installed.

-   **DirectX Shader Compiler support** *(July 2022)* *(Windows only)*  
    Building the DXR example requires an updated version of the DirectX Shader
    Compiler.  
    Download and extract the pre-compiled x64 binaries from
    [github](https://github.com/microsoft/DirectXShaderCompiler/releases).

-   **GLEW** *(2.2.0)*  
    This dependency is required for all OpenGL-based examples.  
    Installation via [vcpkg](#vcpkg) is strongly recommended. Install the vcpkg
    package *glew*.  

-   **GLFW** *(3.3.8)*  
    This dependency is required for all OpenGL- and Vulkan-based examples.  
    Installation via [vcpkg](#vcpkg) is strongly recommended. Install the vcpkg
    package *glfw3*.  

-   **NVIDIA CUDA Toolkit** *(12.x)*  
    This dependency is required for all CUDA-based examples.  
    Please follow the instructions on the
    [CUDA Developer Website](https://developer.nvidia.com/cuda-toolkit).

-   **Vulkan SDK** *(1.2.198.1)*  
    This dependency is required for all Vulkan-based examples.  
    Please follow the instructions on the
    [Vulkan SDK Website](https://vulkan.lunarg.com/sdk/home).  
    For debug builds on Windows, the debug libraries are required to be installed.

The following third-party dependencies are only used by fewer or single
examples, or add additional features to other examples. Installation can be
safely skipped unless you are specifically interested in those examples or
features.

-   **Arnold SDK** *(6.2.0.1)*  
    This dependency is required to build the MDL plugin for Arnold.  
    Please follow the instructions on the
    [Arnold Website](https://www.arnoldrenderer.com/arnold/download/) to
    download the Arnold SDK.

-   **MaterialX** *(github repository, tag: v1.38.7, Windows only)*  
    This dependency adds MaterialX support to the DXR example.  
    Please download a release from
    [github](https://github.com/AcademySoftwareFoundation/MaterialX/releases).  
    The pre-built packages do not contain libs for debug.
    If those are needed, a build from source is required.

-   **NVIDIA CUDA Toolkit** *(8.0)*  
    This exact version is required by the OptiX 7 example to generate LLVM bitcode
    for the closest hit shader with Clang 12.0.1.  
    The old version is available in the
    [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive).

-   **OptiX** *(7.0 to 7.5)*  
    This dependency is required to build the OptiX 7 example.  
    Please follow the instructions on the [OptiX Website](https://developer.nvidia.com/designworks/optix/download).

-   **Qt** *(5.10.1)*  
    This dependency is required for the MDL browser example, a Qt-based browser
    for MDL modules and materials. This browser is also accessible from the DXR
    example.  
    Please follow the instructions on the [Qt Website](https://www.qt.io/).

-   **X-Rite AxF SDK** *(1.9.0)*  
    This dependency is required to build the AxF-to-MDL example.  
    Please send the "Request AxF SDK" document from the
    [X-Rite AxF Website](https://www.xrite.com/axf).


## Building on Windows

1.  Before generating the Visual Studio solution, be sure to
    download and extract or install the third-party libraries listed
    above.  The following steps assume you have extracted the pre-compiled
    binaries to a common third-party directory that is:

        C:/projects/thirdparty

2.  Open CMake-Gui, click `Browse Source...` and select the root
    directory of the MDL SDK source checkout. This directory contains
    the top-level *CMakeLists.txt*.  Pick a build directory that will
    contain the files for your build system and eventually, the compiled
    binaries.

    It is recommended that you build into a subdirectory, not into the repository root.
    *C:/projects/mdl-sdk/build/vs2022* for example is fine, assuming you cloned the repository to:

        C:/projects/mdl-sdk

3.  After clicking ``Configure``, CMake asks you to choose the Generator.
    Select `Visual Studio 17 2022` (or higher), enter `host=x64` as toolset
    and click `Finish`. CMake starts to configure the build and stops several
    times when user input is required to resolve dependencies.

4.  Optionally, you can select or deselect
    [Additional CMake Options](#additional-cmake-options) by checking and
    un-checking the boxes next to the entries that start with *MDL*. Click
    ``Configure`` again to continue.

5.  When red error messages appear in the log, identify the dependency path
    that is requested and resolve the error by specifying the corresponding
    entry in CMake-Gui. Then, click ``Configure`` again to continue. Repeat
    this step until no further errors occur.

    <a name="thirdparty-dependencies-options"></a>
    During this process, you may need to setup the following entries based on the selected components:

    -   **ARNOLD_SDK_DIR** in Ungrouped Entries,  
        for example: *C:/projects/thirdparty/Arnold-6.2.0.1-windows*

    -   **clang_PATH** in Ungrouped Entries (only if not found in the PATH),  
        for example: *C:/Program Files/LLVM-12/bin/clang.exe*

    -   **CMAKE_TOOLCHAIN_FILE** in the CMAKE group,  
        for example: *C:/projects/thirdparty/vcpkg/scripts/buildsystems/vcpkg.cmake*

    -   **CUDA8_PATH** in Ungrouped Entries,  
        for example: *C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0*  
        Version 8 is mandatory here, used for the OptiX 7 example.

    -   **DXC_DIR** in Ungrouped Entries,  
        for example: *C:/projects/thirdparty/dxc_2022_07_18*

    -   **MATERIALX_DIR** in Ungrouped Entries,  
        for example: *C:/projects/thirdparty/git/MaterialX*  
        For MaterialX support, the option **MDL_MSVC_DYNAMIC_RUNTIME_EXAMPLES** has to be enabled.

    -   **python_PATH** in Ungrouped Entries (only if not found in the PATH),  
        for example: *C:/projects/thirdparty/python_3_8_0/bin/python.exe*  

    -   **swig_PATH** in Ungrouped Entries (only if not found in the PATH),  
        for example: *C:/projects/thirdparty/swigwin-4.0.2/swig.exe*

    -   **Qt5_DIR** in Ungrouped Entries,  
        for example: *C:/Qt/5.10.1/msvc2017_64*

    -   **VULKAN_SDK_DIR** in Ungrouped Entries (only if the environment variable VULKAN_SDK is not set),  
        for example: *C:/VulkanSDK/1.2.198.1*

    -   **PANTORA_AXF_DIR** in Ungrouped Entries,  
        for example: *C:/projects/thirdparty/pantora-axf-1.9.0*

    Note: when you installed a new Visual Studio version after installing CUDA,
    you may have to reinstall CUDA to register it correctly with Visual Studio.
    Otherwise, CMake won't find the CUDA compiler.

6.  When all dependencies have been resolved or the corresponding examples
    have been disabled as indicated in the CMake error messages, the log
    will show that the configuration is done.

    Generate the Visual Studio solution by clicking ``Generate`` and open it
    afterwards using ``Open Project``. CMake-Gui is not needed anymore and
    can be closed.

    You can also open the Visual Studio solution directly from the build
    directory.

7.  Use Visual Studio to build the MDL SDK library, the MDL Core library,
    and the examples. When running the examples using the Visual Studio debugger,
    you can provide additional command line arguments by specifying them in the
    individual Visual Studio project settings.

    You can find the example binaries in the corresponding subfolders in *build/examples*.
    To run the examples by double-clicking the executable in the build directories
    or by using the command line, you need to add the location of the built libraries and
    plugins to your environment PATH or copy them into the corresponding example
    binary folder.

    For the *mdl_sdk* examples, you need
    - *libmdl_sdk.dll* from *build/src/prod/lib/mdl_sdk*,
    - *nv_openimageio.dll* from *build/src/shaders/plugin/openimageio*, and
    - *dds.dll* from *build/src/shaders/plugin/dds*.

    For the *mdl_core* examples, you need *libmdl_core.dll* from
    *build/src/prod/lib/mdl_core*.

8. Similarly, the unit tests can be run via the ``RUN_TESTS`` project in the
    ``_cmake`` folder, or individually from the corresponding project in the
    solution. Alternatively, you can run them from the command line. A very
    flexible way to do that is via the ``ctest`` command from the top-level
    build directory, which also sets up the environment correctly (MI_SRC and
    PATH for the library and the plugins).


## Building on Linux

1.  Before generating make files, you need to install the required
    tools and libraries as listed [above](#thirdparty-dependencies-libs).

    Building on Linux requires a developer environment including Python and
    CMake, which can be installed using the package manager (first command
    below). The second command will install the third-party libraries that
    are available in the package management system:

    ```bash
    sudo apt-get install git git-lfs build-essential python cmake
    ```

    Please note that the build also requires clang 12.0.1. Please download the
    binary as described [above](#thirdparty-dependencies-libs). In
    the following, it is assumed that the extracted clang is the only
    clang compiler found in the system path or, for step 3.ii, that it
    has been extracted to (on x86-64):

        $HOME/projects/thirdparty/clang+llvm-12.0.1-x86_64-linux-gnu-ubuntu-16.04/bin/clang

    or (on aarch64):

        $HOME/projects/thirdparty/clang+llvm-12.0.1-aarch64-linux-gnu/bin/clang
    
2.  It is assumed that you checked out the repository in your home directory
    as follows:

    ```bash
    export MDL_SDK_ROOT=$HOME/projects/mdl-sdk
    git lfs install
    git clone https://github.com/NVIDIA/MDL-SDK.git $MDL_SDK_ROOT
    ```

    Before running CMake, create a build directory that will contain
    your make files and switch to that directory. It is recommended
    that you build into a subdirectory, not the repository root:

    ```bash
    export MDL_SDK_BUILD=$MDL_SDK_ROOT/build
    mkdir -p $MDL_SDK_BUILD
    cd $MDL_SDK_BUILD
    ```

3.  To generate your build files, run CMake with the path to the top-level
    *CMakeLists.txt* as the last argument.

    1.  When all dependencies are installed correctly, the default settings
        should complete the configuration without any further user
        interactions:

        ```bash
        cmake ..
        ```

        In this case, you can continue with Step 4.

    2.  Optionally, you can use CMake options and the *-D* flags to customize
        your build.

        One or multiple of these flags can be used to enable and disable
        examples and logging (see
        [Additional CMake Options](#additional-cmake-options)), for example:

        ```bash
        cmake -DMDL_BUILD_SDK_EXAMPLES=OFF -DMDL_BUILD_CORE_EXAMPLES=OFF ..
        ```

        You can also use the flags to point CMake to custom installation
        directories for third-party libraries. Please refer to
        [Windows build](#thirdparty-dependencies-options) for a list of
        supported flags. On Unix-like systems, it is assumed that the
        specified paths contain a directory named *include* for headers
        files and subdirectories named `lib64` or `lib` that contains shared
        libraries. For the Vulkan SDK for example, the call to CMake could look
        as follows:

        ```bash
        cmake -DVULKAN_SDK_DIR=$HOME/projects/thirdparty/vulkansdk-linux-x86_64-1.2.198.1/1.2.198.1/x86_64 ..
        ```

        When a different clang compiler is installed on your system, you
        can provide the path to a clang 12.0.1 by setting the 'clang_Path'
        option (on x86-64):

        ```bash
        cmake -Dclang_PATH=$HOME/projects/thirdparty/clang+llvm-12.0.1-x86_64-linux-gnu-ubuntu-16.04/bin/clang ..
        ```

        or (on aarch64):

        ```bash
        cmake -Dclang_PATH=$HOME/projects/thirdparty/clang+llvm-12.0.1-aarch64-linux-gnu/bin/clang ..
        ```

        The same applies to other dependencies like Python 3.8.

        For builds using a different compiler version, you need to pass the
        compiler names when calling CMake as follows:

        ```bash
        sudo apt-get install gcc-12 g++-12
        cmake -DCMAKE_C_COMPILER=/usr/bin/gcc-12 -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 ..
        ```
        
        To create an optimized build on a Unix-like system, set the build type to *Release*:

        ```bash
        cmake -DCMAKE_BUILD_TYPE=Release ..
        ```

        and replace `Debug` by `Release` below.

    3.  In case CMake is not able to find a working CUDA compiler for the
        examples, make sure the *nvcc* is reachable through the system PATH
        variable before running CMake:

        ```bash
        export PATH=<CUDA_SDK_DIR>/bin:$PATH
        ```

    4.  If Qt5 cannot be found, or you want to use an extracted package
        rather than installing Qt on your system, you can optionally set
        an additional environment variable before calling CMake:

        ```bash
        export Qt5_DIR=$HOME/Qt/5.10.1/gcc_64
        ```

        or pass the Qt5_DIR as CMake option:

        ```bash
        cmake -DQt5_DIR=$HOME/Qt/5.10.1/gcc_64 ..
        ```

4.  After a successful configuration, you can run make from within the
    specified build directory or any subdirectory that corresponds to a
    source directory containing a *CMakeLists.txt*:

    ```bash
    make -j8
    ```

5.  Examples can be run by directly executing the corresponding binary. For
    example, to run the MDL SDK modules example, use:

    ```bash
    cd $MDL_SDK_BUILD/examples/mdl_sdk/modules/Debug
    ./modules
    ```

6.  Similarly, the unit tests can be run via the ``make test`` target, or in a
    more flexible way via the ``ctest`` command, which also sets up the
    environment correctly (MI_SRC and LD_LIBRARY_PATH for the library and the
    plugins):

    ```bash
    cd $MDL_SDK_BUILD
    ctest
    ```

    For debugging, there is a run script next to each test executable. This
    test script also sets up the environment correctly and can be easily
    modified to launch the debugger of your choice.


## Building on Mac OS X

1.  Before generating make files, you need to install the required
    tools and libraries as listed [above](#thirdparty-dependencies-libs).

    Please note that the build requires clang 12.0.1. Please download the
    binary as described [above](#thirdparty-dependencies-libs). In
    the following, it is assumed that it has been extracted to:

        $HOME/projects/thirdparty/clang+llvm-12.0.1-x86_64-apple-darwin/bin/clang

2.  Depending on your workflow, you can use CMake-Gui and follow the [Windows instructions](#building-on-windows)
    or use the command line as described in the [Linux section](#building-on-linux).
    In each case, begin with step 2 of the respective instructions.

    If the brew packages, Python 3.8, CUDA, and Qt have been installed correctly,
    the following CMake options need to be specified:

    -   **clang_PATH** in Ungrouped Entries,  
        for example: *$HOME/projects/thirdparty/clang+llvm-12.0.1-x86_64-apple-darwin/bin/clang*

    -   **python_PATH** in Ungrouped Entries (only if not found in the PATH),  
        for example: */usr/bin/python*

    -   **Qt5_DIR** in Ungrouped Entries,  
        for example: *$HOME/Qt/5.10.1/clang_64*


3.  After successfully configuring and generating make files, switch to the selected build directory and run make:

    ```bash
    cd $MDL_SDK_BUILD
    make -j8
    ```

4.  Examples can be run by directly executing the corresponding binary. For
    example, to run the MDL SDK modules example, use:

    ```bash
    cd $MDL_SDK_BUILD/examples/mdl_sdk/modules/Debug
    ./modules
    ```

5.  Similarly, the unit tests can be run via the ``make test`` target, or in a
    more flexible way via the ``ctest`` command, which also sets up the
    environment correctly (MI_SRC and LD_LIBRARY_PATH for the library and the
    plugins):

    ```bash
    cd $MDL_SDK_BUILD
    ctest
    ```

    For debugging, there is a run script next to each test executable. This
    test script also sets up the environment correctly and can be easily
    modified to launch the debugger of your choice.


## Additional CMake Options

<a name="cmake-options"></a>
The following options enable you to select the components to be built and to
select particular logging information:

-   **MDL_BUILD_SDK_EXAMPLES**  
    [ON/OFF] enable/disable the MDL SDK examples.

-   **MDL_BUILD_CORE_EXAMPLES**  
    [ON/OFF] enable/disable the MDL Core examples.

-   **MDL_BUILD_DOCUMENTATION**  
    [ON/OFF] enable/disable building of the API documentation.

-   **MDL_ENABLE_CUDA_EXAMPLES**  
    [ON/OFF] enable/disable examples that require CUDA.

-   **MDL_ENABLE_D3D12_EXAMPLES**  
    [ON/OFF] enable/disable examples that require D3D12 (Windows only).

-   **MDL_ENABLE_OPENGL_EXAMPLES**  
    [ON/OFF] enable/disable examples that require OpenGL.

-   **MDL_ENABLE_VULKAN_EXAMPLES**  
    [ON/OFF] enable/disable examples that require Vulkan.

-   **MDL_ENABLE_OPTIX7_EXAMPLES**  
    [ON/OFF] enable/disable examples that require OptiX 7 (Linux and Windows only).

-   **MDL_ENABLE_QT_EXAMPLES**  
    [ON/OFF] enable/disable examples that require Qt.
    
-   **MDL_ENABLE_AXF_EXAMPLES**  
    [ON/OFF] enable/disable the AxF to MDL example.

-   **MDL_ENABLE_PYTHON_BINDINGS**  
    [ON/OFF] enable/disable the generation and compilation of the MDL Python Bindings.

-   **MDL_BUILD_ARNOLD_PLUGIN**  
    [ON/OFF] enable/disable the build of the MDL Arnold Plugin.

-   **MDL_ENABLE_MATERIALX**  
    [ON/OFF] enable/disable MaterialX in examples that support it.

-   **MDL_ENABLE_UNIT_TESTS**  
    [ON/OFF] enable/disable the build of unit tests.

-   **MDL_LOG_PLATFORM_INFOS**  
    [ON/OFF] enable/disable the logging of platform and CMake settings.

-   **MDL_LOG_DEPENDENCIES**  
    [ON/OFF] enable/disable the logging of dependencies of the individual targets.

-   **MDL_LOG_FILE_DEPENDENCIES**  
    [ON/OFF] enable/disable the logging of files that are copied to the output folder.

-   **MDL_MSVC_DYNAMIC_RUNTIME_EXAMPLES**  
    [ON/OFF] links the MSCV dynamic runtime (/MD) instead of static (/MT) when
    creating the example executables.

For any help request, please attach the log messages generated when the log
options are enabled.


## Testing the Build

To verify the build, run the examples as described above.


## Building the API Documentation

The documentation is stored in the `doc/` subdirectory. There are two
C++ APIs -- the __MDL SDK API__ and the __MDL Core API__ -- for which
you need to generate the documentation with Doxygen. Please make sure
to use the specified version 1.9.4.

Additional documents are the MDL Specification (PDF) and the `base.mdl`
and `core_definitions.mdl` documentation (HTML), which you do not
need to generate; they are a part of the source code release.

1.  The tools required to build the documentation are listed
    [here](#doc-build-tools). The `dot` tool from GraphViz is optional: it is
    used to generate nicer inheritance diagrams.

2.  The documentation can be built via the `make` target `doc` or the
    corresponding item in the Visual Studio solution. The target is part of the
    default target.

A start page that links all documents can be found in the doc directory:

```bash
$MDL_SDK_ROOT/doc/index.html
```
