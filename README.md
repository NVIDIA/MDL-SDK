# NVIDIA MDL SDK

The NVIDIA MDL SDK enables the easy integration of MDL support into
rendering and material editing applications. The SDK contains components
for loading, inspecting, and editing of material definitions as well as
compiling MDL materials and functions to PTX and LLVM-IR.


## NVIDIA Material Definition Language (MDL)

![MDL example material renderings](doc/images/mdl_material_examples.jpg)

The [NVIDIA Material Definition Language (MDL)](https://www.nvidia.com/mdl) 
is a domain-specific programming language for defining physically-based 
materials for rendering. It allows you to define  *materials* and *functions*,
which you can organize in *modules* and *packages* to create flexible, 
custom-built material catalogs.

Material definitions are written in a declarative style; they define
what to compute -- not how to compute it. This is the central premise in 
MDL where one material definition delivers the same appearance in many
rendering algorithms. Following is a simple example of a diffuse material 
in MDL:

    material diffuse ( color diffuse_color = color(0.7))
        = material(
            surface: material_surface (
                scattering: df::diffuse_reflection_bsdf (
                    tint: diffuse_color
                )
            )
        );

The function definitions in MDL are written in a procedural programming
style. Their use is limited to computing material parameters in a 
side-effect-free manner.

The clear separation of material definitions from function definitions and 
their respective constraints makes possible the optimization of rendering 
algorithms independently of the material definition.


## Pre-compiled Binaries

NVIDIA offers a binary release of the MDL SDK, see 
[https://developer.nvidia.com/mdl-sdk](https://developer.nvidia.com/mdl-sdk). 
The binary release is different in some functionality as documented in the 
[Change Log](CHANGELOG.md).


## Support

- [NVIDIA MDL SDK Forum](https://devtalk.nvidia.com/default/board/253/mdl-sdk/)


## Building the MDL SDK from Source

MDL uses [CMake](http://www.cmake.org/) to generate build files for a 
particular development environment. It is suggested to use CMake 3.11 or later, 
which can be downloaded from the [CMake Website](https://cmake.org/download/).
When using a Unix-like system, you can install the *cmake* package using 
the respective package management systems. On the Mac OS X platform, third party
dependencies can be resolved using the [Homebrew Package Manager](https://brew.sh/index_de).


### Dependencies

The source code requires a C++11 compiler and several third-party libraries 
and tools to build the MDL SDK. Additional third-party libraries are used in 
the examples.

The build with the following x64-platform-compiler combinations has been 
successfully tested:

-   **Windows 10:**     Microsoft Visual Studio 2017 (msvc v141) or 2019 (msvc v142) 
    are recommended. Visual Studio 2015 (msvc v140) is the minimum requirement.
-   **Ubuntu 18.04:**   gcc/g++6 and gcc/g++7
-   **Mac OS X 10.13:** Xcode 8.3.3 (Apple Clang 8.1)

The versions listed with the following dependencies have been
successfully tested. Where not mentioned otherwise, other versions
might work as well.
The tested versions include the ones specified in the
[VFX Reference Platform](https://vfxplatform.com/) CY2020.

<a name="thirdparty-dependencies-libs"></a>
The following third-party libraries are required to build the MDL SDK:

-   **Boost** *(1.67.0 or 1.70.0)* (headers)  
    Linux: Install the *libboost-dev* package.  
    Windows: Download and extract the boost source code from 
    [boost.org](https://www.boost.org/users/download/).  
    Mac OS X: Install the *boost* package using brew.

-   **FreeImage** *(3.17.0)*  
    Linux: Install the *libfreeimage-dev* package.  
    Windows: Download and extract the pre-compiled binaries from 
    [freeimage.sourceforge.net](http://freeimage.sourceforge.net/download.html).  
    Mac OS X: Install the *freeimage* package using brew.

-   **Python2** *(2.7.1)* or **Python3** *(3.7.5)*
    Linux: Install the *python* package.  
    Windows and Max OS X: Download and install Python 2.7 or 3.7 from 
    [python.org](https://www.python.org/downloads/).

-   **Clang 7.0.0**  
    Using version 7.0.0 is mandatory.  
    Pre-compiled binaries can be found on
    [llvm.org](http://releases.llvm.org/download.html#7.0.0).

In order to build and run all examples, you need additional third-party 
libraries. These additional libraries are:

-   **GLEW** *(2.1.0)*  
    Linux: Install the *libglew-dev* package.  
    Windows: Download and extract the pre-compiled binaries from 
    [glew.sourceforge.net](http://glew.sourceforge.net/).  
    Mac OS X: Install the *glew* package using brew.

-   **GLFW** *(3.2.1)*  
    Linux: Install the *libglfw3-dev* package.  
    Windows: Download and extract the pre-compiled x64 binaries from 
    [glfw.org](http://www.glfw.org/download.html).  
    Mac OS X: Install the *glfw* package using brew.

-   **NVIDIA CUDA Toolkit** *(9.0 or later)*  
    Please follow the instructions on the 
    [CUDA Developer Website](https://developer.nvidia.com/cuda-toolkit).

-   **Qt** *(5.10.1 or 5.12.6)*  
    Please follow the instructions on the [Qt Website](https://www.qt.io/).  
    To build with Qt support on Linux, your system's GLIBC needs to be release 
    2.14 or later.

-   **DirectX Raytracing support**  
    Building the DXR example requires:  

    - Windows 10 version 1809 and the corresponding SDK 10.0.17763.0 or
    - Windows 10 version 1909 and the corresponding SDK 10.0.18362.0

    Additionally the optional *Graphic Tools* feature has to be installed.

-   **Arnold SDK** *(6.0.0.0)*
    Please follow the instructions on the [Arnold Website](https://www.arnoldrenderer.com/arnold/download/) to download the Arnold SDK.

-   **MaterialX** *(github repository, branch: v1.38, soon master)*  
    Please checkout the repository from [github](https://github.com/materialx/MaterialX)
    and build the libraries using CMake for Debug, Release and RelWithDebInfo.  

<a name="doc-build-tools"></a>
Required tools to build the documentation:

-   **Doxygen** *(1.8.4)*  
    See the [Doxygen project page](https://sourceforge.net/projects/doxygen/) and 
    the [archive of all releases](https://sourceforge.net/projects/doxygen/files/).

-   **dot from GraphViz** *(2.40.1)* (optional)  
    See the [GraphViz project page](https://www.graphviz.org/).


### Building on Windows

1.  Before generating the Visual Studio 2015 solution, be sure to
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
    *C:/projects/mdl-sdk/build/vs2015* for example is fine, assuming you cloned the repository to:

        C:/projects/mdl-sdk

3.  After clicking ``Configure``, CMake asks you to choose the
    Generator. Select `Visual Studio 14 2015 Win64` (for newer CMake versions
    select `x64` as platform), enter `host=x64` as toolset and click
    `Finish`.  CMake starts to configure the build and stops several
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
    During this process, you will need to setup the following entries:

    -   **BOOST_INCLUDEDIR** in Ungrouped Entries,  
        for example: *C:/projects/thirdparty/boost_1_67_0*, which contains the *boost* directory.  

    -   **FREEIMAGE_DIR** in the FREEIMAGE group,  
        for example: *C:/projects/thirdparty/freeimage_3_17_0/Dist/x64*

    -   **GLEW_DIR** in the GLEW group,  
        for example: *C:/projects/thirdparty/glew-2.1.0*

    -   **GLFW_DIR** in Ungrouped Entries,  
        for example: *C:/projects/thirdparty/glfw-3.2.1.bin.WIN64*

    -   **clang_PATH** in Ungrouped Entries (only if not found in the PATH),  
        for example: *C:/Program Files/LLVM-7/bin/clang.exe*

    -   **python_PATH** in Ungrouped Entries (only of not found in the PATH),  
        for example: *C:/projects/thirdparty/python_2_7_1/bin/python.exe*

    -   **Qt5_DIR** in Ungrouped Entries,  
        for example: *C:/Qt/5.10.1/msvc2015_64*

    -   **ARNOLD_SDK_DIR** in Ungrouped Entries,
        for example: *C:/projects/thirdparty/Arnold-6.0.3.0-windows*

    -   **MATERIALX_REPOSITORY** and **MATERIALX_BUILD**,
        for example: *C:/projects/thirdparty/git/MaterialX*
        and *C:/projects/thirdparty/git/MaterialX/build* respectively.
        Here, *MATERIALX_BUILD*  has to point to the build directory selected while building the MaterialX libraries using CMake.
        For MaterialX support, the option **MDL_MSVC_DYNAMIC_RUNTIME_EXAMPLES** has to be enabled.

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

7.  For Visual Studio 2015, you may have to adapt the `Target Platform Version` under
    `General` in the example and third-party project properties to avoid getting build errors.
    CMake is configured to select the latest SDK installed. Select `10.0.18362.0`
    manually if you have the SDK installed. Otherwise, select version `10.0.17763.0`.
    For later Visual Studio versions, CMake will use the correct platform version.

8.  Use Visual Studio to build the MDL SDK library, the MDL Core library,
    and the examples. When running the examples using the Visual Studio debugger,
    you can provide additional command line arguments by specifying them in the
    individual Visual Studio project settings.

    You can find the example binaries in the corresponding subfolders in *build/examples*.
    To run the examples by double-clicking the executable in the build directories
    or by using the command line, you need to add the location of the built libraries and
    plugins to your environment PATH or copy them into the corresponding example 
    binary folder.

    For the *mdl_sdk* examples, you need *libmdl_sdk.dll* from
    *build/src/prod/lib/mdl_sdk* and additionally,
    *freeimage.dll* and *nv_freeimage.dll* from
    *build/src/shaders/plugin/freeimage*.

    For the *mdl_core* examples, you need *libmdl_core.dll* from
    *build/src/prod/lib/mdl_core* and the *freeimage.dll* from 
    *build/src/shaders/plugin/freeimage* or your third-party downloads.


### Building on Linux

1.  Before generating make files, you need to install the required
    tools and libraries as listed [above](#thirdparty-dependencies-libs).

    Building on Linux requires a developer environment including Python and 
    CMake, which can be installed using the package manager (first command 
    below). The second command will install the third-party libraries that 
    are available in the package management system:

    ```bash
    sudo apt-get install git build-essential python cmake g++
    sudo apt-get install libboost-dev libfreeimage-dev libglew-dev libglfw3-dev
    ```

    Please note that the build also requires clang 7.0.0. Please download the
    binary as described [above](#thirdparty-dependencies-libs). In
    the following, it is assumed that the extracted clang is the only
    clang compiler found in the system path or, for step 3.ii, that it
    has been extracted to:

        $HOME/projects/thirdparty/clang+llvm-7.0.0-x86_64-linux-gnu-ubuntu-16.04/bin/clang
    
2.  It is assumed that you checked out the repository in your home directory 
    as follows:

    ```bash
    export MDL_SDK_ROOT=$HOME/projects/mdl-sdk
    git clone https://github.com/NVIDIA/MDL-SDK.git $MDL_SDK_ROOT
    ```

    Before running CMake, create a build directory that will contain 
    your make files and switch to that directory. It is recommended 
    that you build into a subdirectory, not the repository root: 

    ```bash
    export MDL_SDK_BUILD=$MDL_SDK_ROOT/build/linux-x64-gcc7
    mkdir -p $MDL_SDK_BUILD
    cd $MDL_SDK_BUILD
    ```

3.  To generate your build files, run CMake with the path to the top-level 
    *CMakeLists.txt* as the last argument.

    1.  When all dependencies are installed correctly, the default settings 
        should complete the configuration without any further user 
        interactions:

        ```bash
        cmake ../..
        ```

        In this case, you can continue with Step 4.

    2.  Optionally, you can use CMake options and the *-D* flags to customize 
        your build.

        One or multiple of these flags can be used to enable and disable examples and logging (see 
        [Additional CMake Options](#additional-cmake-options)), for example:

        ```bash
        cmake -DMDL_BUILD_SDK_EXAMPLES=OFF -DMDL_BUILD_CORE_EXAMPLES=OFF ../..
        ```

        You can also use the flags to point CMake to custom installation 
        directories for third-party libraries. Please refer to 
        [Windows build](#thirdparty-dependencies-options) for a list of 
        supported flags. On Unix-like systems, it is assumed that the 
        specified paths contain a directory named *include* for headers 
        files and subdirectories named `lib64` or `lib` that contain shared 
        libraries. For a custom build of the GLEW library for example, the 
        call to CMake could look as follows:

        ```bash
        cmake -DGLEW_DIR=$HOME/projects/thirdparty/glew-2.1.0 ../..
        ```

        When a different clang compiler is installed on your system, you
        can provide the path to a clang 7.0.0 by setting the 'clang_Path'
        option:

        ```bash
        cmake -Dclang_PATH=$HOME/projects/thirdparty/clang+llvm-7.0.0-x86_64-linux-gnu-ubuntu-16.04/bin/clang ../..
        ```
        
        The same applies to other dependencies like Python 2.7.

        For builds using a different compiler version, you need to pass the 
        compiler names when calling CMake as follows:

        ```bash
        sudo apt-get install gcc-6 g++-6
        cmake -DCMAKE_C_COMPILER=/usr/bin/gcc-6 -DCMAKE_CXX_COMPILER=/usr/bin/g++-6 ../..
        ```
        
        To create an optimized build on a Unix-like system, set the build type to *Release*:

        ```bash
        cmake -DCMAKE_BUILD_TYPE=Release ../..
        ```

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
        cmake -DQt5_DIR=$HOME/Qt/5.10.1/gcc_64 ../..
        ```

4.  After a successful configuration, you can run make from within the
    specified build directory or any subdirectory that corresponds to a
    source directory containing a *CMakeLists.txt*:

    ```bash
    make -j8
    ```

5.  Because the different MDL SDK libraries are loaded at runtime, their 
    location has to be provided in order to run an example. Therefore, 
    specify the paths to the built MDL SDK library, MDL Core library, and 
    the image plugins using the `LD_LIBRARY_PATH` variable:

    ```bash
    export LD_LIBRARY_PATH=$MDL_SDK_BUILD/src/prod/lib/mdl_sdk:$MDL_SDK_BUILD/src/prod/lib/mdl_core:$MDL_SDK_BUILD/src/shaders/plugin/freeimage${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    ```

    For example, to run the MDL SDK modules example, use:

    ```bash
    cd $MDL_SDK_BUILD/examples/mdl_sdk/modules
    ./modules
    ```

### Building on Mac OS X

1.  Before generating make files, you need to install the required
    tools and libraries as listed [above](#thirdparty-dependencies-libs).

    Please note that the build requires clang 7.0.0. Please download the
    binary as described [above](#thirdparty-dependencies-libs). In
    the following, it is assumed that it has been extracted to:

        $HOME/projects/thirdparty/clang+llvm-7.0.0-x86_64-apple-darwin/bin/clang

2.  Depending on your workflow, you can use CMake-Gui and follow the [Windows instructions](#building-on-windows) 
    or use the command line as described in the [Linux section](#building-on-linux).
    In each case, begin with step 2 of the respective instructions.

    If the brew packages, Python 2.7, CUDA, and Qt have been installed correctly,
    the following CMake options need to be specified:

    -   **clang_PATH** in Ungrouped Entries,  
        for example: *$HOME/projects/thirdparty/clang+llvm-7.0.0-x86_64-apple-darwin/bin/clang*

    -   **python_PATH** in Ungrouped Entries (only of not found in the PATH),  
        for example: */usr/bin/python*

    -   **Qt5_DIR** in Ungrouped Entries,  
        for example: *$HOME/Qt/5.10.1/clang_64*


3.  After successfully configuring and generating make files, switch to the selected build directory and run make:

    ```bash
    cd $MDL_SDK_BUILD
    make -j8
    ```

4.  Because the different MDL SDK libraries are loaded at runtime, their 
    location has to be provided in order to run an example. Therefore, 
    specify the paths to the built MDL SDK library, MDL Core library, and 
    the image plugins using the `DYLD_LIBRARY_PATH` variable:

    ```bash
    export Qt5_DIR=$HOME/Qt/5.10.1/clang_64
    export DYLD_LIBRARY_PATH=$MDL_SDK_BUILD/src/prod/lib/mdl_sdk:$MDL_SDK_BUILD/src/prod/lib/mdl_core:$MDL_SDK_BUILD/src/shaders/plugin/freeimage:${Qt5_DIR}/lib:${Qt5_DIR}/plugins/imageformats${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}
    ```

    For example, to run the MDL SDK modules example, use:

    ```bash
    cd $MDL_SDK_BUILD/examples/mdl_sdk/modules
    ./modules
    ```

### Additional CMake Options

The following options enable you to select the components to be built and to 
select particular logging information:

-   **MDL_BUILD_SDK_EXAMPLES**  
    [ON/OFF] enable/disable the MDL SDK examples.

-   **MDL_BUILD_CORE_EXAMPLES**  
    [ON/OFF] enable/disable the MDL Core examples.

-   **MDL_ENABLE_CUDA_EXAMPLES**  
    [ON/OFF] enable/disable examples that require CUDA.

-   **MDL_ENABLE_D3D12_EXAMPLES**  
    [ON/OFF] enable/disable examples that require D3D12 (Windows only).

-   **MDL_ENABLE_OPENGL_EXAMPLES**  
    [ON/OFF] enable/disable examples that require OpenGL.

-   **MDL_ENABLE_OPTIX7_EXAMPLES**  
    [ON/OFF] enable/disable examples that require OptiX 7 (Linux and Windows only).

-   **MDL_ENABLE_QT_EXAMPLES**  
    [ON/OFF] enable/disable examples that require Qt.

-   **MDL_BUILD_ARNOLD_PLUGIN**  
    [ON/OFF] enable/disable the build of the MDL Arnold Plugin.

-   **MDL_ENABLE_MATERIALX**  
    [ON/OFF] enable/disable MaterialX in examples that support it.

-   **MDL_LOG_PLATFORM_INFOS**  
    [ON/OFF] enable/disable the logging of platform and CMake settings.

-   **MDL_LOG_DEPENDENCIES**  
    [ON/OFF] enable/disable the logging of dependencies of the individual targets.

-   **MDL_LOG_FILE_DEPENDENCIES**  
    [ON/OFF] enable/disable the logging of files that are copied to the output folder.

-   **MDL_MSVC_DYNAMIC_RUNTIME_EXAMPLES**  
    [ON/OFF] links the MSCV dynamic runtime (/MD) instead of static (/MT) when
    creating the example executables.

By default, all options are set to ON. For any help request, please attach 
the log messages generated when the log options are enabled.


### Testing the Build

To verify the build, run the examples as described above.


### Building the API Documentation

The documentation is stored in the `doc/` subdirectory. There are two 
C++ APIs -- the __MDL SDK API__ and the __MDL Core API__ -- for which 
you need to generate the documentation with Doxygen. Please make sure 
to use the specified version 1.8.4.

Additional documents are the MDL Specification (PDF) and the `base.mdl` 
and `core_definitions.mdl` documentation (HTML), which you do not
need to generate; they are a part of the source code release.

1.  The tools required to build the documentation are listed 
    [here](#doc-build-tools).

2.  The use of the `dot` tool is optional and disabled by default. Enabling 
    it gives you nicer inheritance diagrams. It is only supported on Linux 
    and Mac, since a custom bash script is used to process the diagrams. You 
    can enable the `dot` tool as follows:

    1.  Set the `HAVE_DOT` configuration setting in the respective `Doxyfile`
        to `YES` in `doc/mdl_sdkapi` and `doc/mdl_coreapi`.

    2.  Set the `GRAPHVIZ_DOT_HOME` environment variable to the directory 
        containing the GraphVis `dot` command (not the local `doc/mdl_sdkapi` 
        or `doc/mdl_coreapi` directory with the equally named `dot` shell 
        script, which is just a wrapper). Defaults to /usr/bin on Linux.

    3.  On Mac, set the `PLATFORM` environment variable to a suitable
        setting starting with `macosx-x86`.

3.  Build the __MDL SDK API__ documentation with a plain `doxygen` call:

    ```bash
    cd $MDL_SDK_ROOT/doc/mdl_sdkapi
    doxygen
    ```

    The resulting documentation is placed in an `html/` subdirectory with 
    the start page: 

    ```bash
    $MDL_SDK_ROOT/doc/mdl_sdkapi/html/index.html
    ```

4.  Build the __MDL Core API__ documentation with a plain `doxygen` call:

    ```bash
    cd $MDL_SDK_ROOT/doc/mdl_coreapi
    doxygen
    ```

    The resulting documentation is placed in an `html/` subdirectory with
    the start page:

    ```bash
    $MDL_SDK_ROOT/doc/mdl_coreapi/html/index.html
    ```

A start page that links all documents can be found in the doc directory:

```bash
$MDL_SDK_ROOT/doc/index.html
```

## Repository Structure

The NVIDIA MDL SDK repository consists of the following directories and files:

    include/       - C++ API header files
    examples/      - example programs and MDL files
    src/           - source code for the SDK libraries
    doc/           - API documentation, MDL specification, 
                     core_definitions.mdl and base.mdl documentation
    cmake/         - support files for the CMAKE build system
    
    README.md      - this file: introduction and build instructions
    CHANGELOG.md   - change log and difference to the binary MDL SDK release
    LICENSE.md     - license for the MDL SDK and references to 
                     third-party licenses
    CMakeLists.txt - top level CMAKE file


## Additional Resources

- [NVIDIA MDL Home Page](https://www.nvidia.com/mdl)
- [NVIDIA MDL SDK Forum](https://devtalk.nvidia.com/default/board/253/mdl-sdk/)
