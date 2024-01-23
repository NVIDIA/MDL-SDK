#*****************************************************************************
# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#*****************************************************************************

import sys
import os
import gc
import traceback
import time

print("Current working directory: " + os.getcwd())
print("\n")

# load the binding module
print("About to load the MDL Python Bindings")
# Note, if that import fails. Make sure you have selected the same python runtime version as
# you have used while building the binding.
import pymdlsdk
import pymdl
print("Loaded the MDL Python Bindings")

#--------------------------------------------------------------------------------------------------
# Utilities
#--------------------------------------------------------------------------------------------------

def get_examples_search_path():
    """Try to get the example search path or returns 'mdl' sub folder of the current directory if it failed."""

    # get the environment variable that is used in all MDL SDK examples
    example_sp = os.getenv('MDL_SAMPLES_ROOT')

    # fall back to a path relative to this script file
    if example_sp == None or not os.path.exists(example_sp):
        example_sp = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    if example_sp == None or not os.path.exists(example_sp):
        example_sp = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')

    # go down into the mdl folder
    example_sp = os.path.join(example_sp, 'mdl')

    # fall back to the current folder
    if not os.path.exists(example_sp):
        example_sp = './mdl'

    return os.path.abspath(example_sp)

#--------------------------------------------------------------------------------------------------
# MDL Python Example
#--------------------------------------------------------------------------------------------------

def get_db_module_name(neuray, module_mdl_name):
    """Return the db name of the given module."""

    # When the module is loaded we can access it and all its definitions by accessing the DB
    # for that we need to get a the database name of the module using the factory
    with neuray.get_api_component(pymdlsdk.IMdl_factory) as mdl_factory:
        with mdl_factory.get_db_module_name(module_mdl_name) as istring:
            module_db_name = istring.get_c_str() # note, even though this name is simple and could
                                                 # be constructed by string operations, use the
                                                 # factory to be save in case of unicode encodings
                                                 # and potential upcoming changes in the future

        # shortcut for the function above
        # this chaining is a bit special. In the C++ interface it's not possible without
        # leaking memory. Here we create the smart pointer automatically. However, the IString
        # which is created temporary here is released at the end the `load_module` function, right?
        # This might be unexpected, especially when we rely on the RAII pattern and that
        # objects are disposed at certain points in time (usually before committing a transaction)
        module_db_name_2 = mdl_factory.get_db_module_name(module_mdl_name).get_c_str()
        # note, we plan to map compatible types to python. E.g. the IString class may disappear
        return module_db_name_2

#--------------------------------------------------------------------------------------------------

def get_db_definition_name(neuray, function_mdl_name):
    """Return the db name of the given function definition."""

    with neuray.get_api_component(pymdlsdk.IMdl_factory) as mdl_factory:
        with mdl_factory.get_db_definition_name(function_mdl_name) as istring:
            return istring.get_c_str()

#--------------------------------------------------------------------------------------------------

def load_module(neuray, transaction, module_mdl_name):
    """Load the module given its name.
    Returns true if the module is loaded to database"""

    with neuray.get_api_component(pymdlsdk.IMdl_impexp_api) as imp_exp:
        with neuray.get_api_component(pymdlsdk.IMdl_factory) as mdl_factory:
            # for illustration, we don't use a `with` block for the `context`, instead we release manually
            context = mdl_factory.create_execution_context()

        res = imp_exp.load_module(transaction, module_mdl_name, context)
    context.release()  # same as: context = None
    return res >= 0

#--------------------------------------------------------------------------------------------------

def run_example(neuray):

    # since the MDL SDK has limited database with no scopes and only one transaction
    # we access the transaction in the beginning and keep it open
    with neuray.get_api_component(pymdlsdk.IDatabase) as database, \
         database.get_global_scope() as scope, \
         scope.create_transaction() as transaction:

        # load an MDL module, wrapped in a function
        # the module is loaded using it's MDL name
        # in OV the module would be loaded on the C++ side (rtx.neuraylib.plugin)
        module_mdl_name = "::nvidia::sdk_examples::tutorials"
        if load_module(neuray, transaction, module_mdl_name):
            # When the module is loaded we can access it and all its definitions by accessing the DB
            module_db_name = get_db_module_name(neuray, module_mdl_name)
            print("\nMDL Module name: %s" % module_mdl_name)
            print("MDL Module DB name: %s" % module_db_name)

            # when the module is loaded we can use the high-level python binding to inspect the module 
            module: pymdl.Module = pymdl.Module._fetchFromDb(transaction, module_db_name)

            # inspect the module
            if module:
                print(f" filename: {module.filename}")
                print(f" dbName: {module.dbName}")
                print(f" mdlName: {module.mdlName}")
                print(f" mdlSimpleName: {module.mdlSimpleName}")

                print(f" types:")
                for type in module.types: # exported types are structs or enums
                    print(f" - type name: {type.name}")
                    print(f"   kind: {type.kind}")
                    if type.kind == pymdlsdk.IType.Kind.TK_ENUM:
                        print(f"   enum values: {type.enumValues}")

                print(f" functions:")
                for funcName in module.functions:
                    print(f" * simple name: {funcName}")
                    overloads = module.functions[funcName]
                    for func in overloads: 
                        print(f" - name with signature (overload): {funcName}{func.parameterTypeNames}")
                        matOrFunc = "Material" if func.isMaterial else "Function"
                        print(f"   material or function: {matOrFunc}")
                        print(f"   isExported: {func.isExported}")
                        print(f"   semantics: {func.semantics}")
                        print(f"   returns: ")
                        print(f"     kind: {func.returnValue.type.kind}")
                        print(f"     type name: {func.returnValue.type.name}")
                        print(f"     uniform: {func.returnValue.type.uniform}")
                        print(f"     varying: {func.returnValue.type.varying}")
                        print(f"   parameters:")
                        for param in func.parameters.items():
                            argument = param[1]
                            print(f"     [{param[0]}] parameter name tuple: {func.parameterTypeNames}")
                            print(f"       kind: {argument.type.kind}")
                            print(f"       type name: {argument.type.name}")
                            print(f"       uniform: {argument.type.uniform}")
                            print(f"       varying: {argument.type.varying}")

                        continue ### ONLY TO REDUCE THE AMOUNT OF OUTPUT, feel free to remove the 'continue' 

                        # Annotations
                        if func.annotations:
                            print(f"      Annotations:")
                            anno: pymdl.Annotation
                            for anno in func.annotations:
                                print(f"      - Simple Name: {anno.simpleName}")
                                print(f"        Qualified Name: {anno.name}")
                                arg: pymdl.Argument
                                for arg_name, arg in anno.arguments.items():
                                    print(f"        ({arg.type.kind}) {arg_name}: {arg.value}")

            # close transaction before destroying it
            transaction.commit()

#--------------------------------------------------------------------------------------------------
# Entry Point
#--------------------------------------------------------------------------------------------------

def main():

    # Get the INeuray interface in a suitable smart pointer that works as context manager
    with  pymdlsdk.load_and_get_ineuray('') as neuray:
        if not neuray.is_valid_interface():
            raise Exception('Failed to load the MDL SDK.')

        # configuration settings go here
        # get the component using a context manager
        with neuray.get_api_component(pymdlsdk.IMdl_configuration) as cfg:
            # add default search paths
            cfg.add_mdl_system_paths()
            cfg.add_mdl_user_paths()
            
            # get the example search path that is used for all MDL SDK examples
            # falls back to `mdl` in the current working directory
            example_sp = get_examples_search_path()
            cfg.add_mdl_path(example_sp)

        # Load the 'nv_openimageio' and 'dds' plug-ins
        if not pymdlsdk.load_plugin(neuray, 'nv_openimageio'):
            raise Exception('Failed to load the \'nv_openimageio\' plugin.')
        if not pymdlsdk.load_plugin(neuray, 'dds'):
            raise Exception('Failed to load the \'dds\' plugin.')

        # after the configuration is done, start neuray.
        resultCode = neuray.start()
        if resultCode != 0:
            raise Exception('Failed to initialize the SDK. Result code: ' + resultCode)

        # the actual example that should be illustrated
        run_example(neuray)

        # Shutting the MDL SDK down. Again, a return code of 0 indicates success.
        resultCode = neuray.shutdown()
        if resultCode != 0:
            raise Exception('Failed to shutdown the SDK. Result code: ' + resultCode)

    # Unload the MDL SDK
    if not pymdlsdk.unload():
        raise Exception('Failed to unload the SDK.')
    print('Unloaded the MDL SDK.')


if __name__ == "__main__":
    try:
        # optional binding debugging:
        # some more simple memory debugging output
        # pymdlsdk._enable_print_ref_counts(True)

        main()

        # optional binding debugging:
        # some more simple memory debugging output
        unreachable = gc.collect()
        print("\n%d unreachable objects detected during garbage collection.\n" % unreachable)
        # pymdlsdk._print_open_handle_statistic()

        # sleep to be able to read the output when starting from VS
        time.sleep(2.0)

    except Exception as e:
        print("Unexpected error: ", sys.exc_info()[0])
        print("Exception: ", e)
        print("Traceback: ", traceback.format_exc())
        input("\nPress Enter to continue...")
