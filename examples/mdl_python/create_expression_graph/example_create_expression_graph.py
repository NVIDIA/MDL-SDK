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


# load the binding module
print("About to load the MDL Python Bindings")
import pymdlsdk
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


def load_module(neuray: pymdlsdk.INeuray, transaction: pymdlsdk.ITransaction, mdl_module_name: str) -> str:
    """Load the module given it's db name"""

    with neuray.get_api_component(pymdlsdk.IMdl_impexp_api) as imp_exp, \
         neuray.get_api_component(pymdlsdk.IMdl_factory) as mdl_factory, \
         mdl_factory.create_execution_context() as context:

        if imp_exp.load_module(transaction, mdl_module_name, context) < 0:
            return "" # loading failed

        with mdl_factory.get_db_module_name(mdl_module_name) as istring:
            return istring.get_c_str()

#--------------------------------------------------------------------------------------------------


def get_function_by_name(neuray: pymdlsdk.INeuray, transaction: pymdlsdk.ITransaction, module_db_name: str, mdl_function_name: str) -> str:
    """Access an MDL module and find the first function that matches a given name"""

    with transaction.access_as(pymdlsdk.IModule, module_db_name) as mdl_module:
        if not mdl_module.is_valid_interface():
            return ""

        # check if the function name is provided with signature (we don't do overload resolution then)
        if mdl_function_name[-1] == ')':
            function_db_name: str = module_db_name + "::" + mdl_function_name
            with transaction.access_as(pymdlsdk.IFunction_definition, function_db_name) as mdl_function:
                if not mdl_function.is_valid_interface():
                    return ""
                else:
                    return function_db_name

        # do overload resolution otherwise and return the first function with that name
        # if more than overload is expected this function should be extended to narrow down a unique
        # function by specifying parameter types and names. here we simply return the first 
        with mdl_module.get_function_overloads(mdl_function_name, None) as overloads:
            if overloads.get_length() == 0:
                return ""
            if overloads.get_length() != 1:
                print(f"WARNING: the selected function `{mdl_function}` is not unique in module `{mdl_module.get_mdl_name()}`")

            with overloads.get_element(0) as element:
                with element.get_interface(pymdlsdk.IString) as istring_db_name:
                    return istring_db_name.get_c_str()

#--------------------------------------------------------------------------------------------------

def run_example(neuray):

    # since the MDL SDK has limited database with no scopes and only one transaction
    # we access the transaction in the beginning and keep it open
    with neuray.get_api_component(pymdlsdk.IDatabase) as database, \
         database.get_global_scope() as scope, \
         scope.create_transaction() as transaction:

        # load the two modules we want to use
        core_definitions_module_db_name: str = load_module(neuray, transaction, "::nvidia::core_definitions")
        tutorials_module_db_name: str = load_module(neuray, transaction, "::nvidia::sdk_examples::tutorials")

        if not core_definitions_module_db_name or not tutorials_module_db_name:
            print("Failed to load MDL modules.")
            return

        # get the function definition names
        plastic_db_name: str = get_function_by_name(neuray, transaction, core_definitions_module_db_name, "scratched_plastic_v2")
        checker_db_name: str = get_function_by_name(neuray, transaction, tutorials_module_db_name, "color_checker(float,color,color)")

        # get the actual function definitions
        with transaction.access_as(pymdlsdk.IFunction_definition, plastic_db_name) as plastic_definition, \
             transaction.access_as(pymdlsdk.IFunction_definition, checker_db_name) as checker_definition, \
             neuray.get_api_component(pymdlsdk.IMdl_factory) as mdl_factory, \
             mdl_factory.create_expression_factory(transaction) as ef, \
             mdl_factory.create_value_factory(transaction) as vf:

            if not plastic_definition.is_valid_interface() or not checker_definition.is_valid_interface():
                print("Failed to access function definition.")
                return

            # create the color_checker call
            checker_call_db_name = "mdlexample::color_checker"
            with ef.create_expression_list() as parameters, \
                 vf.create_float(5.0) as scale_value, \
                 ef.create_constant(scale_value) as scale_arg, \
                 vf.create_color(0.25, 0.5, 0.75) as b_value, \
                 ef.create_constant(b_value) as b_arg:
                # we leave the parameter 'a' at its default value (white)

                # add the created constant expressions to the new parameter list
                parameters.add_expression("scale", scale_arg)
                parameters.add_expression("b", b_arg)

                # create a function call and pass the parameter list.
                # Values that are not specified stay at their defaults.
                # Parameter without default need to be specified (visible in the definitions).
                with checker_definition.create_function_call(parameters) as checker_call:
                    transaction.store(checker_call, checker_call_db_name)

                ret: pymdlsdk.ReturnCode = pymdlsdk.ReturnCode()
                with checker_definition.create_function_call(parameters, ret) as checker_call:
                    transaction.store(checker_call, checker_call_db_name)
                    print(f"Return code: {ret.value}")

            # create the plastic call
            plastic_call_db_name = "mdlexample::pastic"
            with ef.create_expression_list() as parameters, \
                 ef.create_call(checker_call_db_name) as call_arg: # connect the checker_call

                # add the call expression to the new parameter list and create a function call
                parameters.add_expression("diffuse_color", call_arg)
                with plastic_definition.create_function_call(parameters) as plastic_call:
                    transaction.store(plastic_call, plastic_call_db_name)

        # access the call
        with transaction.access_as(pymdlsdk.IFunction_call, plastic_call_db_name) as plastic_call:
            if not plastic_call.is_valid_interface():
                return

            # from here we can trigger compilation and code generation

            # alternatively, we can write out the created material graph as an MDL
            # Create the module builder.
            # Note, the MDL modules have a `mdl::` prefix and the module name has to be unique
            module_name = "mdl::new_module_9339b384_b05c_11ec_b909_0242ac120002"
            with neuray.get_api_component(pymdlsdk.IMdl_factory) as mdl_factory, \
                 mdl_factory.create_execution_context() as execution_context, \
                 mdl_factory.create_expression_factory(transaction) as ef, \
                 ef.create_annotation_block() as empty_anno_block:

                # create a module builder and add the material as a variant
                with mdl_factory.create_module_builder(transaction, module_name, pymdlsdk.Mdl_version.MDL_VERSION_1_6, pymdlsdk.Mdl_version.MDL_VERSION_LATEST, execution_context) as module_builder:

                    if not module_builder.is_valid_interface():
                        print("Error: Failed to create a module builder.")
                        return

                    if module_builder.add_variant(
                            "MyGraphMaterial",  # material/function name
                            plastic_call.get_function_definition(),  # prototype
                            plastic_call.get_arguments(),  # default arguments, basically our graph
                            empty_anno_block,  # drop annotations
                            empty_anno_block,  # drop annotations
                            True,  # export the material/function
                            execution_context) != 0:

                        print("Error: Failed to add variant to module builder")
                        for i in range(execution_context.get_messages_count()):
                            print(execution_context.get_message(i).get_string())
                        return

                    # export to file
                    filename = os.path.join(os.getcwd(), "MyGraphModule.mdl")
                    with neuray.get_api_component(pymdlsdk.IMdl_impexp_api) as imp_exp:
                        execution_context.set_option("bundle_resources", True)
                        if imp_exp.export_module(transaction, module_name, filename, execution_context) != 0:
                            print(f"Error: Could not export material graph to module '{filename}'")
                        else:
                            print(f"Success: exported material graph to module '{filename}'")

        # close the transaction
        transaction.commit()


#--------------------------------------------------------------------------------------------------
# Entry Point
#--------------------------------------------------------------------------------------------------

def main():
    print(f"Running Example {__file__} in process with id: {os.getpid()}")

    # Get the INeuray interface in a suitable smart pointer that works as context manager
    with pymdlsdk.load_and_get_ineuray('') as neuray:
        if not neuray.is_valid_interface():
            raise Exception('Failed to load the MDL SDK.')

        # configuration settings go here
        # get the component using a context manager
        with neuray.get_api_component(pymdlsdk.IMdl_configuration) as cfg:
            # get the example search path that is used for all MDL SDK examples
            # falls back to `mdl` in the current working directory
            cfg.add_mdl_system_paths()
            cfg.add_mdl_user_paths()
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
        # pymdlsdk._enable_print_ref_counts(False)

        main()

        # optional binding debugging:
        # some more simple memory debugging output
        # unreachable = gc.collect()
        # print("\n%d unreachable objects detected during garbage collection.\n" % unreachable)
        # pymdlsdk._print_open_handle_statistic()

        # sleep to be able to read the output when starting from VS
        time.sleep(2.0)

    except Exception as e:
        print("Unexpected error: ", sys.exc_info()[0])
        print("Exception: ", e)
        print("Traceback: ", traceback.format_exc())
        input("\nPress Enter to continue...")
