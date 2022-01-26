#*****************************************************************************
# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

print("Current working directory: " + os.getcwd())
print("\n")

# load the binding module
print("About to load the MDL Python Bindings")
# Note, if that import fails. Make sure you have selected the same python runtime version as
# you have used while building the binding.
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

def inspect_definition(neuray, transaction, function_db_name):
    # Here, we don't use the `with`-blocks in order see what happens
    # without the `with`s or manual releases the objects are to be disposed by the garbage collection
    # assuming that for instance the `function_def` object is disposed when leaving the
    # current function as python ref-counting should be able to see that there is no variable that
    # points to the object anymore.

    # access the module by querying it from the DB
    function_def = transaction.access_as(pymdlsdk.IFunction_definition, function_db_name)
    if not function_def.is_valid_interface():
        return

    # since we set 'materials_are_functions', we can treat them the same way
    print("\nThe Functions is a material: %s" % ("true" if function_def.is_material() else "false"))

    # get some factories that allow to dump values and types as strings to the console
    mdl_factory = neuray.get_api_component(pymdlsdk.IMdl_factory)
    expression_factory = mdl_factory.create_expression_factory(transaction)

    # iterate over parameters and get some info on them
    param_count = function_def.get_parameter_count()
    types = function_def.get_parameter_types()
    defaults = function_def.get_defaults()

    function_annotations = function_def.get_annotations() # not used here
    parameter_annotations = function_def.get_parameter_annotations()

    print("\nThe Functions has %d parameters:" % param_count)
    for i in range(param_count):

        # get parameter name
        param_name = function_def.get_parameter_name(i)
        print("*   Name: %s" % param_name)

        # get the parameter type
        param_type = types.get_type(param_name).skip_all_type_aliases()
        kind = param_type.get_kind()
        print("    Type Kind: %s" % kind)

        # get annotations for this parameter
        anno_block = parameter_annotations.get_annotation_block(param_name)
        # check if the block is valid, if not it means there are no annotations for this parameter
        if anno_block.is_valid_interface():
            print("    Parameter has %d Annotations" % anno_block.get_size())
            # loop over the annotations, same schema as always, get name, use the name access details
            for a in range(anno_block.get_size()):
                anno = anno_block.get_annotation(a)
                print("    + Name: %s" % anno.get_name())

                anno_arguments = anno.get_arguments()
                # not every annotation has arguments, in that case the object is invalid
                if anno_arguments.is_valid_interface():
                    # loop over the arguments, get name, get details
                    for p in range(anno_arguments.get_size()):
                        anno_arg_name = anno_arguments.get_name(p)
                        print("      - Argument: %s" % anno_arg_name)
                        anno_arg_expr = anno_arguments.get_expression(anno_arg_name)
                        # Assuming constants here
                        if anno_arg_expr.get_kind() == pymdlsdk.IExpression.Kind.EK_CONSTANT:
                            anno_arg_value = anno_arg_expr.get_interface(pymdlsdk.IExpression_constant).get_value()
                            anno_arg_value_kind = anno_arg_value.get_kind()
                            print("        Value Kind: %s" % anno_arg_value_kind)
                            if anno_arg_value_kind == pymdlsdk.IValue.Kind.VK_FLOAT:
                                v = anno_arg_value.get_interface(pymdlsdk.IValue_float).get_value()
                                print("        Value: %f" % v)


        # get the default value
        # Note, compared to C++ API we can return the specialized value directly so
        # manual casting based on the kind is not required, handling based on the kind is of course
        param_default = defaults.get_expression(param_name)
        if param_default.is_valid_interface():

            # the simplest default value is a constant value
            expr_kind = param_default.get_kind()
            print("    Default Expression Kind: %s" % expr_kind)
            if expr_kind == pymdlsdk.IExpression.Kind.EK_CONSTANT:
                param_default_value = param_default.get_interface(pymdlsdk.IExpression_constant).get_value()
                param_default_value_kind = param_default_value.get_kind()

                # Note, calling get_interface() is not required here, just like for the expressions
                print("    Default Value Kind: %s" % param_default_value_kind)
                dump = False
                if param_default_value_kind == pymdlsdk.IValue.Kind.VK_BOOL:
                    print("    Value: %s" % ("true" if param_default_value.get_interface(pymdlsdk.IValue_bool).get_value() else "false"))
                elif param_default_value_kind == pymdlsdk.IValue.Kind.VK_INT:
                    print("    Value: %d" % param_default_value.get_interface(pymdlsdk.IValue_int).get_value())
                # ...
                elif param_default_value_kind == pymdlsdk.IValue.Kind.VK_FLOAT:
                    print("    Value: %f" % param_default_value.get_interface(pymdlsdk.IValue_float).get_value())
                elif param_default_value_kind == pymdlsdk.IValue.Kind.VK_DOUBLE:
                    print("    Value: %f" % param_default_value.get_interface(pymdlsdk.IValue_double).get_value())
                # ...
                elif param_default_value_kind == pymdlsdk.IValue.Kind.VK_COLOR:
                    param_default_value_color = param_default_value.get_interface(pymdlsdk.IValue_color)
                    r = param_default_value_color.get_value(0).get_interface(pymdlsdk.IValue_float).get_value()
                    g = param_default_value_color.get_value(1).get_interface(pymdlsdk.IValue_float).get_value()
                    b = param_default_value_color.get_value(2).get_interface(pymdlsdk.IValue_float).get_value()
                    print("    Value: %f / %f / %f" % (r, g, b))
                # ...
                else:
                    dump = True

            # if the default value is a function call we get a database name of that call
            elif expr_kind == pymdlsdk.IExpression.Kind.EK_CALL:
                function_call_db_name = param_default.get_interface(pymdlsdk.IExpression_call).get_call()
                print("    Default Attached Call: %s" % function_call_db_name)

            if dump:
                # Note, this dumping of defaults here is very primitive
                # There are more sophisticated ways to deal with parameters, defaults and annotations to come
                default_text = expression_factory.dump(param_default, None, 1)
                print("    Default (dump):  %s" % default_text.get_c_str())

        else:
            print("    Default: (None)")

#--------------------------------------------------------------------------------------------------

def inspect_module(neuray, transaction, module_db_name):

    # access the module by querying it from the DB
    # also get some factories that allow to dump values and types as strings to the console
    with transaction.access_as(pymdlsdk.IModule, module_db_name) as module, \
        neuray.get_api_component(pymdlsdk.IMdl_factory) as mdl_factory, \
        mdl_factory.create_type_factory(transaction) as type_factory, \
        mdl_factory.create_value_factory(transaction) as value_factory:

        if not module.is_valid_interface():
            # little caveat here, there is no break in a 'with'-block (language short coming?)
            return # so return will exit the `inspect_module` function

        print("MDL Module filename: %s" % module.get_filename())

        print("\nThe Module imports the following %d modules:" % module.get_import_count())
        for i in range(module.get_import_count()):
            print("*   %s" % module.get_import(i))

        # Dump exported types.
        with module.get_types() as types:
            print("\nThe module contains the following %d types:" % types.get_size())
            for i in range(types.get_size()):
                with types.get_type(i) as ttype:
                    with type_factory.dump(ttype, 1) as istring:
                        print("*   %s" % istring.get_c_str())

        # Dump exported constants.
        with module.get_constants() as constants:
            print("\nThe module contains the following %d constants:" % constants.get_size())
            for i in range(constants.get_size()):
                name = constants.get_name(i)
                with constants.get_value(i) as constant:
                    with value_factory.dump(constant, None, 1) as result:
                        print("*     {} = {}".format(name, result.get_c_str()))

        # Dump annotation definitions of the module.
        print("\nThe module contains the following %d annotations:" % module.get_annotation_definition_count())
        for i in range(module.get_annotation_definition_count()):
            with module.get_annotation_definition(i) as anno_def:
                print("*   %s" % anno_def.get_mdl_simple_name())
                print("    MDL Name: %s" % anno_def.get_name())
                for p in range(anno_def.get_parameter_count()):
                    if p == 0:
                        print("    Parameters:")
                    print("    - '{}' of type '{}'".format(anno_def.get_parameter_name(p), anno_def.get_mdl_parameter_type_name(p)))

        # Dump function definitions of the module.
        print("\nThe module contains the following %d function definitions:" % module.get_function_count())
        for i in range(module.get_function_count()):
            print("*   %s" % module.get_function(i))

        # Dump material definitions of the module.
        print("\nThe module contains the following %d material definitions:" % module.get_material_count())
        for i in range(module.get_material_count()):
            print("*   %s" % module.get_material(i))

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
    context.release() # same as: context = None
    return res >= 0

#--------------------------------------------------------------------------------------------------

def run_example(neuray):

    # since the MDL SDK has limited database with no scopes and only one transaction
    # we access the transaction in the beginning and keep it open
    with neuray.get_api_component(pymdlsdk.IDatabase) as database:
        with database.get_global_scope() as scope:

            # this transaction is not used in a `with`-block so it lives in the function scope
            # see the end of the function for the `release()` function
            transaction = scope.create_transaction()
    print("Expectation: 'transaction' handle is not freed and still valid. [Actually: is valid = {}]".format(transaction.is_valid_interface()))

    # load an MDL module, wrapped in a function
    # the module is loaded using it's MDL name
    # in OV the module would be loaded on the C++ side (rtx.neuraylib.plugin)
    module_mdl_name = "::nvidia::sdk_examples::tutorials"
    if load_module(neuray, transaction, module_mdl_name):
        # When the module is loaded we can access it and all its definitions by accessing the DB
        module_db_name = get_db_module_name(neuray, module_mdl_name)
        print("\nMDL Module name: %s" % module_mdl_name)
        print("MDL Module DB name: %s" % module_db_name)

        # after loading was successful, we can inspect materials, functions and other stuff
        inspect_module(neuray, transaction, module_db_name)

        # lets look at a definition
        # lets go for a functions, soon material definitions will behave exactly like functions
        # function_mdl_name = module_mdl_name + "::example_function(color,float)"
        function_mdl_name = module_mdl_name + "::example_df(float,float,color,float,color,float,float,float,float,::nvidia::sdk_examples::tutorials::Options,color,float,string,texture_2d)"
        function_db_name = get_db_definition_name(neuray, function_mdl_name)
        print("\nMDL Function name: %s" % function_mdl_name)
        print("MDL Function DB name: %s" % function_db_name)
        inspect_definition(neuray, transaction, function_db_name)
        print("Inspection Done")


    # load an invalid MDL module
    if load_module(neuray, transaction, "::invalid"):
        # this will not be reached...
        inspect_module(neuray, transaction, "::invalid")

    # changes made with this transaction are committed (here currently we did not edit anything)
    # if that is not done, you will get a warning when the transaction is released
    # Note, we run garbage collection here first in order to release MDL objects that have not been
    # freed after a `with` or released manually. For instance the `function_def` in `inspect_definition`
    # seems to still live, especially when a debugger is attached.
    # Question would be if we should run `gc.collect()` internally before committing a transaction
    # (or calling pymdlsdk.shutdown) or if that is very bad practice.
    gc.collect()
    transaction.commit()

    # this manually releases the transaction. It has the same effect as the __exit__ function that
    # is called at the end of a `with`-block
    # if release is not called, python would clean up the object when all references to the object
    # go out of scope, the GC finds out the object is not used anymore or when the application exits
    # transaction.release()

    # this behaves like calling release()
    transaction = None

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
            # get the example search path that is used for all MDL SDK examples
            # falls back to `mdl` in the current working directory
            example_sp = get_examples_search_path()
            cfg.add_mdl_path(example_sp)

            # unify the handling of materials and functions
            cfg.set_materials_are_functions(True)

        # Load the 'nv_freeimage' plug-in
        if not pymdlsdk.load_plugin(neuray, 'nv_freeimage'):
            raise Exception('Failed to load the \'nv_freeimage\' plugin.')

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

    except Exception as e:
        print("Unexpected error: ", sys.exc_info()[0])
        print("Exception: ", e)
        print("Traceback: ", traceback.format_exc())
        input("\nPress Enter to continue...")
