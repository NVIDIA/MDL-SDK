import unittest
import os
import gc

# local testing only
import pymdlsdk
BindingModuleName: str = 'pymdlsdk'
BindingModule = pymdlsdk

# the unittest base class, in case testing system already needs a specializing
UnittestFrameworkBase = unittest.TestCase

class SDK():
    neuray: pymdlsdk.INeuray = None
    transaction: pymdlsdk.ITransaction = None
    mdlFactory: pymdlsdk.IMdl_factory = None

    def get_examples_search_path(self):
        """Try to get the example search path or returns 'mdl' sub folder of the current directory if it failed."""

        # get the environment variable that is used in all MDL SDK examples
        example_sp = os.getenv('MDL_SAMPLES_ROOT')

        # fall back to a path relative to this script file
        if example_sp is None or not os.path.exists(example_sp):
            example_sp = os.path.join(os.path.dirname(os.path.realpath(__file__)))
        if example_sp is None or not os.path.exists(example_sp):
            example_sp = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')

        # go down into the mdl folder
        example_sp = os.path.join(example_sp, 'mdl')

        # fall back to the current folder
        if not os.path.exists(example_sp):
            example_sp = './mdl'

        return os.path.abspath(example_sp)

    def get_src_shaders_mdl_search_path(self):  # pragma: no cover
        """Try to get the search path for modules like ::nvidia::core_definitions or
        ::nvidia::axf_to_mdl. Returns the empty string if that directory is the sames as
        get_examples_search_path(), or in case of failure."""
        shader_sp = os.getenv('MDL_SRC_SHADERS_MDL')
        if shader_sp is None or not os.path.exists(shader_sp):
            return ""
        return os.path.abspath(shader_sp)

    def load(self, addExampleSearchPath: bool = True, addExampleResourcePath: bool = True, loadImagePlugins: bool = True, loadDistillerPlugin: bool = False,
        enableTemporaryNames: bool = False, locale: str = ""):
        """Initialize the SDK and get some common interface for basic testing"""

        # load neuray
        self.neuray = pymdlsdk.load_and_get_ineuray('')
        if not self.neuray.is_valid_interface():
            raise Exception('Failed to load the MDL SDK.')  # pragma: no cover

        # add MDL search paths
        with self.neuray.get_api_component(pymdlsdk.IMdl_configuration) as cfg:
            cfg: pymdlsdk.IMdl_configuration = cfg
            # add default search paths
            cfg.add_mdl_system_paths()
            cfg.add_mdl_user_paths()
            cfg.set_expose_names_of_let_expressions(enableTemporaryNames)

            # get the example search path that is used for all MDL SDK examples
            # falls back to `mdl` in the current working directory
            example_sp: str = self.get_examples_search_path()
            src_shader_mdl_sp: str = self.get_src_shaders_mdl_search_path()
            if addExampleSearchPath:
                cfg.add_mdl_path(example_sp)
                if src_shader_mdl_sp:  # pragma: no cover
                    cfg.add_mdl_path(src_shader_mdl_sp)
            if addExampleResourcePath:
                cfg.add_resource_path(example_sp)
                if src_shader_mdl_sp:  # pragma: no cover
                    cfg.add_resource_path(src_shader_mdl_sp)

        # Load plugins
        if loadImagePlugins:
            if not pymdlsdk.load_plugin(self.neuray, 'nv_openimageio'):
                raise Exception('Failed to load the \'nv_openimageio\' plugin.')  # pragma: no cover
            if not pymdlsdk.load_plugin(self.neuray, 'dds'):
                raise Exception('Failed to load the \'dds\' plugin.')  # pragma: no cover

        if loadDistillerPlugin:
            if not pymdlsdk.load_plugin(self.neuray, 'mdl_distiller'):
                raise Exception('Failed to load the \'mdl_distiller\' plugin.')  # pragma: no cover

        # localization
        if locale:
            i18n: pymdlsdk.IMdl_i18n_configuration = self.neuray.get_api_component(pymdlsdk.IMdl_i18n_configuration)
            if i18n.set_locale(locale) != 0:
                raise Exception(f"Failed to set locale to '{locale}'.")  # pragma: no cover

        # start neuray
        resultCode = self.neuray.start()
        if resultCode != 0:
            raise Exception('Failed to initialize the SDK. Result code: ' + resultCode)  # pragma: no cover

        # create a DB transaction
        with self.neuray.get_api_component(pymdlsdk.IDatabase) as database, \
             database.get_global_scope() as scope:
            self.transaction = scope.create_transaction()

        # fetch other components we need
        self.mdlFactory = self.neuray.get_api_component(pymdlsdk.IMdl_factory)

    def unload(self, commitTransaction: bool = True):
        """Release all components created in the 'load' function"""
        if commitTransaction:
            # gc.collect()  # uncomment for better debugging experience in VS Code
            # for automated test, don't run GC since it has to work without
            self.transaction.commit()
        self.transaction = None
        self.mdlFactory = None
        self.neuray.shutdown()
        self.neuray = None
        pymdlsdk._print_open_handle_statistic()

        # Unload the MDL SDK
        if not pymdlsdk.unload():
            raise Exception('Failed to unload the SDK.')  # pragma: no cover
