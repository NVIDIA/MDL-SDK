import pymdlsdk
import os

class SDK():
    neuray: pymdlsdk.INeuray = None
    transaction: pymdlsdk.ITransaction = None


    def _get_examples_search_path(self):
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


    def load(self, addExampleSearchPath: bool = True, loadImagePlugins: bool = True):
        """Initialize the SDK and get some common interface for basic testing"""

        # load neuray
        self.neuray = pymdlsdk.load_and_get_ineuray('')
        if not self.neuray.is_valid_interface():
            raise Exception('Failed to load the MDL SDK.')

        # add MDL search paths
        with self.neuray.get_api_component(pymdlsdk.IMdl_configuration) as cfg:
            # add default search paths
            cfg.add_mdl_system_paths()
            cfg.add_mdl_user_paths()

            # get the example search path that is used for all MDL SDK examples
            # falls back to `mdl` in the current working directory
            if addExampleSearchPath:
                example_sp: str = self.get_examples_search_path()
                cfg.add_mdl_path(example_sp)

        # Load plugins
        if loadImagePlugins:
            if not pymdlsdk.load_plugin(self.neuray, 'nv_openimageio'):
                raise Exception('Failed to load the \'nv_openimageio\' plugin.')
            if not pymdlsdk.load_plugin(self.neuray, 'dds'):
                raise Exception('Failed to load the \'dds\' plugin.')

        # start neuray
        resultCode = self.neuray.start()
        if resultCode != 0:
            raise Exception('Failed to initialize the SDK. Result code: ' + resultCode)

        # create a DB transaction
        with self.neuray.get_api_component(pymdlsdk.IDatabase) as database, \
            database.get_global_scope() as scope:
                self.transaction = scope.create_transaction()


    def unload(self, commitTransaction: bool = True):
        """Release all components created in the 'load' function"""
        if commitTransaction:
            self.transaction.commit()
        self.transaction = None
        self.neuray = None
        pymdlsdk._print_open_handle_statistic()

        # Unload the MDL SDK
        if not pymdlsdk.unload():
            raise Exception('Failed to unload the SDK.')
