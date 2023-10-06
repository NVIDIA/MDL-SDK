import pymdlsdk
import os

class SDK():
    neuray: pymdlsdk.INeuray = None
    transaction: pymdlsdk.ITransaction = None
    mdlFactory: pymdlsdk.IMdl_factory = None


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


    def load(self, addExampleSearchPath: bool = True, loadImagePlugins: bool = True, loadDistillerPlugin: bool = False):
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
                example_sp: str = self._get_examples_search_path()
                cfg.add_mdl_path(example_sp)

        # Load plugins
        if loadImagePlugins:
            if not pymdlsdk.load_plugin(self.neuray, 'nv_openimageio'):
                raise Exception('Failed to load the \'nv_openimageio\' plugin.')
            if not pymdlsdk.load_plugin(self.neuray, 'dds'):
                raise Exception('Failed to load the \'dds\' plugin.')

        if loadDistillerPlugin:
            if not pymdlsdk.load_plugin(self.neuray, 'mdl_distiller'):
                raise Exception('Failed to load the \'mdl_distiller\' plugin.')

        # start neuray
        resultCode = self.neuray.start()
        if resultCode != 0:
            raise Exception('Failed to initialize the SDK. Result code: ' + resultCode)

        # create a DB transaction
        with self.neuray.get_api_component(pymdlsdk.IDatabase) as database, \
             database.get_global_scope() as scope:
            self.transaction = scope.create_transaction()

        # fetch other components we need
        self.mdlFactory = self.neuray.get_api_component(pymdlsdk.IMdl_factory)

    def unload(self, commitTransaction: bool = True):
        """Release all components created in the 'load' function"""
        if commitTransaction:
            self.transaction.commit()
        self.transaction = None
        self.mdlFactory = None
        self.neuray = None
        pymdlsdk._print_open_handle_statistic()

        # Unload the MDL SDK
        if not pymdlsdk.unload():
            raise Exception('Failed to unload the SDK.')

    def log_context_messages(self, context: pymdlsdk.IMdl_execution_context) -> bool:
        """print all messages from the context. Return false if there have been errors"""
        if context.get_messages_count() == 0:
            return True
        hasErrors: bool = context.get_error_messages_count() > 0
        for i in range(context.get_messages_count()):
            message: pymdlsdk.IMessage = context.get_message(i)
            level: str = "         "
            if message.get_severity() == 0:  # pymdlsdk.MESSAGE_SEVERITY_FATAL
                level = "fatal:   "
                hasErrors = True
            elif message.get_severity() == 1:  # pymdlsdk.MESSAGE_SEVERITY_ERROR
                level = "error:   "
                hasErrors = True
            elif message.get_severity() == 2:  # pymdlsdk.MESSAGE_SEVERITY_WARNING
                level = "warning: "
            elif message.get_severity() == 3:  # pymdlsdk.MESSAGE_SEVERITY_INFO
                level = "info:    "
            elif message.get_severity() == 4:  # pymdlsdk.MESSAGE_SEVERITY_VERBOSE
                level = "verbose: "
            elif message.get_severity() == 5:  # pymdlsdk.MESSAGE_SEVERITY_DEBUG
                level = "debug:   "
            print(f"{level} {message.get_string()}")
        return not hasErrors


    def load_module(self, qualifiedModuleName: str):
        """Load the module given its name.
        Returns the database name if loaded successfully otherwise empty string"""

        impExp: pymdlsdk.IMdl_impexp_api
        context: pymdlsdk.IMdl_execution_context
        with self.neuray.get_api_component(pymdlsdk.IMdl_impexp_api) as impExp, \
             self.mdlFactory.create_execution_context() as context:
            res = impExp.load_module(self.transaction, qualifiedModuleName, context)
            if not self.log_context_messages(context) or res < 0:
                return ""
            dbName: pymdlsdk.IString = self.mdlFactory.get_db_module_name(qualifiedModuleName)
            if dbName.is_valid_interface():
                return dbName.get_c_str()
        return ""
