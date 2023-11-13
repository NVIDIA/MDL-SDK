try:  # pragma: no cover
    # testing from within a package or CI
    from .setup import BindingModule, UnittestFrameworkBase
    pymdlsdk = BindingModule
except ImportError:  # pragma: no cover
    # local testing
    from setup import UnittestFrameworkBase
    import pymdlsdk


# shared test helper function
class UnittestBase(UnittestFrameworkBase):
    # sdk: SDK  # declared in derived classes

    def assertIsValidInterface(self, iinterface: pymdlsdk.IInterface):
        self.assertIsNotNone(iinterface)
        self.assertTrue(iinterface.is_valid_interface())
        # UPCOMING ref1: int = iinterface.__iinterface_refs__()
        # UPCOMING ptr1: int = iinterface.__iinterface_ptr_as_uint64__()
        iinterface2 = iinterface  # this does not increase interface ref count
        # UPCOMING self.assertEqual(ref1, iinterface2.__iinterface_refs__())
        # UPCOMING self.assertEqual(ptr1, iinterface2.__iinterface_ptr_as_uint64__())  # same iinterface
        iinterface = None  # doesn't matter which python variable is reset
        # UPCOMING elf.assertEqual(ref1, iinterface2.__iinterface_refs__())
        # UPCOMING elf.assertEqual(ptr1, iinterface2.__iinterface_ptr_as_uint64__())

        shortTime = iinterface2.get_interface(type(iinterface2))
        self.assertTrue(shortTime.is_valid_interface())
        # UPCOMING self.assertEqual(ref1 + 1, iinterface2.__iinterface_refs__())
        shortTime.release()
        # UPCOMING self.assertEqual(ref1, iinterface2.__iinterface_refs__())
        self.assertFalse(shortTime.is_valid_interface())

        with iinterface2.get_interface(pymdlsdk.IInterface) as iinterface3:
            # UPCOMING self.assertEqual(ref1 + 1, iinterface3.__iinterface_refs__())
            with iinterface3.get_interface(type(iinterface2)) as iinterface4:
                # UPCOMING self.assertEqual(ref1 + 2, iinterface4.__iinterface_refs__())
                iinterface4.compare_iid(type(iinterface2).IID())
                iinterface2.compare_iid(pymdlsdk.IInterface.IID())
                # self.assertEqual(iinterface2.get_iid(), iinterface3.get_iid())
        # UPCOMING self.assertEqual(ref1, iinterface2.__iinterface_refs__())

    def log_context_messages(self, context: pymdlsdk.IMdl_execution_context) -> bool:
        """print all messages from the context. Return false if there have been errors"""
        if context.get_messages_count() == 0:
            return True
        hasErrors: bool = context.get_error_messages_count() > 0
        for i in range(context.get_messages_count()):
            message: pymdlsdk.IMessage = context.get_message(i)
            # UPCOMING level: str = "         "
            # UPCOMING if message.get_severity() == pymdlsdk.Message_severity.MESSAGE_SEVERITY_FATAL:
            # UPCOMING     level = "fatal:   "
            # UPCOMING     hasErrors = True
            # UPCOMING elif message.get_severity() == pymdlsdk.Message_severity.MESSAGE_SEVERITY_ERROR:
            # UPCOMING     level = "error:   "
            # UPCOMING     hasErrors = True
            # UPCOMING elif message.get_severity() == pymdlsdk.Message_severity.MESSAGE_SEVERITY_WARNING:
            # UPCOMING     level = "warning: "
            # UPCOMING elif message.get_severity() == pymdlsdk.Message_severity.MESSAGE_SEVERITY_INFO:
            # UPCOMING     level = "info:    "
            # UPCOMING elif message.get_severity() == pymdlsdk.Message_severity.MESSAGE_SEVERITY_VERBOSE:
            # UPCOMING     level = "verbose: "
            # UPCOMING elif message.get_severity() == pymdlsdk.Message_severity.MESSAGE_SEVERITY_DEBUG:
            # UPCOMING     level = "debug:   "
            # UPCOMING print(f"{level} {message.get_string()}")
            print(f"Context Message: {message.get_string()}")
        return not hasErrors

    def assertContextNoErrors(self, context: pymdlsdk.IMdl_execution_context):
        self.assertTrue(self.log_context_messages(context))

    def load_module(self, qualifiedModuleName: str):
        """Load the module given its name.
        Returns the database name if loaded successfully otherwise empty string"""

        impExp: pymdlsdk.IMdl_impexp_api
        context: pymdlsdk.IMdl_execution_context
        with self.sdk.neuray.get_api_component(pymdlsdk.IMdl_impexp_api) as impExp, \
             self.sdk.mdlFactory.create_execution_context() as context:
            res = impExp.load_module(self.sdk.transaction, qualifiedModuleName, context)
            if not self.log_context_messages(context) or res < 0:
                return ""
            dbName: pymdlsdk.IString = self.sdk.mdlFactory.get_db_module_name(qualifiedModuleName)
            if dbName.is_valid_interface():
                return dbName.get_c_str()
        self.assertTrue(False, "Code path reached that should not be reached")  # pragma: no cover

    def assertStartswith(self, text: str, prefix: str):
        self.assertIsInstance(text, str)
        self.assertTrue(text.startswith(prefix))
