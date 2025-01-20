try:  # pragma: no cover
    # testing from within a package or CI
    from .setup import BindingModule, UnittestFrameworkBase
    pymdlsdk = BindingModule
except ImportError:  # pragma: no cover
    # local testing
    from setup import UnittestFrameworkBase
    import pymdlsdk

import warnings

# shared test helper function
class UnittestBase(UnittestFrameworkBase):
    # sdk: SDK  # declared in derived classes

    def assertNotNullOrEmpty(self, text: str, msg=None):
        self.assertIsNotNone(text, msg)
        self.assertNotEqual(text, '', msg)

    def assertZero(self, number: int, msg=None):
        self.assertEqual(number, 0, msg)

    def assertIsValidInterface(self, iinterface: pymdlsdk.IInterface, msg=None):
        self.assertIsNotNone(iinterface, msg)
        self.assertTrue(iinterface.is_valid_interface(), msg)
        ref1: int = iinterface.__iinterface_refs__()
        ptr1: int = iinterface.__iinterface_ptr_as_uint64__()
        iinterface2 = iinterface  # this does not increase interface ref count
        self.assertEqual(ref1, iinterface2.__iinterface_refs__(), msg)
        self.assertEqual(ptr1, iinterface2.__iinterface_ptr_as_uint64__(), msg)  # same iinterface
        iinterface = None  # doesn't matter which python variable is reset
        self.assertEqual(ref1, iinterface2.__iinterface_refs__(), msg)
        self.assertEqual(ptr1, iinterface2.__iinterface_ptr_as_uint64__(), msg)

        shortTime = iinterface2.get_interface(type(iinterface2))
        self.assertTrue(shortTime.is_valid_interface(), msg)
        self.assertLessEqual(ref1, iinterface2.__iinterface_refs__(), msg)
        shortTime.release()
        self.assertEqual(ref1, iinterface2.__iinterface_refs__(), msg)
        self.assertFalse(shortTime.is_valid_interface(), msg)

        with iinterface2.get_interface(pymdlsdk.IInterface) as iinterface3:
            self.assertTrue(iinterface3.is_valid_interface(), msg)
            self.assertLessEqual(ref1, iinterface3.__iinterface_refs__(), msg)
            with iinterface3.get_interface(type(iinterface2)) as iinterface4:
                self.assertTrue(iinterface4.is_valid_interface(), msg)
                self.assertLessEqual(ref1, iinterface4.__iinterface_refs__(), msg)
                iinterface4.compare_iid(type(iinterface2).IID())
                iinterface2.compare_iid(pymdlsdk.IInterface.IID())
                # note, this is not guaranteed:
                # for example when getting an IMaterial_instance from an IFunction_call, a new
                # object of type IMaterial_instance is returned. This means the get interface will
                # return a different object outside the inheritance hierarchy
                # self.assertEqual(iinterface2.get_iid(), iinterface3.get_iid())
        self.assertEqual(ref1, iinterface2.__iinterface_refs__(), msg)
        self.assertNotEqual(iinterface2.get_iid(), pymdlsdk.Uuid(), msg)

    def assertIsNotValidInterface(self, iinterface: pymdlsdk.IInterface, msg=None):
        self.assertIsNotNone(iinterface, msg)
        self.assertFalse(iinterface.is_valid_interface(), msg)

    def log_context_messages(self, context: pymdlsdk.IMdl_execution_context) -> bool:
        """print all messages from the context. Return false if there have been errors"""
        if context.get_messages_count() == 0:
            return True
        hasErrors: bool = context.get_error_messages_count() > 0
        for i in range(context.get_messages_count()):
            message: pymdlsdk.IMessage = context.get_message(i)
            kind: pymdlsdk.IMessage.Kind = message.get_kind()
            self.assertNotEqual(kind, pymdlsdk.IMessage.Kind.MSG_FORCE_32_BIT)
            level: str = ""
            if message.get_severity() == pymdlsdk.Message_severity.MESSAGE_SEVERITY_FATAL:  # pragma: no cover
                level = "fatal:"
                hasErrors = True
            elif message.get_severity() == pymdlsdk.Message_severity.MESSAGE_SEVERITY_ERROR:  # pragma: no cover
                level = "error:"
                hasErrors = True
            elif message.get_severity() == pymdlsdk.Message_severity.MESSAGE_SEVERITY_WARNING:  # pragma: no cover
                level = "warning:"
            elif message.get_severity() == pymdlsdk.Message_severity.MESSAGE_SEVERITY_INFO:  # pragma: no cover
                level = "info:"
            elif message.get_severity() == pymdlsdk.Message_severity.MESSAGE_SEVERITY_VERBOSE:  # pragma: no cover
                level = "verbose:"
            elif message.get_severity() == pymdlsdk.Message_severity.MESSAGE_SEVERITY_DEBUG:  # pragma: no cover
                level = "debug:"
            print(f"{level} {message.get_string()}")
        return not hasErrors

    def assertContextNoErrors(self, context: pymdlsdk.IMdl_execution_context, msg=None):
        self.assertTrue(self.log_context_messages(context), msg)

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

    def assertStartswith(self, text: str, prefix: str, msg=None):
        self.assertIsInstance(text, str, msg)
        self.assertTrue(text.startswith(prefix), msg)

    def assertMd5Hash(self, filename: str, hash: str):
        import hashlib
        import os
        hash_md5 = hashlib.md5()
        self.assertTrue(os.path.isfile(filename), f"File for MD5-hashing does not exist: {filename}")
        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        self.assertEqual(hash_md5.hexdigest(), hash, f"MD5-hash not matching: {filename}")

    def assertDeprecationWarning(self, func):
        with self.assertWarns(DeprecationWarning):
            return func()
        
    def assertException(self, expectedException, func):
        if expectedException == None:  # pragma: no cover
            return func()  # allow to pass none when testing 
        with self.assertRaises(expectedException):
            func()  # no return needed
