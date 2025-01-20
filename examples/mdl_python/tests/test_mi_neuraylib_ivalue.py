import unittest
import os
import inspect

try:  # pragma: no cover
    # testing from within a package or CI
    from .setup import SDK, BindingModule
    from .unittest_base import UnittestBase
    pymdlsdk = BindingModule
except ImportError:  # pragma: no cover
    # local testing
    from setup import SDK
    from unittest_base import UnittestBase
    import pymdlsdk


class MainIValue(UnittestBase):
    sdk: SDK = None
    tf: pymdlsdk.IType_factory = None
    vf: pymdlsdk.IValue_factory = None
    imp_exp: pymdlsdk.IMdl_impexp_api = None

    @classmethod
    def setUpClass(self):
        print(f"Running tests in {__file__} in process with id: {os.getpid()}")
        self.sdk = SDK()
        self.sdk.load(addExampleSearchPath=True, loadImagePlugins=False, locale="de")
        self.tf = self.sdk.mdlFactory.create_type_factory(self.sdk.transaction)
        self.vf = self.sdk.mdlFactory.create_value_factory(self.sdk.transaction)
        self.imp_exp = self.sdk.neuray.get_api_component(pymdlsdk.IMdl_impexp_api)
        self.imp_exp.load_module(self.sdk.transaction, "::base")  # load something to get builtins

    @classmethod
    def tearDownClass(self):
        self.imp_exp = None
        self.vf = None
        self.tf = None
        self.sdk.unload()
        self.sdk = None
        print(f"\nFinished tests in {__file__}\n")

    def test_setupIsDone(self):
        self.assertIsNotNone(self.sdk)
        self.assertIsValidInterface(self.sdk.neuray)
        self.assertIsValidInterface(self.sdk.transaction)
        self.assertIsValidInterface(self.sdk.mdlFactory)
        self.assertIsValidInterface(self.tf)
        self.assertIsValidInterface(self.vf)

    def test_IValue_bool(self):
        tBool: pymdlsdk.IType_bool = self.tf.create_bool()
        self.assertIsNotNone(tBool)
        self.assertEqual(tBool.get_kind(), pymdlsdk.IType.Kind.TK_BOOL)
        v: pymdlsdk.IValue = self.vf.create(tBool)
        self.assertIsNotNone(v)
        vBool: pymdlsdk.IValue_bool = v.get_interface(pymdlsdk.IValue_bool)
        self.assertIsNotNone(vBool)
        self.assertEqual(vBool.get_kind(), pymdlsdk.IValue.Kind.VK_BOOL)
        self.assertEqual(vBool.get_value(), False)
        vBool.set_value(True)
        self.assertEqual(vBool.get_value(), True)
        vBool = self.vf.create_bool(False)
        self.assertIsNotNone(vBool)
        self.assertEqual(vBool.get_value(), False)
        vBool = self.vf.create_bool(True)
        self.assertIsNotNone(vBool)
        self.assertEqual(vBool.get_value(), True)
        self.assertTrue(True)
        vAtomic: pymdlsdk.IValue_atomic = vBool.get_interface(pymdlsdk.IValue_atomic)
        self.assertEqual(vAtomic.get_type().get_iid(), vBool.get_type().get_iid())
        self.assertEqual(vAtomic.get_kind(), pymdlsdk.IValue.Kind.VK_BOOL)
        self.assertEqual(tBool.get_iid(), vBool.get_type().get_iid())
        self.assertTrue(vBool.compare_iid(vBool.get_iid()))
        self.assertTrue(vBool.compare_iid(v.get_iid()))
        self.assertTrue(vBool.compare_iid(vAtomic.get_iid()))
        self.assertFalse(vBool.compare_iid(tBool.get_iid()))

    def run_modifier_tests(self, itype):
        mods: int = itype.get_all_type_modifiers()
        skipped = itype.skip_all_type_aliases()
        if mods == 0:
            self.assertEqual(itype.get_iid(), skipped.get_iid())  # equal iid
        else:
            self.assertNotEqual(itype.get_iid(), skipped.get_iid())  # equal iid

    def run_compound_test(self, compoundTypeClass, compoundValueClass, value: pymdlsdk.IValue_compound):
        valueCompound = value.get_interface(compoundValueClass)
        if not valueCompound.is_valid_interface():
            return  # allows to just try every type
        self.assertIsValidInterface(valueCompound)
        for i in range(valueCompound.get_size()):
            element: pymdlsdk.IValue = valueCompound.get_value(i)
            self.assertTrue(element.is_valid_interface())
            element2: pymdlsdk.IInterface = valueCompound.get_value_as(pymdlsdk.IInterface, i)
            self.assertTrue(element2.is_valid_interface())
            notValid: pymdlsdk.IInterface = valueCompound.get_value_as(pymdlsdk.IType, i+1337)
            self.assertFalse(notValid.is_valid_interface())

        self.assertIsInstance(value.get_kind(), pymdlsdk.IValue.Kind)
        self.assertIsInstance(valueCompound.get_kind(), pymdlsdk.IValue.Kind)
        self.assertEqual(value.get_kind(), valueCompound.get_kind())
        self.assertTrue(valueCompound.compare_iid(pymdlsdk.IValue.IID()))
        self.assertTrue(valueCompound.compare_iid(compoundValueClass.IID()))
        self.assertIsNotNone(valueCompound.get_iid())
        with value.get_interface(compoundValueClass) as valueCompound2:
            self.assertIsNotNone(valueCompound2)
            self.assertTrue(valueCompound2.is_valid_interface())
            self.assertEqual(valueCompound.get_iid(), valueCompound2.get_iid())
            component: pymdlsdk.IValue = valueCompound2.get_value(0)
            self.assertIsValidInterface(component)
            self.assertEqual(valueCompound2.set_value(0, component), 0)
        typeCompound = valueCompound.get_type()
        self.assertTrue(typeCompound.is_valid_interface())
        self.assertTrue(typeCompound.compare_iid(pymdlsdk.IType.IID()))
        self.assertTrue(typeCompound.compare_iid(compoundTypeClass.IID()))
        self.run_modifier_tests(typeCompound)
        with typeCompound.get_interface(compoundTypeClass) as typeCompound2:
            self.assertIsNotNone(typeCompound2)
            self.assertTrue(typeCompound2.is_valid_interface())
            self.assertEqual(typeCompound.get_iid(), typeCompound2.get_iid())

    def run_ivalue_test(self, ivalueClass, valueTyped: pymdlsdk.IValue):
        self.assertIsValidInterface(valueTyped)
        value: pymdlsdk.IValue = None
        for funcName, funcObj in inspect.getmembers(ivalueClass):
            if funcName == "get_interface":
                value = funcObj(valueTyped, pymdlsdk.IValue)
        self.assertIsValidInterface(value)
        with value.get_interface(ivalueClass) as value2:
            self.assertIsNotNone(value2)
            self.assertTrue(value2.is_valid_interface())
            self.assertEqual(valueTyped.get_iid(), value2.get_iid())
        pythonType = type(valueTyped)
        castedBack = value.get_interface(pythonType)
        self.assertEqual(castedBack.get_iid(), valueTyped.get_iid())
        self.assertEqual(value.get_kind(), valueTyped.get_kind())
        self.assertEqual(value.get_iid(), valueTyped.get_iid())
        self.assertEqual(value.get_type().get_iid(), valueTyped.get_type().get_iid())
        self.assertIsInstance(value.get_kind(), pymdlsdk.IValue.Kind)
        self.assertIsInstance(valueTyped.get_kind(), pymdlsdk.IValue.Kind)
        self.run_compound_test(pymdlsdk.IType_compound, pymdlsdk.IValue_compound, value)
        self.run_compound_test(pymdlsdk.IType_vector, pymdlsdk.IValue_vector, value)
        self.run_compound_test(pymdlsdk.IType_matrix, pymdlsdk.IValue_matrix, value)
        self.run_compound_test(pymdlsdk.IType_structure, pymdlsdk.IValue_structure, value)
        self.run_compound_test(pymdlsdk.IType_color, pymdlsdk.IValue_color, value)
        self.run_compound_test(pymdlsdk.IType_array, pymdlsdk.IValue_array, value)
        asAtomic: pymdlsdk.IValue_atomic = value.get_interface(pymdlsdk.IValue_atomic)
        if asAtomic.is_valid_interface():
            v = valueTyped.get_value()
            valueTyped.set_value(v)
        if value.get_kind() == pymdlsdk.IValue.Kind.VK_ENUM:
            asEnumeration: pymdlsdk.IValue_enumeration = value.get_interface(pymdlsdk.IValue_enumeration)
            self.assertIsValidInterface(asEnumeration)
            asEnumeration.set_value(1)
            asEnumeration.set_index(asEnumeration.get_index())
            asEnumeration.set_name(asEnumeration.get_name())
            asEnumeration.set_value(0)


    def run_itype_test(self, itypeClass, ivalueClass, typeTyped: pymdlsdk.IType, symbol: str = ""):
        self.assertIsValidInterface(typeTyped)
        itype: pymdlsdk.IType = None
        for funcName, funcObj in inspect.getmembers(itypeClass):
            if funcName == "get_interface":
                itype = funcObj(typeTyped, pymdlsdk.IType)
        self.assertIsNotNone(itype)
        with itype.get_interface(itypeClass) as itype2:
            self.assertIsNotNone(itype2)
            self.assertTrue(itype2.is_valid_interface())
            self.assertEqual(typeTyped.get_iid(), itype2.get_iid())
        self.assertTrue(itype.is_valid_interface())
        self.assertIsInstance(itype.is_declarative(), bool)
        self.assertIsInstance(typeTyped.is_declarative(), bool)
        self.run_modifier_tests(typeTyped)
        self.run_modifier_tests(itype)
        mod: int = itype.get_all_type_modifiers()
        pythonType = type(typeTyped)
        castedBack = itype.get_interface(pythonType)
        self.assertEqual(castedBack.get_iid(), typeTyped.get_iid())
        self.assertEqual(itype.get_kind(), typeTyped.get_kind())
        self.assertEqual(itype.get_iid(), typeTyped.get_iid())
        self.assertTrue(typeTyped.compare_iid(pymdlsdk.IType.IID()))
        self.assertTrue(typeTyped.compare_iid(itype.get_iid()))
        self.assertIsInstance(itype.get_kind(), pymdlsdk.IType.Kind)
        self.assertIsInstance(typeTyped.get_kind(), pymdlsdk.IType.Kind)
        typeDf: pymdlsdk.IType_df = itype.get_interface(pymdlsdk.IType_df)
        if typeDf.is_valid_interface():
            self.assertIsInstance(typeDf.is_declarative(), bool)
            self.run_modifier_tests(typeDf)
        if itype.get_kind() == pymdlsdk.IType.Kind.TK_BSDF:
            self.assertIsValidInterface(typeDf)
            self.assertIsInstance(typeDf.get_kind(), pymdlsdk.IType.Kind)
            self.assertEqual(typeDf.get_kind(), pymdlsdk.IType.Kind.TK_BSDF)
            return
        if itype.get_kind() == pymdlsdk.IType.Kind.TK_EDF:
            return
        if itype.get_kind() == pymdlsdk.IType.Kind.TK_VDF:
            return
        if itype.get_kind() == pymdlsdk.IType.Kind.TK_HAIR_BSDF:
            return
        if itype.get_kind() == pymdlsdk.IType.Kind.TK_VECTOR:
            elementType: pymdlsdk.IType = typeTyped.get_element_type()
            atomicType: pymdlsdk.IType_atomic = elementType.get_interface(pymdlsdk.IType_atomic)
            self.assertTrue(atomicType.is_valid_interface())
            for i in range(typeTyped.get_size()):
                compType: pymdlsdk.IType = typeTyped.get_component_type(i)
                self.assertEqual(compType.get_kind(), elementType.get_kind())
        if itypeClass == pymdlsdk.IType_matrix:
            elementType: pymdlsdk.IType = typeTyped.get_element_type()
            columnType: pymdlsdk.IType_vector = elementType.get_interface(pymdlsdk.IType_vector)
            self.assertTrue(columnType.is_valid_interface())
        if itypeClass == pymdlsdk.IType_texture:
            self.assertEqual(typeTyped.get_shape(), pymdlsdk.IType_texture.Shape.TS_3D)
        if itypeClass == pymdlsdk.IType_structure:
            self.assertEqual(typeTyped.get_symbol(), symbol)
            if symbol == "::material":
                self.assertEqual(typeTyped.get_predefined_id(), pymdlsdk.IType_structure.Predefined_id.SID_MATERIAL)
                self.assertTrue(typeTyped.is_declarative())
                category: pymdlsdk.IStruct_category = typeTyped.get_struct_category()
                self.assertIsValidInterface(category)
                cid: pymdlsdk.IStruct_category.Predefined_id = category.get_predefined_id()
                self.assertEqual(cid, pymdlsdk.IStruct_category.Predefined_id.CID_MATERIAL_CATEGORY)
                cat_symbol: str = category.get_symbol()
                self.assertEqual(cat_symbol, "::material_category")
                annoBlock: pymdlsdk.IAnnotation_block = category.get_annotations()
                self.assertIsNotNone(annoBlock)
            if symbol == "::material_surface":
                self.assertEqual(typeTyped.get_predefined_id(), pymdlsdk.IType_structure.Predefined_id.SID_MATERIAL_SURFACE)
                self.assertTrue(typeTyped.is_declarative())
                category: pymdlsdk.IStruct_category = typeTyped.get_struct_category()
                self.assertFalse(category.is_valid_interface())
            for i in range(typeTyped.get_size()):
                fieldName: str = typeTyped.get_field_name(i)
                self.assertEqual(typeTyped.find_field(fieldName), i)
                annoBlock: pymdlsdk.IAnnotation_block = typeTyped.get_field_annotations(i)
                self.assertIsNotNone(annoBlock)
            fieldType: pymdlsdk.IType = typeTyped.get_field_type(1)
            self.assertEqual(fieldType.get_kind(), pymdlsdk.IType.Kind.TK_STRUCT)
            annoBlock: pymdlsdk.IAnnotation_block = typeTyped.get_annotations()
            self.assertIsNotNone(annoBlock)
        if itypeClass == pymdlsdk.IType_enumeration:
            self.assertEqual(typeTyped.get_symbol(), "::intensity_mode")
            self.assertEqual(typeTyped.get_predefined_id(), pymdlsdk.IType_enumeration.Predefined_id.EID_INTENSITY_MODE)
            for i in range(typeTyped.get_size()):
                name: str = typeTyped.get_value_name(i)
                code: int = typeTyped.get_value_code(i)
                self.assertEqual(typeTyped.find_value(name), i)
                self.assertEqual(typeTyped.find_value(code), i)
                return_code: pymdlsdk.ReturnCode = pymdlsdk.ReturnCode()
                code2: int = typeTyped.get_value_code(i, return_code)
                self.assertEqual(code, code2)
                self.assertZero(return_code.value)
                code3, err = self.assertDeprecationWarning(lambda: typeTyped.get_value_code_with_ret(i))
                self.assertEqual(code, code3)
                self.assertZero(err)
                annoBlock: pymdlsdk.IAnnotation_block = typeTyped.get_value_annotations(i)
                self.assertIsNotNone(annoBlock)
            annoBlock: pymdlsdk.IAnnotation_block = typeTyped.get_annotations()
            self.assertIsNotNone(annoBlock)
        isDeferredSizedArray: bool = False
        if itypeClass == pymdlsdk.IType_array:
            elementType: pymdlsdk.IType = typeTyped.get_element_type()
            self.assertIsValidInterface(elementType)
            if typeTyped.is_immediate_sized():
                self.assertEqual(typeTyped.get_size(), 10)
            else:
                self.assertEqual(typeTyped.get_deferred_size(), "X")
                isDeferredSizedArray = True
        asCompound: pymdlsdk.IType_compound = typeTyped.get_interface(pymdlsdk.IType_compound)
        if asCompound.is_valid_interface():
            self.assertIsValidInterface(asCompound)
            self.assertEqual(asCompound.get_kind(), typeTyped.get_kind())
            self.assertEqual(typeTyped.is_declarative(), asCompound.is_declarative())
            size: int = asCompound.get_size()
            self.assertEqual(size, typeTyped.get_size())
            if isDeferredSizedArray:
                self.assertEqual(size, -1)
            else:
                self.assertGreater(size, 0)
            fieldType: pymdlsdk.IType = asCompound.get_component_type(0)
            self.assertEqual(fieldType.get_iid(), typeTyped.get_component_type(0).get_iid())
            self.assertIsValidInterface(fieldType)
        asAtomic: pymdlsdk.IType_atomic = typeTyped.get_interface(pymdlsdk.IType_atomic)
        if asAtomic.is_valid_interface():
            self.assertIsValidInterface(asAtomic)
            self.run_modifier_tests(asAtomic)
            self.assertFalse(asAtomic.is_declarative())
        asResource: pymdlsdk.IType_resource = typeTyped.get_interface(pymdlsdk.IType_resource)
        if asResource.is_valid_interface():
            self.assertIsValidInterface(asResource)
            self.run_modifier_tests(asResource)
            self.assertTrue(asResource.get_kind() == pymdlsdk.IType.Kind.TK_TEXTURE or
                            asResource.get_kind() == pymdlsdk.IType.Kind.TK_LIGHT_PROFILE or
                            asResource.get_kind() == pymdlsdk.IType.Kind.TK_BSDF_MEASUREMENT)
            self.assertFalse(asResource.is_declarative())
        asReference: pymdlsdk.IType_reference = typeTyped.get_interface(pymdlsdk.IType_reference)
        if asReference.is_valid_interface():
            self.assertIsValidInterface(asReference)
            self.run_modifier_tests(asReference)
            self.assertFalse(asReference.is_declarative())
            self.assertTrue(asReference.get_kind() == pymdlsdk.IType.Kind.TK_TEXTURE or
                            asReference.get_kind() == pymdlsdk.IType.Kind.TK_LIGHT_PROFILE or
                            asReference.get_kind() == pymdlsdk.IType.Kind.TK_BSDF_MEASUREMENT or
                            asReference.get_kind() == pymdlsdk.IType.Kind.TK_BSDF or
                            asReference.get_kind() == pymdlsdk.IType.Kind.TK_EDF or
                            asReference.get_kind() == pymdlsdk.IType.Kind.TK_HAIR_BSDF or
                            asReference.get_kind() == pymdlsdk.IType.Kind.TK_VDF)

        # create a value and run tests on it
        if not isDeferredSizedArray:
            ivalue: pymdlsdk.IValue = self.vf.create(typeTyped)
            ivalueTyped = ivalue.get_interface(ivalueClass)
            self.run_ivalue_test(ivalueClass, ivalueTyped)

    def test_istruct_categories(self):
        material_cat: pymdlsdk.IStruct_category = self.tf.create_struct_category("::material_category")
        material_cat2: pymdlsdk.IStruct_category = self.tf.get_predefined_struct_category(pymdlsdk.IStruct_category.Predefined_id.CID_MATERIAL_CATEGORY)
        self.assertIsValidInterface(material_cat)
        self.assertIsValidInterface(material_cat2)
        self.assertEqual(self.tf.compare(material_cat, material_cat2), 0)
        cat_list_1: pymdlsdk.IStruct_category_list = self.tf.create_struct_category_list()
        cat_list_1.add_struct_category("a", material_cat)
        cat_list_2: pymdlsdk.IStruct_category_list = self.tf.create_struct_category_list()
        self.assertEqual(cat_list_1.get_size(), 1)
        self.assertEqual(cat_list_1.get_index("a"), 0)
        self.assertEqual(cat_list_1.get_name(0), "a")
        self.assertEqual(self.tf.compare(cat_list_1.get_struct_category(0), material_cat), 0)
        self.assertNotEqual(self.tf.compare(cat_list_1, cat_list_2), 0)
        cat_list_2.add_struct_category("a", material_cat2)
        cat_list_2.set_struct_category("a", material_cat)
        cat_list_2.set_struct_category(0, material_cat)
        self.assertEqual(self.tf.compare(cat_list_1, cat_list_2), 0)
        cat_list_2.add_struct_category("b", material_cat2)
        cat_list_3: pymdlsdk.IStruct_category_list = self.tf.clone(cat_list_2)
        self.assertEqual(self.tf.compare(cat_list_2, cat_list_3), 0)
        self.assertIsValidInterface(cat_list_3)
        self.assertEqual(cat_list_3.get_size(), 2)
        self.assertEqual(self.tf.dump(material_cat).get_c_str(), '"::material_category"')
        self.assertEqual(self.tf.dump(cat_list_3).get_c_str(), 'struct_category_list [\n    0: a = "::material_category"\n    1: b = "::material_category"\n]')
        a: pymdlsdk.IType_structure = self.tf.create_struct("::material")
        b: pymdlsdk.IType_structure = self.tf.get_predefined_struct(pymdlsdk.IType_structure.Predefined_id.SID_MATERIAL)
        c: pymdlsdk.IType_structure = self.tf.create_struct("::material_surface")
        self.assertEqual(self.tf.from_same_struct_category(a, b), 1)
        self.assertEqual(self.tf.from_same_struct_category(a, c), -1)

    def test_ivalues(self):
        tBool: pymdlsdk.IType_bool = self.tf.create_bool()
        tInt: pymdlsdk.IType_int = self.tf.create_int()
        tFloat: pymdlsdk.IType_float = self.tf.create_float()
        tFloat4: pymdlsdk.IType_vector = self.tf.create_vector(tFloat, 4)
        self.run_itype_test(pymdlsdk.IType_vector, pymdlsdk.IValue_vector, tFloat4)
        self.run_itype_test(pymdlsdk.IType_matrix, pymdlsdk.IValue_matrix, self.tf.create_matrix(tFloat4, 4))
        tDouble: pymdlsdk.IType_double = self.tf.create_double()
        self.run_itype_test(pymdlsdk.IType_bool, pymdlsdk.IValue_bool, tBool)
        self.run_itype_test(pymdlsdk.IType_int, pymdlsdk.IValue_int, tInt)
        self.run_itype_test(pymdlsdk.IType_float, pymdlsdk.IValue_float, tFloat)
        self.run_itype_test(pymdlsdk.IType_double, pymdlsdk.IValue_double, tDouble)
        self.run_itype_test(pymdlsdk.IType_string, pymdlsdk.IValue_string, self.tf.create_string())
        self.run_itype_test(pymdlsdk.IType_color, pymdlsdk.IValue_color, self.tf.create_color())
        self.run_itype_test(pymdlsdk.IType_texture, pymdlsdk.IValue_texture, self.tf.create_texture(pymdlsdk.IType_texture.Shape.TS_3D))
        self.run_itype_test(pymdlsdk.IType_light_profile, pymdlsdk.IValue_light_profile, self.tf.create_light_profile())
        self.run_itype_test(pymdlsdk.IType_bsdf_measurement, pymdlsdk.IValue_bsdf_measurement, self.tf.create_bsdf_measurement())
        self.run_itype_test(pymdlsdk.IType_enumeration, pymdlsdk.IValue_enumeration, self.tf.create_enum("::intensity_mode"), symbol="::intensity_mode")
        self.run_itype_test(pymdlsdk.IType_structure, pymdlsdk.IValue_structure, self.tf.create_struct("::material_surface"), symbol="::material_surface")
        self.run_itype_test(pymdlsdk.IType_structure, pymdlsdk.IValue_structure, self.tf.create_struct("::material"), symbol="::material")
        self.run_itype_test(pymdlsdk.IType_enumeration, pymdlsdk.IValue_enumeration, self.tf.get_predefined_enum(pymdlsdk.IType_enumeration.Predefined_id.EID_INTENSITY_MODE), symbol="::intensity_mode")
        self.run_itype_test(pymdlsdk.IType_structure, pymdlsdk.IValue_structure, self.tf.get_predefined_struct(pymdlsdk.IType_structure.Predefined_id.SID_MATERIAL_SURFACE), symbol="::material_surface")
        self.run_itype_test(pymdlsdk.IType_array, pymdlsdk.IValue_array, self.tf.create_immediate_sized_array(tInt, 10))
        self.run_itype_test(pymdlsdk.IType_array, pymdlsdk.IValue_array, self.tf.create_deferred_sized_array(tInt, "X"))
        tbsdf: pymdlsdk.IType_bsdf = self.tf.create_bsdf()
        self.run_itype_test(pymdlsdk.IType_bsdf, None, tbsdf)
        tedf: pymdlsdk.IType_edf = self.tf.create_edf()
        self.run_itype_test(pymdlsdk.IType_edf, None, tedf)
        tvdf: pymdlsdk.IType_vdf = self.tf.create_vdf()
        self.run_itype_test(pymdlsdk.IType_vdf, None, tvdf)
        thair: pymdlsdk.IType_hair_bsdf = self.tf.create_hair_bsdf()
        self.run_itype_test(pymdlsdk.IType_hair_bsdf, None, thair)
        tList: pymdlsdk.IType_list = self.tf.create_type_list()
        tList.add_type("xyz", tedf)
        tList.set_type("xyz", tvdf)
        self.assertEqual(tList.get_index("xyz"), 0)
        self.assertEqual(tList.get_name(0), "xyz")

    def test_ivalues_factory(self):
        tf: pymdlsdk.IType_factory = self.vf.get_type_factory()
        self.assertIsValidInterface(tf)
        vBool: pymdlsdk.IValue_bool = self.vf.create_bool(True)
        self.assertIsValidInterface(vBool)
        self.assertTrue(vBool.get_value())
        vInt: pymdlsdk.IValue_int = self.vf.create_int(42)
        self.assertIsValidInterface(vInt)
        self.assertEqual(vInt.get_value(), 42)
        vInt2: pymdlsdk.IValue_int = self.vf.clone(vInt).get_interface(pymdlsdk.IValue_int)
        self.assertEqual(vInt2.get_value(), 42)
        self.assertZero(self.vf.compare(vInt, vInt2))
        valueDumped: str = self.vf.dump(vInt, "answer").get_c_str()
        self.assertEqual(valueDumped, 'int answer = 42')
        tIntensityMode: pymdlsdk.IType_enumeration = tf.create_enum("::intensity_mode")
        vEnum: pymdlsdk.IValue_enumeration = self.vf.create_enum(tIntensityMode, 1)
        self.assertIsValidInterface(vEnum)
        self.assertEqual(vEnum.get_name(), "intensity_power")
        vFloat: pymdlsdk.IValue_float = self.vf.create_float(1.23)
        self.assertIsValidInterface(vFloat)
        self.assertAlmostEqual(vFloat.get_value(), 1.23)
        vDouble: pymdlsdk.IValue_double = self.vf.create_double(2.34)
        self.assertIsValidInterface(vDouble)
        self.assertAlmostEqual(vDouble.get_value(), 2.34)
        vString: pymdlsdk.IValue_string = self.vf.create_string("MDL")
        self.assertIsValidInterface(vString)
        self.assertEqual(vString.get_value(), "MDL")
        tFloat: pymdlsdk.IType_float = self.tf.create_float()
        tFloat2: pymdlsdk.IType_vector = self.tf.create_vector(tFloat, 2)
        vVector: pymdlsdk.IValue_vector = self.vf.create_vector(tFloat2)
        self.assertIsValidInterface(vVector)
        self.assertEqual(vVector.get_size(), 2)
        tFloat23: pymdlsdk.IType_vector = self.tf.create_matrix(tFloat2, 3)
        vMatrix: pymdlsdk.IValue_matrix = self.vf.create_matrix(tFloat23)
        self.assertIsValidInterface(vMatrix)
        self.assertEqual(vMatrix.get_size(), 3)
        tMaterial: pymdlsdk.IType_structure = self.tf.get_predefined_struct(pymdlsdk.IType_structure.Predefined_id.SID_MATERIAL)
        vMaterial: pymdlsdk.IValue_structure = self.vf.create_struct(tMaterial)
        self.assertIsValidInterface(vMaterial)
        vMaterialThinWalled: pymdlsdk.IValue = vMaterial.get_field("thin_walled")
        self.assertIsValidInterface(vMaterialThinWalled)
        self.assertEqual(vMaterialThinWalled.get_kind(), pymdlsdk.IValue.Kind.VK_BOOL)
        vMaterialThinWalled2: pymdlsdk.IValue_bool = vMaterial.get_field_as(pymdlsdk.IValue_bool, "thin_walled")
        self.assertIsValidInterface(vMaterialThinWalled2)
        self.assertZero(vMaterial.set_field("thin_walled", vBool))
        vNotExisting: pymdlsdk.IValue_color = vMaterial.get_field_as(pymdlsdk.IValue_color, "NOT_EXISTING")
        self.assertIsNotValidInterface(vNotExisting)
        tArray: pymdlsdk.IType_array = self.tf.create_immediate_sized_array(tFloat2, 10)
        vArray: pymdlsdk.IValue_array = self.vf.create_array(tArray)
        self.assertIsValidInterface(vArray)
        self.assertEqual(vArray.get_size(), 10)
        tArrayDeferredSized: pymdlsdk.IType_array = self.tf.create_deferred_sized_array(tFloat2, "N")
        vArrayDeferredSized: pymdlsdk.IValue_array = self.vf.create_array(tArrayDeferredSized)
        self.assertIsValidInterface(vArrayDeferredSized)
        self.assertZero(vArrayDeferredSized.set_size(10))
        self.assertEqual(vArrayDeferredSized.get_size(), 10)
        tbsdf: pymdlsdk.IType_bsdf = self.tf.create_bsdf()
        vInvalidDf: pymdlsdk.IValue_invalid_df = self.vf.create_invalid_df(tbsdf)
        self.assertIsValidInterface(vInvalidDf)
        self.assertEqual(vInvalidDf.get_type().get_iid(), tbsdf.get_iid())
        self.assertEqual(vInvalidDf.get_kind(), pymdlsdk.IValue.Kind.VK_INVALID_DF)
        vList: pymdlsdk.IValue_list = self.vf.create_value_list()
        self.assertIsValidInterface(vList)
        vList.add_value("a", vInt)
        vList.add_value("b", vEnum)
        vList.set_value("b", vString)
        self.assertEqual(vList.get_index("b"), 1)
        self.assertEqual(vList.get_name(0), "a")
        b: pymdlsdk.IValue_string = vList.get_value("b").get_interface(pymdlsdk.IValue_string)
        self.assertIsValidInterface(b)
        self.assertEqual(b.get_value(), "MDL")
        vList2: pymdlsdk.IValue_list = self.vf.clone(vList).get_interface(pymdlsdk.IValue_list)
        self.assertEqual(vList2.get_size(), 2)
        self.assertZero(self.vf.compare(vList, vList2))

    def test_itype_factory(self):
        tColor: pymdlsdk.IType_color = self.tf.create_color()
        tUniformColor: pymdlsdk.IType_alias = self.tf.create_alias(tColor, pymdlsdk.IType.Modifier.MK_UNIFORM.value, None)
        self.assertIsValidInterface(tUniformColor)
        self.run_modifier_tests(tUniformColor)
        self.assertEqual(tUniformColor.get_symbol(), None)
        self.assertEqual(tUniformColor.get_aliased_type().get_iid(), tColor.get_iid())
        self.assertEqual(tUniformColor.get_type_modifiers(), pymdlsdk.IType.Modifier.MK_UNIFORM.value)
        self.assertFalse(tUniformColor.is_declarative())
        self.assertEqual(tUniformColor.get_kind(), pymdlsdk.IType.Kind.TK_ALIAS)
        tFloat3: pymdlsdk.IType = self.tf.create_from_mdl_type_name("float3")
        self.assertIsValidInterface(tFloat3)
        tFloat3: pymdlsdk.IType_vector = tFloat3.get_interface(pymdlsdk.IType_vector)
        self.assertEqual(tFloat3.get_element_type().get_kind(), pymdlsdk.IType.Kind.TK_FLOAT)
        self.assertEqual(tFloat3.get_size(), 3)
        tMaterial: pymdlsdk.IType_structure = self.tf.get_predefined_struct(pymdlsdk.IType_structure.Predefined_id.SID_MATERIAL)
        self.assertEqual(self.tf.is_compatible(tMaterial, tMaterial), 1)
        builtin: str = self.tf.get_mdl_module_name(tMaterial).get_c_str()
        self.assertEqual(builtin, '::%3Cbuiltins%3E')
        self.assertEqual(self.tf.get_mdl_type_name(tFloat3).get_c_str(), "float3")

    def test_string_localized(self):
        i18n: pymdlsdk.IMdl_i18n_configuration = self.sdk.neuray.get_api_component(pymdlsdk.IMdl_i18n_configuration)
        self.assertIsValidInterface(i18n)
        self.assertEqual(i18n.get_locale(), "de")
        systemKeyword: str = i18n.get_system_keyword()
        systemLocale: str = i18n.get_system_locale()

        dbName: str = self.load_module("::nvidia::core_definitions")
        self.assertNotNullOrEmpty(dbName)
        mod: pymdlsdk.IModule = self.sdk.transaction.access_as(pymdlsdk.IModule, dbName)
        self.assertIsValidInterface(mod)
        overloads: pymdlsdk.IArray = mod.get_function_overloads("worley_noise_texture")
        self.assertIsValidInterface(overloads)
        self.assertEqual(overloads.get_length(), 1)
        dbName: pymdlsdk.IString = overloads.get_element_as(pymdlsdk.IString, 0)
        self.assertNotNullOrEmpty(dbName.get_c_str())
        funcDef: pymdlsdk.IFunction_definition = self.sdk.transaction.access_as(pymdlsdk.IFunction_definition, dbName.get_c_str())
        self.assertIsValidInterface(funcDef)
        annoBlock: pymdlsdk.IAnnotation_block = funcDef.get_annotations()
        for i in range(annoBlock.get_size()):
            anno: pymdlsdk.IAnnotation = annoBlock.get_annotation(i)
            name: str = anno.get_name()
            if name == "::anno::description(string)":
                args: pymdlsdk.IExpression_list = anno.get_arguments()
                argExpr: pymdlsdk.IExpression_constant = args.get_expression_as(pymdlsdk.IExpression_constant, "description")
                self.assertIsValidInterface(argExpr)
                argValue: pymdlsdk.IValue = argExpr.get_value()
                argValueStr: pymdlsdk.IValue_string = argValue.get_interface(pymdlsdk.IValue_string)
                argValueStrDe: pymdlsdk.IValue_string_localized = argValue.get_interface(pymdlsdk.IValue_string_localized)
                self.assertIsValidInterface(argValueStr)
                self.assertIsValidInterface(argValueStrDe)
                self.assertEqual(argValueStr.get_value(), argValueStrDe.get_value())
                self.assertEqual(argValueStrDe.get_value(), "Erzeugt eine Textur mit wabenartigem Muster")
                self.assertEqual(argValueStrDe.get_original_value(), "Allow texturing with a cell forming pattern")
                self.assertEqual(argValueStrDe.get_kind(), pymdlsdk.IValue.Kind.VK_STRING)
                t: pymdlsdk.IType = argValueStrDe.get_type()
                self.assertEqual(t.get_kind(), pymdlsdk.IType.Kind.TK_STRING)
                vf: pymdlsdk.IValue_factory = self.sdk.mdlFactory.create_value_factory(self.sdk.transaction)
                copy: pymdlsdk.IValue = vf.clone(argValueStrDe)
                copyValueStrDe: pymdlsdk.IValue_string_localized = copy.get_interface(pymdlsdk.IValue_string_localized)
                copyValueStrDe.set_value("TEST")
                self.assertEqual(copyValueStrDe.get_value(), "TEST")


# run all tests of this file
if __name__ == '__main__':
    unittest.main()  # pragma: no cover
