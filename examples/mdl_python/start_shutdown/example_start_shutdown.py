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
# Entry Point
#--------------------------------------------------------------------------------------------------

def main():

    # Get the INeuray interface in a suitable smart pointer.
    neuray = pymdlsdk.load_and_get_ineuray('')
    if not neuray.is_valid_interface():
        raise Exception('Failed to load the MDL SDK.')
    print('Loaded the MDL SDK.')

    # configuration settings go here, none in this example

    # after the configuration is done, start neuray.
    resultCode = neuray.start()
    if resultCode != 0:
        raise Exception('Failed to initialize the SDK. Result code: ' + resultCode)

    print('Started the MDL SDK. Status: ' + str(neuray.get_status()))

    # scene graph manipulations and rendering calls go here, none in this example.
    # ...

    # Shutting the MDL SDK down. Again, a return code of 0 indicates success.
    resultCode = neuray.shutdown()
    if resultCode != 0:
        raise Exception('Failed to shutdown the SDK. Result code: ' + resultCode)
    print('Shutdown the MDL SDK.')

    # make sure the object is disposed.
    neuray = None
    # alternatively, neuray as well as other pymdlsdk objects can be created using a context
    # manager by wrapping them in with-statements, e.g.:
    #
    # with pymdlsdk.INeuray.load_and_get_ineuray() as neuray:
    #     # do something with neuray

    # before shutting down, we might have objects not collected yet by the garbage collection
    # this can lead to crashes as shutdown can potentially make C++ objects invalid without
    # Python noticing it. Using context managers or setting objects to 'None' should make
    # this unnecessary.
    # gc.collect()

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
