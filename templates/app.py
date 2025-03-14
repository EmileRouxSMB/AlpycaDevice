# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# app.py - Application module
#
# Python Interface Generator for AlpycaDevice
#
# Author:   Robert B. Denny <rdenny@dc3.com> (rbd)
#
# Python Compatibility: Requires Python 3.7 or later
#
# -----------------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2022-2023 Bob Denny
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------
# Edit History:
# 19-Jan-2023   rbd Initial edit
# 24-May-2023   rbd For upgraded templates with multi-device support
# 31-May-2023   Changes for Alpaca protocol conformance, and connected
# check in templates. Corrct capitalizaton of responder class names.
# Replace dunder naming with real names from YAML. Enhance to provide
# put paraeter handling with names and conversion functions that will
# raise an exception. Major upgrade to the templates.
# 08-Nov-2023   rbd GitHub #6 to_int and to_float are gone, remove from
#               import statements from shr in module template.
# 08-Nov-2023   rbd GitHub #8 Include property Connected/
# 08-Noc-2023   rbd Change other ommon property responder classes to
#               lower case andremove redundant ().
# 08-Nov-2023   rbd GitHub #9 Corrections to avoid adding fragments
#               of on_put() to existinv correct on_put().
# 28-Nov-2023   rbd GitHub #9 Missing line and extra parenthesis in
#               class Connected. Also substitute {memname} in mod_hdr
#               to avoid naked {memhdr} in the templates.
# 14-Feb-2024   rbd Overhaul to use JSON instead of YAML, use the JSON
#               from the Omni Simulator swagger for Platform 7 changes.
# 16-Feb-2024   rbd DeviceState has template code for construction
# 13-Sep-2024   rbd Add support for enum classes within the responder
#               modules. These come from separate files not JSON.
#               GitHub issue #12
# 15-Sep-2024   rbd Fix handling of string parameters. Catch parameter
#               types not supported here.
# 15-Sep-2024   rbd Fix multiple parameter related issues for both
#               properties and methods. Add checks for incoming integer
#               values for enums, fail if integer is not one of the
#               values in the enum. This addresses GitHub issue #13
# 16-Sep-2024   rbd InvalidValueException error messages now use
#               f-strings instead of concatenation.
# 07-Jan-2025   rbd 1.1 use the new JSON input with the correct 'Id' casing for Switch.

import json
import os.path

mod_hdr = '''
# -*- coding: utf-8 -*-
#
# -----------------------------------------------------------------------------
# {devname}.py - Alpaca API responders for {Devname}
#
# Author:   Your R. Name <your@email.org> (abc)
#
# -----------------------------------------------------------------------------
# Edit History:
#   Generated by Python Interface Generator for AlpycaDevice
#
# ??-???-????   abc Initial edit

from falcon import Request, Response, HTTPBadRequest, before
from logging import Logger
from shr import PropertyResponse, MethodResponse, PreProcessRequest, \\
                StateValue, get_request_field, to_bool
from exceptions import *        # Nothing but exception classes

logger: Logger = None

# ----------------------
# MULTI-INSTANCE SUPPORT
# ----------------------
# If this is > 0 then it means that multiple devices of this type are supported.
# Each responder on_get() and on_put() is called with a devnum parameter to indicate
# which instance of the device (0-based) is being called by the client. Leave this
# set to 0 for the simple case of controlling only one instance of this device type.
#
maxdev = 0                      # Single instance

# -----------
# DEVICE INFO
# -----------
# Static metadata not subject to configuration changes
## EDIT FOR YOUR DEVICE ##
class {Devname}Metadata:
    """ Metadata describing the {Devname} Device. Edit for your device"""
    Name = 'Sample {Devname}'
    Version = '##DRIVER VERSION AS STRING##'
    Description = 'My ASCOM {Devname}'
    DeviceType = '{Devname}'
    DeviceID = '##GENERATE A NEW GUID AND PASTE HERE##' # https://guidgenerator.com/online-guid-generator.aspx
    Info = 'Alpaca Sample Device\\nImplements I{Devname}\\nASCOM Initiative'
    MaxDeviceNumber = maxdev
    InterfaceVersion = ##YOUR DEVICE INTERFACE VERSION##        # I{Devname}Vxxx

{enum_block}
# --------------------
# RESOURCE CONTROLLERS
# --------------------

@before(PreProcessRequest(maxdev))
class action:
    def on_put(self, req: Request, resp: Response, devnum: int):
        resp.text = MethodResponse(req, NotImplementedException()).json

@before(PreProcessRequest(maxdev))
class commandblind:
    def on_put(self, req: Request, resp: Response, devnum: int):
        resp.text = MethodResponse(req, NotImplementedException()).json

@before(PreProcessRequest(maxdev))
class commandbool:
    def on_put(self, req: Request, resp: Response, devnum: int):
        resp.text = MethodResponse(req, NotImplementedException()).json

@before(PreProcessRequest(maxdev))
class commandstring:
    def on_put(self, req: Request, resp: Response, devnum: int):
        resp.text = MethodResponse(req, NotImplementedException()).json

@before(PreProcessRequest(maxdev))
class connect:
    def on_put(self, req: Request, resp: Response, devnum: int):
        try:
            # ------------------------
            ### CONNECT THE DEVICE ###
            # ------------------------
            resp.text = MethodResponse(req).json
        except Exception as ex:
            resp.text = MethodResponse(req,
                            DriverException(0x500, '{Devname}.Connect failed', ex)).json

@before(PreProcessRequest(maxdev))
class connected:
    def on_get(self, req: Request, resp: Response, devnum: int):
        try:
            # -------------------------------------
            is_conn = ### READ CONN STATE ###
            # -------------------------------------
            resp.text = PropertyResponse(is_conn, req).json
        except Exception as ex:
            resp.text = MethodResponse(req, DriverException(0x500, '{Devname}.Connected failed', ex)).json

    def on_put(self, req: Request, resp: Response, devnum: int):
        conn_str = get_request_field('Connected', req)
        conn = to_bool(conn_str)              # Raises 400 Bad Request if str to bool fails

        try:
            # --------------------------------------
            ### CONNECT OR DISCONNECT THE DEVICE ###
            # --------------------------------------
            resp.text = MethodResponse(req).json
        except Exception as ex:
            resp.text = MethodResponse(req, # Put is actually like a method :-(
                            DriverException(0x500, '{Devname}.Connected failed', ex)).json

@before(PreProcessRequest(maxdev))
class connecting:
    def on_get(self, req: Request, resp: Response, devnum: int):
        try:
            # ------------------------------
            val = ## GET CONNECTING STATE ##
            # ------------------------------
            resp.text = PropertyResponse(val, req).json
        except Exception as ex:
            resp.text = PropertyResponse(None, req,
                            DriverException(0x500, '{Devname}.Connecting failed', ex)).json

@before(PreProcessRequest(maxdev))
class description:
    def on_get(self, req: Request, resp: Response, devnum: int):
        resp.text = PropertyResponse({Devname}Metadata.Description, req).json

@before(PreProcessRequest(maxdev))
class devicestate:

    def on_get(self, req: Request, resp: Response, devnum: int):
        if not ##IS DEV CONNECTED##:
            resp.text = PropertyResponse(None, req,
                            NotConnectedException()).json
            return
        try:
            # ----------------------
            val = []
            # val.append(StateValue('## NAME ##', ## GET VAL ##))
            # Repeat for each of the operational states per the device spec
            # ----------------------
            resp.text = PropertyResponse(val, req).json
        except Exception as ex:
            resp.text = PropertyResponse(None, req,
                            DriverException(0x500, '{devname}.Devicestate failed', ex)).json


class disconnect:
    def on_put(self, req: Request, resp: Response, devnum: int):
        try:
            # ---------------------------
            ### DISCONNECT THE DEVICE ###
            # ---------------------------
            resp.text = MethodResponse(req).json
        except Exception as ex:
            resp.text = MethodResponse(req,
                            DriverException(0x500, '{Devname}.Disconnect failed', ex)).json

@before(PreProcessRequest(maxdev))
class driverinfo:
    def on_get(self, req: Request, resp: Response, devnum: int):
        resp.text = PropertyResponse({Devname}Metadata.Info, req).json

@before(PreProcessRequest(maxdev))
class interfaceversion:
    def on_get(self, req: Request, resp: Response, devnum: int):
        resp.text = PropertyResponse({Devname}Metadata.InterfaceVersion, req).json

@before(PreProcessRequest(maxdev))
class driverversion():
    def on_get(self, req: Request, resp: Response, devnum: int):
        resp.text = PropertyResponse({Devname}Metadata.Version, req).json

@before(PreProcessRequest(maxdev))
class name():
    def on_get(self, req: Request, resp: Response, devnum: int):
        resp.text = PropertyResponse({Devname}Metadata.Name, req).json

@before(PreProcessRequest(maxdev))
class supportedactions:
    def on_get(self, req: Request, resp: Response, devnum: int):
        resp.text = PropertyResponse([], req).json  # Not PropertyNotImplemented

'''
cls_tmpl = '''@before(PreProcessRequest(maxdev))
class {memname}:

'''

get_tmpl = '''    def on_get(self, req: Request, resp: Response, devnum: int):
        if not ##IS DEV CONNECTED##:
            resp.text = PropertyResponse(None, req,
                            NotConnectedException()).json
            return
        {GETPARAMS}
        try:
            # ----------------------
            val = ## GET PROPERTY ##
            # ----------------------
            resp.text = PropertyResponse(val, req).json
        except Exception as ex:
            resp.text = PropertyResponse(None, req,
                            DriverException(0x500, '{Devname}.{Memname} failed', ex)).json

'''

put_tmpl = '''    def on_put(self, req: Request, resp: Response, devnum: int):
        if not ## IS DEV CONNECTED ##:
            resp.text = PropertyResponse(None, req,
                            NotConnectedException()).json
            return
        {GETPARAMS}
        try:
            # -----------------------------
            ### DEVICE OPERATION(PARAM) ###
            # -----------------------------
            resp.text = MethodResponse(req).json
        except Exception as ex:
            resp.text = MethodResponse(req,
                            DriverException(0x500, '{Devname}.{Memname} failed', ex)).json

'''

params_tmpl_str = '''
        {param} = get_request_field('{Param}', req)         # Raises 400 bad request if missing
'''

params_tmpl_cvt = '''
        {param}str = get_request_field('{Param}', req)      # Raises 400 bad request if missing
        try:
            {param} = {cvtfunc}({param}str)
        except:
            resp.text = MethodResponse(req,
                            InvalidValueException(f'{Param} {{param}str} not a valid {ptype}.')).json
            return
'''

# Only  valid for integer enums
params_tmpl_enum = '''
        {param}str = get_request_field('{Param}', req)      # Raises 400 bad request if missing
        try:
            {param} = int({param}str)
        except:
            resp.text = MethodResponse(req,
                            InvalidValueException(f'{Param} {{param}str} not a valid integer.')).json
            return
        if not {param} in {enumvals}:
            resp.text = MethodResponse(req,
                            InvalidValueException(f'{Param} {{param}} not a valid enum value.')).json
            return
'''


# Skip these common members, they are in the device main template
common_mems = ['action', 'commandblind', 'commandbool', 'commandstring', 'connect', 'connected', 'connecting',
               'devicestate', 'description', 'disconnect', 'driverinfo', 'driverversion', 'interfaceversion',
               'name', 'supportedactions']

def main():
    with open('AlpacaDeviceAPI_v2_plat7-0.4.1.json') as f:  # Has corrected 'Id' casing for Switch
        toptree = json.load(f)

    seendevs = []
    mf = None

    for path, meths in toptree['paths'].items():
        print(f'{path}')
        bits = path.split('/')
        if bits[1] == 'management' or bits[1] == 'simulator':
            continue
        devname = bits[3]
        Devname = devname.title()
        memname = bits[5]
        Memname = memname.title()
        if memname in common_mems:
            continue;
        if not devname in seendevs:
            if not mf is None and not mf.closed:
                mf.close
            mf = open(f'{devname}.py', 'w')
            temp = mod_hdr.replace('{devname}', devname)
            if os.path.exists(f'enum/{devname}_enum.py'):
                ef = open(f'enum/{devname}_enum.py')
                enumtxt = ef.read()
                ef.close()
                temp = temp.replace('{enum_block}', enumtxt)
            else:
                temp = temp.replace('{enum_block}', '')
            mf.write(temp.replace('{Devname}', Devname))
            seendevs.append(devname)
        mf.write(cls_tmpl.replace('{memname}', memname))
        for meth, meta in meths.items():
            # TODO -- Yes I know this can be refactored! TFB
            if meth == 'get':
                temp = get_tmpl.replace('{Devname}', Devname)
                temp = temp.replace('{Memname}', Memname)
                getparams = ''
                for param in meths['get']['parameters']:
                    Pname = param['name']
                    pname = Pname.lower()
                    if pname == 'clientid' or pname == 'clienttransactionid' or pname == 'devicenumber':
                        continue;
                    if '$ref' in param['schema']:
                        ref = param['schema']['$ref'].split('/')[3]     # Enum NAME
                        refdict = toptree['components']['schemas'][ref]
                        ptype = refdict['type']                         # Data type of enum
                        if ptype != 'integer':
                            raise Exception('Oops enum with non-integer values')
                        if 'enum' in refdict:
                            ptype = 'enum'                              # Enum parameter template range check
                            enumvals = refdict['enum']                  # List of valid integer values
                        else:
                            raise Exception('Oops, $ref of type other than enum')
                        #{'enum': [0, 1, 2], 'type': 'integer', 'description': 'The telescope axes', 'format': 'int32'}
                    else:
                        ptype = param['schema']['type']
                    if ptype == 'string':
                        ptemp = params_tmpl_str
                        ptemp += '        ### INTEPRET AS NEEDED OR FAIL ###  # Raise Alpaca InvalidValueException with details!'
                    elif ptype == 'boolean':
                        ptemp = params_tmpl_cvt
                        ptemp = ptemp.replace('{cvtfunc}', 'to_bool')
                    elif ptype == 'integer':
                        ptemp = params_tmpl_cvt
                        ptemp = ptemp.replace('{cvtfunc}', 'int')
                        ptemp += '        ### RANGE CHECK AS NEEDED ###  # Raise Alpaca InvalidValueException with details!'
                    elif ptype == 'number':
                        ptemp = params_tmpl_cvt
                        ptemp = ptemp.replace('{cvtfunc}', 'float')
                        ptemp += '        ### RANGE CHECK AS NEEDED ###  # Raise Alpaca InvalidValueException with details!'
                    elif ptype == 'enum':
                        ptemp = params_tmpl_enum
                        ptemp = ptemp.replace('{enumvals}', str(enumvals))
                    else:
                        raise Exception('Unsupported parameter type {ptype}')
                    ptemp = ptemp.replace('{param}', pname)      # Parameter name
                    ptemp = ptemp.replace('{Param}', Pname)
                    ptemp = ptemp.replace('{ptype}', ptype)
                    getparams += ptemp
                    #mf.write('#------ 1 ------\n')                                  # Direct RequestBody with parameters
                mf.write (temp.replace('{GETPARAMS}', getparams))
            else:
                temp = put_tmpl.replace('{Devname}', Devname)
                temp = temp.replace('{Memname}', Memname)
                getparams = ''
                if 'content' in  meths['put']['requestBody']:
                    for param in meths['put']['requestBody']['content']['multipart/form-data']['schema']['properties'].items():
                        Pname = param[0]
                        pname = Pname.lower()
                        if pname == 'clientid' or pname == 'clienttransactionid':
                            continue;
                        if '$ref' in param[1]:
                            ref = param[1]['$ref'].split('/')[3]
                            refdict = toptree['components']['schemas'][ref]
                            ptype = refdict['type']                         # Data type of enum
                            if ptype != 'integer':
                                raise Exception('Oops enum with non-integer values')
                            if 'enum' in refdict:
                                ptype = 'enum'                              # Enum parameter template range check
                                enumvals = refdict['enum']                  # List of valid integer values
                            else:
                                raise Exception('Oops, $ref of type other than enum')
                        else:
                            ptype = param[1]['type']
                        if ptype == 'string':
                            ptemp = params_tmpl_str
                            ptemp += '        ### INTEPRET AS NEEDED OR FAIL ###  # Raise Alpaca InvalidValueException with details!'
                        elif ptype == 'boolean':
                            ptemp = params_tmpl_cvt
                            ptemp = ptemp.replace('{cvtfunc}', 'to_bool')
                        elif ptype == 'integer':
                            ptemp = params_tmpl_cvt
                            ptemp = ptemp.replace('{cvtfunc}', 'int')
                            ptemp += '        ### RANGE CHECK AS NEEDED ###  # Raise Alpaca InvalidValueException with details!'
                        elif ptype == 'number':
                            ptemp = params_tmpl_cvt
                            ptemp = ptemp.replace('{cvtfunc}', 'float')
                            ptemp += '        ### RANGE CHECK AS NEEDED ###  # Raise Alpaca InvalidValueException with details!'
                        elif ptype == 'enum':
                            ptemp = params_tmpl_enum
                            ptemp = ptemp.replace('{enumvals}', str(enumvals))
                        else:
                            raise Exception('Unsupported parameter type {ptype}')
                        ptemp = ptemp.replace('{param}', pname)      # Parameter name
                        ptemp = ptemp.replace('{Param}', Pname)
                        ptemp = ptemp.replace('{ptype}', ptype)

                        getparams += ptemp
                    #mf.write('#------ 1 ------\n')                                  # Direct RequestBody with parameters
                    mf.write (temp.replace('{GETPARAMS}', getparams))

    mf.close()
    print('end')


# ========================
if __name__ == '__main__':
    main()
# ========================
