#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
image_overrides = Tree('gui_images', prefix='gui_images')

block_cipher = None


a = Analysis(['multiplex_ms_gui.py'],
             pathex=[],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,  
          [],
		  image_overrides,
          name='MultiplexMS - Companion tool',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None,
		  icon = 'gui_images/icon.ico')

info_plist = {'addition_prop': 'additional_value'}
app = BUNDLE(exe,
             name='MultiplexMS - Companion tool.app',
             icon='gui_images/icon.icns',
             bundle_identifier=None,
             info_plist=info_plist
            )
