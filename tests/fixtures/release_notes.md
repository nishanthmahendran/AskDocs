# Lumen Notes 4.2 Release Notes

This is a fictional document written for the AskDocs test suite. Lumen Notes
is an imaginary note-taking app.

## New features

- **Command palette.** Press Ctrl+Shift+P (Cmd+Shift+P on macOS) to open the
  new command palette and run any action by typing its name.
- **PDF export.** Choose File > Export > PDF to save the current note as a PDF.
  Images and tables are included; embedded videos are replaced by a link.
- **Dark theme schedule.** The dark theme can now switch on automatically at
  sunset.

## Changes

- Sync now runs every five minutes instead of every fifteen minutes.
- The maximum attachment size was raised from 25 MB to 100 MB.

## Removed and deprecated

- Windows 8.1 is no longer supported. Lumen Notes 4.2 requires Windows 10 or
  later, or macOS 12 or later.
- The legacy plain-text sync format was removed. Notes stored in that format
  are converted automatically the first time 4.2 starts.

## Fixed

- Fixed a crash when pasting very large tables.
- Search now finds words with accents, such as "café", when you type "cafe".
