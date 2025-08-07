![tablemutant logo](/resources/icons/tablemutant/icon.png)

# TableMutant

TableMutant lets you generate new columns in datasets using LLMs. It provides both a CLI and a Toga-based GUI.

## Installation

Requires Python 3.12+.

Using uv (recommended):
```
uv sync
```

Or using pip:
```
pip install -e .
```

## Running

CLI:
```
tablemutant --model TheBloke/Llama-2-7B-GGUF/llama-2-7b.Q4_K_M.gguf \
            --table data.csv \
            --instructions "Summarize the content in columns 0 and 1"
```

GUI:
```
tablemutant --gui
```

## Briefcase packaging

This project is configured for Briefcase packaging via [tool.briefcase] in the pyproject. Reference: https://briefcase.readthedocs.io/en/stable/

### Install Briefcase

Using uv:
```
uv add --dev briefcase
```

Or pip:
```
pip install briefcase
```

### Dev run

Run your app with a managed, editable development bundle:
```
briefcase dev
```

### Create, build, package

Create application scaffolding for your platform:
```
briefcase create
```

Build an app bundle:
```
briefcase build
```

Generate a distributable installer:
```
briefcase package
```

Notes:
- Some platforms require SDKs/tools (e.g., Xcode for macOS/iOS, Android SDK for Android). See Briefcase docs.
- First runs may download toolchains; allow time and follow prompts.

## App icons (Toga / Briefcase)

App icon locations are configured in [tool.briefcase.app.tablemutant] and per-platform sections in [`toml.table()`](pyproject.toml:1).

Icon folders:
- resources/icons/tablemutant/  -> generic fallback (icon.png)
- resources/icons/macos/        -> macOS icon.icns
- resources/icons/windows/      -> Windows icon.ico
- resources/icons/linux/        -> Linux icon.png
- resources/icons/ios/          -> placeholder
- resources/icons/android/      -> placeholder

Quick-start with placeholders:
1) Place a square high-res PNG at repo root as logo.png
2) Populate placeholders:
```
python resources/icons_setup.py
```
3) Replace generated placeholders with proper per-platform assets when ready.

For detailed Toga app icon guidance, see:
- Toga docs: https://toga.readthedocs.io/en/stable/
- Briefcase icons: https://briefcase.readthedocs.io/en/stable/reference/config/app.html#icon

## Entry point

Briefcase runs the application via:
- entry_point = "tablemutant.__main__:main"

Main entry is defined in [`python.main()`](tablemutant/__main__.py:13).