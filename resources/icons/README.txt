This directory contains app icon assets for Briefcase packaging.

Icon pipeline:
- Base artwork: place a high-res square PNG (at least 1024x1024) in base/logo.png
- Platform-specific folders contain correctly named files that Briefcase expects.
- You can replace any placeholder with your final artwork; keep filenames.

Folders:
- resources/icons/tablemutant/   -> Generic fallback (icon.png)
- resources/icons/macos/         -> macOS ICNS (icon.icns). If not available, Briefcase will rasterize from PNG, but ICNS is recommended.
- resources/icons/windows/       -> Windows ICO (icon.ico). If not available, Briefcase will rasterize from PNG, but ICO is recommended.
- resources/icons/linux/         -> PNG (icon.png). Multiple sizes can be added if desired.
- resources/icons/ios/           -> iOS AppIcon.appiconset (not fully provisioned here)
- resources/icons/android/       -> Android mipmap set (not fully provisioned here)

Quick start:
1) Replace each placeholder with final artwork while preserving filenames.
2) For best results:
   - macOS: export icon.icns with required sizes (16,32,64,128,256,512,1024 and @2x variants)
   - Windows: export icon.ico with sizes (16,24,32,48,64,128,256)
   - Linux: provide at least a 256x256 PNG as icon.png
   - iOS/Android: leave as-is until mobile targets are needed.