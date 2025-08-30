#!/bin/bash

# Script to package the enhanced Neural MR Bot

# Create a directory for the enhanced version
mkdir -p neural-mr-bot-enhanced

# Copy all the enhanced files
cp neural-language-model-transformer.js neural-mr-bot-enhanced/neural-language-model.js
cp neural-data-manager-enhanced.js neural-mr-bot-enhanced/neural-data-manager.js
cp app-integrator-enhanced.js neural-mr-bot-enhanced/app-integrator.js
cp index-enhanced.html neural-mr-bot-enhanced/index.html
cp README-enhanced.md neural-mr-bot-enhanced/README.md
cp service-worker.js neural-mr-bot-enhanced/service-worker.js
cp manifest.json neural-mr-bot-enhanced/manifest.json
cp download-resources.sh neural-mr-bot-enhanced/download-resources.sh

# Copy existing files that don't need changes
cp neural-conversation-engine.js neural-mr-bot-enhanced/
cp js/demo.js neural-mr-bot-enhanced/
cp js/test.js neural-mr-bot-enhanced/
cp css/styles.css neural-mr-bot-enhanced/

# Create necessary directories
mkdir -p neural-mr-bot-enhanced/css
mkdir -p neural-mr-bot-enhanced/js
mkdir -p neural-mr-bot-enhanced/js/lib
mkdir -p neural-mr-bot-enhanced/fonts
mkdir -p neural-mr-bot-enhanced/icons
mkdir -p neural-mr-bot-enhanced/data
mkdir -p neural-mr-bot-enhanced/models

# Copy enhanced CSS
cp css/output.css neural-mr-bot-enhanced/css/
cp css/styles.css neural-mr-bot-enhanced/css/

# Copy font CSS
cp fonts/bangla-fonts.css neural-mr-bot-enhanced/fonts/

# Create a README for the js/lib directory
echo "# JavaScript Libraries

This directory contains JavaScript libraries used by the application:

- tf.min.js: TensorFlow.js library for neural network operations

These files should be downloaded using the download-resources.sh script." > neural-mr-bot-enhanced/js/lib/README.md

# Create a README for the fonts directory
echo "# Font Files

This directory should contain font files for proper text display:

- NotoSansBengali-Regular.woff2
- NotoSansBengali-Regular.woff
- NotoSansBengali-Bold.woff2
- NotoSansBengali-Bold.woff

These files should be downloaded using the download-resources.sh script." > neural-mr-bot-enhanced/fonts/README.md

# Create a README for the icons directory
echo "# PWA Icons

This directory should contain icons for the Progressive Web App:

- icon-72x72.png
- icon-96x96.png
- icon-128x128.png
- icon-144x144.png
- icon-152x152.png
- icon-192x192.png
- icon-384x384.png
- icon-512x512.png

These files should be generated using the download-resources.sh script." > neural-mr-bot-enhanced/icons/README.md

# Create zip file
cd neural-mr-bot-enhanced
zip -r ../neural-mr-bot-enhanced.zip *
cd ..

echo "Package created: neural-mr-bot-enhanced.zip"