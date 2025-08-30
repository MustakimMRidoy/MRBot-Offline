#!/bin/bash

# Script to download resources for offline support
# This script downloads TensorFlow.js and Bengali fonts

# Create directories if they don't exist
mkdir -p js/lib
mkdir -p fonts

echo "Downloading TensorFlow.js..."
curl -L https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.18.0/dist/tf.min.js -o js/lib/tf.min.js

echo "Downloading Bengali fonts..."
# Download Noto Sans Bengali fonts
curl -L https://fonts.gstatic.com/s/notosansbengali/v20/Cn-SJsCGWQxOjaGwMQ6fIiMywrNJIky6nvd8.woff2 -o fonts/NotoSansBengali-Regular.woff2
curl -L https://fonts.gstatic.com/s/notosansbengali/v20/Cn-SJsCGWQxOjaGwMQ6fIiMywrNJIky6nvd8.woff -o fonts/NotoSansBengali-Regular.woff
curl -L https://fonts.gstatic.com/s/notosansbengali/v20/Cn-TJsCGWQxOjaGwMQ6fIiMywrNJIky6nvd8.woff2 -o fonts/NotoSansBengali-Bold.woff2
curl -L https://fonts.gstatic.com/s/notosansbengali/v20/Cn-TJsCGWQxOjaGwMQ6fIiMywrNJIky6nvd8.woff -o fonts/NotoSansBengali-Bold.woff

echo "Creating placeholder icons for PWA..."
mkdir -p icons

# Generate placeholder icons using ImageMagick if available
if command -v convert &> /dev/null; then
    echo "Using ImageMagick to generate icons..."
    
    # Generate icons of different sizes
    for size in 72 96 128 144 152 192 384 512; do
        convert -size ${size}x${size} xc:#5D5CDE -fill white -gravity center -font Arial -pointsize $(($size/3)) -annotate 0 "MR" icons/icon-${size}x${size}.png
    done
else
    echo "ImageMagick not found. Please manually create icons in the icons directory."
    echo "You need icons of sizes: 72x72, 96x96, 128x128, 144x144, 152x152, 192x192, 384x384, 512x512"
fi

echo "Done! Resources downloaded for offline support."