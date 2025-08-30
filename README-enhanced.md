# Neural MR Bot - Enhanced Version

This is an enhanced version of the Neural MR Bot language teaching assistant with significant improvements to the architecture, capabilities, and user experience.

## Key Improvements

### 1. Transformer-Based Model Architecture
- Replaced the simple LSTM-based Seq2Seq model with a more powerful Transformer architecture
- Implemented attention mechanisms for better context understanding
- Added positional encoding for sequence awareness
- Improved tokenization with subword support for better language handling
- Enhanced response generation with beam search decoding

### 2. Multi-Language Support
- Expanded beyond English and Bengali to support multiple languages
- Replaced language toggle with a dropdown menu for language selection
- Implemented dynamic language storage in the data manager
- Added subword tokenization for better cross-language support
- Included language-specific fonts for proper text display

### 3. Speech Recognition/Synthesis
- Added voice input capabilities using Web Speech API
- Implemented text-to-speech for listening to bot responses
- Added language detection and appropriate voice selection
- Included UI elements for microphone and speaker controls
- Enhanced accessibility with voice interaction

### 4. Offline Support
- Implemented service worker for offline functionality
- Downloaded and stored TensorFlow.js locally
- Generated static Tailwind CSS file
- Downloaded and stored Bengali fonts locally
- Added PWA support with manifest file
- Created a download script for required resources

### 5. Progressive Learning (Curriculum Learning)
- Added difficulty level and topic tagging to training data
- Modified trainModel function to implement curriculum learning
- Created performance tracking metrics for adaptive learning
- Implemented logic to progress through difficulty levels based on performance
- Enhanced data manager to support metadata for curriculum learning

## File Structure

- `neural-language-model-transformer.js`: Enhanced model with transformer architecture
- `neural-data-manager-enhanced.js`: Improved data manager with multi-language support
- `app-integrator-enhanced.js`: Updated app integrator with speech features
- `index-enhanced.html`: Updated UI with language dropdown and speech controls
- `service-worker.js`: Service worker for offline functionality
- `manifest.json`: PWA manifest file
- `css/output.css`: Static Tailwind CSS file for offline support
- `fonts/bangla-fonts.css`: Bengali font definitions
- `download-resources.sh`: Script to download required resources

## Usage

1. Run the download script to get required resources:
   ```
   chmod +x download-resources.sh
   ./download-resources.sh
   ```

2. Open `index-enhanced.html` in a browser to use the enhanced version.

3. Select your preferred language from the dropdown menu.

4. Use the microphone button to speak your input or type in the text box.

5. Use the speaker button to listen to the bot's responses.

## Offline Usage

The app is designed to work offline after the initial load. The service worker caches all necessary resources, allowing you to use the app without an internet connection.

## Progressive Learning

The bot adapts to your skill level over time. As you interact with it, it will:

1. Start with beginner-level content
2. Track your performance and understanding
3. Gradually introduce more complex language concepts
4. Adjust difficulty based on your feedback

## Technical Details

### Transformer Architecture
- Multi-head attention mechanism
- Positional encoding for sequence awareness
- Feed-forward neural networks
- Layer normalization and residual connections
- Beam search decoding for better responses

### Subword Tokenization
- Handles unknown words better than word-level tokenization
- Works across multiple languages
- Improves vocabulary efficiency
- Better handles morphologically rich languages

### Web Speech API Integration
- SpeechRecognition for voice input
- SpeechSynthesis for voice output
- Language-specific voice selection
- Real-time speech processing

### Service Worker
- Caches app resources for offline use
- Intercepts network requests
- Serves cached content when offline
- Updates cache when online

## Browser Compatibility

The enhanced version works best in modern browsers that support:
- Web Speech API
- Service Workers
- IndexedDB
- ES6+ JavaScript features

## License

This project is licensed under the MIT License - see the LICENSE file for details.