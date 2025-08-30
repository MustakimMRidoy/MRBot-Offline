# Neural MR Bot Language Teacher

A self-learning AI language teaching assistant that learns from conversations and text inputs without requiring manual training.

## Features

- **Self-Learning Neural Network**: The bot learns from conversations and improves over time
- **No Predefined Data Files**: Unlike traditional chatbots, this system doesn't rely on static data files
- **Bilingual Support**: Practice both English and Bengali (বাংলা)
- **Feedback System**: Users can correct the bot's responses to help it learn
- **Export/Import Learning**: Save and load the trained neural model
- **Multiple Learning Modes**: Conversation, vocabulary, grammar, and reading practice

## Technical Architecture

### Core Components

1. **Neural Language Model** (`neural-language-model.js`)
   - TensorFlow.js-based sequence-to-sequence model
   - Tokenization and text processing
   - Training and prediction functionality
   - Model saving/loading capabilities

2. **Neural Data Manager** (`neural-data-manager.js`)
   - Manages training data from conversations and texts
   - Handles model persistence
   - User profile management
   - Import/export functionality

3. **Neural Conversation Engine** (`neural-conversation-engine.js`)
   - Processes user inputs
   - Manages conversation context
   - Handles special commands
   - Provides feedback mechanisms

4. **Application Integrator** (`app-integrator.js`)
   - Connects all components
   - Manages UI interactions
   - Handles model export/import
   - Provides user feedback interface

### How It Works

1. **Learning Process**:
   - The system starts with minimal knowledge
   - As users interact with the bot, it collects conversation pairs
   - The neural model is trained on these pairs
   - User corrections provide additional training data
   - Text inputs can be processed to extract more training examples

2. **Conversation Flow**:
   - User sends a message
   - The conversation engine processes the input
   - The neural model generates a response
   - The response is displayed to the user
   - User can provide feedback on the response
   - The model learns from the feedback

3. **Data Storage**:
   - Conversation history is stored in the browser
   - Model weights are saved using TensorFlow.js storage
   - User profile and preferences are maintained
   - All data can be exported and imported

## Getting Started

### Prerequisites

- Modern web browser with JavaScript enabled
- Internet connection (for loading TensorFlow.js)

### Installation

1. Clone this repository or download the files
2. Open `index.html` in a web browser
3. Start conversing with the bot

### Usage

- Type messages in the input field and press Send
- Use the language toggle to switch between English and Bengali
- Click on the tool buttons to access different learning modes
- Provide feedback on the bot's responses to help it learn
- Export your trained model to save progress
- Import a previously trained model to continue learning

## Commands

- `/english` or `speak english` - Switch to English
- `/bengali` or `speak bengali` - Switch to Bengali
- `/conversation` - Practice conversation
- `/vocabulary` - Learn vocabulary
- `/grammar` - Practice grammar
- `/reading` - Practice reading
- `/help` - Show help information
- `/status` - Show current status
- `/reset` - Reset conversation history

## Development

### Project Structure

```
neural-mr-bot/
├── css/
│   └── styles.css
├── js/
│   ├── neural-language-model.js
│   ├── neural-data-manager.js
│   ├── neural-conversation-engine.js
│   └── app-integrator.js
├── models/
│   └── (saved models will be stored here)
├── data/
│   └── (training data will be stored here)
├── index.html
└── README.md
```

### Extending the System

To extend the functionality:

1. **Add new learning modes**:
   - Update the `NeuralConversationEngine` class with new modes
   - Add corresponding UI elements in `index.html`

2. **Enhance the neural model**:
   - Modify the model architecture in `NeuralLanguageModel.createModel()`
   - Adjust training parameters for better performance

3. **Add more languages**:
   - Update language detection in `NeuralConversationEngine.detectLanguage()`
   - Add language toggle UI elements

## Limitations

- **Initial Learning Period**: The bot starts with minimal knowledge and improves over time
- **Browser Resources**: Training neural networks in the browser can be resource-intensive
- **Data Persistence**: Data is stored in the browser and may be lost if storage is cleared
- **Simple Model Architecture**: The current implementation uses a basic Seq2Seq model

## Future Improvements

- **Advanced Model Architecture**: Implement transformer-based models like BERT or GPT
- **Multi-language Support**: Add more languages beyond English and Bengali
- **Speech Recognition/Synthesis**: Add voice interaction capabilities
- **Offline Support**: Enable full functionality without internet connection
- **Progressive Learning**: Implement curriculum learning for better progression

## License

This project is open source and available under the MIT License.

## Acknowledgments

- TensorFlow.js team for making neural networks possible in the browser
- NinjaTech AI for the original concept and requirements