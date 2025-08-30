/**
 * MR Bot Neural Language Teacher - Neural Conversation Engine
 * 
 * This module provides a simplified conversation engine that uses the neural language model
 * instead of rule-based logic. It handles user interactions and manages the conversation flow.
 */

class NeuralConversationEngine {
    constructor() {
        this.languageModel = null;
        this.dataManager = null;
        this.initialized = false;
        
        // Conversation state
        this.context = {
            language: 'en',
            targetLanguage: 'bn',
            userLevel: 'beginner',
            history: [],
            currentTopic: null,
            learningFocus: null
        };
        
        // Learning modes
        this.learningModes = {
            CONVERSATION: 'conversation',
            VOCABULARY: 'vocabulary',
            GRAMMAR: 'grammar',
            READING: 'reading',
            WRITING: 'writing',
            PRONUNCIATION: 'pronunciation'
        };
        
        this.currentMode = this.learningModes.CONVERSATION;
    }

    /**
     * Initialize the conversation engine
     * @param {Object} languageModel - The neural language model instance
     * @param {Object} dataManager - The data manager instance
     */
    async initialize(languageModel, dataManager) {
        console.log("Initializing Neural Conversation Engine...");
        
        if (!languageModel || !dataManager) {
            console.error("Language model and data manager are required");
            return false;
        }
        
        this.languageModel = languageModel;
        this.dataManager = dataManager;
        
        // Load user profile
        const userProfile = this.dataManager.getUserProfile();
        if (userProfile) {
            this.context.language = userProfile.nativeLanguage || 'en';
            this.context.targetLanguage = userProfile.targetLanguage || 'bn';
            this.context.userLevel = userProfile.proficiencyLevel || 'beginner';
        }
        
        this.initialized = true;
        console.log("Neural Conversation Engine initialized successfully");
        return true;
    }

    /**
     * Process user input and generate a response
     * @param {string} userInput - The user's input text
     * @returns {Object} - Response object with text and metadata
     */
    async processInput(userInput) {
        if (!this.initialized || !this.languageModel) {
            console.error("Conversation engine not initialized. Call initialize() first.");
            return {
                text: "I'm still initializing. Please try again in a moment.",
                metadata: { status: 'initializing' }
            };
        }
        
        console.log(`Processing user input: "${userInput}"`);
        
        try {
            // Add to conversation history
            this.context.history.push({
                role: 'user',
                text: userInput,
                timestamp: new Date().toISOString()
            });
            
            // Detect commands
            const commandResponse = this.processCommands(userInput);
            if (commandResponse) {
                return commandResponse;
            }
            
            // Generate response using the neural model
            const response = await this.languageModel.generateResponse(userInput);
            
            // Add response to conversation history
            this.context.history.push({
                role: 'bot',
                text: response,
                timestamp: new Date().toISOString()
            });
            
            // Limit history size
            if (this.context.history.length > 50) {
                this.context.history.shift();
            }
            
            // Add to training data
            await this.dataManager.addConversation(userInput, response);
            
            // Return response with metadata
            return {
                text: response,
                metadata: {
                    status: 'success',
                    mode: this.currentMode,
                    language: this.detectLanguage(response),
                    learningFocus: this.context.learningFocus
                }
            };
        } catch (error) {
            console.error("Error processing input:", error);
            return {
                text: "I'm having trouble processing your request. Please try again.",
                metadata: { status: 'error', error: error.message }
            };
        }
    }

    /**
     * Process special commands in user input
     * @param {string} userInput - The user's input text
     * @returns {Object|null} - Response object if command detected, null otherwise
     */
    processCommands(userInput) {
        const input = userInput.toLowerCase().trim();
        
        // Language switching commands
        if (input === '/english' || input === 'speak english') {
            this.context.language = 'en';
            return {
                text: "I'll speak English now.",
                metadata: { status: 'command', command: 'switchLanguage', language: 'en' }
            };
        }
        
        if (input === '/bengali' || input === 'speak bengali' || input === 'speak bangla') {
            this.context.language = 'bn';
            return {
                text: "আমি এখন বাংলায় কথা বলব।",
                metadata: { status: 'command', command: 'switchLanguage', language: 'bn' }
            };
        }
        
        // Learning mode commands
        if (input === '/conversation' || input === 'practice conversation') {
            this.currentMode = this.learningModes.CONVERSATION;
            return {
                text: "Let's practice conversation! What would you like to talk about?",
                metadata: { status: 'command', command: 'switchMode', mode: this.currentMode }
            };
        }
        
        if (input === '/vocabulary' || input === 'learn vocabulary') {
            this.currentMode = this.learningModes.VOCABULARY;
            return {
                text: "Let's focus on vocabulary. I'll help you learn new words.",
                metadata: { status: 'command', command: 'switchMode', mode: this.currentMode }
            };
        }
        
        if (input === '/grammar' || input === 'learn grammar') {
            this.currentMode = this.learningModes.GRAMMAR;
            return {
                text: "Let's work on grammar. I'll help you understand sentence structures.",
                metadata: { status: 'command', command: 'switchMode', mode: this.currentMode }
            };
        }
        
        if (input === '/reading' || input === 'practice reading') {
            this.currentMode = this.learningModes.READING;
            return {
                text: "Let's practice reading. I'll provide texts for you to read and understand.",
                metadata: { status: 'command', command: 'switchMode', mode: this.currentMode }
            };
        }
        
        // Help command
        if (input === '/help' || input === 'help') {
            return {
                text: this.getHelpText(),
                metadata: { status: 'command', command: 'help' }
            };
        }
        
        // Status command
        if (input === '/status' || input === 'status') {
            return {
                text: this.getStatusText(),
                metadata: { status: 'command', command: 'status' }
            };
        }
        
        // Reset command
        if (input === '/reset' || input === 'reset') {
            this.context.history = [];
            return {
                text: "Conversation history has been reset.",
                metadata: { status: 'command', command: 'reset' }
            };
        }
        
        // No command detected
        return null;
    }

    /**
     * Get help text with available commands
     * @returns {string} - Help text
     */
    getHelpText() {
        return `
# MR Bot Neural Language Teacher - Help

## Language Commands
- **/english** or **speak english** - Switch to English
- **/bengali** or **speak bengali** - Switch to Bengali

## Learning Mode Commands
- **/conversation** - Practice conversation
- **/vocabulary** - Learn new vocabulary
- **/grammar** - Practice grammar
- **/reading** - Practice reading
- **/writing** - Practice writing

## Other Commands
- **/help** - Show this help
- **/status** - Show current status
- **/reset** - Reset conversation history

## Learning Features
- I learn from our conversations
- You can teach me by correcting my responses
- I can help you practice both English and Bengali
- The more we talk, the better I get at helping you!
`;
    }

    /**
     * Get current status text
     * @returns {string} - Status text
     */
    getStatusText() {
        const trainingMetrics = this.languageModel.trainingMetrics;
        
        return `
# Current Status

- **Current language**: ${this.context.language === 'en' ? 'English' : 'Bengali'}
- **Target language**: ${this.context.targetLanguage === 'en' ? 'English' : 'Bengali'}
- **Learning mode**: ${this.currentMode}
- **User level**: ${this.context.userLevel}
- **Conversation history**: ${this.context.history.length} messages

## Learning Progress
- **Training sessions**: ${trainingMetrics.totalTrainingSessions}
- **Examples learned**: ${trainingMetrics.totalExamples}
- **Last training**: ${trainingMetrics.lastTrainingTime || 'Never'}
- **Current accuracy**: ${(trainingMetrics.accuracy * 100).toFixed(2)}%
`;
    }

    /**
     * Submit user feedback for a response
     * @param {string} userInput - The original user input
     * @param {string} botResponse - The bot's response
     * @param {string} correctedResponse - The user's correction
     */
    async submitFeedback(userInput, botResponse, correctedResponse) {
        if (!this.initialized) {
            console.error("Conversation engine not initialized. Call initialize() first.");
            return false;
        }
        
        try {
            // Record feedback
            await this.languageModel.recordFeedback(userInput, botResponse, correctedResponse);
            
            // Update conversation history
            const lastBotIndex = this.context.history.findIndex(
                msg => msg.role === 'bot' && msg.text === botResponse
            );
            
            if (lastBotIndex !== -1) {
                this.context.history[lastBotIndex].text = correctedResponse;
                this.context.history[lastBotIndex].corrected = true;
            }
            
            return true;
        } catch (error) {
            console.error("Error submitting feedback:", error);
            return false;
        }
    }

    /**
     * Detect the language of a text (simplified)
     * @param {string} text - The text to analyze
     * @returns {string} - Detected language code ('en' or 'bn')
     */
    detectLanguage(text) {
        // Simple language detection based on character codes
        // Bengali Unicode range: 0980-09FF
        let bengaliCount = 0;
        let englishCount = 0;
        
        for (let i = 0; i < text.length; i++) {
            const charCode = text.charCodeAt(i);
            if (charCode >= 0x0980 && charCode <= 0x09FF) {
                bengaliCount++;
            } else if ((charCode >= 65 && charCode <= 90) || (charCode >= 97 && charCode <= 122)) {
                englishCount++;
            }
        }
        
        return bengaliCount > englishCount ? 'bn' : 'en';
    }

    /**
     * Get conversation history
     * @returns {Array} - Conversation history
     */
    getConversationHistory() {
        return this.context.history;
    }

    /**
     * Set user level
     * @param {string} level - User proficiency level
     */
    setUserLevel(level) {
        this.context.userLevel = level;
        
        // Update user profile
        this.dataManager.updateUserProfile({
            proficiencyLevel: level
        });
    }

    /**
     * Set target language
     * @param {string} language - Target language code
     */
    setTargetLanguage(language) {
        this.context.targetLanguage = language;
        
        // Update user profile
        this.dataManager.updateUserProfile({
            targetLanguage: language
        });
    }

    /**
     * Set learning focus
     * @param {string} focus - Learning focus area
     */
    setLearningFocus(focus) {
        this.context.learningFocus = focus;
    }
}

// Export the class for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { NeuralConversationEngine };
} else {
    // For browser use
    window.NeuralConversationEngine = NeuralConversationEngine;
}