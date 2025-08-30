/**
 * MR Bot Neural Language Teacher - Enhanced Neural Data Manager
 * 
 * This module manages data for the neural language model, including:
 * - Storing and retrieving conversation history
 * - Managing training data with metadata for curriculum learning
 * - Supporting multiple languages dynamically
 * - Handling model persistence
 * - Importing and exporting data
 */

class NeuralDataManager {
    constructor() {
        this.initialized = false;
        
        // Storage for training data with enhanced structure
        this.trainingData = {
            conversations: [],
            texts: {},  // Changed from {en: [], bn: []} to dynamic object
            feedback: []
        };
        
        // Storage for model data
        this.modelData = {
            tokenizer: null,
            weights: null,
            config: null
        };
        
        // User profile data with expanded language support
        this.userProfile = {
            nativeLanguage: 'en',
            targetLanguage: 'bn',
            proficiencyLevel: 'beginner',
            learningGoals: [],
            learningHistory: [],
            supportedLanguages: ['en', 'bn', 'hi', 'es', 'fr', 'de', 'ja', 'zh', 'ru', 'ar']  // Added more languages
        };
        
        // Storage keys
        this.storageKeys = {
            trainingData: 'mr-bot-training-data',
            modelData: 'mr-bot-model-data',
            userProfile: 'mr-bot-user-profile',
            conversationHistory: 'mr-bot-conversation-history'
        };
    }

    /**
     * Initialize the data manager
     */
    async initialize() {
        console.log("Initializing Enhanced Neural Data Manager...");
        
        try {
            // Load data from storage
            await this.loadFromStorage();
            
            // Initialize empty arrays for all supported languages
            this.userProfile.supportedLanguages.forEach(lang => {
                if (!this.trainingData.texts[lang]) {
                    this.trainingData.texts[lang] = [];
                }
            });
            
            this.initialized = true;
            console.log("Enhanced Neural Data Manager initialized successfully");
            return true;
        } catch (error) {
            console.error("Error initializing data manager:", error);
            return false;
        }
    }

    /**
     * Load data from storage (localStorage in browser, simulated here)
     */
    async loadFromStorage() {
        console.log("Loading data from storage...");
        
        try {
            // In a browser environment, we would use localStorage or IndexedDB
            // For our implementation, we'll simulate loading from storage
            
            // Check if we have data in localStorage
            const loadData = (key) => {
                // In a real implementation, this would be:
                // const data = localStorage.getItem(key);
                // return data ? JSON.parse(data) : null;
                
                // For simulation, we'll return null to indicate no saved data
                return null;
            };
            
            // Load training data
            const trainingData = loadData(this.storageKeys.trainingData);
            if (trainingData) {
                this.trainingData = trainingData;
                
                // Ensure all supported languages have an array
                this.userProfile.supportedLanguages.forEach(lang => {
                    if (!this.trainingData.texts[lang]) {
                        this.trainingData.texts[lang] = [];
                    }
                });
                
                console.log(`Loaded ${this.trainingData.conversations.length} conversations and texts in ${Object.keys(this.trainingData.texts).length} languages`);
            }
            
            // Load model data
            const modelData = loadData(this.storageKeys.modelData);
            if (modelData) {
                this.modelData = modelData;
                console.log("Loaded model data");
            }
            
            // Load user profile
            const userProfile = loadData(this.storageKeys.userProfile);
            if (userProfile) {
                // Merge with default profile to ensure all fields exist
                this.userProfile = {...this.userProfile, ...userProfile};
                console.log("Loaded user profile");
            }
            
            return true;
        } catch (error) {
            console.error("Error loading from storage:", error);
            return false;
        }
    }

    /**
     * Save data to storage
     * @param {string} key - The storage key
     * @param {Object} data - The data to save
     */
    async saveToStorage(key, data) {
        console.log(`Saving data to storage: ${key}`);
        
        try {
            // In a browser environment, we would use localStorage or IndexedDB
            // For our implementation, we'll simulate saving to storage
            
            // In a real implementation, this would be:
            // localStorage.setItem(key, JSON.stringify(data));
            
            console.log(`Data saved to ${key}`);
            return true;
        } catch (error) {
            console.error(`Error saving to storage (${key}):`, error);
            return false;
        }
    }

    /**
     * Add a conversation to the training data with metadata
     * @param {string} userInput - The user's input
     * @param {string} botResponse - The bot's response
     * @param {Object} metadata - Additional metadata (language, difficulty, etc.)
     */
    async addConversation(userInput, botResponse, metadata = {}) {
        if (!this.initialized) {
            console.error("Data manager not initialized. Call initialize() first.");
            return false;
        }
        
        try {
            // Add to conversations with metadata
            this.trainingData.conversations.push({
                input: userInput,
                response: botResponse,
                timestamp: new Date().toISOString(),
                language: metadata.language || 'en',
                difficulty: metadata.difficulty || 'beginner',
                topic: metadata.topic || 'general'
            });
            
            // Save to storage
            await this.saveToStorage(this.storageKeys.trainingData, this.trainingData);
            
            return true;
        } catch (error) {
            console.error("Error adding conversation:", error);
            return false;
        }
    }

    /**
     * Add text for training with metadata
     * @param {string} text - The text to add
     * @param {string} language - The language of the text
     * @param {Object} metadata - Additional metadata (difficulty, topic, etc.)
     */
    async addTrainingText(text, language, metadata = {}) {
        if (!this.initialized) {
            console.error("Data manager not initialized. Call initialize() first.");
            return false;
        }
        
        if (!text || !language || text.trim().length === 0) {
            console.error("Invalid text or language");
            return false;
        }
        
        try {
            // Add to texts
            if (!this.trainingData.texts[language]) {
                this.trainingData.texts[language] = [];
            }
            
            this.trainingData.texts[language].push({
                content: text,
                timestamp: new Date().toISOString(),
                difficulty: metadata.difficulty || 'beginner',
                topic: metadata.topic || 'general',
                source: metadata.source || 'user'
            });
            
            // Save to storage
            await this.saveToStorage(this.storageKeys.trainingData, this.trainingData);
            
            return true;
        } catch (error) {
            console.error("Error adding training text:", error);
            return false;
        }
    }

    /**
     * Add user feedback with metadata
     * @param {string} userInput - The user's input
     * @param {string} botResponse - The bot's response
     * @param {string} correctResponse - The correct response provided by the user
     * @param {Object} metadata - Additional metadata
     */
    async addFeedback(userInput, botResponse, correctResponse, metadata = {}) {
        if (!this.initialized) {
            console.error("Data manager not initialized. Call initialize() first.");
            return false;
        }
        
        try {
            // Add to feedback
            this.trainingData.feedback.push({
                input: userInput,
                botResponse: botResponse,
                correctResponse: correctResponse,
                timestamp: new Date().toISOString(),
                language: metadata.language || this.userProfile.targetLanguage,
                difficulty: metadata.difficulty || this.userProfile.proficiencyLevel,
                topic: metadata.topic || 'general'
            });
            
            // Save to storage
            await this.saveToStorage(this.storageKeys.trainingData, this.trainingData);
            
            return true;
        } catch (error) {
            console.error("Error adding feedback:", error);
            return false;
        }
    }

    /**
     * Get all training data for the model
     * @returns {Array} - Array of text strings for training
     */
    async getTrainingData() {
        if (!this.initialized) {
            console.error("Data manager not initialized. Call initialize() first.");
            return [];
        }
        
        try {
            const trainingTexts = [];
            
            // Add conversation inputs and responses
            for (const conv of this.trainingData.conversations) {
                trainingTexts.push(conv.input);
                trainingTexts.push(conv.response);
            }
            
            // Add texts from all languages
            for (const lang in this.trainingData.texts) {
                for (const textObj of this.trainingData.texts[lang]) {
                    // Split text into sentences
                    const sentences = textObj.content.split(/[.!?]+/).filter(s => s.trim().length > 0);
                    trainingTexts.push(...sentences.map(s => s.trim()));
                }
            }
            
            // Add feedback
            for (const feedback of this.trainingData.feedback) {
                trainingTexts.push(feedback.input);
                trainingTexts.push(feedback.correctResponse);
            }
            
            return trainingTexts;
        } catch (error) {
            console.error("Error getting training data:", error);
            return [];
        }
    }

    /**
     * Get conversation pairs for training with metadata
     * @param {Object} options - Options for filtering data
     * @returns {Array} - Array of [input, output, metadata] triples
     */
    async getConversationPairs(options = {}) {
        if (!this.initialized) {
            console.error("Data manager not initialized. Call initialize() first.");
            return [];
        }
        
        try {
            const pairs = [];
            
            // Add conversation pairs with metadata
            for (const conv of this.trainingData.conversations) {
                // Filter by language if specified
                if (options.language && conv.language !== options.language) {
                    continue;
                }
                
                pairs.push([
                    conv.input, 
                    conv.response, 
                    {
                        language: conv.language || 'en',
                        difficulty: conv.difficulty || 'beginner',
                        topic: conv.topic || 'general',
                        timestamp: conv.timestamp
                    }
                ]);
            }
            
            // Add feedback pairs (input to correct response) with metadata
            for (const feedback of this.trainingData.feedback) {
                // Filter by language if specified
                if (options.language && feedback.language !== options.language) {
                    continue;
                }
                
                pairs.push([
                    feedback.input, 
                    feedback.correctResponse,
                    {
                        language: feedback.language || 'en',
                        difficulty: feedback.difficulty || 'beginner',
                        topic: feedback.topic || 'general',
                        timestamp: feedback.timestamp,
                        isFeedback: true
                    }
                ]);
            }
            
            // Add text pairs (each sentence with the next one) with metadata
            for (const lang in this.trainingData.texts) {
                // Filter by language if specified
                if (options.language && lang !== options.language) {
                    continue;
                }
                
                for (const textObj of this.trainingData.texts[lang]) {
                    const sentences = textObj.content.split(/[.!?]+/).filter(s => s.trim().length > 0);
                    for (let i = 0; i < sentences.length - 1; i++) {
                        pairs.push([
                            sentences[i].trim(), 
                            sentences[i + 1].trim(),
                            {
                                language: lang,
                                difficulty: textObj.difficulty || 'beginner',
                                topic: textObj.topic || 'general',
                                timestamp: textObj.timestamp
                            }
                        ]);
                    }
                }
            }
            
            return pairs;
        } catch (error) {
            console.error("Error getting conversation pairs:", error);
            return [];
        }
    }

    /**
     * Get training data for a specific language
     * @param {string} language - The language code
     * @returns {Array} - Array of text objects for the specified language
     */
    async getLanguageData(language) {
        if (!this.initialized) {
            console.error("Data manager not initialized. Call initialize() first.");
            return [];
        }
        
        try {
            if (!this.trainingData.texts[language]) {
                return [];
            }
            
            return this.trainingData.texts[language];
        } catch (error) {
            console.error(`Error getting data for language ${language}:`, error);
            return [];
        }
    }

    /**
     * Get all supported languages
     * @returns {Array} - Array of language codes
     */
    getSupportedLanguages() {
        return this.userProfile.supportedLanguages;
    }

    /**
     * Add a new supported language
     * @param {string} languageCode - The language code to add
     * @returns {boolean} - Success status
     */
    async addSupportedLanguage(languageCode) {
        if (!languageCode || typeof languageCode !== 'string' || languageCode.length < 2) {
            console.error("Invalid language code");
            return false;
        }
        
        try {
            // Check if language is already supported
            if (!this.userProfile.supportedLanguages.includes(languageCode)) {
                this.userProfile.supportedLanguages.push(languageCode);
                
                // Initialize empty array for the new language
                if (!this.trainingData.texts[languageCode]) {
                    this.trainingData.texts[languageCode] = [];
                }
                
                // Save to storage
                await this.saveToStorage(this.storageKeys.userProfile, this.userProfile);
                
                console.log(`Added support for language: ${languageCode}`);
                return true;
            }
            
            return false;
        } catch (error) {
            console.error(`Error adding language ${languageCode}:`, error);
            return false;
        }
    }

    /**
     * Save tokenizer data
     * @param {Object} tokenizerData - The tokenizer data to save
     */
    async saveTokenizerData(tokenizerData) {
        if (!this.initialized) {
            console.error("Data manager not initialized. Call initialize() first.");
            return false;
        }
        
        try {
            this.modelData.tokenizer = tokenizerData;
            await this.saveToStorage(this.storageKeys.modelData, this.modelData);
            return true;
        } catch (error) {
            console.error("Error saving tokenizer data:", error);
            return false;
        }
    }

    /**
     * Get tokenizer data
     * @returns {Object|null} - The tokenizer data or null if not found
     */
    async getTokenizerData() {
        if (!this.initialized) {
            console.error("Data manager not initialized. Call initialize() first.");
            return null;
        }
        
        return this.modelData.tokenizer;
    }

    /**
     * Save model weights
     * @param {Object} weights - The model weights to save
     */
    async saveModelWeights(weights) {
        if (!this.initialized) {
            console.error("Data manager not initialized. Call initialize() first.");
            return false;
        }
        
        try {
            this.modelData.weights = weights;
            await this.saveToStorage(this.storageKeys.modelData, this.modelData);
            return true;
        } catch (error) {
            console.error("Error saving model weights:", error);
            return false;
        }
    }

    /**
     * Get model weights
     * @returns {Object|null} - The model weights or null if not found
     */
    async getModelWeights() {
        if (!this.initialized) {
            console.error("Data manager not initialized. Call initialize() first.");
            return null;
        }
        
        return this.modelData.weights;
    }

    /**
     * Save model configuration
     * @param {Object} config - The model configuration to save
     */
    async saveModelConfig(config) {
        if (!this.initialized) {
            console.error("Data manager not initialized. Call initialize() first.");
            return false;
        }
        
        try {
            this.modelData.config = config;
            await this.saveToStorage(this.storageKeys.modelData, this.modelData);
            return true;
        } catch (error) {
            console.error("Error saving model config:", error);
            return false;
        }
    }

    /**
     * Get model configuration
     * @returns {Object|null} - The model configuration or null if not found
     */
    async getModelConfig() {
        if (!this.initialized) {
            console.error("Data manager not initialized. Call initialize() first.");
            return null;
        }
        
        return this.modelData.config;
    }

    /**
     * Update user profile
     * @param {Object} profileData - The profile data to update
     */
    async updateUserProfile(profileData) {
        if (!this.initialized) {
            console.error("Data manager not initialized. Call initialize() first.");
            return false;
        }
        
        try {
            this.userProfile = { ...this.userProfile, ...profileData };
            await this.saveToStorage(this.storageKeys.userProfile, this.userProfile);
            return true;
        } catch (error) {
            console.error("Error updating user profile:", error);
            return false;
        }
    }

    /**
     * Get user profile
     * @returns {Object} - The user profile
     */
    getUserProfile() {
        return this.userProfile;
    }

    /**
     * Export all data for backup or transfer
     * @returns {Object} - All data
     */
    async exportAllData() {
        if (!this.initialized) {
            console.error("Data manager not initialized. Call initialize() first.");
            return null;
        }
        
        try {
            const exportData = {
                trainingData: this.trainingData,
                modelData: this.modelData,
                userProfile: this.userProfile,
                exportDate: new Date().toISOString(),
                version: '2.0.0'  // Updated version
            };
            
            return exportData;
        } catch (error) {
            console.error("Error exporting data:", error);
            return null;
        }
    }

    /**
     * Import data from a previous export
     * @param {Object} importData - The data to import
     */
    async importAllData(importData) {
        try {
            if (!importData || !importData.version) {
                throw new Error("Invalid import data");
            }
            
            // Import training data
            if (importData.trainingData) {
                this.trainingData = importData.trainingData;
                
                // Ensure all supported languages have an array
                this.userProfile.supportedLanguages.forEach(lang => {
                    if (!this.trainingData.texts[lang]) {
                        this.trainingData.texts[lang] = [];
                    }
                });
            }
            
            // Import model data
            if (importData.modelData) {
                this.modelData = importData.modelData;
            }
            
            // Import user profile
            if (importData.userProfile) {
                // Merge with default profile to ensure all fields exist
                this.userProfile = {...this.userProfile, ...importData.userProfile};
            }
            
            // Save all to storage
            await this.saveToStorage(this.storageKeys.trainingData, this.trainingData);
            await this.saveToStorage(this.storageKeys.modelData, this.modelData);
            await this.saveToStorage(this.storageKeys.userProfile, this.userProfile);
            
            this.initialized = true;
            console.log("Data imported successfully");
            
            return true;
        } catch (error) {
            console.error("Error importing data:", error);
            return false;
        }
    }

    /**
     * Clear all data (for testing or reset)
     */
    async clearAllData() {
        try {
            // Reset training data with support for multiple languages
            this.trainingData = {
                conversations: [],
                texts: {},
                feedback: []
            };
            
            // Initialize empty arrays for all supported languages
            this.userProfile.supportedLanguages.forEach(lang => {
                this.trainingData.texts[lang] = [];
            });
            
            // Reset model data
            this.modelData = {
                tokenizer: null,
                weights: null,
                config: null
            };
            
            // Reset user profile but keep supported languages
            const supportedLanguages = [...this.userProfile.supportedLanguages];
            this.userProfile = {
                nativeLanguage: 'en',
                targetLanguage: 'bn',
                proficiencyLevel: 'beginner',
                learningGoals: [],
                learningHistory: [],
                supportedLanguages: supportedLanguages
            };
            
            // Clear storage
            // In a real implementation, this would be:
            // localStorage.removeItem(this.storageKeys.trainingData);
            // localStorage.removeItem(this.storageKeys.modelData);
            // localStorage.removeItem(this.storageKeys.userProfile);
            // localStorage.removeItem(this.storageKeys.conversationHistory);
            
            console.log("All data cleared");
            return true;
        } catch (error) {
            console.error("Error clearing data:", error);
            return false;
        }
    }
}

// Export the class for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { NeuralDataManager };
} else {
    // For browser use
    window.NeuralDataManager = NeuralDataManager;
}