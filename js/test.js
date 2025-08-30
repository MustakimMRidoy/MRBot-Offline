/**
 * Neural MR Bot Language Teacher - Test Script
 * 
 * This script runs basic tests to verify the functionality of the neural language model.
 */

// Test data
const testData = {
    conversations: [
        { input: "Hello", response: "Hi there! How can I help you today?" },
        { input: "How are you?", response: "I'm doing well, thank you for asking! How about you?" },
        { input: "What is your name?", response: "I'm MR Bot, a self-learning language teaching assistant." },
        { input: "Goodbye", response: "Goodbye! Have a great day!" }
    ],
    texts: [
        "Language learning is the process by which humans acquire the capacity to perceive, produce, and use words to understand and communicate.",
        "A neural network is a series of algorithms that endeavors to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates."
    ]
};

/**
 * Run tests for the neural language model
 */
async function runTests() {
    console.log("Running Neural MR Bot tests...");
    
    try {
        // Test 1: Initialize components
        console.log("Test 1: Initializing components...");
        
        const dataManager = new NeuralDataManager();
        await dataManager.initialize();
        console.log("✓ Data manager initialized");
        
        const languageModel = new NeuralLanguageModel();
        await languageModel.initialize(dataManager);
        console.log("✓ Language model initialized");
        
        const conversationEngine = new NeuralConversationEngine();
        await conversationEngine.initialize(languageModel, dataManager);
        console.log("✓ Conversation engine initialized");
        
        // Test 2: Add training data
        console.log("\nTest 2: Adding training data...");
        
        for (const conv of testData.conversations) {
            await dataManager.addConversation(conv.input, conv.response);
        }
        console.log(`✓ Added ${testData.conversations.length} conversations`);
        
        for (const text of testData.texts) {
            await dataManager.addTrainingText(text, 'en');
        }
        console.log(`✓ Added ${testData.texts.length} texts`);
        
        // Test 3: Get training data
        console.log("\nTest 3: Getting training data...");
        
        const trainingData = await dataManager.getTrainingData();
        console.log(`✓ Got ${trainingData.length} training examples`);
        
        const conversationPairs = await dataManager.getConversationPairs();
        console.log(`✓ Got ${conversationPairs.length} conversation pairs`);
        
        // Test 4: Train model
        console.log("\nTest 4: Training model...");
        
        const trainingResult = await languageModel.trainModel(conversationPairs, { epochs: 2 });
        console.log(`✓ Model trained: ${trainingResult}`);
        
        // Test 5: Generate responses
        console.log("\nTest 5: Generating responses...");
        
        const testInputs = ["Hello", "How are you?", "What is your name?", "Goodbye"];
        
        for (const input of testInputs) {
            const response = await languageModel.generateResponse(input);
            console.log(`Input: "${input}" → Response: "${response}"`);
        }
        
        // Test 6: Process user input
        console.log("\nTest 6: Processing user input...");
        
        const processResult = await conversationEngine.processInput("Hello, can you help me learn a language?");
        console.log(`✓ Processed input: "${processResult.text}"`);
        
        // Test 7: Export model
        console.log("\nTest 7: Exporting model...");
        
        const exportData = await languageModel.exportModel('json');
        console.log(`✓ Model exported: ${exportData !== null}`);
        
        console.log("\nAll tests completed successfully!");
        return true;
    } catch (error) {
        console.error("Test failed:", error);
        return false;
    }
}

// Run tests when script is loaded directly
if (typeof window !== 'undefined' && window.runNeuralTests) {
    runTests().then(success => {
        console.log(`Tests ${success ? 'passed' : 'failed'}`);
    });
}

// Export for Node.js environment
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { runTests };
}