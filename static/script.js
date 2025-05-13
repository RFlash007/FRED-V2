const messageForm = document.getElementById('message-form');
const messageInput = document.getElementById('message-input');
const chatHistory = document.getElementById('chat-history');
const modelSelect = document.getElementById('model-select');
const ollamaUrlInput = document.getElementById('ollama-url');
// const memoryTabsContainer = document.querySelector('.memory-tabs'); // REMOVED (Unused)
const memoryContentDisplay = document.getElementById('memory-content-display');
// const historyList = document.getElementById('history-list'); // Removed unused variable
const darkModeSwitch = document.getElementById('dark-mode-switch');
const fredVisualization = document.getElementById('fred-visualization');
const bodyElement = document.body;
const sidebarToggle = document.getElementById('sidebar-toggle');
// const idleModeSwitch = document.getElementById('idle-mode-switch'); // REMOVED (Feature removed)

// --- Helper Functions ---

function appendMessage(role, content, addLoading = false) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', role);
    if (addLoading && role === 'assistant') {
        messageDiv.classList.add('loading-placeholder');
    }

    const paragraph = document.createElement('p');
    // Basic Markdown-like formatting (bold, italic) - could be expanded
    content = content
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // Bold
        .replace(/\*(.*?)\*/g, '<em>$1</em>');       // Italic
    paragraph.innerHTML = content; // Use innerHTML for basic formatting
    messageDiv.appendChild(paragraph);

    // Remove initial system message if it exists and this is the first *real* message
    const systemMessage = chatHistory.querySelector('.message.system');
    if (systemMessage && chatHistory.children.length === 1 && (role === 'user' || role === 'assistant')) {
        systemMessage.remove();
    }

    chatHistory.appendChild(messageDiv);
    // Smooth scroll to bottom
    chatHistory.scrollTo({
        top: chatHistory.scrollHeight,
        behavior: 'smooth'
    });
    return messageDiv;
}

function adjustTextareaHeight(textarea) {
    textarea.style.height = 'auto'; // Temporarily shrink
    const scrollHeight = textarea.scrollHeight;
    const maxHeight = parseInt(window.getComputedStyle(textarea).maxHeight, 10) || 150;

    if (scrollHeight > maxHeight) {
        textarea.style.height = maxHeight + 'px';
        textarea.style.overflowY = 'auto'; // Show scrollbar if max height reached
    } else {
        textarea.style.height = scrollHeight + 'px';
        textarea.style.overflowY = 'hidden'; // Hide scrollbar if not needed
    }
}

// --- Dark Mode ---
function applyDarkMode(isDark) {
    if (isDark) {
        bodyElement.classList.add('dark-mode');
    } else {
        bodyElement.classList.remove('dark-mode');
    }
}

function toggleDarkMode() {
    const isDarkMode = bodyElement.classList.toggle('dark-mode');
    localStorage.setItem('darkMode', isDarkMode);
    darkModeSwitch.checked = isDarkMode;
}

// --- Sidebar Toggle ---
function toggleSidebar() {
    // Toggle the sidebar-collapsed class on the body element
    bodyElement.classList.toggle('sidebar-collapsed');
    
    // Store the state in localStorage
    const isCollapsed = bodyElement.classList.contains('sidebar-collapsed');
    localStorage.setItem('sidebarCollapsed', isCollapsed);
    
    // Trigger resize to center the mind map properly
    if (window.mindMap) {
        setTimeout(() => window.mindMap.resize(), 300); // Wait for transition to complete
    }
}

// --- Memory Management Functions --- // REMOVED/COMMENTED OUT
/*
function updateMemoryDisplay(message) {
    if (memoryContentDisplay) {
        memoryContentDisplay.textContent = message;
    }
}

function toggleIdleMode() {
    const isEnabled = idleModeSwitch.checked;
    
    // Update UI immediately to show loading state
    updateMemoryDisplay(`${isEnabled ? 'Enabling' : 'Disabling'} idle association discovery...`);
    
    // Send request to API
    fetch('/api/memory/idle-mode', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ enabled: isEnabled })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            updateMemoryDisplay(
                isEnabled 
                ? 'Idle mode activated. Discovering memory associations in the background.' 
                : 'Idle mode deactivated.'
            );
            
            // Activate visualization if idle mode is enabled
            if (window.mindMap && isEnabled) {
                window.mindMap.setActive(true);
            }
        } else {
            // Revert the toggle if there was an error
            idleModeSwitch.checked = !isEnabled;
            updateMemoryDisplay(`Error: Could not ${isEnabled ? 'enable' : 'disable'} idle mode.`);
        }
    })
    .catch(error => {
        console.error('Error toggling idle mode:', error);
        idleModeSwitch.checked = !isEnabled;
        updateMemoryDisplay('Error communicating with the server.');
    });
}

function fetchMemoryStatus() {
    fetch('/api/memory/status')
        .then(response => response.json())
        .then(data => {
            // Update idle mode toggle to match current state
            if (idleModeSwitch) {
                idleModeSwitch.checked = data.idle_mode;
            }
            
            // Update memory display with status information
            const nodeCount = data.node_count;
            updateMemoryDisplay(
                data.idle_mode 
                ? `Idle mode active. ${nodeCount} memories in the system.` 
                : `${nodeCount} memories in the system.`
            );
        })
        .catch(error => {
            console.error('Error fetching memory status:', error);
            // Optionally update display with error message
            // updateMemoryDisplay('Could not fetch memory status.'); 
        });
}
*/

// --- Initial Setup ---
// Initialize mind map visualization on page load
document.addEventListener('DOMContentLoaded', () => {
    // The mind map will be initialized through mind-map.js
    // but we can make sure we have the visualization container ready
    const fredVis = document.getElementById('fred-visualization');
    if (fredVis) {
        fredVis.setAttribute('aria-label', 'Mind map visualization');
    }
    
    // Initial textarea height
    adjustTextareaHeight(messageInput);
    
    // Check local storage for sidebar state
    const sidebarCollapsed = localStorage.getItem('sidebarCollapsed') === 'true';
    if (sidebarCollapsed) {
        bodyElement.classList.add('sidebar-collapsed');
    }
    
    // Add click event listener to sidebar toggle button
    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', toggleSidebar);
    }
    
    // Setup idle mode toggle - REMOVED
    /*
    if (idleModeSwitch) {
        idleModeSwitch.addEventListener('change', toggleIdleMode);
        // Fetch current status to set toggle correctly
        fetchMemoryStatus();
    }
    */
    
    console.log('Fred UI Initialized');

    // Add chat toggle button functionality - REMOVED
    /*
    const chatToggleBtn = document.createElement('button');
    chatToggleBtn.id = 'chat-toggle';
    chatToggleBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>';
    document.querySelector('.chat-area').appendChild(chatToggleBtn);
    
    chatToggleBtn.addEventListener('click', function() {
        chatHistory.classList.toggle('expanded');
    });
    */
});

// Check local storage on load
const savedDarkMode = localStorage.getItem('darkMode') === 'true';
applyDarkMode(savedDarkMode);
darkModeSwitch.checked = savedDarkMode;

// Add listener to toggle
darkModeSwitch.addEventListener('change', toggleDarkMode);

// --- Event Listeners ---

messageForm.addEventListener('submit', async (event) => {
    event.preventDefault(); // Prevent default form submission

    const userMessage = messageInput.value.trim();
    if (!userMessage) return; // Don't send empty messages

    const selectedModel = modelSelect.value;
    const ollamaUrl = ollamaUrlInput.value.trim(); // Trim whitespace

    if (!ollamaUrl) {
        appendMessage('error', 'Server Endpoint URL cannot be empty.');
        return;
    }

    appendMessage('user', userMessage);
    messageInput.value = ''; // Clear input field
    adjustTextareaHeight(messageInput); // Reset textarea height
    messageInput.focus(); // Keep focus on input

    // Activate visualization
    fredVisualization.classList.add('active');
    
    // Start mind map animation
    if (window.mindMap) {
        window.mindMap.setActive(true);
    }

    const assistantMessageDiv = appendMessage('assistant', '', true); // Add loading placeholder
    const assistantParagraph = assistantMessageDiv.querySelector('p');

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'text/event-stream'
            },
            body: JSON.stringify({
                message: userMessage,
                model: selectedModel,
                ollama_url: ollamaUrl
            })
        });

        if (!response.ok || !response.body) {
            const errorData = response.ok ? { error: 'Response body is missing' } : await response.json();
            throw new Error(errorData.error || `Server responded with status ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let firstChunk = true;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            let lines = buffer.split('\n\n');
            buffer = lines.pop(); // Keep incomplete line

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const jsonStr = line.substring(6);
                        const data = JSON.parse(jsonStr);

                        if (data.type === 'chunk') {
                            if (firstChunk) {
                                assistantParagraph.textContent = '';
                                assistantMessageDiv.classList.remove('loading-placeholder');
                                firstChunk = false;
                            }
                            // Append content incrementally, handling potential basic markdown
                            assistantParagraph.innerHTML += data.content
                                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                                .replace(/\*(.*?)\*/g, '<em>$1</em>');
                            chatHistory.scrollTo({ top: chatHistory.scrollHeight, behavior: 'auto' }); // Use auto scroll during streaming
                        } else if (data.type === 'done') {
                            console.log("Stream finished (done message).");
                            assistantMessageDiv.classList.remove('loading-placeholder');
                            chatHistory.scrollTo({ top: chatHistory.scrollHeight, behavior: 'smooth' }); // Smooth scroll at the end
                            reader.cancel();
                            // Deactivate visualization
                            fredVisualization.classList.remove('active');
                            
                            // Stop mind map animation
                            if (window.mindMap) {
                                window.mindMap.setActive(false);
                            }
                            return;
                        } else if (data.type === 'error') {
                            throw new Error(data.content);
                        }
                    } catch (parseError) {
                        console.error("Error parsing SSE data:", parseError, "Raw line:", line);
                    }
                }
            }
        }
        console.log("Stream finished (reader done).");
        assistantMessageDiv.classList.remove('loading-placeholder');
        chatHistory.scrollTo({ top: chatHistory.scrollHeight, behavior: 'smooth' });

    } catch (error) {
        console.error('Chat Error:', error);
        assistantParagraph.textContent = `Error: ${error.message}`;
        assistantMessageDiv.classList.remove('assistant', 'loading-placeholder');
        assistantMessageDiv.classList.add('error');
        chatHistory.scrollTo({ top: chatHistory.scrollHeight, behavior: 'smooth' });
    } finally {
        // Ensure visualization is always deactivated
        fredVisualization.classList.remove('active');
        
        // Always stop mind map animation
        if (window.mindMap) {
            window.mindMap.setActive(false);
        }
    }
});

// Textarea Input Handling (Auto-resize and Enter key)
messageInput.addEventListener('input', () => adjustTextareaHeight(messageInput));
messageInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        messageForm.requestSubmit();
    }
});

// Memory Tab Interaction
// memoryTabsContainer.addEventListener('click', (event) => {
//     if (event.target.classList.contains('memory-tab')) {
//         // Remove active class from all tabs
//         memoryTabsContainer.querySelectorAll('.memory-tab').forEach(tab => tab.classList.remove('active'));
//         // Add active class to the clicked tab
//         event.target.classList.add('active');
//         // Update placeholder content (replace with actual logic later)
//         const memoryType = event.target.dataset.memory;
//         // Add a temporary loading state visual
//         memoryContentDisplay.textContent = `Loading ${memoryType} view...`;
//         memoryContentDisplay.style.opacity = '0.5';
//         setTimeout(() => { // Simulate loading
//             memoryContentDisplay.textContent = `Accessing ${memoryType} memory...`;
//             memoryContentDisplay.style.opacity = '0.7';
//         }, 300);
//         console.log(`Switched to ${memoryType} memory view.`);
//         // TODO: Add logic to fetch and display actual memory content
//     }
// });

// Initial Setup
adjustTextareaHeight(messageInput);
console.log('Fred UI Initialized');

// --- Potential Future Enhancements ---
// TODO: Load/save chat history (localStorage or backend)
// TODO: Dynamic model list population (requires backend endpoint)
// TODO: Settings persistence (localStorage)
// TODO: Implement actual memory content loading/display
// TODO: More robust Markdown rendering

console.log('Clay UI Chat Initialized');
// TODO: Add logic to load/save chat history from localStorage
// TODO: Add logic to dynamically populate models from Ollama API (requires a new backend endpoint)
// TODO: Add logic to save/load settings from localStorage 