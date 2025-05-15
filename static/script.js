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

const activityLog = document.getElementById('activity-log'); // Get the activity log div

// --- Helper Functions ---

function logToolActivity(message) {
    if (!activityLog) return; // Safety check
    const activityEntry = document.createElement('p');
    activityEntry.textContent = message; // Use textContent for security
    activityLog.appendChild(activityEntry);
    // Scroll to the bottom of the activity log
    activityLog.scrollTop = activityLog.scrollHeight;
}

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
    let fullResponse = ""; // To accumulate the full response for potential Markdown parsing later

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: userMessage,
                model: selectedModel,
                ollama_url: ollamaUrl,
                // max_tool_iterations: 3 // Example, can be configurable
            }),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ error: 'Unknown error occurred, server response not JSON.' }));
            throw new Error(`HTTP error! status: ${response.status}, message: ${errorData.error || 'Server error'}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        assistantMessageDiv.classList.remove('loading-placeholder'); // Remove placeholder once stream starts

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            
            // Process buffer line by line (assuming server sends JSON objects separated by newlines)
            let newlineIndex;
            while ((newlineIndex = buffer.indexOf('\n')) >= 0) {
                const line = buffer.substring(0, newlineIndex).trim();
                buffer = buffer.substring(newlineIndex + 1);

                if (line) {
                    try {
                        const chunk = JSON.parse(line);
                        if (chunk.type === 'tool_activity' && chunk.content) {
                            logToolActivity(chunk.content);
                        } else if (chunk.response) {
                            fullResponse += chunk.response;
                            // Basic Markdown-like formatting for streaming
                            assistantParagraph.innerHTML = fullResponse
                                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                                .replace(/\*(.*?)\*/g, '<em>$1</em>')
                                .replace(/\n/g, '<br>'); // Replace explicit newlines if any
                        } else if (chunk.error) {
                            // Handle errors streamed from the backend
                            console.error("Streamed error:", chunk.error);
                            appendMessage('error', `Server error: ${chunk.error}`);
                            // Optionally stop processing further messages if a critical error occurs
                            // return; 
                        }
                        // Add other chunk types if needed (e.g., tool_call_result)
                    } catch (e) {
                        // This might happen if a non-JSON line is encountered or if JSON is malformed
                        // It could also be a raw string if the server sometimes doesn't send JSON
                        console.warn('Received non-JSON line or malformed JSON in stream:', line, e);
                        // As a fallback, append raw line if it's not an empty string from keep-alive pings
                        if (line.trim()) {
                             fullResponse += line; // Add it to the main response
                             assistantParagraph.innerHTML = fullResponse
                                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                                .replace(/\*(.*?)\*/g, '<em>$1</em>')
                                .replace(/\n/g, '<br>');
                        }
                    }
                }
                chatHistory.scrollTop = chatHistory.scrollHeight; // Keep scrolled to bottom
            }
        }
        
        // Final processing of any remaining buffer content (if stream ends without newline)
        if (buffer.trim()) {
            try {
                const chunk = JSON.parse(buffer.trim());
                 if (chunk.type === 'tool_activity' && chunk.content) {
                    logToolActivity(chunk.content);
                } else if (chunk.response) {
                    fullResponse += chunk.response;
                } else if (chunk.error) {
                    console.error("Streamed error (final buffer):", chunk.error);
                    appendMessage('error', `Server error: ${chunk.error}`);
                }
            } catch (e) {
                 console.warn('Received non-JSON final buffer content:', buffer.trim(), e);
                 fullResponse += buffer.trim(); // Add it to the main response
            }
        }

        // Update with the complete response for final Markdown rendering if needed,
        // or if server only sends one complete JSON at the end.
        // For now, progressive rendering handles it.
        // assistantParagraph.innerHTML = marked.parse(fullResponse); // If using a library like 'marked'

    } catch (error) {
        console.error('Error sending message:', error);
        assistantParagraph.textContent = `Error: ${error.message}`;
        assistantMessageDiv.classList.remove('loading-placeholder');
        appendMessage('error', `Failed to get response from Fred: ${error.message}`);
    } finally {
        // Deactivate visualization after response or error
        fredVisualization.classList.remove('active');
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